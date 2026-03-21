"""
models/semantic_memory.py

Context-aware semantic memory for TGNN.

Memory v2 goals:
- do not rely only on token identity
- read memory using node + context
- fuse memory through a learned gate
- support delayed activation from training loop
- support selective weighted EMA writes
- keep output shape compatible with current TGNN:
    augmented_features = [node_features ; memory_vector]

This file is the memory-core redesign only.
Training schedule, confidence/frequency/novelty weighting, and model-side
context construction are handled in later files.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import config
from graph_construction.vocabulary import load_vocab


class SemanticMemory(nn.Module):
    """
    Context-aware semantic memory for temporal/API-like nodes.

    Read path:
        token embedding + context query -> sparse attention over memory bank
        -> gated fusion with current node features
        -> return augmented features [node_features ; fused_memory]

    Write path:
        weighted EMA updates to token-associated memory slots
        using contextual write candidates.
    """

    def __init__(
        self,
        vocab_path: str,
        memory_dim: int,
        node_feat_dim: int,
        query_context_dim: Optional[int] = None,
        write_context_dim: Optional[int] = None,
        query_hidden_dim: Optional[int] = None,
        gate_hidden_dim: Optional[int] = None,
        read_temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        super().__init__()

        self.vocab = load_vocab(vocab_path)
        if self.vocab is None:
            raise FileNotFoundError(f"Vocabulary file not found or failed to load: {vocab_path}")

        self.vocab_size = self.vocab.size()
        self.memory_dim = memory_dim
        self.node_feat_dim = node_feat_dim
        self.query_context_dim = query_context_dim
        self.write_context_dim = write_context_dim

        self.query_hidden_dim = (
            query_hidden_dim
            if query_hidden_dim is not None
            else getattr(config, "MEMORY_QUERY_HIDDEN_DIM", 256)
        )
        self.gate_hidden_dim = (
            gate_hidden_dim
            if gate_hidden_dim is not None
            else getattr(config, "MEMORY_GATE_HIDDEN_DIM", 128)
        )
        self.read_temperature = (
            read_temperature
            if read_temperature is not None
            else getattr(config, "MEMORY_READ_TEMPERATURE", 1.0)
        )
        self.top_k = (
            top_k
            if top_k is not None
            else getattr(config, "MEMORY_TOP_K", 64)
        )

        # 1. Token embedding (identity prior)
        self.token_embedding = nn.Embedding(self.vocab_size, memory_dim)

        # 2. Memory bank
        self.memory_keys = nn.Embedding(self.vocab_size, memory_dim)
        self.memory_values = nn.Embedding(self.vocab_size, memory_dim)

        # 3. Base context-aware query projection
        base_query_in_dim = memory_dim + node_feat_dim
        self.base_query_proj = nn.Sequential(
            nn.Linear(base_query_in_dim, self.query_hidden_dim),
            nn.LayerNorm(self.query_hidden_dim),
            nn.GELU(),
            nn.Linear(self.query_hidden_dim, memory_dim),
        )

        # 3b. Explicitly registered extra context projections
        self.extra_context_proj = None
        if self.query_context_dim is not None:
            self.extra_context_proj = nn.Sequential(
                nn.Linear(self.query_context_dim, self.query_hidden_dim),
                nn.LayerNorm(self.query_hidden_dim),
                nn.GELU(),
                nn.Linear(self.query_hidden_dim, self.memory_dim),
            )

        self.extra_write_context_proj = None
        if self.write_context_dim is not None:
            self.extra_write_context_proj = nn.Sequential(
                nn.Linear(self.write_context_dim, self.query_hidden_dim),
                nn.LayerNorm(self.query_hidden_dim),
                nn.GELU(),
                nn.Linear(self.query_hidden_dim, self.memory_dim),
            )

        # 4. Gated read
        gate_in_dim = node_feat_dim + memory_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, self.gate_hidden_dim),
            nn.LayerNorm(self.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, memory_dim),
        )

        # 5. Write candidate projection
        self.write_candidate_proj = nn.Sequential(
            nn.Linear(base_query_in_dim, self.query_hidden_dim),
            nn.LayerNorm(self.query_hidden_dim),
            nn.GELU(),
            nn.Linear(self.query_hidden_dim, memory_dim),
        )

        # Cache from latest read for optional inspection/debugging
        self._last_read: Dict[str, torch.Tensor] = {}

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.memory_keys.weight)
        nn.init.xavier_uniform_(self.memory_values.weight)

        modules_to_init = [
            self.base_query_proj,
            self.gate_mlp,
            self.write_candidate_proj,
        ]

        if self.extra_context_proj is not None:
            modules_to_init.append(self.extra_context_proj)
        if self.extra_write_context_proj is not None:
            modules_to_init.append(self.extra_write_context_proj)

        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _clamp_gate(self, gate: torch.Tensor) -> torch.Tensor:
        min_gate = getattr(config, "MEMORY_MIN_GATE", 0.05)
        max_gate = getattr(config, "MEMORY_MAX_GATE", 0.95)
        return gate * (max_gate - min_gate) + min_gate

    def _build_query(
        self,
        api_indices: torch.LongTensor,
        node_features: torch.Tensor,
        query_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_emb = self.token_embedding(api_indices)
        base_input = torch.cat([token_emb, node_features], dim=-1)
        query = self.base_query_proj(base_input)

        if query_context is not None:
            if query_context.dim() != 2:
                raise ValueError(
                    f"query_context must be 2D [N, D], got shape={tuple(query_context.shape)}"
                )
            if self.extra_context_proj is None:
                raise ValueError(
                    "query_context was provided, but extra_context_proj is not initialized. "
                    "Pass query_context_dim when building SemanticMemory."
                )
            if self.query_context_dim is None or query_context.size(-1) != self.query_context_dim:
                raise ValueError(
                    f"query_context dim mismatch: got {query_context.size(-1)}, "
                    f"expected {self.query_context_dim}"
                )

            query = query + self.extra_context_proj(query_context)

        query = F.normalize(query, dim=-1)
        return query

    def _build_write_candidate(
        self,
        api_indices: torch.LongTensor,
        node_features: torch.Tensor,
        write_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_emb = self.token_embedding(api_indices)
        base_input = torch.cat([token_emb, node_features], dim=-1)
        candidate = self.write_candidate_proj(base_input)

        if write_context is not None:
            if write_context.dim() != 2:
                raise ValueError(
                    f"write_context must be 2D [N, D], got shape={tuple(write_context.shape)}"
                )
            if self.extra_write_context_proj is None:
                raise ValueError(
                    "write_context was provided, but extra_write_context_proj is not initialized. "
                    "Pass write_context_dim when building SemanticMemory."
                )
            if self.write_context_dim is None or write_context.size(-1) != self.write_context_dim:
                raise ValueError(
                    f"write_context dim mismatch: got {write_context.size(-1)}, "
                    f"expected {self.write_context_dim}"
                )

            candidate = candidate + self.extra_write_context_proj(write_context)

        candidate = F.normalize(candidate, dim=-1)
        return candidate

    def read(
        self,
        api_indices: torch.LongTensor,
        node_features: torch.Tensor,
        query_context: Optional[torch.Tensor] = None,
        enable_memory: bool = True,
        return_debug: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            fused_memory: [N, memory_dim]
        """
        N = node_features.size(0)

        if N == 0:
            out = node_features.new_zeros((0, self.memory_dim))
            return (out, {}) if return_debug else out

        if not enable_memory:
            out = node_features.new_zeros((N, self.memory_dim))
            return (out, {}) if return_debug else out

        query = self._build_query(
            api_indices=api_indices,
            node_features=node_features,
            query_context=query_context,
        )  # [N, memory_dim]

        local_vals = self.memory_values(api_indices)  # [N, memory_dim]

        # Sparse global contextual retrieval over the memory bank
        global_keys = F.normalize(self.memory_keys.weight, dim=-1)  # [V, D]
        scores = torch.matmul(query, global_keys.T) / max(self.read_temperature, 1e-6)

        top_k = min(self.top_k, self.vocab_size)
        topk_scores, topk_idx = scores.topk(top_k, dim=-1)  # [N, K]
        topk_attn = F.softmax(topk_scores, dim=-1)          # [N, K]
        topk_vals = self.memory_values.weight[topk_idx]     # [N, K, D]
        global_read = (topk_attn.unsqueeze(-1) * topk_vals).sum(dim=1)  # [N, D]

        # Blend token-local prior with context retrieval
        raw_memory = 0.5 * local_vals + 0.5 * global_read

        # Learned gate
        gate_input = torch.cat([node_features, raw_memory], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        gate = self._clamp_gate(gate)

        fused_memory = gate * raw_memory

        if return_debug:
            debug = {
                "query": query.detach(),
                "topk_idx": topk_idx.detach(),
                "topk_attn": topk_attn.detach(),
                "local_vals": local_vals.detach(),
                "global_read": global_read.detach(),
                "gate": gate.detach(),
                "fused_memory": fused_memory.detach(),
            }
            self._last_read = debug
            return fused_memory, debug

        return fused_memory

    def forward(
        self,
        api_indices: torch.LongTensor,
        node_features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        query_context: Optional[torch.Tensor] = None,
        enable_memory: bool = True,
        return_debug: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward returns augmented node features:
            [node_features ; fused_memory]
        """
        if return_debug:
            fused_memory, debug = self.read(
                api_indices=api_indices,
                node_features=node_features,
                query_context=query_context,
                enable_memory=enable_memory,
                return_debug=True,
            )
        else:
            fused_memory = self.read(
                api_indices=api_indices,
                node_features=node_features,
                query_context=query_context,
                enable_memory=enable_memory,
                return_debug=False,
            )
            debug = {}

        augmented = torch.cat([node_features, fused_memory], dim=-1)

        if return_debug:
            debug["augmented"] = augmented.detach()
            if timestamps is not None:
                debug["timestamps"] = timestamps.detach()
            return augmented, debug

        return augmented

    @torch.no_grad()
    def write(
        self,
        api_indices: torch.LongTensor,
        node_features: torch.Tensor,
        momentum: float = 0.9,
        write_strength: Optional[torch.Tensor] = None,
        write_context: Optional[torch.Tensor] = None,
        enable_memory: bool = True,
    ) -> None:
        """
        Selective weighted EMA write.
        """
        if not enable_memory:
            return

        if api_indices.numel() == 0:
            return

        candidate = self._build_write_candidate(
            api_indices=api_indices,
            node_features=node_features,
            write_context=write_context,
        )  # [N, memory_dim]

        if write_strength is None:
            write_strength = torch.ones(
                api_indices.size(0),
                device=api_indices.device,
                dtype=node_features.dtype,
            )
        else:
            if write_strength.dim() == 2 and write_strength.size(1) == 1:
                write_strength = write_strength.squeeze(1)
            if write_strength.dim() != 1:
                raise ValueError(
                    f"write_strength must be [N] or [N,1], got shape={tuple(write_strength.shape)}"
                )
            write_strength = write_strength.to(node_features.device, dtype=node_features.dtype)

        min_s = getattr(config, "MEMORY_WRITE_MIN_STRENGTH", 0.05)
        max_s = getattr(config, "MEMORY_WRITE_MAX_STRENGTH", 1.0)
        write_strength = write_strength.clamp(min=min_s, max=max_s)

        unique_ids = torch.unique(api_indices)

        for idx in unique_ids:
            mask = api_indices == idx
            if not mask.any():
                continue

            cand_i = candidate[mask]
            strength_i = write_strength[mask]

            strength_sum = strength_i.sum().clamp_min(1e-6)
            weighted_cand = (cand_i * strength_i.unsqueeze(-1)).sum(dim=0) / strength_sum

            avg_strength = strength_i.mean().item()

            # avg_strength=1.0 -> standard momentum
            # avg_strength=0.0 -> no update
            effective_momentum = 1.0 - (1.0 - momentum) * avg_strength

            old_key = self.memory_keys.weight[idx]
            old_val = self.memory_values.weight[idx]

            new_key = effective_momentum * old_key + (1.0 - effective_momentum) * weighted_cand
            new_val = effective_momentum * old_val + (1.0 - effective_momentum) * weighted_cand

            self.memory_keys.weight[idx].copy_(F.normalize(new_key, dim=-1))
            self.memory_values.weight[idx].copy_(new_val)

    @torch.no_grad()
    def get_memory_vectors(self, api_indices: torch.LongTensor) -> torch.Tensor:
        return self.memory_values(api_indices)

    @torch.no_grad()
    def get_token_embeddings(self, api_indices: torch.LongTensor) -> torch.Tensor:
        return self.token_embedding(api_indices)


def build_semantic_memory(
    vocab_path: str,
    memory_dim: int,
    node_feat_dim: int,
    query_context_dim: Optional[int] = None,
    write_context_dim: Optional[int] = None,
) -> SemanticMemory:
    return SemanticMemory(
        vocab_path=vocab_path,
        memory_dim=memory_dim,
        node_feat_dim=node_feat_dim,
        query_context_dim=query_context_dim,
        write_context_dim=write_context_dim,
        query_hidden_dim=getattr(config, "MEMORY_QUERY_HIDDEN_DIM", 256),
        gate_hidden_dim=getattr(config, "MEMORY_GATE_HIDDEN_DIM", 128),
        read_temperature=getattr(config, "MEMORY_READ_TEMPERATURE", 1.0),
        top_k=getattr(config, "MEMORY_TOP_K", 64),
    )