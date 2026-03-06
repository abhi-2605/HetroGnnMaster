"""
models/semantic_memory.py

Semantic Memory Module for Ransomware TGNN.

Provides a persistent, learnable memory bank indexed by API vocabulary.
Each API gets a 128-dim memory vector that accumulates cross-graph
behavioral context during training.

Architecture:
    Memory Bank:  nn.Embedding(325, 128)  → one slot per API
    Read:         Attention-based retrieval (time-aware)
    Write:        EMA update after each batch
    Output:       node_features [775] + memory [128] = [903-dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from graph_construction.vocabulary import load_vocab


# ══════════════════════════════════════════════════════════════════════════════
#  SemanticMemory
# ══════════════════════════════════════════════════════════════════════════════

class SemanticMemory(nn.Module):
    """
    Persistent API behaviour memory bank.

    Memory flow:
        READ  → api_idx → attention blend over memory bank → 128-dim vector
        WRITE → after optimizer.step(), EMA-update memory slots for seen APIs

    Usage:
        memory  = SemanticMemory(vocab_size=325)
        enriched = memory(api_indices, node_features, timestamps)
        # → [N, 903]  (775 static + 128 dynamic)

        # after loss.backward() + optimizer.step():
        memory.write(api_indices, node_features.detach())
    """

    def __init__(
        self,
        vocab_size: int,
        memory_dim: int = 128,
        node_feat_dim: int = 775,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size    = vocab_size
        self.memory_dim    = memory_dim
        self.node_feat_dim = node_feat_dim
        self.output_dim    = node_feat_dim + memory_dim   # 903

        # ── Memory Bank ──────────────────────────────────────────────────────
        # padding_idx=1 → <PAD> slot is always zero, never updated
        self.memory_bank = nn.Embedding(
            vocab_size, memory_dim, padding_idx=1
        )

        # ── Read: query projection (node features → query space) ─────────────
        self.query_proj = nn.Linear(node_feat_dim, memory_dim, bias=False)

        # ── Read: multi-head soft attention over full memory bank ─────────────
        self.attention = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ── Read: time-aware gate ─────────────────────────────────────────────
        # timestamp (scalar) → gate vector to scale memory by temporal position
        self.time_gate = nn.Sequential(
            nn.Linear(1, memory_dim),
            nn.Sigmoid(),
        )

        # ── Output normalisation ──────────────────────────────────────────────
        self.layer_norm = nn.LayerNorm(memory_dim)
        self.dropout    = nn.Dropout(dropout)

        self._init_weights()

    # ── initialisation ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.normal_(self.memory_bank.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.memory_bank.weight[0].zero_()   # <UNK>
            self.memory_bank.weight[1].zero_()   # <PAD>

    # ── READ ──────────────────────────────────────────────────────────────────

    def read(
        self,
        api_indices:   torch.LongTensor,          # [N]
        node_features: torch.Tensor,              # [N, 775]
        timestamps:    torch.Tensor = None,       # [N]  optional
    ) -> torch.Tensor:
        """
        Retrieve memory for each node.

        Three-step process:
          1. Direct lookup  → hard memory for this API
          2. Soft attention → blend from behaviorally similar APIs
          3. Time gate      → scale by position in execution sequence

        Returns:
            memory_vectors: [N, 128]
        """
        # 1. Direct lookup  [N, 128]
        direct = self.memory_bank(api_indices)

        # 2. Soft attention over entire memory bank
        queries = self.query_proj(node_features).unsqueeze(0)   # [1, N, 128]
        keys    = self.memory_bank.weight.unsqueeze(0)           # [1, V, 128]

        attended, _ = self.attention(
            query=queries,
            key=keys,
            value=keys,
        )                                                         # [1, N, 128]
        attended = attended.squeeze(0)                           # [N, 128]

        # 3. Blend: direct carries more weight (it is the API's own memory)
        blended = 0.7 * direct + 0.3 * attended                 # [N, 128]

        # 4. Time-aware gating
        if timestamps is not None:
            t = timestamps.float().unsqueeze(-1)                 # [N, 1]
            # Normalise to [0, 1] within this graph
            t_min, t_max = t.min(), t.max()
            t = (t - t_min) / (t_max - t_min + 1e-8)
            gate   = self.time_gate(t)                           # [N, 128]
            blended = blended * gate

        out = self.layer_norm(blended)
        out = self.dropout(out)
        return out                                               # [N, 128]

    # ── WRITE ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def write(
        self,
        api_indices:    torch.LongTensor,         # [N]
        node_features:  torch.Tensor,             # [N, 775]  detached
        momentum:       float = 0.9,
    ) -> None:
        """
        Update memory bank after each optimiser step (EMA).

        For every unique API seen in this batch:
          new_mem = momentum * old_mem + (1 - momentum) * mean_signal

        momentum=0.9 → memory changes slowly → prevents catastrophic
        forgetting while still accumulating cross-graph context.

        Call pattern in training loop:
            loss.backward()
            optimizer.step()
            memory.write(api_indices, node_features.detach())  ← here
        """
        # Re-use the already-trained query_proj to project 775 → 128
        projected = self.query_proj(node_features)              # [N, 128]

        for api_idx in api_indices.unique():
            idx = api_idx.item()
            if idx <= 1:          # skip <UNK>=0 and <PAD>=1
                continue

            mask   = (api_indices == api_idx)
            signal = projected[mask].mean(dim=0)                # [128]

            old    = self.memory_bank.weight[idx]               # [128]
            updated = momentum * old + (1.0 - momentum) * signal
            self.memory_bank.weight[idx].copy_(updated)

    # ── FORWARD ───────────────────────────────────────────────────────────────

    def forward(
        self,
        api_indices:   torch.LongTensor,          # [N]
        node_features: torch.Tensor,              # [N, 775]
        timestamps:    torch.Tensor = None,       # [N]  optional
    ) -> torch.Tensor:
        """
        Read memory and concatenate with node features.

        Returns:
            enriched: [N, 903]   (775 static + 128 dynamic)
        """
        mem      = self.read(api_indices, node_features, timestamps)
        enriched = torch.cat([node_features, mem], dim=-1)
        return enriched                                          # [N, 903]

    # ── UTILITIES ─────────────────────────────────────────────────────────────

    def reset_memory(self) -> None:
        """Reset all memory slots to initial state."""
        self._init_weights()

    def get_api_memory(self, api_idx: int) -> torch.Tensor:
        """Return memory vector for a single API. [128]"""
        return self.memory_bank.weight[api_idx].detach().clone()

    def memory_stats(self) -> dict:
        """Diagnostic stats about the current memory state."""
        w     = self.memory_bank.weight.detach()
        norms = w.norm(dim=-1)
        return {
            "mean_norm" : norms.mean().item(),
            "max_norm"  : norms.max().item(),
            "min_norm"  : norms.min().item(),
            "mean_value": w.mean().item(),
            "std_value" : w.std().item(),
        }

    def __repr__(self) -> str:
        return (
            f"SemanticMemory("
            f"vocab={self.vocab_size}, "
            f"memory_dim={self.memory_dim}, "
            f"input_dim={self.node_feat_dim}, "
            f"output_dim={self.output_dim})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_semantic_memory(
    vocab_path:    str = "results/api_vocabulary.json",
    memory_dim:    int = 128,
    node_feat_dim: int = 775,
    **kwargs,
) -> SemanticMemory:
    """
    Build SemanticMemory from the saved vocabulary.

    Example:
        memory = build_semantic_memory()
        # SemanticMemory(vocab=325, memory_dim=128, output_dim=903)
    """
    vocab = load_vocab(vocab_path)
    if vocab is None:
        raise FileNotFoundError(
            f"Vocabulary not found at '{vocab_path}'.\n"
            "Run:  python scripts/extract_vocab.py"
        )
    return SemanticMemory(
        vocab_size=vocab.size(),
        memory_dim=memory_dim,
        node_feat_dim=node_feat_dim,
        **kwargs,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== SemanticMemory Smoke Test ===\n")

    memory = build_semantic_memory()
    print(f"Module  : {memory}")
    print(f"Out dim : {memory.output_dim}")

    N = 32
    api_idx   = torch.randint(2, 325, (N,))
    node_feat = torch.randn(N, 775)
    timestamps = torch.linspace(0, 100, N)

    # ── forward (read + concat) ───────────────────────────────────
    enriched = memory(api_idx, node_feat, timestamps)
    assert enriched.shape == (N, 903), f"Shape mismatch: {enriched.shape}"
    print(f"\nForward : {list(node_feat.shape)} → {list(enriched.shape)}  ✅")

    # ── write (post-optimizer step) ───────────────────────────────
    memory.write(api_idx, node_feat.detach())
    print(f"Write   : memory updated ✅")

    # ── stats ─────────────────────────────────────────────────────
    stats = memory.memory_stats()
    print(f"\nMemory stats after write:")
    for k, v in stats.items():
        print(f"  {k:<14}: {v:.6f}")

    # ── reset ─────────────────────────────────────────────────────
    memory.reset_memory()
    print(f"\nReset   : memory cleared ✅")

    print("\n✅ semantic_memory.py is working correctly!")

