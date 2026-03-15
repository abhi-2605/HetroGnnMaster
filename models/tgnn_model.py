"""
models/tgnn_model.py

Schema-aware Memory-Augmented Heterogeneous Temporal GNN for ransomware detection.

Phase 1 upgrades:
- active-schema driven node/edge setup
- fixed type-slot combiner (stable across batches)
- explicit TemporalLSTM branch preserved
- relation-complete message passing through HeteroTemporalConv
- safer memory handling
- defensive pooling-dim checks
- LSTM path diagnostic
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import config
from models.temporal_layers import (
    HeteroTemporalConv,
    TemporalAttentionPooling,
    TemporalLSTM,
)
from models.semantic_memory import build_semantic_memory
from models.memory_utils import memory_regularization_loss
from utils.schema_manager import ensure_active_schema
from utils.schema_utils import (
    get_active_schema,
    get_canonical_edge_dims,
    get_canonical_node_dims,
    get_edge_types,
    get_node_types,
    get_nonempty_edge_types,
)


EdgeType = Tuple[str, str, str]


# ══════════════════════════════════════════════════════════════════════════════
#  Type-slot combiner
# ══════════════════════════════════════════════════════════════════════════════

class TypeSlotAttentionCombiner(nn.Module):
    """
    Combine per-type pooled embeddings with stable slot semantics.

    Input:
        type_embeddings: [B, T, D]
        type_mask      : [B, T]  True for present, False for absent

    Output:
        graph_embedding: [B, D]
    """

    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, type_embeddings: torch.Tensor, type_mask: torch.Tensor) -> torch.Tensor:
        """
        type_embeddings: [B, T, D]
        type_mask:       [B, T] (True = present, False = absent)
        """
        B, _, _ = type_embeddings.shape

        query = self.query.expand(B, -1, -1)  # [B, 1, D]

        # True means "ignore" for MultiheadAttention
        key_padding_mask = ~type_mask  # [B, T]

        out, _ = self.attn(
            query=query,
            key=type_embeddings,
            value=type_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # [B, 1, D]

        out = out.squeeze(1)  # [B, D]
        out = self.norm(out)
        out = self.dropout(out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  HeteroTemporalGNN
# ══════════════════════════════════════════════════════════════════════════════

class HeteroTemporalGNN(nn.Module):
    """
    Schema-aware memory-augmented heterogeneous temporal GNN.

    Important:
    - TemporalLSTM remains the explicit order-sensitive temporal branch.
    - HeteroTemporalConv handles graph relations, including api->temporal->api,
      but does not replace sequence modeling.
    """

    def __init__(
        self,
        schema: Optional[Dict] = None,
        hidden_dim_api: int = 128,
        hidden_dim_resource: int = 64,
        pooled_type_dim: int = 128,
        num_gnn_layers: Optional[int] = None,
        lstm_hidden_dim: Optional[int] = None,
        lstm_num_layers: Optional[int] = None,
        num_classes: Optional[int] = None,
        spatial_layer: Optional[str] = None,
        num_attention_heads: Optional[int] = None,
        temporal_heads: Optional[int] = None,
        type_combiner_heads: int = 4,
        dropout: Optional[float] = None,
        lstm_dropout: Optional[float] = None,
        bidirectional: Optional[bool] = None,
        use_memory: bool = True,
        memory_dim: int = 128,
        vocab_path: str = "results/api_vocabulary.json",
    ):
        super().__init__()

        self.schema = schema if schema is not None else ensure_active_schema()

        # ── config defaults ───────────────────────────────────────────────────
        num_gnn_layers = num_gnn_layers or config.NUM_GNN_LAYERS
        lstm_hidden_dim = lstm_hidden_dim or config.LSTM_HIDDEN_DIM
        lstm_num_layers = lstm_num_layers or config.LSTM_NUM_LAYERS
        num_classes = num_classes or config.NUM_CLASSES
        spatial_layer = spatial_layer or config.SPATIAL_LAYER
        num_attention_heads = num_attention_heads or config.NUM_ATTENTION_HEADS
        temporal_heads = temporal_heads or config.TEMPORAL_HEADS
        dropout = config.DROPOUT if dropout is None else dropout
        lstm_dropout = config.LSTM_DROPOUT if lstm_dropout is None else lstm_dropout
        bidirectional = config.BIDIRECTIONAL if bidirectional is None else bidirectional

        # ── schema-derived structure ──────────────────────────────────────────
        self.node_types: List[str] = get_node_types(self.schema)
        self.edge_types: List[EdgeType] = get_edge_types(self.schema)
        self.nonempty_edge_types: List[EdgeType] = get_nonempty_edge_types(self.schema)
        self.node_dims: Dict[str, int] = get_canonical_node_dims(self.schema)
        self.edge_dims: Dict[EdgeType, Optional[int]] = get_canonical_edge_dims(self.schema)

        self.hidden_dim_api = hidden_dim_api
        self.hidden_dim_resource = hidden_dim_resource
        self.pooled_type_dim = pooled_type_dim
        self.num_gnn_layers = num_gnn_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_classes = num_classes
        self.num_attention_heads = num_attention_heads
        self.temporal_heads = temporal_heads
        self.type_combiner_heads = type_combiner_heads
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_memory = use_memory
        self.memory_dim = memory_dim

        if "api" not in self.node_types:
            raise ValueError("Active schema must contain 'api' node type.")

        # ── explicit stable type slot order ───────────────────────────────────
        self.type_slot_order = list(self.node_types)
        self.resource_node_types = [nt for nt in self.node_types if nt != "api"]

        # ── memory enrichment ─────────────────────────────────────────────────
        base_api_dim = self.node_dims["api"]

        if use_memory:
            self.memory = build_semantic_memory(
                vocab_path=vocab_path,
                memory_dim=memory_dim,
                node_feat_dim=base_api_dim,
            )
            self.node_dims["api"] = base_api_dim + memory_dim
            print(
                f"[Model] SemanticMemory ON  │ API input: "
                f"{base_api_dim} + {memory_dim} = {self.node_dims['api']}"
            )
        else:
            self.memory = None
            print(f"[Model] SemanticMemory OFF │ API input: {base_api_dim}")

        # ── derived dims ──────────────────────────────────────────────────────
        self.api_out_dim_per_layer = hidden_dim_api * num_attention_heads
        self.resource_out_dim_per_layer = hidden_dim_resource * num_attention_heads
        self.lstm_output_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim

        if self.lstm_output_dim % temporal_heads != 0:
            raise ValueError(
                f"lstm_output_dim ({self.lstm_output_dim}) must be divisible by "
                f"temporal_heads ({temporal_heads})."
            )

        # ══════════════════════════════════════════════════════════════════════
        # 1. Input adapters (schema-driven)
        # ══════════════════════════════════════════════════════════════════════
        self.input_projections = nn.ModuleDict()
        for nt in self.node_types:
            in_dim = self.node_dims[nt]
            out_dim = hidden_dim_api if nt == "api" else hidden_dim_resource
            self.input_projections[nt] = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # ══════════════════════════════════════════════════════════════════════
        # 2. Heterogeneous GNN layers
        # ══════════════════════════════════════════════════════════════════════
        self.gnn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        current_dims: Dict[str, int] = {}
        for nt in self.node_types:
            current_dims[nt] = hidden_dim_api if nt == "api" else hidden_dim_resource

        out_channels_dict: Dict[str, int] = {}
        for nt in self.node_types:
            out_channels_dict[nt] = hidden_dim_api if nt == "api" else hidden_dim_resource

        edge_dim_dict = {
            et: int(self.edge_dims.get(et, 0) or 0)
            for et in self.edge_types
        }

        receiving_types = {dst for _, _, dst in self.nonempty_edge_types}

        for _ in range(num_gnn_layers):
            layer = HeteroTemporalConv(
                in_channels_dict=dict(current_dims),
                out_channels_dict=dict(out_channels_dict),
                edge_types=self.edge_types,
                layer_type=spatial_layer,
                num_heads=num_attention_heads,
                edge_dim_dict=edge_dim_dict,
                dropout=dropout,
                aggr="sum",
            )
            self.gnn_layers.append(layer)

            norm_dict = nn.ModuleDict()
            for nt in self.node_types:
                if nt in receiving_types:
                    norm_dim = (
                        self.api_out_dim_per_layer if nt == "api"
                        else self.resource_out_dim_per_layer
                    )
                    norm_dict[nt] = nn.LayerNorm(norm_dim)
            self.norm_layers.append(norm_dict)

            for nt in self.node_types:
                if nt in receiving_types:
                    current_dims[nt] = (
                        self.api_out_dim_per_layer if nt == "api"
                        else self.resource_out_dim_per_layer
                    )

        # Actual final hidden dims per node type after GNN stack.
        # Some types may remain at 64 if they never receive messages.
        self.final_hidden_dims = dict(current_dims)

        # ══════════════════════════════════════════════════════════════════════
        # 3. Explicit temporal branch on API sequence
        # ══════════════════════════════════════════════════════════════════════
        self.temporal_lstm = TemporalLSTM(
            input_dim=self.api_out_dim_per_layer,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        self.api_pooling = TemporalAttentionPooling(
            input_dim=self.lstm_output_dim,
            num_heads=temporal_heads,
        )

        # ══════════════════════════════════════════════════════════════════════
        # 4. Per-type pooling projections to common pooled dim
        # ══════════════════════════════════════════════════════════════════════
        self.type_pool_projections = nn.ModuleDict()

        # API pooled vector comes from LSTM branch
        self.type_pool_projections["api"] = nn.Sequential(
            nn.Linear(self.lstm_output_dim, pooled_type_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Other types pooled after GNN branch using ACTUAL final dim
        for nt in self.resource_node_types:
            in_dim = self.final_hidden_dims[nt]
            self.type_pool_projections[nt] = nn.Sequential(
                nn.Linear(in_dim, pooled_type_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # ══════════════════════════════════════════════════════════════════════
        # 5. Stable-slot type combiner
        # ══════════════════════════════════════════════════════════════════════
        self.type_combiner = TypeSlotAttentionCombiner(
            input_dim=pooled_type_dim,
            num_heads=type_combiner_heads,
            dropout=dropout,
        )

        # ══════════════════════════════════════════════════════════════════════
        # 6. Behavioral summary features
        # ══════════════════════════════════════════════════════════════════════
        self.behav_dim = len(self.type_slot_order) * 3

        # ══════════════════════════════════════════════════════════════════════
        # 7. Classifier
        # ══════════════════════════════════════════════════════════════════════
        trunk_in = pooled_type_dim + self.behav_dim

        self.shared_trunk = nn.Sequential(
            nn.Linear(trunk_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_ransomware = nn.Linear(128, 1)
        self.head_benign = nn.Linear(128, 1)
        self.head_malware = nn.Linear(128, 1)

        self.total_embedding_dim = pooled_type_dim

        print(f"[Model] Active node types     : {self.node_types}")
        print(f"[Model] Active edge types     : {self.edge_types}")
        print(f"[Model] Non-empty edge types  : {self.nonempty_edge_types}")
        print(f"[Model] Type slot order       : {self.type_slot_order}")
        print(f"[Model] Canonical node dims   : {self.node_dims}")
        print(f"[Model] Final hidden dims     : {self.final_hidden_dims}")
        print(f"[Model] Graph embedding dim   : {self.total_embedding_dim}")
        print(f"[Model] Behavioral feat dim   : {self.behav_dim}")

    # ──────────────────────────────────────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _pool_resource_type(
        self,
        h_nt: torch.Tensor,
        batch_nt: torch.Tensor,
        batch_size: int,
        node_type: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Mean-pool node embeddings of one non-api type per graph, then project.
        """
        pooled = []
        feat_dim = h_nt.size(-1)

        for b in range(batch_size):
            mask = batch_nt == b
            if mask.any():
                pooled.append(h_nt[mask].mean(dim=0))
            else:
                pooled.append(torch.zeros(feat_dim, device=device, dtype=h_nt.dtype))

        pooled = torch.stack(pooled, dim=0)

        expected_in = self.type_pool_projections[node_type][0].in_features
        if pooled.size(1) != expected_in:
            raise ValueError(
                f"Type pooling dim mismatch for '{node_type}': "
                f"pooled has dim {pooled.size(1)}, projection expects {expected_in}."
            )

        return self.type_pool_projections[node_type](pooled)

    def _build_type_slots(
        self,
        h_dict: Dict[str, torch.Tensor],
        batch_dict: Dict[str, torch.Tensor],
        api_graph_emb: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build fixed-order type slot embeddings and presence mask.

        Returns:
            type_slots: [B, T, D]
            type_mask : [B, T]
        """
        slot_embeddings = []
        slot_masks = []

        for nt in self.type_slot_order:
            if nt == "api":
                emb = self.type_pool_projections["api"](api_graph_emb)
                mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                slot_embeddings.append(emb)
                slot_masks.append(mask)
                continue

            if nt in h_dict and nt in batch_dict and batch_dict[nt].numel() > 0:
                emb = self._pool_resource_type(
                    h_nt=h_dict[nt],
                    batch_nt=batch_dict[nt],
                    batch_size=batch_size,
                    node_type=nt,
                    device=device,
                )

                present = torch.zeros(batch_size, dtype=torch.bool, device=device)
                present[batch_dict[nt].unique()] = True
            else:
                emb = torch.zeros(
                    batch_size,
                    self.pooled_type_dim,
                    device=device,
                )
                present = torch.zeros(batch_size, dtype=torch.bool, device=device)

            slot_embeddings.append(emb)
            slot_masks.append(present)

        type_slots = torch.stack(slot_embeddings, dim=1)  # [B, T, D]
        type_mask = torch.stack(slot_masks, dim=1)        # [B, T]
        return type_slots, type_mask

    def _compute_behavioral_features(
        self,
        h_dict: Dict[str, torch.Tensor],
        batch_dict: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        For each node type slot:
            [mean activation, std activation, log1p(node count)]
        """
        behav_parts: List[torch.Tensor] = []

        for nt in self.type_slot_order:
            if nt in h_dict and nt in batch_dict and batch_dict[nt].numel() > 0:
                h_nt = h_dict[nt]
                b_nt = batch_dict[nt]

                mean_per_feat = scatter(h_nt, b_nt, dim=0, dim_size=batch_size, reduce="mean")
                centered = h_nt - mean_per_feat[b_nt]
                var_per_feat = scatter(
                    centered * centered,
                    b_nt,
                    dim=0,
                    dim_size=batch_size,
                    reduce="mean",
                )
                std_per_feat = torch.sqrt(var_per_feat + 1e-8)

                g_means = mean_per_feat.mean(dim=-1)
                g_stds = std_per_feat.mean(dim=-1)

                node_ones = torch.ones(h_nt.size(0), device=device)
                g_counts = torch.log1p(
                    scatter(node_ones, b_nt, dim=0, dim_size=batch_size, reduce="sum")
                )

                behav_parts.extend([g_means, g_stds, g_counts])
            else:
                behav_parts.extend([
                    torch.zeros(batch_size, device=device),
                    torch.zeros(batch_size, device=device),
                    torch.zeros(batch_size, device=device),
                ])

        return torch.stack(behav_parts, dim=-1)  # [B, T*3]

    # ──────────────────────────────────────────────────────────────────────────
    # forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
        batch_dict: Dict[str, torch.Tensor],
        edge_attr_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
        timestamps_dict: Optional[Dict[str, torch.Tensor]] = None,
        api_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        # ── LSTM path diagnostic ──────────────────────────────────────────────
        if not hasattr(self, "_lstm_diag"):
            self._lstm_diag = {"with_api": 0, "zero_api": 0}

        api_x = x_dict.get("api", None)
        if api_x is not None and api_x.size(0) > 0:
            self._lstm_diag["with_api"] += 1
        else:
            self._lstm_diag["zero_api"] += 1

        total_seen = self._lstm_diag["with_api"] + self._lstm_diag["zero_api"]
        if total_seen % 200 == 0:
            print(
                f"[LSTM-CHECK] graphs_seen={total_seen} | "
                f"with_api={self._lstm_diag['with_api']} | "
                f"zero_api={self._lstm_diag['zero_api']}"
            )

        if "api" not in batch_dict or batch_dict["api"].numel() == 0:
            raise ValueError("Batch has no API nodes; cannot run HeteroTemporalGNN forward.")

        if self.use_memory and self.memory is not None and api_indices is None:
            raise ValueError("Memory is enabled but api_indices is missing.")

        batch_size = int(batch_dict["api"].max().item()) + 1

        # ── memory enrichment ────────────────────────────────────────────────
        if self.use_memory and self.memory is not None:
            api_ts = timestamps_dict.get("api", None) if timestamps_dict is not None else None
            x_dict = dict(x_dict)
            x_dict["api"] = self.memory(
                api_indices.to(device),
                x_dict["api"],
                api_ts,
            )

        # ── 1. input projections ─────────────────────────────────────────────
        h_dict: Dict[str, torch.Tensor] = {}
        for nt, x in x_dict.items():
            if nt in self.input_projections:
                h_dict[nt] = self.input_projections[nt](x)
            else:
                h_dict[nt] = x

        # ── 2. stacked hetero GNN layers ────────────────────────────────────
        for gnn_layer, norm_dict in zip(self.gnn_layers, self.norm_layers):
            h_out = gnn_layer(
                x_dict=h_dict,
                edge_index_dict=edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )

            for nt in h_dict:
                if nt in h_out:
                    if nt in norm_dict:
                        h_out[nt] = norm_dict[nt](h_out[nt])

                    if h_dict[nt].size(-1) == h_out[nt].size(-1):
                        h_out[nt] = h_out[nt] + h_dict[nt]
                else:
                    h_out[nt] = h_dict[nt]

            for nt in h_out:
                h_out[nt] = F.relu(h_out[nt])
                h_out[nt] = F.dropout(h_out[nt], p=self.dropout, training=self.training)

            h_dict = h_out

        # ── 3. explicit temporal branch on API nodes ────────────────────────
        api_batch = batch_dict["api"]
        api_ts = timestamps_dict.get("api", None) if timestamps_dict is not None else None

        api_temporal = self.temporal_lstm(
            x=h_dict["api"],
            batch=api_batch,
            timestamps=api_ts,
        )

        api_graph_emb = self.api_pooling(
            x=api_temporal,
            batch=api_batch,
        )  # [B, lstm_output_dim]

        # ── 4. fixed type slots + type combiner ─────────────────────────────
        type_slots, type_mask = self._build_type_slots(
            h_dict=h_dict,
            batch_dict=batch_dict,
            api_graph_emb=api_graph_emb,
            batch_size=batch_size,
            device=device,
        )

        graph_embedding = self.type_combiner(type_slots, type_mask)  # [B, pooled_type_dim]

        # ── 5. behavioral features in same stable type order ────────────────
        behav_feats = self._compute_behavioral_features(
            h_dict=h_dict,
            batch_dict=batch_dict,
            batch_size=batch_size,
            device=device,
        )  # [B, behav_dim]

        # ── 6. classifier ────────────────────────────────────────────────────
        x = torch.cat([graph_embedding, behav_feats], dim=-1)
        z = self.shared_trunk(x)

        logits = torch.cat(
            [
                self.head_ransomware(z),
                self.head_benign(z),
                self.head_malware(z),
            ],
            dim=-1,
        )
        return logits

    # ──────────────────────────────────────────────────────────────────────────
    # training-loop helpers
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_memory(
        self,
        api_indices: torch.LongTensor,
        node_features: torch.Tensor,
        momentum: float = 0.9,
    ) -> None:
        if self.use_memory and self.memory is not None:
            self.memory.write(
                api_indices=api_indices.to(node_features.device),
                node_features=node_features.detach(),
                momentum=momentum,
            )

    def memory_loss(self, lambda_reg: float = 1e-4) -> torch.Tensor:
        if self.use_memory and self.memory is not None:
            return memory_regularization_loss(
                self.memory.memory_bank.weight,
                lambda_reg,
            )
        return torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

def create_hetero_tgnn(
    schema: Optional[Dict] = None,
    use_memory: Optional[bool] = None,
    memory_dim: Optional[int] = None,
    **kwargs,
) -> HeteroTemporalGNN:
    """
    Factory for schema-aware TGNN.
    """
    schema = schema if schema is not None else get_active_schema()
    use_memory = config.USE_MEMORY if use_memory is None else use_memory
    memory_dim = config.MEMORY_DIM if memory_dim is None else memory_dim

    model = HeteroTemporalGNN(
        schema=schema,
        use_memory=use_memory,
        memory_dim=memory_dim,
        vocab_path=str(config.VOCAB_SAVE_PATH),
        **kwargs,
    )
    return model


if __name__ == "__main__":
    schema = ensure_active_schema()
    model = create_hetero_tgnn(schema=schema, use_memory=config.USE_MEMORY)
    print(model)