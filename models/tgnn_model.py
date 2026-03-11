"""
models/tgnn_model.py

Memory-Augmented Heterogeneous Temporal GNN for Ransomware Detection.

Dimension flow:
    API nodes:  [N, 775] → [Memory] → [N, 903] → [InputProj] → [N, 128]
                → [HeteroConv × L] → [N, 768] → [LSTM] → [N, 256]
                → [AttnPool] → [B, 256]
    Resources:  [N, 6]   → [InputProj] → [N, 64]
                → [HeteroConv × L] → [N, 384] → [MeanPool] → [B, 64]
    Combined:   [B, 256 + 4×64] = [B, 512] → [MLP] → [B, 3]

Author: Abhishek Menon
Date:   2026-03-04
"""
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import config
from models.temporal_layers import (
    HeteroTemporalConv,
    TemporalLSTM,
    TemporalAttentionPooling,
)
from models.semantic_memory import SemanticMemory, build_semantic_memory
from models.memory_utils import memory_regularization_loss
from graph_construction.vocabulary import load_vocab


# ══════════════════════════════════════════════════════════════════════════════
#  HeteroTemporalGNN
# ══════════════════════════════════════════════════════════════════════════════

class HeteroTemporalGNN(nn.Module):
    """
    Memory-augmented Heterogeneous Temporal GNN for ransomware detection.

    New vs previous version:
        - SemanticMemory module (Phase 2) enriches API nodes before GNN
        - API input dim: 775 → 903  (775 static + 128 memory)
        - Two new training-loop helpers: update_memory() and memory_loss()
        - Everything else (GNN, LSTM, pooling, classifier) unchanged
    """

    def __init__(
        self,
        node_types:          Optional[List[str]]                    = None,
        edge_types:          Optional[List[Tuple[str, str, str]]]   = None,
        node_dims:           Optional[Dict[str, int]]               = None,
        hidden_dim_api:      int   = 128,
        hidden_dim_resource: int   = 64,
        num_gnn_layers:      int   = None,
        lstm_hidden_dim:     int   = None,
        lstm_num_layers:     int   = None,
        num_classes:         int   = None,
        spatial_layer:       Optional[str]  = None,
        num_attention_heads: int   = None,
        temporal_heads:      int   = None,
        dropout:             float = None,
        lstm_dropout:        float = None,
        bidirectional:       Optional[bool] = None,
        # ── memory ──────────────────────────────────────────────────────────
        use_memory:  bool = True,
        memory_dim:  int  = 128,
        vocab_path:  str  = "results/api_vocabulary.json",
    ):
        super().__init__()

        # ── resolve config defaults ───────────────────────────────────────────
        node_types          = node_types          or config.NODE_TYPES
        edge_types          = edge_types          or config.EDGE_TYPES
        num_gnn_layers      = num_gnn_layers      or config.NUM_GNN_LAYERS
        lstm_hidden_dim     = lstm_hidden_dim     or config.LSTM_HIDDEN_DIM
        lstm_num_layers     = lstm_num_layers     or config.LSTM_NUM_LAYERS
        num_classes         = num_classes         or config.NUM_CLASSES
        spatial_layer       = spatial_layer       or config.SPATIAL_LAYER
        num_attention_heads = num_attention_heads or config.NUM_ATTENTION_HEADS
        temporal_heads      = temporal_heads      or config.TEMPORAL_HEADS
        dropout             = dropout             if dropout      is not None else config.DROPOUT
        lstm_dropout        = lstm_dropout        if lstm_dropout is not None else config.LSTM_DROPOUT
        bidirectional       = bidirectional       if bidirectional is not None else config.BIDIRECTIONAL

        # base node dims (copy so we don't mutate caller)
        if node_dims is None:
            node_dims = {
                "api":      config.API_NODE_DIM,       # 775
                "file":     config.RESOURCE_NODE_DIM,  # 6
                "registry": config.RESOURCE_NODE_DIM,
                "process":  config.RESOURCE_NODE_DIM,
                "network":  config.RESOURCE_NODE_DIM,
            }
        else:
            node_dims = dict(node_dims)

        # ── store hyper-params ────────────────────────────────────────────────
        self.node_types          = node_types
        self.edge_types          = edge_types
        self.hidden_dim_api      = hidden_dim_api
        self.hidden_dim_resource = hidden_dim_resource
        self.num_gnn_layers      = num_gnn_layers
        self.lstm_hidden_dim     = lstm_hidden_dim
        self.num_classes         = num_classes
        self.num_attention_heads = num_attention_heads
        self.temporal_heads      = temporal_heads
        self.dropout             = dropout
        self.bidirectional       = bidirectional
        self.use_memory          = use_memory
        self.memory_dim          = memory_dim

        # ── derived dims ──────────────────────────────────────────────────────
        # output of one GNN layer per node type (heads are concatenated)
        self.api_out_dim_per_layer      = hidden_dim_api      * num_attention_heads  # 128×6=768
        self.resource_out_dim_per_layer = hidden_dim_resource * num_attention_heads  # 64×6=384
        self.lstm_output_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim  # 256

        # ── sanity checks ─────────────────────────────────────────────────────
        if self.lstm_output_dim % temporal_heads != 0:
            raise ValueError(
                f"lstm_output_dim ({self.lstm_output_dim}) must be divisible "
                f"by temporal_heads ({temporal_heads}). "
                f"Got {self.lstm_output_dim} % {temporal_heads} = "
                f"{self.lstm_output_dim % temporal_heads}."
            )

        # ── which node types receive messages ─────────────────────────────────
        self.receiving_types = {dst for _, _, dst in edge_types}

        # ══════════════════════════════════════════════════════════════════════
        #  Phase 2 — Semantic Memory
        # ══════════════════════════════════════════════════════════════════════
        if use_memory:
            self.memory = build_semantic_memory(
                vocab_path    = vocab_path,
                memory_dim    = memory_dim,
                node_feat_dim = node_dims["api"],   # 775
            )
            # API input dimension is now enriched
            node_dims["api"] = node_dims["api"] + memory_dim   # 775 + 128 = 903
            print(f"[Model] SemanticMemory ON  │ API input: "
                  f"{node_dims['api'] - memory_dim} + {memory_dim} = {node_dims['api']}")
        else:
            self.memory = None
            print(f"[Model] SemanticMemory OFF │ API input: {node_dims['api']}")

        self.node_dims = node_dims   # final dims used by input projections

        # ══════════════════════════════════════════════════════════════════════
        #  1.  Input projections (one per node type)
        # ══════════════════════════════════════════════════════════════════════
        self.input_projections = nn.ModuleDict()
        for nt in node_types:
            in_dim  = node_dims[nt]
            out_dim = hidden_dim_api if nt == "api" else hidden_dim_resource
            self.input_projections[nt] = nn.Sequential(
                nn.Linear(in_dim, out_dim),   # e.g. 903→128 or 6→64
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # ══════════════════════════════════════════════════════════════════════
        #  2.  Heterogeneous GNN layers
        # ══════════════════════════════════════════════════════════════════════
        self.gnn_layers  = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # track actual dims fed into each layer
        current_dims: Dict[str, int] = {
            "api":      hidden_dim_api,
            "file":     hidden_dim_resource,
            "registry": hidden_dim_resource,
            "process":  hidden_dim_resource,
            "network":  hidden_dim_resource,
        }

        for _ in range(num_gnn_layers):
            out_channels_dict = {
                "api":      hidden_dim_api,
                "file":     hidden_dim_resource,
                "registry": hidden_dim_resource,
                "process":  hidden_dim_resource,
                "network":  hidden_dim_resource,
            }

            # edge dims: only temporal api→api edges carry features
            edge_dim_dict: Dict[Tuple, int] = {}
            for et in edge_types:
                src, rel, dst = et
                edge_dim_dict[et] = (
                    config.TEMPORAL_EDGE_DIM
                    if rel == "temporal" and src == "api" and dst == "api"
                    else 0
                )

            self.gnn_layers.append(
                HeteroTemporalConv(
                    in_channels_dict  = dict(current_dims),
                    out_channels_dict = out_channels_dict,
                    edge_types        = edge_types,
                    layer_type        = spatial_layer,
                    num_heads         = num_attention_heads,
                    edge_dim_dict     = edge_dim_dict,
                    dropout           = dropout,
                    aggr              = "sum",
                )
            )

            # LayerNorm only for node types that receive messages
            norm_dict = nn.ModuleDict()
            for nt in node_types:
                if nt in self.receiving_types:
                    norm_dim = (
                        self.api_out_dim_per_layer
                        if nt == "api"
                        else self.resource_out_dim_per_layer
                    )
                    norm_dict[nt] = nn.LayerNorm(norm_dim)
            self.norm_layers.append(norm_dict)

            # update dims for next layer
            for nt in node_types:
                if nt in self.receiving_types:
                    current_dims[nt] = (
                        self.api_out_dim_per_layer
                        if nt == "api"
                        else self.resource_out_dim_per_layer
                    )

        # ══════════════════════════════════════════════════════════════════════
        #  3.  Temporal LSTM on API node sequence
        # ══════════════════════════════════════════════════════════════════════
        self.temporal_lstm = TemporalLSTM(
            input_dim     = self.api_out_dim_per_layer,  # 768
            hidden_dim    = lstm_hidden_dim,             # 128
            num_layers    = lstm_num_layers,
            dropout       = lstm_dropout,
            bidirectional = bidirectional,
        )

        # ══════════════════════════════════════════════════════════════════════
        #  4.  Temporal attention pooling (graph-level embedding)
        # ══════════════════════════════════════════════════════════════════════
        self.api_pooling = TemporalAttentionPooling(
            input_dim = self.lstm_output_dim,  # 256
            num_heads = temporal_heads,        # 8
        )

        # ══════════════════════════════════════════════════════════════════════
        #  5.  Resource pooling (dual-path: 64-dim or 384-dim input)
        # ══════════════════════════════════════════════════════════════════════
        self.resource_poolings_64  = nn.ModuleDict()
        self.resource_poolings_384 = nn.ModuleDict()

        resource_node_types = [nt for nt in node_types if nt != "api"]
        for nt in resource_node_types:
            # Path A: node type never received GNN messages → still 64-dim
            self.resource_poolings_64[nt] = nn.Sequential(
                nn.Linear(hidden_dim_resource, hidden_dim_resource),   # 64→64
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # Path B: node type received GNN messages → 384-dim
            self.resource_poolings_384[nt] = nn.Sequential(
                nn.Linear(self.resource_out_dim_per_layer, hidden_dim_resource),  # 384→64
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # ══════════════════════════════════════════════════════════════════════
        #  6.  Classifier — shared trunk + 3 per-class heads       [CHANGE 1]
        #      (replaces single self.classifier MLP, but keeps total_embedding_dim)
        # ══════════════════════════════════════════════════════════════════════
        num_resource_types       = len(resource_node_types)  # 4
        self.total_embedding_dim = (
            self.lstm_output_dim + num_resource_types * hidden_dim_resource
        )   # 256 + 4×64 = 512

        behav_dim = (num_resource_types + 1) * 3   # 5 node types × 3 stats = 15
        trunk_in  = self.total_embedding_dim + behav_dim  # 512 + 15 = 527

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
        self.head_benign     = nn.Linear(128, 1)
        self.head_malware    = nn.Linear(128, 1)

        print(f"[Model] Total embedding dim : {self.total_embedding_dim}")
        print(f"[Model] Classifier          : shared_trunk({trunk_in}→256→128) + 3 heads")

    # ══════════════════════════════════════════════════════════════════════════
    #  forward
    # ══════════════════════════════════════════════════════════════════════════

    def forward(
        self,
        x_dict:          Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        batch_dict:      Dict[str, torch.Tensor],
        edge_attr_dict:  Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        timestamps_dict: Optional[Dict[str, torch.Tensor]] = None,
        api_indices:     Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
            device = next(self.parameters()).device
            B = int(max(v.max().item() for v in batch_dict.values())) + 1

            # ── memory enrichment ────────────────────────────────────────────────
            if self.use_memory and self.memory is not None and api_indices is not None:
                api_ts = (
                    timestamps_dict.get("api", None)
                    if timestamps_dict is not None else None
                )
                x_dict = dict(x_dict)          # shallow copy — don't mutate caller
                x_dict["api"] = self.memory(
                    api_indices.to(device),
                    x_dict["api"],
                    api_ts,
                )

            # ── 1. Input projections ──────────────────────────────────────────────
            # api: 903→128, resources: 6→64
            h_dict: Dict[str, torch.Tensor] = {}
            for nt, x in x_dict.items():
                h_dict[nt] = (self.input_projections[nt](x)
                            if nt in self.input_projections else x)

            # ── 2. Stacked HeteroGNN layers ───────────────────────────────────────
            for gnn_layer, norm_dict in zip(self.gnn_layers, self.norm_layers):
                h_out = gnn_layer(
                    x_dict          = h_dict,
                    edge_index_dict = edge_index_dict,
                    edge_attr_dict  = edge_attr_dict,
                )

                for nt in h_dict:
                    if nt in h_out:
                        # LayerNorm
                        if nt in norm_dict:
                            h_out[nt] = norm_dict[nt](h_out[nt])
                        # Residual (only when dims match)
                        if h_dict[nt].size(-1) == h_out[nt].size(-1):
                            h_out[nt] = h_out[nt] + h_dict[nt]
                    else:
                        # Node type received no messages — keep previous embedding
                        h_out[nt] = h_dict[nt]

                for nt in h_out:
                    h_out[nt] = F.relu(h_out[nt])
                    h_out[nt] = F.dropout(h_out[nt], p=self.dropout, training=self.training)

                h_dict = h_out

            # ── 3. Temporal LSTM on API nodes ─────────────────────────────────────
            api_batch = batch_dict["api"]
            api_ts    = (
                timestamps_dict.get("api", None)
                if timestamps_dict is not None else None
            )
            api_temporal = self.temporal_lstm(
                x = h_dict["api"],
                batch      = api_batch,
                timestamps = api_ts,
            )  # [N_api, lstm_output_dim]

            # ── 4. Attention pooling → graph-level API embedding ──────────────────
            api_graph_emb = self.api_pooling(
                x     = api_temporal,
                batch = api_batch,
            )  # [B, 256]

            # ── 5. Resource pooling per resource node type ────────────────────────
            resource_node_types = [nt for nt in self.node_types if nt != "api"]
            resource_embeddings: List[torch.Tensor] = []

            for nt in resource_node_types:
                if nt not in h_dict or nt not in batch_dict:
                    resource_embeddings.append(
                        torch.zeros(B, self.hidden_dim_resource, device=device)
                    )
                    continue

                h_res      = h_dict[nt]
                batch_res  = batch_dict[nt]
                actual_dim = h_res.size(-1)

                pooled = []
                for b in range(B):
                    mask = batch_res == b
                    pooled.append(
                        h_res[mask].mean(0) if mask.any()
                        else torch.zeros(actual_dim, device=device)
                    )
                pooled = torch.stack(pooled)  # [B, actual_dim]

                if actual_dim == self.hidden_dim_resource:           # 64
                    pooled = self.resource_poolings_64[nt](pooled)
                elif actual_dim == self.resource_out_dim_per_layer:  # 384
                    pooled = self.resource_poolings_384[nt](pooled)
                else:
                    # safety net: project from unexpected dim — avoids crash
                    emergency = nn.Linear(actual_dim, self.hidden_dim_resource).to(device)
                    pooled    = emergency(pooled)

                resource_embeddings.append(pooled)  # [B, 64]

            # ── 6. Combine embeddings ─────────────────────────────────────────────
            combined_embedding = torch.cat(
                [api_graph_emb, *resource_embeddings], dim=1
            )  # [B, 512]

            # ── Behavioural statistics                                  [CHANGE 2]
            # 5 node types × 3 stats (mean-activation, std-activation, log1p-count)
            # = 15 features, making trunk_in = 512 + 15 = 527
            behav_parts: List[torch.Tensor] = []
            for nt in ["api"] + resource_node_types:
                if nt in h_dict and nt in batch_dict:
                    h_nt = h_dict[nt]
                    b_nt = batch_dict[nt]
                    means, stds, counts = [], [], []
                    for b in range(B):
                        mask = b_nt == b
                        if mask.any():
                            feats = h_nt[mask]
                            means.append(feats.mean())
                            stds.append(
                                feats.std()
                                if feats.size(0) > 1
                                else feats.new_zeros(1).squeeze()
                            )
                            counts.append(torch.log1p(mask.float().sum()))
                        else:
                            means.append(h_nt.new_zeros(1).squeeze())
                            stds.append(h_nt.new_zeros(1).squeeze())
                            counts.append(h_nt.new_zeros(1).squeeze())
                    behav_parts.extend([
                        torch.stack(means),
                        torch.stack(stds),
                        torch.stack(counts),
                    ])
                else:
                    behav_parts.extend([
                        torch.zeros(B, device=device),
                        torch.zeros(B, device=device),
                        torch.zeros(B, device=device),
                    ])

            behav_feats = torch.stack(behav_parts, dim=-1)  # [B, 15]

            # ── Shared trunk + 3 per-class heads             [CHANGE 1 + CHANGE 2]
            x      = torch.cat([combined_embedding, behav_feats], dim=-1)  # [B, 527]
            z      = self.shared_trunk(x)
            logits = torch.cat([
                self.head_ransomware(z),
                self.head_benign(z),
                self.head_malware(z),
            ], dim=-1)  # [B, 3]
            return logits


    # ──────────────────────────────────────────────────────────────────────────
    #  Training-loop helpers
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_memory(
        self,
        api_indices:   torch.LongTensor,
        node_features: torch.Tensor,
        momentum:      float = 0.9,
    ) -> None:
        """
        Write new behavioral knowledge into memory bank.

        Call pattern:
            # after loss.backward() and optimizer.step()
            model.update_memory(api_indices, raw_api_features)
        """
        if self.use_memory and self.memory is not None:
            self.memory.write(
                api_indices   = api_indices.to(node_features.device),
                node_features = node_features.detach(),
                momentum      = momentum,
            )

    def memory_loss(self, lambda_reg: float = 1e-4) -> torch.Tensor:
        """
        L2 regularisation on memory bank vectors.

        Call pattern:
            total_loss = task_loss + model.memory_loss()
        """
        if self.use_memory and self.memory is not None:
            return memory_regularization_loss(
                self.memory.memory_bank.weight,
                lambda_reg,
            )
        return torch.tensor(0.0, requires_grad=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

def create_hetero_tgnn(
    spatial_layer: str           = "gat",
    use_memory:    bool          = True,
    device:        Optional[str] = None,
    **kwargs,
) -> HeteroTemporalGNN:
    """
    Factory function.

    Example:
        model = create_hetero_tgnn(spatial_layer='gat', device='mps')
    """
    model = HeteroTemporalGNN(
        spatial_layer = spatial_layer,
        use_memory    = use_memory,
        **kwargs,
    )
    if device is not None:
        model = model.to(device)
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import config

    print("=" * 60)
    print("  HeteroTemporalGNN smoke test")
    print("=" * 60)

    B      = 4    # batch size
    N_api  = 80   # total API nodes across batch
    N_res  = 20   # resource nodes per type

    model = create_hetero_tgnn(spatial_layer="gat", use_memory=True)
    model.eval()
    device = torch.device("cpu")

    x_dict = {
        "api":      torch.randn(N_api, 775),
        "file":     torch.randn(N_res, 6),
        "registry": torch.randn(N_res, 6),
        "process":  torch.randn(N_res, 6),
        "network":  torch.randn(N_res, 6),
    }
    edge_index_dict = {
        ("api", "temporal",  "api"):
            torch.randint(0, N_api, (2, 200)),
        ("api", "accesses",  "file"):
            torch.stack([torch.randint(0, N_api, (100,)),
                         torch.randint(0, N_res,  (100,))]),
        ("api", "modifies",  "registry"):
            torch.stack([torch.randint(0, N_api, (100,)),
                         torch.randint(0, N_res,  (100,))]),
        ("api", "spawns",    "process"):
            torch.stack([torch.randint(0, N_api, (100,)),
                         torch.randint(0, N_res,  (100,))]),
        ("api", "connects",  "network"):
            torch.stack([torch.randint(0, N_api, (100,)),
                         torch.randint(0, N_res,  (100,))]),
    }
    batch_dict = {
        "api":      torch.repeat_interleave(torch.arange(B), N_api // B),
        "file":     torch.repeat_interleave(torch.arange(B), N_res // B),
        "registry": torch.repeat_interleave(torch.arange(B), N_res // B),
        "process":  torch.repeat_interleave(torch.arange(B), N_res // B),
        "network":  torch.repeat_interleave(torch.arange(B), N_res // B),
    }
    timestamps_dict = {"api": torch.linspace(0, 100, N_api)}
    api_indices     = torch.randint(2, 325, (N_api,))

    with torch.no_grad():
        logits = model(
            x_dict          = x_dict,
            edge_index_dict = edge_index_dict,
            batch_dict      = batch_dict,
            timestamps_dict = timestamps_dict,
            api_indices     = api_indices,
        )
    print(f"\n[Smoke] logits shape : {logits.shape}  (expected [{B}, 3])")
    assert logits.shape == (B, 3), f"Shape mismatch: {logits.shape}"

    model.update_memory(api_indices, x_dict["api"])
    print(f"[Smoke] Memory write ✓")

    mem_loss = model.memory_loss()
    print(f"[Smoke] Memory loss  {mem_loss.item():.8f} ✓")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Smoke] Parameters   total={total:,}  trainable={trainable:,}")
    print("\n✅  tgnn_model.py working correctly!")
