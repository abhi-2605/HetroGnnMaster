# """
# Heterogeneous Temporal Graph Neural Network (Hetero-TGNN) for Ransomware Detection.

# Architecture:
# 1. Heterogeneous node inputs (HeteroData-style dictionaries):
#    - 'api' nodes: 775-dim (768 LLM + 7 metadata)
#    - 'file', 'registry', 'process', 'network' nodes: 6-dim each
#    - Temporal edges 'api'→'api': 14-dim edge features

# 2. HeteroTemporalConv layers (from temporal_layers.py):
#    - Edge-type-specific TemporalGATConv / TemporalTransformerConv
#    - Full use of source/destination node embeddings and temporal edge features
#    - Proper bipartite handling for cross-type edges (api→file, process→api, etc.)

# 3. Temporal LSTM on API node sequences:
#    - Input: enriched API embeddings from last HeteroTemporalConv layer
#    - Sorted by per-node timestamps inside each graph

# 4. Temporal attention pooling on API nodes:
#    - Multi-head attention over time to produce one embedding per graph

# 5. Optional resource node pooling:
#    - Mean pooling per node type (file/registry/process/network)
#    - Projected and concatenated with API graph embedding

# 6. Classification head:
#    - MLP mapping combined graph embedding to 3 classes:
#      [Ransomware, Benign, Other Malware]

# Author: Abhishek Menon
# Date: 2026-02-13 (FOOLPROOF FIXED)
# """

# from typing import Dict, Tuple, List, Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from pathlib import Path
# import sys

# PROJECT_ROOT = Path(__file__).parent.parent
# sys.path.append(str(PROJECT_ROOT))

# import config
# from models.temporal_layers import (
#     HeteroTemporalConv,
#     TemporalLSTM,
#     TemporalAttentionPooling,
# )


# class HeteroTemporalGNN(nn.Module):
#     """
#     Fully heterogeneous temporal GNN for ransomware detection.

#     Expects HeteroData-style inputs broken out by node_type and edge_type.
#     """

#     def __init__(
#         self,
#         node_types: Optional[List[str]] = None,
#         edge_types: Optional[List[Tuple[str, str, str]]] = None,
#         node_dims: Optional[Dict[str, int]] = None,
#         hidden_dim_api: int = 128,
#         hidden_dim_resource: int = 64,
#         num_gnn_layers: int = None,
#         lstm_hidden_dim: int = None,
#         lstm_num_layers: int = None,
#         num_classes: int = None,
#         spatial_layer: Optional[str] = None,  # 'gat' or 'transformer'
#         num_attention_heads: int = None,
#         temporal_heads: int = None,
#         dropout: float = None,
#         lstm_dropout: float = None,
#         bidirectional: Optional[bool] = None,
#     ):
#         super().__init__()

#         # ----------------------------
#         # Config defaults
#         # ----------------------------
#         node_types = node_types or config.NODE_TYPES
#         edge_types = edge_types or config.EDGE_TYPES

#         if node_dims is None:
#             node_dims = {
#                 "api": config.API_NODE_DIM,
#                 "file": config.RESOURCE_NODE_DIM,
#                 "registry": config.RESOURCE_NODE_DIM,
#                 "process": config.RESOURCE_NODE_DIM,
#                 "network": config.RESOURCE_NODE_DIM,
#             }

#         num_gnn_layers = num_gnn_layers or config.NUM_GNN_LAYERS
#         lstm_hidden_dim = lstm_hidden_dim or config.LSTM_HIDDEN_DIM
#         lstm_num_layers = lstm_num_layers or config.LSTM_NUM_LAYERS
#         num_classes = num_classes or config.NUM_CLASSES
#         spatial_layer = spatial_layer or config.SPATIAL_LAYER
#         num_attention_heads = num_attention_heads or config.NUM_ATTENTION_HEADS
#         temporal_heads = temporal_heads or config.TEMPORAL_HEADS
#         dropout = dropout or config.DROPOUT
#         lstm_dropout = lstm_dropout or config.LSTM_DROPOUT
#         bidirectional = bidirectional if bidirectional is not None else config.BIDIRECTIONAL

#         self.node_types = node_types
#         self.edge_types = edge_types
#         self.node_dims = node_dims
#         self.hidden_dim_api = hidden_dim_api
#         self.hidden_dim_resource = hidden_dim_resource
#         self.num_gnn_layers = num_gnn_layers
#         self.lstm_hidden_dim = lstm_hidden_dim
#         self.lstm_num_layers = lstm_num_layers
#         self.num_classes = num_classes
#         self.spatial_layer = spatial_layer
#         self.num_attention_heads = num_attention_heads
#         self.temporal_heads = temporal_heads
#         self.dropout = dropout
#         self.lstm_dropout = lstm_dropout
#         self.bidirectional = bidirectional

#         # API and resource output dims per layer (after heads concatenation)
#         self.api_out_dim_per_layer = hidden_dim_api * num_attention_heads  
#         self.resource_out_dim_per_layer = hidden_dim_resource * num_attention_heads  
#         print(f"🔍 API out dim: {self.api_out_dim_per_layer}")      # Should be 768
#         print(f"🔍 Resource out dim: {self.resource_out_dim_per_layer}")  # Should be 384
#         self.lstm_output_dim = (
#             lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim
#         )

#         # Dimension check for attention pooling
#         if self.lstm_output_dim % self.temporal_heads != 0:
#             raise ValueError(
#                 f"LSTM output dim ({self.lstm_output_dim}) must be divisible "
#                 f"by temporal_heads ({self.temporal_heads})."
#             )

#         # Identify which node types receive messages (will be updated by GNN)
#         self.receiving_types = set()
#         for src, rel, dst in edge_types:
#             self.receiving_types.add(dst)

#         print(f"🔍 Receiving types (will have {self.resource_out_dim_per_layer}-dim): {self.receiving_types}")
#         print(f"🔍 Non-receiving types (stay at {self.hidden_dim_resource}-dim): {set(node_types) - self.receiving_types - {'api'}}")

#         # ----------------------------
#         # 1. Input linear projections per node type
#         # ----------------------------
#         self.input_projections = nn.ModuleDict()
#         for nt in node_types:
#             in_dim = node_dims[nt]
#             if nt == "api":
#                 out_dim = hidden_dim_api
#             else:
#                 out_dim = hidden_dim_resource

#             self.input_projections[nt] = nn.Sequential(
#                 nn.Linear(in_dim, out_dim),
#                 nn.LayerNorm(out_dim),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#             )

#         # ----------------------------
#         # 2. Heterogeneous temporal GNN layers
#         # ----------------------------
#         self.gnn_layers = nn.ModuleList()
#         self.norm_layers = nn.ModuleList()

#         # Track ACTUAL dimensions per node type across layers
#         current_dims = {
#             "api": hidden_dim_api,
#             "file": hidden_dim_resource,
#             "registry": hidden_dim_resource,
#             "process": hidden_dim_resource,
#             "network": hidden_dim_resource,
#         }

#         for layer_idx in range(num_gnn_layers):
#             # Out channels: what each destination node type will get (before concat)
#             out_channels_dict = {
#                 "api": hidden_dim_api,
#                 "file": hidden_dim_resource,
#                 "registry": hidden_dim_resource,
#                 "process": hidden_dim_resource,
#                 "network": hidden_dim_resource,
#             }

#             # Edge dims: only temporal API→API edges carry 14-dim features
#             edge_dim_dict = {}
#             for et in edge_types:
#                 src, rel, dst = et
#                 if rel == "temporal" and src == "api" and dst == "api":
#                     edge_dim_dict[et] = config.TEMPORAL_EDGE_DIM
#                 else:
#                     edge_dim_dict[et] = 0

#             hetero_conv = HeteroTemporalConv(
#                 in_channels_dict=current_dims,  # Use actual current dimensions
#                 out_channels_dict=out_channels_dict,
#                 edge_types=edge_types,
#                 layer_type=spatial_layer,
#                 num_heads=num_attention_heads,
#                 edge_dim_dict=edge_dim_dict,
#                 dropout=dropout,
#                 aggr="sum",
#             )
#             self.gnn_layers.append(hetero_conv)

#             # LayerNorm only for node types that RECEIVE messages
#             norm_dict = nn.ModuleDict()
#             for nt in node_types:
#                 if nt in self.receiving_types:
#                     # Receiving types have dimension = out_per_layer (heads * hidden)
#                     if nt == "api":
#                         norm_dim = self.api_out_dim_per_layer
#                     else:
#                         norm_dim = self.resource_out_dim_per_layer
#                     norm_dict[nt] = nn.LayerNorm(norm_dim)
#             self.norm_layers.append(norm_dict)

#             # Update current_dims for next layer
#             for nt in node_types:
#                 if nt in self.receiving_types:
#                     if nt == "api":
#                         current_dims[nt] = self.api_out_dim_per_layer
#                     else:
#                         current_dims[nt] = self.resource_out_dim_per_layer
#                 # Non-receiving types keep their dimension

#         # ----------------------------
#         # 3. Temporal LSTM (on API sequences)
#         # ----------------------------
#         self.temporal_lstm = TemporalLSTM(
#             input_dim=self.api_out_dim_per_layer,
#             hidden_dim=lstm_hidden_dim,
#             num_layers=lstm_num_layers,
#             dropout=lstm_dropout,
#             bidirectional=bidirectional,
#         )

#         # ----------------------------
#         # 4. Temporal attention pooling (on API nodes → graph-level embedding)
#         # ----------------------------
#         self.api_pooling = TemporalAttentionPooling(
#             input_dim=self.lstm_output_dim,
#             num_heads=self.temporal_heads,
#         )

#         # ----------------------------
#         # 5. Resource node pooling (DUAL-PATH - FOOLPROOF)
#         # ----------------------------
#         # Create TWO pooling paths: one for 64-dim, one for 384-dim
#         self.resource_poolings_64 = nn.ModuleDict()   # For non-receiving types
#         self.resource_poolings_384 = nn.ModuleDict()  # For receiving types

#         resource_node_types = [nt for nt in node_types if nt != "api"]

#         for nt in resource_node_types:
#             # Path 1: 64-dim input (no GNN updates)
#             self.resource_poolings_64[nt] = nn.Sequential(
#                 nn.Linear(self.hidden_dim_resource, self.hidden_dim_resource),  # 64→64
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#             )
            
#             # Path 2: 384-dim input (received GNN updates)
#             self.resource_poolings_384[nt] = nn.Sequential(
#                 nn.Linear(self.resource_out_dim_per_layer, self.hidden_dim_resource),  # 384→64
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#             )

#         print(f"🔧 Dual-path pooling created for: {resource_node_types}")

#         # ----------------------------
#         # 6. Final classifier
#         # ----------------------------
#         # Combined embedding = API (lstm_output_dim) + resources (4 types × 64)
#         num_resource_types = len([nt for nt in node_types if nt != "api"])
#         self.total_embedding_dim = self.lstm_output_dim + (num_resource_types * self.hidden_dim_resource)
#         self.classifier_hidden_dim = 256

#         print(f"📊 Total embedding dim: {self.total_embedding_dim}")
#         print(f"📊 Classifier hidden: {self.classifier_hidden_dim}")
        
#         self.classifier = nn.Sequential(
#             nn.Linear(self.total_embedding_dim, self.classifier_hidden_dim),
#             nn.LayerNorm(self.classifier_hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.classifier_hidden_dim, self.classifier_hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.classifier_hidden_dim // 2, num_classes),
#         )

#     def forward(
#         self,
#         x_dict: Dict[str, torch.Tensor],
#         edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
#         edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
#         batch_dict: Optional[Dict[str, torch.Tensor]] = None,
#         timestamps_dict: Optional[Dict[str, torch.Tensor]] = None,
#     ) -> torch.Tensor:
#         """
#         Forward pass.

#         Args:
#             x_dict: {node_type: [N_type, feat_dim]}
#             edge_index_dict: {(src, rel, dst): [2, E_type]}
#             edge_attr_dict: {(src, rel, dst): [E_type, edge_dim]} or None
#             batch_dict: {node_type: [N_type]} batch assignment
#             timestamps_dict: {node_type: [N_type]} timestamps for LSTM sorting

#         Returns:
#             logits [batch_size, num_classes]
#         """
#         device = next(iter(x_dict.values())).device

#         # 1) Input projection
#         x_dict_projected = {}
#         for nt, x in x_dict.items():
#             x_dict_projected[nt] = self.input_projections[nt](x)

#         # 2) Multi-layer heterogeneous GNN
#         h_dict = x_dict_projected
#         for layer_idx, (gnn_layer, norm_dict) in enumerate(zip(self.gnn_layers, self.norm_layers)):
#             # Apply GNN
#             h_out = gnn_layer(
#                 x_dict=h_dict,
#                 edge_index_dict=edge_index_dict,
#                 edge_attr_dict=edge_attr_dict,
#             )

#             # Apply layer norm + residual (only for receiving types)
#             for nt in h_dict.keys():
#                 if nt in h_out:
#                     # This node type received updates
#                     if nt in norm_dict:
#                         h_out[nt] = norm_dict[nt](h_out[nt])
                    
#                     # Residual connection (if dimensions match)
#                     if h_dict[nt].size(-1) == h_out[nt].size(-1):
#                         h_out[nt] = h_out[nt] + h_dict[nt]
#                 else:
#                     # This node type didn't receive updates, keep previous embeddings
#                     h_out[nt] = h_dict[nt]

#             # Apply activation
#             for nt in h_out.keys():
#                 h_out[nt] = F.relu(h_out[nt])
#                 h_out[nt] = F.dropout(h_out[nt], p=self.dropout, training=self.training)

#             h_dict = h_out

#         # 3) Temporal LSTM on API nodes
#         api_h = h_dict["api"]
#         api_batch = batch_dict["api"]
#         api_timestamps = timestamps_dict.get("api", None) if timestamps_dict is not None else None

#         api_h_temporal = self.temporal_lstm(
#             x=api_h,
#             batch=api_batch,
#             timestamps=api_timestamps,
#         )

#         # 4) Temporal attention pooling on API → graph embedding
#         api_graph_embedding = self.api_pooling(
#             x=api_h_temporal,
#             batch=api_batch,
#         )
#         # shape: [batch_size, lstm_output_dim]

#         # 5) Resource pooling (FOOLPROOF VERSION with dual-path)
#         batch_size = int(api_batch.max().item()) + 1
#         resource_embeddings = []

#         for nt in self.node_types:
#             if nt == "api":
#                 continue

#             if nt not in h_dict or nt not in batch_dict:
#                 # Node type doesn't exist in this batch → zero embedding
#                 dummy_emb = torch.zeros(batch_size, self.hidden_dim_resource, device=device)
#                 resource_embeddings.append(dummy_emb)
#                 continue

#             h_resource = h_dict[nt]
#             batch_resource = batch_dict[nt]
            
#             # Get ACTUAL dimension from tensor (NOT from config)
#             actual_dim = h_resource.size(-1)

#             # Mean pooling per graph
#             pooled = []
#             for b in range(batch_size):
#                 mask = batch_resource == b
#                 if mask.sum() > 0:
#                     pooled_b = h_resource[mask].mean(dim=0)
#                 else:
#                     pooled_b = torch.zeros(actual_dim, device=device)
#                 pooled.append(pooled_b)

#             pooled = torch.stack(pooled)  # [batch_size, actual_dim]
            
#             # ✅ FOOLPROOF: Choose correct pooling path based on ACTUAL dimension
#             if actual_dim == self.hidden_dim_resource:
#                 # Path 1: No GNN updates, stays at 64-dim
#                 pooled = self.resource_poolings_64[nt](pooled)
#             elif actual_dim == self.resource_out_dim_per_layer:
#                 # Path 2: Received GNN updates, now 384-dim
#                 pooled = self.resource_poolings_384[nt](pooled)
#             else:
#                 # Unexpected dimension - create emergency projection
#                 print(f"⚠️ WARNING: Unexpected dimension {actual_dim} for {nt}, creating emergency projection")
#                 emergency_proj = nn.Linear(actual_dim, self.hidden_dim_resource).to(device)
#                 pooled = emergency_proj(pooled)
            
#             # Result: [batch_size, 64]
#             resource_embeddings.append(pooled)

#         # 6) Concatenate all embeddings
#         combined_embedding = torch.cat(
#             [api_graph_embedding] + resource_embeddings,
#             dim=1
#         )
#         # shape: [batch_size, total_embedding_dim]

#         # 7) Classification
#         logits = self.classifier(combined_embedding)
#         return logits


# def create_hetero_tgnn(spatial_layer: str = 'gat', device: Optional[str] = None, **kwargs):
#     """
#     Factory function to create HeteroTemporalGNN model.
    
#     Args:
#         spatial_layer: 'gat' or 'transformer'
#         device: Device to place model on
#         **kwargs: Additional arguments passed to HeteroTemporalGNN
        
#     Returns:
#         HeteroTemporalGNN model on specified device
#     """
#     model = HeteroTemporalGNN(
#         spatial_layer=spatial_layer,
#         **kwargs,
#     )
    
#     if device is not None:
#         model = model.to(device)
    
#     return model
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
        dropout             = dropout             if dropout     is not None else config.DROPOUT
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

        # ── which node types receive messages ────────────────────────────────
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
        #  1. Input projections (one per node type)
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
        #  2. Heterogeneous GNN layers
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
                    if (rel == "temporal" and src == "api" and dst == "api")
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
                    norm_dim = (self.api_out_dim_per_layer
                                if nt == "api"
                                else self.resource_out_dim_per_layer)
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
        #  3. Temporal LSTM  (on API node sequence)
        # ══════════════════════════════════════════════════════════════════════
        self.temporal_lstm = TemporalLSTM(
            input_dim    = self.api_out_dim_per_layer,  # 768
            hidden_dim   = lstm_hidden_dim,             # 128
            num_layers   = lstm_num_layers,
            dropout      = lstm_dropout,
            bidirectional= bidirectional,
        )

        # ══════════════════════════════════════════════════════════════════════
        #  4. Temporal attention pooling  → graph-level embedding
        # ══════════════════════════════════════════════════════════════════════
        self.api_pooling = TemporalAttentionPooling(
            input_dim = self.lstm_output_dim,   # 256
            num_heads = temporal_heads,         # 8
        )

        # ══════════════════════════════════════════════════════════════════════
        #  5. Resource pooling  (dual-path: 64-dim or 384-dim input)
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
        #  6. Classifier MLP
        # ══════════════════════════════════════════════════════════════════════
        num_resource_types       = len(resource_node_types)
        self.total_embedding_dim = (
            self.lstm_output_dim + num_resource_types * hidden_dim_resource
        )   # 256 + 4×64 = 512

        clf_hidden = 256
        self.classifier = nn.Sequential(
            nn.Linear(self.total_embedding_dim, clf_hidden),   # 512→256
            nn.LayerNorm(clf_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden, clf_hidden // 2),            # 256→128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden // 2, num_classes),           # 128→3
        )

        print(f"[Model] Total embedding dim : {self.total_embedding_dim}")
        print(f"[Model] Classifier          : "
              f"{self.total_embedding_dim}→{clf_hidden}→{clf_hidden//2}→{num_classes}")

    # ══════════════════════════════════════════════════════════════════════════
    #  forward
    # ══════════════════════════════════════════════════════════════════════════

    def forward(
        self,
        x_dict:          Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict:  Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        batch_dict:      Optional[Dict[str, torch.Tensor]]                  = None,
        timestamps_dict: Optional[Dict[str, torch.Tensor]]                  = None,
        api_indices:     Optional[torch.LongTensor]                          = None,
    ) -> torch.Tensor:
        """
        Args:
            x_dict          : {node_type: Tensor[N, feat_dim]}
            edge_index_dict : {(src,rel,dst): LongTensor[2, E]}
            edge_attr_dict  : {(src,rel,dst): Tensor[E, edge_dim]}  optional
            batch_dict      : {node_type: LongTensor[N]}  batch assignment
            timestamps_dict : {node_type: Tensor[N]}  timestamps
            api_indices     : LongTensor[N_api]  vocab indices for memory read/write

        Returns:
            logits : Tensor[batch_size, num_classes]
        """
        device = next(iter(x_dict.values())).device

        # ── 0. SemanticMemory enrichment (API nodes) ──────────────────────────
        # [N_api, 775] → [N_api, 903]
        if self.use_memory and self.memory is not None and api_indices is not None:
            api_ts  = (timestamps_dict.get("api", None)
                       if timestamps_dict is not None else None)
            x_dict  = dict(x_dict)          # shallow copy — don't mutate caller
            x_dict["api"] = self.memory(
                api_indices.to(device),
                x_dict["api"],
                api_ts,
            )

        # ── 1. Input projections ──────────────────────────────────────────────
        # api: 903→128,  resources: 6→64
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
                h_out[nt] = F.dropout(h_out[nt], p=self.dropout,
                                      training=self.training)
            h_dict = h_out

        # ── 3. Temporal LSTM on API nodes ─────────────────────────────────────
        api_batch = batch_dict["api"]
        api_ts    = (timestamps_dict.get("api", None)
                     if timestamps_dict is not None else None)

        api_temporal = self.temporal_lstm(
            x          = h_dict["api"],   # [N_api, 768]
            batch      = api_batch,
            timestamps = api_ts,
        )                                 # [N_api, 256]

        # ── 4. Attention pooling → graph-level API embedding ──────────────────
        api_graph_emb = self.api_pooling(
            x     = api_temporal,    # [N_api, 256]
            batch = api_batch,
        )                            # [B, 256]

        # ── 5. Resource pooling ───────────────────────────────────────────────
        B                   = int(api_batch.max().item()) + 1
        resource_embeddings = []

        for nt in self.node_types:
            if nt == "api":
                continue

            if nt not in h_dict or nt not in batch_dict:
                resource_embeddings.append(
                    torch.zeros(B, self.hidden_dim_resource, device=device)
                )
                continue

            h_res      = h_dict[nt]               # [N_nt, actual_dim]
            batch_res  = batch_dict[nt]
            actual_dim = h_res.size(-1)

            # mean-pool per graph in batch
            pooled = []
            for b in range(B):
                mask = (batch_res == b)
                pooled.append(
                    h_res[mask].mean(dim=0) if mask.sum() > 0
                    else torch.zeros(actual_dim, device=device)
                )
            pooled = torch.stack(pooled)          # [B, actual_dim]

            # route to correct projection based on actual tensor dim
            if actual_dim == self.hidden_dim_resource:           # 64
                pooled = self.resource_poolings_64[nt](pooled)
            elif actual_dim == self.resource_out_dim_per_layer:  # 384
                pooled = self.resource_poolings_384[nt](pooled)
            else:
                # Safety net — project from unexpected dim (avoids crash)
                emergency = nn.Linear(
                    actual_dim, self.hidden_dim_resource
                ).to(device)
                pooled = emergency(pooled)

            resource_embeddings.append(pooled)    # [B, 64]

        # ── 6. Classify ───────────────────────────────────────────────────────
        combined = torch.cat(
            [api_graph_emb] + resource_embeddings, dim=1
        )                                          # [B, 512]
        return self.classifier(combined)           # [B, 3]

    # ══════════════════════════════════════════════════════════════════════════
    #  Memory helpers  (called from training loop)
    # ══════════════════════════════════════════════════════════════════════════

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
            loss.backward()
            optimizer.step()
            model.update_memory(api_indices, raw_api_features)   ← here
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
                self.memory.memory_bank.weight, lambda_reg
            )
        return torch.tensor(0.0, requires_grad=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

def create_hetero_tgnn(
    spatial_layer: str  = "gat",
    device:        Optional[str] = None,
    use_memory:    bool = True,
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

    # ── build model ──────────────────────────────────────────────
    model = create_hetero_tgnn(spatial_layer="gat", use_memory=True)
    model.eval()

    device = torch.device("cpu")

    # ── dummy inputs ─────────────────────────────────────────────
    x_dict = {
        "api":      torch.randn(N_api, 775),
        "file":     torch.randn(N_res, 6),
        "registry": torch.randn(N_res, 6),
        "process":  torch.randn(N_res, 6),
        "network":  torch.randn(N_res, 6),
    }

    edge_index_dict = {
        ("api", "temporal", "api"): torch.randint(0, N_api, (2, 200)),
        ("api", "accesses",  "file"):    torch.stack([
            torch.randint(0, N_api, (50,)),
            torch.randint(0, N_res, (50,)),
        ]),
    }

    edge_attr_dict = {
        ("api", "temporal", "api"): torch.randn(200, 14),
    }

    batch_dict = {
        "api":      torch.repeat_interleave(torch.arange(B), N_api // B),
        "file":     torch.repeat_interleave(torch.arange(B), N_res // B),
        "registry": torch.repeat_interleave(torch.arange(B), N_res // B),
        "process":  torch.repeat_interleave(torch.arange(B), N_res // B),
        "network":  torch.repeat_interleave(torch.arange(B), N_res // B),
    }

    timestamps_dict = {
        "api": torch.linspace(0, 100, N_api),
    }

    api_indices = torch.randint(2, 325, (N_api,))

    # ── forward pass ─────────────────────────────────────────────
    with torch.no_grad():
        logits = model(
            x_dict          = x_dict,
            edge_index_dict = edge_index_dict,
            edge_attr_dict  = edge_attr_dict,
            batch_dict      = batch_dict,
            timestamps_dict = timestamps_dict,
            api_indices     = api_indices,
        )

    print(f"\n[Smoke] Input  API : {list(x_dict['api'].shape)}")
    print(f"[Smoke] Output     : {list(logits.shape)}  (expected [{B}, 3])")
    assert logits.shape == (B, 3), f"Shape mismatch: {logits.shape}"

    # ── memory write ─────────────────────────────────────────────
    model.update_memory(api_indices, x_dict["api"])
    print(f"[Smoke] Memory write  ✅")

    # ── memory loss ───────────────────────────────────────────────
    mem_loss = model.memory_loss()
    print(f"[Smoke] Memory loss   {mem_loss.item():.8f}  ✅")

    # ── param count ──────────────────────────────────────────────
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Smoke] Parameters    total={total:,}  trainable={trainable:,}")
    print("\n✅  tgnn_model.py working correctly!")