"""
Heterogeneous Temporal Graph Neural Network (Hetero-TGNN) for Ransomware Detection.

Architecture:
1. Heterogeneous node inputs (HeteroData-style dictionaries):
   - 'api' nodes: 775-dim (768 LLM + 7 metadata)
   - 'file', 'registry', 'process', 'network' nodes: 6-dim each
   - Temporal edges 'api'→'api': 14-dim edge features

2. HeteroTemporalConv layers (from temporal_layers.py):
   - Edge-type-specific TemporalGATConv / TemporalTransformerConv
   - Full use of source/destination node embeddings and temporal edge features
   - Proper bipartite handling for cross-type edges (api→file, process→api, etc.)

3. Temporal LSTM on API node sequences:
   - Input: enriched API embeddings from last HeteroTemporalConv layer
   - Sorted by per-node timestamps inside each graph

4. Temporal attention pooling on API nodes:
   - Multi-head attention over time to produce one embedding per graph

5. Optional resource node pooling:
   - Mean pooling per node type (file/registry/process/network)
   - Projected and concatenated with API graph embedding

6. Classification head:
   - MLP mapping combined graph embedding to 3 classes:
     [Ransomware, Benign, Other Malware]

Author: Abhishek Menon
Date: 2026-02-13 (FOOLPROOF FIXED)
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


class HeteroTemporalGNN(nn.Module):
    """
    Fully heterogeneous temporal GNN for ransomware detection.

    Expects HeteroData-style inputs broken out by node_type and edge_type.
    """

    def __init__(
        self,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[Tuple[str, str, str]]] = None,
        node_dims: Optional[Dict[str, int]] = None,
        hidden_dim_api: int = 128,
        hidden_dim_resource: int = 64,
        num_gnn_layers: int = None,
        lstm_hidden_dim: int = None,
        lstm_num_layers: int = None,
        num_classes: int = None,
        spatial_layer: Optional[str] = None,  # 'gat' or 'transformer'
        num_attention_heads: int = None,
        temporal_heads: int = None,
        dropout: float = None,
        lstm_dropout: float = None,
        bidirectional: Optional[bool] = None,
    ):
        super().__init__()

        # ----------------------------
        # Config defaults
        # ----------------------------
        node_types = node_types or config.NODE_TYPES
        edge_types = edge_types or config.EDGE_TYPES

        if node_dims is None:
            node_dims = {
                "api": config.API_NODE_DIM,
                "file": config.RESOURCE_NODE_DIM,
                "registry": config.RESOURCE_NODE_DIM,
                "process": config.RESOURCE_NODE_DIM,
                "network": config.RESOURCE_NODE_DIM,
            }

        num_gnn_layers = num_gnn_layers or config.NUM_GNN_LAYERS
        lstm_hidden_dim = lstm_hidden_dim or config.LSTM_HIDDEN_DIM
        lstm_num_layers = lstm_num_layers or config.LSTM_NUM_LAYERS
        num_classes = num_classes or config.NUM_CLASSES
        spatial_layer = spatial_layer or config.SPATIAL_LAYER
        num_attention_heads = num_attention_heads or config.NUM_ATTENTION_HEADS
        temporal_heads = temporal_heads or config.TEMPORAL_HEADS
        dropout = dropout or config.DROPOUT
        lstm_dropout = lstm_dropout or config.LSTM_DROPOUT
        bidirectional = bidirectional if bidirectional is not None else config.BIDIRECTIONAL

        self.node_types = node_types
        self.edge_types = edge_types
        self.node_dims = node_dims
        self.hidden_dim_api = hidden_dim_api
        self.hidden_dim_resource = hidden_dim_resource
        self.num_gnn_layers = num_gnn_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.num_classes = num_classes
        self.spatial_layer = spatial_layer
        self.num_attention_heads = num_attention_heads
        self.temporal_heads = temporal_heads
        self.dropout = dropout
        self.lstm_dropout = lstm_dropout
        self.bidirectional = bidirectional

        # API and resource output dims per layer (after heads concatenation)
        self.api_out_dim_per_layer = hidden_dim_api * num_attention_heads  
        self.resource_out_dim_per_layer = hidden_dim_resource * num_attention_heads  
        print(f"🔍 API out dim: {self.api_out_dim_per_layer}")      # Should be 768
        print(f"🔍 Resource out dim: {self.resource_out_dim_per_layer}")  # Should be 384
        self.lstm_output_dim = (
            lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim
        )

        # Dimension check for attention pooling
        if self.lstm_output_dim % self.temporal_heads != 0:
            raise ValueError(
                f"LSTM output dim ({self.lstm_output_dim}) must be divisible "
                f"by temporal_heads ({self.temporal_heads})."
            )

        # Identify which node types receive messages (will be updated by GNN)
        self.receiving_types = set()
        for src, rel, dst in edge_types:
            self.receiving_types.add(dst)

        print(f"🔍 Receiving types (will have {self.resource_out_dim_per_layer}-dim): {self.receiving_types}")
        print(f"🔍 Non-receiving types (stay at {self.hidden_dim_resource}-dim): {set(node_types) - self.receiving_types - {'api'}}")

        # ----------------------------
        # 1. Input linear projections per node type
        # ----------------------------
        self.input_projections = nn.ModuleDict()
        for nt in node_types:
            in_dim = node_dims[nt]
            if nt == "api":
                out_dim = hidden_dim_api
            else:
                out_dim = hidden_dim_resource

            self.input_projections[nt] = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # ----------------------------
        # 2. Heterogeneous temporal GNN layers
        # ----------------------------
        self.gnn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # Track ACTUAL dimensions per node type across layers
        current_dims = {
            "api": hidden_dim_api,
            "file": hidden_dim_resource,
            "registry": hidden_dim_resource,
            "process": hidden_dim_resource,
            "network": hidden_dim_resource,
        }

        for layer_idx in range(num_gnn_layers):
            # Out channels: what each destination node type will get (before concat)
            out_channels_dict = {
                "api": hidden_dim_api,
                "file": hidden_dim_resource,
                "registry": hidden_dim_resource,
                "process": hidden_dim_resource,
                "network": hidden_dim_resource,
            }

            # Edge dims: only temporal API→API edges carry 14-dim features
            edge_dim_dict = {}
            for et in edge_types:
                src, rel, dst = et
                if rel == "temporal" and src == "api" and dst == "api":
                    edge_dim_dict[et] = config.TEMPORAL_EDGE_DIM
                else:
                    edge_dim_dict[et] = 0

            hetero_conv = HeteroTemporalConv(
                in_channels_dict=current_dims,  # Use actual current dimensions
                out_channels_dict=out_channels_dict,
                edge_types=edge_types,
                layer_type=spatial_layer,
                num_heads=num_attention_heads,
                edge_dim_dict=edge_dim_dict,
                dropout=dropout,
                aggr="sum",
            )
            self.gnn_layers.append(hetero_conv)

            # LayerNorm only for node types that RECEIVE messages
            norm_dict = nn.ModuleDict()
            for nt in node_types:
                if nt in self.receiving_types:
                    # Receiving types have dimension = out_per_layer (heads * hidden)
                    if nt == "api":
                        norm_dim = self.api_out_dim_per_layer
                    else:
                        norm_dim = self.resource_out_dim_per_layer
                    norm_dict[nt] = nn.LayerNorm(norm_dim)
            self.norm_layers.append(norm_dict)

            # Update current_dims for next layer
            for nt in node_types:
                if nt in self.receiving_types:
                    if nt == "api":
                        current_dims[nt] = self.api_out_dim_per_layer
                    else:
                        current_dims[nt] = self.resource_out_dim_per_layer
                # Non-receiving types keep their dimension

        # ----------------------------
        # 3. Temporal LSTM (on API sequences)
        # ----------------------------
        self.temporal_lstm = TemporalLSTM(
            input_dim=self.api_out_dim_per_layer,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        # ----------------------------
        # 4. Temporal attention pooling (on API nodes → graph-level embedding)
        # ----------------------------
        self.api_pooling = TemporalAttentionPooling(
            input_dim=self.lstm_output_dim,
            num_heads=self.temporal_heads,
        )

        # ----------------------------
        # 5. Resource node pooling (DUAL-PATH - FOOLPROOF)
        # ----------------------------
        # Create TWO pooling paths: one for 64-dim, one for 384-dim
        self.resource_poolings_64 = nn.ModuleDict()   # For non-receiving types
        self.resource_poolings_384 = nn.ModuleDict()  # For receiving types

        resource_node_types = [nt for nt in node_types if nt != "api"]

        for nt in resource_node_types:
            # Path 1: 64-dim input (no GNN updates)
            self.resource_poolings_64[nt] = nn.Sequential(
                nn.Linear(self.hidden_dim_resource, self.hidden_dim_resource),  # 64→64
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            
            # Path 2: 384-dim input (received GNN updates)
            self.resource_poolings_384[nt] = nn.Sequential(
                nn.Linear(self.resource_out_dim_per_layer, self.hidden_dim_resource),  # 384→64
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        print(f"🔧 Dual-path pooling created for: {resource_node_types}")

        # ----------------------------
        # 6. Final classifier
        # ----------------------------
        # Combined embedding = API (lstm_output_dim) + resources (4 types × 64)
        num_resource_types = len([nt for nt in node_types if nt != "api"])
        self.total_embedding_dim = self.lstm_output_dim + (num_resource_types * self.hidden_dim_resource)
        self.classifier_hidden_dim = 256

        print(f"📊 Total embedding dim: {self.total_embedding_dim}")
        print(f"📊 Classifier hidden: {self.classifier_hidden_dim}")
        
        self.classifier = nn.Sequential(
            nn.Linear(self.total_embedding_dim, self.classifier_hidden_dim),
            nn.LayerNorm(self.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.classifier_hidden_dim, self.classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.classifier_hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        batch_dict: Optional[Dict[str, torch.Tensor]] = None,
        timestamps_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_dict: {node_type: [N_type, feat_dim]}
            edge_index_dict: {(src, rel, dst): [2, E_type]}
            edge_attr_dict: {(src, rel, dst): [E_type, edge_dim]} or None
            batch_dict: {node_type: [N_type]} batch assignment
            timestamps_dict: {node_type: [N_type]} timestamps for LSTM sorting

        Returns:
            logits [batch_size, num_classes]
        """
        device = next(iter(x_dict.values())).device

        # 1) Input projection
        x_dict_projected = {}
        for nt, x in x_dict.items():
            x_dict_projected[nt] = self.input_projections[nt](x)

        # 2) Multi-layer heterogeneous GNN
        h_dict = x_dict_projected
        for layer_idx, (gnn_layer, norm_dict) in enumerate(zip(self.gnn_layers, self.norm_layers)):
            # Apply GNN
            h_out = gnn_layer(
                x_dict=h_dict,
                edge_index_dict=edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )

            # Apply layer norm + residual (only for receiving types)
            for nt in h_dict.keys():
                if nt in h_out:
                    # This node type received updates
                    if nt in norm_dict:
                        h_out[nt] = norm_dict[nt](h_out[nt])
                    
                    # Residual connection (if dimensions match)
                    if h_dict[nt].size(-1) == h_out[nt].size(-1):
                        h_out[nt] = h_out[nt] + h_dict[nt]
                else:
                    # This node type didn't receive updates, keep previous embeddings
                    h_out[nt] = h_dict[nt]

            # Apply activation
            for nt in h_out.keys():
                h_out[nt] = F.relu(h_out[nt])
                h_out[nt] = F.dropout(h_out[nt], p=self.dropout, training=self.training)

            h_dict = h_out

        # 3) Temporal LSTM on API nodes
        api_h = h_dict["api"]
        api_batch = batch_dict["api"]
        api_timestamps = timestamps_dict.get("api", None) if timestamps_dict is not None else None

        api_h_temporal = self.temporal_lstm(
            x=api_h,
            batch=api_batch,
            timestamps=api_timestamps,
        )

        # 4) Temporal attention pooling on API → graph embedding
        api_graph_embedding = self.api_pooling(
            x=api_h_temporal,
            batch=api_batch,
        )
        # shape: [batch_size, lstm_output_dim]

        # 5) Resource pooling (FOOLPROOF VERSION with dual-path)
        batch_size = int(api_batch.max().item()) + 1
        resource_embeddings = []

        for nt in self.node_types:
            if nt == "api":
                continue

            if nt not in h_dict or nt not in batch_dict:
                # Node type doesn't exist in this batch → zero embedding
                dummy_emb = torch.zeros(batch_size, self.hidden_dim_resource, device=device)
                resource_embeddings.append(dummy_emb)
                continue

            h_resource = h_dict[nt]
            batch_resource = batch_dict[nt]
            
            # Get ACTUAL dimension from tensor (NOT from config)
            actual_dim = h_resource.size(-1)

            # Mean pooling per graph
            pooled = []
            for b in range(batch_size):
                mask = batch_resource == b
                if mask.sum() > 0:
                    pooled_b = h_resource[mask].mean(dim=0)
                else:
                    pooled_b = torch.zeros(actual_dim, device=device)
                pooled.append(pooled_b)

            pooled = torch.stack(pooled)  # [batch_size, actual_dim]
            
            # ✅ FOOLPROOF: Choose correct pooling path based on ACTUAL dimension
            if actual_dim == self.hidden_dim_resource:
                # Path 1: No GNN updates, stays at 64-dim
                pooled = self.resource_poolings_64[nt](pooled)
            elif actual_dim == self.resource_out_dim_per_layer:
                # Path 2: Received GNN updates, now 384-dim
                pooled = self.resource_poolings_384[nt](pooled)
            else:
                # Unexpected dimension - create emergency projection
                print(f"⚠️ WARNING: Unexpected dimension {actual_dim} for {nt}, creating emergency projection")
                emergency_proj = nn.Linear(actual_dim, self.hidden_dim_resource).to(device)
                pooled = emergency_proj(pooled)
            
            # Result: [batch_size, 64]
            resource_embeddings.append(pooled)

        # 6) Concatenate all embeddings
        combined_embedding = torch.cat(
            [api_graph_embedding] + resource_embeddings,
            dim=1
        )
        # shape: [batch_size, total_embedding_dim]

        # 7) Classification
        logits = self.classifier(combined_embedding)
        return logits


def create_hetero_tgnn(spatial_layer: str = 'gat', device: Optional[str] = None, **kwargs):
    """
    Factory function to create HeteroTemporalGNN model.
    
    Args:
        spatial_layer: 'gat' or 'transformer'
        device: Device to place model on
        **kwargs: Additional arguments passed to HeteroTemporalGNN
        
    Returns:
        HeteroTemporalGNN model on specified device
    """
    model = HeteroTemporalGNN(
        spatial_layer=spatial_layer,
        **kwargs,
    )
    
    if device is not None:
        model = model.to(device)
    
    return model
