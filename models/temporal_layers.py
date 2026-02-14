"""
Temporal layers for (heterogeneous) Graph Neural Networks.

Components (homogeneous, existing):
1. TemporalGATConv: GAT with temporal edge features (edge_attr) + BIPARTITE SUPPORT
2. TemporalTransformerConv: Transformer-style spatial aggregation + BIPARTITE SUPPORT
3. TemporalLSTM: Bi-directional LSTM for sequential modeling
4. TemporalAttentionPooling: Multi-head temporal pooling

New heterogeneous wrappers:
5. HeteroTemporalConv: Edge-type-specific temporal GNN via HeteroConv

Author: AbhisheK Menon
Date: 2026-02-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    TransformerConv,
    MessagePassing,
    HeteroConv
)
from torch_geometric.utils import softmax
from typing import Optional, Tuple, Dict, List, Union
import math


# ============================================================
# 1. TemporalGATConv with BIPARTITE SUPPORT
# ============================================================

class TemporalGATConv(MessagePassing):
    """
    Graph Attention Network with optional temporal edge features.
    
    SUPPORTS BIPARTITE GRAPHS:
    - Homogeneous: x is tensor, src=dst (e.g., API->API)
    - Bipartite: x is tuple (x_src, x_dst) (e.g., API->File)
    
    Args:
        in_channels: int or tuple (in_src, in_dst)
        out_channels: int (output per head)
        edge_dim: edge feature dimension
        heads: number of attention heads
        dropout: dropout rate
        concat: concatenate heads (True) or average (False)
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: int = 0,
        heads: int = 6,
        dropout: float = 0.3,
        concat: bool = True,
        **kwargs
    ):
        super().__init__(aggr="add", node_dim=0, **kwargs)

        # Handle bipartite: (in_src, in_dst) or homogeneous: in_channels
        if isinstance(in_channels, int):
            self.in_channels_src = in_channels
            self.in_channels_dst = in_channels
        else:
            self.in_channels_src, self.in_channels_dst = in_channels

        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.dropout = dropout
        self.concat = concat

        # Linear transformation for source and destination
        self.lin_src = nn.Linear(self.in_channels_src, heads * out_channels)
        self.lin_dst = nn.Linear(self.in_channels_dst, heads * out_channels)

        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # Edge feature handling
        if edge_dim > 0:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
            self.att_edge = nn.Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.att_edge = None

        # Bias
        if concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
            nn.init.xavier_uniform_(self.att_edge)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features
                - Homogeneous: [N, in_channels]
                - Bipartite: ([N_src, in_src], [N_dst, in_dst])
            edge_index: [2, E] where edges go from src to dst
            edge_attr: [E, edge_dim] or None
        
        Returns:
            [N_dst, heads * out_channels] if concat else [N_dst, out_channels]
        """
        # Handle bipartite vs homogeneous
        if isinstance(x, torch.Tensor):
            x_src = x_dst = x
        else:
            x_src, x_dst = x

        # Transform source and destination
        x_src = self.lin_src(x_src).view(-1, self.heads, self.out_channels)
        x_dst = self.lin_dst(x_dst).view(-1, self.heads, self.out_channels)

        # Process edge attributes
        if self.edge_dim > 0 and edge_attr is not None and self.lin_edge is not None:
            edge_attr_emb = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        else:
            E = edge_index.size(1)
            edge_attr_emb = x_src.new_zeros(E, self.heads, self.out_channels)

        # Propagate messages
        out = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            edge_attr=edge_attr_emb,
        )

        # Concatenate or average heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        return out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        # x_i: destination features [E, heads, out_channels]
        # x_j: source features [E, heads, out_channels]
        # edge_attr: [E, heads, out_channels]

        alpha_src = (x_j * self.att_src).sum(dim=-1)  # [E, heads]
        alpha_dst = (x_i * self.att_dst).sum(dim=-1)  # [E, heads]

        if self.att_edge is not None:
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha_src + alpha_dst + alpha_edge
        else:
            alpha = alpha_src + alpha_dst

        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)


# ============================================================
# 2. TemporalTransformerConv with BIPARTITE SUPPORT
# ============================================================

class TemporalTransformerConv(MessagePassing):
    """
    Transformer-style graph convolution with bipartite support.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: int = 0,
        heads: int = 6,
        dropout: float = 0.3,
        concat: bool = True,
        **kwargs
    ):
        super().__init__(aggr="add", node_dim=0, **kwargs)

        self.conv = TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            edge_dim=edge_dim if edge_dim > 0 else None,
            dropout=dropout,
        )

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.conv(x, edge_index, edge_attr=edge_attr)


# ============================================================
# 3. TemporalLSTM (unchanged)
# ============================================================

class TemporalLSTM(nn.Module):
    """
    Bi-directional LSTM for sequential node modeling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = x.device
        batch_size = int(batch.max().item()) + 1

        outputs = []

        for b in range(batch_size):
            mask = batch == b
            x_b = x[mask]

            if timestamps is not None:
                t_b = timestamps[mask]
                order = torch.argsort(t_b)
                x_b = x_b[order]

            x_b = x_b.unsqueeze(0)
            out_b, _ = self.lstm(x_b)
            out_b = out_b.squeeze(0)

            if timestamps is not None:
                inv_order = torch.argsort(order)
                out_b = out_b[inv_order]

            outputs.append(out_b)

        return torch.cat(outputs, dim=0)


# ============================================================
# 4. TemporalAttentionPooling (unchanged)
# ============================================================

class TemporalAttentionPooling(nn.Module):
    """
    Multi-head temporal attention pooling.
    """

    def __init__(self, input_dim: int, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> List[torch.Tensor]:
        batch_size = int(batch.max().item()) + 1
        graph_embeddings = []

        for b in range(batch_size):
            mask = batch == b
            x_b = x[mask].unsqueeze(0)  # [1, seq_len, input_dim]

            q = self.query(x_b.mean(dim=1, keepdim=True))  # [1, 1, input_dim]
            k = self.key(x_b)
            v = self.value(x_b)

            # Multi-head attention
            q = q.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_scores, dim=-1)

            out_b = torch.matmul(attn_weights, v)
            out_b = out_b.transpose(1, 2).reshape(1, -1)
            emb_b = out_b.squeeze(0).squeeze(0)
            emb_b = self.norm(emb_b)
            graph_embeddings.append(emb_b)

        return torch.stack(graph_embeddings)


# ============================================================
# 5. HeteroTemporalConv (FIXED)
# ============================================================

class HeteroTemporalConv(nn.Module):
    """
    Heterogeneous temporal graph convolution.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        out_channels_dict: Dict[str, int],
        edge_types: List[Tuple[str, str, str]],
        layer_type: str = 'gat',
        num_heads: int = 6,
        edge_dim_dict: Optional[Dict[Tuple[str, str, str], int]] = None,
        dropout: float = 0.3,
        aggr: str = 'sum',
    ):
        super().__init__()

        self.in_channels_dict = in_channels_dict
        self.out_channels_dict = out_channels_dict
        self.edge_types = edge_types
        self.layer_type = layer_type.lower()
        self.num_heads = num_heads
        self.edge_dim_dict = edge_dim_dict if edge_dim_dict is not None else {}
        self.dropout = dropout
        self.aggr = aggr

        # Create per-edge-type convolutions
        convs = {}
        for edge_type in edge_types:
            src, rel, dst = edge_type

            # Determine if bipartite
            if src == dst:
                # Homogeneous
                in_ch = in_channels_dict[src]
            else:
                # Bipartite
                in_ch = (in_channels_dict[src], in_channels_dict[dst])

            out_ch = out_channels_dict[dst]
            edge_dim = self.edge_dim_dict.get(edge_type, 0)

            # Create layer
            if self.layer_type == 'gat':
                conv = TemporalGATConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    edge_dim=edge_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            elif self.layer_type == 'transformer':
                conv = TemporalTransformerConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    edge_dim=edge_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            else:
                raise ValueError(f"Unknown layer_type: {layer_type}")

            # Store with string key for ModuleDict
            key = f"{src}__{rel}__{dst}"
            convs[key] = conv

        self.convs = nn.ModuleDict(convs)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (FIXED to handle missing node types).
        """
        out_dict = {}

        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict:
                continue

            src, rel, dst = edge_type

            # FIXED: Check if src and dst exist in x_dict before accessing
            if src not in x_dict or dst not in x_dict:
                continue

            edge_index = edge_index_dict[edge_type]

            # Get edge attributes
            if edge_attr_dict is not None and edge_type in edge_attr_dict:
                edge_attr = edge_attr_dict[edge_type]
            else:
                edge_dim = self.edge_dim_dict.get(edge_type, 0)
                if edge_dim > 0:
                    E = edge_index.size(1)
                    device = x_dict[src].device
                    edge_attr = torch.zeros(E, edge_dim, device=device)
                else:
                    edge_attr = None

            # Get convolution
            key = f"{src}__{rel}__{dst}"
            conv = self.convs[key]

            # Prepare input (bipartite or homogeneous)
            if src == dst:
                x_input = x_dict[src]
            else:
                x_input = (x_dict[src], x_dict[dst])

            # Apply convolution
            out = conv(x_input, edge_index, edge_attr)

            # Aggregate to destination
            if dst not in out_dict:
                out_dict[dst] = out
            else:
                if self.aggr == 'sum':
                    out_dict[dst] = out_dict[dst] + out
                elif self.aggr == 'mean':
                    out_dict[dst] = (out_dict[dst] + out) / 2
                elif self.aggr == 'max':
                    out_dict[dst] = torch.max(out_dict[dst], out)

        return out_dict


# ============================================================
# Test
# ============================================================

if __name__ == '__main__':
    # Test heterogeneous temporal convolution
    print("Testing HeteroTemporalConv...")

    in_dims = {'api': 32, 'file': 16, 'process': 24, 'registry': 20}
    out_dims = {'api': 64, 'file': 64, 'process': 64, 'registry': 64}
    edge_types = [
        ('api', 'temporal', 'api'),
        ('api', 'operates_on', 'file'),
    ]
    edge_dim_dict = {
        ('api', 'temporal', 'api'): 14,
        ('api', 'operates_on', 'file'): 0,
    }

    num_apis = 20
    num_files = 15

    # Create dummy data
    x_dict = {
        'api': torch.randn(num_apis, in_dims['api']),
        'file': torch.randn(num_files, in_dims['file']),
    }

    edge_index_dict = {
        ('api', 'temporal', 'api'): torch.randint(0, num_apis, (2, 40)),
        ('api', 'operates_on', 'file'): torch.stack([
            torch.randint(0, num_apis, (15,)),
            torch.randint(0, num_files, (15,)),
        ]),
    }

    edge_attr_dict = {
        ('api', 'temporal', 'api'): torch.randn(40, 14),
    }

    # Create layer
    layer = HeteroTemporalConv(
        in_channels_dict=in_dims,
        out_channels_dict=out_dims,
        edge_types=edge_types,
        layer_type='gat',
        num_heads=6,
        edge_dim_dict=edge_dim_dict,
        dropout=0.3,
        aggr='sum',
    )

    # Forward
    out = layer(x_dict, edge_index_dict, edge_attr_dict)

    print("Output node types:", list(out.keys()))
    for k, v in out.items():
        print(f"  {k}: {v.shape}")

    # Verify shapes
    assert out['api'].shape == (num_apis, 6 * 128), f"API shape wrong: {out['api'].shape}"
    assert out['file'].shape == (num_files, 6 * 128), f"File shape wrong: {out['file'].shape}"

    print("✅ All tests passed!")
