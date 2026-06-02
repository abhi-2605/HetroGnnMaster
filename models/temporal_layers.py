"""
Temporal layers for heterogeneous temporal GNNs.

Components:
1. TemporalGATConv: GAT-style relation conv with optional edge features
2. TemporalTransformerConv: Transformer-style relation conv with optional edge features
3a. EdgeAwareSAGEConv: GraphSAGE-style mean aggregation with edge features
3b. HybridGATSAGEConv: GAT + EdgeAwareSAGE with disagreement-aware gating
4. TemporalLSTM: explicit ordered sequence model for API temporal spine
5. TemporalAttentionPooling: graph-level temporal pooling
6. HeteroTemporalConv: schema-friendly heterogeneous relation message passing

Important design note:
- TemporalLSTM remains the explicit order-sensitive temporal branch.
- HeteroTemporalConv processes graph relations, including api->temporal->api,
  but does NOT replace sequence modeling.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.utils import softmax
import config

EdgeType = Tuple[str, str, str]


def edge_type_to_key(edge_type: EdgeType) -> str:
    """Stable string key for ModuleDict storage."""
    src, rel, dst = edge_type
    return f"{src}__{rel}__{dst}"


# ============================================================
# 1. TemporalGATConv with bipartite support
# ============================================================

class TemporalGATConv(MessagePassing):
    """
    GAT-style message passing with optional edge features.

    Supports:
    - homogeneous relations: x is Tensor
    - bipartite relations: x is (x_src, x_dst)
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: int = 0,
        heads: int = 6,
        dropout: float = 0.3,
        concat: bool = True,
        **kwargs,
    ):
        super().__init__(aggr="add", node_dim=0, **kwargs)

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

        self.lin_src = nn.Linear(self.in_channels_src, heads * out_channels)
        self.lin_dst = nn.Linear(self.in_channels_dst, heads * out_channels)

        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim > 0:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
            self.att_edge = nn.Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.att_edge = None

        if concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        if self.lin_src.bias is not None:
            nn.init.zeros_(self.lin_src.bias)
        if self.lin_dst.bias is not None:
            nn.init.zeros_(self.lin_dst.bias)

        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
            if self.lin_edge.bias is not None:
                nn.init.zeros_(self.lin_edge.bias)
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
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")

        if isinstance(x, torch.Tensor):
            x_src = x_dst = x
        else:
            x_src, x_dst = x

        x_src = self.lin_src(x_src).view(-1, self.heads, self.out_channels)
        x_dst = self.lin_dst(x_dst).view(-1, self.heads, self.out_channels)

        if self.edge_dim > 0 and edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_attr_emb = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        else:
            E = edge_index.size(1)
            edge_attr_emb = x_src.new_zeros(E, self.heads, self.out_channels)

        out = self.propagate(
            edge_index=edge_index,
            x=(x_src, x_dst),
            edge_attr=edge_attr_emb,
        )

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
        alpha_src = (x_j * self.att_src).sum(dim=-1)
        alpha_dst = (x_i * self.att_dst).sum(dim=-1)

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
# 2. TemporalTransformerConv with bipartite support
# ============================================================

class TemporalTransformerConv(MessagePassing):
    """
    Transformer-style relation convolution with optional edge features.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: int = 0,
        heads: int = 6,
        dropout: float = 0.3,
        concat: bool = True,
        **kwargs,
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
# 3a. EdgeAwareSAGEConv — inductive, edge-feature-preserving
# ============================================================

class EdgeAwareSAGEConv(MessagePassing):
    """
    GraphSAGE-style mean aggregation with edge feature support.

    Edge features are fused additively into neighbour messages before mean
    aggregation:
        m_uv = h_u + W_e e_uv
        h_v = W_self h_v + W_neigh MEAN(m_uv)

    This preserves edge signal without introducing per-pair attention weights.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: int = 0,
        normalize: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(aggr="mean", node_dim=0, **kwargs)

        if isinstance(in_channels, int):
            self.in_src = in_channels
            self.in_dst = in_channels
        else:
            self.in_src, self.in_dst = in_channels

        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.normalize = normalize
        self.dropout_p = dropout

        self.lin_self = nn.Linear(self.in_dst, out_channels, bias=False)
        self.lin_neigh = nn.Linear(self.in_src, out_channels, bias=False)

        if edge_dim > 0:
            self.lin_edge = nn.Linear(edge_dim, self.in_src, bias=False)
        else:
            self.lin_edge = None

        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")

        if isinstance(x, torch.Tensor):
            x_src = x_dst = x
        else:
            x_src, x_dst = x

        agg = self.propagate(
            edge_index=edge_index,
            x=x_src,
            edge_attr=edge_attr,
            size=(x_src.size(0), x_dst.size(0)),
        )

        out = self.lin_self(x_dst) + self.lin_neigh(agg) + self.bias

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        out = F.dropout(out, p=self.dropout_p, training=self.training)
        return out

    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.lin_edge is not None and edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            return x_j + self.lin_edge(edge_attr)
        return x_j


# ============================================================
# 3b. HybridGATSAGEConv — attention + inductive fallback
# ============================================================

class HybridGATSAGEConv(nn.Module):
    """
    Parallel TemporalGATConv + EdgeAwareSAGEConv with disagreement-aware gating.

    Both branches receive the same relation input and edge_attr. The final output
    has the same width as TemporalGATConv with concat=True:
        output_dim = heads * out_channels

    Fusion:
        d = |h_gat - h_sage|
        g = sigmoid(MLP([h_gat || h_sage || scale * d]))
        h = g * h_gat + (1 - g) * h_sage

    The gate is initialized slightly toward SAGE:
        gate ~= sigmoid(-0.85) ~= 0.30

    So early training starts roughly as:
        30% GAT + 70% SAGE

    Monitoring:
        last_gate and last_disagreement are detached tensors from the latest
        forward pass, intended for train/validation logging.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_dim: int = 0,
        heads: int = 6,
        dropout: float = 0.3,
        gate_hidden_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.heads = heads
        self.output_dim = heads * out_channels
        self.dropout = dropout

        self.gate_init_bias = float(getattr(config, "HYBRID_GATE_INIT_BIAS", -0.85))
        self.disagreement_scale = float(getattr(config, "HYBRID_DISAGREEMENT_SCALE", 2.0))

        self.gat_conv = TemporalGATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
            **kwargs,
        )

        self.sage_conv = EdgeAwareSAGEConv(
            in_channels=in_channels,
            out_channels=self.output_dim,
            edge_dim=edge_dim,
            normalize=True,
            dropout=dropout,
        )

        if gate_hidden_dim is None:
            gate_hidden_dim = max(self.output_dim // 2, 32)

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.output_dim * 3, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, self.output_dim),
        )

        # Start slightly biased toward SAGE:
        # sigmoid(-0.85) ≈ 0.30 → 30% GAT / 70% SAGE.
        if self.gate_mlp[-1].bias is not None:
            nn.init.constant_(self.gate_mlp[-1].bias, self.gate_init_bias)

        self.last_gate: Optional[torch.Tensor] = None
        self.last_disagreement: Optional[torch.Tensor] = None

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_gat = self.gat_conv(x, edge_index, edge_attr)
        h_sage = self.sage_conv(x, edge_index, edge_attr)

        diff = torch.abs(h_gat - h_sage)

        gate_input = torch.cat(
            [h_gat, h_sage, self.disagreement_scale * diff],
            dim=-1,
        )

        gate = torch.sigmoid(self.gate_mlp(gate_input))

        out = gate * h_gat + (1.0 - gate) * h_sage

        self.last_gate = gate.detach()
        self.last_disagreement = diff.detach()

        return out

    def get_gate_stats(self) -> Dict[str, float]:
        if self.last_gate is None:
            return {}

        gate = self.last_gate.float()
        stats = {
            "gate_mean": gate.mean().item(),
            "gate_std": gate.std(unbiased=False).item(),
            "gate_min": gate.min().item(),
            "gate_max": gate.max().item(),
            "sage_mean": (1.0 - gate).mean().item(),
        }

        if self.last_disagreement is not None:
            stats["disagreement_mean"] = self.last_disagreement.float().mean().item()
            stats["disagreement_std"] = self.last_disagreement.float().std(unbiased=False).item()

        return stats


# ============================================================
# 4. TemporalLSTM (explicit temporal spine)
# ============================================================

class TemporalLSTM(nn.Module):
    """
    Bi-directional LSTM for ordered sequential modeling.

    This is the explicit order-sensitive temporal branch and should not be
    replaced by generic relation message passing.
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
        if x.size(0) == 0:
            return x.new_zeros((0, self.output_dim))

        batch_size = int(batch.max().item()) + 1

        sequences = []
        inverse_orders = []

        for b in range(batch_size):
            mask = batch == b
            x_b = x[mask].contiguous()

            if x_b.size(0) == 0:
                sequences.append(None)
                inverse_orders.append(None)
                continue

            if timestamps is not None:
                t_b = timestamps[mask]
                order = torch.argsort(t_b)
                x_b = x_b[order].contiguous()
                inverse_orders.append(torch.argsort(order))
            else:
                inverse_orders.append(None)

            sequences.append(x_b)

        valid_sequences = [seq for seq in sequences if seq is not None]
        if len(valid_sequences) == 0:
            return x.new_zeros((0, self.output_dim))

        lengths = torch.tensor(
            [s.size(0) for s in valid_sequences],
            dtype=torch.long,
            device=x.device,
        )

        padded = torch.nn.utils.rnn.pad_sequence(
            valid_sequences,
            batch_first=True,
        ).contiguous()

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            padded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # FIX: force contiguous on packed data before LSTM
        packed = torch.nn.utils.rnn.PackedSequence(
            packed.data.contiguous(),
            packed.batch_sizes,
            packed.sorted_indices,
            packed.unsorted_indices,
        )

        self.lstm.flatten_parameters()

        # Run the LSTM in fp32 with cuDNN ENABLED (fast) but error-free.
        # cuDNN's fused RNN kernel is fast, but under AMP autocast the fp16
        # input vs. fp32 LSTM weights mismatch is what threw the original
        # "cudnn error" (hence the old enabled=False workaround). Forcing fp32
        # here matches the weights so the fast kernel runs without error, and
        # RNNs are kept in fp32 under AMP for numerical stability anyway.
        # No effect on CPU (autocast(enabled=False) is a no-op there).
        _dev_type = packed.data.device.type
        with torch.autocast(device_type=_dev_type, enabled=False):
            if packed.data.dtype != torch.float32:
                packed = torch.nn.utils.rnn.PackedSequence(
                    packed.data.float(),
                    packed.batch_sizes,
                    packed.sorted_indices,
                    packed.unsorted_indices,
                )
            packed_out, _ = self.lstm(packed)
        out_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
        )
        out_padded = out_padded.contiguous()

        outputs = []
        valid_idx = 0
        for i in range(len(sequences)):
            if sequences[i] is None:
                continue

            out_b = out_padded[valid_idx, : lengths[valid_idx]].contiguous()
            if inverse_orders[i] is not None:
                out_b = out_b[inverse_orders[i]].contiguous()

            outputs.append(out_b)
            valid_idx += 1

        return torch.cat(outputs, dim=0).contiguous()


# ============================================================
# 5. TemporalAttentionPooling
# ============================================================

class TemporalAttentionPooling(nn.Module):
    """
    Multi-head temporal attention pooling over node sequences per graph.
    """

    def __init__(self, input_dim: int, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0:
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            return x.new_zeros((batch_size, self.input_dim))

        batch_size = int(batch.max().item()) + 1
        graph_embeddings = []

        for b in range(batch_size):
            mask = batch == b
            x_b = x[mask]

            if x_b.size(0) == 0:
                graph_embeddings.append(x.new_zeros(self.input_dim))
                continue

            x_b = x_b.unsqueeze(0)  # [1, seq_len, input_dim]

            q = self.query(x_b.mean(dim=1, keepdim=True))
            k = self.key(x_b)
            v = self.value(x_b)

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

        return torch.stack(graph_embeddings, dim=0)


# ============================================================
# 6. HeteroTemporalConv
# ============================================================

class HeteroTemporalConv(nn.Module):
    """
    Heterogeneous relation-aware graph convolution.

    Notes:
    - Processes all provided edge types relation-by-relation.
    - Supports optional edge features per relation.
    - Safely skips missing node/edge types in a batch.
    - Does NOT replace the explicit TemporalLSTM sequence path.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        out_channels_dict: Dict[str, int],
        edge_types: List[EdgeType],
        layer_type: str = "gat",
        num_heads: int = 6,
        edge_dim_dict: Optional[Dict[EdgeType, int]] = None,
        dropout: float = 0.3,
        aggr: str = "sum",
    ):
        super().__init__()

        self.in_channels_dict = dict(in_channels_dict)
        self.out_channels_dict = dict(out_channels_dict)
        self.edge_types = list(edge_types)
        self.layer_type = layer_type.lower()
        self.num_heads = num_heads
        self.edge_dim_dict = dict(edge_dim_dict) if edge_dim_dict is not None else {}
        self.dropout = dropout
        self.aggr = aggr.lower()
        self._last_hybrid_gate_stats: Dict[str, Dict[str, float]] = {}

        if self.aggr not in {"sum", "mean", "max"}:
            raise ValueError(f"Unsupported aggregation mode: {aggr}")

        convs = {}
        for edge_type in self.edge_types:
            src, _, dst = edge_type

            if src not in self.in_channels_dict:
                raise KeyError(f"Missing in_channels for source node type '{src}'")
            if dst not in self.in_channels_dict:
                raise KeyError(f"Missing in_channels for destination node type '{dst}'")
            if dst not in self.out_channels_dict:
                raise KeyError(f"Missing out_channels for destination node type '{dst}'")

            if src == dst:
                in_ch = self.in_channels_dict[src]
            else:
                in_ch = (self.in_channels_dict[src], self.in_channels_dict[dst])

            out_ch = self.out_channels_dict[dst]
            edge_dim = int(self.edge_dim_dict.get(edge_type, 0) or 0)

            if self.layer_type == "gat":
                conv = TemporalGATConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    edge_dim=edge_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            elif self.layer_type == "transformer":
                conv = TemporalTransformerConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    edge_dim=edge_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            elif self.layer_type == "sage":
                conv = EdgeAwareSAGEConv(
                    in_channels=in_ch,
                    out_channels=out_ch * num_heads,
                    edge_dim=edge_dim,
                    normalize=True,
                    dropout=dropout,
                )
            elif self.layer_type in {"hybrid_gat_sage", "hybrid"}:
                conv = HybridGATSAGEConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    edge_dim=edge_dim,
                    heads=num_heads,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unknown layer_type: {layer_type}")

            convs[edge_type_to_key(edge_type)] = conv

        self.convs = nn.ModuleDict(convs)

    def get_registered_edge_types(self) -> List[EdgeType]:
        return list(self.edge_types)

    def get_hybrid_gate_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Return latest per-relation hybrid gate statistics.

        Empty unless layer_type is hybrid_gat_sage/hybrid and a forward pass has
        already been executed.
        """
        return dict(self._last_hybrid_gate_stats)

    def get_active_edge_types(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
    ) -> List[EdgeType]:
        """
        Return relation types that are actually usable in this forward pass.
        """
        active = []
        for edge_type in self.edge_types:
            src, _, dst = edge_type
            if edge_type not in edge_index_dict:
                continue
            if src not in x_dict or dst not in x_dict:
                continue
            edge_index = edge_index_dict[edge_type]
            if edge_index is None or edge_index.numel() == 0:
                continue
            active.append(edge_type)
        return active

    def _prepare_edge_attr(
        self,
        edge_type: EdgeType,
        edge_index: torch.Tensor,
        edge_attr_dict: Optional[Dict[EdgeType, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Return relation edge_attr if present, else zero-fill only when the relation
        is configured to expect edge features.
        """
        expected_dim = int(self.edge_dim_dict.get(edge_type, 0) or 0)

        if edge_attr_dict is not None and edge_type in edge_attr_dict:
            edge_attr = edge_attr_dict[edge_type]
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            return edge_attr.to(device=device, dtype=dtype)

        if expected_dim > 0:
            E = edge_index.size(1)
            return torch.zeros(E, expected_dim, device=device, dtype=dtype)

        return None

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
        edge_attr_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply per-relation convolution and aggregate to destination node types.

        Returns:
            Dict[node_type, updated_node_features]
        """
        out_dict: Dict[str, torch.Tensor] = {}
        contrib_count: Dict[str, int] = {}
        self._last_hybrid_gate_stats = {}

        # Gate the per-relation stats computation behind HYBRID_GATE_MONITOR.
        # When True (default) behavior is identical to before; when False we
        # skip get_gate_stats() and its ~5-7 .item() GPU syncs per hybrid conv
        # (downstream collectors already handle an empty stats dict gracefully).
        monitor_gate = getattr(config, "HYBRID_GATE_MONITOR", True)

        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict:
                continue

            src, _, dst = edge_type

            if src not in x_dict or dst not in x_dict:
                continue

            x_src = x_dict[src]
            x_dst = x_dict[dst]
            edge_index = edge_index_dict[edge_type]

            if not isinstance(edge_index, torch.Tensor):
                raise TypeError(f"edge_index for {edge_type} is not a tensor")

            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                raise ValueError(
                    f"edge_index for {edge_type} must have shape [2, E], "
                    f"got {tuple(edge_index.shape)}"
                )

            if edge_index.size(1) == 0:
                continue

            edge_attr = self._prepare_edge_attr(
                edge_type=edge_type,
                edge_index=edge_index,
                edge_attr_dict=edge_attr_dict,
                device=x_src.device,
                dtype=x_src.dtype,
            )

            conv = self.convs[edge_type_to_key(edge_type)]

            if src == dst:
                x_input = x_src
            else:
                x_input = (x_src, x_dst)

            out = conv(x_input, edge_index, edge_attr)

            if monitor_gate and hasattr(conv, "get_gate_stats"):
                stats = conv.get_gate_stats()
                if stats:
                    self._last_hybrid_gate_stats[edge_type_to_key(edge_type)] = stats

            if dst not in out_dict:
                out_dict[dst] = out
                contrib_count[dst] = 1
            else:
                if self.aggr == "sum":
                    out_dict[dst] = out_dict[dst] + out
                elif self.aggr == "mean":
                    out_dict[dst] = out_dict[dst] + out
                    contrib_count[dst] += 1
                elif self.aggr == "max":
                    out_dict[dst] = torch.max(out_dict[dst], out)

        if self.aggr == "mean":
            for dst, count in contrib_count.items():
                if count > 1:
                    out_dict[dst] = out_dict[dst] / float(count)

        return out_dict
