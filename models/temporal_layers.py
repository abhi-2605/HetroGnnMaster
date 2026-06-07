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

        # Run the LSTM in fp32 with cuDNN enabled (fast). cuDNN's fused RNN
        # rejects fp16 (dtype) and very long / capped-graph sequences
        # (CUDNN_STATUS_NOT_SUPPORTED, "non-contiguous input"). We force fp32 +
        # contiguous so it's happy on the ~96% of normal graphs, and fall back
        # to the native kernel ONLY for the rare big-graph batches cuDNN can't
        # handle — keeping speed everywhere else and never skipping a batch.
        # No effect on CPU (autocast(enabled=False) is a no-op there).
        _dev_type = packed.data.device.type
        with torch.autocast(device_type=_dev_type, enabled=False):
            if packed.data.dtype != torch.float32 or not packed.data.is_contiguous():
                packed = torch.nn.utils.rnn.PackedSequence(
                    packed.data.float().contiguous(),
                    packed.batch_sizes,
                    packed.sorted_indices,
                    packed.unsorted_indices,
                )
            try:
                packed_out, _ = self.lstm(packed)
            except RuntimeError as e:
                if "cudnn" in str(e).lower():
                    # Rare big/long-sequence graph cuDNN can't process: retry
                    # this one batch on the native kernel (slower, robust).
                    with torch.backends.cudnn.flags(enabled=False):
                        packed_out, _ = self.lstm(packed)
                else:
                    raise
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
# 4b. Time-feature encoding + Mamba temporal backbone (OPTIONAL, drop-in)
#     Additive: does NOT modify TemporalLSTM. mamba-ssm is imported lazily so
#     this file still imports fine on machines without it (LSTM path unaffected).
# ============================================================

def signed_log(x: torch.Tensor) -> torch.Tensor:
    """Monotonic, per-element scale compression. Preserves sign, order and
    relative magnitude (bigger stays bigger) while bounding huge values
    (1e6 -> ~14). Deterministic — no batch/chunk statistics — so it never
    flattens discrimination or shifts branch balance the way BatchNorm did,
    and it is invariant to batching/chunking."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


class Time2Vec(nn.Module):
    """Learnable time encoding: one linear term + (out_dim-1) periodic (sin)
    bases, so the model can represent *periodicity* directly — the
    'these events recur at interval X' signal — not just magnitude."""

    def __init__(self, out_dim: int):
        super().__init__()
        assert out_dim >= 2, "Time2Vec out_dim must be >= 2"
        self.lin = nn.Linear(1, 1)
        self.per = nn.Linear(1, out_dim - 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:   # t: [N, 1]
        return torch.cat([self.lin(t), torch.sin(self.per(t))], dim=-1)  # [N, out_dim]


class TimingEncoder(nn.Module):
    """Encode the per-node temporal edge features (the 14 dims) into a compact,
    learned time embedding WITHOUT killing the meaning of each value:
      - signed-log per-dim  -> bounds scale, keeps order/sign/identity (each
                               feature stays in its own slot; bigger stays bigger)
      - Time2Vec on delta   -> makes periodicity representable
      - small MLP           -> the model LEARNS what each value means / how it
                               weighs in (can amplify, not just dampen)
      - LayerNorm on OUTPUT -> keeps the embedding on a scale comparable to the
                               node embedding so the backbone actually USES it
                               (not 'pulled down' to a tiny ignored signal)
    NOTE: normalization is on the LEARNED embedding, not a global flatten of the
    raw features (the latter is what caused the earlier semantic monoculture)."""

    def __init__(self, n_features: int = 14, embed_dim: int = 16,
                 t2v_dim: int = 8, delta_index: int = 0):
        super().__init__()
        self.delta_index = delta_index
        self.t2v = Time2Vec(t2v_dim)
        self.mlp = nn.Sequential(
            nn.Linear(n_features + t2v_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # feats: [N, n_features] raw
        x = signed_log(feats)
        delta = signed_log(feats[:, self.delta_index:self.delta_index + 1])
        z = torch.cat([x, self.t2v(delta)], dim=-1)
        return self.norm(self.mlp(z))   # [N, embed_dim]


class TemporalMamba(nn.Module):
    """Drop-in replacement for TemporalLSTM.

    Same interface — forward(x, batch, timestamps) — plus an optional
    `step_features` arg carrying the raw 14-dim per-node timing (encoded
    internally by TimingEncoder). Handles long sequences natively (linear,
    strong long-range), so NO chunking is needed. Bidirectional via a forward
    pass + a separately-padded reversed pass (padding never contaminates the
    SSM state). Output is projected to `output_dim` (== lstm_output_dim) so the
    rest of the model (pooling/fusion/classifier) is untouched.

    mamba-ssm is imported lazily: this class is only constructed when the
    'mamba' backbone is selected, so the module imports fine without it."""

    def __init__(self, input_dim: int, output_dim: int, d_model: int = 256,
                 num_layers: int = 2, bidirectional: bool = True,
                 time_embed_dim: int = 0, time_feature_dim: int = 14):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as e:
            raise ImportError(
                "TEMPORAL_BACKBONE='mamba' needs mamba-ssm. On the CUDA GPU box:\n"
                "    pip install causal-conv1d mamba-ssm\n"
                "(Blackwell/sm_120 may require a source build of a recent version.)"
            ) from e

        self.bidirectional = bidirectional
        self.output_dim = output_dim
        self.time_encoder = (
            TimingEncoder(time_feature_dim, time_embed_dim) if time_embed_dim > 0 else None
        )
        eff_in = input_dim + (time_embed_dim if self.time_encoder is not None else 0)
        self.in_proj = nn.Linear(eff_in, d_model)

        self.fwd_blocks = nn.ModuleList([Mamba(d_model=d_model) for _ in range(num_layers)])
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_blocks = nn.ModuleList([Mamba(d_model=d_model) for _ in range(num_layers)])
            self.bwd_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.out_proj = nn.Linear(d_model * (2 if bidirectional else 1), output_dim)
        self._warned_fp32 = False   # one-time slow-path guard (see forward)

    @staticmethod
    def _run_stack(seq, blocks, norms):   # seq: [B, L, d_model] -> [B, L, d_model]
        h = seq
        for blk, nrm in zip(blocks, norms):
            h = h + blk(nrm(h))           # pre-norm residual
        return h

    def forward(self, x, batch, timestamps=None, step_features=None):
        if x.size(0) == 0:
            return x.new_zeros((0, self.output_dim))

        # fp32-speed guard: Mamba's fused kernel is fastest in bf16/fp16 (autocast).
        # Running on CUDA OUTSIDE autocast = slow fp32 path. Warn ONCE so a silent
        # slowdown can't sneak in (LSTM speed is unaffected — this is mamba-only).
        if (not self._warned_fp32) and x.is_cuda:
            try:
                in_autocast = torch.is_autocast_enabled()
            except Exception:
                in_autocast = True  # don't false-alarm if the API differs by version
            if not in_autocast:
                import warnings
                warnings.warn(
                    "TemporalMamba is running on CUDA OUTSIDE autocast (fp32 = SLOW "
                    "path). Run under torch.autocast(bf16/fp16) for full speed; the "
                    "LSTM's fp32 carve-out must NOT wrap the Mamba backbone.",
                    RuntimeWarning,
                )
            self._warned_fp32 = True

        if self.time_encoder is not None and step_features is not None:
            x = torch.cat([x, self.time_encoder(step_features)], dim=-1)
        x = self.in_proj(x)

        batch_size = int(batch.max().item()) + 1
        seqs, idxs, invs = [], [], []
        for b in range(batch_size):
            idx = (batch == b).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            x_b = x[idx]
            if timestamps is not None:
                order = torch.argsort(timestamps[idx])
                x_b = x_b[order]
                invs.append(torch.argsort(order))
            else:
                invs.append(None)
            seqs.append(x_b)
            idxs.append(idx)

        if not seqs:
            return x.new_zeros((0, self.output_dim))

        lengths = [s.size(0) for s in seqs]
        padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)   # [B, Lmax, d_model]
        out_f = self._run_stack(padded, self.fwd_blocks, self.fwd_norms)
        if self.bidirectional:
            # reverse each REAL sequence, then pad -> padding stays on the right,
            # so it never feeds the SSM before the real tokens.
            rev = torch.nn.utils.rnn.pad_sequence(
                [torch.flip(s, dims=[0]) for s in seqs], batch_first=True
            )
            out_b = self._run_stack(rev, self.bwd_blocks, self.bwd_norms)

        feat_dim = out_f.size(-1) * (2 if self.bidirectional else 1)
        combined = x.new_zeros((batch.size(0), feat_dim))
        for j in range(len(seqs)):
            L = lengths[j]
            f = out_f[j, :L]
            if self.bidirectional:
                b = torch.flip(out_b[j, :L], dims=[0])   # undo the reversal
                so = torch.cat([f, b], dim=-1)
            else:
                so = f
            if invs[j] is not None:
                so = so[invs[j]]                          # back to original node order
            combined[idxs[j]] = so

        return self.out_proj(combined)                    # [N, output_dim]


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
