#!/usr/bin/env python3
"""
baseline_models/compare_static_vs_existing_tgnn.py
--------------------------------------------------

Static GCN vs Static GAT vs existing TGNN checkpoint.

Goal:
    Prove whether static graph structure alone is enough.

Models:
    1. Static GCN structure-only
    2. Static GAT structure-only
    3. Existing TGNN checkpoint

Tracks:
    1. clean
    2. bigram_attack
    3. structure_attack
    4. combined_attack

Important:
    - Static GCN/GAT are trained fresh by default.
    - Existing TGNN is NOT trained. It is loaded from results/checkpoints/best_model.pt.
    - Static GCN/GAT do NOT receive API semantic embeddings.
    - Static GCN/GAT do NOT receive temporal edges or timestamps.
    - Static GCN/GAT do NOT receive behavioral shortcut count/stat features.
    - Structure attack changes non-temporal topology.
"""

from __future__ import annotations

import copy
import json
import pickle
import random
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import HeteroConv, GraphConv, GATConv
from torch_geometric.utils import scatter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config
from graph_construction.pytorch_dataset import hetero_collate, RansomwareGraphDataset
from graph_construction.vocabulary import load_vocab
from models.memory_utils import get_api_indices
from models.tgnn_model import create_hetero_tgnn
from utils.schema_manager import ensure_active_schema
from utils.schema_utils import (
    get_canonical_node_dims,
    get_edge_types,
    get_node_types,
    normalize_graph_node_features,
)
from utils.schema_validator import validate_batch_or_raise

from baseline_models.prove_classical_model_attacks import (
    build_custom_eval_manifest,
    compute_threat_vs_benign,
    label_to_name,
    print_metrics,
    _stats_from_info_df,
)


# ============================================================
# Local controls
# ============================================================

RANDOM_STATE = 42
ATTACK_BASE_SEED = 1337

OUT_DIR = Path(config.RESULTS_DIR) / "static_vs_existing_tgnn_strict"
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

STATIC_GCN_CKPT = MODEL_DIR / "static_gcn_strict_no_semantic_no_behavior_best.pt"
STATIC_GAT_CKPT = MODEL_DIR / "static_gat_strict_no_semantic_no_behavior_best.pt"
EXISTING_TGNN_CKPT = Path(config.RESULTS_DIR) / "checkpoints" / "best_model.pt"

# Force fresh static training to avoid accidentally loading older shortcut-based models.
FORCE_RETRAIN_STATIC = False

EVAL_TRACKS = ["clean", "bigram_attack", "structure_attack", "combined_attack"]

STATIC_EPOCHS = 30
STATIC_PATIENCE = 8
STATIC_LR = 1e-3
STATIC_WEIGHT_DECAY = 1e-4
STATIC_HIDDEN_DIM = 128
STATIC_POOLED_DIM = 128
STATIC_LAYERS = 2
STATIC_GAT_HEADS = 4

STATIC_COMPARE_TRAIN_FRAC = 0.50

# Static model restrictions.
STATIC_REMOVE_API_SEMANTIC_FEATURES = True
STATIC_API_FEATURE_REPLACEMENT = "zeros"  # "zeros" or "ones"
STATIC_USE_BEHAVIORAL_FEATURES = False

# Bigram/order attack.
STATIC_BIGRAM_ATTACK_MODE = "benign_interleaving"  # "benign_interleaving" or "full_shuffle"
STATIC_FULL_SHUFFLE_PRESERVE_FIRST = False
STATIC_FULL_SHUFFLE_PRESERVE_LAST = False
STATIC_INTERLEAVE_EVERY_K = 1
STATIC_INTERLEAVE_RATIO = 2.00
STATIC_INTERLEAVE_MAX_ADD = 30000

# Structure/topology attack.
STATIC_STRUCTURE_ATTACK_MODE = "combined_static"  # "edge_drop", "edge_rewire", "node_injection", "combined_static"
STATIC_EDGE_DROP_RATIO = 0.30
STATIC_EDGE_REWIRE_RATIO = 0.30
STATIC_NODE_INJECTION_RATIO = 0.50
STATIC_NODE_INJECTION_MAX_ADD_PER_TYPE = 3000

PADDING_API_POOL = [
    "sleep", "gettickcount", "closehandle", "getsystemtimeasfiletime",
    "queryperformancecounter", "getcurrentprocessid", "getcurrentthreadid",
    "getmodulehandlew", "getprocaddress", "virtualquery", "heapalloc",
    "heapfree", "entercriticalsection", "leavecriticalsection",
    "waitforsingleobject", "setevent", "resetevent", "createfilew",
    "readfile", "writefile", "getversion", "getlasterror",
    "initializecriticalsection", "deletecriticalsection", "gettemppathw",
    "getsystemdirectoryw",
]


# ============================================================
# Basic helpers
# ============================================================

def set_seed(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_label(all_categories: str) -> int:
    cats = str(all_categories).lower()
    if "ransomware" in cats:
        return 0
    if "benign" in cats:
        return 1
    return 2


def compute_metrics(predictions, labels):
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0, labels=[0, 1, 2]
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0, labels=[0, 1, 2]
    )
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0, labels=[0, 1, 2]
    )
    cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])

    return {
        "accuracy": float(acc),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(w_p),
        "weighted_recall": float(w_r),
        "weighted_f1": float(w_f1),
        "confusion_matrix": cm.tolist(),
    }


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = torch.tensor(alpha) if isinstance(alpha, list) else alpha

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        class_mask = F.one_hot(targets, num_classes=inputs.size(1)).float()
        probs = (probs * class_mask).sum(dim=1)
        focal_w = (1 - probs) ** self.gamma
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        loss = focal_w * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            loss = self.alpha[targets] * loss

        return loss.mean() if self.reduction == "mean" else loss.sum()


def safe_edge_attr_dict(batch_data):
    try:
        return batch_data.edge_attr_dict
    except (AttributeError, KeyError):
        return None


def safe_batch_dict(batch_data):
    try:
        return batch_data.batch_dict
    except (AttributeError, KeyError):
        return None


def _sample_seed_from_id(sample_id: str, base_seed: int) -> int:
    return (hash(sample_id) ^ base_seed) & 0xFFFFFFFF


def ensure_api_timestamps_for_attack(graph):
    try:
        tmp_proc = RansomwareGraphDataset.__new__(RansomwareGraphDataset)
        graph = tmp_proc._ensure_api_timestamps(graph)
    except Exception:
        pass
    return graph


# ============================================================
# API sequence helpers
# ============================================================

def _get_api_names_list(graph) -> List[str] | None:
    if not hasattr(graph, "node_types") or "api" not in graph.node_types:
        return None
    api_store = graph["api"]
    for attr in ["api_names", "names", "api_name"]:
        if hasattr(api_store, attr):
            return list(getattr(api_store, attr))
    return None


def _set_api_names_list(graph, new_names: List[str]) -> bool:
    if not hasattr(graph, "node_types") or "api" not in graph.node_types:
        return False
    api_store = graph["api"]
    for attr in ["api_names", "names", "api_name"]:
        if hasattr(api_store, attr):
            setattr(api_store, attr, new_names)
            return True
    return False


def _sort_api_order_from_graph(graph) -> Tuple[List[int], torch.Tensor | None]:
    if not hasattr(graph, "node_types") or "api" not in graph.node_types:
        return [], None
    if not hasattr(graph["api"], "timestamps") or not isinstance(graph["api"].timestamps, torch.Tensor):
        return [], None

    ts = graph["api"].timestamps
    if ts is None or ts.numel() == 0:
        return [], None

    order = torch.argsort(ts)
    return order.tolist(), ts


def _normalize_api_name(x) -> str:
    return str(x).strip().lower().replace(" ", "_") if x is not None else "unknown_api"


def extract_api_token_list_from_graph(graph) -> List[str]:
    try:
        if graph is None:
            return []
        if not hasattr(graph, "node_types") or "api" not in graph.node_types:
            return []

        names = _get_api_names_list(graph)
        if names is None:
            return []

        names = list(names)

        if hasattr(graph["api"], "timestamps") and graph["api"].timestamps is not None:
            try:
                ts = graph["api"].timestamps
                ts = ts.detach().cpu().numpy() if hasattr(ts, "detach") else np.asarray(ts)
                if len(ts) == len(names):
                    names = [name for _, name in sorted(zip(ts.tolist(), names), key=lambda x: x[0])]
            except Exception:
                pass

        return [_normalize_api_name(a) for a in names]
    except Exception:
        return []


def bigram_set(tokens: List[str]) -> Set[Tuple[str, str]]:
    if len(tokens) < 2:
        return set()
    return set(zip(tokens[:-1], tokens[1:]))


def compute_bigram_overlap(clean_tokens: List[str], attacked_tokens: List[str]) -> float:
    clean_b = bigram_set(clean_tokens)
    if not clean_b:
        return 1.0
    attacked_b = bigram_set(attacked_tokens)
    return float(len(clean_b & attacked_b) / len(clean_b))


def compute_unigram_overlap(clean_tokens: List[str], attacked_tokens: List[str]) -> float:
    clean_u = set(clean_tokens)
    if not clean_u:
        return 1.0
    attacked_u = set(attacked_tokens)
    return float(len(clean_u & attacked_u) / len(clean_u))


def compute_length_ratio(clean_tokens: List[str], attacked_tokens: List[str]) -> float:
    if len(clean_tokens) == 0:
        return 1.0
    return float(len(attacked_tokens) / len(clean_tokens))


# ============================================================
# Bigram/order attack
# ============================================================

def apply_full_shuffle_attack_local(graph, sample_id: str):
    graph = copy.deepcopy(graph)
    graph = ensure_api_timestamps_for_attack(graph)

    sorted_idx, ts = _sort_api_order_from_graph(graph)
    if not sorted_idx or ts is None or ts.numel() < 2:
        return graph

    order = torch.argsort(ts)
    sorted_ts = ts[order].clone()

    rng = random.Random(_sample_seed_from_id(sample_id, ATTACK_BASE_SEED + 10))

    middle = sorted_idx[:]
    first = []
    last = []

    if STATIC_FULL_SHUFFLE_PRESERVE_FIRST and middle:
        first = [middle[0]]
        middle = middle[1:]

    if STATIC_FULL_SHUFFLE_PRESERVE_LAST and middle:
        last = [middle[-1]]
        middle = middle[:-1]

    rng.shuffle(middle)
    permuted = first + middle + last

    new_ts = ts.clone()
    new_ts[torch.tensor(permuted, dtype=torch.long)] = sorted_ts
    graph["api"].timestamps = new_ts

    return graph


def apply_benign_interleaving_attack_local(graph, sample_id: str):
    graph = copy.deepcopy(graph)
    graph = ensure_api_timestamps_for_attack(graph)

    if not hasattr(graph, "node_types") or "api" not in graph.node_types:
        return graph
    if not hasattr(graph["api"], "x") or not isinstance(graph["api"].x, torch.Tensor):
        return graph

    names = _get_api_names_list(graph)
    if names is None:
        return graph

    x = graph["api"].x
    n = int(x.shape[0])
    if n <= 1 or len(names) != n:
        return graph

    rng = random.Random(_sample_seed_from_id(sample_id, ATTACK_BASE_SEED + 2001))

    order, ts = _sort_api_order_from_graph(graph)
    if not order:
        order = list(range(n))
        ts = None

    max_by_ratio = int(round(n * STATIC_INTERLEAVE_RATIO))
    target_add = max(1, min(max_by_ratio, STATIC_INTERLEAVE_MAX_ADD))

    insert_after_positions = list(range(0, max(0, len(order) - 1), max(1, STATIC_INTERLEAVE_EVERY_K)))
    if not insert_after_positions:
        insert_after_positions = [0]

    if len(insert_after_positions) > target_add:
        insert_after_positions = sorted(rng.sample(insert_after_positions, target_add))
    else:
        target_add = len(insert_after_positions)

    proto_idx = [order[rng.randrange(len(order))] for _ in range(target_add)]
    new_x = x[torch.tensor(proto_idx, dtype=torch.long)].clone()
    if new_x.numel() > 0:
        new_x = new_x + 0.001 * torch.randn_like(new_x)

    graph["api"].x = torch.cat([x, new_x], dim=0)

    new_api_indices = list(range(n, n + target_add))
    new_names = [rng.choice(PADDING_API_POOL) for _ in range(target_add)]
    _set_api_names_list(graph, list(names) + new_names)

    interleave_map = {}
    for pos, new_idx in zip(insert_after_positions, new_api_indices):
        interleave_map[pos] = new_idx

    attacked_order = []
    for i, old_idx in enumerate(order):
        attacked_order.append(old_idx)
        if i in interleave_map:
            attacked_order.append(interleave_map[i])

    if ts is not None and isinstance(ts, torch.Tensor) and ts.numel() == n:
        old_ts = ts.clone()
        base_sorted_ts = old_ts[torch.tensor(order, dtype=torch.long)].float()

        attacked_len = len(attacked_order)
        if attacked_len > 1:
            start_t = float(base_sorted_ts[0].item())
            end_t = float(base_sorted_ts[-1].item())
            if end_t <= start_t:
                end_t = start_t + float(attacked_len) * 1e-3

            attacked_ts_sorted = torch.linspace(
                start_t,
                end_t + 1e-3 * attacked_len,
                steps=attacked_len,
                dtype=old_ts.dtype,
            )
        else:
            attacked_ts_sorted = old_ts.new_tensor([float(base_sorted_ts[0].item())])

        new_ts = old_ts.new_empty(n + target_add)
        for idx, tval in zip(attacked_order, attacked_ts_sorted):
            new_ts[idx] = tval

        graph["api"].timestamps = new_ts

    for attr_name, fill_value in [("repeat_count", 1), ("run_duration", 0)]:
        if hasattr(graph["api"], attr_name):
            tensor = getattr(graph["api"], attr_name)
            if isinstance(tensor, torch.Tensor) and tensor.numel() == n:
                extra = torch.full((target_add,), fill_value, dtype=tensor.dtype)
                setattr(graph["api"], attr_name, torch.cat([tensor, extra], dim=0))

    graph["api"].num_nodes = n + target_add
    return graph


def apply_bigram_attack_local(graph, sample_id: str):
    if STATIC_BIGRAM_ATTACK_MODE == "full_shuffle":
        return apply_full_shuffle_attack_local(graph, sample_id)
    if STATIC_BIGRAM_ATTACK_MODE == "benign_interleaving":
        return apply_benign_interleaving_attack_local(graph, sample_id)
    raise ValueError(f"Unknown STATIC_BIGRAM_ATTACK_MODE: {STATIC_BIGRAM_ATTACK_MODE}")


# ============================================================
# Structure/topology attack
# ============================================================

def _node_count(graph, node_type: str) -> int:
    if not hasattr(graph, "node_types") or node_type not in graph.node_types:
        return 0
    store = graph[node_type]
    if hasattr(store, "num_nodes") and store.num_nodes is not None:
        return int(store.num_nodes)
    if hasattr(store, "x") and isinstance(store.x, torch.Tensor):
        return int(store.x.shape[0])
    return 0


def _is_temporal_edge_type(edge_type) -> bool:
    return len(edge_type) == 3 and edge_type[1] == "temporal"


def _non_temporal_edge_types(graph):
    return [et for et in list(getattr(graph, "edge_types", [])) if not _is_temporal_edge_type(et)]


def _edge_attr_keys(store):
    keys = []
    for key in ["edge_attr", "edge_weight", "timestamps", "time_delta", "delta_t"]:
        if hasattr(store, key):
            val = getattr(store, key)
            if isinstance(val, torch.Tensor):
                keys.append(key)
    return keys


def _filter_edge_store_columns(store, keep_idx: torch.Tensor):
    if hasattr(store, "edge_index") and isinstance(store.edge_index, torch.Tensor):
        store.edge_index = store.edge_index[:, keep_idx]

    for key in _edge_attr_keys(store):
        val = getattr(store, key)
        try:
            if val.size(0) == keep_idx.numel():
                setattr(store, key, val[keep_idx])
        except Exception:
            pass


def apply_edge_drop_attack(graph, sample_id: str, drop_ratio: float):
    graph = copy.deepcopy(graph)
    rng = random.Random(_sample_seed_from_id(sample_id, ATTACK_BASE_SEED + 3001))

    for et in _non_temporal_edge_types(graph):
        store = graph[et]
        if not hasattr(store, "edge_index") or not isinstance(store.edge_index, torch.Tensor):
            continue

        edge_index = store.edge_index
        e = int(edge_index.shape[1])
        if e <= 1:
            continue

        drop_n = int(round(e * drop_ratio))
        drop_n = max(0, min(drop_n, e - 1))
        if drop_n <= 0:
            continue

        drop_set = set(rng.sample(range(e), drop_n))
        keep = [i for i in range(e) if i not in drop_set]
        keep_idx = torch.tensor(keep, dtype=torch.long)
        _filter_edge_store_columns(store, keep_idx)

    return graph


def apply_edge_rewire_attack(graph, sample_id: str, rewire_ratio: float):
    graph = copy.deepcopy(graph)
    rng = random.Random(_sample_seed_from_id(sample_id, ATTACK_BASE_SEED + 4001))

    for et in _non_temporal_edge_types(graph):
        src_type, _, dst_type = et
        src_n = _node_count(graph, src_type)
        dst_n = _node_count(graph, dst_type)

        if src_n <= 0 or dst_n <= 0:
            continue

        store = graph[et]
        if not hasattr(store, "edge_index") or not isinstance(store.edge_index, torch.Tensor):
            continue

        edge_index = store.edge_index.clone()
        e = int(edge_index.shape[1])
        if e <= 1:
            continue

        rewire_n = int(round(e * rewire_ratio))
        rewire_n = max(1, min(rewire_n, e))

        chosen = rng.sample(range(e), rewire_n)

        for col in chosen:
            edge_index[0, col] = rng.randrange(src_n)
            edge_index[1, col] = rng.randrange(dst_n)

        store.edge_index = edge_index

    return graph


def _inject_nodes_for_type(graph, node_type: str, add_n: int, rng: random.Random):
    if add_n <= 0:
        return []

    store = graph[node_type]
    old_n = _node_count(graph, node_type)

    if not hasattr(store, "x") or not isinstance(store.x, torch.Tensor):
        return []

    x = store.x
    if old_n <= 0:
        return []

    proto_idx = torch.tensor([rng.randrange(old_n) for _ in range(add_n)], dtype=torch.long)
    new_x = x[proto_idx].clone()

    # For injected API nodes, do not give semantic embedding signal.
    if node_type == "api":
        new_x = torch.zeros_like(new_x)

    store.x = torch.cat([x, new_x], dim=0)
    store.num_nodes = old_n + add_n

    # Keep node-level tensors aligned with x after node injection.
    # This prevents TGNN preprocessing crashes like:
    # IndexError in _compress_consecutive_api_runs due to timestamps shorter than api.x.
    for attr_name, fill_mode in [
        ("timestamps", "increasing_time"),
        ("repeat_count", "ones"),
        ("run_duration", "zeros"),
    ]:
        if hasattr(store, attr_name):
            tensor = getattr(store, attr_name)

            if isinstance(tensor, torch.Tensor) and tensor.numel() == old_n:
                if fill_mode == "increasing_time":
                    last_t = float(tensor.max().item()) if tensor.numel() > 0 else 0.0
                    extra = tensor.new_tensor(
                        [last_t + 1e-3 * (i + 1) for i in range(add_n)]
                    )
                elif fill_mode == "ones":
                    extra = torch.ones(add_n, dtype=tensor.dtype, device=tensor.device)
                else:
                    extra = torch.zeros(add_n, dtype=tensor.dtype, device=tensor.device)

                setattr(store, attr_name, torch.cat([tensor, extra], dim=0))

    # Keep API names aligned too.
    if node_type == "api":
        names = _get_api_names_list(graph)
        if names is not None:
            _set_api_names_list(
                graph,
                list(names) + [rng.choice(PADDING_API_POOL) for _ in range(add_n)],
            )

        assert graph["api"].x.size(0) == graph["api"].num_nodes

        if hasattr(graph["api"], "timestamps"):
            assert graph["api"].timestamps.numel() == graph["api"].num_nodes, (
                "API timestamps are not aligned after node injection."
            )

        names_after = _get_api_names_list(graph)
        if names_after is not None:
            assert len(names_after) == graph["api"].num_nodes, (
                "API names are not aligned after node injection."
            )

    return list(range(old_n, old_n + add_n))


def _append_edges(graph, et, new_edges: List[Tuple[int, int]]):
    if not new_edges:
        return

    store = graph[et]
    new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()

    if hasattr(store, "edge_index") and isinstance(store.edge_index, torch.Tensor):
        old_e = int(store.edge_index.shape[1])
        store.edge_index = torch.cat([store.edge_index, new_edge_index], dim=1)
    else:
        old_e = 0
        store.edge_index = new_edge_index

    added_e = len(new_edges)

    for key in _edge_attr_keys(store):
        val = getattr(store, key)
        try:
            if val.size(0) == old_e:
                pad_shape = list(val.shape)
                pad_shape[0] = added_e
                pad = torch.zeros(*pad_shape, dtype=val.dtype)
                setattr(store, key, torch.cat([val, pad], dim=0))
        except Exception:
            pass


def apply_node_injection_attack(graph, sample_id: str, injection_ratio: float):
    graph = copy.deepcopy(graph)
    rng = random.Random(_sample_seed_from_id(sample_id, ATTACK_BASE_SEED + 5001))

    injected_by_type = {}

    for nt in list(getattr(graph, "node_types", [])):
        old_n = _node_count(graph, nt)
        if old_n <= 0:
            continue

        add_n = int(round(old_n * injection_ratio))
        add_n = max(0, min(add_n, STATIC_NODE_INJECTION_MAX_ADD_PER_TYPE))

        if add_n <= 0:
            continue

        injected_by_type[nt] = _inject_nodes_for_type(graph, nt, add_n, rng)

    for et in _non_temporal_edge_types(graph):
        src_type, _, dst_type = et
        src_old_n = _node_count(graph, src_type)
        dst_old_n = _node_count(graph, dst_type)

        src_new = injected_by_type.get(src_type, [])
        dst_new = injected_by_type.get(dst_type, [])

        new_edges = []

        for src_idx in src_new:
            if dst_old_n > 0:
                new_edges.append((src_idx, rng.randrange(dst_old_n)))

        for dst_idx in dst_new:
            if src_old_n > 0:
                new_edges.append((rng.randrange(src_old_n), dst_idx))

        _append_edges(graph, et, new_edges)

    return graph


def apply_structure_attack_local(graph, sample_id: str):
    if STATIC_STRUCTURE_ATTACK_MODE == "edge_drop":
        return apply_edge_drop_attack(graph, sample_id, STATIC_EDGE_DROP_RATIO)

    if STATIC_STRUCTURE_ATTACK_MODE == "edge_rewire":
        return apply_edge_rewire_attack(graph, sample_id, STATIC_EDGE_REWIRE_RATIO)

    if STATIC_STRUCTURE_ATTACK_MODE == "node_injection":
        return apply_node_injection_attack(graph, sample_id, STATIC_NODE_INJECTION_RATIO)

    if STATIC_STRUCTURE_ATTACK_MODE == "edge_drop_rewire":

        g = apply_edge_drop_attack(graph, sample_id, STATIC_EDGE_DROP_RATIO)

        g = apply_edge_rewire_attack(g, sample_id, STATIC_EDGE_REWIRE_RATIO)

        return g

    if STATIC_STRUCTURE_ATTACK_MODE == "combined_static":
        g = apply_edge_drop_attack(graph, sample_id, STATIC_EDGE_DROP_RATIO)
        g = apply_edge_rewire_attack(g, sample_id, STATIC_EDGE_REWIRE_RATIO)
        g = apply_node_injection_attack(g, sample_id, STATIC_NODE_INJECTION_RATIO)
        return g

    raise ValueError(f"Unknown STATIC_STRUCTURE_ATTACK_MODE: {STATIC_STRUCTURE_ATTACK_MODE}")


def apply_named_attack_local(graph, sample_id: str, attack_name: str):
    if attack_name == "clean":
        return graph

    if attack_name == "bigram_attack":
        return apply_bigram_attack_local(graph, sample_id)

    if attack_name == "structure_attack":
        return apply_structure_attack_local(graph, sample_id)

    if attack_name == "combined_attack":
        g = apply_bigram_attack_local(graph, sample_id)
        g = apply_structure_attack_local(g, sample_id)
        return g

    raise ValueError(f"Unknown attack_name: {attack_name}")


# ============================================================
# Static graph preparation
# ============================================================

def remove_temporal_information(graph):
    graph = copy.deepcopy(graph)

    for et in list(getattr(graph, "edge_types", [])):
        if _is_temporal_edge_type(et):
            try:
                del graph[et]
            except Exception:
                pass

    for nt in getattr(graph, "node_types", []):
        try:
            if hasattr(graph[nt], "timestamps"):
                delattr(graph[nt], "timestamps")
        except Exception:
            pass

    return graph


def remove_api_semantic_features(graph):
    if not STATIC_REMOVE_API_SEMANTIC_FEATURES:
        return graph

    if not hasattr(graph, "node_types") or "api" not in graph.node_types:
        return graph

    if not hasattr(graph["api"], "x") or graph["api"].x is None:
        return graph

    x = graph["api"].x

    if STATIC_API_FEATURE_REPLACEMENT == "ones":
        graph["api"].x = torch.ones_like(x)
    elif STATIC_API_FEATURE_REPLACEMENT == "zeros":
        graph["api"].x = torch.zeros_like(x)
        assert torch.all(graph["api"].x == 0), "API semantic features were not fully zeroed."
    else:
        raise ValueError("STATIC_API_FEATURE_REPLACEMENT must be 'zeros' or 'ones'.")

    return graph


def prepare_static_graph(graph):
    graph = remove_temporal_information(graph)
    graph = remove_api_semantic_features(graph)
    return graph


# ============================================================
# Eval track info
# ============================================================

def build_eval_track_info(manifest_df: pd.DataFrame, attack_name: str):
    labels = []
    sample_ids = []
    origins = []
    graph_paths = []
    bigram_overlaps = []
    unigram_overlaps = []
    length_ratios = []

    for _, row in manifest_df.iterrows():
        graph_path = Path(row["graph_path"])
        sample_id = str(row["sample_id"])

        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        graph = ensure_api_timestamps_for_attack(graph)
        clean_tokens = extract_api_token_list_from_graph(graph)

        attacked_graph = apply_named_attack_local(graph, sample_id, attack_name)
        attacked_tokens = extract_api_token_list_from_graph(attacked_graph)

        labels.append(int(row["label"]))
        sample_ids.append(sample_id)
        origins.append(str(row.get("origin", "")))
        graph_paths.append(str(graph_path))

        bigram_overlaps.append(
            1.0 if attack_name == "clean" else compute_bigram_overlap(clean_tokens, attacked_tokens)
        )
        unigram_overlaps.append(
            1.0 if attack_name == "clean" else compute_unigram_overlap(clean_tokens, attacked_tokens)
        )
        length_ratios.append(
            1.0 if attack_name == "clean" else compute_length_ratio(clean_tokens, attacked_tokens)
        )

    info_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "origin": origins,
            "graph_path": graph_paths,
            "label": labels,
            "attack_name": [attack_name] * len(sample_ids),
            "bigram_overlap_vs_clean": bigram_overlaps,
            "unigram_overlap_vs_clean": unigram_overlaps,
            "length_ratio_vs_clean": length_ratios,
        }
    )

    print(
        f"[INFO] Eval track prepared: {attack_name} | "
        f"samples={len(info_df)} | "
        f"bigram_overlap_mean={info_df['bigram_overlap_vs_clean'].mean():.4f} | "
        f"length_ratio_mean={info_df['length_ratio_vs_clean'].mean():.4f}"
    )

    return info_df


# ============================================================
# Datasets
# ============================================================

class StaticSplitDataset(Dataset):
    def __init__(self, split_name: str, schema: Dict, graph_processor):
        self.split_name = split_name
        self.schema = schema
        self.graph_processor = graph_processor

        metadata_df = pd.read_csv(config.METADATA_PATH)
        metadata_df["_md5"] = metadata_df["md5"].astype(str).str.strip().str.lower()
        self.label_map = {
            row["_md5"]: get_label(row["all_categories"])
            for _, row in metadata_df.iterrows()
        }

        split_dir = Path(config.ACTIVE_GRAPH_DIR) / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        self.rows = []
        for graph_path in sorted(split_dir.glob("*.pkl")):
            sample_id = graph_path.stem.strip().lower()
            if sample_id not in self.label_map:
                continue
            self.rows.append(
                {
                    "sample_id": sample_id,
                    "graph_path": str(graph_path),
                    "label": int(self.label_map[sample_id]),
                    "origin": split_name,
                }
            )

        if split_name == "train":
            if not (0.0 < STATIC_COMPARE_TRAIN_FRAC <= 1.0):
                raise ValueError("STATIC_COMPARE_TRAIN_FRAC must be in (0, 1].")

            original_n = len(self.rows)
            keep_n = max(1, int(round(original_n * STATIC_COMPARE_TRAIN_FRAC)))
            rng = random.Random(RANDOM_STATE)
            rng.shuffle(self.rows)
            self.rows = self.rows[:keep_n]

            print(f"[INFO] StaticSplitDataset[train] subsampled: {keep_n}/{original_n} ({STATIC_COMPARE_TRAIN_FRAC:.0%})")

        if not self.rows:
            raise RuntimeError(f"No graphs found for split: {split_name}")

        print(f"[INFO] StaticSplitDataset[{split_name}] samples={len(self.rows)}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        graph_path = Path(row["graph_path"])

        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        graph = self.graph_processor._ensure_api_timestamps(graph)
        graph = prepare_static_graph(graph)

        if config.NORMALIZE_NODE_FEATURE_DIMS:
            graph = normalize_graph_node_features(graph, schema=self.schema, inplace=True)

        graph = self.graph_processor._compress_consecutive_api_runs(graph)
        graph = self.graph_processor._behavior_aware_api_cap(graph)
        graph = self.graph_processor._ensure_api_aux_tensors(graph)

        graph = prepare_static_graph(graph)

        graph.sample_id = row["sample_id"]
        graph.graph_path = str(graph_path)
        graph.origin = row["origin"]
        graph.attack_name = "train_or_val"
        graph.bigram_overlap_vs_clean = 1.0
        graph.unigram_overlap_vs_clean = 1.0
        graph.length_ratio_vs_clean = 1.0

        return graph, int(row["label"])


class EvalManifestDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, schema: Dict, graph_processor, attack_name: str, static_mode: bool):
        self.df = pd.read_csv(Path(manifest_csv))
        self.schema = schema
        self.graph_processor = graph_processor
        self.attack_name = attack_name
        self.static_mode = static_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        graph_path = Path(row["graph_path"])
        label = int(row["label"])
        sample_id = str(row["sample_id"])

        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        graph = self.graph_processor._ensure_api_timestamps(graph)
        graph = apply_named_attack_local(graph, sample_id, self.attack_name)

        if self.static_mode:
            graph = prepare_static_graph(graph)

        if config.NORMALIZE_NODE_FEATURE_DIMS:
            graph = normalize_graph_node_features(graph, schema=self.schema, inplace=True)

        graph = self.graph_processor._compress_consecutive_api_runs(graph)
        graph = self.graph_processor._behavior_aware_api_cap(graph)
        graph = self.graph_processor._ensure_api_aux_tensors(graph)

        if self.static_mode:
            graph = prepare_static_graph(graph)

        graph.sample_id = sample_id
        graph.graph_path = str(graph_path)
        graph.origin = str(row.get("origin", ""))
        graph.attack_name = self.attack_name
        graph.bigram_overlap_vs_clean = float(row.get("bigram_overlap_vs_clean", 1.0))
        graph.unigram_overlap_vs_clean = float(row.get("unigram_overlap_vs_clean", 1.0))
        graph.length_ratio_vs_clean = float(row.get("length_ratio_vs_clean", 1.0))

        return graph, label


def custom_collate(batch, schema, validate_batch=False):
    if not batch:
        return None, None

    sample_ids, graph_paths, origins, attack_names = [], [], [], []
    bigram_overlaps, unigram_overlaps, length_ratios = [], [], []

    for graph, _ in batch:
        sample_ids.append(getattr(graph, "sample_id", ""))
        graph_paths.append(getattr(graph, "graph_path", ""))
        origins.append(getattr(graph, "origin", ""))
        attack_names.append(getattr(graph, "attack_name", ""))
        bigram_overlaps.append(getattr(graph, "bigram_overlap_vs_clean", 1.0))
        unigram_overlaps.append(getattr(graph, "unigram_overlap_vs_clean", 1.0))
        length_ratios.append(getattr(graph, "length_ratio_vs_clean", 1.0))

    batched_graph, labels = hetero_collate(batch, schema=schema, validate_batch=validate_batch)

    if batched_graph is None or labels is None:
        return batched_graph, labels

    batched_graph.sample_id = sample_ids
    batched_graph.graph_path = graph_paths
    batched_graph.origin = origins
    batched_graph.attack_name = attack_names
    batched_graph.bigram_overlap_vs_clean = bigram_overlaps
    batched_graph.unigram_overlap_vs_clean = unigram_overlaps
    batched_graph.length_ratio_vs_clean = length_ratios

    return batched_graph, labels


# ============================================================
# Static model
# ============================================================

class StaticHeteroGNN(nn.Module):
    def __init__(self, schema: Dict, kind: str = "gcn"):
        super().__init__()

        if kind not in {"gcn", "gat"}:
            raise ValueError("kind must be 'gcn' or 'gat'")

        self.kind = kind
        self.node_types = get_node_types(schema)
        self.edge_types = [et for et in get_edge_types(schema) if et[1] != "temporal"]
        self.node_dims = get_canonical_node_dims(schema)

        self.input_proj = nn.ModuleDict()
        for nt in self.node_types:
            self.input_proj[nt] = nn.Sequential(
                nn.Linear(int(self.node_dims[nt]), STATIC_HIDDEN_DIM),
                nn.LayerNorm(STATIC_HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(getattr(config, "DROPOUT", 0.3)),
            )

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(STATIC_LAYERS):
            convs = {}
            for et in self.edge_types:
                if kind == "gcn":
                    convs[et] = GraphConv(
                        in_channels=(STATIC_HIDDEN_DIM, STATIC_HIDDEN_DIM),
                        out_channels=STATIC_HIDDEN_DIM,
                    )
                else:
                    if STATIC_HIDDEN_DIM % STATIC_GAT_HEADS != 0:
                        raise ValueError("STATIC_HIDDEN_DIM must be divisible by STATIC_GAT_HEADS.")
                    convs[et] = GATConv(
                        in_channels=(STATIC_HIDDEN_DIM, STATIC_HIDDEN_DIM),
                        out_channels=STATIC_HIDDEN_DIM // STATIC_GAT_HEADS,
                        heads=STATIC_GAT_HEADS,
                        concat=True,
                        dropout=getattr(config, "DROPOUT", 0.3),
                        add_self_loops=False,
                    )

            self.layers.append(HeteroConv(convs, aggr="sum"))
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(STATIC_HIDDEN_DIM) for nt in self.node_types}))

        self.type_proj = nn.ModuleDict()
        for nt in self.node_types:
            self.type_proj[nt] = nn.Sequential(
                nn.Linear(STATIC_HIDDEN_DIM, STATIC_POOLED_DIM),
                nn.ReLU(),
                nn.Dropout(getattr(config, "DROPOUT", 0.3)),
            )

        classifier_in = STATIC_POOLED_DIM

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(getattr(config, "DROPOUT", 0.3)),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(getattr(config, "DROPOUT", 0.3)),
            nn.Linear(128, 3),
        )

    def forward(self, x_dict, edge_index_dict, batch_dict, edge_attr_dict=None, timestamps_dict=None, api_indices=None, current_epoch=None):
        device = next(self.parameters()).device

        h_dict = {nt: self.input_proj[nt](x) for nt, x in x_dict.items() if nt in self.input_proj}

        static_edge_index_dict = {
            et: ei
            for et, ei in edge_index_dict.items()
            if et in self.edge_types and ei is not None and ei.numel() > 0
        }

        for layer, norm in zip(self.layers, self.norms):
            out = layer(h_dict, static_edge_index_dict) if static_edge_index_dict else {}
            new_h = {}

            for nt in h_dict:
                if nt in out:
                    z = norm[nt](out[nt])
                    if z.shape == h_dict[nt].shape:
                        z = z + h_dict[nt]
                    new_h[nt] = F.dropout(F.relu(z), p=getattr(config, "DROPOUT", 0.3), training=self.training)
                else:
                    new_h[nt] = h_dict[nt]

            h_dict = new_h

        if "api" in batch_dict and batch_dict["api"].numel() > 0:
            batch_size = int(batch_dict["api"].max().item()) + 1
        else:
            first_batch = next(iter(batch_dict.values()))
            batch_size = int(first_batch.max().item()) + 1

        type_embs = []
        type_masks = []

        for nt in self.node_types:
            if nt in h_dict and nt in batch_dict and batch_dict[nt].numel() > 0:
                pooled = scatter(h_dict[nt], batch_dict[nt], dim=0, dim_size=batch_size, reduce="mean")
                emb = self.type_proj[nt](pooled)
                mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                mask[batch_dict[nt].unique()] = True
            else:
                emb = torch.zeros(batch_size, STATIC_POOLED_DIM, device=device)
                mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

            type_embs.append(emb)
            type_masks.append(mask)

        type_embs = torch.stack(type_embs, dim=1)
        type_masks = torch.stack(type_masks, dim=1)

        denom = type_masks.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        graph_emb = (type_embs * type_masks.unsqueeze(-1).float()).sum(dim=1) / denom

        return self.classifier(graph_emb)


# ============================================================
# Train/eval
# ============================================================

def make_loader(dataset, schema, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=partial(custom_collate, schema=schema, validate_batch=False),
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )


def forward_model(model, batch_data, labels, criterion, device, schema, vocab=None, static_mode=True, eval_epoch=None):
    labels = labels.to(device)
    batch_data = batch_data.to(device)

    edge_attr_dict = safe_edge_attr_dict(batch_data)
    batch_dict = safe_batch_dict(batch_data)

    if static_mode:
        timestamps_dict = None
        api_indices = None
    else:
        timestamps_dict = {}
        for nt in batch_data.x_dict.keys():
            if hasattr(batch_data[nt], "timestamps"):
                timestamps_dict[nt] = batch_data[nt].timestamps
        timestamps_dict = timestamps_dict or None

        try:
            api_names = batch_data["api"].api_names
            api_indices = get_api_indices(api_names, vocab, device=device)
        except Exception:
            n = batch_data["api"].x.size(0)
            api_indices = torch.zeros(n, dtype=torch.long, device=device)

        validate_batch_or_raise(
            x_dict=batch_data.x_dict,
            edge_index_dict=batch_data.edge_index_dict,
            edge_attr_dict=edge_attr_dict,
            batch_dict=batch_dict,
            schema=schema,
            require_api=True,
            require_api_indices_if_memory=(api_indices is not None),
            api_indices=api_indices,
        )

    logits = model(
        x_dict=batch_data.x_dict,
        edge_index_dict=batch_data.edge_index_dict,
        edge_attr_dict=edge_attr_dict,
        batch_dict=batch_dict,
        timestamps_dict=timestamps_dict,
        api_indices=api_indices,
        current_epoch=eval_epoch,
    )

    loss = criterion(logits, labels)
    return logits, loss, labels


def train_static_model(kind: str, schema: Dict, train_loader, val_loader, device, criterion):
    model = StaticHeteroGNN(schema=schema, kind=kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=STATIC_LR, weight_decay=STATIC_WEIGHT_DECAY)

    best_macro_f1 = -1.0
    best_state = None
    patience_left = STATIC_PATIENCE

    for epoch in range(1, STATIC_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        steps = 0

        for batch_item in train_loader:
            if batch_item is None:
                continue

            batch_data, labels = batch_item
            if batch_data is None or labels is None:
                continue

            optimizer.zero_grad(set_to_none=True)

            _, loss, _ = forward_model(
                model=model,
                batch_data=batch_data,
                labels=labels,
                criterion=criterion,
                device=device,
                schema=schema,
                static_mode=True,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += float(loss.item())
            steps += 1

        val_metrics, _ = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            schema=schema,
            model_name=f"Static {kind.upper()}",
            static_mode=True,
        )

        val_macro_f1 = float(val_metrics["macro_f1"])
        avg_train_loss = train_loss / max(steps, 1)

        print(
            f"[TRAIN][Static {kind.upper()}] "
            f"epoch={epoch:03d} train_loss={avg_train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_left = STATIC_PATIENCE
        else:
            patience_left -= 1

        if patience_left <= 0:
            print(f"[INFO] Early stopping Static {kind.upper()} at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_macro_f1


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, schema, model_name: str, static_mode: bool, vocab=None, eval_epoch=None):
    model.eval()

    all_preds = []
    all_labels = []
    rows = []
    total_loss = 0.0
    steps = 0

    for batch_item in loader:
        if batch_item is None:
            continue

        batch_data, labels = batch_item
        if batch_data is None or labels is None:
            continue

        logits, loss, labels = forward_model(
            model=model,
            batch_data=batch_data,
            labels=labels,
            criterion=criterion,
            device=device,
            schema=schema,
            vocab=vocab,
            static_mode=static_mode,
            eval_epoch=eval_epoch,
        )

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        total_loss += float(loss.item())
        steps += 1

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        sample_ids = getattr(batch_data, "sample_id", [""] * labels.size(0))
        graph_paths = getattr(batch_data, "graph_path", [""] * labels.size(0))
        origins = getattr(batch_data, "origin", [""] * labels.size(0))
        attack_names = getattr(batch_data, "attack_name", [""] * labels.size(0))
        bigram_overlaps = getattr(batch_data, "bigram_overlap_vs_clean", [1.0] * labels.size(0))
        unigram_overlaps = getattr(batch_data, "unigram_overlap_vs_clean", [1.0] * labels.size(0))
        length_ratios = getattr(batch_data, "length_ratio_vs_clean", [1.0] * labels.size(0))

        for i in range(labels.size(0)):
            rows.append(
                {
                    "sample_id": sample_ids[i],
                    "origin": origins[i],
                    "graph_path": graph_paths[i],
                    "attack_name": attack_names[i],
                    "bigram_overlap_vs_clean": float(bigram_overlaps[i]),
                    "unigram_overlap_vs_clean": float(unigram_overlaps[i]),
                    "length_ratio_vs_clean": float(length_ratios[i]),
                    "true_label": int(labels[i].item()),
                    "true_name": label_to_name(int(labels[i].item())),
                    "pred_label": int(preds[i].item()),
                    "pred_name": label_to_name(int(preds[i].item())),
                    "prob_ransomware": float(probs[i, 0].item()),
                    "prob_benign": float(probs[i, 1].item()),
                    "prob_other_malware": float(probs[i, 2].item()),
                    "correct": int(preds[i].item() == labels[i].item()),
                    "model": model_name,
                }
            )

    if not all_labels:
        raise RuntimeError(f"No evaluation samples processed for {model_name}")

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / max(steps, 1)
    return metrics, rows


def load_tgnn_existing(device):
    if not EXISTING_TGNN_CKPT.exists():
        raise FileNotFoundError(f"Existing TGNN checkpoint not found: {EXISTING_TGNN_CKPT}")

    model = create_hetero_tgnn(use_memory=config.USE_MEMORY).to(device)
    ckpt = torch.load(EXISTING_TGNN_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    print(f"[INFO] Loaded existing TGNN checkpoint: {EXISTING_TGNN_CKPT}")

    eval_epoch = int(ckpt.get("epoch", config.EPOCHS))
    return model, eval_epoch


def make_eval_loader(track_name, manifest_df, info_df, schema, graph_processor, static_mode):
    manifest_path = OUT_DIR / f"tmp_{'static' if static_mode else 'tgnn'}_{track_name}_manifest.csv"

    tmp_df = manifest_df.copy()
    tmp_df["attack_name"] = track_name
    tmp_df["bigram_overlap_vs_clean"] = info_df["bigram_overlap_vs_clean"].tolist()
    tmp_df["unigram_overlap_vs_clean"] = info_df["unigram_overlap_vs_clean"].tolist()
    tmp_df["length_ratio_vs_clean"] = info_df["length_ratio_vs_clean"].tolist()
    tmp_df.to_csv(manifest_path, index=False)

    dataset = EvalManifestDataset(
        manifest_csv=manifest_path,
        schema=schema,
        graph_processor=graph_processor,
        attack_name=track_name,
        static_mode=static_mode,
    )

    return make_loader(dataset, schema=schema, shuffle=False)


def metric_cell(metrics):
    tvb = compute_threat_vs_benign(np.array(metrics["confusion_matrix"]))
    return f"{metrics['accuracy'] * 100:.2f}% / {tvb['threats_missed']} missed"


def save_report(rows, out_path):
    lines = []
    lines.append("# Static GNN vs Existing TGNN")
    lines.append("")
    lines.append("| Model | Clean Acc / Missed | Bigram Attack Acc / Missed | Structure Attack Acc / Missed | Combined Attack Acc / Missed |")
    lines.append("|---|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['model']} | "
            f"{row.get('clean', '—')} | "
            f"{row.get('bigram_attack', '—')} | "
            f"{row.get('structure_attack', '—')} | "
            f"{row.get('combined_attack', '—')} |"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main():
    set_seed(RANDOM_STATE)

    print(f"[INFO] Output directory: {OUT_DIR}")
    print(f"[INFO] Static model directory: {MODEL_DIR}")
    print(f"[INFO] Existing TGNN checkpoint: {EXISTING_TGNN_CKPT}")
    print(f"[INFO] Force retrain static models: {FORCE_RETRAIN_STATIC}")
    print(f"[INFO] Static train fraction: {STATIC_COMPARE_TRAIN_FRAC}")
    print(f"[INFO] Static uses behavioral shortcut features: {STATIC_USE_BEHAVIORAL_FEATURES}")
    print(f"[INFO] Remove API semantic features from static models: {STATIC_REMOVE_API_SEMANTIC_FEATURES}")
    print(f"[INFO] Static API feature replacement: {STATIC_API_FEATURE_REPLACEMENT}")
    print(f"[INFO] Bigram attack mode: {STATIC_BIGRAM_ATTACK_MODE}")
    print(f"[INFO] Structure attack mode: {STATIC_STRUCTURE_ATTACK_MODE}")
    print(f"[INFO] Edge drop ratio: {STATIC_EDGE_DROP_RATIO}")
    print(f"[INFO] Edge rewire ratio: {STATIC_EDGE_REWIRE_RATIO}")
    print(f"[INFO] Node injection ratio: {STATIC_NODE_INJECTION_RATIO}")

    device = torch.device(config.DEVICE)
    print(f"[INFO] Device: {device}")

    schema = ensure_active_schema()
    if not isinstance(schema, dict):
        with open(config.ACTIVE_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)

    graph_processor = RansomwareGraphDataset.__new__(RansomwareGraphDataset)
    graph_processor.schema = schema

    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)

    train_dataset = StaticSplitDataset("train", schema, graph_processor)
    val_dataset = StaticSplitDataset("val", schema, graph_processor)

    train_loader = make_loader(train_dataset, schema=schema, shuffle=True)
    val_loader = make_loader(val_dataset, schema=schema, shuffle=False)

    if STATIC_GCN_CKPT.exists() and not FORCE_RETRAIN_STATIC:
        static_gcn = StaticHeteroGNN(schema=schema, kind="gcn").to(device)
        static_gcn.load_state_dict(torch.load(STATIC_GCN_CKPT, map_location=device, weights_only=False)["model_state_dict"])
        print(f"[INFO] Loaded Static GCN from {STATIC_GCN_CKPT}")
    else:
        static_gcn, best_f1 = train_static_model("gcn", schema, train_loader, val_loader, device, criterion)
        torch.save(
            {
                "model_state_dict": static_gcn.state_dict(),
                "best_val_macro_f1": best_f1,
                "model": "Static GCN strict structure-only",
                "train_fraction": STATIC_COMPARE_TRAIN_FRAC,
                "remove_api_semantic_features": STATIC_REMOVE_API_SEMANTIC_FEATURES,
                "use_behavioral_features": STATIC_USE_BEHAVIORAL_FEATURES,
            },
            STATIC_GCN_CKPT,
        )
        print(f"[INFO] Saved Static GCN checkpoint: {STATIC_GCN_CKPT}")

    if STATIC_GAT_CKPT.exists() and not FORCE_RETRAIN_STATIC:
        static_gat = StaticHeteroGNN(schema=schema, kind="gat").to(device)
        static_gat.load_state_dict(torch.load(STATIC_GAT_CKPT, map_location=device, weights_only=False)["model_state_dict"])
        print(f"[INFO] Loaded Static GAT from {STATIC_GAT_CKPT}")
    else:
        static_gat, best_f1 = train_static_model("gat", schema, train_loader, val_loader, device, criterion)
        torch.save(
            {
                "model_state_dict": static_gat.state_dict(),
                "best_val_macro_f1": best_f1,
                "model": "Static GAT strict structure-only",
                "train_fraction": STATIC_COMPARE_TRAIN_FRAC,
                "remove_api_semantic_features": STATIC_REMOVE_API_SEMANTIC_FEATURES,
                "use_behavioral_features": STATIC_USE_BEHAVIORAL_FEATURES,
            },
            STATIC_GAT_CKPT,
        )
        print(f"[INFO] Saved Static GAT checkpoint: {STATIC_GAT_CKPT}")

    vocab = load_vocab(config.RESULTS_DIR / "api_vocabulary.json")
    tgnn_model, tgnn_eval_epoch = load_tgnn_existing(device=device)

    if getattr(config, "CUSTOM_EVAL_REBUILD_MANIFEST", True) or not Path(config.CUSTOM_EVAL_MANIFEST_PATH).exists():
        custom_manifest_df = build_custom_eval_manifest()
    else:
        custom_manifest_df = pd.read_csv(config.CUSTOM_EVAL_MANIFEST_PATH)
        print(f"[INFO] Using existing custom eval manifest: {config.CUSTOM_EVAL_MANIFEST_PATH}")

    eval_tracks = {}
    for track_name in EVAL_TRACKS:
        eval_tracks[track_name] = build_eval_track_info(custom_manifest_df, attack_name=track_name)

    loaders_static = {}
    loaders_tgnn = {}

    for track_name in EVAL_TRACKS:
        info_df = eval_tracks[track_name]

        loaders_static[track_name] = make_eval_loader(
            track_name=track_name,
            manifest_df=custom_manifest_df,
            info_df=info_df,
            schema=schema,
            graph_processor=graph_processor,
            static_mode=True,
        )

        loaders_tgnn[track_name] = make_eval_loader(
            track_name=track_name,
            manifest_df=custom_manifest_df,
            info_df=info_df,
            schema=schema,
            graph_processor=graph_processor,
            static_mode=False,
        )

    model_specs = [
        ("Static GCN strict structure-only", static_gcn, True, None),
        ("Static GAT strict structure-only", static_gat, True, None),
        ("Existing TGNN", tgnn_model, False, tgnn_eval_epoch),
    ]

    summary = {
        "description": "Strict Static GCN/GAT vs existing TGNN under order and topology attacks.",
        "output_dir": str(OUT_DIR),
        "checkpoints": {
            "static_gcn": str(STATIC_GCN_CKPT),
            "static_gat": str(STATIC_GAT_CKPT),
            "existing_tgnn": str(EXISTING_TGNN_CKPT),
        },
        "static_setup": {
            "train_fraction": STATIC_COMPARE_TRAIN_FRAC,
            "remove_temporal_edges": True,
            "remove_timestamps": True,
            "remove_api_semantic_features": STATIC_REMOVE_API_SEMANTIC_FEATURES,
            "api_feature_replacement": STATIC_API_FEATURE_REPLACEMENT,
            "use_behavioral_features": STATIC_USE_BEHAVIORAL_FEATURES,
        },
        "attacks": {
            "bigram_attack": {
                "mode": STATIC_BIGRAM_ATTACK_MODE,
                "interleave_every_k": STATIC_INTERLEAVE_EVERY_K,
                "interleave_ratio": STATIC_INTERLEAVE_RATIO,
                "interleave_max_add": STATIC_INTERLEAVE_MAX_ADD,
                "full_shuffle_preserve_first": STATIC_FULL_SHUFFLE_PRESERVE_FIRST,
                "full_shuffle_preserve_last": STATIC_FULL_SHUFFLE_PRESERVE_LAST,
            },
            "structure_attack": {
                "mode": STATIC_STRUCTURE_ATTACK_MODE,
                "edge_drop_ratio": STATIC_EDGE_DROP_RATIO,
                "edge_rewire_ratio": STATIC_EDGE_REWIRE_RATIO,
                "node_injection_ratio": STATIC_NODE_INJECTION_RATIO,
                "node_injection_max_add_per_type": STATIC_NODE_INJECTION_MAX_ADD_PER_TYPE,
            },
        },
        "tracks": EVAL_TRACKS,
        "attack_stats": {
            track_name: _stats_from_info_df(eval_tracks[track_name])
            for track_name in EVAL_TRACKS
        },
        "results": {},
    }

    report_rows = []

    for model_name, model, static_mode, eval_epoch in model_specs:
        row = {"model": model_name}
        summary["results"][model_name] = {}

        for track_name in EVAL_TRACKS:
            loader = loaders_static[track_name] if static_mode else loaders_tgnn[track_name]

            metrics, pred_rows = evaluate_model(
                model=model,
                loader=loader,
                criterion=criterion,
                device=device,
                schema=schema,
                model_name=model_name,
                static_mode=static_mode,
                vocab=vocab,
                eval_epoch=eval_epoch,
            )

            print_metrics(metrics, f"{model_name} | {track_name.upper()}")

            safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
            pred_path = OUT_DIR / f"{safe_name}_{track_name}_predictions.csv"
            pd.DataFrame(pred_rows).to_csv(pred_path, index=False)

            summary["results"][model_name][track_name] = {
                "metrics": metrics,
                "tvb": compute_threat_vs_benign(np.array(metrics["confusion_matrix"])),
                "predictions_csv": str(pred_path),
            }

            row[track_name] = metric_cell(metrics)

        report_rows.append(row)

    summary_path = OUT_DIR / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = OUT_DIR / "comparison_report.md"
    save_report(report_rows, report_path)

    print(f"\n[INFO] Saved summary JSON: {summary_path}")
    print(f"[INFO] Saved report: {report_path}")
    print(f"[INFO] Outputs saved in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()