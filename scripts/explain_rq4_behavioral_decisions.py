#!/usr/bin/env python3
"""
scripts/explain_rq4_behavioral_decisions.py

RQ4: Model Interpretability

This script loads an already-trained TGNN checkpoint, samples graphs from a
chosen split, runs normal model prediction, and writes JSON reasoning reports.

Main goal:
    Explain WHY a graph was predicted as Threat/Benign using what the model
    actually sees:
        - API node feature saliency
        - semantic / structural / temporal branch gate weights
        - hybrid GAT/SAGE summary
        - temporal API chain around salient API nodes
        - graph relations touching salient API nodes
        - connected process/file/registry/network nodes when present
        - available node metadata/names if present in the .pkl graph

Example:
    python scripts/explain_rq4_behavioral_decisions.py --split test --max_samples 1
    python scripts/explain_rq4_behavioral_decisions.py --split val --max_samples 300
    python scripts/explain_rq4_behavioral_decisions.py --split train --max_samples 50 --seed 123

Outputs:
    results/rq4_explanations/
        rq4_summary_<split>.json
        predictions_<split>.csv
        reports/
            <sample_id>_rq4_report.json

Important:
    This is an interpretability aid, not a formal causal proof.
    It does not claim that hand-written keyword tags were used by the model.
    It explains the model decision from saliency + graph structure + temporal
    context + available metadata.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config
from graph_construction.pytorch_dataset import RansomwareGraphDataset, hetero_collate
from graph_construction.vocabulary import load_vocab
from models.memory_utils import get_api_indices
from models.tgnn_model import create_hetero_tgnn
from utils.schema_manager import ensure_active_schema
from utils.schema_validator import validate_batch_or_raise


EdgeType = Tuple[str, str, str]
BRANCH_NAMES = ["semantic", "structural", "temporal"]

THREAT_LABELS = {0, 2}
BENIGN_LABEL = 1


# ============================================================
# Label helpers
# ============================================================

def label_to_name(label: int) -> str:
    return config.CLASS_NAMES.get(int(label), f"Class_{label}")


def label_to_binary_name(label: int) -> str:
    return "Threat" if int(label) in THREAT_LABELS else "Benign"


def edge_type_to_str(edge_type: EdgeType) -> str:
    return f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"


def safe_json_value(v: Any) -> Any:
    """
    Convert tensors / numpy scalars / paths into JSON-safe values.
    """
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return safe_json_value(v.item())
        if v.numel() <= 20:
            return v.detach().cpu().tolist()
        return {
            "tensor_shape": list(v.shape),
            "tensor_dtype": str(v.dtype),
        }

    if isinstance(v, np.generic):
        return v.item()

    if isinstance(v, Path):
        return str(v)

    if isinstance(v, (str, int, float, bool)) or v is None:
        return v

    if isinstance(v, (list, tuple)):
        if len(v) <= 20:
            return [safe_json_value(x) for x in v]
        return {
            "list_length": len(v),
            "first_items": [safe_json_value(x) for x in list(v)[:10]],
        }

    if isinstance(v, dict):
        return {
            str(k): safe_json_value(val)
            for k, val in list(v.items())[:50]
        }

    return str(v)


# ============================================================
# Graph object introspection
# ============================================================

def get_node_x(graph, node_type: str) -> Optional[torch.Tensor]:
    try:
        return graph[node_type].x
    except Exception:
        try:
            return graph.x_dict.get(node_type, None)
        except Exception:
            return None


def get_node_store(graph, node_type: str):
    try:
        return graph[node_type]
    except Exception:
        return None


def get_edge_store(graph, edge_type: EdgeType):
    try:
        return graph[edge_type]
    except Exception:
        return None


def get_node_types(graph) -> List[str]:
    try:
        return list(graph.node_types)
    except Exception:
        try:
            return list(graph.x_dict.keys())
        except Exception:
            return []


def get_edge_types(graph) -> List[EdgeType]:
    try:
        return list(graph.edge_types)
    except Exception:
        try:
            return list(graph.edge_index_dict.keys())
        except Exception:
            return []


def extract_edge_index_dict(graph) -> Dict[EdgeType, torch.Tensor]:
    try:
        return dict(graph.edge_index_dict)
    except Exception:
        out = {}
        try:
            for et in get_edge_types(graph):
                out[et] = graph[et].edge_index
        except Exception:
            pass
        return out


def extract_edge_attr_dict(graph) -> Dict[EdgeType, torch.Tensor]:
    try:
        return dict(graph.edge_attr_dict)
    except Exception:
        out = {}
        try:
            for et in get_edge_types(graph):
                if hasattr(graph[et], "edge_attr"):
                    out[et] = graph[et].edge_attr
        except Exception:
            pass
        return out


def list_public_attrs(obj: Any) -> Dict[str, Any]:
    """
    Return a shallow summary of public attributes/keys for graph/node/edge stores.
    This is used to discover available metadata from .pkl files.
    """
    out = {}

    if obj is None:
        return out

    # PyG stores often support keys()
    try:
        for k in obj.keys():
            try:
                out[str(k)] = safe_json_value(obj[k])
            except Exception:
                try:
                    out[str(k)] = safe_json_value(getattr(obj, k))
                except Exception:
                    out[str(k)] = "<unreadable>"
    except Exception:
        pass

    # Also inspect __dict__
    try:
        for k, v in vars(obj).items():
            if str(k).startswith("_"):
                continue
            if str(k) in out:
                continue
            out[str(k)] = safe_json_value(v)
    except Exception:
        pass

    return out


def metadata_candidates_for_node_type(node_type: str) -> List[str]:
    base = [
        "names", "name", "labels", "label", "ids", "id", "node_ids",
        "raw_names", "raw_name", "objects", "object_names", "paths", "path",
        "values", "value", "keys", "key", "addresses", "address",
    ]

    if node_type == "api":
        return [
            "api_names", "api_name", "api_calls", "api_sequence", "raw_api_names",
            "names", "name", "calls", "call_names", "function_names",
            "syscall_names", "winapi_names",
        ] + base

    if node_type == "file":
        return [
            "file_names", "file_paths", "paths", "path", "filenames", "names",
            "objects", "object_names",
        ] + base

    if node_type == "registry":
        return [
            "registry_keys", "registry_paths", "reg_keys", "keys", "paths",
            "names", "values",
        ] + base

    if node_type == "process":
        return [
            "process_names", "process_ids", "pids", "image_names", "cmdlines",
            "command_lines", "names", "ids",
        ] + base

    if node_type == "network":
        return [
            "domains", "ips", "ip_addresses", "addresses", "ports",
            "urls", "hosts", "names",
        ] + base

    return base


def coerce_sequence_metadata(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return [safe_json_value(value)]
        if value.numel() <= 100000:
            return value.detach().cpu().tolist()
        return None

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.Series):
        return value.tolist()

    if isinstance(value, (list, tuple)):
        return list(value)

    # string is a single metadata value, not per-node sequence
    if isinstance(value, str):
        return [value]

    return None


def get_node_metadata_sequence(graph, node_type: str, expected_len: Optional[int] = None) -> Tuple[List[str], Dict[str, Any]]:
    """
    Tries hard to find human-readable node names/metadata in the graph.

    If unavailable, returns node IDs like api_0, file_1, etc.
    Also returns metadata about where names came from.
    """
    store = get_node_store(graph, node_type)
    candidates = metadata_candidates_for_node_type(node_type)

    search_locations = []

    # Node store attributes
    if store is not None:
        search_locations.append(("node_store", store))

    # Graph-level attributes
    search_locations.append(("graph", graph))

    # Some graphs store metadata in dictionaries
    for attr in ["metadata", "node_metadata", "raw_metadata", "aux", "node_aux", "extras"]:
        try:
            val = getattr(graph, attr)
            search_locations.append((f"graph.{attr}", val))
        except Exception:
            pass

    for loc_name, obj in search_locations:
        # dictionary-style
        if isinstance(obj, dict):
            # direct node type dict
            possible_objs = [obj]
            if node_type in obj and isinstance(obj[node_type], dict):
                possible_objs.append(obj[node_type])

            for possible in possible_objs:
                for key in candidates:
                    if key in possible:
                        seq = coerce_sequence_metadata(possible.get(key))
                        if seq is not None:
                            if expected_len is None or len(seq) == expected_len:
                                return [str(x) for x in seq], {
                                    "source": f"{loc_name}.{key}",
                                    "available": True,
                                }

        # attribute-style
        for key in candidates:
            try:
                val = getattr(obj, key)
                seq = coerce_sequence_metadata(val)
                if seq is not None:
                    if expected_len is None or len(seq) == expected_len:
                        return [str(x) for x in seq], {
                            "source": f"{loc_name}.{key}",
                            "available": True,
                        }
            except Exception:
                pass

            # PyG store can be dict-like
            try:
                val = obj[key]
                seq = coerce_sequence_metadata(val)
                if seq is not None:
                    if expected_len is None or len(seq) == expected_len:
                        return [str(x) for x in seq], {
                            "source": f"{loc_name}[{key}]",
                            "available": True,
                        }
            except Exception:
                pass

    n = int(expected_len or 0)
    if n <= 0:
        x = get_node_x(graph, node_type)
        if isinstance(x, torch.Tensor):
            n = int(x.size(0))

    fallback = [f"{node_type}_{i}" for i in range(n)]
    return fallback, {
        "source": "fallback_node_id",
        "available": False,
        "note": (
            f"No per-node metadata/name sequence found for node type '{node_type}'. "
            "Using stable node IDs from graph indices."
        ),
    }


def build_node_name_maps(graph) -> Dict[str, Dict[str, Any]]:
    maps = {}
    for nt in get_node_types(graph):
        x = get_node_x(graph, nt)
        n = int(x.size(0)) if isinstance(x, torch.Tensor) and x.dim() >= 1 else 0
        names, source_info = get_node_metadata_sequence(graph, nt, expected_len=n)
        maps[nt] = {
            "names": names,
            "source": source_info,
        }
    return maps


def get_api_timestamps_from_graph(graph) -> Optional[torch.Tensor]:
    candidates = []

    getters = [
        lambda: graph["api"].timestamps,
        lambda: graph["api"].timestamp,
        lambda: graph.api_timestamps,
        lambda: graph.timestamps,
    ]

    for getter in getters:
        try:
            candidates.append(getter())
        except Exception:
            pass

    for c in candidates:
        if c is None:
            continue
        if isinstance(c, torch.Tensor):
            return c.detach().float().view(-1)
        try:
            return torch.tensor(c, dtype=torch.float32).view(-1)
        except Exception:
            continue

    return None


def get_api_timestamps_from_batch(batch_data) -> Optional[torch.Tensor]:
    try:
        ts = batch_data["api"].timestamps
        if isinstance(ts, torch.Tensor):
            return ts.detach().float().view(-1)
    except Exception:
        pass

    try:
        graph_timestamps = object.__getattribute__(batch_data, "graph_timestamps")
        if graph_timestamps is not None and "api" in graph_timestamps:
            return graph_timestamps["api"].detach().float().view(-1)
    except Exception:
        pass

    return None


def node_ref(node_type: str, node_idx: int, node_name_maps: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    names = node_name_maps.get(node_type, {}).get("names", [])
    name = names[node_idx] if 0 <= node_idx < len(names) else f"{node_type}_{node_idx}"
    return {
        "node_type": node_type,
        "node_index": int(node_idx),
        "display_name": str(name),
        "metadata_available": bool(node_name_maps.get(node_type, {}).get("source", {}).get("available", False)),
        "metadata_source": node_name_maps.get(node_type, {}).get("source", {}).get("source"),
    }


# ============================================================
# Dataset / model helpers
# ============================================================

def load_dataset_for_split(split: str, schema):
    return RansomwareGraphDataset(
        data_dir=str(config.DATA_DIR),
        split=split,
        cache_dir=str(config.ACTIVE_GRAPH_DIR),
        schema=schema,
        validate_graphs=True,
    )


def choose_indices(dataset, max_samples: int, seed: int, sample_mode: str) -> List[int]:
    n = len(dataset)
    all_indices = list(range(n))

    if max_samples <= 0 or max_samples >= n:
        max_samples = n

    rng = random.Random(seed)

    if sample_mode == "first":
        return all_indices[:max_samples]

    if sample_mode == "random":
        rng.shuffle(all_indices)
        return all_indices[:max_samples]

    raise ValueError(f"Unknown sample_mode: {sample_mode}")


def _extract_timestamps(batch_data) -> Optional[dict]:
    try:
        graph_timestamps = object.__getattribute__(batch_data, "graph_timestamps")
        if graph_timestamps is not None:
            return graph_timestamps
    except (AttributeError, KeyError):
        pass

    ts = {}
    try:
        for nt in batch_data.x_dict.keys():
            if hasattr(batch_data[nt], "timestamps"):
                ts[nt] = batch_data[nt].timestamps
    except (KeyError, AttributeError):
        pass

    return ts or None


def _extract_api_indices(batch_data, vocab, device) -> torch.LongTensor:
    try:
        api_names = batch_data["api"].api_names
        return get_api_indices(api_names, vocab, device=device)
    except (AttributeError, KeyError):
        n = batch_data["api"].x.size(0)
        return torch.zeros(n, dtype=torch.long, device=device)


def _validate_batch_before_forward(batch_data, schema, api_indices=None):
    edge_attr_dict = batch_data.edge_attr_dict if hasattr(batch_data, "edge_attr_dict") else None
    batch_dict = batch_data.batch_dict if hasattr(batch_data, "batch_dict") else None

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


def get_branch_weights(model) -> Optional[Dict[str, float]]:
    weights = getattr(model, "_last_branch_weights", None)
    if weights is None or not isinstance(weights, torch.Tensor):
        return None
    if weights.dim() != 2 or weights.size(1) != 3 or weights.size(0) < 1:
        return None

    w = weights.detach().float().cpu()[0]
    return {
        "semantic": float(w[0].item()),
        "structural": float(w[1].item()),
        "temporal": float(w[2].item()),
        "dominant": BRANCH_NAMES[int(torch.argmax(w).item())],
    }


def get_hybrid_stats(model) -> Dict[str, Any]:
    if hasattr(model, "get_hybrid_gate_stats"):
        try:
            return model.get_hybrid_gate_stats()
        except Exception:
            return {}
    return {}


def load_model(checkpoint_path: Path, schema, device):
    model = create_hetero_tgnn(schema=schema, use_memory=config.USE_MEMORY).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing = model.load_state_dict(ckpt["model_state_dict"], strict=False)

    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(f"[INFO] Checkpoint epoch : {ckpt.get('epoch', 'unknown')}")

    if hasattr(missing, "missing_keys") and missing.missing_keys:
        print(f"[WARN] Missing keys    : {missing.missing_keys}")
    if hasattr(missing, "unexpected_keys") and missing.unexpected_keys:
        print(f"[WARN] Unexpected keys : {missing.unexpected_keys}")

    model.eval()
    return model, ckpt


# ============================================================
# Saliency
# ============================================================

def compute_gradient_x_input_saliency(
    model,
    batch_data,
    schema,
    vocab,
    device,
    target_class: Optional[int] = None,
):
    """
    Computes gradient × input saliency for API node features.

    This reflects sensitivity of the chosen class logit to the model's actual
    API node feature vectors. It does not depend on keyword heuristics.
    """
    batch_data = batch_data.to(device)

    x_dict = {}
    for nt, x in batch_data.x_dict.items():
        x_clone = x.detach().clone().to(device)
        if nt == "api":
            x_clone.requires_grad_(True)
        x_dict[nt] = x_clone

    timestamps_dict = _extract_timestamps(batch_data)
    api_indices = _extract_api_indices(batch_data, vocab, device)
    _validate_batch_before_forward(batch_data, schema, api_indices=api_indices)

    model.zero_grad(set_to_none=True)

    logits = model(
        x_dict=x_dict,
        edge_index_dict=batch_data.edge_index_dict,
        edge_attr_dict=batch_data.edge_attr_dict if hasattr(batch_data, "edge_attr_dict") else None,
        batch_dict=batch_data.batch_dict if hasattr(batch_data, "batch_dict") else None,
        timestamps_dict=timestamps_dict,
        api_indices=api_indices,
        current_epoch=int(getattr(config, "EPOCHS", 0)),
    )

    probs = torch.softmax(logits, dim=1)
    pred = int(logits.argmax(dim=1)[0].item())

    if target_class is None:
        target_class = pred

    score = logits[0, int(target_class)]
    score.backward()

    api_x = x_dict.get("api")
    if api_x is None or api_x.grad is None:
        saliency = None
    else:
        grad = api_x.grad.detach()
        saliency = (grad * api_x.detach()).abs().sum(dim=-1)
        if saliency.numel() > 0:
            saliency = saliency / (saliency.max().clamp_min(1e-8))

    return {
        "logits": logits.detach().cpu(),
        "probs": probs.detach().cpu(),
        "pred_label": pred,
        "target_class": int(target_class),
        "api_saliency": saliency.detach().cpu() if saliency is not None else None,
        "branch_weights": get_branch_weights(model),
        "hybrid_stats": get_hybrid_stats(model),
    }


def topk_indices(values: Optional[torch.Tensor], k: int) -> List[int]:
    if values is None or values.numel() == 0:
        return []
    k = min(int(k), int(values.numel()))
    return torch.topk(values, k=k).indices.cpu().tolist()


# ============================================================
# Structural and temporal explanation builders
# ============================================================

def graph_basic_stats(graph) -> Dict[str, Any]:
    stats = {
        "node_counts": {},
        "edge_counts": {},
        "api_count": 0,
        "total_nodes": 0,
        "total_edges": 0,
    }

    for nt in get_node_types(graph):
        x = get_node_x(graph, nt)
        n = int(x.size(0)) if isinstance(x, torch.Tensor) and x.dim() >= 1 else 0
        stats["node_counts"][nt] = n
        stats["total_nodes"] += n
        if nt == "api":
            stats["api_count"] = n

    edge_index_dict = extract_edge_index_dict(graph)
    for et, edge_index in edge_index_dict.items():
        e = int(edge_index.size(1)) if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 else 0
        stats["edge_counts"][edge_type_to_str(et)] = e
        stats["total_edges"] += e

    stats["edge_node_ratio"] = float(stats["total_edges"] / max(stats["total_nodes"], 1))
    return stats


def build_api_context_windows(
    node_name_maps: Dict[str, Dict[str, Any]],
    timestamps: Optional[torch.Tensor],
    saliency: Optional[torch.Tensor],
    top_indices: List[int],
    window: int = 3,
) -> List[Dict[str, Any]]:
    api_names = node_name_maps.get("api", {}).get("names", [])
    n = len(api_names)

    if timestamps is not None:
        ts = timestamps.detach().float().cpu().view(-1)
    else:
        ts = None

    contexts = []

    for idx in top_indices:
        start = max(0, idx - window)
        end = min(n, idx + window + 1)

        center_ts = float(ts[idx].item()) if ts is not None and idx < ts.numel() else None

        events = []
        for j in range(start, end):
            api_ref = node_ref("api", j, node_name_maps)
            ts_j = float(ts[j].item()) if ts is not None and j < ts.numel() else None

            prev_gap = None
            next_gap = None
            if ts is not None and j > 0 and j < ts.numel():
                prev_gap = float((ts[j] - ts[j - 1]).item())
            if ts is not None and j + 1 < ts.numel():
                next_gap = float((ts[j + 1] - ts[j]).item())

            events.append({
                **api_ref,
                "timestamp": ts_j,
                "relative_time_from_salient_api": None if center_ts is None or ts_j is None else float(ts_j - center_ts),
                "saliency": None if saliency is None or j >= saliency.numel() else float(saliency[j].item()),
                "time_gap_from_previous_api": prev_gap,
                "time_gap_to_next_api": next_gap,
                "model_feature_basis": (
                    "This API node was evaluated through its graph node feature vector. "
                    "If display_name is generic, the graph did not expose a real API name."
                ),
            })

        contexts.append({
            "salient_api": node_ref("api", idx, node_name_maps),
            "saliency": None if saliency is None or idx >= saliency.numel() else float(saliency[idx].item()),
            "timestamp": center_ts,
            "local_sequence_window": events,
            "temporal_interpretation": summarize_temporal_window(events),
        })

    return contexts


def summarize_temporal_window(events: List[Dict[str, Any]]) -> str:
    if not events:
        return "No temporal context was available."

    zero_gap = 0
    positive_gap = 0
    max_gap = None

    for ev in events:
        gap = ev.get("time_gap_to_next_api")
        if gap is None:
            continue
        if abs(float(gap)) < 1e-9:
            zero_gap += 1
        else:
            positive_gap += 1
            max_gap = float(gap) if max_gap is None else max(max_gap, float(gap))

    if zero_gap and positive_gap:
        return (
            f"The local API window contains bursty repeated/immediate transitions "
            f"({zero_gap} zero-time gaps) plus delayed transitions "
            f"(largest observed local gap={max_gap})."
        )
    if zero_gap:
        return (
            f"The local API window is burst-like, with {zero_gap} immediate/zero-time "
            "API transitions."
        )
    if positive_gap:
        return (
            f"The local API window has separated temporal transitions, with largest "
            f"observed local gap={max_gap}."
        )

    return "Timestamp gaps were not informative in this local window."


def classify_relation_behavior(edge_type: EdgeType) -> str:
    src, rel, dst = edge_type

    if src == "api" and dst == "api" and rel == "temporal":
        return "API execution/order transition"
    if src == "process" and dst == "api":
        return "process executed or invoked API behavior"
    if src == "api" and dst == "file":
        return "API interacted with file/resource nodes"
    if src == "api" and dst == "registry":
        return "API interacted with registry/resource nodes"
    if src == "api" and dst == "network":
        return "API interacted with network/resource nodes"
    if dst == "api":
        return f"{src} nodes point into salient API behavior"
    if src == "api":
        return f"salient API behavior points to {dst} nodes"
    return f"{src} to {dst} relation"


def build_salient_neighborhoods(
    graph,
    node_name_maps: Dict[str, Dict[str, Any]],
    salient_api_indices: List[int],
    max_edges_per_relation: int = 50,
) -> Dict[str, Any]:
    """
    For each salient API node, list actual relation connections touching it.

    This gives analyst-readable evidence such as:
        api_1 -> temporal -> api_2
        process_0 -> executes -> api_1
        api_5 -> operates_on -> file_0
    """
    edge_index_dict = extract_edge_index_dict(graph)
    salient_set = set(int(i) for i in salient_api_indices)

    per_api = {
        int(i): {
            "api_node": node_ref("api", int(i), node_name_maps),
            "incoming_edges": [],
            "outgoing_edges": [],
            "connected_node_type_counts": Counter(),
            "relation_counts": Counter(),
        }
        for i in salient_api_indices
    }

    relation_summaries = []
    global_relation_counts = Counter()
    connected_resources = defaultdict(Counter)

    for et, edge_index in edge_index_dict.items():
        if not isinstance(edge_index, torch.Tensor) or edge_index.dim() != 2:
            continue

        src, rel, dst = et
        edge_key = edge_type_to_str(et)
        rows = edge_index[0].detach().cpu().tolist()
        cols = edge_index[1].detach().cpu().tolist()

        examples = []
        touched = 0

        for eidx, (r, c) in enumerate(zip(rows, cols)):
            r = int(r)
            c = int(c)

            matched_api_idx = None
            direction = None

            if src == "api" and r in salient_set:
                matched_api_idx = r
                direction = "outgoing_from_salient_api"
            elif dst == "api" and c in salient_set:
                matched_api_idx = c
                direction = "incoming_to_salient_api"
            else:
                continue

            src_ref = node_ref(src, r, node_name_maps)
            dst_ref = node_ref(dst, c, node_name_maps)

            edge_record = {
                "edge_index_position": int(eidx),
                "edge_type": edge_key,
                "relation_meaning": classify_relation_behavior(et),
                "direction_relative_to_salient_api": direction,
                "source": src_ref,
                "relation": rel,
                "destination": dst_ref,
                "plain_english": build_edge_sentence(src_ref, rel, dst_ref, et),
            }

            touched += 1
            global_relation_counts[edge_key] += 1

            per_api[matched_api_idx]["relation_counts"][edge_key] += 1

            if direction == "outgoing_from_salient_api":
                per_api[matched_api_idx]["outgoing_edges"].append(edge_record)
                per_api[matched_api_idx]["connected_node_type_counts"][dst] += 1
                if dst != "api":
                    connected_resources[dst][dst_ref["display_name"]] += 1
            else:
                per_api[matched_api_idx]["incoming_edges"].append(edge_record)
                per_api[matched_api_idx]["connected_node_type_counts"][src] += 1
                if src != "api":
                    connected_resources[src][src_ref["display_name"]] += 1

            if len(examples) < max_edges_per_relation:
                examples.append(edge_record)

        if touched:
            relation_summaries.append({
                "edge_type": edge_key,
                "relation_meaning": classify_relation_behavior(et),
                "num_edges_touching_salient_api_nodes": int(touched),
                "sampled_examples": examples,
            })

    per_api_out = []
    for api_idx, info in per_api.items():
        connected_counts = {
            str(k): int(v) for k, v in info["connected_node_type_counts"].items()
        }
        rel_counts = {
            str(k): int(v) for k, v in info["relation_counts"].items()
        }

        per_api_out.append({
            "api_index": int(api_idx),
            "api_node": info["api_node"],
            "relation_counts": rel_counts,
            "connected_node_type_counts": connected_counts,
            "incoming_edges": info["incoming_edges"][:max_edges_per_relation],
            "outgoing_edges": info["outgoing_edges"][:max_edges_per_relation],
            "why_this_neighborhood_matters": explain_api_neighborhood(
                api_node=info["api_node"],
                relation_counts=rel_counts,
                connected_counts=connected_counts,
                incoming_edges=info["incoming_edges"],
                outgoing_edges=info["outgoing_edges"],
            ),
        })

    return {
        "global_relation_counts_touching_salient_api_nodes": {
            str(k): int(v) for k, v in global_relation_counts.items()
        },
        "connected_resource_summary": {
            nt: dict(counter.most_common(20))
            for nt, counter in connected_resources.items()
        },
        "relation_summaries": relation_summaries,
        "per_salient_api_neighborhoods": per_api_out,
    }


def build_edge_sentence(src_ref: Dict[str, Any], rel: str, dst_ref: Dict[str, Any], edge_type: EdgeType) -> str:
    src_type, _, dst_type = edge_type
    src_name = src_ref["display_name"]
    dst_name = dst_ref["display_name"]

    if src_type == "api" and dst_type == "api" and rel == "temporal":
        return f"{src_name} was followed by {dst_name} in the API temporal sequence."

    if src_type == "process" and dst_type == "api":
        return f"Process node {src_name} executed or invoked API node {dst_name}."

    if src_type == "api" and dst_type == "file":
        return f"API node {src_name} operated on file/resource node {dst_name}."

    if src_type == "api" and dst_type == "registry":
        return f"API node {src_name} modified or interacted with registry node {dst_name}."

    if src_type == "api" and dst_type == "network":
        return f"API node {src_name} connected to or interacted with network node {dst_name}."

    return f"{src_type} node {src_name} --{rel}--> {dst_type} node {dst_name}."


def explain_api_neighborhood(
    api_node: Dict[str, Any],
    relation_counts: Dict[str, int],
    connected_counts: Dict[str, int],
    incoming_edges: List[Dict[str, Any]],
    outgoing_edges: List[Dict[str, Any]],
) -> List[str]:
    lines = []
    api_name = api_node["display_name"]

    if relation_counts:
        rel_text = ", ".join([f"{k}:{v}" for k, v in relation_counts.items()])
        lines.append(f"{api_name} sits in model-salient relations: {rel_text}.")

    if connected_counts:
        conn_text = ", ".join([f"{k}:{v}" for k, v in connected_counts.items()])
        lines.append(f"{api_name} is directly connected to node types: {conn_text}.")

    if incoming_edges:
        incoming_types = Counter([e["source"]["node_type"] for e in incoming_edges])
        lines.append(
            f"Incoming context into {api_name}: "
            + ", ".join([f"{k}:{v}" for k, v in incoming_types.items()])
            + "."
        )

    if outgoing_edges:
        outgoing_types = Counter([e["destination"]["node_type"] for e in outgoing_edges])
        lines.append(
            f"Outgoing context from {api_name}: "
            + ", ".join([f"{k}:{v}" for k, v in outgoing_types.items()])
            + "."
        )

    if not lines:
        lines.append(
            f"{api_name} was salient from its feature vector, but no direct graph relations "
            "touching this API were found in the stored edge index."
        )

    return lines


def build_graph_metadata_inventory(graph, node_name_maps: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Show what metadata the .pkl actually exposes.
    This answers: did the graph contain real node names or only feature tensors?
    """
    inventory = {
        "graph_public_attrs_sample": list_public_attrs(graph),
        "node_metadata_sources": {},
        "node_store_public_attrs": {},
        "edge_store_public_attrs": {},
    }

    for nt in get_node_types(graph):
        inventory["node_metadata_sources"][nt] = node_name_maps.get(nt, {}).get("source", {})
        store = get_node_store(graph, nt)
        attrs = list_public_attrs(store)
        # Avoid dumping huge tensors.
        inventory["node_store_public_attrs"][nt] = {
            k: v for k, v in attrs.items()
            if k not in {"x"}
        }

    for et in get_edge_types(graph):
        store = get_edge_store(graph, et)
        attrs = list_public_attrs(store)
        inventory["edge_store_public_attrs"][edge_type_to_str(et)] = {
            k: v for k, v in attrs.items()
            if k not in {"edge_index", "edge_attr"}
        }

    return inventory


def build_behavioral_rationale(
    pred_label: int,
    probs: torch.Tensor,
    branch_weights: Optional[Dict[str, float]],
    top_contexts: List[Dict[str, Any]],
    salient_neighborhoods: Dict[str, Any],
    graph_stats: Dict[str, Any],
) -> List[str]:
    lines = []

    pred_name = label_to_name(pred_label)
    binary_name = label_to_binary_name(pred_label)

    lines.append(
        f"The model predicted {pred_name} ({binary_name}) with probability "
        f"{float(probs[0, pred_label].item()):.4f}."
    )

    if branch_weights:
        dom = branch_weights["dominant"]
        if dom == "semantic":
            branch_meaning = (
                "the raw/projected API-node feature summary carried the strongest graph-level signal"
            )
        elif dom == "structural":
            branch_meaning = (
                "the post-GNN graph structure around API/resource/process relations carried the strongest signal"
            )
        else:
            branch_meaning = (
                "the LSTM/temporal ordering of API behavior carried the strongest signal"
            )

        lines.append(
            f"The adaptive fusion leaned most on the {dom} branch "
            f"(semantic={branch_weights['semantic']:.3f}, "
            f"structural={branch_weights['structural']:.3f}, "
            f"temporal={branch_weights['temporal']:.3f}); this means {branch_meaning}."
        )

    salient_names = []
    for ctx in top_contexts:
        api = ctx.get("salient_api", {})
        salient_names.append(api.get("display_name", "unknown"))

    if salient_names:
        lines.append(
            "The highest-saliency API nodes were: "
            + ", ".join(salient_names[:10])
            + ". These are the API nodes whose feature vectors most influenced the selected class logit."
        )

    global_rel = salient_neighborhoods.get("global_relation_counts_touching_salient_api_nodes", {})
    if global_rel:
        top_rel = ", ".join([f"{k}:{v}" for k, v in Counter(global_rel).most_common(8)])
        lines.append(
            "Those salient API nodes were embedded in these graph relations: "
            f"{top_rel}."
        )

    connected_resources = salient_neighborhoods.get("connected_resource_summary", {})
    if connected_resources:
        resource_parts = []
        for nt, items in connected_resources.items():
            names = list(items.keys())[:5]
            resource_parts.append(f"{nt} nodes={names}")
        lines.append(
            "The salient API regions touched external graph objects/resources: "
            + "; ".join(resource_parts)
            + "."
        )

    edge_counts = graph_stats.get("edge_counts", {})
    if edge_counts:
        nonzero_edges = {k: v for k, v in edge_counts.items() if v}
        if nonzero_edges:
            top_edges = ", ".join([f"{k}:{v}" for k, v in Counter(nonzero_edges).most_common(8)])
            lines.append(
                "At whole-graph level, the dominant relation volumes were: "
                f"{top_edges}."
            )

    if pred_label in THREAT_LABELS:
        lines.append(
            "Security interpretation: the model treated this graph as a threat because high-saliency "
            "API feature vectors occurred inside a structured behavioral context, especially temporal "
            "API transitions and process/API execution links. If file, registry, or network edges are "
            "present around salient APIs, those resource interactions strengthen the behavioral alert."
        )
    else:
        lines.append(
            "Security interpretation: the model treated this graph as benign because the salient API "
            "feature vectors and their local graph/temporal context did not cross the learned threat boundary."
        )

    return lines


# ============================================================
# Report builder
# ============================================================

def explain_one_graph(
    graph,
    label: int,
    sample_id: str,
    model,
    schema,
    vocab,
    device,
    top_k: int,
    window: int,
    include_metadata_inventory: bool,
) -> Dict[str, Any]:
    batch_data, labels = hetero_collate([(graph, int(label))], schema=schema, validate_batch=False)
    if batch_data is None:
        raise RuntimeError(f"Failed to collate graph {sample_id}")

    result = compute_gradient_x_input_saliency(
        model=model,
        batch_data=batch_data,
        schema=schema,
        vocab=vocab,
        device=device,
        target_class=None,
    )

    logits = result["logits"]
    probs = result["probs"]
    pred_label = int(result["pred_label"])
    saliency = result["api_saliency"]

    node_name_maps = build_node_name_maps(graph)

    timestamps = get_api_timestamps_from_graph(graph)
    if timestamps is None:
        timestamps = get_api_timestamps_from_batch(batch_data)

    top_indices = topk_indices(saliency, top_k)

    top_contexts = build_api_context_windows(
        node_name_maps=node_name_maps,
        timestamps=timestamps,
        saliency=saliency,
        top_indices=top_indices,
        window=window,
    )

    salient_neighborhoods = build_salient_neighborhoods(
        graph=graph,
        node_name_maps=node_name_maps,
        salient_api_indices=top_indices,
    )

    stats = graph_basic_stats(graph)

    rationale = build_behavioral_rationale(
        pred_label=pred_label,
        probs=probs,
        branch_weights=result["branch_weights"],
        top_contexts=top_contexts,
        salient_neighborhoods=salient_neighborhoods,
        graph_stats=stats,
    )

    report = {
        "sample_id": sample_id,
        "true_label": int(label),
        "true_name": label_to_name(int(label)),
        "true_binary": label_to_binary_name(int(label)),
        "pred_label": pred_label,
        "pred_name": label_to_name(pred_label),
        "pred_binary": label_to_binary_name(pred_label),
        "correct_3class": bool(pred_label == int(label)),
        "correct_threat_vs_benign": bool((pred_label in THREAT_LABELS) == (int(label) in THREAT_LABELS)),
        "probabilities": {
            "ransomware": float(probs[0, 0].item()),
            "benign": float(probs[0, 1].item()),
            "other_malware": float(probs[0, 2].item()),
        },
        "logits": {
            "ransomware": float(logits[0, 0].item()),
            "benign": float(logits[0, 1].item()),
            "other_malware": float(logits[0, 2].item()),
        },
        "branch_gate_weights": result["branch_weights"],
        "hybrid_gat_sage_summary": result["hybrid_stats"],
        "graph_basic_stats": stats,
        "node_metadata_sources": {
            nt: node_name_maps[nt]["source"]
            for nt in node_name_maps
        },
        "top_salient_api_nodes": top_contexts,
        "salient_behavioral_neighborhoods": salient_neighborhoods,
        "analyst_rationale": rationale,
        "interpretability_basis": {
            "saliency_method": "gradient_x_input over API node feature vectors",
            "what_model_actually_used": [
                "API node feature vectors",
                "heterogeneous graph relations",
                "hybrid GAT/SAGE spatial message passing",
                "LSTM temporal sequence modeling",
                "adaptive semantic/structural/temporal branch fusion",
                "final classifier logits",
            ],
            "metadata_policy": (
                "If real node names/metadata exist in the .pkl graph they are displayed. "
                "If not, stable graph node IDs such as api_1 or process_0 are used."
            ),
            "not_used_as_model_input": (
                "No hand-written keyword suspiciousness labels are used to create the prediction."
            ),
        },
        "interpretation_note": (
            "This explanation combines model saliency, graph connectivity, temporal context, "
            "adaptive branch weights, and available graph metadata. It is an analyst aid, not a formal causal proof."
        ),
    }

    if include_metadata_inventory:
        report["pkl_metadata_inventory"] = build_graph_metadata_inventory(graph, node_name_maps)

    return report


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate RQ4 interpretability reports for TGNN decisions.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=getattr(config, "RANDOM_SEED", 42))
    parser.add_argument("--sample_mode", choices=["random", "first"], default="random")
    parser.add_argument("--top_k", type=int, default=10, help="Top salient API nodes per graph.")
    parser.add_argument("--window", type=int, default=3, help="Temporal context window around salient API nodes.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(config.CHECKPOINTS_DIR / "best_model.pt"),
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(config.RESULTS_DIR / "rq4_explanations"),
        help="Output directory.",
    )
    parser.add_argument(
        "--include_metadata_inventory",
        action="store_true",
        help="Include a shallow scan of public graph/node/edge metadata in each report.",
    )
    args = parser.parse_args()

    config.set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(config.DEVICE)
    print(f"[INFO] Device: {device}")

    schema = ensure_active_schema()
    dataset = load_dataset_for_split(args.split, schema=schema)
    indices = choose_indices(
        dataset=dataset,
        max_samples=args.max_samples,
        seed=args.seed,
        sample_mode=args.sample_mode,
    )

    print(f"[INFO] Split       : {args.split}")
    print(f"[INFO] Dataset size: {len(dataset)}")
    print(f"[INFO] Explaining  : {len(indices)} samples")
    print(f"[INFO] Sample mode : {args.sample_mode}")

    vocab = load_vocab(config.RESULTS_DIR / "api_vocabulary.json")

    model, ckpt = load_model(
        checkpoint_path=Path(args.checkpoint),
        schema=schema,
        device=device,
    )

    out_dir = Path(args.out_dir)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for rank, idx in enumerate(indices, start=1):
        try:
            graph, label = dataset[idx]
            sample_id = getattr(graph, "sample_id", None)
            if sample_id is None:
                try:
                    sample_id = dataset.graph_files[idx].stem
                except Exception:
                    sample_id = f"{args.split}_{idx}"

            print(f"[INFO] ({rank}/{len(indices)}) Explaining sample: {sample_id}")

            report = explain_one_graph(
                graph=graph,
                label=int(label),
                sample_id=str(sample_id),
                model=model,
                schema=schema,
                vocab=vocab,
                device=device,
                top_k=args.top_k,
                window=args.window,
                include_metadata_inventory=args.include_metadata_inventory,
            )

            report_path = reports_dir / f"{sample_id}_rq4_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            probs = report["probabilities"]
            branch = report.get("branch_gate_weights") or {}

            summary_rows.append({
                "sample_id": sample_id,
                "dataset_index": int(idx),
                "true_label": report["true_label"],
                "true_name": report["true_name"],
                "true_binary": report["true_binary"],
                "pred_label": report["pred_label"],
                "pred_name": report["pred_name"],
                "pred_binary": report["pred_binary"],
                "correct_3class": int(report["correct_3class"]),
                "correct_threat_vs_benign": int(report["correct_threat_vs_benign"]),
                "prob_ransomware": probs["ransomware"],
                "prob_benign": probs["benign"],
                "prob_other_malware": probs["other_malware"],
                "branch_semantic": branch.get("semantic"),
                "branch_structural": branch.get("structural"),
                "branch_temporal": branch.get("temporal"),
                "branch_dominant": branch.get("dominant"),
                "report_path": str(report_path),
            })

        except Exception as e:
            print(f"[ERROR] Failed sample index={idx}: {type(e).__name__}: {e}")
            summary_rows.append({
                "sample_id": f"{args.split}_{idx}",
                "dataset_index": int(idx),
                "error": f"{type(e).__name__}: {e}",
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / f"predictions_{args.split}.csv"
    summary_df.to_csv(summary_csv, index=False)

    if "error" in summary_df.columns:
        processed_df = summary_df[summary_df["error"].isna()]
    else:
        processed_df = summary_df

    if len(processed_df) > 0:
        rq4_summary = {
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": ckpt.get("epoch"),
            "num_requested": int(len(indices)),
            "num_processed": int(len(processed_df)),
            "num_errors": int(len(summary_df) - len(processed_df)),
            "accuracy_3class": float(processed_df["correct_3class"].mean()),
            "accuracy_threat_vs_benign": float(processed_df["correct_threat_vs_benign"].mean()),
            "predicted_class_counts": {
                str(k): int(v) for k, v in processed_df["pred_name"].value_counts().to_dict().items()
            },
            "predicted_binary_counts": {
                str(k): int(v) for k, v in processed_df["pred_binary"].value_counts().to_dict().items()
            },
            "branch_dominant_counts": {
                str(k): int(v) for k, v in processed_df["branch_dominant"].value_counts(dropna=True).to_dict().items()
            },
            "mean_branch_weights": {
                "semantic": None if processed_df["branch_semantic"].dropna().empty else float(processed_df["branch_semantic"].mean()),
                "structural": None if processed_df["branch_structural"].dropna().empty else float(processed_df["branch_structural"].mean()),
                "temporal": None if processed_df["branch_temporal"].dropna().empty else float(processed_df["branch_temporal"].mean()),
            },
            "reports_dir": str(reports_dir),
            "summary_csv": str(summary_csv),
        }
    else:
        rq4_summary = {
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "num_requested": int(len(indices)),
            "num_processed": 0,
            "num_errors": int(len(summary_df)),
            "reports_dir": str(reports_dir),
            "summary_csv": str(summary_csv),
        }

    summary_json = out_dir / f"rq4_summary_{args.split}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rq4_summary, f, indent=2)

    print("\n" + "=" * 72)
    print("RQ4 Explanation Summary")
    print("=" * 72)
    print(f"Split                    : {args.split}")
    print(f"Requested samples        : {len(indices)}")
    print(f"Processed samples        : {rq4_summary['num_processed']}")
    print(f"Errors                   : {rq4_summary['num_errors']}")
    if rq4_summary["num_processed"] > 0:
        print(f"3-class accuracy         : {rq4_summary['accuracy_3class']:.4f}")
        print(f"Threat-vs-benign accuracy: {rq4_summary['accuracy_threat_vs_benign']:.4f}")
        print(f"Predicted classes        : {rq4_summary['predicted_class_counts']}")
        print(f"Predicted binary         : {rq4_summary['predicted_binary_counts']}")
        print(f"Dominant branches        : {rq4_summary['branch_dominant_counts']}")
        print(f"Mean branch weights      : {rq4_summary['mean_branch_weights']}")
    print(f"Reports directory        : {reports_dir}")
    print(f"Summary CSV              : {summary_csv}")
    print(f"Summary JSON             : {summary_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()
