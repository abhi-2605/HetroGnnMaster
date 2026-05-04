#!/usr/bin/env python3
"""
scripts/component_attribution_improved.py

Improved TGNN component sensitivity / ablation analysis.

What this improves over the old script
--------------------------------------
1) Finer-grained ablations:
   - semantic_zero_bert
   - semantic_shuffle_bert
   - semantic_noise_bert
   - remove_all_structure_keep_temporal
   - remove_edge_api_file
   - remove_edge_api_registry
   - remove_edge_api_network
   - remove_edge_process_api
   - remove_edge_api_temporal
   - temporal_zero_lstm
   - temporal_shuffle_timestamps
   - temporal_reverse_timestamps
   - temporal_collapse_timestamps
   - combined_no_semantic_no_structure
   - combined_no_semantic_no_temporal
   - combined_no_structure_no_temporal

2) Better metrics per ablation:
   - predicted-class confidence drop
   - true-class confidence drop
   - probability delta vector
   - decision flip
   - threat-vs-benign flip
   - margin drop (top1 - top2)

3) Keeps the same real preprocessing path as your evaluation path

4) Produces:
   - per-sample CSV
   - aggregate summary JSON
   - class-wise summary in terminal

Interpretation
--------------
This is still an approximate component sensitivity analysis, not a full
mechanistic proof. But it is much stronger than a single coarse branch ablation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
from graph_construction.pytorch_dataset import hetero_collate, RansomwareGraphDataset
from graph_construction.vocabulary import load_vocab
from models.memory_utils import get_api_indices
from models.tgnn_model import create_hetero_tgnn
from utils.schema_manager import ensure_active_schema
from utils.schema_utils import normalize_graph_node_features
from utils.schema_validator import validate_batch_or_raise


DEVICE = torch.device(config.DEVICE)
BERT_DIM = 768
EPSILON = 1e-9

CLASS_NAMES = {
    0: "Ransomware",
    1: "Benign",
    2: "Other Malware",
}

EDGE_TYPE_NAMES = {
    ("api", "temporal", "api"): "api_temporal_api",
    ("api", "operates_on", "file"): "api_file",
    ("api", "modifies", "registry"): "api_registry",
    ("api", "connects_to", "network"): "api_network",
    ("process", "executes", "api"): "process_api",
}

ABLATIONS = [
    "semantic_zero_bert",
    "semantic_shuffle_bert",
    "semantic_noise_bert",
    "remove_all_structure_keep_temporal",
    "remove_edge_api_file",
    "remove_edge_api_registry",
    "remove_edge_api_network",
    "remove_edge_process_api",
    "remove_edge_api_temporal",
    "temporal_zero_lstm",
    "temporal_shuffle_timestamps",
    "temporal_reverse_timestamps",
    "temporal_collapse_timestamps",
    "combined_no_semantic_no_structure",
    "combined_no_semantic_no_temporal",
    "combined_no_structure_no_temporal",
]


# ============================================================================
# Metadata helpers
# ============================================================================

def get_true_label(row: dict) -> int:
    all_cats = str(row.get("all_categories", "")).strip().lower()
    if "ransomware" in all_cats:
        return 0
    if "benign" in all_cats:
        return 1
    return 2


def load_metadata(csv_path: Path) -> Dict[str, dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "md5" not in df.columns:
        raise KeyError("Expected 'md5' column in metadata CSV.")
    if "all_categories" not in df.columns:
        raise KeyError("Expected 'all_categories' column in metadata CSV.")

    df["md5"] = df["md5"].astype(str).str.strip().str.lower()
    lookup = {row["md5"]: row.to_dict() for _, row in df.iterrows()}
    print(f"[INFO] Loaded {len(lookup):,} metadata rows from {csv_path}")
    return lookup


# ============================================================================
# Model / schema / processor setup
# ============================================================================

def load_model_and_schema(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    schema_obj = ensure_active_schema()
    if isinstance(schema_obj, dict):
        active_schema = schema_obj
    else:
        with open(config.ACTIVE_SCHEMA_PATH, "r", encoding="utf-8") as f:
            active_schema = json.load(f)

    model = create_hetero_tgnn(
        schema=active_schema,
        use_memory=config.USE_MEMORY,
    ).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    eval_epoch = int(ckpt.get("epoch", config.EPOCHS))
    vocab = load_vocab(config.RESULTS_DIR / "api_vocabulary.json")

    graph_processor = RansomwareGraphDataset.__new__(RansomwareGraphDataset)
    graph_processor.schema = active_schema

    print(f"[INFO] Loaded checkpoint : {checkpoint_path}")
    print(f"[INFO] Running on device : {DEVICE}")
    print(f"[INFO] Checkpoint epoch  : {eval_epoch}")

    return model, active_schema, vocab, graph_processor, eval_epoch


# ============================================================================
# Graph loading / preprocessing
# ============================================================================

def load_graph(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _num_edges(graph, et) -> int:
    if et not in graph.edge_types:
        return 0
    store = graph[et]
    if not hasattr(store, "edge_index") or store.edge_index is None:
        return 0
    if store.edge_index.dim() != 2 or store.edge_index.size(0) != 2:
        return 0
    return int(store.edge_index.size(1))


def ensure_api_temporal_backbone(graph):
    """
    Guarantee a usable api->temporal->api backbone exists.
    """
    g = copy.deepcopy(graph)

    if "api" not in g.node_types:
        return g
    if not hasattr(g["api"], "x") or g["api"].x is None:
        return g

    num_api = int(g["api"].x.size(0))
    if num_api <= 0:
        return g

    temporal_et = ("api", "temporal", "api")

    if _num_edges(g, temporal_et) > 0:
        return g

    _ = g[temporal_et]
    store = g[temporal_et]
    device = g["api"].x.device

    if num_api == 1:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
    else:
        src = torch.arange(num_api - 1, dtype=torch.long, device=device)
        dst = torch.arange(1, num_api, dtype=torch.long, device=device)
        edge_index = torch.stack([src, dst], dim=0)

    store.edge_index = edge_index
    store.edge_attr = torch.zeros(
        (edge_index.size(1), int(config.TEMPORAL_EDGE_DIM)),
        dtype=torch.float32,
        device=device,
    )
    return g


def preprocess_graph_for_attribution(graph, schema: Dict, graph_processor):
    graph = graph_processor._ensure_api_timestamps(graph)

    if config.NORMALIZE_NODE_FEATURE_DIMS:
        graph = normalize_graph_node_features(graph, schema=schema, inplace=True)

    graph = graph_processor._compress_consecutive_api_runs(graph)
    graph = graph_processor._behavior_aware_api_cap(graph)
    graph = graph_processor._ensure_api_aux_tensors(graph)

    graph = ensure_api_temporal_backbone(graph)

    for node_type in graph.node_types:
        if node_type != "api" and hasattr(graph[node_type], "timestamps"):
            try:
                del graph[node_type]["timestamps"]
            except Exception:
                try:
                    delattr(graph[node_type], "timestamps")
                except Exception:
                    pass

    return graph


# ============================================================================
# Batch conversion
# ============================================================================

def graph_to_single_batch(graph, schema: Dict):
    collate = partial(
        hetero_collate,
        schema=schema,
        validate_batch=False,
    )
    batch_data, labels = collate([(graph, 0)])
    if batch_data is None:
        raise RuntimeError("Failed to collate graph into batch.")
    return batch_data.to(DEVICE)


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


def _extract_api_indices(batch_data, vocab, device):
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
        batch_dict=batch_data.batch_dict if hasattr(batch_data, "batch_dict") else None,
        schema=schema,
        require_api=True,
        require_api_indices_if_memory=(api_indices is not None),
        api_indices=api_indices,
    )


# ============================================================================
# Graph mutation helpers
# ============================================================================

def zero_bert_dims(graph):
    g2 = copy.deepcopy(graph)
    if "api" in g2.node_types and hasattr(g2["api"], "x") and g2["api"].x is not None:
        if g2["api"].x.size(1) >= BERT_DIM:
            g2["api"].x[:, :BERT_DIM] = 0.0
    return g2


def shuffle_bert_dims(graph):
    g2 = copy.deepcopy(graph)
    if "api" in g2.node_types and hasattr(g2["api"], "x") and g2["api"].x is not None:
        x = g2["api"].x
        if x.size(0) > 1 and x.size(1) >= BERT_DIM:
            perm = torch.randperm(x.size(0), device=x.device)
            x[:, :BERT_DIM] = x[perm, :BERT_DIM]
            g2["api"].x = x
    return g2


def noise_bert_dims(graph, sigma: float = 0.25):
    g2 = copy.deepcopy(graph)
    if "api" in g2.node_types and hasattr(g2["api"], "x") and g2["api"].x is not None:
        x = g2["api"].x
        if x.size(1) >= BERT_DIM:
            x[:, :BERT_DIM] = x[:, :BERT_DIM] + sigma * torch.randn_like(x[:, :BERT_DIM])
            g2["api"].x = x
    return g2


def remove_edge_type(graph, edge_type: Tuple[str, str, str], keep_temporal_backbone: bool = True):
    g2 = copy.deepcopy(graph)

    if edge_type in g2.edge_types:
        store = g2[edge_type]
        if hasattr(store, "edge_index") and store.edge_index is not None:
            device = store.edge_index.device
            store.edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        if hasattr(store, "edge_attr") and store.edge_attr is not None:
            ea = store.edge_attr
            if ea.dim() == 2:
                store.edge_attr = torch.zeros((0, ea.size(1)), dtype=ea.dtype, device=ea.device)
            else:
                store.edge_attr = torch.zeros((0,), dtype=ea.dtype, device=ea.device)

    if keep_temporal_backbone:
        g2 = ensure_api_temporal_backbone(g2)
    return g2


def remove_all_structure_keep_temporal(graph):
    g2 = copy.deepcopy(graph)
    for et in list(g2.edge_types):
        if et == ("api", "temporal", "api"):
            continue
        store = g2[et]
        if hasattr(store, "edge_index") and store.edge_index is not None:
            device = store.edge_index.device
            store.edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        if hasattr(store, "edge_attr") and store.edge_attr is not None:
            ea = store.edge_attr
            if ea.dim() == 2:
                store.edge_attr = torch.zeros((0, ea.size(1)), dtype=ea.dtype, device=ea.device)
            else:
                store.edge_attr = torch.zeros((0,), dtype=ea.dtype, device=ea.device)
    return ensure_api_temporal_backbone(g2)


def shuffle_api_timestamps(graph):
    g2 = copy.deepcopy(graph)
    if "api" not in g2.node_types:
        return g2
    if not hasattr(g2["api"], "timestamps") or g2["api"].timestamps is None:
        return g2
    ts = g2["api"].timestamps
    if ts.numel() > 1:
        perm = torch.randperm(ts.numel(), device=ts.device)
        g2["api"].timestamps = ts[perm]
    return g2


def reverse_api_timestamps(graph):
    g2 = copy.deepcopy(graph)
    if "api" not in g2.node_types:
        return g2
    if not hasattr(g2["api"], "timestamps") or g2["api"].timestamps is None:
        return g2
    ts = g2["api"].timestamps
    if ts.numel() > 1:
        sorted_ts, _ = torch.sort(ts)
        rev = torch.flip(sorted_ts, dims=[0])
        g2["api"].timestamps = rev
    return g2


def collapse_api_timestamps(graph):
    g2 = copy.deepcopy(graph)
    if "api" not in g2.node_types:
        return g2
    if not hasattr(g2["api"], "timestamps") or g2["api"].timestamps is None:
        return g2
    ts = g2["api"].timestamps
    if ts.numel() > 0:
        g2["api"].timestamps = torch.zeros_like(ts)
    return g2


# ============================================================================
# Forward helpers
# ============================================================================

@torch.no_grad()
def forward_probs(model, batch_data, vocab, schema, eval_epoch):
    api_indices = _extract_api_indices(batch_data, vocab, DEVICE)
    _validate_batch_before_forward(batch_data, schema, api_indices=api_indices)

    logits = model(
        x_dict=batch_data.x_dict,
        edge_index_dict=batch_data.edge_index_dict,
        edge_attr_dict=batch_data.edge_attr_dict if hasattr(batch_data, "edge_attr_dict") else None,
        batch_dict=batch_data.batch_dict if hasattr(batch_data, "batch_dict") else None,
        timestamps_dict=_extract_timestamps(batch_data),
        api_indices=api_indices,
        current_epoch=eval_epoch,
    )

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    probs = F.softmax(logits[0], dim=-1)
    return probs


@torch.no_grad()
def get_probs(model, graph, schema, vocab, eval_epoch):
    batch_data = graph_to_single_batch(graph, schema)
    probs = forward_probs(model, batch_data, vocab, schema, eval_epoch)
    return probs


@torch.no_grad()
def get_probs_temporal_zero_hook(model, graph, schema, vocab, eval_epoch):
    def _zero_lstm_output(module, inputs, output):
        if isinstance(output, torch.Tensor):
            return torch.zeros_like(output)
        return output

    handle = model.temporal_lstm.register_forward_hook(_zero_lstm_output)
    try:
        probs = get_probs(model, graph, schema, vocab, eval_epoch)
    finally:
        handle.remove()
    return probs


# ============================================================================
# Metrics helpers
# ============================================================================

def probs_to_margin(probs: torch.Tensor) -> float:
    vals, _ = torch.topk(probs, k=2)
    return float((vals[0] - vals[1]).item())


def threat_binary(label: int) -> int:
    return 0 if label == 1 else 1  # benign=0, threat=1


def build_effect_record(base_probs: torch.Tensor, ablated_probs: torch.Tensor, true_label: int):
    pred_class = int(base_probs.argmax().item())
    ablated_pred_class = int(ablated_probs.argmax().item())

    base_conf_pred = float(base_probs[pred_class].item())
    ablated_conf_pred = float(ablated_probs[pred_class].item())

    base_conf_true = float(base_probs[true_label].item())
    ablated_conf_true = float(ablated_probs[true_label].item())

    delta_pred = abs(base_conf_pred - ablated_conf_pred)
    delta_true = abs(base_conf_true - ablated_conf_true)

    base_margin = probs_to_margin(base_probs)
    ablated_margin = probs_to_margin(ablated_probs)

    return {
        "pred_class_before": pred_class,
        "pred_name_before": CLASS_NAMES[pred_class],
        "pred_class_after": ablated_pred_class,
        "pred_name_after": CLASS_NAMES[ablated_pred_class],
        "decision_flip": int(pred_class != ablated_pred_class),
        "threat_binary_flip": int(threat_binary(pred_class) != threat_binary(ablated_pred_class)),
        "base_pred_conf": round(base_conf_pred, 6),
        "ablated_pred_conf": round(ablated_conf_pred, 6),
        "pred_conf_drop_abs": round(delta_pred, 6),
        "base_true_conf": round(base_conf_true, 6),
        "ablated_true_conf": round(ablated_conf_true, 6),
        "true_conf_drop_abs": round(delta_true, 6),
        "base_margin": round(base_margin, 6),
        "ablated_margin": round(ablated_margin, 6),
        "margin_drop_abs": round(abs(base_margin - ablated_margin), 6),
        "prob_ransom_before": round(float(base_probs[0].item()), 6),
        "prob_benign_before": round(float(base_probs[1].item()), 6),
        "prob_malware_before": round(float(base_probs[2].item()), 6),
        "prob_ransom_after": round(float(ablated_probs[0].item()), 6),
        "prob_benign_after": round(float(ablated_probs[1].item()), 6),
        "prob_malware_after": round(float(ablated_probs[2].item()), 6),
        "delta_prob_ransom": round(float(abs(base_probs[0] - ablated_probs[0]).item()), 6),
        "delta_prob_benign": round(float(abs(base_probs[1] - ablated_probs[1]).item()), 6),
        "delta_prob_malware": round(float(abs(base_probs[2] - ablated_probs[2]).item()), 6),
    }


# ============================================================================
# Ablation executor
# ============================================================================

def run_named_ablation(model, graph, schema, vocab, eval_epoch, ablation_name: str):
    g = ensure_api_temporal_backbone(graph)

    if ablation_name == "semantic_zero_bert":
        return get_probs(model, zero_bert_dims(g), schema, vocab, eval_epoch)

    if ablation_name == "semantic_shuffle_bert":
        return get_probs(model, shuffle_bert_dims(g), schema, vocab, eval_epoch)

    if ablation_name == "semantic_noise_bert":
        return get_probs(model, noise_bert_dims(g), schema, vocab, eval_epoch)

    if ablation_name == "remove_all_structure_keep_temporal":
        return get_probs(model, remove_all_structure_keep_temporal(g), schema, vocab, eval_epoch)

    if ablation_name == "remove_edge_api_file":
        return get_probs(model, remove_edge_type(g, ("api", "operates_on", "file")), schema, vocab, eval_epoch)

    if ablation_name == "remove_edge_api_registry":
        return get_probs(model, remove_edge_type(g, ("api", "modifies", "registry")), schema, vocab, eval_epoch)

    if ablation_name == "remove_edge_api_network":
        return get_probs(model, remove_edge_type(g, ("api", "connects_to", "network")), schema, vocab, eval_epoch)

    if ablation_name == "remove_edge_process_api":
        return get_probs(model, remove_edge_type(g, ("process", "executes", "api")), schema, vocab, eval_epoch)

    if ablation_name == "remove_edge_api_temporal":
        # remove temporal edges, then restore minimal backbone so forward remains runnable
        return get_probs(model, remove_edge_type(g, ("api", "temporal", "api"), keep_temporal_backbone=True), schema, vocab, eval_epoch)

    if ablation_name == "temporal_zero_lstm":
        return get_probs_temporal_zero_hook(model, g, schema, vocab, eval_epoch)

    if ablation_name == "temporal_shuffle_timestamps":
        return get_probs(model, shuffle_api_timestamps(g), schema, vocab, eval_epoch)

    if ablation_name == "temporal_reverse_timestamps":
        return get_probs(model, reverse_api_timestamps(g), schema, vocab, eval_epoch)

    if ablation_name == "temporal_collapse_timestamps":
        return get_probs(model, collapse_api_timestamps(g), schema, vocab, eval_epoch)

    if ablation_name == "combined_no_semantic_no_structure":
        g2 = remove_all_structure_keep_temporal(zero_bert_dims(g))
        return get_probs(model, g2, schema, vocab, eval_epoch)

    if ablation_name == "combined_no_semantic_no_temporal":
        g2 = zero_bert_dims(g)
        return get_probs_temporal_zero_hook(model, g2, schema, vocab, eval_epoch)

    if ablation_name == "combined_no_structure_no_temporal":
        g2 = remove_all_structure_keep_temporal(g)
        return get_probs_temporal_zero_hook(model, g2, schema, vocab, eval_epoch)

    raise ValueError(f"Unknown ablation: {ablation_name}")


# ============================================================================
# Core analysis
# ============================================================================

def analyse_single_sample(
    model,
    pkl_path: Path,
    true_label: int,
    schema: Dict,
    vocab,
    graph_processor,
    eval_epoch: int,
) -> List[dict]:
    md5 = pkl_path.stem.lower()

    try:
        graph = load_graph(pkl_path)
        graph = preprocess_graph_for_attribution(
            graph,
            schema=schema,
            graph_processor=graph_processor,
        )
    except Exception as exc:
        return [{
            "md5": md5,
            "true_label": true_label,
            "true_name": CLASS_NAMES[true_label],
            "error": f"preprocess_failed: {exc}",
        }]

    base_record = {
        "md5": md5,
        "true_label": true_label,
        "true_name": CLASS_NAMES[true_label],
    }

    for nt in ("api", "file", "registry", "network", "process"):
        if nt in graph.node_types and hasattr(graph[nt], "x") and graph[nt].x is not None:
            base_record[f"n_{nt}_nodes"] = int(graph[nt].x.shape[0])
        else:
            base_record[f"n_{nt}_nodes"] = 0

    total_edges = 0
    per_edge = {}
    for et in graph.edge_types:
        n_edges = _num_edges(graph, et)
        total_edges += n_edges
        per_edge[f"edges_{EDGE_TYPE_NAMES.get(et, str(et))}"] = n_edges
    base_record["total_edges"] = total_edges
    base_record.update(per_edge)

    try:
        base_probs = get_probs(model, ensure_api_temporal_backbone(graph), schema, vocab, eval_epoch)
    except Exception as exc:
        rec = dict(base_record)
        rec["error"] = f"base_forward_failed: {exc}"
        return [rec]

    rows = []
    for ablation_name in ABLATIONS:
        rec = dict(base_record)
        rec["ablation"] = ablation_name

        try:
            ablated_probs = run_named_ablation(
                model=model,
                graph=graph,
                schema=schema,
                vocab=vocab,
                eval_epoch=eval_epoch,
                ablation_name=ablation_name,
            )
            rec.update(build_effect_record(base_probs, ablated_probs, true_label))
            rec["correct_before"] = int(int(base_probs.argmax().item()) == true_label)
            rec["correct_after"] = int(int(ablated_probs.argmax().item()) == true_label)
            rec["error"] = ""
        except Exception as exc:
            rec["error"] = str(exc)

        rows.append(rec)

    return rows


# ============================================================================
# Aggregate reporting
# ============================================================================

def summarize_results(results: List[dict]) -> dict:
    valid = [r for r in results if not r.get("error")]
    summary = {
        "n_rows": len(results),
        "n_valid_rows": len(valid),
        "by_ablation": {},
    }

    grouped = defaultdict(list)
    for r in valid:
        grouped[r["ablation"]].append(r)

    for ablation, rows in grouped.items():
        summary["by_ablation"][ablation] = {
            "n": len(rows),
            "mean_pred_conf_drop": float(np.mean([r["pred_conf_drop_abs"] for r in rows])),
            "mean_true_conf_drop": float(np.mean([r["true_conf_drop_abs"] for r in rows])),
            "mean_margin_drop": float(np.mean([r["margin_drop_abs"] for r in rows])),
            "decision_flip_rate": float(np.mean([r["decision_flip"] for r in rows])),
            "threat_binary_flip_rate": float(np.mean([r["threat_binary_flip"] for r in rows])),
            "correct_after_rate": float(np.mean([r["correct_after"] for r in rows])),
        }

    return summary


def print_summary(results: List[dict], split_name: str):
    valid = [r for r in results if not r.get("error")]
    if not valid:
        print(f"[WARN] No valid results for {split_name}")
        return

    print(f"\n{'=' * 88}")
    print(f"IMPROVED COMPONENT SENSITIVITY SUMMARY — {split_name.upper()}")
    print(f"{'=' * 88}")

    grouped = defaultdict(list)
    for r in valid:
        grouped[r["ablation"]].append(r)

    for ablation in ABLATIONS:
        rows = grouped.get(ablation, [])
        if not rows:
            continue
        mean_pred_drop = np.mean([r["pred_conf_drop_abs"] for r in rows])
        mean_true_drop = np.mean([r["true_conf_drop_abs"] for r in rows])
        flip_rate = np.mean([r["decision_flip"] for r in rows]) * 100.0
        threat_flip = np.mean([r["threat_binary_flip"] for r in rows]) * 100.0
        print(
            f"{ablation:<34} "
            f"pred_drop={mean_pred_drop:>7.4f} "
            f"true_drop={mean_true_drop:>7.4f} "
            f"flip={flip_rate:>6.2f}% "
            f"threat_flip={threat_flip:>6.2f}%"
        )


def save_results_csv(results: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = sorted({k for r in results for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Saved CSV: {path}")


def save_summary_json(summary: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved JSON: {path}")


# ============================================================================
# Split analysis
# ============================================================================

def analyse_split(
    model,
    split_dir: Path,
    metadata: Dict[str, dict],
    schema: Dict,
    vocab,
    graph_processor,
    eval_epoch: int,
    output_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> List[dict]:
    pkl_files = sorted(split_dir.glob("*.pkl"))
    if max_samples is not None:
        pkl_files = pkl_files[:max_samples]

    if not pkl_files:
        print(f"[WARN] No .pkl files found in {split_dir}")
        return []

    print(f"\n[INFO] Analysing split: {split_dir.name.upper()} ({len(pkl_files):,} files)")
    results = []

    for i, pkl_path in enumerate(pkl_files, start=1):
        md5 = pkl_path.stem.lower()
        row = metadata.get(md5)
        if row is None:
            continue

        true_label = get_true_label(row)
        sample_rows = analyse_single_sample(
            model=model,
            pkl_path=pkl_path,
            true_label=true_label,
            schema=schema,
            vocab=vocab,
            graph_processor=graph_processor,
            eval_epoch=eval_epoch,
        )

        for r in sample_rows:
            r["split"] = split_dir.name
            results.append(r)

        if i % 100 == 0:
            print(f"  ... {i}/{len(pkl_files)} processed")

    print_summary(results, split_dir.name)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_results_csv(results, output_dir / f"attribution_{split_dir.name}.csv")
        save_summary_json(summarize_results(results), output_dir / f"attribution_{split_dir.name}_summary.json")

    return results


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Improved TGNN component sensitivity / ablation analysis")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default="results/component_attribution_improved")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint or config.CUSTOM_EVAL_CHECKPOINT_PATH)
    graph_root = Path(config.ACTIVE_GRAPH_DIR)
    meta_path = Path(config.METADATA_PATH)
    out_dir = Path(args.output_dir)

    model, schema, vocab, graph_processor, eval_epoch = load_model_and_schema(ckpt_path)
    metadata = load_metadata(meta_path)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    all_results = []
    for split in splits:
        split_dir = graph_root / split
        if not split_dir.exists():
            print(f"[WARN] Missing split directory: {split_dir}")
            continue

        split_results = analyse_split(
            model=model,
            split_dir=split_dir,
            metadata=metadata,
            schema=schema,
            vocab=vocab,
            graph_processor=graph_processor,
            eval_epoch=eval_epoch,
            output_dir=out_dir,
            max_samples=args.max_samples,
        )
        all_results.extend(split_results)

    if len(splits) > 1 and all_results:
        print_summary(all_results, "all_splits")
        save_results_csv(all_results, out_dir / "attribution_all_splits.csv")
        save_summary_json(summarize_results(all_results), out_dir / "attribution_all_splits_summary.json")

    print("[DONE]")


if __name__ == "__main__":
    main()