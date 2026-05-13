#!/usr/bin/env python3
"""
scripts/component_attribution.py

Branch-wise attribution for the trained TGNN using inference-time ablations. -- slide 10 

Components:
    1. semantic   -> zero api.x[:, :768]
    2. structural -> remove non-temporal structure while preserving
                     a minimal runnable api->temporal->api backbone
    3. temporal   -> zero TemporalLSTM output via hook

Important:
- This follows the SAME preprocessing path as evaluate_zeroday.py
- This follows the SAME forward signature as the real TGNN
- Attribution is computed on the predicted class confidence
- It preserves the model's required API temporal backbone so inference remains valid
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

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
    2: "Other Malwares",
}


# ============================================================================
# Metadata helpers
# ============================================================================

def get_true_label(row: dict) -> int:
    all_cats = str(row.get("all_categories", "")).strip().lower()
    if "ransomware" in all_cats:
        return 0
    elif "benign" in all_cats:
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

    This is necessary because the trained TGNN expects API node states to be
    widened before entering temporal_lstm. Without a valid temporal/API update
    path, API features can remain at 128 instead of the expected 768.
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

    # Create zero temporal edge_attr with expected dim
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

    # Critical runtime fix
    graph = ensure_api_temporal_backbone(graph)

    # keep batching consistent
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
        batch_dict=batch_dict,
        schema=schema,
        require_api=True,
        require_api_indices_if_memory=(api_indices is not None),
        api_indices=api_indices,
    )


# ============================================================================
# Ablations
# ============================================================================

def zero_bert_dims(graph):
    g2 = copy.deepcopy(graph)
    if "api" in g2.node_types and hasattr(g2["api"], "x") and g2["api"].x is not None:
        g2["api"].x[:, :BERT_DIM] = 0.0
    return g2


def remove_all_edges_except_minimal_temporal(graph):
    """
    Structural ablation that remains runnable for this TGNN.

    We remove all heterogeneous structure, but restore a minimal
    api->temporal->api backbone so the model can still produce valid
    API hidden states before temporal_lstm.
    """
    g2 = copy.deepcopy(graph)

    for et in list(g2.edge_types):
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

    g2 = ensure_api_temporal_backbone(g2)
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
def get_confidence(model, graph, cls, schema, vocab, eval_epoch):
    probs = get_probs(model, graph, schema, vocab, eval_epoch)
    return float(probs[cls].item()), probs


@torch.no_grad()
def get_temporal_ablation_confidence(model, graph, cls, schema, vocab, eval_epoch):
    def _zero_lstm_output(module, inputs, output):
        if isinstance(output, torch.Tensor):
            return torch.zeros_like(output)
        return output

    handle = model.temporal_lstm.register_forward_hook(_zero_lstm_output)
    try:
        conf, probs = get_confidence(model, graph, cls, schema, vocab, eval_epoch)
    finally:
        handle.remove()

    return conf, probs


# ============================================================================
# Core attribution
# ============================================================================

def compute_component_attribution(model, graph, schema, vocab, eval_epoch) -> dict:
    graph = ensure_api_temporal_backbone(graph)

    probs = get_probs(
        model=model,
        graph=graph,
        schema=schema,
        vocab=vocab,
        eval_epoch=eval_epoch,
    )

    pred_class = int(probs.argmax().item())
    conf_base = float(probs[pred_class].item())

    graph_no_sem = zero_bert_dims(graph)
    graph_no_sem = ensure_api_temporal_backbone(graph_no_sem)
    conf_no_sem, _ = get_confidence(
        model=model,
        graph=graph_no_sem,
        cls=pred_class,
        schema=schema,
        vocab=vocab,
        eval_epoch=eval_epoch,
    )
    delta_sem = abs(conf_base - conf_no_sem)

    graph_no_str = remove_all_edges_except_minimal_temporal(graph)
    conf_no_str, _ = get_confidence(
        model=model,
        graph=graph_no_str,
        cls=pred_class,
        schema=schema,
        vocab=vocab,
        eval_epoch=eval_epoch,
    )
    delta_str = abs(conf_base - conf_no_str)

    conf_no_tmp, _ = get_temporal_ablation_confidence(
        model=model,
        graph=graph,
        cls=pred_class,
        schema=schema,
        vocab=vocab,
        eval_epoch=eval_epoch,
    )
    delta_tmp = abs(conf_base - conf_no_tmp)

    total = delta_sem + delta_str + delta_tmp + EPSILON
    attr_sem = delta_sem / total
    attr_str = delta_str / total
    attr_tmp = delta_tmp / total

    dominant = ["semantic", "structural", "temporal"][
        int(np.argmax([delta_sem, delta_str, delta_tmp]))
    ]

    return {
        "pred_class": pred_class,
        "pred_name": CLASS_NAMES[pred_class],
        "confidence": round(conf_base, 4),
        "prob_ransom": round(float(probs[0].item()), 4),
        "prob_benign": round(float(probs[1].item()), 4),
        "prob_malware": round(float(probs[2].item()), 4),
        "conf_base": round(conf_base, 4),
        "conf_no_semantic": round(conf_no_sem, 4),
        "conf_no_structural": round(conf_no_str, 4),
        "conf_no_temporal": round(conf_no_tmp, 4),
        "delta_semantic": round(delta_sem, 4),
        "delta_structural": round(delta_str, 4),
        "delta_temporal": round(delta_tmp, 4),
        "attr_semantic_pct": round(attr_sem, 4),
        "attr_structural_pct": round(attr_str, 4),
        "attr_temporal_pct": round(attr_tmp, 4),
        "dominant_factor": dominant,
    }


# ============================================================================
# Per-sample analysis
# ============================================================================

def analyse_single_sample(
    model,
    pkl_path: Path,
    true_label: int,
    schema: Dict,
    vocab,
    graph_processor,
    eval_epoch: int,
) -> Optional[dict]:
    md5 = pkl_path.stem.lower()

    try:
        graph = load_graph(pkl_path)
        graph = preprocess_graph_for_attribution(
            graph,
            schema=schema,
            graph_processor=graph_processor,
        )
    except Exception as exc:
        print(f"[WARN] Failed preprocessing {pkl_path.name}: {exc}")
        return None

    record = {
        "md5": md5,
        "true_label": true_label,
        "true_name": CLASS_NAMES[true_label],
    }

    for nt in ("api", "file", "registry", "network", "process"):
        if nt in graph.node_types and hasattr(graph[nt], "x") and graph[nt].x is not None:
            record[f"n_{nt}_nodes"] = int(graph[nt].x.shape[0])
        else:
            record[f"n_{nt}_nodes"] = 0

    total_edges = 0
    for et in graph.edge_types:
        store = graph[et]
        if hasattr(store, "edge_index") and store.edge_index is not None:
            total_edges += int(store.edge_index.shape[1])
    record["total_edges"] = total_edges

    try:
        attr = compute_component_attribution(
            model=model,
            graph=graph,
            schema=schema,
            vocab=vocab,
            eval_epoch=eval_epoch,
        )
        record.update(attr)
        record["correct"] = int(attr["pred_class"] == true_label)
    except Exception as exc:
        record["error"] = str(exc)

    return record


# ============================================================================
# Aggregate reporting
# ============================================================================

def _print_aggregate_stats(results: List[dict], split_name: str):
    valid = [r for r in results if "error" not in r and "attr_semantic_pct" in r]
    if not valid:
        print(f"[WARN] No valid attribution results for {split_name}")
        return

    print(f"\n{'=' * 72}")
    print(f"COMPONENT ATTRIBUTION SUMMARY — {split_name.upper()}")
    print(f"{'=' * 72}")
    print(f"Total valid samples: {len(valid)}")

    for cls_id, cls_name in CLASS_NAMES.items():
        subset = [r for r in valid if r["true_label"] == cls_id]
        if not subset:
            continue

        sem = np.mean([r["attr_semantic_pct"] for r in subset])
        st = np.mean([r["attr_structural_pct"] for r in subset])
        tmp = np.mean([r["attr_temporal_pct"] for r in subset])
        dom = Counter(r["dominant_factor"] for r in subset).most_common(1)[0]

        print(
            f"{cls_name:<16} n={len(subset):>4} | "
            f"sem={sem*100:>6.1f}% | "
            f"struct={st*100:>6.1f}% | "
            f"temp={tmp*100:>6.1f}% | "
            f"top={dom[0]} ({dom[1] / len(subset) * 100:.0f}%)"
        )


def _save_results_csv(results: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    fixed_cols = [
        "md5", "split", "true_label", "true_name",
        "pred_class", "pred_name", "correct", "confidence",
        "prob_ransom", "prob_benign", "prob_malware",
        "n_api_nodes", "n_file_nodes", "n_registry_nodes",
        "n_network_nodes", "n_process_nodes", "total_edges",
        "conf_base", "conf_no_semantic", "conf_no_structural", "conf_no_temporal",
        "delta_semantic", "delta_structural", "delta_temporal",
        "attr_semantic_pct", "attr_structural_pct", "attr_temporal_pct",
        "dominant_factor", "error",
    ]

    all_keys = {k for r in results for k in r.keys()}
    fieldnames = [c for c in fixed_cols if c in all_keys] + sorted(all_keys - set(fixed_cols))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"[INFO] Saved: {path}")


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
    output_csv: Optional[Path] = None,
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
        result = analyse_single_sample(
            model=model,
            pkl_path=pkl_path,
            true_label=true_label,
            schema=schema,
            vocab=vocab,
            graph_processor=graph_processor,
            eval_epoch=eval_epoch,
        )

        if result is not None:
            result["split"] = split_dir.name
            results.append(result)

        if i % 100 == 0:
            print(f"  ... {i}/{len(pkl_files)} processed")

    _print_aggregate_stats(results, split_dir.name)

    if output_csv is not None and results:
        _save_results_csv(results, output_csv)

    return results


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TGNN component attribution via ablation")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint or config.CUSTOM_EVAL_CHECKPOINT_PATH)
    graph_root = Path(config.ACTIVE_GRAPH_DIR)
    meta_path = Path(config.METADATA_PATH)

    model, schema, vocab, graph_processor, eval_epoch = load_model_and_schema(ckpt_path)
    metadata = load_metadata(meta_path)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    out_dir = Path(args.output_dir) if args.output_dir else None

    all_results = []

    for split in splits:
        split_dir = graph_root / split
        if not split_dir.exists():
            print(f"[WARN] Split directory not found: {split_dir}")
            continue

        out_csv = None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / f"attribution_{split}.csv"

        split_results = analyse_split(
            model=model,
            split_dir=split_dir,
            metadata=metadata,
            schema=schema,
            vocab=vocab,
            graph_processor=graph_processor,
            eval_epoch=eval_epoch,
            output_csv=out_csv,
            max_samples=args.max_samples,
        )
        all_results.extend(split_results)

    if len(splits) > 1 and all_results:
        _print_aggregate_stats(all_results, "ALL_SPLITS")
        if out_dir is not None:
            _save_results_csv(all_results, out_dir / "attribution_all_splits.csv")

    print("[DONE]")


if __name__ == "__main__":
    main()