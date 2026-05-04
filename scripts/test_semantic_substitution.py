#!/usr/bin/env python3
"""
scripts/test_semantic_vector_substitution.py

Semantic-vector substitution test for TGNN (SEMANTIC MEMORY OFF setting).

Purpose
-------
Test whether the trained TGNN is relying on exact semantic embedding identities
("mugging up") or whether it is robust to semantically similar substitutions
inside api.x[:, :768].

This version is appropriate when:
    [Model] SemanticMemory OFF │ api memory disabled

Because in that case, changing api_names alone is not the correct intervention.
The correct target is the semantic block inside API node features.

What this script does
---------------------
1. Loads the trained TGNN checkpoint
2. Resolves split membership the same way as your other robust scripts
3. Builds an explicit sample list from metadata + split membership
4. Runs the SAME preprocessing path as your evaluation-style scripts
5. Builds three evaluation tracks:
      - clean
      - semantic_substitution  (replace semantic vectors with similar ones)
      - random_substitution    (replace semantic vectors with random other ones)
6. Compares prediction stability and confidence

Why this version fixes the earlier issue
----------------------------------------
The earlier version used the first N alphabetically sorted .pkl files from a split
directory. That can create a very skewed class slice and can make the "clean"
baseline look wrong. This version instead:
  - resolves train/val/test membership explicitly
  - samples records from metadata-aligned split membership
  - optionally mirrors the train malware subsample used elsewhere

Interpretation
--------------
- If semantic_substitution stays close to clean, but random_substitution drops:
    evidence against simple memorization of exact semantic vectors
- If both collapse similarly:
    evidence that exact embedding identity matters strongly
- If semantic_substitution also collapses sharply:
    evidence that the model is brittle to semantic perturbation

Notes
-----
- This changes api.x[:, :768], not api_names
- Graph structure is preserved
- Temporal structure is preserved
- This uses an explicit record list, not directory prefix slicing
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import re
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

import config
from graph_construction.pytorch_dataset import hetero_collate, RansomwareGraphDataset
from graph_construction.vocabulary import load_vocab
from models.memory_utils import get_api_indices
from models.tgnn_model import create_hetero_tgnn
from utils.schema_manager import ensure_active_schema
from utils.schema_utils import normalize_graph_node_features
from utils.schema_validator import validate_batch_or_raise, validate_graph_or_raise


DEVICE = torch.device(config.DEVICE)
RANDOM_STATE = 42
SEM_DIM = 768
EPS = 1e-12

CLASS_NAMES = {
    0: "Ransomware",
    1: "Benign",
    2: "Other Malware",
}

# Fraction of API nodes to perturb
SUBSTITUTE_FRACTION = 1.00

# Keep top-k frequent API nodes unchanged per sample if desired
KEEP_TOP_K = 0

# Similar substitution settings
SIMILARITY_TOP_K = 5
MIN_API_NODES_FOR_ATTACK = 2

# Optional train mirroring controls
APPLY_TRAIN_MALWARE_SUBSAMPLE = True
TRAIN_MALWARE_KEEP_FRAC = 0.80

# Optional exclusion file logic mirroring your other scripts
OPTIONAL_EXCLUSION_FILE = getattr(config, "BASELINE_TRAIN_EXCLUSION_FILE", None)
EXCLUDED_TRAIN_MD5S = set()


# ============================================================================
# Label / metrics helpers
# ============================================================================

def get_true_label_from_categories(all_categories: str) -> int:
    cats = str(all_categories).strip().lower()
    if "ransomware" in cats:
        return 0
    if "benign" in cats:
        return 1
    return 2


def normalize_md5_like(value: str) -> str:
    value = str(value).strip().lower()
    return value.replace(".json", "").replace(".pkl", "")


def parse_exclusion_lines(lines: List[str]) -> set[str]:
    ids: set[str] = set()
    md5_pattern = re.compile(r"\b([0-9a-f]{32})\b", re.IGNORECASE)
    for line in lines:
        for match in md5_pattern.findall(str(line).strip().lower()):
            ids.add(normalize_md5_like(match))
    return ids


def load_optional_exclusion_file(path_value) -> set[str]:
    if not path_value:
        return set()

    path = Path(path_value)
    if not path.exists():
        print(f"[WARN] Baseline exclusion file not found: {path}")
        return set()

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    ids = parse_exclusion_lines(lines)
    print(f"[INFO] Loaded {len(ids)} train exclusion IDs from {path}")
    return ids


def get_all_train_exclusions() -> set[str]:
    ids = {normalize_md5_like(x) for x in EXCLUDED_TRAIN_MD5S}
    ids |= load_optional_exclusion_file(OPTIONAL_EXCLUSION_FILE)
    return ids


def load_metadata(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "md5" not in df.columns:
        raise KeyError("Expected 'md5' column in metadata CSV.")
    if "all_categories" not in df.columns:
        raise KeyError("Expected 'all_categories' column in metadata CSV.")

    df["_md5"] = df["md5"].astype(str).str.strip().str.lower()
    df["_label"] = df["all_categories"].apply(get_true_label_from_categories)
    print(f"[INFO] Loaded metadata rows: {len(df):,}")
    return df


def compute_metrics(y_true: List[int], y_pred: List[int]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2]
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=[0, 1, 2]
    )
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0, labels=[0, 1, 2]
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(w_f1),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def compute_threat_vs_benign(confusion_mat: np.ndarray) -> dict:
    tp = confusion_mat[0, 0] + confusion_mat[0, 2] + confusion_mat[2, 0] + confusion_mat[2, 2]
    fn = confusion_mat[0, 1] + confusion_mat[2, 1]
    tn = confusion_mat[1, 1]
    fp = confusion_mat[1, 0] + confusion_mat[1, 2]
    return {
        "TP": int(tp),
        "FN": int(fn),
        "TN": int(tn),
        "FP": int(fp),
        "threats_missed": int(fn),
    }


def print_metrics(metrics: dict, title: str):
    cm = np.array(metrics["confusion_matrix"])
    tvb = compute_threat_vs_benign(cm)

    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"Macro F1      : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1   : {metrics['weighted_f1']:.4f}")
    print(f"Threats missed: {tvb['threats_missed']}")
    print(f"{'-' * 80}")
    for i in range(3):
        print(
            f"{CLASS_NAMES[i]:<16} "
            f"P={metrics['per_class_precision'][i]:.3f}  "
            f"R={metrics['per_class_recall'][i]:.3f}  "
            f"F1={metrics['per_class_f1'][i]:.3f}  "
            f"N={int(metrics['per_class_support'][i])}"
        )
    print(f"{'-' * 80}")
    print(cm)
    print(f"{'=' * 80}")


# ============================================================================
# Split helpers
# ============================================================================

def _normalize_md5_set(values) -> set[str]:
    return {str(x).strip().lower() for x in values if str(x).strip()}


def audit_split_sets(train_md5s: set[str], val_md5s: set[str], test_md5s: set[str], strict: bool = True):
    tv = train_md5s & val_md5s
    tt = train_md5s & test_md5s
    vt = val_md5s & test_md5s
    union_total = len(train_md5s | val_md5s | test_md5s)

    print("[AUDIT] Split membership")
    print(f"        train unique : {len(train_md5s)}")
    print(f"        val unique   : {len(val_md5s)}")
    print(f"        test unique  : {len(test_md5s)}")
    print(f"        train ∩ val  : {len(tv)}")
    print(f"        train ∩ test : {len(tt)}")
    print(f"        val ∩ test   : {len(vt)}")
    print(f"        union total  : {union_total}")

    if strict and (tv or tt or vt):
        raise RuntimeError(
            "Split leakage detected. train/val/test are not disjoint.\n"
            f"train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}."
        )


def load_split_sets(df: pd.DataFrame, split_root: Path):
    manifest_path = getattr(config, "SPLIT_MANIFEST_PATH", None)
    if manifest_path:
        manifest_path = Path(manifest_path)
        if manifest_path.exists():
            print(f"[INFO] Using split manifest: {manifest_path}")
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            train_md5s = _normalize_md5_set(data["train"])
            val_md5s = _normalize_md5_set(data["val"])
            test_md5s = _normalize_md5_set(data["test"])
            audit_split_sets(train_md5s, val_md5s, test_md5s, strict=True)
            return train_md5s, val_md5s, test_md5s

    candidate_cols = ["split", "partition", "set", "fold_split"]
    split_col = next((c for c in candidate_cols if c in df.columns), None)

    if split_col is not None:
        print(f"[INFO] Using metadata split column: {split_col}")
        split_vals = df[split_col].astype(str).str.strip().str.lower()

        train_aliases = {"train", "training"}
        val_aliases = {"val", "valid", "validation", "dev"}
        test_aliases = {"test", "testing"}

        train_md5s = _normalize_md5_set(df.loc[split_vals.isin(train_aliases), "_md5"])
        val_md5s = _normalize_md5_set(df.loc[split_vals.isin(val_aliases), "_md5"])
        test_md5s = _normalize_md5_set(df.loc[split_vals.isin(test_aliases), "_md5"])

        if train_md5s or val_md5s or test_md5s:
            audit_split_sets(train_md5s, val_md5s, test_md5s, strict=True)
            return train_md5s, val_md5s, test_md5s

    print("[WARN] No split manifest or metadata split column found.")
    print("[WARN] Falling back to folder-based split membership from ACTIVE_GRAPH_DIR.")

    def md5s_in_split(split: str) -> set[str]:
        split_dir = split_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        return {p.stem.lower() for p in split_dir.glob("*.pkl")}

    train_md5s = md5s_in_split("train")
    val_md5s = md5s_in_split("val")
    test_md5s = md5s_in_split("test")

    audit_split_sets(train_md5s, val_md5s, test_md5s, strict=True)
    return train_md5s, val_md5s, test_md5s


def subsample_other_malware(df_split: pd.DataFrame, keep_frac: float, split_name: str) -> pd.DataFrame:
    if not (0.0 < keep_frac <= 1.0):
        raise ValueError(f"keep_frac must be in (0,1], got {keep_frac}")

    df_split = df_split.copy()
    df_ransom = df_split[df_split["_label"] == 0]
    df_benign = df_split[df_split["_label"] == 1]
    df_malware = df_split[df_split["_label"] == 2]

    original_malware = len(df_malware)
    keep_n = int(round(original_malware * keep_frac))
    if keep_n < original_malware:
        df_malware = df_malware.sample(n=keep_n, random_state=RANDOM_STATE)

    df_out = pd.concat([df_ransom, df_benign, df_malware], axis=0)
    df_out = df_out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print(
        f"[INFO] {split_name}: kept malware {len(df_malware)}/{original_malware} "
        f"({keep_frac:.0%}), final rows={len(df_out)}"
    )
    return df_out


def build_records_for_split(
    df: pd.DataFrame,
    split_name: str,
    split_root: Path,
    max_samples: Optional[int],
    sample_seed: int = RANDOM_STATE,
) -> List[dict]:
    train_md5s, val_md5s, test_md5s = load_split_sets(df, split_root)
    split_map = {
        "train": train_md5s,
        "val": val_md5s,
        "test": test_md5s,
    }
    if split_name not in split_map:
        raise ValueError(f"Unknown split_name: {split_name}")

    selected_md5s = split_map[split_name]
    split_df = df[df["_md5"].isin(selected_md5s)].copy()

    if split_name == "train":
        excluded_ids = get_all_train_exclusions()
        if excluded_ids:
            before = len(split_df)
            split_df = split_df[~split_df["_md5"].isin(excluded_ids)].copy()
            print(f"[INFO] Removed {before - len(split_df)} train rows via exclusion IDs")

        if APPLY_TRAIN_MALWARE_SUBSAMPLE:
            split_df = subsample_other_malware(split_df, TRAIN_MALWARE_KEEP_FRAC, "TRAIN")

    split_df = split_df.sample(frac=1.0, random_state=sample_seed).reset_index(drop=True)
    if max_samples is not None:
        split_df = split_df.iloc[:max_samples].copy()

    records = []
    for _, row in split_df.iterrows():
        graph_path = split_root / split_name / f"{row['_md5']}.pkl"
        if not graph_path.exists():
            continue
        records.append({
            "sample_id": row["_md5"],
            "graph_path": str(graph_path.resolve()),
            "label": int(row["_label"]),
            "all_categories": str(row["all_categories"]),
            "split": split_name,
        })

    class_counts = Counter(r["label"] for r in records)
    print(f"[INFO] Final records for {split_name}: {len(records)}")
    print(
        f"[INFO] Class counts for {split_name}: "
        f"ransom={class_counts.get(0, 0)} benign={class_counts.get(1, 0)} malware={class_counts.get(2, 0)}"
    )
    return records


# ============================================================================
# Model / schema / processor
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

    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Eval epoch: {eval_epoch}")

    return model, active_schema, vocab, graph_processor, eval_epoch


# ============================================================================
# Graph utilities
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


def preprocess_graph(graph, schema: Dict, graph_processor):
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

    if getattr(config, "FAIL_ON_SCHEMA_MISMATCH", False):
        validate_graph_or_raise(
            graph,
            schema=schema,
            require_api_nodes=True,
            strict=getattr(config, "SCHEMA_STRICT_VALIDATION", True),
        )

    return graph


# ============================================================================
# Attack helpers
# ============================================================================

def _cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    x_norm = x / (x.norm(dim=1, keepdim=True) + EPS)
    return x_norm @ x_norm.t()


def _choose_nodes_to_replace(api_names: List[str], fraction: float, keep_top_k: int, seed: int) -> List[int]:
    freq = Counter([str(x).strip().lower() for x in api_names])
    protected = set()
    if keep_top_k > 0:
        protected = {tok for tok, _ in freq.most_common(keep_top_k)}

    candidate_indices = [i for i, tok in enumerate(api_names) if str(tok).strip().lower() not in protected]
    if not candidate_indices:
        return []

    rng = random.Random(seed)
    replace_n = int(round(len(candidate_indices) * fraction))
    replace_n = max(1, min(replace_n, len(candidate_indices)))
    return sorted(rng.sample(candidate_indices, replace_n))


def apply_semantic_vector_substitution(graph, sample_id: str) -> Tuple[object, float]:
    g = copy.deepcopy(graph)

    if "api" not in g.node_types or not hasattr(g["api"], "x") or g["api"].x is None:
        return g, 0.0

    x = g["api"].x
    n = int(x.size(0))
    if n < MIN_API_NODES_FOR_ATTACK or x.size(1) < SEM_DIM:
        return g, 0.0

    api_names = list(getattr(g["api"], "api_names", [f"api_{i}" for i in range(n)]))
    seed = (hash(sample_id) ^ RANDOM_STATE ^ 1111) & 0xFFFFFFFF
    selected = _choose_nodes_to_replace(api_names, SUBSTITUTE_FRACTION, KEEP_TOP_K, seed)
    if not selected:
        return g, 0.0

    sem = x[:, :SEM_DIM].clone()
    sim = _cosine_matrix(sem.detach())

    rng = random.Random(seed)
    changed = 0

    for idx in selected:
        sims = sim[idx].clone()
        sims[idx] = -1e9
        topk = min(SIMILARITY_TOP_K, n - 1)
        if topk <= 0:
            continue

        nn_idx = torch.topk(sims, k=topk).indices.detach().cpu().tolist()
        chosen = rng.choice(nn_idx)
        sem[idx] = sem[chosen]
        changed += 1

    g["api"].x[:, :SEM_DIM] = sem
    realized = changed / max(n, 1)
    return g, realized


def apply_random_vector_substitution(graph, sample_id: str) -> Tuple[object, float]:
    g = copy.deepcopy(graph)

    if "api" not in g.node_types or not hasattr(g["api"], "x") or g["api"].x is None:
        return g, 0.0

    x = g["api"].x
    n = int(x.size(0))
    if n < MIN_API_NODES_FOR_ATTACK or x.size(1) < SEM_DIM:
        return g, 0.0

    api_names = list(getattr(g["api"], "api_names", [f"api_{i}" for i in range(n)]))
    seed = (hash(sample_id) ^ RANDOM_STATE ^ 2222) & 0xFFFFFFFF
    selected = _choose_nodes_to_replace(api_names, SUBSTITUTE_FRACTION, KEEP_TOP_K, seed)
    if not selected:
        return g, 0.0

    sem = x[:, :SEM_DIM].clone()
    rng = random.Random(seed)
    changed = 0

    for idx in selected:
        candidates = [j for j in range(n) if j != idx]
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        sem[idx] = sem[chosen]
        changed += 1

    g["api"].x[:, :SEM_DIM] = sem
    realized = changed / max(n, 1)
    return g, realized


def apply_named_attack(graph, sample_id: str, attack_name: str):
    if attack_name == "clean":
        return graph, 0.0
    if attack_name == "semantic_substitution":
        return apply_semantic_vector_substitution(graph, sample_id)
    if attack_name == "random_substitution":
        return apply_random_vector_substitution(graph, sample_id)
    raise ValueError(f"Unknown attack_name: {attack_name}")


# ============================================================================
# Dataset / collate
# ============================================================================

class RecordGraphDataset(Dataset):
    def __init__(
        self,
        records: List[dict],
        schema: Dict,
        graph_processor,
        attack_name: str,
        validate_graphs: bool = True,
    ):
        self.records = list(records)
        self.schema = schema
        self.graph_processor = graph_processor
        self.attack_name = attack_name
        self.validate_graphs = validate_graphs

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records[idx]
        graph_path = Path(row["graph_path"])
        sample_id = str(row["sample_id"])
        true_label = int(row["label"])

        graph = load_graph(graph_path)
        graph = preprocess_graph(graph, schema=self.schema, graph_processor=self.graph_processor)
        graph, realized = apply_named_attack(graph, sample_id, self.attack_name)

        graph.sample_id = sample_id
        graph.graph_path = str(graph_path)
        graph.attack_name = self.attack_name
        graph.realized_sub_fraction = float(realized)

        return graph, true_label


def custom_collate(batch, schema, validate_batch=False):
    if not batch:
        return None, None

    sample_ids = []
    graph_paths = []
    attack_names = []
    realized_fracs = []

    for graph, _ in batch:
        sample_ids.append(getattr(graph, "sample_id", ""))
        graph_paths.append(getattr(graph, "graph_path", ""))
        attack_names.append(getattr(graph, "attack_name", ""))
        realized_fracs.append(getattr(graph, "realized_sub_fraction", 0.0))

    batched_graph, labels = hetero_collate(batch, schema=schema, validate_batch=validate_batch)
    if batched_graph is None or labels is None:
        return batched_graph, labels

    batched_graph.sample_id = sample_ids
    batched_graph.graph_path = graph_paths
    batched_graph.attack_name = attack_names
    batched_graph.realized_sub_fraction = realized_fracs
    return batched_graph, labels


# ============================================================================
# Forward helpers
# ============================================================================

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


@torch.no_grad()
def run_eval(model, loader, vocab, schema, eval_epoch: int):
    all_true = []
    all_pred = []
    rows = []

    for batch_item in loader:
        if batch_item is None or not isinstance(batch_item, (tuple, list)) or len(batch_item) != 2:
            continue

        batch_data, labels = batch_item
        if batch_data is None or labels is None:
            continue

        batch_data = batch_data.to(DEVICE)
        labels = labels.to(DEVICE)

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

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        sample_ids = getattr(batch_data, "sample_id", [""] * labels.size(0))
        graph_paths = getattr(batch_data, "graph_path", [""] * labels.size(0))
        attack_names = getattr(batch_data, "attack_name", [""] * labels.size(0))
        realized_fracs = getattr(batch_data, "realized_sub_fraction", [0.0] * labels.size(0))

        for i in range(labels.size(0)):
            true_y = int(labels[i].item())
            pred_y = int(preds[i].item())

            all_true.append(true_y)
            all_pred.append(pred_y)

            rows.append({
                "sample_id": sample_ids[i],
                "graph_path": graph_paths[i],
                "attack_name": attack_names[i],
                "realized_sub_fraction": float(realized_fracs[i]),
                "true_label": true_y,
                "true_name": CLASS_NAMES[true_y],
                "pred_label": pred_y,
                "pred_name": CLASS_NAMES[pred_y],
                "prob_ransomware": float(probs[i, 0].item()),
                "prob_benign": float(probs[i, 1].item()),
                "prob_other_malware": float(probs[i, 2].item()),
                "correct": int(true_y == pred_y),
            })

    metrics = compute_metrics(all_true, all_pred)
    return metrics, rows


# ============================================================================
# Split analysis
# ============================================================================

def compare_runs(clean_rows: List[dict], attacked_rows: List[dict]) -> dict:
    clean_map = {r["sample_id"]: r for r in clean_rows}
    attacked_map = {r["sample_id"]: r for r in attacked_rows}
    common_ids = sorted(set(clean_map) & set(attacked_map))

    if not common_ids:
        return {
            "prediction_flip_rate": 0.0,
            "threat_to_benign_flip_rate": 0.0,
            "mean_realized_sub_fraction": 0.0,
            "mean_confidence_drop_on_clean_pred": 0.0,
        }

    pred_flips = 0
    threat_to_benign = 0
    realized = []
    conf_drops = []

    for sid in common_ids:
        c = clean_map[sid]
        a = attacked_map[sid]

        if c["pred_label"] != a["pred_label"]:
            pred_flips += 1

        if c["true_label"] in [0, 2] and c["pred_label"] != 1 and a["pred_label"] == 1:
            threat_to_benign += 1

        realized.append(float(a.get("realized_sub_fraction", 0.0)))

        clean_pred = int(c["pred_label"])
        clean_conf = float(
            c["prob_ransomware"] if clean_pred == 0 else
            c["prob_benign"] if clean_pred == 1 else
            c["prob_other_malware"]
        )
        attacked_same_class_conf = float(
            a["prob_ransomware"] if clean_pred == 0 else
            a["prob_benign"] if clean_pred == 1 else
            a["prob_other_malware"]
        )
        conf_drops.append(clean_conf - attacked_same_class_conf)

    n = max(len(common_ids), 1)
    return {
        "prediction_flip_rate": float(pred_flips / n),
        "threat_to_benign_flip_rate": float(threat_to_benign / n),
        "mean_realized_sub_fraction": float(np.mean(realized)) if realized else 0.0,
        "mean_confidence_drop_on_clean_pred": float(np.mean(conf_drops)) if conf_drops else 0.0,
    }


def analyse_records(
    model,
    records: List[dict],
    split_name: str,
    schema: Dict,
    vocab,
    graph_processor,
    eval_epoch: int,
    output_dir: Path,
):
    collate = partial(custom_collate, schema=schema, validate_batch=False)

    def make_loader(attack_name: str):
        ds = RecordGraphDataset(
            records=records,
            schema=schema,
            graph_processor=graph_processor,
            attack_name=attack_name,
            validate_graphs=getattr(config, "CUSTOM_EVAL_VALIDATE_GRAPHS", True),
        )
        return DataLoader(
            ds,
            batch_size=getattr(config, "BATCH_SIZE", 16),
            shuffle=False,
            collate_fn=collate,
            num_workers=getattr(config, "NUM_WORKERS", 0),
            pin_memory=getattr(config, "PIN_MEMORY", False),
        )

    clean_loader = make_loader("clean")
    semantic_loader = make_loader("semantic_substitution")
    random_loader = make_loader("random_substitution")

    clean_metrics, clean_rows = run_eval(model, clean_loader, vocab, schema, eval_epoch)
    semantic_metrics, semantic_rows = run_eval(model, semantic_loader, vocab, schema, eval_epoch)
    random_metrics, random_rows = run_eval(model, random_loader, vocab, schema, eval_epoch)

    print_metrics(clean_metrics, f"TGNN | CLEAN | {split_name.upper()}")
    print_metrics(semantic_metrics, f"TGNN | SEMANTIC VECTOR SUBSTITUTION | {split_name.upper()}")
    print_metrics(random_metrics, f"TGNN | RANDOM VECTOR SUBSTITUTION | {split_name.upper()}")

    semantic_compare = compare_runs(clean_rows, semantic_rows)
    random_compare = compare_runs(clean_rows, random_rows)

    print(f"\nSemantic substitution flip rate      : {semantic_compare['prediction_flip_rate']:.4f}")
    print(f"Semantic threat->benign flip rate    : {semantic_compare['threat_to_benign_flip_rate']:.4f}")
    print(f"Semantic mean realized sub fraction  : {semantic_compare['mean_realized_sub_fraction']:.4f}")
    print(f"Semantic mean confidence drop        : {semantic_compare['mean_confidence_drop_on_clean_pred']:.4f}")

    print(f"\nRandom substitution flip rate        : {random_compare['prediction_flip_rate']:.4f}")
    print(f"Random threat->benign flip rate      : {random_compare['threat_to_benign_flip_rate']:.4f}")
    print(f"Random mean realized sub fraction    : {random_compare['mean_realized_sub_fraction']:.4f}")
    print(f"Random mean confidence drop          : {random_compare['mean_confidence_drop_on_clean_pred']:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clean_rows).to_csv(output_dir / f"semantic_vector_clean_{split_name}.csv", index=False)
    pd.DataFrame(semantic_rows).to_csv(output_dir / f"semantic_vector_semantic_sub_{split_name}.csv", index=False)
    pd.DataFrame(random_rows).to_csv(output_dir / f"semantic_vector_random_sub_{split_name}.csv", index=False)

    summary = {
        "split": split_name,
        "n_clean": len(clean_rows),
        "n_semantic_sub": len(semantic_rows),
        "n_random_sub": len(random_rows),
        "clean_metrics": clean_metrics,
        "semantic_substitution_metrics": semantic_metrics,
        "random_substitution_metrics": random_metrics,
        "semantic_substitution_comparison": semantic_compare,
        "random_substitution_comparison": random_compare,
        "settings": {
            "substitute_fraction": SUBSTITUTE_FRACTION,
            "keep_top_k": KEEP_TOP_K,
            "similarity_top_k": SIMILARITY_TOP_K,
            "semantic_dim": SEM_DIM,
            "apply_train_malware_subsample": APPLY_TRAIN_MALWARE_SUBSAMPLE,
            "train_malware_keep_frac": TRAIN_MALWARE_KEEP_FRAC,
        },
    }

    with open(output_dir / f"semantic_vector_substitution_{split_name}.summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Saved outputs in: {output_dir}")
    return summary


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TGNN semantic-vector substitution test")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default="results/semantic_vector_substitution_test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = Path(args.checkpoint or config.CUSTOM_EVAL_CHECKPOINT_PATH)
    graph_root = Path(config.ACTIVE_GRAPH_DIR)
    meta_path = Path(config.METADATA_PATH)
    out_dir = Path(args.output_dir)

    model, schema, vocab, graph_processor, eval_epoch = load_model_and_schema(ckpt_path)
    df = load_metadata(meta_path)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    summaries = []

    for split in splits:
        split_records = build_records_for_split(
            df=df,
            split_name=split,
            split_root=graph_root,
            max_samples=args.max_samples,
            sample_seed=args.seed,
        )

        if not split_records:
            print(f"[WARN] No usable records for split: {split}")
            continue

        summary = analyse_records(
            model=model,
            records=split_records,
            split_name=split,
            schema=schema,
            vocab=vocab,
            graph_processor=graph_processor,
            eval_epoch=eval_epoch,
            output_dir=out_dir,
        )
        summaries.append(summary)

    if len(summaries) > 1:
        with open(out_dir / "semantic_vector_substitution_all_summaries.json", "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"[INFO] Saved merged summary: {out_dir / 'semantic_vector_substitution_all_summaries.json'}")

    print("[DONE]")


if __name__ == "__main__":
    main()