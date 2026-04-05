#!/usr/bin/env python3
"""
scripts/evaluate_zero_day_custom.py

Evaluate an already-trained TGNN checkpoint on a custom zero-day / mixed
evaluation set built on the fly from existing graph folders.

Selection policy from config.py:
    - 200 benign from ACTIVE_GRAPH_DIR/{val,test}
    - 145 new unseen malware from CUSTOM_EVAL_NEW_MALWARE_DIR (all)
    - 55 old malware from ACTIVE_GRAPH_DIR/{val,test}
    - all ransomware from ACTIVE_GRAPH_DIR/{val,test}

Outputs:
    - manifest CSV
    - predictions CSV
    - results JSON
"""

from __future__ import annotations

import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config
from graph_construction.pytorch_dataset import hetero_collate
from graph_construction.vocabulary import load_vocab
from models.memory_utils import get_api_indices
from models.tgnn_model import create_hetero_tgnn
from utils.schema_manager import ensure_active_schema
from utils.schema_validator import validate_batch_or_raise


# ============================================================
# Loss
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = torch.tensor(alpha) if isinstance(alpha, list) else alpha
        else:
            self.alpha = None

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


# ============================================================
# Label helpers
# ============================================================

def category_to_label(all_categories: str) -> int:
    cats = str(all_categories).strip().lower()
    if "ransomware" in cats:
        return 0
    elif "benign" in cats:
        return 1
    else:
        return 2


def label_to_name(label: int) -> str:
    return config.CLASS_NAMES.get(int(label), f"Class_{label}")


# ============================================================
# Metrics
# ============================================================

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


def compute_threat_vs_benign(confusion_mat: np.ndarray) -> dict:
    TP = confusion_mat[0, 0] + confusion_mat[0, 2] + confusion_mat[2, 0] + confusion_mat[2, 2]
    FN = confusion_mat[0, 1] + confusion_mat[2, 1]
    TN = confusion_mat[1, 1]
    FP = confusion_mat[1, 0] + confusion_mat[1, 2]

    threat_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    threat_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    threat_f1 = (
        2 * threat_precision * threat_recall / (threat_precision + threat_recall)
        if (threat_precision + threat_recall) > 0 else 0.0
    )

    benign_recall = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    benign_precision = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    benign_f1 = (
        2 * benign_precision * benign_recall / (benign_precision + benign_recall)
        if (benign_precision + benign_recall) > 0 else 0.0
    )

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    return {
        "TP": int(TP),
        "FN": int(FN),
        "TN": int(TN),
        "FP": int(FP),
        "threat_recall": float(threat_recall),
        "threat_precision": float(threat_precision),
        "threat_f1": float(threat_f1),
        "benign_recall": float(benign_recall),
        "benign_precision": float(benign_precision),
        "benign_f1": float(benign_f1),
        "accuracy_2class": float(accuracy),
        "threats_missed": int(FN),
    }


def print_metrics(metrics, split="Custom Eval"):
    print(f"\n{'=' * 72}")
    print(f"{split} Metrics")
    print(f"{'=' * 72}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Loss:        {metrics.get('loss', 0.0):.4f}")
    print("\nPer-Class Metrics")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
    print(f"{'-' * 72}")
    for i in range(3):
        print(
            f"{config.CLASS_NAMES[i]:<20} "
            f"{metrics['per_class_precision'][i]:<12.4f} "
            f"{metrics['per_class_recall'][i]:<12.4f} "
            f"{metrics['per_class_f1'][i]:<12.4f} "
            f"{int(metrics['per_class_support'][i]):<12}"
        )
    print("\nConfusion Matrix")
    print(np.array(metrics["confusion_matrix"]))
    print(f"{'=' * 72}")


# ============================================================
# Manifest building
# ============================================================

def load_metadata_labels() -> Dict[str, int]:
    df = pd.read_csv(config.METADATA_PATH)

    if "md5" not in df.columns:
        raise KeyError("Expected 'md5' column in metadata CSV.")
    if "all_categories" not in df.columns:
        raise KeyError("Expected 'all_categories' column in metadata CSV.")

    df["_md5"] = df["md5"].astype(str).str.strip().str.lower()

    return {
        row["_md5"]: category_to_label(row["all_categories"])
        for _, row in df.iterrows()
    }


def discover_old_split_graphs(label_map: Dict[str, int]) -> List[dict]:
    rows = []

    for split in config.CUSTOM_EVAL_SOURCE_SPLITS:
        split_dir = Path(config.ACTIVE_GRAPH_DIR) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split dir: {split_dir}")

        for graph_path in sorted(split_dir.glob("*.pkl")):
            md5 = graph_path.stem.strip().lower()
            if md5 not in label_map:
                continue

            label = int(label_map[md5])
            rows.append({
                "sample_id": md5,
                "graph_path": str(graph_path.resolve()),
                "label": label,
                "class_name": label_to_name(label),
                "origin": f"old_{split}",
                "is_new_unseen": False,
            })

    return rows


def discover_new_unseen_graphs(old_sample_ids: set[str]) -> List[dict]:
    new_dir = Path(config.CUSTOM_EVAL_NEW_MALWARE_DIR)

    if not new_dir.exists():
        if config.CUSTOM_EVAL_FAIL_IF_NEW_MALWARE_DIR_MISSING:
            raise FileNotFoundError(f"Missing CUSTOM_EVAL_NEW_MALWARE_DIR: {new_dir}")
        print(f"[WARN] New malware dir missing: {new_dir}")
        return []

    rows = []
    overlap = []

    for graph_path in sorted(new_dir.glob("*.pkl")):
        sample_id = graph_path.stem.strip().lower()

        if config.CUSTOM_EVAL_ENFORCE_NEW_UNSEEN_DISJOINT and sample_id in old_sample_ids:
            overlap.append(sample_id)
            continue

        rows.append({
            "sample_id": sample_id,
            "graph_path": str(graph_path.resolve()),
            "label": int(config.CUSTOM_EVAL_NEW_MALWARE_LABEL),
            "class_name": label_to_name(int(config.CUSTOM_EVAL_NEW_MALWARE_LABEL)),
            "origin": "new_unseen",
            "is_new_unseen": True,
        })

    if overlap:
        raise RuntimeError(
            "Overlap detected between new unseen malware and old val/test pools. "
            f"Count={len(overlap)}. Example={overlap[:10]}"
        )

    return rows


def sample_rows(rows: List[dict], n: int, name: str) -> List[dict]:
    if len(rows) < n:
        msg = f"Requested {n} {name} samples, but only found {len(rows)}."
        if name == "benign" and config.CUSTOM_EVAL_FAIL_IF_NOT_ENOUGH_BENIGN:
            raise RuntimeError(msg)
        if name == "old_malware" and config.CUSTOM_EVAL_FAIL_IF_NOT_ENOUGH_OLD_MALWARE:
            raise RuntimeError(msg)
        print(f"[WARN] {msg} Using all available.")
        return list(rows)

    rows = list(rows)
    if config.CUSTOM_EVAL_SHUFFLE_BEFORE_SAMPLE:
        rng = random.Random(config.RANDOM_SEED)
        rng.shuffle(rows)

    return rows[:n]


def build_manifest() -> pd.DataFrame:
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    label_map = load_metadata_labels()
    old_rows = discover_old_split_graphs(label_map)
    old_sample_ids = {r["sample_id"] for r in old_rows}
    new_rows = discover_new_unseen_graphs(old_sample_ids)

    benign_pool = [r for r in old_rows if r["label"] == 1]
    ransomware_pool = [r for r in old_rows if r["label"] == 0]
    old_malware_pool = [r for r in old_rows if r["label"] == 2]

    selected_benign = sample_rows(benign_pool, int(config.CUSTOM_EVAL_TARGET_BENIGN), "benign")
    selected_old_malware = sample_rows(old_malware_pool, int(config.CUSTOM_EVAL_TARGET_OLD_MALWARE), "old_malware")
    selected_ransomware = ransomware_pool if config.CUSTOM_EVAL_INCLUDE_ALL_RANSOMWARE else []
    selected_new_malware = new_rows if config.CUSTOM_EVAL_INCLUDE_ALL_NEW_MALWARE else []

    final_rows = selected_benign + selected_old_malware + selected_new_malware + selected_ransomware
    if not final_rows:
        raise RuntimeError("Custom evaluation manifest is empty.")

    df = pd.DataFrame(final_rows).drop_duplicates(subset=["graph_path"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["class_name"] = df["label"].map(label_to_name)

    config.CUSTOM_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CUSTOM_EVAL_MANIFEST_PATH, index=False)

    print("\n[INFO] Custom manifest created:")
    print(f"       path   : {config.CUSTOM_EVAL_MANIFEST_PATH}")
    print(f"       total  : {len(df)}")
    print(f"       benign : {(df['label'] == 1).sum()}")
    print(f"       ransom : {(df['label'] == 0).sum()}")
    print(f"       other  : {(df['label'] == 2).sum()}")
    print(f"       origin : {dict(Counter(df['origin'].tolist()))}")

    return df


# ============================================================
# Dataset
# ============================================================

class CustomManifestDataset(Dataset):
    def __init__(self, manifest_csv: str | Path):
        self.manifest_path = Path(manifest_csv)
        self.df = pd.read_csv(self.manifest_path)

        required = {"graph_path", "label", "sample_id"}
        missing = required - set(self.df.columns)
        if missing:
            raise KeyError(f"Manifest missing required columns: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        graph_path = Path(row["graph_path"])
        label = int(row["label"])
        sample_id = str(row["sample_id"])

        if not graph_path.exists():
            raise FileNotFoundError(f"Graph not found: {graph_path}")

        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        graph.sample_id = sample_id
        graph.graph_path = str(graph_path)

        return graph, label


# ============================================================
# Forward / validate helpers
# ============================================================

def _extract_timestamps(batch_data) -> dict:
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
def run_eval(model, loader, criterion, vocab, schema, device, eval_epoch=None):
    model.eval()

    all_preds = []
    all_labels = []
    prediction_rows = []
    total_loss = 0.0
    total_graphs = 0

    for batch_data, labels in loader:
        if batch_data is None or labels is None:
            continue

        labels = labels.to(device)
        batch_data = batch_data.to(device)

        api_indices = _extract_api_indices(batch_data, vocab, device)
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

        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        total_loss += float(loss.item()) * labels.size(0)
        total_graphs += labels.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        sample_ids = getattr(batch_data, "sample_id", None)
        graph_paths = getattr(batch_data, "graph_path", None)

        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(labels.size(0))]
        if graph_paths is None:
            graph_paths = [""] * labels.size(0)

        for i in range(labels.size(0)):
            prediction_rows.append({
                "sample_id": sample_ids[i] if isinstance(sample_ids, list) else str(sample_ids),
                "graph_path": graph_paths[i] if isinstance(graph_paths, list) else str(graph_paths),
                "true_label": int(labels[i].item()),
                "true_name": label_to_name(int(labels[i].item())),
                "pred_label": int(preds[i].item()),
                "pred_name": label_to_name(int(preds[i].item())),
                "prob_ransomware": float(probs[i, 0].item()),
                "prob_benign": float(probs[i, 1].item()),
                "prob_other_malware": float(probs[i, 2].item()),
                "correct": int(preds[i].item() == labels[i].item()),
            })

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / max(total_graphs, 1)
    return metrics, prediction_rows


# ============================================================
# Main
# ============================================================

def main():
    if not config.USE_CUSTOM_ZERO_DAY_EVAL:
        print("USE_CUSTOM_ZERO_DAY_EVAL is False in config.py. Exiting.")
        return

    config.set_seed()
    device = torch.device(config.DEVICE)
    print(f"[INFO] Device: {device}")

    schema = ensure_active_schema()
    if isinstance(schema, dict):
        active_schema = schema
    else:
        with open(config.ACTIVE_SCHEMA_PATH, "r", encoding="utf-8") as f:
            active_schema = json.load(f)

    if config.CUSTOM_EVAL_REBUILD_MANIFEST or not Path(config.CUSTOM_EVAL_MANIFEST_PATH).exists():
        manifest_df = build_manifest()
    else:
        manifest_df = pd.read_csv(config.CUSTOM_EVAL_MANIFEST_PATH)
        print(f"[INFO] Using existing manifest: {config.CUSTOM_EVAL_MANIFEST_PATH}")

    dataset = CustomManifestDataset(config.CUSTOM_EVAL_MANIFEST_PATH)

    collate = lambda batch: hetero_collate(
        batch,
        schema=active_schema,
        validate_batch=getattr(config, "CUSTOM_EVAL_VALIDATE_BATCH", False),
    )

    loader = DataLoader(
        dataset,
        batch_size=config.CUSTOM_EVAL_BATCH_SIZE,
        shuffle=getattr(config, "CUSTOM_EVAL_SHUFFLE", False),
        collate_fn=collate,
        num_workers=config.CUSTOM_EVAL_NUM_WORKERS,
        pin_memory=config.CUSTOM_EVAL_PIN_MEMORY,
    )

    vocab = load_vocab(config.RESULTS_DIR / "api_vocabulary.json")

    model = create_hetero_tgnn(use_memory=config.USE_MEMORY).to(device)

    checkpoint_path = Path(config.CUSTOM_EVAL_CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    eval_epoch = int(ckpt.get("epoch", config.EPOCHS))

    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(f"[INFO] Checkpoint epoch : {eval_epoch}")
    print(f"[INFO] Custom eval batch size: {config.CUSTOM_EVAL_BATCH_SIZE}")

    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)

    metrics, prediction_rows = run_eval(
        model=model,
        loader=loader,
        criterion=criterion,
        vocab=vocab,
        schema=active_schema,
        device=device,
        eval_epoch=eval_epoch,
    )

    print_metrics(metrics, split="Custom Zero-Day Eval")

    cm = np.array(metrics["confusion_matrix"])
    tvb = compute_threat_vs_benign(cm)

    print("\nThreat vs Benign")
    print(f"  Threat Precision : {tvb['threat_precision']:.4f}")
    print(f"  Threat Recall    : {tvb['threat_recall']:.4f}")
    print(f"  Threat F1        : {tvb['threat_f1']:.4f}")
    print(f"  Benign Precision : {tvb['benign_precision']:.4f}")
    print(f"  Benign Recall    : {tvb['benign_recall']:.4f}")
    print(f"  Benign F1        : {tvb['benign_f1']:.4f}")
    print(f"  Threats Missed   : {tvb['threats_missed']}")

    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(config.CUSTOM_EVAL_PREDICTIONS_PATH, index=False)

    results = {
        "manifest_path": str(config.CUSTOM_EVAL_MANIFEST_PATH),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": eval_epoch,
        "num_samples": int(len(manifest_df)),
        "custom_eval_batch_size": int(config.CUSTOM_EVAL_BATCH_SIZE),
        "label_distribution": {
            "ransomware": int((manifest_df["label"] == 0).sum()),
            "benign": int((manifest_df["label"] == 1).sum()),
            "other_malware": int((manifest_df["label"] == 2).sum()),
        },
        "origin_distribution": {
            k: int(v) for k, v in Counter(manifest_df["origin"].tolist()).items()
        },
        "metrics_3class": metrics,
        "metrics_threat_vs_benign": tvb,
    }

    with open(config.CUSTOM_EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[INFO] Saved manifest    : {config.CUSTOM_EVAL_MANIFEST_PATH}")
    print(f"[INFO] Saved predictions : {config.CUSTOM_EVAL_PREDICTIONS_PATH}")
    print(f"[INFO] Saved results     : {config.CUSTOM_EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()