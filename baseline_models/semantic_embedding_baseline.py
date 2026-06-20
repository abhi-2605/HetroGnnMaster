#!/usr/bin/env python3
"""
baseline_models/semantic_embedding_baseline.py
-----------------------------------------------

Semantic-embedding baseline (pretrained MPNet API embeddings + trained classifier)
vs the existing TGNN checkpoint.

Goal
    Test whether pretrained SEMANTIC understanding of API names alone -- with NO
    graph structure and NO temporal modelling -- is enough. This is the
    "semantic-only paradigm" of the literature (Sentence-BERT / BERT / MPNet:
    reimers2019sbert, devlin2019bert, mpnet2020; BEACON's semantic representation
    perera2023beacon). It is the trainable, fair operationalisation of that
    paradigm (in place of a zero-shot LLM): same training regime, same eval set,
    same metrics, real probabilities.

    Crucially, the proposed TGNN's semantic branch uses these SAME MPNet
    embeddings. So "pool the MPNet embeddings + a classifier" measures whether
    semantic embeddings alone suffice; any gap vs the TGNN is the value added by
    fusing structure + temporal on top of semantics (RQ3).

Models
    1. Semantic-embedding classifier  (mean-pooled 768-d MPNet vectors -> MLP).
                                        Trained fresh.
    2. Existing TGNN checkpoint        (loaded, NOT trained).

Tracks  (ONLY attacks meaningful to an order-agnostic, graph-free semantic model)
    1. clean
    2. unigram_attack   -> API rename / substitution (breaks token identity)
    NOTE: bigram_attack (reordering) is excluded -- mean pooling is order-
    agnostic, so reordering has no effect. structure_attack is excluded -- no
    graph. Attacks are matched to the paradigm under test.

Representation / OOV (Option 1: cached dict + OOV -> drop)
    - Each API name -> its pretrained 768-d MPNet vector from cache/api_embeddings.pkl
      (323 known APIs), looked up by normalised name.
    - A trace -> MEAN of its known-API vectors (order + structure discarded).
    - Tokens NOT in the cached dict (unseen, or renamed by the attack) are simply
      dropped from the average. Under the rename attack almost all tokens drop out,
      so the pooled vector collapses -> prediction degrades (the fixed-vocabulary
      limitation, analogous to the TF-IDF collapse).

Fairness / setup
    - Trained classifier head (parity with RF/XGB/BiLSTM/GNN), same val/test
      splits, seed 42, focal loss + weighted oversampling.
    - Training EXCLUDES the train_augment_700 md5s; 50% train fraction to match
      the static GCN/GAT baseline.
    - 475-sample custom-eval manifest for the clean + attack comparison.

Where to run
    - The MLP trains/evaluates on CPU fine (it is tiny; the only real cost is
      loading graphs once to extract token sequences).
    - The TGNN comparison needs CUDA (Mamba backbone). Run the full script on the
      GPU box, OR set INCLUDE_TGNN = False to produce the semantic-baseline tracks
      locally.

Outputs (results/semantic_baseline/)
    semantic_embedding_<track>_predictions.csv
    existing_tgnn_<track>_predictions.csv        (if INCLUDE_TGNN)
    comparison_summary.json
    comparison_report.md
    models/semantic_embedding_best.pt
"""

from __future__ import annotations

import copy
import csv
import json
import pickle
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config
from graph_construction.vocabulary import load_vocab
from utils.schema_manager import ensure_active_schema
from models.tgnn_model import create_hetero_tgnn
from graph_construction.pytorch_dataset import RansomwareGraphDataset

# Reuse the EXACT data / attack / metric machinery used by the other baselines.
from baseline_models.prove_classical_model_attacks import (
    build_text_features_from_metadata,    # clean ordered-token sequences for train/val
    build_text_features_from_manifest,    # attacked ordered-token sequences for eval tracks
    load_split_sets,
    subsample_other_malware,
    get_all_train_exclusions,
    get_label,
    compute_metrics,
    compute_threat_vs_benign,
    compute_drop_summary,
    label_to_name,
    print_metrics,
    _stats_from_info_df,
    _normalize_api_name,
    # graph-side TGNN eval (reused verbatim)
    CustomManifestDataset,
    custom_eval_collate,
    run_tgnn_eval,
    FocalLoss,
)


# ============================================================
# Local controls
# ============================================================

RANDOM_STATE = 42

OUT_DIR = Path(config.RESULTS_DIR) / "semantic_baseline"
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEM_CKPT = MODEL_DIR / "semantic_embedding_best.pt"
# Point this at the no-700 checkpoint (the BiLSTM run used best_model_1.pt).
EXISTING_TGNN_CKPT = Path(config.RESULTS_DIR) / "checkpoints" / "best_model_1.pt"

API_EMB_PATH = PROJECT_ROOT / "cache" / "api_embeddings.pkl"

# Reuse the saved checkpoint if it exists (set True to force fresh training).
FORCE_RETRAIN_SEM = False

# Compare against the TGNN in the same run (needs CUDA / Mamba). Set False to
# run only the semantic-baseline tracks locally.
INCLUDE_TGNN = True

# Only paradigm-relevant attacks (order-agnostic + graph-free => rename only).
SEM_EVAL_TRACKS = ["clean", "unigram_attack"]

# ---- Eval-set sourcing (THIS FILE ONLY) -----------------------------------
# Build the eval set from val+test folders, NOT the zero-day old/all folders.
# Both zero-day sources are OFF; flip to True when you decide to use them.
USE_OLD_ZERO_DAY = False
USE_ALL_ZERO_DAY = False
# Target composition sourced from val+test (benign=1 / ransom=0 / malware=2).
SEM_EVAL_TARGETS = {1: 200, 0: 75, 2: 225}

# --- MLP head hyperparameters ---
SEM_EPOCHS = 60
SEM_PATIENCE = 10
SEM_LR = 1e-3
SEM_WEIGHT_DECAY = 1e-4
SEM_HIDDEN_DIM = 256
SEM_DROPOUT = 0.3
SEM_BATCH_SIZE = 64
SEM_LOG_EVERY = 25          # print intra-epoch progress every N batches

# --- training-set composition (match the other baselines) ---
SEM_TRAIN_FRAC = 0.50                      # match static GCN/GAT (STATIC_COMPARE_TRAIN_FRAC=0.50)
APPLY_MALWARE_SUBSAMPLE_TO_TRAIN = False
MALWARE_KEEP_FRAC_TRAIN = 0.8
AUGMENT_EXCLUDE_CSV = PROJECT_ROOT / "train_augment_700.csv"
# 200 malware added to val/test on 2026-06-07 -- excluded from the eval/test set.
EVAL_EXCLUDE_CSV = PROJECT_ROOT / "val_test_added_200.csv"


# ============================================================
# Basic helpers
# ============================================================

def set_seed(seed: int = RANDOM_STATE):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_augment_exclusions() -> set:
    """md5s of the 700 augmentation samples to keep OUT of training."""
    if not AUGMENT_EXCLUDE_CSV.exists():
        print(f"[WARN] augment-exclude file not found: {AUGMENT_EXCLUDE_CSV} (excluding 0)")
        return set()
    md5s = set()
    with open(AUGMENT_EXCLUDE_CSV, newline="") as f:
        for row in csv.DictReader(f):
            md5s.add(str(row["md5"]).strip().lower())
    print(f"[INFO] augment exclusions loaded: {len(md5s)} md5s from {AUGMENT_EXCLUDE_CSV.name}")
    return md5s


def load_eval_exclusions() -> set:
    """md5s of the 200 val/test malware added 2026-06-07 to keep OUT of the eval/test set."""
    if not EVAL_EXCLUDE_CSV.exists():
        print(f"[WARN] eval-exclude file not found: {EVAL_EXCLUDE_CSV} (excluding 0)")
        return set()
    md5s = {str(r["md5"]).strip().lower() for r in csv.DictReader(open(EVAL_EXCLUDE_CSV))}
    print(f"[INFO] eval exclusions loaded: {len(md5s)} md5s from {EVAL_EXCLUDE_CSV.name}")
    return md5s


def filter_eval_manifest(manifest_df):
    """Drop the excluded val/test added-200 md5s from the eval manifest."""
    excl = load_eval_exclusions()
    if not excl:
        return manifest_df
    sid = manifest_df["sample_id"].astype(str).str.strip().str.lower()
    before = len(manifest_df)
    out = manifest_df[~sid.isin(excl)].reset_index(drop=True)
    print(f"[INFO] eval manifest: removed {before - len(out)} of {before} "
          f"(val/test added-200 excluded) -> {len(out)} samples")
    return out


def build_semantic_eval_manifest():
    """LOCAL eval manifest for THIS semantic baseline only.

    Sources benign/ransom/malware from the val+test folders to the targets in
    SEM_EVAL_TARGETS, excludes the 200 added-malware, and by default does NOT
    use the zero-day old/ or all/ folders. Flip USE_OLD_ZERO_DAY / USE_ALL_ZERO_DAY
    to re-enable those later. Deterministic (seed=RANDOM_STATE)."""
    import random
    rng = random.Random(RANDOM_STATE)

    meta = {str(r["md5"]).strip().lower(): r for r in csv.DictReader(open(config.METADATA_PATH))}
    excl = load_eval_exclusions()
    split_root = Path(config.ACTIVE_GRAPH_DIR)

    pools = {0: [], 1: [], 2: []}
    for split in ["val", "test"]:
        for p in sorted((split_root / split).glob("*.pkl")):
            md5 = p.stem.strip().lower()
            if md5 in excl:
                continue
            r = meta.get(md5)
            if r is None:
                continue  # need a metadata label (skips integer-named / unlabeled pkls)
            lab = get_label(r["all_categories"])
            pools[lab].append({
                "sample_id": md5, "graph_path": str(p.resolve()),
                "label": lab, "origin": f"old_{split}", "is_new_unseen": False,
            })

    rows = []
    for lab, n in SEM_EVAL_TARGETS.items():
        cand = pools[lab][:]
        rng.shuffle(cand)
        if len(cand) < n:
            print(f"[WARN] semantic eval: only {len(cand)}/{n} available for {label_to_name(lab)}")
        rows.extend(cand[:n])

    zd_root = split_root.parent / "zero_day_unseen"
    if USE_OLD_ZERO_DAY:
        for p in sorted((zd_root / "old").glob("*.pkl")):
            md5 = p.stem.strip().lower()
            if md5 in excl:
                continue
            rows.append({"sample_id": md5, "graph_path": str(p.resolve()),
                         "label": 2, "origin": "old_zero_day", "is_new_unseen": True})
    if USE_ALL_ZERO_DAY:
        lbl = int(getattr(config, "CUSTOM_EVAL_NEW_MALWARE_LABEL", 2))
        for p in sorted((zd_root / "all").glob("*.pkl")):
            md5 = p.stem.strip().lower()
            if md5 in excl:
                continue
            rows.append({"sample_id": md5, "graph_path": str(p.resolve()),
                         "label": lbl, "origin": "new_unseen", "is_new_unseen": True})

    df = pd.DataFrame(rows)
    df["class_name"] = df["label"].map(label_to_name)
    print(f"[INFO] semantic eval manifest (val+test only, in-memory): total={len(df)} | "
          f"benign={int((df.label == 1).sum())} ransom={int((df.label == 0).sum())} "
          f"malware={int((df.label == 2).sum())} | "
          f"USE_OLD_ZERO_DAY={USE_OLD_ZERO_DAY} USE_ALL_ZERO_DAY={USE_ALL_ZERO_DAY}")
    return df


# ============================================================
# Semantic embedding lookup + pooling (cached dict, OOV -> drop)
# ============================================================

def load_embedding_lookup() -> tuple[Dict[str, np.ndarray], int]:
    """Load cache/api_embeddings.pkl and key it by NORMALISED api name so it
    matches the normalised tokens produced by build_text_features_*."""
    if not API_EMB_PATH.exists():
        raise FileNotFoundError(f"API embeddings not found: {API_EMB_PATH}")
    raw = pickle.load(open(API_EMB_PATH, "rb"))
    lookup = {_normalize_api_name(k): np.asarray(v, dtype=np.float32) for k, v in raw.items()}
    dim = len(next(iter(lookup.values())))
    print(f"[INFO] loaded {len(lookup)} API embeddings (dim={dim}) from {API_EMB_PATH.name}")
    return lookup, dim


def pool_sequence(seq: str, lookup: Dict[str, np.ndarray], dim: int) -> np.ndarray:
    """Mean-pool the known-API embeddings of a trace. OOV / renamed tokens drop out.
    Empty / all-OOV traces -> zero vector."""
    vecs = [lookup[t] for t in seq.split() if t in lookup]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


# ============================================================
# Dataset
# ============================================================

class SemDataset(Dataset):
    """In-memory dataset of (pooled-embedding vector, label) plus optional meta."""

    def __init__(self, sequences: List[str], labels: List[int], lookup, dim,
                 meta: List[dict] | None = None):
        self.x = np.stack([pool_sequence(s, lookup, dim) for s in sequences]).astype(np.float32)
        self.labels = [int(y) for y in labels]
        self.meta = meta if meta is not None else [{} for _ in labels]
        # fraction of samples whose pooled vector is all-zero (fully OOV) -- useful diagnostic
        self.zero_frac = float((np.abs(self.x).sum(axis=1) == 0).mean())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), int(self.labels[idx]), self.meta[idx]


def sem_collate(batch):
    xs, labels, metas = zip(*batch)
    return torch.stack(xs), torch.tensor(labels, dtype=torch.long), list(metas)


# ============================================================
# Model
# ============================================================

class SemanticMLP(nn.Module):
    """Small MLP over the pooled 768-d MPNet vector (semantics only)."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int = 3, dropout: float = SEM_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Train / evaluate the semantic MLP
# ============================================================

def make_oversampler(labels: List[int]) -> WeightedRandomSampler:
    counts = Counter(labels)
    class_w = {c: 1.0 / max(counts[c], 1) for c in counts}
    weights = [class_w[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def train_semantic_model(train_ds, val_ds, in_dim, device, criterion):
    model = SemanticMLP(in_dim=in_dim, hidden=SEM_HIDDEN_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SEM_LR, weight_decay=SEM_WEIGHT_DECAY)

    sampler = make_oversampler(train_ds.labels)
    train_loader = DataLoader(train_ds, batch_size=SEM_BATCH_SIZE, sampler=sampler, collate_fn=sem_collate)
    val_loader = DataLoader(val_ds, batch_size=SEM_BATCH_SIZE, shuffle=False, collate_fn=sem_collate)

    best_macro_f1 = -1.0
    best_state = None
    patience_left = SEM_PATIENCE
    n_batches = len(train_loader)

    for epoch in range(1, SEM_EPOCHS + 1):
        model.train()
        train_loss, steps = 0.0, 0
        for x, labels, _ in train_loader:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += float(loss.item())
            steps += 1
            if steps % SEM_LOG_EVERY == 0 or steps == n_batches:
                print(f"    [epoch {epoch:03d}] batch {steps}/{n_batches}  "
                      f"running_loss={train_loss / max(steps, 1):.4f}", flush=True)

        val_metrics, _ = evaluate_semantic_model(model, val_loader, criterion, device, "Semantic-embedding (val)")
        val_macro_f1 = float(val_metrics["macro_f1"])
        print(f"[TRAIN][Semantic] epoch={epoch:03d} train_loss={train_loss / max(steps, 1):.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_macro_f1:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_left = SEM_PATIENCE
        else:
            patience_left -= 1
        if patience_left <= 0:
            print(f"[INFO] Early stopping Semantic-embedding at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_macro_f1


@torch.no_grad()
def evaluate_semantic_model(model, loader, criterion, device, model_name: str):
    model.eval()
    all_preds, all_labels, rows = [], [], []
    total_loss, steps = 0.0, 0

    for x, labels, metas in loader:
        x, labels_d = x.to(device), labels.to(device)
        logits = model(x)
        loss = criterion(logits, labels_d)
        total_loss += float(loss.item())
        steps += 1
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

        for i in range(labels.size(0)):
            m = metas[i]
            rows.append({
                "sample_id": m.get("sample_id", ""),
                "origin": m.get("origin", ""),
                "graph_path": m.get("graph_path", ""),
                "attack_name": m.get("attack_name", ""),
                "bigram_overlap_vs_clean": float(m.get("bigram_overlap_vs_clean", 1.0)),
                "unigram_overlap_vs_clean": float(m.get("unigram_overlap_vs_clean", 1.0)),
                "length_ratio_vs_clean": float(m.get("length_ratio_vs_clean", 1.0)),
                "true_label": int(labels[i].item()),
                "true_name": label_to_name(int(labels[i].item())),
                "pred_label": int(preds[i].item()),
                "pred_name": label_to_name(int(preds[i].item())),
                "prob_ransomware": float(probs[i, 0].item()),
                "prob_benign": float(probs[i, 1].item()),
                "prob_other_malware": float(probs[i, 2].item()),
                "correct": int(preds[i].item() == labels[i].item()),
                "model": model_name,
            })

    if not all_labels:
        raise RuntimeError(f"No samples evaluated for {model_name}")
    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / max(steps, 1)
    return metrics, rows


# ============================================================
# Reporting
# ============================================================

def metric_cell(metrics):
    tvb = compute_threat_vs_benign(np.array(metrics["confusion_matrix"]))
    return f"{metrics['accuracy'] * 100:.2f}% / {tvb['threats_missed']} missed"


def save_report(rows, out_path):
    lines = ["# Semantic-embedding baseline vs Existing TGNN", ""]
    header = "| Model | " + " | ".join(t.replace("_", " ").title() + " Acc / Missed" for t in SEM_EVAL_TRACKS) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(SEM_EVAL_TRACKS))
    for row in rows:
        cells = " | ".join(row.get(t, "—") for t in SEM_EVAL_TRACKS)
        lines.append(f"| {row['model']} | {cells} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main():
    set_seed(RANDOM_STATE)
    device = torch.device(config.DEVICE)
    print(f"[INFO] Output directory : {OUT_DIR}")
    print(f"[INFO] Device           : {device}")
    print(f"[INFO] Eval tracks      : {SEM_EVAL_TRACKS}  (bigram/structure excluded -- not relevant to an order-agnostic, graph-free semantic model)")
    print(f"[INFO] Include TGNN     : {INCLUDE_TGNN}")

    lookup, emb_dim = load_embedding_lookup()

    # ---- custom-eval manifest (same as the other baselines) ----
    # Local eval manifest: val+test only, 200 excluded, zero-day old/all OFF by default.
    custom_manifest_df = build_semantic_eval_manifest()
    custom_manifest_df = filter_eval_manifest(custom_manifest_df)  # belt-and-suspenders (builder already excludes the 200)

    # ---- training / validation token sequences ----
    metadata_path = Path(config.METADATA_PATH)
    split_root = Path(config.ACTIVE_GRAPH_DIR)
    df = pd.read_csv(metadata_path)
    df["_md5"] = df["md5"].astype(str).str.strip().str.lower()
    if "all_categories" not in df.columns:
        raise KeyError("Expected metadata column 'all_categories' not found.")

    train_md5s, val_md5s, _ = load_split_sets(df, split_root)
    df_train = df[df["_md5"].isin(train_md5s)].copy()
    df_val = df[df["_md5"].isin(val_md5s)].copy()

    exclusions = set(get_all_train_exclusions()) | load_augment_exclusions()
    if exclusions:
        before = len(df_train)
        df_train = df_train[~df_train["_md5"].isin(exclusions)].copy()
        print(f"[INFO] Removed {before - len(df_train)} train rows via exclusions "
              f"(incl. {len(load_augment_exclusions())} augment md5s)")

    if APPLY_MALWARE_SUBSAMPLE_TO_TRAIN:
        df_train = subsample_other_malware(df_train, MALWARE_KEEP_FRAC_TRAIN, "TRAIN")

    if SEM_TRAIN_FRAC < 1.0:
        n0 = len(df_train)
        df_train = df_train.sample(frac=SEM_TRAIN_FRAC, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[INFO] train subsampled to {len(df_train)}/{n0} ({SEM_TRAIN_FRAC:.0%}) to match the static-GNN baseline")

    train_dir = split_root / "train"
    val_dir = split_root / "val"
    X_train_text, y_train, _ = build_text_features_from_metadata(df_train, train_dir)
    X_val_text, y_val, _ = build_text_features_from_metadata(df_val, val_dir)

    train_ds = SemDataset(X_train_text, y_train, lookup, emb_dim)
    val_ds = SemDataset(X_val_text, y_val, lookup, emb_dim)
    print(f"[INFO] train={len(y_train)} {dict(Counter(y_train))} (all-OOV frac={train_ds.zero_frac:.3f}) | "
          f"val={len(y_val)} {dict(Counter(y_val))} (all-OOV frac={val_ds.zero_frac:.3f})")

    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)

    # ---- train (or load) the semantic MLP ----
    if SEM_CKPT.exists() and not FORCE_RETRAIN_SEM:
        ckpt = torch.load(SEM_CKPT, map_location=device, weights_only=False)
        sem_model = SemanticMLP(in_dim=emb_dim, hidden=SEM_HIDDEN_DIM).to(device)
        sem_model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded semantic MLP from {SEM_CKPT}")
    else:
        sem_model, best_f1 = train_semantic_model(train_ds, val_ds, emb_dim, device, criterion)
        torch.save(
            {
                "model_state_dict": sem_model.state_dict(),
                "best_val_macro_f1": best_f1,
                "model": "Semantic-embedding (mean-pooled MPNet, semantics-only)",
                "emb_dim": emb_dim,
                "pooling": "mean",
                "oov_policy": "drop (cached dict only)",
                "excluded_augment_md5s": len(load_augment_exclusions()),
                "train_size": len(y_train),
                "hyperparams": {
                    "hidden": SEM_HIDDEN_DIM, "dropout": SEM_DROPOUT,
                    "lr": SEM_LR, "weight_decay": SEM_WEIGHT_DECAY, "epochs": SEM_EPOCHS,
                },
            },
            SEM_CKPT,
        )
        print(f"[INFO] Saved semantic MLP checkpoint: {SEM_CKPT}")

    # ---- build eval tracks (attacked token sequences -> pooled vectors) ----
    sem_eval_loaders = {}
    attack_stats = {}
    for track in SEM_EVAL_TRACKS:
        X_text, y_eval, info_df = build_text_features_from_manifest(custom_manifest_df, attack_name=track)
        attack_stats[track] = _stats_from_info_df(info_df)
        meta = info_df.to_dict("records")
        for m in meta:
            m["attack_name"] = track
        ds = SemDataset(X_text, y_eval, lookup, emb_dim, meta=meta)
        print(f"[INFO] eval track '{track}': n={len(ds)} all-OOV frac={ds.zero_frac:.3f}")
        sem_eval_loaders[track] = DataLoader(ds, batch_size=SEM_BATCH_SIZE, shuffle=False, collate_fn=sem_collate)

    # ---- optional TGNN graph eval on the same tracks (reused verbatim) ----
    tgnn_model = tgnn_loaders = tgnn_vocab = active_schema = tgnn_eval_epoch = None
    if INCLUDE_TGNN:
        schema = ensure_active_schema()
        if isinstance(schema, dict):
            active_schema = schema
        else:
            with open(config.ACTIVE_SCHEMA_PATH, "r", encoding="utf-8") as f:
                active_schema = json.load(f)
        graph_processor = RansomwareGraphDataset.__new__(RansomwareGraphDataset)
        graph_processor.schema = active_schema
        collate = partial(custom_eval_collate, schema=active_schema, validate_batch=False)

        def make_tgnn_loader(track):
            _, _, info_df = build_text_features_from_manifest(custom_manifest_df, attack_name=track)
            manifest_path = OUT_DIR / f"tmp_tgnn_{track}_manifest.csv"
            tmp_df = custom_manifest_df.copy()
            tmp_df["attack_name"] = track
            tmp_df["bigram_overlap_vs_clean"] = info_df["bigram_overlap_vs_clean"].tolist()
            tmp_df["unigram_overlap_vs_clean"] = info_df["unigram_overlap_vs_clean"].tolist()
            tmp_df["length_ratio_vs_clean"] = info_df["length_ratio_vs_clean"].tolist()
            tmp_df.to_csv(manifest_path, index=False)
            ds = CustomManifestDataset(
                manifest_csv=manifest_path, schema=active_schema, graph_processor=graph_processor,
                attack_name=track, validate_graphs=getattr(config, "CUSTOM_EVAL_VALIDATE_GRAPHS", True),
            )
            return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate,
                              num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

        tgnn_loaders = {track: make_tgnn_loader(track) for track in SEM_EVAL_TRACKS}
        tgnn_vocab = load_vocab(config.RESULTS_DIR / "api_vocabulary.json")
        tgnn_model = create_hetero_tgnn(use_memory=config.USE_MEMORY).to(device)
        if not EXISTING_TGNN_CKPT.exists():
            raise FileNotFoundError(f"TGNN checkpoint not found: {EXISTING_TGNN_CKPT}")
        ckpt = torch.load(EXISTING_TGNN_CKPT, map_location=device, weights_only=False)
        tgnn_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        tgnn_eval_epoch = int(ckpt.get("epoch", config.EPOCHS))
        print(f"[INFO] Loaded existing TGNN checkpoint: {EXISTING_TGNN_CKPT} (epoch {tgnn_eval_epoch})")

    # ---- run all models on all tracks ----
    summary = {
        "description": "Semantic-embedding baseline (mean-pooled MPNet, semantics-only) vs existing TGNN under rename attack.",
        "output_dir": str(OUT_DIR),
        "checkpoints": {"semantic_embedding": str(SEM_CKPT), "existing_tgnn": str(EXISTING_TGNN_CKPT)},
        "sem_setup": {
            "model": "mean-pooled MPNet embeddings -> MLP (semantics-only, no graph, no order)",
            "embedding_source": str(API_EMB_PATH),
            "embedding_dim": emb_dim,
            "pooling": "mean",
            "oov_policy": "drop (cached dict only, OOV -> excluded from pool)",
            "train_size": len(y_train),
            "val_size": len(y_val),
            "excluded_augment_md5s": len(load_augment_exclusions()),
            "train_fraction": SEM_TRAIN_FRAC,
        },
        "tracks": SEM_EVAL_TRACKS,
        "tracks_note": "bigram (reorder) + structure attacks excluded; mean pooling is order-agnostic and there is no graph.",
        "attack_stats": attack_stats,
        "results": {},
    }
    report_rows = []

    # ---- Semantic-embedding ----
    sem_name = "Semantic-embedding"
    summary["results"][sem_name] = {}
    row = {"model": sem_name}
    clean_metrics = None
    for track in SEM_EVAL_TRACKS:
        metrics, pred_rows = evaluate_semantic_model(sem_model, sem_eval_loaders[track], criterion, device, sem_name)
        print_metrics(metrics, f"{sem_name} | {track.upper()}")
        pred_path = OUT_DIR / f"semantic_embedding_{track}_predictions.csv"
        pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
        entry = {
            "metrics": metrics,
            "tvb": compute_threat_vs_benign(np.array(metrics["confusion_matrix"])),
            "predictions_csv": str(pred_path),
        }
        if track == "clean":
            clean_metrics = metrics
        else:
            entry["drop_vs_clean"] = compute_drop_summary(clean_metrics, metrics)
        summary["results"][sem_name][track] = entry
        row[track] = metric_cell(metrics)
    report_rows.append(row)

    # ---- TGNN ----
    if INCLUDE_TGNN:
        tg_name = "Existing TGNN"
        summary["results"][tg_name] = {}
        row = {"model": tg_name}
        clean_metrics = None
        for track in SEM_EVAL_TRACKS:
            metrics, pred_rows = run_tgnn_eval(
                tgnn_model, tgnn_loaders[track], criterion, tgnn_vocab, active_schema, device,
                eval_epoch=tgnn_eval_epoch,
            )
            print_metrics(metrics, f"{tg_name} | {track.upper()}")
            pred_path = OUT_DIR / f"existing_tgnn_{track}_predictions.csv"
            pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
            entry = {
                "metrics": metrics,
                "tvb": compute_threat_vs_benign(np.array(metrics["confusion_matrix"])),
                "predictions_csv": str(pred_path),
            }
            if track == "clean":
                clean_metrics = metrics
            else:
                entry["drop_vs_clean"] = compute_drop_summary(clean_metrics, metrics)
            summary["results"][tg_name][track] = entry
            row[track] = metric_cell(metrics)
        report_rows.append(row)

    # ---- save artifacts ----
    summary_path = OUT_DIR / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_report(report_rows, OUT_DIR / "comparison_report.md")
    print(f"\n[INFO] Saved summary JSON : {summary_path}")
    print(f"[INFO] Saved report      : {OUT_DIR / 'comparison_report.md'}")
    print(f"[INFO] Outputs in        : {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
