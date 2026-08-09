#!/usr/bin/env python3
"""
baseline_models/sequential_dl_baseline.py
------------------------------------------

Sequential deep-learning baseline (BiLSTM over the ordered API-call sequence)
vs the existing TGNN checkpoint.

Goal
    Test whether sequence / ordering alone -- with NO graph structure and NO
    pretrained semantic embeddings -- is enough. This is the "sequence-only
    paradigm" of the literature (Pascanu 2015, Athiwaratkun 2017, Nguyen 2023).
    Any gap vs the TGNN is then attributable to the structural + semantic
    branches the sequence model lacks.

Models
    1. Sequential BiLSTM  (learned token-ID embeddings over ordered API tokens,
                           attention pooling, 3-class head). Trained fresh.
    2. Existing TGNN checkpoint (loaded, NOT trained).

Tracks  (ONLY attacks that are meaningful to a sequence model)
    1. clean
    2. unigram_attack   -> API rename / substitution   (breaks token identity)
    3. bigram_attack    -> reordering                   (breaks local ordering)
    4. combined_attack  -> both
    NOTE: structure_attack is intentionally NOT run here. A sequence model has
    no graph, so topology attacks are not relevant to it (that attack belongs to
    the static-GNN comparison). Attacks are matched to the paradigm under test.

Fairness / setup
    - BiLSTM uses a SELF-CONTAINED token vocabulary built from its own training
      sequences only (no leakage from the TGNN vocab). Renamed tokens therefore
      fall to <UNK> under attack -- the same memorisation weakness that collapses
      the TF-IDF baselines.
    - Training EXCLUDES the train_augment_700 md5s, so the baseline's training
      data matches the TGNN checkpoint that did not see those 700 samples.
    - Same val/test splits, seed 42, focal loss + weighted oversampling
      (parity with the TGNN training regime).
    - The 475-sample custom-eval set is used for the clean + attack comparison,
      identical to the classical / static-GNN baselines.

Where to run
    - The BiLSTM trains/evaluates on CPU/MPS fine (it is light).
    - The TGNN comparison requires CUDA (the Mamba backbone). Run the full script
      on the GPU box, OR set INCLUDE_TGNN = False to produce the BiLSTM-only
      tracks locally and compare against the already-saved TGNN prediction CSVs.

Outputs (results/sequential_baseline/)
    sequential_bilstm_<track>_predictions.csv
    existing_tgnn_<track>_predictions.csv        (if INCLUDE_TGNN)
    comparison_summary.json
    comparison_report.md
    models/sequential_bilstm_best.pt
"""

from __future__ import annotations

import copy
import csv
import json
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config
from graph_construction.vocabulary import load_vocab
from utils.schema_manager import ensure_active_schema
from models.tgnn_model import create_hetero_tgnn
from graph_construction.pytorch_dataset import RansomwareGraphDataset

# Reuse the EXACT data / attack / metric machinery used by the other baselines,
# so the sequence model trains and is evaluated on identical token data.
from baseline_models.prove_classical_model_attacks import (
    EVAL_TRACKS,                          # ["clean","unigram_attack","bigram_attack","combined_attack"]
    build_custom_eval_manifest,
    build_text_features_from_metadata,    # clean ordered-token sequences for train/val
    build_text_features_from_manifest,    # attacked ordered-token sequences for eval tracks
    load_split_sets,
    subsample_other_malware,
    get_all_train_exclusions,
    compute_metrics,
    compute_threat_vs_benign,
    compute_drop_summary,
    label_to_name,
    print_metrics,
    _stats_from_info_df,
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

OUT_DIR = Path(config.RESULTS_DIR) / "sequential_baseline"
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQ_CKPT = MODEL_DIR / "sequential_bilstm_best.pt"
EXISTING_TGNN_CKPT = Path(config.RESULTS_DIR) / "checkpoints" / "best_model_1.pt"

# ---- Eval set = held-out VAL + TEST only (no zero-day) -----------------------
# Match the canonical B3 baseline setup: the §5.6 paradigm comparison uses a clean
# held-out val+test mix; the zero-day unseen malware belongs to the §5.8
# generalisation experiment. Scope these overrides to THIS script only (config.py
# on disk untouched) and redirect the manifest cache into OUT_DIR so the shared
# manifest CSV is not clobbered.
config.CUSTOM_EVAL_INCLUDE_ALL_NEW_MALWARE = False   # drop the 170 zero-day rows
config.CUSTOM_EVAL_REBUILD_MANIFEST = True           # force a fresh val+test build
config.CUSTOM_EVAL_MANIFEST_PATH = OUT_DIR / "custom_eval_manifest_valtest.csv"

# Reuse the saved checkpoint if it exists (set True to force fresh training).
FORCE_RETRAIN_SEQ = False

# Compare against the TGNN in the same run (needs CUDA / Mamba). Set False to
# run only the BiLSTM tracks locally.
INCLUDE_TGNN = True

# Only paradigm-relevant attacks (no structure attack for a sequence model).
SEQ_EVAL_TRACKS = list(EVAL_TRACKS)

# --- BiLSTM hyperparameters ---
SEQ_EPOCHS = 40
SEQ_PATIENCE = 8
SEQ_LR = 1e-3
SEQ_WEIGHT_DECAY = 1e-4
SEQ_EMB_DIM = 128
SEQ_HIDDEN_DIM = 128
SEQ_NUM_LAYERS = 2
SEQ_DROPOUT = 0.3
SEQ_BIDIRECTIONAL = True
SEQ_BATCH_SIZE = 64
SEQ_MAX_LEN = 2048          # per-sequence cap fed to the LSTM (even subsample if longer)
SEQ_MIN_TOKEN_FREQ = 1      # drop tokens rarer than this from the BiLSTM vocab
SEQ_LOG_EVERY = 20          # print intra-epoch progress every N batches

# --- training-set composition ---
SEQ_TRAIN_FRAC = 0.50                      # match the static GCN/GAT baseline (STATIC_COMPARE_TRAIN_FRAC=0.50)
APPLY_MALWARE_SUBSAMPLE_TO_TRAIN = False   # keep the full train set (minus the 700)
MALWARE_KEEP_FRAC_TRAIN = 0.8              # only used if the flag above is True
AUGMENT_EXCLUDE_CSV = PROJECT_ROOT / "train_augment_700.csv"
# 200 malware added to val/test on 2026-06-07 -- excluded from the eval/test set.
EVAL_EXCLUDE_CSV = PROJECT_ROOT / "val_test_added_200.csv"

# Special token ids for the self-contained BiLSTM vocabulary.
UNK_ID = 0
PAD_ID = 1


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


# ============================================================
# Self-contained token vocabulary (built from train sequences only)
# ============================================================

def build_token_vocab(train_sequences: List[str]) -> Dict[str, int]:
    counter: Counter = Counter()
    for seq in train_sequences:
        counter.update(seq.split())
    token2id: Dict[str, int] = {"<UNK>": UNK_ID, "<PAD>": PAD_ID}
    next_id = 2
    for tok, cnt in counter.most_common():
        if cnt < SEQ_MIN_TOKEN_FREQ:
            continue
        token2id[tok] = next_id
        next_id += 1
    print(f"[INFO] BiLSTM token vocab size = {len(token2id)} "
          f"(from {len(train_sequences)} train sequences, min_freq={SEQ_MIN_TOKEN_FREQ})")
    return token2id


def seq_to_ids(seq: str, token2id: Dict[str, int], max_len: int = SEQ_MAX_LEN) -> List[int]:
    toks = seq.split()
    if not toks:
        return [UNK_ID]                       # never feed an empty sequence
    if len(toks) > max_len:                    # even subsample to preserve overall shape
        idx = np.linspace(0, len(toks) - 1, max_len).round().astype(int)
        toks = [toks[i] for i in idx]
    return [token2id.get(t, UNK_ID) for t in toks]


# ============================================================
# Datasets
# ============================================================

class SeqDataset(Dataset):
    """In-memory dataset of (token-id sequence, label) plus optional meta."""

    def __init__(self, sequences: List[str], labels: List[int], token2id: Dict[str, int],
                 meta: List[dict] | None = None):
        self.ids = [seq_to_ids(s, token2id) for s in sequences]
        self.labels = [int(y) for y in labels]
        self.meta = meta if meta is not None else [{} for _ in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.ids[idx], dtype=torch.long),
            int(self.labels[idx]),
            self.meta[idx],
        )


def seq_collate(batch):
    seqs, labels, metas = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_t = int(lengths.max().item())
    padded = torch.full((len(seqs), max_t), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    return padded, lengths, torch.tensor(labels, dtype=torch.long), list(metas)


# ============================================================
# Model
# ============================================================

class SequenceClassifier(nn.Module):
    """BiLSTM over ordered API tokens with masked attention pooling."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden: int, layers: int,
                 dropout: float, bidir: bool, num_classes: int = 3, pad_idx: int = PAD_ID):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim, hidden, num_layers=layers, batch_first=True,
            bidirectional=bidir, dropout=dropout if layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidir else 1)
        self.attn = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = ids != PAD_ID                                   # [B,T]
        emb = self.emb(ids)                                    # [B,T,E]
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))  # [B,T,H]
        scores = self.attn(out).squeeze(-1)                    # [B,T]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)   # [B,T,1]
        pooled = (out * weights).sum(dim=1)                    # [B,H]
        return self.fc(self.dropout(pooled))


# ============================================================
# Train / evaluate the BiLSTM
# ============================================================

def make_oversampler(labels: List[int]) -> WeightedRandomSampler:
    counts = Counter(labels)
    class_w = {c: 1.0 / max(counts[c], 1) for c in counts}
    weights = [class_w[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def train_sequence_model(train_ds, val_ds, vocab_size, device, criterion):
    model = SequenceClassifier(
        vocab_size=vocab_size, emb_dim=SEQ_EMB_DIM, hidden=SEQ_HIDDEN_DIM,
        layers=SEQ_NUM_LAYERS, dropout=SEQ_DROPOUT, bidir=SEQ_BIDIRECTIONAL,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SEQ_LR, weight_decay=SEQ_WEIGHT_DECAY)

    sampler = make_oversampler(train_ds.labels)
    train_loader = DataLoader(train_ds, batch_size=SEQ_BATCH_SIZE, sampler=sampler, collate_fn=seq_collate)
    val_loader = DataLoader(val_ds, batch_size=SEQ_BATCH_SIZE, shuffle=False, collate_fn=seq_collate)

    best_macro_f1 = -1.0
    best_state = None
    patience_left = SEQ_PATIENCE

    n_batches = len(train_loader)
    for epoch in range(1, SEQ_EPOCHS + 1):
        model.train()
        train_loss, steps = 0.0, 0
        for padded, lengths, labels, _ in train_loader:
            padded, labels = padded.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(padded, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += float(loss.item())
            steps += 1
            if steps % SEQ_LOG_EVERY == 0 or steps == n_batches:
                print(f"    [epoch {epoch:03d}] batch {steps}/{n_batches}  "
                      f"running_loss={train_loss / max(steps, 1):.4f}", flush=True)

        val_metrics, _ = evaluate_sequence_model(model, val_loader, criterion, device, "Sequential BiLSTM (val)")
        val_macro_f1 = float(val_metrics["macro_f1"])
        print(f"[TRAIN][BiLSTM] epoch={epoch:03d} train_loss={train_loss / max(steps, 1):.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_macro_f1:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_left = SEQ_PATIENCE
        else:
            patience_left -= 1
        if patience_left <= 0:
            print(f"[INFO] Early stopping BiLSTM at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_macro_f1


@torch.no_grad()
def evaluate_sequence_model(model, loader, criterion, device, model_name: str):
    model.eval()
    all_preds, all_labels, rows = [], [], []
    total_loss, steps = 0.0, 0

    for padded, lengths, labels, metas in loader:
        padded, labels_d = padded.to(device), labels.to(device)
        logits = model(padded, lengths)
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
    lines = ["# Sequential BiLSTM vs Existing TGNN", ""]
    header = "| Model | " + " | ".join(t.replace("_", " ").title() + " Acc / Missed" for t in SEQ_EVAL_TRACKS) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(SEQ_EVAL_TRACKS))
    for row in rows:
        cells = " | ".join(row.get(t, "—") for t in SEQ_EVAL_TRACKS)
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
    print(f"[INFO] Eval tracks      : {SEQ_EVAL_TRACKS}  (structure attack excluded -- not relevant to a sequence model)")
    print(f"[INFO] Include TGNN     : {INCLUDE_TGNN}")

    # ---- custom-eval manifest (475 samples), same as the other baselines ----
    if getattr(config, "CUSTOM_EVAL_REBUILD_MANIFEST", True) or not Path(config.CUSTOM_EVAL_MANIFEST_PATH).exists():
        custom_manifest_df = build_custom_eval_manifest()
    else:
        custom_manifest_df = pd.read_csv(config.CUSTOM_EVAL_MANIFEST_PATH)
        print(f"[INFO] Using existing custom eval manifest: {config.CUSTOM_EVAL_MANIFEST_PATH}")

    custom_manifest_df = filter_eval_manifest(custom_manifest_df)

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

    # exclusions: standard train exclusions + the 700 augmentation md5s
    exclusions = set(get_all_train_exclusions()) | load_augment_exclusions()
    if exclusions:
        before = len(df_train)
        df_train = df_train[~df_train["_md5"].isin(exclusions)].copy()
        print(f"[INFO] Removed {before - len(df_train)} train rows via exclusions "
              f"(incl. {len(load_augment_exclusions())} augment md5s)")

    if APPLY_MALWARE_SUBSAMPLE_TO_TRAIN:
        df_train = subsample_other_malware(df_train, MALWARE_KEEP_FRAC_TRAIN, "TRAIN")

    # Match the static GCN/GAT baseline's 50% train fraction (same seed) so the
    # two baselines train on the same amount of data.
    if SEQ_TRAIN_FRAC < 1.0:
        n0 = len(df_train)
        df_train = df_train.sample(frac=SEQ_TRAIN_FRAC, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[INFO] train subsampled to {len(df_train)}/{n0} ({SEQ_TRAIN_FRAC:.0%}) "
              f"to match the static-GNN baseline")

    train_dir = split_root / "train"
    val_dir = split_root / "val"
    X_train_text, y_train, _ = build_text_features_from_metadata(df_train, train_dir)
    X_val_text, y_val, _ = build_text_features_from_metadata(df_val, val_dir)
    print(f"[INFO] train sequences={len(y_train)} {dict(Counter(y_train))} | "
          f"val sequences={len(y_val)} {dict(Counter(y_val))}")

    # ---- self-contained BiLSTM token vocabulary (train-only) ----
    token2id = build_token_vocab(X_train_text)
    vocab_size = len(token2id)

    train_ds = SeqDataset(X_train_text, y_train, token2id)
    val_ds = SeqDataset(X_val_text, y_val, token2id)

    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)

    # ---- train (or load) the BiLSTM ----
    if SEQ_CKPT.exists() and not FORCE_RETRAIN_SEQ:
        ckpt = torch.load(SEQ_CKPT, map_location=device, weights_only=False)
        token2id = ckpt["token2id"]
        vocab_size = len(token2id)
        seq_model = SequenceClassifier(
            vocab_size=vocab_size, emb_dim=SEQ_EMB_DIM, hidden=SEQ_HIDDEN_DIM,
            layers=SEQ_NUM_LAYERS, dropout=SEQ_DROPOUT, bidir=SEQ_BIDIRECTIONAL,
        ).to(device)
        seq_model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded BiLSTM from {SEQ_CKPT}")
    else:
        seq_model, best_f1 = train_sequence_model(train_ds, val_ds, vocab_size, device, criterion)
        torch.save(
            {
                "model_state_dict": seq_model.state_dict(),
                "token2id": token2id,
                "best_val_macro_f1": best_f1,
                "model": "Sequential BiLSTM (token-ID, order-only)",
                "excluded_augment_md5s": len(load_augment_exclusions()),
                "train_size": len(y_train),
                "hyperparams": {
                    "emb_dim": SEQ_EMB_DIM, "hidden": SEQ_HIDDEN_DIM, "layers": SEQ_NUM_LAYERS,
                    "dropout": SEQ_DROPOUT, "bidirectional": SEQ_BIDIRECTIONAL,
                    "max_len": SEQ_MAX_LEN, "lr": SEQ_LR, "weight_decay": SEQ_WEIGHT_DECAY,
                },
            },
            SEQ_CKPT,
        )
        print(f"[INFO] Saved BiLSTM checkpoint: {SEQ_CKPT}")

    # ---- build the eval tracks (attacked token sequences) ----
    seq_eval_loaders = {}
    attack_stats = {}
    for track in SEQ_EVAL_TRACKS:
        X_text, y_eval, info_df = build_text_features_from_manifest(custom_manifest_df, attack_name=track)
        attack_stats[track] = _stats_from_info_df(info_df)
        meta = info_df.to_dict("records")
        for m in meta:
            m["attack_name"] = track
        ds = SeqDataset(X_text, y_eval, token2id, meta=meta)
        seq_eval_loaders[track] = DataLoader(ds, batch_size=SEQ_BATCH_SIZE, shuffle=False, collate_fn=seq_collate)

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
            X_text, y_eval, info_df = build_text_features_from_manifest(custom_manifest_df, attack_name=track)
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

        tgnn_loaders = {track: make_tgnn_loader(track) for track in SEQ_EVAL_TRACKS}
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
        "description": "Sequential BiLSTM (order-only) vs existing TGNN under token-level attacks.",
        "output_dir": str(OUT_DIR),
        "checkpoints": {"sequential_bilstm": str(SEQ_CKPT), "existing_tgnn": str(EXISTING_TGNN_CKPT)},
        "seq_setup": {
            "model": "BiLSTM token-ID, attention pooling (order-only, no graph, no MPNet)",
            "train_size": len(y_train),
            "val_size": len(y_val),
            "excluded_augment_md5s": len(load_augment_exclusions()),
            "apply_malware_subsample_to_train": APPLY_MALWARE_SUBSAMPLE_TO_TRAIN,
            "token_vocab_size": vocab_size,
            "max_len": SEQ_MAX_LEN,
            "bidirectional": SEQ_BIDIRECTIONAL,
            "hidden_dim": SEQ_HIDDEN_DIM,
            "num_layers": SEQ_NUM_LAYERS,
        },
        "tracks": SEQ_EVAL_TRACKS,
        "tracks_note": "structure_attack intentionally excluded; attacks are matched to the sequence paradigm.",
        "attack_stats": attack_stats,
        "results": {},
    }
    report_rows = []

    # ---- BiLSTM ----
    seq_name = "Sequential BiLSTM"
    summary["results"][seq_name] = {}
    row = {"model": seq_name}
    clean_metrics = None
    for track in SEQ_EVAL_TRACKS:
        metrics, pred_rows = evaluate_sequence_model(seq_model, seq_eval_loaders[track], criterion, device, seq_name)
        print_metrics(metrics, f"{seq_name} | {track.upper()}")
        pred_path = OUT_DIR / f"sequential_bilstm_{track}_predictions.csv"
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
        summary["results"][seq_name][track] = entry
        row[track] = metric_cell(metrics)
    report_rows.append(row)

    # ---- TGNN ----
    if INCLUDE_TGNN:
        tg_name = "Existing TGNN"
        summary["results"][tg_name] = {}
        row = {"model": tg_name}
        clean_metrics = None
        for track in SEQ_EVAL_TRACKS:
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
