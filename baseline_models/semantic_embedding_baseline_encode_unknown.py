#!/usr/bin/env python3
"""
baseline_models/semantic_embedding_baseline_encode_unknown.py
-------------------------------------------------------------

Semantic-embedding baseline -- ENCODE-UNKNOWN variant.

This is the SAME baseline as semantic_embedding_baseline.py (mean-pooled MPNet
embeddings + trained MLP, semantics-only), with ONE difference in how unseen /
renamed API names are represented:

    * semantic_embedding_baseline.py            -> OOV tokens are DROPPED
                                                   (cached dict only).
    * THIS file (encode-unknown)                -> OOV tokens are RE-ENCODED live
                                                   with sentence-transformers/
                                                   all-mpnet-base-v2, EXACTLY as the
                                                   real pipeline does
                                                   (pipeline_v2/enrich_graphs_v2.py,
                                                   ENCODE_UNKNOWN_APIS_BY_NAME=True).

Why both: the production node enrichment never drops an unseen API -- it embeds
the name so it lands in the same semantic space. So this variant is the FAITHFUL
representation of the TGNN's own semantic branch; the drop variant is the
strict/pessimistic one. Running both lets us see the gap.

Everything else (model, training regime, eval set, attacks, TGNN comparison,
output artifacts) is identical to the drop variant. Outputs go to a separate
directory so the two can be compared side by side.

Models
    1. Semantic-embedding classifier (mean-pooled MPNet vectors -> MLP). Trained fresh.
    2. Existing TGNN checkpoint (loaded, NOT trained).

Tracks  (only paradigm-relevant attacks)
    1. clean
    2. unigram_attack   -> API rename / substitution
    (bigram/structure excluded: pooling is order-agnostic and there is no graph.)

Requires: sentence-transformers with all-mpnet-base-v2 (needed to encode unknown
names). The TGNN comparison additionally needs CUDA (Mamba). Set INCLUDE_TGNN=False
to run the semantic-baseline tracks without the TGNN.

Outputs (results/semantic_baseline_encode_unknown/)
    semantic_embedding_<track>_predictions.csv
    existing_tgnn_<track>_predictions.csv        (if INCLUDE_TGNN)
    comparison_summary.json
    comparison_report.md
    models/semantic_embedding_encode_unknown_best.pt
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

from baseline_models.prove_classical_model_attacks import (
    build_custom_eval_manifest,
    build_text_features_from_metadata,
    build_text_features_from_manifest,
    load_split_sets,
    subsample_other_malware,
    get_all_train_exclusions,
    compute_metrics,
    compute_threat_vs_benign,
    compute_drop_summary,
    label_to_name,
    print_metrics,
    _stats_from_info_df,
    _normalize_api_name,
    CustomManifestDataset,
    custom_eval_collate,
    run_tgnn_eval,
    FocalLoss,
)


# ============================================================
# Local controls
# ============================================================

RANDOM_STATE = 42

# BiLSTM (reported model), val+test-only run -> write to a SEPARATE folder so the
# earlier Mamba and zero-day-inclusive runs are all preserved.
OUT_DIR = Path(config.RESULTS_DIR) / "semantic_baseline_encode_unknown_bilstm_valtest"
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---- Eval set = held-out VAL + TEST only (no zero-day) -----------------------
# For the §5.6 baseline comparison the eval set must be a clean held-out mix; the
# zero-day unseen malware belongs to the separate generalisation experiment (§5.8).
# Scope these overrides to THIS script (config.py on disk is untouched, so the
# other baselines keep the default zero-day-inclusive manifest). Redirect the
# manifest cache into OUT_DIR so the shared manifest CSV is not clobbered.
config.CUSTOM_EVAL_INCLUDE_ALL_NEW_MALWARE = False   # drop the 170 zero-day rows
config.CUSTOM_EVAL_REBUILD_MANIFEST = True           # force a fresh val+test build
config.CUSTOM_EVAL_MANIFEST_PATH = OUT_DIR / "custom_eval_manifest_valtest.csv"

# Reuse the already-trained semantic MLP from the original encode-unknown run so
# the semantic side is byte-identical and ONLY the eval set / TGNN reference change.
SEM_CKPT = (Path(config.RESULTS_DIR) / "semantic_baseline_encode_unknown"
            / "models" / "semantic_embedding_encode_unknown_best.pt")
# Reported model is the BiLSTM checkpoint best_model_1.pt (not the Mamba best_model.pt).
EXISTING_TGNN_CKPT = Path(config.RESULTS_DIR) / "checkpoints" / "best_model_1.pt"

API_EMB_PATH = PROJECT_ROOT / "cache" / "api_embeddings.pkl"

FORCE_RETRAIN_SEM = False   # reuse saved checkpoint if it exists
INCLUDE_TGNN = True

# THE difference vs the drop variant: re-encode unknown API names live (matches
# pipeline_v2 ENCODE_UNKNOWN_APIS_BY_NAME=True). Set False to fall back to zeros.
ENCODE_UNKNOWN_BY_NAME = True

# clean = no attack; unigram_attack = Tier-1 meaning-preserving rename (+padding);
# random_rename_attack = Tier-2 out-of-context rename (behaviourally-unrelated names).
SEM_EVAL_TRACKS = ["clean", "unigram_attack", "random_rename_attack"]

# --- MLP head hyperparameters (identical to the drop variant) ---
SEM_EPOCHS = 60
SEM_PATIENCE = 10
SEM_LR = 1e-3
SEM_WEIGHT_DECAY = 1e-4
SEM_HIDDEN_DIM = 256
SEM_DROPOUT = 0.3
SEM_BATCH_SIZE = 64
SEM_LOG_EVERY = 25

# --- training-set composition (match the other baselines) ---
SEM_TRAIN_FRAC = 0.50
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
# Semantic vectorizer: cached lookup + LIVE encode of unknown names
# (faithful to pipeline_v2/enrich_graphs_v2.py)
# ============================================================

class SemanticVectorizer:
    """Maps a trace's API tokens to a mean-pooled MPNet vector.
    Known tokens -> cached api_embeddings.pkl vector.
    Unknown tokens -> live MPNet encode (ENCODE_UNKNOWN_BY_NAME=True), else zeros.
    Call fit_tokens(sequences) before pool() so every token has a vector."""

    def __init__(self, encode_unknown: bool = True):
        if not API_EMB_PATH.exists():
            raise FileNotFoundError(f"API embeddings not found: {API_EMB_PATH}")
        raw = pickle.load(open(API_EMB_PATH, "rb"))
        self.lookup: Dict[str, np.ndarray] = {
            _normalize_api_name(k): np.asarray(v, dtype=np.float32) for k, v in raw.items()
        }
        self.dim = len(next(iter(self.lookup.values())))
        self.n_cached = len(self.lookup)
        self.encode_unknown = encode_unknown
        self.embedder = None
        self.n_encoded = 0
        print(f"[INFO] loaded {self.n_cached} cached API embeddings (dim={self.dim}) "
              f"from {API_EMB_PATH.name} | encode_unknown={self.encode_unknown}")

    def _ensure_embedder(self):
        if self.embedder is None:
            from pipeline_v2.enrich_graphs_v2 import APIEmbedder
            self.embedder = APIEmbedder(batch_size=128)
            print("[INFO] loaded all-mpnet-base-v2 for live encoding of unknown API names")

    def fit_tokens(self, sequences: List[str], tag: str = ""):
        """Ensure every token across `sequences` has a vector (encode new unknowns)."""
        uniq = set()
        for s in sequences:
            uniq.update(s.split())
        unknown = sorted(t for t in uniq if t not in self.lookup)
        if not unknown:
            return
        if self.encode_unknown:
            self._ensure_embedder()
            enc = np.asarray(self.embedder.encode(unknown), dtype=np.float32)
            for t, v in zip(unknown, enc):
                self.lookup[t] = v
            self.n_encoded += len(unknown)
            print(f"[INFO] [{tag}] live-encoded {len(unknown)} unknown API names "
                  f"(cumulative encoded={self.n_encoded})")
        else:
            z = np.zeros(self.dim, dtype=np.float32)
            for t in unknown:
                self.lookup[t] = z
            print(f"[INFO] [{tag}] zeroed {len(unknown)} unknown API names (encode_unknown=False)")

    def pool(self, seq: str) -> np.ndarray:
        vecs = [self.lookup[t] for t in seq.split() if t in self.lookup]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)


# ============================================================
# Dataset / model (identical to the drop variant)
# ============================================================

class SemDataset(Dataset):
    def __init__(self, sequences: List[str], labels: List[int], vectorizer: SemanticVectorizer,
                 meta: List[dict] | None = None):
        self.x = np.stack([vectorizer.pool(s) for s in sequences]).astype(np.float32)
        self.labels = [int(y) for y in labels]
        self.meta = meta if meta is not None else [{} for _ in labels]
        self.zero_frac = float((np.abs(self.x).sum(axis=1) == 0).mean())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), int(self.labels[idx]), self.meta[idx]


def sem_collate(batch):
    xs, labels, metas = zip(*batch)
    return torch.stack(xs), torch.tensor(labels, dtype=torch.long), list(metas)


class SemanticMLP(nn.Module):
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
# Train / evaluate (identical to the drop variant)
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
    lines = ["# Semantic-embedding baseline (encode-unknown) vs Existing TGNN", ""]
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
    print(f"[INFO] Encode unknown   : {ENCODE_UNKNOWN_BY_NAME}  (matches pipeline_v2 ENCODE_UNKNOWN_APIS_BY_NAME)")
    print(f"[INFO] Include TGNN     : {INCLUDE_TGNN}")

    vectorizer = SemanticVectorizer(encode_unknown=ENCODE_UNKNOWN_BY_NAME)
    emb_dim = vectorizer.dim

    if getattr(config, "CUSTOM_EVAL_REBUILD_MANIFEST", True) or not Path(config.CUSTOM_EVAL_MANIFEST_PATH).exists():
        custom_manifest_df = build_custom_eval_manifest()
    else:
        custom_manifest_df = pd.read_csv(config.CUSTOM_EVAL_MANIFEST_PATH)
        print(f"[INFO] Using existing custom eval manifest: {config.CUSTOM_EVAL_MANIFEST_PATH}")

    custom_manifest_df = filter_eval_manifest(custom_manifest_df)

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

    # encode any unknown tokens BEFORE pooling (faithful to the pipeline)
    vectorizer.fit_tokens(X_train_text, tag="train")
    vectorizer.fit_tokens(X_val_text, tag="val")

    train_ds = SemDataset(X_train_text, y_train, vectorizer)
    val_ds = SemDataset(X_val_text, y_val, vectorizer)
    print(f"[INFO] train={len(y_train)} {dict(Counter(y_train))} (all-zero frac={train_ds.zero_frac:.3f}) | "
          f"val={len(y_val)} {dict(Counter(y_val))} (all-zero frac={val_ds.zero_frac:.3f})")

    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)

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
                "model": "Semantic-embedding encode-unknown (mean-pooled MPNet, semantics-only)",
                "emb_dim": emb_dim,
                "pooling": "mean",
                "oov_policy": "encode unknown by name (live MPNet) -- matches pipeline_v2",
                "encode_unknown_by_name": ENCODE_UNKNOWN_BY_NAME,
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
        vectorizer.fit_tokens(X_text, tag=f"eval:{track}")     # encode renamed/unknown tokens live
        meta = info_df.to_dict("records")
        for m in meta:
            m["attack_name"] = track
        ds = SemDataset(X_text, y_eval, vectorizer, meta=meta)
        print(f"[INFO] eval track '{track}': n={len(ds)} all-zero frac={ds.zero_frac:.3f}")
        sem_eval_loaders[track] = DataLoader(ds, batch_size=SEM_BATCH_SIZE, shuffle=False, collate_fn=sem_collate)

    # ---- optional TGNN graph eval on the same tracks ----
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
        "description": "Semantic-embedding baseline (mean-pooled MPNet, ENCODE-UNKNOWN variant) vs existing TGNN under rename attack.",
        "output_dir": str(OUT_DIR),
        "checkpoints": {"semantic_embedding": str(SEM_CKPT), "existing_tgnn": str(EXISTING_TGNN_CKPT)},
        "sem_setup": {
            "model": "mean-pooled MPNet embeddings -> MLP (semantics-only)",
            "embedding_source": str(API_EMB_PATH),
            "embedding_dim": emb_dim,
            "pooling": "mean",
            "oov_policy": "encode unknown by name (live MPNet) -- matches pipeline_v2 ENCODE_UNKNOWN_APIS_BY_NAME",
            "encode_unknown_by_name": ENCODE_UNKNOWN_BY_NAME,
            "live_encoded_unknown_count": vectorizer.n_encoded,
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

    sem_name = "Semantic-embedding (encode-unknown)"
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

    summary_path = OUT_DIR / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_report(report_rows, OUT_DIR / "comparison_report.md")
    print(f"\n[INFO] Saved summary JSON : {summary_path}")
    print(f"[INFO] Saved report      : {OUT_DIR / 'comparison_report.md'}")
    print(f"[INFO] Outputs in        : {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
