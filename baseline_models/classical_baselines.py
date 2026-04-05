#!/usr/bin/env python3
"""
baseline_models/classical_baselines.py
--------------------------------------
Tier 1 baselines: Random Forest + XGBoost

Input source:
    Existing graph .pkl files from ACTIVE_GRAPH_DIR/{train,val,test}

Features used:
    - API names
    - API order
    - API unigram / bigram TF-IDF only

Explicitly NOT used:
    - graph['api'].x
    - enriched edge_attr
    - memory outputs
    - TGNN embeddings
    - node counts
    - edge counts
    - timestamp summaries
    - any graph-derived flat summary features

Split logic (strict, fair evaluation):
    1. Prefer config.SPLIT_MANIFEST_PATH if defined
    2. Else prefer metadata split column if present
    3. Else fallback to folder-based split membership
    4. In all cases, enforce disjoint train/val/test

IMPORTANT:
    Labels are derived ONLY from metadata column: all_categories

EXPERIMENT:
    Keep only a fraction of OTHER MALWARE (label=2) in each split.
    Ransomware and benign are kept unchanged.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
    XGB_IMPORT_ERROR = None
except Exception as e:
    xgb = None
    HAS_XGBOOST = False
    XGB_IMPORT_ERROR = str(e)

import config


# ─────────────────────────────────────────────────────────────
# EXPERIMENT CONTROL
# ─────────────────────────────────────────────────────────────

RANDOM_STATE = 42
MALWARE_KEEP_FRAC = 0.80   # keep 50% of label=2 in train/val/test
APPLY_MALWARE_SUBSAMPLE_TO_TRAIN = True
APPLY_MALWARE_SUBSAMPLE_TO_VAL = True
APPLY_MALWARE_SUBSAMPLE_TO_TEST = True


# ─────────────────────────────────────────────────────────────
# 1. LABEL MAPPING — ONLY FROM all_categories
# ─────────────────────────────────────────────────────────────

def get_label(all_categories: str) -> int:
    cats = str(all_categories).lower()
    if "ransomware" in cats:
        return 0
    elif "benign" in cats:
        return 1
    else:
        return 2


CLASS_NAMES = ["Ransomware", "Benign", "Other Malware"]
LABEL_NAME_MAP = {0: "Ransomware", 1: "Benign", 2: "Other Malware"}


# ─────────────────────────────────────────────────────────────
# 2. SAFE GRAPH LOAD
# ─────────────────────────────────────────────────────────────

def load_graph(graph_path: Path):
    try:
        with open(graph_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load graph: {graph_path} ({e})")
        return None


# ─────────────────────────────────────────────────────────────
# 3. API SEQUENCE EXTRACTION ONLY
# ─────────────────────────────────────────────────────────────

def _normalize_api_name(x) -> str:
    return str(x).strip().lower().replace(" ", "_") if x is not None else "unknown_api"


def extract_api_sequence_from_graph(graph) -> str:
    """
    Build ordered API sequence string from symbolic API names only.
    Explicitly does NOT use graph['api'].x or any graph summary statistics.
    """
    try:
        if graph is None:
            return "empty"
        if not hasattr(graph, "node_types") or "api" not in graph.node_types:
            return "empty"

        api_store = graph["api"]

        api_names = None
        for attr in ["api_names", "names", "api_name"]:
            if hasattr(api_store, attr):
                api_names = getattr(api_store, attr)
                break

        if api_names is None:
            return "empty"

        api_names = list(api_names)
        if len(api_names) == 0:
            return "empty"

        if hasattr(api_store, "timestamps") and api_store.timestamps is not None:
            try:
                ts = api_store.timestamps
                ts = ts.detach().cpu().numpy() if hasattr(ts, "detach") else np.asarray(ts)
                if len(ts) == len(api_names):
                    api_names = [name for _, name in sorted(zip(ts.tolist(), api_names), key=lambda x: x[0])]
            except Exception:
                pass

        api_names = [_normalize_api_name(a) for a in api_names]
        return " ".join(api_names) if api_names else "empty"

    except Exception:
        return "empty"


# ─────────────────────────────────────────────────────────────
# 4. SPLIT LOADING (STRICT / FAIR)
# ─────────────────────────────────────────────────────────────

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
        msg = (
            "Split leakage detected. train/val/test are not disjoint.\n"
            f"train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}.\n"
            "Fix the split source before trusting baseline results."
        )
        raise RuntimeError(msg)


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


# ─────────────────────────────────────────────────────────────
# 5. LABEL AUDIT HELPERS
# ─────────────────────────────────────────────────────────────

def print_label_counts(name: str, series: pd.Series):
    counts = series.apply(get_label).value_counts().sort_index()
    print(f"[AUDIT] {name} label counts from all_categories")
    for cls_id in [0, 1, 2]:
        print(f"        {LABEL_NAME_MAP[cls_id]}: {int(counts.get(cls_id, 0))}")


def subsample_other_malware(df_split: pd.DataFrame, keep_frac: float, split_name: str) -> pd.DataFrame:
    """
    Keep all ransomware (0) and benign (1).
    Randomly keep only a fraction of other malware (2).
    """
    if not (0.0 < keep_frac <= 1.0):
        raise ValueError(f"keep_frac must be in (0,1], got {keep_frac}")

    df_split = df_split.copy()
    df_split["_label"] = df_split["all_categories"].apply(get_label)

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
    return df_out.drop(columns=["_label"])


# ─────────────────────────────────────────────────────────────
# 6. DATASET BUILDING — TEXT ONLY
# ─────────────────────────────────────────────────────────────

def build_text_features(metadata_df: pd.DataFrame, graph_dir: Path):
    api_sequences = []
    labels = []
    md5s = []

    if "all_categories" not in metadata_df.columns:
        raise KeyError("Expected column 'all_categories' not found in metadata.")

    missing_graphs = 0
    failed_graphs = 0

    for _, row in metadata_df.iterrows():
        md5 = str(row["md5"]).strip().lower()
        graph_path = graph_dir / f"{md5}.pkl"

        if not graph_path.exists():
            missing_graphs += 1
            continue

        graph = load_graph(graph_path)
        if graph is None:
            failed_graphs += 1
            continue

        seq = extract_api_sequence_from_graph(graph)
        label = get_label(row["all_categories"])

        api_sequences.append(seq)
        labels.append(label)
        md5s.append(md5)

    print(f"[INFO] Graph scan {graph_dir.name}: kept={len(labels)} missing={missing_graphs} failed={failed_graphs}")
    return api_sequences, labels, md5s


# ─────────────────────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, model_name: str):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    threats_missed = int(cm[0, 1] + cm[2, 1])

    print(f"\n{'='*72}")
    print(f"MODEL: {model_name}")
    print(f"{'='*72}")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Macro-F1         : {macro_f1:.4f}")
    print(f"Threats Missed   : {threats_missed}")
    print(f"{'-'*72}")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls:<16} P={p[i]:.3f}  R={r[i]:.3f}  F1={f[i]:.3f}")
    print(f"{'-'*72}")
    print("Confusion Matrix (rows=true, cols=pred)")
    print(f"{'':<16}{'Ransom':>10}{'Benign':>10}{'Malware':>10}")
    for i, cls in enumerate(CLASS_NAMES):
        row = "".join(f"{cm[i, j]:>10}" for j in range(3))
        print(f"{cls:<16}{row}")
    print(f"{'='*72}\n")

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "threats_missed": threats_missed,
        "ransom_P": round(p[0], 3),
        "ransom_R": round(r[0], 3),
        "ransom_F1": round(f[0], 3),
        "benign_P": round(p[1], 3),
        "benign_R": round(r[1], 3),
        "benign_F1": round(f[1], 3),
        "malware_P": round(p[2], 3),
        "malware_R": round(r[2], 3),
        "malware_F1": round(f[2], 3),
    }


# ─────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    metadata_path = Path(config.METADATA_PATH)
    split_root = Path(config.ACTIVE_GRAPH_DIR)

    out_dir = Path("baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)
    df["_md5"] = df["md5"].astype(str).str.strip().str.lower()

    if "all_categories" not in df.columns:
        raise KeyError("Expected metadata column 'all_categories' not found.")

    print_label_counts("FULL METADATA", df["all_categories"])

    train_md5s, val_md5s, test_md5s = load_split_sets(df, split_root)

    df_train = df[df["_md5"].isin(train_md5s)].copy()
    df_val = df[df["_md5"].isin(val_md5s)].copy()
    df_test = df[df["_md5"].isin(test_md5s)].copy()

    print(f"[INFO] Split sizes from metadata rows before malware subsampling:")
    print(f"       train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

    print_label_counts("TRAIN BEFORE", df_train["all_categories"])
    print_label_counts("VAL BEFORE", df_val["all_categories"])
    print_label_counts("TEST BEFORE", df_test["all_categories"])

    if APPLY_MALWARE_SUBSAMPLE_TO_TRAIN:
        df_train = subsample_other_malware(df_train, MALWARE_KEEP_FRAC, "TRAIN")
    if APPLY_MALWARE_SUBSAMPLE_TO_VAL:
        df_val = subsample_other_malware(df_val, MALWARE_KEEP_FRAC, "VAL")
    if APPLY_MALWARE_SUBSAMPLE_TO_TEST:
        df_test = subsample_other_malware(df_test, MALWARE_KEEP_FRAC, "TEST")

    print(f"[INFO] Split sizes from metadata rows after malware subsampling:")
    print(f"       train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

    print_label_counts("TRAIN AFTER", df_train["all_categories"])
    print_label_counts("VAL AFTER", df_val["all_categories"])
    print_label_counts("TEST AFTER", df_test["all_categories"])

    train_dir = split_root / "train"
    val_dir = split_root / "val"
    test_dir = split_root / "test"

    print("[INFO] Extracting API-sequence-only features from graph PKLs...")
    X_train_text, y_train, train_ids = build_text_features(df_train, train_dir)
    X_val_text, y_val, val_ids = build_text_features(df_val, val_dir)
    X_test_text, y_test, test_ids = build_text_features(df_test, test_dir)

    print(f"[INFO] Loaded usable samples:")
    print(f"       train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    print("[INFO] TF-IDF vectorising API sequences...")
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        min_df=2,
        sublinear_tf=True,
    )

    X_train = vec.fit_transform(X_train_text)
    X_val = vec.transform(X_val_text)
    X_test = vec.transform(X_test_text)

    print(f"[INFO] Final feature dims:")
    print(f"       train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    results = []

    print("[INFO] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)

    results.append(evaluate(y_train, rf.predict(X_train), "Random Forest (train)"))
    results.append(evaluate(y_val, rf.predict(X_val), "Random Forest (val)"))
    results.append(evaluate(y_test, rf.predict(X_test), "Random Forest (test)"))

    if HAS_XGBOOST:
        print("[INFO] Training XGBoost...")
        label_counts = Counter(y_train)
        total = len(y_train)
        sample_weights = np.array([total / (3 * label_counts[y]) for y in y_train], dtype=np.float32)

        xgb_clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            early_stopping_rounds=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        xgb_clf.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        results.append(evaluate(y_train, xgb_clf.predict(X_train), "XGBoost (train)"))
        results.append(evaluate(y_val, xgb_clf.predict(X_val), "XGBoost (val)"))
        results.append(evaluate(y_test, xgb_clf.predict(X_test), "XGBoost (test)"))
    else:
        print(f"[WARN] Skipping XGBoost because it is unavailable: {XGB_IMPORT_ERROR}")

    results_df = pd.DataFrame(results)
    results_path = out_dir / "classical_results.csv"
    results_df.to_csv(results_path, index=False)

    feature_info = {
        "metadata_path": str(metadata_path),
        "active_graph_dir": str(split_root),
        "label_column_used": "all_categories",
        "feature_family": "api_tfidf_only",
        "tfidf_vocab_size": int(len(vec.vocabulary_)),
        "random_state": RANDOM_STATE,
        "malware_keep_frac": MALWARE_KEEP_FRAC,
        "apply_malware_subsample_to_train": APPLY_MALWARE_SUBSAMPLE_TO_TRAIN,
        "apply_malware_subsample_to_val": APPLY_MALWARE_SUBSAMPLE_TO_VAL,
        "apply_malware_subsample_to_test": APPLY_MALWARE_SUBSAMPLE_TO_TEST,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "split_manifest_path": str(getattr(config, "SPLIT_MANIFEST_PATH", "")),
        "xgboost_available": bool(HAS_XGBOOST),
    }
    with open(out_dir / "classical_feature_info.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=2)

    print(f"[INFO] Results saved to: {results_path}")
    print(f"[INFO] Feature schema saved to: {out_dir / 'classical_feature_info.json'}")
    print()
    cols_to_show = ["model", "macro_f1", "ransom_R", "benign_R", "malware_R", "threats_missed", "accuracy"]
    print(results_df[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()