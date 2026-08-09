#!/usr/bin/env python3
"""
baseline_models/prove_classical_model_attacks.py
------------------------------------------------
Evaluate classical TF-IDF baselines and an already-trained TGNN checkpoint
under FOUR evaluation tracks on the same custom eval set:

1) clean
2) unigram_attack
   - lexical API-name substitution
   - optional heavy frequency dilution / benign padding
3) bigram_attack
   - benign interleaving or full shuffle
4) combined_attack
   - unigram_attack + bigram_attack together

Why this script exists
----------------------
- Unigram TF-IDF mainly depends on token identity / token frequency.
- Bigram TF-IDF mainly depends on local adjacent order.
- TGNN uses graph structure + temporal/message passing, so we compare whether it
  degrades differently under the same attacked graphs.

Important note about the unigram attack
---------------------------------------
This unigram attack is deliberately lexical:
- it renames API strings to alternate / alias-like / OOV-looking names
- it can also add large benign-looking padding
This is NOT a perfect "semantic-preserving" program transformation.
It is an adversarial lexical stress test against token-count models.

Models compared
---------------
Classical:
    - Random Forest
    - XGBoost
TF-IDF variants:
    - unigram
    - unigram + bigram
    - bigram

TGNN:
    - loaded from existing checkpoint
    - NO training here

Outputs
-------
robustness_compare_dual/
    comparison_summary.json

    random_forest_unigram_clean_predictions.csv
    random_forest_unigram_unigram_attack_predictions.csv
    random_forest_unigram_bigram_attack_predictions.csv
    random_forest_unigram_combined_attack_predictions.csv
    ...

    xgboost_bigram_clean_predictions.csv
    xgboost_bigram_unigram_attack_predictions.csv
    xgboost_bigram_bigram_attack_predictions.csv
    xgboost_bigram_combined_attack_predictions.csv

    tgnn_clean_predictions.csv
    tgnn_unigram_attack_predictions.csv
    tgnn_bigram_attack_predictions.csv
    tgnn_combined_attack_predictions.csv
"""

from __future__ import annotations

import copy
import json
import pickle
import random
import re
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset

try:
    import xgboost as xgb

    HAS_XGBOOST = True
    XGB_IMPORT_ERROR = None
except Exception as e:
    xgb = None
    HAS_XGBOOST = False
    XGB_IMPORT_ERROR = str(e)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config
from graph_construction.pytorch_dataset import hetero_collate, RansomwareGraphDataset
from graph_construction.vocabulary import load_vocab
from models.memory_utils import get_api_indices
from models.tgnn_model import create_hetero_tgnn
from utils.schema_manager import ensure_active_schema
from utils.schema_utils import normalize_graph_node_features
from utils.schema_validator import validate_batch_or_raise, validate_graph_or_raise


# ============================================================
# Experiment controls
# ============================================================

RANDOM_STATE = 42

# Classical baseline train controls
MALWARE_KEEP_FRAC_TRAIN = 0.80
APPLY_MALWARE_SUBSAMPLE_TO_TRAIN = True

# Fairness cap for huge API sequences
API_SEQ_HARD_LIMIT = 100_000
API_SEQ_REDUCED_TO = 30_000

# Optional train exclusions
OPTIONAL_EXCLUSION_FILE = getattr(config, "BASELINE_TRAIN_EXCLUSION_FILE", None)
EXCLUDED_TRAIN_MD5S: Set[str] = set()

ATTACK_BASE_SEED = 1337

# ============================================================
# UNIGRAM ATTACK CONTROLS
# ============================================================

# 1) lexical renaming
UNIGRAM_RENAME_FRACTION = 1.00
UNIGRAM_KEEP_CORE_APIS = 0
UNIGRAM_USE_SUFFIX_VARIANTS = True

# turn padding off
UNIGRAM_ENABLE_PADDING = True
UNIGRAM_ATTACK_PADDING_RATIO = 2.0
UNIGRAM_ATTACK_MIN_ADD = 500
UNIGRAM_ATTACK_MAX_ADD = 5000

# ============================================================
# BIGRAM ATTACK CONTROLS
# ============================================================

BIGRAM_ATTACK_MODE = "benign_interleaving"    # "benign_interleaving" or "full_shuffle"

FULL_SHUFFLE_PRESERVE_FIRST = False
FULL_SHUFFLE_PRESERVE_LAST = False

INTERLEAVE_EVERY_K = 1
INTERLEAVE_RATIO = 1.00
INTERLEAVE_MAX_ADD = 30000

# ============================================================
# COMBINED ATTACK CONTROLS
# ============================================================

COMBINED_APPLY_UNIGRAM_FIRST = True

# ============================================================
# Pools / aliases
# ============================================================

PADDING_API_POOL = [
    "sleep",
    "gettickcount",
    "closehandle",
    "getsystemtimeasfiletime",
    "queryperformancecounter",
    "getcurrentprocessid",
    "getcurrentthreadid",
    "getmodulehandlew",
    "getprocaddress",
    "virtualquery",
    "heapalloc",
    "heapfree",
    "entercriticalsection",
    "leavecriticalsection",
    "waitforsingleobject",
    "setevent",
    "resetevent",
    "createfilew",
    "readfile",
    "writefile",
    "getversion",
    "getlasterror",
    "initializecriticalsection",
    "deletecriticalsection",
    "gettemppathw",
    "getsystemdirectoryw",
]

# Alias-like substitutions for lexical attack.
# The goal is exact-token disruption, not perfect semantic preservation.
API_RENAME_MAP: Dict[str, List[str]] = {
    "createfilew": ["openfilew", "create_file_w", "open_handle_filew"],
    "readfile": ["read_file_bytes", "fetchfile", "read_file_buffer"],
    "writefile": ["write_file_bytes", "storefile", "write_file_buffer"],
    "closehandle": ["releasehandle", "close_handle_obj", "disposehandle"],
    "getprocaddress": ["resolveprocaddr", "lookup_proc_address", "findprocaddress"],
    "getmodulehandlew": ["resolve_module_w", "get_mod_handle_w", "modulehandle_lookup"],
    "virtualalloc": ["reservevirtualmem", "virtual_alloc_region", "allocvirtmem"],
    "virtualprotect": ["setvirtualprotect", "change_virtual_protect", "protectvirtmem"],
    "virtualfree": ["releasevirtualmem", "virtual_free_region", "freevirtmem"],
    "virtualquery": ["queryvirtualmem", "virtual_query_region", "virtmemquery"],
    "loadlibraryw": ["load_library_w", "importlibraryw", "maplibraryw"],
    "createremotethread": ["spawnremotethread", "create_remote_thread_x", "injectthreadremote"],
    "openprocess": ["open_target_process", "acquireprocesshandle", "procopen"],
    "terminateprocess": ["killprocesshandle", "stopprocessnow", "proc_terminate_x"],
    "regopenkeyexw": ["registry_open_key_w", "openregkeyexw", "regkeyopenxw"],
    "regsetvalueexw": ["registry_set_value_w", "setregvalueexw", "regvaluesetxw"],
    "regqueryvalueexw": ["registry_query_value_w", "queryregvalueexw", "regvaluequeryxw"],
    "createprocessw": ["spawnprocessw", "create_process_wide", "proccreatew"],
    "winexec": ["execute_win_process", "launchwinexec", "winexec_call"],
    "shellexecutew": ["shell_execute_wide", "startexternalshellw", "execshellw"],
    "internetopenw": ["openinternetw", "netsessionopenw", "internet_open_handle_w"],
    "internetconnectw": ["connectinternetw", "netsessionconnectw", "internet_connect_handle_w"],
    "httpopenrequestw": ["openhttprequestw", "http_request_open_w", "httprequestopenwide"],
    "httpsendrequestw": ["sendhttprequestw", "http_request_send_w", "httprequestsendwide"],
    "recv": ["socket_receive_bytes", "net_recv_buffer", "sockrecvx"],
    "send": ["socket_send_bytes", "net_send_buffer", "socksendx"],
    "connect": ["socket_connect_target", "net_connect_x", "sockconnectx"],
    "bind": ["socket_bind_local", "net_bind_x", "sockbindx"],
    "sleep": ["sleepdelay", "pauseexecution", "threadsleepx"],
    "waitforsingleobject": ["wait_single_object_x", "objectwaitsingle", "syncwaitsingle"],
    "setevent": ["signal_event_obj", "eventsetx", "raiseeventsignal"],
    "resetevent": ["clear_event_obj", "eventresetx", "reseteventsignal"],
    "heapalloc": ["heap_allocate_block", "allocheapmem", "heapallocx"],
    "heapfree": ["heap_release_block", "freeheapmem", "heapfreex"],
    "gettickcount": ["querytickcount", "getticksystem", "tickcountreadx"],
    "getcurrentprocessid": ["querycurrentpid", "getprocidcurrent", "currentprocessidx"],
    "getcurrentthreadid": ["querycurrenttid", "getthreadidcurrent", "currentthreadidx"],
}

TFIDF_VARIANTS: List[Tuple[str, Tuple[int, int]]] = [
    ("unigram", (1, 1)),
    ("unigram_bigram", (1, 2)),
    ("bigram", (2, 2)),
]

EVAL_TRACKS = ["clean", "unigram_attack", "bigram_attack", "combined_attack"]

OUT_DIR = Path("robustness_compare_dual")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Ransomware", "Benign", "Other Malware"]
LABEL_NAME_MAP = {0: "Ransomware", 1: "Benign", 2: "Other Malware"}


# ============================================================
# Generic helpers
# ============================================================

def normalize_md5_like(value: str) -> str:
    value = str(value).strip().lower()
    return value.replace(".json", "").replace(".pkl", "")


def parse_exclusion_lines(lines: List[str]) -> Set[str]:
    ids: Set[str] = set()
    md5_pattern = re.compile(r"\b([0-9a-f]{32})\b", re.IGNORECASE)
    for line in lines:
        for match in md5_pattern.findall(str(line).strip().lower()):
            ids.add(normalize_md5_like(match))
    return ids


def load_optional_exclusion_file(path_value) -> Set[str]:
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


def get_all_train_exclusions() -> Set[str]:
    ids = {normalize_md5_like(x) for x in EXCLUDED_TRAIN_MD5S}
    ids |= load_optional_exclusion_file(OPTIONAL_EXCLUSION_FILE)
    return ids


def get_label(all_categories: str) -> int:
    cats = str(all_categories).lower()
    if "ransomware" in cats:
        return 0
    if "benign" in cats:
        return 1
    return 2


def label_to_name(label: int) -> str:
    return LABEL_NAME_MAP[int(label)]


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
    tp = confusion_mat[0, 0] + confusion_mat[0, 2] + confusion_mat[2, 0] + confusion_mat[2, 2]
    fn = confusion_mat[0, 1] + confusion_mat[2, 1]
    tn = confusion_mat[1, 1]
    fp = confusion_mat[1, 0] + confusion_mat[1, 2]

    threat_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    threat_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    threat_f1 = (
        2 * threat_precision * threat_recall / (threat_precision + threat_recall)
        if (threat_precision + threat_recall) > 0 else 0.0
    )
    benign_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    benign_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    benign_f1 = (
        2 * benign_precision * benign_recall / (benign_precision + benign_recall)
        if (benign_precision + benign_recall) > 0 else 0.0
    )
    acc_2class = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "TP": int(tp),
        "FN": int(fn),
        "TN": int(tn),
        "FP": int(fp),
        "threat_recall": float(threat_recall),
        "threat_precision": float(threat_precision),
        "threat_f1": float(threat_f1),
        "benign_recall": float(benign_recall),
        "benign_precision": float(benign_precision),
        "benign_f1": float(benign_f1),
        "accuracy_2class": float(acc_2class),
        "threats_missed": int(fn),
    }


def compute_drop_summary(clean_metrics: dict, attacked_metrics: dict) -> dict:
    clean_tvb = compute_threat_vs_benign(np.array(clean_metrics["confusion_matrix"]))
    attacked_tvb = compute_threat_vs_benign(np.array(attacked_metrics["confusion_matrix"]))

    return {
        "accuracy_drop": float(clean_metrics["accuracy"] - attacked_metrics["accuracy"]),
        "macro_f1_drop": float(clean_metrics["macro_f1"] - attacked_metrics["macro_f1"]),
        "weighted_f1_drop": float(clean_metrics["weighted_f1"] - attacked_metrics["weighted_f1"]),
        "ransomware_recall_drop": float(
            clean_metrics["per_class_recall"][0] - attacked_metrics["per_class_recall"][0]
        ),
        "benign_recall_drop": float(
            clean_metrics["per_class_recall"][1] - attacked_metrics["per_class_recall"][1]
        ),
        "other_malware_recall_drop": float(
            clean_metrics["per_class_recall"][2] - attacked_metrics["per_class_recall"][2]
        ),
        "threats_missed_increase": int(attacked_tvb["threats_missed"] - clean_tvb["threats_missed"]),
    }


def print_metrics(metrics, title: str):
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


# ============================================================
# Manifest / split helpers
# ============================================================

def load_metadata_labels() -> Dict[str, int]:
    df = pd.read_csv(config.METADATA_PATH)
    if "md5" not in df.columns:
        raise KeyError("Expected 'md5' column in metadata CSV.")
    if "all_categories" not in df.columns:
        raise KeyError("Expected 'all_categories' column in metadata CSV.")
    df["_md5"] = df["md5"].astype(str).str.strip().str.lower()
    return {row["_md5"]: get_label(row["all_categories"]) for _, row in df.iterrows()}


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
            rows.append(
                {
                    "sample_id": md5,
                    "graph_path": str(graph_path.resolve()),
                    "label": label,
                    "class_name": label_to_name(label),
                    "origin": f"old_{split}",
                    "is_new_unseen": False,
                }
            )
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
        rows.append(
            {
                "sample_id": sample_id,
                "graph_path": str(graph_path.resolve()),
                "label": int(config.CUSTOM_EVAL_NEW_MALWARE_LABEL),
                "class_name": label_to_name(int(config.CUSTOM_EVAL_NEW_MALWARE_LABEL)),
                "origin": "new_unseen",
                "is_new_unseen": True,
            }
        )

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


def build_custom_eval_manifest() -> pd.DataFrame:
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

    print("\n[INFO] Custom eval manifest created:")
    print(f"       path   : {config.CUSTOM_EVAL_MANIFEST_PATH}")
    print(f"       total  : {len(df)}")
    print(f"       benign : {(df['label'] == 1).sum()}")
    print(f"       ransom : {(df['label'] == 0).sum()}")
    print(f"       other  : {(df['label'] == 2).sum()}")
    print(f"       origin : {dict(Counter(df['origin'].tolist()))}")
    return df


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


def load_graph(graph_path: Path):
    try:
        with open(graph_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load graph: {graph_path} ({e})")
        return None


# ============================================================
# Token utilities / stats
# ============================================================

def reduce_api_sequence(api_names: list[str], hard_limit: int, reduced_to: int) -> list[str]:
    n = len(api_names)
    if n <= hard_limit:
        return api_names
    if reduced_to <= 0:
        return []
    if reduced_to >= n:
        return api_names
    idx = np.linspace(0, n - 1, num=reduced_to, dtype=int)
    return [api_names[i] for i in idx]


def _normalize_api_name(x) -> str:
    return str(x).strip().lower().replace(" ", "_") if x is not None else "unknown_api"


def extract_api_token_list_from_graph(graph) -> List[str]:
    try:
        if graph is None:
            return []
        if not hasattr(graph, "node_types") or "api" not in graph.node_types:
            return []

        api_store = graph["api"]
        api_names = None
        for attr in ["api_names", "names", "api_name"]:
            if hasattr(api_store, attr):
                api_names = getattr(api_store, attr)
                break

        if api_names is None:
            return []

        api_names = list(api_names)
        if len(api_names) == 0:
            return []

        if hasattr(api_store, "timestamps") and api_store.timestamps is not None:
            try:
                ts = api_store.timestamps
                ts = ts.detach().cpu().numpy() if hasattr(ts, "detach") else np.asarray(ts)
                if len(ts) == len(api_names):
                    api_names = [name for _, name in sorted(zip(ts.tolist(), api_names), key=lambda x: x[0])]
            except Exception:
                pass

        api_names = reduce_api_sequence(api_names, API_SEQ_HARD_LIMIT, API_SEQ_REDUCED_TO)
        api_names = [_normalize_api_name(a) for a in api_names]
        return api_names
    except Exception:
        return []


def extract_api_sequence_from_graph(graph) -> str:
    tokens = extract_api_token_list_from_graph(graph)
    return " ".join(tokens) if tokens else "empty"


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
# Attack helpers
# ============================================================

def _sample_seed_from_id(sample_id: str, base_seed: int) -> int:
    return (hash(sample_id) ^ base_seed) & 0xFFFFFFFF


def ensure_api_timestamps_for_attack(graph):
    try:
        tmp_proc = RansomwareGraphDataset.__new__(RansomwareGraphDataset)
        graph = tmp_proc._ensure_api_timestamps(graph)
    except Exception:
        pass
    return graph


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


def _alias_for_token(token: str, rng: random.Random) -> str:
    token = _normalize_api_name(token)
    if token in API_RENAME_MAP and API_RENAME_MAP[token]:
        return rng.choice(API_RENAME_MAP[token])

    if not UNIGRAM_USE_SUFFIX_VARIANTS:
        return token

    choices = [
        f"{token}_alt",
        f"{token}_variant",
        f"{token}_compat",
        f"{token}_shim",
        f"{token}_x",
    ]
    return rng.choice(choices)


# --- Tier-2 out-of-context rename ------------------------------------------
# A pool of real but behaviourally UNRELATED Windows API names (GUI / window /
# cursor / locale / drawing / multimedia). Renaming a malware API to one of these
# preserves the behaviour (the sample still does the same thing) but destroys the
# recoverable semantics: a semantic model that re-encodes the name lands on the
# WRONG behaviour rather than a near-synonym, so this stress-tests whether a
# detector relies on genuine behavioural understanding vs recoverable API meaning.
OUT_OF_CONTEXT_API_POOL = [
    "getcursorpos", "setcursorpos", "getwindowrect", "showwindow",
    "getdesktopwindow", "getforegroundwindow", "messagebeep", "loadcursorw",
    "loadiconw", "getsystemmetrics", "getwindowtextw", "setwindowtextw",
    "invalidaterect", "beginpaint", "endpaint", "getdc", "releasedc",
    "textoutw", "createfontw", "selectobject", "getlocaleinfow",
    "getdateformatw", "gettimeformatw", "charupperw", "charlowerw",
    "lstrlenw", "wsprintfw", "playsoundw", "clienttoscreen", "screentoclient",
]


def _random_alias_for_token(token: str, rng: random.Random) -> str:
    """Out-of-context rename: map a token to a random, behaviourally-unrelated
    real API name (see OUT_OF_CONTEXT_API_POOL). Carries no recoverable
    relationship to the original, unlike the meaning-preserving _alias_for_token."""
    return rng.choice(OUT_OF_CONTEXT_API_POOL)


def apply_full_shuffle_attack(graph, sample_id: str):
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
    if FULL_SHUFFLE_PRESERVE_FIRST and middle:
        first = [middle[0]]
        middle = middle[1:]
    if FULL_SHUFFLE_PRESERVE_LAST and middle:
        last = [middle[-1]]
        middle = middle[:-1]

    rng.shuffle(middle)
    permuted = first + middle + last

    new_ts = ts.clone()
    new_ts[torch.tensor(permuted, dtype=torch.long)] = sorted_ts
    graph["api"].timestamps = new_ts
    return graph


def apply_frequency_dilution_attack(graph, sample_id: str, alias_fn=None):
    """
    Unigram-specific attack:
    1) rename many API tokens to alias/OOV-looking variants
    2) optionally add large benign-looking padding

    alias_fn selects the rename style: the default `_alias_for_token` produces
    meaning-preserving aliases (synonyms / suffix variants -- Tier-1); passing
    `_random_alias_for_token` produces out-of-context names (Tier-2 stress test).
    Everything else (rename fraction, padding) is identical, so the two tiers
    differ ONLY in whether the renamed name keeps recoverable semantics.
    """
    if alias_fn is None:
        alias_fn = _alias_for_token
    graph = copy.deepcopy(graph)
    graph = ensure_api_timestamps_for_attack(graph)

    names = _get_api_names_list(graph)
    if names is None:
        return graph

    if not hasattr(graph["api"], "x") or not isinstance(graph["api"].x, torch.Tensor):
        return graph

    x = graph["api"].x
    n = int(x.shape[0])
    if n <= 0 or len(names) != n:
        return graph

    rng = random.Random(_sample_seed_from_id(sample_id, ATTACK_BASE_SEED + 1001))

    # ---- lexical rename phase ----
    token_counts = Counter(_normalize_api_name(x) for x in names)
    protected_tokens = {tok for tok, _ in token_counts.most_common(max(0, UNIGRAM_KEEP_CORE_APIS))}

    candidate_indices = [
        i for i, tok in enumerate(names) if _normalize_api_name(tok) not in protected_tokens
    ]
    rename_n = int(round(len(candidate_indices) * UNIGRAM_RENAME_FRACTION))
    rename_n = min(rename_n, len(candidate_indices))

    rename_idx_set = set(rng.sample(candidate_indices, rename_n)) if rename_n > 0 else set()

    renamed_names = []
    for i, tok in enumerate(names):
        norm_tok = _normalize_api_name(tok)
        if i in rename_idx_set:
            renamed_names.append(alias_fn(norm_tok, rng))
        else:
            renamed_names.append(norm_tok)

    _set_api_names_list(graph, renamed_names)

    # ---- heavy padding phase ----
    if UNIGRAM_ENABLE_PADDING:
        add_n = int(round(n * UNIGRAM_ATTACK_PADDING_RATIO))
        add_n = max(UNIGRAM_ATTACK_MIN_ADD, add_n)
        add_n = min(UNIGRAM_ATTACK_MAX_ADD, add_n)

        proto_idx = [rng.randrange(n) for _ in range(add_n)]
        new_x = x[torch.tensor(proto_idx, dtype=torch.long)].clone()
        if new_x.numel() > 0:
            new_x = new_x + 0.001 * torch.randn_like(new_x)

        graph["api"].x = torch.cat([graph["api"].x, new_x], dim=0)

        repeated_pool = []
        for api_name in PADDING_API_POOL:
            repeated_pool.extend([api_name] * 8)

        new_names = [rng.choice(repeated_pool) for _ in range(add_n)]
        _set_api_names_list(graph, list(_get_api_names_list(graph)) + new_names)

        if hasattr(graph["api"], "timestamps") and isinstance(graph["api"].timestamps, torch.Tensor):
            old_ts = graph["api"].timestamps
            if old_ts.numel() == n:
                last_t = float(old_ts.max().item()) if old_ts.numel() > 0 else 0.0
                delta = torch.linspace(1e-3, 1e-3 * add_n, steps=add_n, dtype=old_ts.dtype)
                new_ts = old_ts.new_tensor(last_t) + delta
                graph["api"].timestamps = torch.cat([old_ts, new_ts], dim=0)

        if hasattr(graph["api"], "repeat_count") and isinstance(graph["api"].repeat_count, torch.Tensor):
            rc = graph["api"].repeat_count
            if rc.numel() == n:
                graph["api"].repeat_count = torch.cat([rc, torch.ones(add_n, dtype=rc.dtype)], dim=0)

        if hasattr(graph["api"], "run_duration") and isinstance(graph["api"].run_duration, torch.Tensor):
            rd = graph["api"].run_duration
            if rd.numel() == n:
                graph["api"].run_duration = torch.cat([rd, torch.zeros(add_n, dtype=rd.dtype)], dim=0)

        graph["api"].num_nodes = n + add_n
    else:
        graph["api"].num_nodes = n

    return graph


def apply_out_of_context_rename_attack(graph, sample_id: str):
    """Tier-2 stress attack: identical to the Tier-1 rename+padding, but renames
    to out-of-context (behaviourally-unrelated) real API names instead of
    meaning-preserving aliases. Tests whether a detector survives when the
    semantic channel is not just perturbed but actively misleading."""
    return apply_frequency_dilution_attack(graph, sample_id, alias_fn=_random_alias_for_token)


def apply_benign_interleaving_attack(graph, sample_id: str):
    """
    Bigram-specific attack:
    insert benign-looking APIs between existing APIs to break adjacency.
    """
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

    max_by_ratio = int(round(n * INTERLEAVE_RATIO))
    target_add = max(1, min(max_by_ratio, INTERLEAVE_MAX_ADD))

    insert_after_positions = list(range(0, max(0, len(order) - 1), max(1, INTERLEAVE_EVERY_K)))
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

    if hasattr(graph["api"], "repeat_count") and isinstance(graph["api"].repeat_count, torch.Tensor):
        rc = graph["api"].repeat_count
        if rc.numel() == n:
            graph["api"].repeat_count = torch.cat([rc, torch.ones(target_add, dtype=rc.dtype)], dim=0)

    if hasattr(graph["api"], "run_duration") and isinstance(graph["api"].run_duration, torch.Tensor):
        rd = graph["api"].run_duration
        if rd.numel() == n:
            graph["api"].run_duration = torch.cat([rd, torch.zeros(target_add, dtype=rd.dtype)], dim=0)

    graph["api"].num_nodes = n + target_add
    return graph


def apply_bigram_attack(graph, sample_id: str):
    if BIGRAM_ATTACK_MODE == "full_shuffle":
        return apply_full_shuffle_attack(graph, sample_id)
    return apply_benign_interleaving_attack(graph, sample_id)


def apply_named_attack(graph, sample_id: str, attack_name: str):
    if attack_name == "clean":
        return graph

    if attack_name == "unigram_attack":
        return apply_frequency_dilution_attack(graph, sample_id)

    if attack_name == "random_rename_attack":
        return apply_out_of_context_rename_attack(graph, sample_id)

    if attack_name == "bigram_attack":
        return apply_bigram_attack(graph, sample_id)

    if attack_name == "reorder_attack":
        # Pure reorder: permute the SAME calls (shuffle timestamps), insert nothing.
        # Distinct from bigram_attack (benign_interleaving), which distributes/inserts
        # calls. Both are realistic attacker moves; a sequence model is sensitive to
        # order, so this isolates the ordering axis.
        return apply_full_shuffle_attack(graph, sample_id)

    if attack_name == "combined_attack":
        if COMBINED_APPLY_UNIGRAM_FIRST:
            g = apply_frequency_dilution_attack(graph, sample_id)
            g = apply_bigram_attack(g, sample_id)
            return g
        g = apply_bigram_attack(graph, sample_id)
        g = apply_frequency_dilution_attack(g, sample_id)
        return g

    if attack_name == "combined_all_attack":
        # Realistic worst case: rename (Tier-1) + reorder + distribute, all three axes.
        g = apply_frequency_dilution_attack(graph, sample_id)   # rename (+ padding)
        g = apply_full_shuffle_attack(g, sample_id)             # reorder the calls
        g = apply_benign_interleaving_attack(g, sample_id)      # distribute/insert calls
        return g

    raise ValueError(f"Unknown attack_name: {attack_name}")


# ============================================================
# Classical data prep
# ============================================================

def subsample_other_malware(df_split: pd.DataFrame, keep_frac: float, split_name: str) -> pd.DataFrame:
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


def build_text_features_from_metadata(metadata_df: pd.DataFrame, graph_dir: Path):
    api_sequences = []
    labels = []
    md5s = []

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


def build_text_features_from_manifest(manifest_df: pd.DataFrame, attack_name: str):
    api_sequences = []
    labels = []
    sample_ids = []
    origins = []
    graph_paths = []
    bigram_overlaps = []
    unigram_overlaps = []
    length_ratios = []

    missing_graphs = 0
    failed_graphs = 0

    for _, row in manifest_df.iterrows():
        graph_path = Path(row["graph_path"])
        sample_id = str(row["sample_id"])

        if not graph_path.exists():
            missing_graphs += 1
            continue

        graph = load_graph(graph_path)
        if graph is None:
            failed_graphs += 1
            continue

        graph = ensure_api_timestamps_for_attack(graph)
        clean_tokens = extract_api_token_list_from_graph(graph)

        attacked_graph = apply_named_attack(graph, sample_id, attack_name)
        attacked_tokens = extract_api_token_list_from_graph(attacked_graph)

        seq_tokens = clean_tokens if attack_name == "clean" else attacked_tokens
        seq = " ".join(seq_tokens) if seq_tokens else "empty"

        api_sequences.append(seq)
        labels.append(int(row["label"]))
        sample_ids.append(sample_id)
        origins.append(str(row["origin"]))
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

    print(
        f"[INFO] Custom graph scan ({attack_name}): "
        f"kept={len(labels)} missing={missing_graphs} failed={failed_graphs}"
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
    return api_sequences, labels, info_df


def save_prediction_csv(info_df: pd.DataFrame, y_true, y_pred, model_name: str, out_path: Path):
    pred_df = pd.DataFrame(
        {
            "sample_id": info_df["sample_id"].tolist(),
            "origin": info_df["origin"].tolist(),
            "graph_path": info_df["graph_path"].tolist(),
            "attack_name": info_df["attack_name"].tolist() if "attack_name" in info_df.columns else [""] * len(info_df),
            "bigram_overlap_vs_clean": info_df["bigram_overlap_vs_clean"].tolist()
            if "bigram_overlap_vs_clean" in info_df.columns else [None] * len(info_df),
            "unigram_overlap_vs_clean": info_df["unigram_overlap_vs_clean"].tolist()
            if "unigram_overlap_vs_clean" in info_df.columns else [None] * len(info_df),
            "length_ratio_vs_clean": info_df["length_ratio_vs_clean"].tolist()
            if "length_ratio_vs_clean" in info_df.columns else [None] * len(info_df),
            "true_label": list(map(int, y_true)),
            "true_name": [label_to_name(x) for x in y_true],
            "pred_label": list(map(int, y_pred)),
            "pred_name": [label_to_name(x) for x in y_pred],
            "correct": [int(a == b) for a, b in zip(y_true, y_pred)],
            "model": model_name,
        }
    )
    pred_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved predictions: {out_path}")


def evaluate_and_print(y_true, y_pred, title: str):
    metrics = compute_metrics(y_pred, y_true)
    print_metrics(metrics, title)
    return metrics


def make_random_forest(_y_train):
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_xgboost(y_train):
    label_counts = Counter(y_train)
    total = len(y_train)
    sample_weights = np.array([total / (3 * label_counts[y]) for y in y_train], dtype=np.float32)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    class WrappedXGB:
        def __init__(self, inner, train_weights):
            self.inner = inner
            self.train_weights = train_weights

        def fit(self, X, y):
            self.inner.fit(X, y, sample_weight=self.train_weights, verbose=False)
            return self

        def predict(self, X):
            return self.inner.predict(X)

    return WrappedXGB(model, sample_weights)


def _stats_from_info_df(info_df: pd.DataFrame) -> dict:
    return {
        "bigram_overlap_mean": float(info_df["bigram_overlap_vs_clean"].mean()),
        "bigram_overlap_median": float(info_df["bigram_overlap_vs_clean"].median()),
        "unigram_overlap_mean": float(info_df["unigram_overlap_vs_clean"].mean()),
        "unigram_overlap_median": float(info_df["unigram_overlap_vs_clean"].median()),
        "length_ratio_mean": float(info_df["length_ratio_vs_clean"].mean()),
        "length_ratio_median": float(info_df["length_ratio_vs_clean"].median()),
    }


def fit_and_eval_classical_family_multi_attack(
    family_name: str,
    model_ctor,
    X_train_text,
    y_train,
    eval_tracks: Dict[str, tuple[list[str], list[int], pd.DataFrame]],
    out_dir: Path,
):
    results = {}

    for vec_name, ngram_range in TFIDF_VARIANTS:
        print(f"\n[INFO] {family_name}: running variant = {vec_name}  ngram_range={ngram_range}")

        vec = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=3000,
            min_df=2,
            sublinear_tf=True,
        )

        X_train = vec.fit_transform(X_train_text)
        transformed = {name: vec.transform(track[0]) for name, track in eval_tracks.items()}

        model = model_ctor(y_train)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        train_metrics = evaluate_and_print(y_train, train_pred, f"{family_name} [{vec_name}] | TRAIN")

        variant_results = {
            "tfidf_vocab_size": int(len(vec.vocabulary_)),
            "ngram_range": [int(ngram_range[0]), int(ngram_range[1])],
            "train": train_metrics,
        }

        preds_by_track = {}
        metrics_by_track = {}

        for track_name in EVAL_TRACKS:
            _, y_track, info_df = eval_tracks[track_name]
            preds = model.predict(transformed[track_name])
            preds_by_track[track_name] = preds

            title = f"{family_name} [{vec_name}] | {track_name.upper()} custom eval"
            metrics = evaluate_and_print(y_track, preds, title)
            metrics_by_track[track_name] = metrics

            file_prefix = family_name.lower().replace(" ", "_")
            save_prediction_csv(
                info_df,
                y_track,
                preds,
                f"{family_name} [{vec_name}]",
                out_dir / f"{file_prefix}_{vec_name}_{track_name}_predictions.csv",
            )

            variant_results[f"{track_name}_custom_eval"] = metrics
            variant_results[f"{track_name}_tvb"] = compute_threat_vs_benign(np.array(metrics["confusion_matrix"]))
            variant_results[f"{track_name}_stats"] = _stats_from_info_df(info_df)

        for attack_name in ["unigram_attack", "bigram_attack", "combined_attack"]:
            variant_results[f"drop_clean_to_{attack_name}"] = compute_drop_summary(
                metrics_by_track["clean"],
                metrics_by_track[attack_name],
            )

        results[vec_name] = variant_results

    return results


# ============================================================
# TGNN eval
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


class CustomManifestDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        schema: Dict,
        graph_processor,
        attack_name: str,
        validate_graphs: bool = True,
    ):
        self.manifest_path = Path(manifest_csv)
        self.df = pd.read_csv(self.manifest_path)
        self.schema = schema
        self.graph_processor = graph_processor
        self.attack_name = attack_name
        self.validate_graphs = validate_graphs

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

        graph = self.graph_processor._ensure_api_timestamps(graph)
        graph = apply_named_attack(graph, sample_id, self.attack_name)

        if config.NORMALIZE_NODE_FEATURE_DIMS:
            graph = normalize_graph_node_features(graph, schema=self.schema, inplace=True)

        graph = self.graph_processor._compress_consecutive_api_runs(graph)
        graph = self.graph_processor._behavior_aware_api_cap(graph)
        graph = self.graph_processor._ensure_api_aux_tensors(graph)

        if self.validate_graphs and config.FAIL_ON_SCHEMA_MISMATCH:
            validate_graph_or_raise(
                graph,
                schema=self.schema,
                require_api_nodes=True,
                strict=config.SCHEMA_STRICT_VALIDATION,
            )

        graph.sample_id = sample_id
        graph.graph_path = str(graph_path)
        graph.origin = str(row.get("origin", ""))
        graph.attack_name = str(row.get("attack_name", self.attack_name))
        graph.bigram_overlap_vs_clean = float(row.get("bigram_overlap_vs_clean", 1.0))
        graph.unigram_overlap_vs_clean = float(row.get("unigram_overlap_vs_clean", 1.0))
        graph.length_ratio_vs_clean = float(row.get("length_ratio_vs_clean", 1.0))
        return graph, label


def custom_eval_collate(batch, schema, validate_batch=False):
    if not batch:
        return None, None

    sample_ids = []
    graph_paths = []
    origins = []
    attack_names = []
    bigram_overlaps = []
    unigram_overlaps = []
    length_ratios = []

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
def run_tgnn_eval(model, loader, criterion, vocab, schema, device, eval_epoch=None):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    prediction_rows = []

    for batch_item in loader:
        if batch_item is None or not isinstance(batch_item, (tuple, list)) or len(batch_item) != 2:
            continue

        batch_data, labels = batch_item
        if batch_data is None or labels is None:
            continue

        labels = labels.to(device)
        batch_data = batch_data.to(device)

        timestamps_dict = _extract_timestamps(batch_data)
        api_indices = _extract_api_indices(batch_data, vocab, device)
        _validate_batch_before_forward(batch_data, schema, api_indices=api_indices)

        logits = model(
            x_dict=batch_data.x_dict,
            edge_index_dict=batch_data.edge_index_dict,
            edge_attr_dict=batch_data.edge_attr_dict if hasattr(batch_data, "edge_attr_dict") else None,
            batch_dict=batch_data.batch_dict if hasattr(batch_data, "batch_dict") else None,
            timestamps_dict=timestamps_dict,
            api_indices=api_indices,
            current_epoch=eval_epoch,
        )

        loss = criterion(logits, labels)
        total_loss += float(loss.item())

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

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
            prediction_rows.append(
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
                }
            )

    if len(all_labels) == 0:
        raise RuntimeError("No TGNN evaluation samples were processed.")

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics, prediction_rows


# ============================================================
# Main
# ============================================================

def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    if getattr(config, "CUSTOM_EVAL_REBUILD_MANIFEST", True) or not Path(config.CUSTOM_EVAL_MANIFEST_PATH).exists():
        custom_manifest_df = build_custom_eval_manifest()
    else:
        custom_manifest_df = pd.read_csv(config.CUSTOM_EVAL_MANIFEST_PATH)
        print(f"[INFO] Using existing custom eval manifest: {config.CUSTOM_EVAL_MANIFEST_PATH}")

    metadata_path = Path(config.METADATA_PATH)
    split_root = Path(config.ACTIVE_GRAPH_DIR)

    df = pd.read_csv(metadata_path)
    df["_md5"] = df["md5"].astype(str).str.strip().str.lower()

    if "all_categories" not in df.columns:
        raise KeyError("Expected metadata column 'all_categories' not found.")

    train_md5s, _, _ = load_split_sets(df, split_root)
    df_train = df[df["_md5"].isin(train_md5s)].copy()

    excluded_ids = get_all_train_exclusions()
    if excluded_ids:
        before = len(df_train)
        df_train = df_train[~df_train["_md5"].isin(excluded_ids)].copy()
        print(f"[INFO] Removed {before - len(df_train)} train rows via exclusion IDs")

    if APPLY_MALWARE_SUBSAMPLE_TO_TRAIN:
        df_train = subsample_other_malware(df_train, MALWARE_KEEP_FRAC_TRAIN, "TRAIN")

    train_dir = split_root / "train"
    X_train_text, y_train, _ = build_text_features_from_metadata(df_train, train_dir)

    eval_tracks: Dict[str, tuple[list[str], list[int], pd.DataFrame]] = {}
    for track_name in EVAL_TRACKS:
        eval_tracks[track_name] = build_text_features_from_manifest(custom_manifest_df, attack_name=track_name)

    baseline_results = {}

    print("\n[INFO] Running Random Forest multi-attack ablations...")
    baseline_results["random_forest"] = fit_and_eval_classical_family_multi_attack(
        family_name="Random Forest",
        model_ctor=make_random_forest,
        X_train_text=X_train_text,
        y_train=y_train,
        eval_tracks=eval_tracks,
        out_dir=OUT_DIR,
    )

    if HAS_XGBOOST:
        print("\n[INFO] Running XGBoost multi-attack ablations...")
        baseline_results["xgboost"] = fit_and_eval_classical_family_multi_attack(
            family_name="XGBoost",
            model_ctor=make_xgboost,
            X_train_text=X_train_text,
            y_train=y_train,
            eval_tracks=eval_tracks,
            out_dir=OUT_DIR,
        )
    else:
        print(f"[WARN] Skipping XGBoost because unavailable: {XGB_IMPORT_ERROR}")
        baseline_results["xgboost"] = {"available": False, "import_error": XGB_IMPORT_ERROR}

    # -----------------------------
    # TGNN: evaluation only
    # -----------------------------
    device = torch.device(config.DEVICE)
    print(f"[INFO] TGNN device: {device}")

    schema = ensure_active_schema()
    if isinstance(schema, dict):
        active_schema = schema
    else:
        with open(config.ACTIVE_SCHEMA_PATH, "r", encoding="utf-8") as f:
            active_schema = json.load(f)

    graph_processor = RansomwareGraphDataset.__new__(RansomwareGraphDataset)
    graph_processor.schema = active_schema

    collate = partial(custom_eval_collate, schema=active_schema, validate_batch=False)

    def make_manifest_and_loader(track_name: str):
        _, _, info_df = eval_tracks[track_name]
        manifest_path = OUT_DIR / f"tmp_{track_name}_manifest.csv"

        tmp_df = custom_manifest_df.copy()
        tmp_df["attack_name"] = track_name
        tmp_df["bigram_overlap_vs_clean"] = info_df["bigram_overlap_vs_clean"].tolist()
        tmp_df["unigram_overlap_vs_clean"] = info_df["unigram_overlap_vs_clean"].tolist()
        tmp_df["length_ratio_vs_clean"] = info_df["length_ratio_vs_clean"].tolist()
        tmp_df.to_csv(manifest_path, index=False)

        dataset = CustomManifestDataset(
            manifest_csv=manifest_path,
            schema=active_schema,
            graph_processor=graph_processor,
            attack_name=track_name,
            validate_graphs=getattr(config, "CUSTOM_EVAL_VALIDATE_GRAPHS", True),
        )

        loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )
        return loader

    loaders = {track_name: make_manifest_and_loader(track_name) for track_name in EVAL_TRACKS}

    vocab = load_vocab(config.RESULTS_DIR / "api_vocabulary.json")
    model = create_hetero_tgnn(use_memory=config.USE_MEMORY).to(device)

    checkpoint_path = Path(config.CUSTOM_EVAL_CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    eval_epoch = int(ckpt.get("epoch", config.EPOCHS))

    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)

    tgnn_metrics_by_track = {}
    tgnn_rows_by_track = {}

    for track_name in EVAL_TRACKS:
        metrics, rows = run_tgnn_eval(
            model=model,
            loader=loaders[track_name],
            criterion=criterion,
            vocab=vocab,
            schema=active_schema,
            device=device,
            eval_epoch=eval_epoch,
        )
        tgnn_metrics_by_track[track_name] = metrics
        tgnn_rows_by_track[track_name] = rows
        print_metrics(metrics, f"TGNN checkpoint | {track_name.upper()} custom eval")
        pd.DataFrame(rows).to_csv(OUT_DIR / f"tgnn_{track_name}_predictions.csv", index=False)

    tgnn_results = {}
    for track_name in EVAL_TRACKS:
        _, _, info_df = eval_tracks[track_name]
        tgnn_results[f"{track_name}_custom_eval"] = tgnn_metrics_by_track[track_name]
        tgnn_results[f"{track_name}_tvb"] = compute_threat_vs_benign(
            np.array(tgnn_metrics_by_track[track_name]["confusion_matrix"])
        )
        tgnn_results[f"{track_name}_stats"] = _stats_from_info_df(info_df)

    for attack_name in ["unigram_attack", "bigram_attack", "combined_attack"]:
        tgnn_results[f"drop_clean_to_{attack_name}"] = compute_drop_summary(
            tgnn_metrics_by_track["clean"],
            tgnn_metrics_by_track[attack_name],
        )

    summary = {
        "unigram_attack": {
            "name": "lexical_rename_plus_frequency_dilution",
            "rename_fraction": UNIGRAM_RENAME_FRACTION,
            "keep_core_apis": UNIGRAM_KEEP_CORE_APIS,
            "use_suffix_variants": UNIGRAM_USE_SUFFIX_VARIANTS,
            "enable_padding": UNIGRAM_ENABLE_PADDING,
            "padding_ratio": UNIGRAM_ATTACK_PADDING_RATIO,
            "min_add": UNIGRAM_ATTACK_MIN_ADD,
            "max_add": UNIGRAM_ATTACK_MAX_ADD,
        },
        "bigram_attack": {
            "name": BIGRAM_ATTACK_MODE,
            "full_shuffle_preserve_first": FULL_SHUFFLE_PRESERVE_FIRST,
            "full_shuffle_preserve_last": FULL_SHUFFLE_PRESERVE_LAST,
            "interleave_every_k": INTERLEAVE_EVERY_K,
            "interleave_ratio": INTERLEAVE_RATIO,
            "interleave_max_add": INTERLEAVE_MAX_ADD,
        },
        "combined_attack": {
            "apply_unigram_first": COMBINED_APPLY_UNIGRAM_FIRST,
        },
        "tfidf_variants": [
            {"name": name, "ngram_range": [int(rng[0]), int(rng[1])]}
            for name, rng in TFIDF_VARIANTS
        ],
        "train_setup": {
            "metadata_path": str(metadata_path),
            "active_graph_dir": str(split_root),
            "custom_eval_manifest_path": str(config.CUSTOM_EVAL_MANIFEST_PATH),
            "malware_keep_frac_train": MALWARE_KEEP_FRAC_TRAIN,
            "apply_malware_subsample_to_train": APPLY_MALWARE_SUBSAMPLE_TO_TRAIN,
            "api_seq_hard_limit": API_SEQ_HARD_LIMIT,
            "api_seq_reduced_to": API_SEQ_REDUCED_TO,
            "excluded_train_ids": len(excluded_ids),
            "train_samples_used": len(y_train),
            "custom_eval_samples_used": len(eval_tracks["clean"][1]),
        },
        "eval_track_stats": {
            track_name: _stats_from_info_df(eval_tracks[track_name][2]) for track_name in EVAL_TRACKS
        },
        "baseline": baseline_results,
        "tgnn": tgnn_results,
    }

    with open(OUT_DIR / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[INFO] Saved summary JSON: {OUT_DIR / 'comparison_summary.json'}")
    print(f"[INFO] Outputs saved in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()