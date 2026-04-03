#!/usr/bin/env python3
"""
scripts/benign_operation_analyser.py
------------------------------------
Analyse only BENIGN samples from the active enriched graph cache and infer
their likely benign operation type from graph structure + API names.

Uses paths from config.py:
- config.ACTIVE_GRAPH_DIR
- config.METADATA_PATH

Usage
-----
python -m scripts.benign_operation_analyser
python -m scripts.benign_operation_analyser --split train
python -m scripts.benign_operation_analyser --split val --output benign_val_ops.csv
"""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import config


# ─────────────────────────────────────────────────
# 1. Metadata loader
# ─────────────────────────────────────────────────

def load_metadata(csv_path: Path) -> Dict[str, dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "md5" not in df.columns:
        raise ValueError("Metadata CSV must contain column 'md5'")

    # Support either column name
    category_col = None
    if "all_categories" in df.columns:
        category_col = "all_categories"
    elif "all_category_names" in df.columns:
        category_col = "all_category_names"
    else:
        raise ValueError(
            "Metadata CSV must contain either 'all_categories' or 'all_category_names'"
        )

    lookup = {}
    for _, row in df.iterrows():
        md5 = str(row["md5"]).strip().lower()
        record = {k: row[k] for k in df.columns}
        record["_category_col"] = category_col
        lookup[md5] = record

    print(f"[INFO] Loaded {len(lookup):,} metadata rows from {csv_path}")
    return lookup


# ─────────────────────────────────────────────────
# 2. TGNN label mapping
# ─────────────────────────────────────────────────

def get_tgnn_label(row: dict) -> tuple[int, str]:
    category_col = row.get("_category_col", "all_categories")
    all_cats = str(row.get(category_col, "")).lower()

    if "ransomware" in all_cats:
        return 0, "Ransomware"
    elif "benign" in all_cats:
        return 1, "Benign"
    else:
        return 2, "Other Malware"


# ─────────────────────────────────────────────────
# 3. Benign behavior classifier
# ─────────────────────────────────────────────────

API_PATTERNS = {
    "network_communication": [
        "internetopen", "httpsend", "httpopen", "wsaconnect", "wsasend",
        "connect", "send", "recv", "getaddrinfo", "dnsquery",
        "internetreadfile", "internetwritefile", "wsastartup",
        "socket", "bind", "listen", "accept",
    ],
    "file_io": [
        "createfile", "readfile", "writefile", "deletefile", "movefile",
        "copyfile", "openfile", "setfileattributes", "getfileattributes",
        "findfile", "createdir", "removedir", "setendoffile",
        "flushfilebuffers",
    ],
    "registry_access": [
        "regopen", "regcreate", "regset", "regquery", "regdelete",
        "regclose", "regenumkey", "regenumvalue",
    ],
    "process_management": [
        "createprocess", "openprocess", "terminateprocess", "shellexecute",
        "createthread", "openthread", "suspendthread", "resumethread",
        "virtualalloc", "virtualfree", "loadlibrary", "freelibrary",
        "winexec",
    ],
    "crypto_or_encoding": [
        "cryptgenrandom", "bcryptgenrandom", "cryptencrypt", "cryptdecrypt",
        "crypthashdata", "base64", "rtldecompressbuffer",
    ],
    "system_info": [
        "getsysteminfo", "getcomputername", "getusername", "getversion",
        "gettickcount", "querysysteminfo", "ntquerysystem",
    ],
    "ui_or_display": [
        "findwindow", "sendmessage", "postmessage", "setwindowtext",
        "createwindow", "showwindow", "messagebox",
    ],
}

BENIGN_OP_DESCRIPTIONS = {
    "web_or_cloud_app": "Browser / cloud sync / web client",
    "network_service": "Network daemon / background service",
    "system_tool_or_installer": "System tool, updater, or installer",
    "file_operation_app": "Document editor / file manager / archiver",
    "system_configuration": "Registry / system config tool",
    "gui_application": "GUI desktop application",
    "crypto_or_security_tool": "Crypto library / security / AV helper",
    "system_monitor_or_utility": "System monitor / diagnostic / CLI utility",
    "simple_utility": "Simple utility (no strong signals)",
    "unknown_parse_error": "Could not determine (graph parse error)",
}

BENIGN_TYPE_RULES = [
    (lambda n, e, b: b.get("network_communication", 0) > 5 and b.get("file_io", 0) > 5,
     "web_or_cloud_app"),
    (lambda n, e, b: b.get("network_communication", 0) > 5,
     "network_service"),
    (lambda n, e, b: b.get("registry_access", 0) > 5 and b.get("process_management", 0) > 3,
     "system_tool_or_installer"),
    (lambda n, e, b: b.get("file_io", 0) > 5,
     "file_operation_app"),
    (lambda n, e, b: b.get("registry_access", 0) > 5,
     "system_configuration"),
    (lambda n, e, b: b.get("ui_or_display", 0) > 3,
     "gui_application"),
    (lambda n, e, b: b.get("crypto_or_encoding", 0) > 2,
     "crypto_or_security_tool"),
    (lambda n, e, b: b.get("system_info", 0) > 3,
     "system_monitor_or_utility"),
]


def classify_api_bucket(api_names: List[str]) -> Counter:
    bucket_counts = Counter()

    for name in api_names:
        name_lower = str(name).lower()
        matched = False
        for bucket, keywords in API_PATTERNS.items():
            if any(kw in name_lower for kw in keywords):
                bucket_counts[bucket] += 1
                matched = True
                break
        if not matched:
            bucket_counts["other"] += 1

    return bucket_counts


def identify_benign_type(graph) -> dict:
    try:
        node_summary = {}
        for nt in graph.node_types:
            store = graph[nt]
            if hasattr(store, "x") and store.x is not None:
                node_summary[nt] = int(store.x.shape[0])

        edge_summary = {}
        for et in graph.edge_types:
            store = graph[et]
            if hasattr(store, "edge_index") and store.edge_index is not None:
                edge_summary[str(et)] = int(store.edge_index.shape[1])

        api_names = []
        if "api" in graph.node_types and hasattr(graph["api"], "api_names"):
            try:
                api_names = list(graph["api"].api_names or [])
            except Exception:
                api_names = []

        bucket_hits = classify_api_bucket(api_names)

        has_network = node_summary.get("network", 0) > 2
        has_file = node_summary.get("file", 0) > 5
        has_reg = node_summary.get("registry", 0) > 3

        op_type = "simple_utility"
        for cond_fn, label in BENIGN_TYPE_RULES:
            if cond_fn(node_summary, edge_summary, bucket_hits):
                op_type = label
                break
        else:
            if has_network and has_file:
                op_type = "web_or_cloud_app"
            elif has_network:
                op_type = "network_service"
            elif has_file and has_reg:
                op_type = "system_tool_or_installer"
            elif has_file:
                op_type = "file_operation_app"
            elif has_reg:
                op_type = "system_configuration"
            else:
                op_type = "simple_utility"

        return {
            "benign_op_type": op_type,
            "node_summary": node_summary,
            "edge_summary": edge_summary,
            "api_bucket_hits": dict(bucket_hits.most_common(5)),
            "num_api_nodes": node_summary.get("api", 0),
            "api_names_sample": api_names[:10],
        }

    except Exception as exc:
        return {
            "benign_op_type": "unknown_parse_error",
            "error": str(exc),
            "node_summary": {},
            "edge_summary": {},
            "api_bucket_hits": {},
            "num_api_nodes": 0,
            "api_names_sample": [],
        }


# ─────────────────────────────────────────────────
# 4. Single benign sample analyser
# ─────────────────────────────────────────────────

def analyse_sample(pkl_path: Path, metadata: Dict[str, dict], verbose: bool = False) -> Optional[dict]:
    md5 = pkl_path.stem.lower()
    row = metadata.get(md5)

    if row is None:
        return None

    tgnn_cls, tgnn_name = get_tgnn_label(row)

    # Only benign
    if tgnn_cls != 1:
        return None

    result = {
        "md5": md5,
        "pkl_path": str(pkl_path),
        "found_in_metadata": True,
        "filename": row.get("filename", ""),
        "year": row.get("year", ""),
        "family": row.get("family", ""),
        "category_name": row.get("category_name", ""),
        "all_category_names": row.get(row.get("_category_col", "all_categories"), ""),
        "num_api_calls": row.get("num_api_calls", ""),
        "tgnn_class": tgnn_cls,
        "tgnn_class_name": tgnn_name,
    }

    try:
        with open(pkl_path, "rb") as f:
            graph = pickle.load(f)

        result["graph_loaded"] = True
        result["node_types"] = list(graph.node_types)
        result["edge_types"] = [str(et) for et in graph.edge_types]

        benign_info = identify_benign_type(graph)
        result.update(benign_info)

    except Exception as exc:
        result["graph_loaded"] = False
        result["graph_error"] = str(exc)
        result["benign_op_type"] = "unknown_parse_error"

    if verbose:
        print("\n" + "=" * 70)
        print(f"MD5            : {md5}")
        print(f"File           : {result['filename']}")
        print(f"TGNN label     : {result['tgnn_class_name']}")
        print(f"Benign type    : {result.get('benign_op_type', 'n/a')}")
        print(f"Description    : {BENIGN_OP_DESCRIPTIONS.get(result.get('benign_op_type', ''), '')}")
        print(f"API nodes      : {result.get('num_api_nodes', 0)}")
        print(f"Node summary   : {result.get('node_summary', {})}")
        print(f"API buckets    : {result.get('api_bucket_hits', {})}")
        print(f"API sample     : {result.get('api_names_sample', [])[:5]}")
        print("=" * 70)

    return result


# ─────────────────────────────────────────────────
# 5. Batch benign analyser
# ─────────────────────────────────────────────────

def analyse_split(split_dir: Path, metadata: Dict[str, dict], output_csv: Optional[Path] = None) -> List[dict]:
    pkl_files = sorted(split_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"[WARN] No .pkl files found in {split_dir}")
        return []

    print(f"[INFO] Scanning split: {split_dir} ({len(pkl_files):,} graph files)")
    results = []

    for i, pkl_file in enumerate(pkl_files, 1):
        result = analyse_sample(pkl_file, metadata, verbose=False)
        if result is not None:
            results.append(result)

        if i % 250 == 0:
            print(f"  ... processed {i}/{len(pkl_files)}")

    benign_ops = Counter(r.get("benign_op_type", "n/a") for r in results)

    print("\n" + "=" * 70)
    print(f"BENIGN SUMMARY FOR SPLIT: {split_dir.name.upper()}")
    print("=" * 70)
    print(f"Total benign samples analysed: {len(results)}")
    print("Benign operation types:")
    for op_type, count in benign_ops.most_common():
        desc = BENIGN_OP_DESCRIPTIONS.get(op_type, op_type)
        print(f"  {op_type:<28} {count:>5}   ({desc})")

    if output_csv is not None:
        flat_keys = [
            "md5", "pkl_path", "filename", "year", "family",
            "category_name", "all_category_names", "num_api_calls",
            "tgnn_class", "tgnn_class_name", "graph_loaded", "graph_error",
            "benign_op_type", "num_api_nodes",
        ]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"[INFO] Saved: {output_csv}")

    return results


def analyse_all_splits(graph_root: Path, metadata: Dict[str, dict], output_dir: Optional[Path] = None) -> List[dict]:
    all_results = []

    for split in ["train", "val", "test"]:
        split_dir = graph_root / split
        if not split_dir.exists():
            print(f"[WARN] Split not found: {split_dir}")
            continue

        output_csv = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_csv = output_dir / f"benign_{split}_operation_summary.csv"

        split_results = analyse_split(split_dir, metadata, output_csv=output_csv)
        all_results.extend(split_results)

    overall = Counter(r.get("benign_op_type", "n/a") for r in all_results)

    print("\n" + "=" * 70)
    print("OVERALL BENIGN OPERATION SUMMARY")
    print("=" * 70)
    print(f"Total benign samples analysed: {len(all_results)}")
    for op_type, count in overall.most_common():
        desc = BENIGN_OP_DESCRIPTIONS.get(op_type, op_type)
        print(f"  {op_type:<28} {count:>5}   ({desc})")

    return all_results


# ─────────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to analyse",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory for CSV files",
    )
    parser.add_argument(
        "--single",
        default=None,
        help="Analyse a single benign graph by md5 or filename within the chosen split",
    )
    args = parser.parse_args()

    graph_root = Path(config.ACTIVE_GRAPH_DIR)
    metadata_path = Path(config.METADATA_PATH)

    print(f"[INFO] Graph root   : {graph_root}")
    print(f"[INFO] Metadata CSV : {metadata_path}")

    metadata = load_metadata(metadata_path)

    if args.single:
        split = args.split if args.split != "all" else "train"
        split_dir = graph_root / split

        name = args.single
        if not name.endswith(".pkl"):
            name = name.replace(".json", "") + ".pkl"

        pkl_path = split_dir / name
        result = analyse_sample(pkl_path, metadata, verbose=True)

        if result is None:
            print("[INFO] Sample is either not found in metadata or not benign.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.split == "all":
        analyse_all_splits(graph_root, metadata, output_dir=output_dir)
    else:
        split_dir = graph_root / args.split
        output_csv = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_csv = output_dir / f"benign_{args.split}_operation_summary.csv"
        analyse_split(split_dir, metadata, output_csv=output_csv)


if __name__ == "__main__":
    main()