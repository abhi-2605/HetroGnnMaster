#!/usr/bin/env python3
"""
scripts/check_api_embeddings_in_graphs.py

Inspect cached PyG HeteroData .pkl graphs and check whether API embeddings /
API node features are attached.

Examples:

    python -m scripts.check_api_embeddings_in_graphs --split val --max_samples 2

    python -m scripts.check_api_embeddings_in_graphs --split test

    python -m scripts.check_api_embeddings_in_graphs --folder /path/to/graphs --max_samples 10

Outputs:
    results/api_embedding_check/
        api_embedding_check_<name>.csv
        api_embedding_check_<name>.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config


# ============================================================
# Helpers
# ============================================================

def safe_hasattr(obj: Any, name: str) -> bool:
    try:
        return hasattr(obj, name)
    except Exception:
        return False


def safe_getattr(obj: Any, name: str, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def tensor_shape(x: Any):
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    return None


def tensor_numel(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.numel())
    return 0


def tensor_dtype(x: Any):
    if isinstance(x, torch.Tensor):
        return str(x.dtype)
    return None


def tensor_stats(x: Any) -> Dict[str, Any]:
    if not isinstance(x, torch.Tensor):
        return {
            "present": False,
            "shape": None,
            "dtype": None,
            "numel": 0,
            "mean": None,
            "std": None,
            "zero_frac": None,
            "nan_frac": None,
        }

    if x.numel() == 0:
        return {
            "present": True,
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "numel": 0,
            "mean": None,
            "std": None,
            "zero_frac": None,
            "nan_frac": None,
        }

    xf = x.detach().float().reshape(-1)
    finite = torch.isfinite(xf)
    finite_x = xf[finite]

    return {
        "present": True,
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "numel": int(x.numel()),
        "mean": float(finite_x.mean().item()) if finite_x.numel() > 0 else None,
        "std": float(finite_x.std(unbiased=False).item()) if finite_x.numel() > 0 else None,
        "zero_frac": float((xf == 0).float().mean().item()),
        "nan_frac": float((~finite).float().mean().item()),
    }


def get_graph_files(folder: Path, max_samples: Optional[int] = None) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    files = sorted(folder.glob("*.pkl"))

    if max_samples is not None:
        files = files[:max_samples]

    return files


def resolve_folder(args) -> Path:
    if args.folder:
        return Path(args.folder).expanduser().resolve()

    if not args.split:
        raise ValueError("Provide either --split or --folder")

    return Path(config.ACTIVE_GRAPH_DIR) / args.split


def get_node_types(graph) -> List[str]:
    try:
        return list(graph.node_types)
    except Exception:
        try:
            return list(graph.x_dict.keys())
        except Exception:
            return []


def get_api_store(graph):
    try:
        return graph["api"]
    except Exception:
        return None


def get_api_x(graph):
    try:
        return graph["api"].x
    except Exception:
        try:
            return graph.x_dict.get("api", None)
        except Exception:
            return None


def get_api_names(graph):
    candidates = []

    try:
        candidates.append(graph["api"].api_names)
    except Exception:
        pass

    try:
        candidates.append(graph.api_names)
    except Exception:
        pass

    try:
        candidates.append(getattr(graph, "api_names"))
    except Exception:
        pass

    for c in candidates:
        if c is None:
            continue
        if isinstance(c, torch.Tensor):
            continue
        if isinstance(c, (list, tuple)):
            return [str(v) for v in c]
        if isinstance(c, str):
            return [c]

    return []


def get_possible_api_embedding_fields(api_store) -> Dict[str, Any]:
    """
    Checks common names where API embeddings/vectors may be stored.
    """

    candidate_names = [
        "api_embeddings",
        "api_embedding",
        "embeddings",
        "embedding",
        "semantic_embeddings",
        "semantic_embedding",
        "x_semantic",
        "api_x",
        "api_indices",
        "api_ids",
        "token_ids",
        "input_ids",
        "api_names",
        "timestamps",
    ]

    found = {}

    if api_store is None:
        return found

    for name in candidate_names:
        val = safe_getattr(api_store, name, None)
        if val is not None:
            found[name] = val

    try:
        for key in api_store.keys():
            if key not in found:
                found[key] = api_store[key]
    except Exception:
        pass

    return found


# ============================================================
# Per-graph inspection
# ============================================================

def inspect_graph(graph_path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "sample_id": graph_path.stem,
        "graph_path": str(graph_path),
        "load_ok": False,
        "error": None,
    }

    try:
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        row["load_ok"] = True

        node_types = get_node_types(graph)
        api_store = get_api_store(graph)
        api_x = get_api_x(graph)
        api_names = get_api_names(graph)

        row["node_types"] = "|".join(node_types)
        row["has_api_node_type"] = "api" in node_types
        row["has_api_store"] = api_store is not None

        api_x_stats = tensor_stats(api_x)

        row["has_api_x"] = api_x_stats["present"]
        row["api_x_shape"] = api_x_stats["shape"]
        row["api_x_dtype"] = api_x_stats["dtype"]
        row["api_x_numel"] = api_x_stats["numel"]
        row["api_x_mean"] = api_x_stats["mean"]
        row["api_x_std"] = api_x_stats["std"]
        row["api_x_zero_frac"] = api_x_stats["zero_frac"]
        row["api_x_nan_frac"] = api_x_stats["nan_frac"]

        if isinstance(api_x, torch.Tensor) and api_x.dim() >= 2:
            row["num_api_nodes"] = int(api_x.size(0))
            row["api_feature_dim"] = int(api_x.size(1))
        elif isinstance(api_x, torch.Tensor) and api_x.dim() == 1:
            row["num_api_nodes"] = int(api_x.size(0))
            row["api_feature_dim"] = 1
        else:
            row["num_api_nodes"] = 0
            row["api_feature_dim"] = None

        row["has_api_names"] = len(api_names) > 0
        row["api_names_count"] = len(api_names)
        row["api_unique_names_count"] = len(set(api_names))
        row["api_top_10"] = "|".join([f"{k}:{v}" for k, v in Counter(api_names).most_common(10)])

        possible_fields = get_possible_api_embedding_fields(api_store)

        embedding_like_fields = []
        tensor_fields = []

        for name, val in possible_fields.items():
            if isinstance(val, torch.Tensor):
                tensor_fields.append(f"{name}:{list(val.shape)}:{val.dtype}")

                lname = name.lower()
                if (
                    "embed" in lname
                    or "semantic" in lname
                    or name in {"x", "x_semantic", "api_x"}
                ):
                    embedding_like_fields.append(f"{name}:{list(val.shape)}:{val.dtype}")
            else:
                if "embed" in name.lower() or "semantic" in name.lower():
                    embedding_like_fields.append(f"{name}:non_tensor:{type(val).__name__}")

        row["api_store_tensor_fields"] = "|".join(tensor_fields)
        row["api_embedding_like_fields"] = "|".join(embedding_like_fields)
        row["has_embedding_like_field"] = len(embedding_like_fields) > 0

        # Main decision:
        # In your graphs, api.x with shape [num_api_nodes, 775] is the attached API feature/embedding matrix.
        row["has_api_embedding_attached"] = (
            row["has_api_x"]
            and row["num_api_nodes"] > 0
            and row["api_feature_dim"] is not None
            and int(row["api_feature_dim"]) > 1
        )

        row["embedding_status"] = (
            "OK_API_X_PRESENT"
            if row["has_api_embedding_attached"]
            else "MISSING_OR_EMPTY_API_X"
        )

        return row

    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        return row


# ============================================================
# Summary
# ============================================================

def build_summary(df: pd.DataFrame, folder: Path, max_samples: Optional[int]) -> Dict[str, Any]:
    total = int(len(df))
    loaded = int(df["load_ok"].sum()) if "load_ok" in df.columns else 0

    def count_true(col: str) -> int:
        if col not in df.columns:
            return 0
        return int(df[col].fillna(False).astype(bool).sum())

    summary = {
        "folder": str(folder),
        "max_samples": max_samples,
        "total_files_checked": total,
        "loaded_ok": loaded,
        "load_errors": total - loaded,
        "has_api_node_type": count_true("has_api_node_type"),
        "has_api_store": count_true("has_api_store"),
        "has_api_x": count_true("has_api_x"),
        "has_api_names": count_true("has_api_names"),
        "has_embedding_like_field": count_true("has_embedding_like_field"),
        "has_api_embedding_attached": count_true("has_api_embedding_attached"),
        "missing_api_embedding_attached": total - count_true("has_api_embedding_attached"),
    }

    if "api_feature_dim" in df.columns:
        dims = pd.to_numeric(df["api_feature_dim"], errors="coerce").dropna()
        summary["api_feature_dim_counts"] = {
            str(int(k)): int(v)
            for k, v in dims.value_counts().sort_index().to_dict().items()
        }

    if "num_api_nodes" in df.columns:
        vals = pd.to_numeric(df["num_api_nodes"], errors="coerce").dropna()
        if len(vals) > 0:
            summary["num_api_nodes"] = {
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "min": int(vals.min()),
                "max": int(vals.max()),
            }

    return summary


def print_summary(summary: Dict[str, Any]):
    print("\nAPI Embedding Check Summary")
    print("=" * 72)
    print(f"Folder                         : {summary['folder']}")
    print(f"Max samples                    : {summary['max_samples']}")
    print(f"Total files checked            : {summary['total_files_checked']}")
    print(f"Loaded OK                      : {summary['loaded_ok']}")
    print(f"Load errors                    : {summary['load_errors']}")
    print(f"Has API node type              : {summary['has_api_node_type']}")
    print(f"Has API store                  : {summary['has_api_store']}")
    print(f"Has API x                      : {summary['has_api_x']}")
    print(f"Has API names                  : {summary['has_api_names']}")
    print(f"Has embedding-like field       : {summary['has_embedding_like_field']}")
    print(f"Has API embedding attached     : {summary['has_api_embedding_attached']}")
    print(f"Missing API embedding attached : {summary['missing_api_embedding_attached']}")

    if "api_feature_dim_counts" in summary:
        print("\nAPI feature dim counts:")
        for dim, count in summary["api_feature_dim_counts"].items():
            print(f"  dim={dim:<6} count={count}")

    if "num_api_nodes" in summary:
        s = summary["num_api_nodes"]
        print("\nAPI node count:")
        print(f"  mean   : {s['mean']:.2f}")
        print(f"  median : {s['median']:.2f}")
        print(f"  min    : {s['min']}")
        print(f"  max    : {s['max']}")

    print("=" * 72)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Check whether .pkl HeteroData graphs have API embeddings/API node features attached."
    )

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split name under config.ACTIVE_GRAPH_DIR, e.g. train, val, test.",
    )

    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Explicit folder containing .pkl graphs. Overrides --split.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of graphs to inspect. If omitted, checks all graphs.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(config.RESULTS_DIR / "api_embedding_check"),
        help="Output folder.",
    )

    args = parser.parse_args()

    folder = resolve_folder(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = folder.name
    if args.split:
        name = f"split_{args.split}"
    if args.max_samples is not None:
        name = f"{name}_max_{args.max_samples}"

    print("[INFO] API embedding inspection")
    print(f"       folder      : {folder}")
    print(f"       max_samples : {args.max_samples}")
    print(f"       out_dir     : {out_dir}")

    graph_files = get_graph_files(folder, max_samples=args.max_samples)

    if not graph_files:
        raise RuntimeError(f"No .pkl files found in folder: {folder}")

    rows = []

    for i, graph_path in enumerate(graph_files, start=1):
        row = inspect_graph(graph_path)
        rows.append(row)

        status = row.get("embedding_status", "ERROR")
        sample_id = row.get("sample_id", graph_path.stem)

        print(
            f"[{i:>5}/{len(graph_files):<5}] "
            f"{sample_id}  {status}  "
            f"api_x_shape={row.get('api_x_shape')}"
        )

    df = pd.DataFrame(rows)

    csv_path = out_dir / f"api_embedding_check_{name}.csv"
    json_path = out_dir / f"api_embedding_check_{name}.json"

    df.to_csv(csv_path, index=False)

    summary = build_summary(df, folder=folder, max_samples=args.max_samples)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)

    missing = df[df["has_api_embedding_attached"] != True]

    if len(missing) > 0:
        missing_path = out_dir / f"api_embedding_missing_{name}.csv"
        missing.to_csv(missing_path, index=False)
        print(f"\n[WARN] Missing/empty API embedding rows saved: {missing_path}")

    print("\n[INFO] Saved:")
    print(f"  CSV summary  : {csv_path}")
    print(f"  JSON summary : {json_path}")


if __name__ == "__main__":
    main()