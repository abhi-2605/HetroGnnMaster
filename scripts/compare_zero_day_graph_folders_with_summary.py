#!/usr/bin/env python3
"""
scripts/compare_zero_day_graph_folders.py

Compare .pkl graph files from two zero-day folders:

    config.CACHE_DIR / "zero_day_unseen" / "old"
    config.CACHE_DIR / "zero_day_unseen" / "all"

Purpose:
    Understand why unseen graphs are missed by comparing them against older /
    same-family graphs at graph-structure, node-feature, edge-feature, API,
    temporal, and schema levels.

Outputs:
    results/zero_day_compare/
        graph_level_summary.csv
        folder_level_summary.json
        pairwise_same_filename_comparison.csv
        api_token_frequency_old.csv
        api_token_frequency_all.csv
        api_token_frequency_delta.csv
        node_type_stats_by_folder.csv
        edge_type_stats_by_folder.csv
        feature_distribution_by_folder.csv

Notes:
    - Works with PyG HeteroData-style graphs.
    - Attempts to be defensive because cached graph objects can vary.
    - Does not require the model/checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config


EdgeType = Tuple[str, str, str]


# ============================================================
# Generic helpers
# ============================================================

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return default
            return float(x.detach().float().mean().item())
        return float(x)
    except Exception:
        return default


def tensor_stats(x: Optional[torch.Tensor], prefix: str) -> Dict[str, Any]:
    if x is None or not isinstance(x, torch.Tensor):
        return {
            f"{prefix}_present": False,
            f"{prefix}_numel": 0,
            f"{prefix}_dim": None,
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_zero_frac": None,
            f"{prefix}_nan_frac": None,
        }

    if x.numel() == 0:
        return {
            f"{prefix}_present": True,
            f"{prefix}_numel": 0,
            f"{prefix}_dim": list(x.shape),
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_zero_frac": None,
            f"{prefix}_nan_frac": None,
        }

    xf = x.detach().float().view(-1)
    finite = torch.isfinite(xf)
    finite_x = xf[finite]

    if finite_x.numel() == 0:
        mean = std = min_v = max_v = None
    else:
        mean = float(finite_x.mean().item())
        std = float(finite_x.std(unbiased=False).item())
        min_v = float(finite_x.min().item())
        max_v = float(finite_x.max().item())

    return {
        f"{prefix}_present": True,
        f"{prefix}_numel": int(x.numel()),
        f"{prefix}_dim": list(x.shape),
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": min_v,
        f"{prefix}_max": max_v,
        f"{prefix}_zero_frac": float((xf == 0).float().mean().item()),
        f"{prefix}_nan_frac": float((~finite).float().mean().item()),
    }


def get_node_types(graph) -> List[str]:
    try:
        return list(graph.node_types)
    except Exception:
        try:
            return [k for k in graph.x_dict.keys()]
        except Exception:
            return []


def get_edge_types(graph) -> List[EdgeType]:
    try:
        return list(graph.edge_types)
    except Exception:
        try:
            return list(graph.edge_index_dict.keys())
        except Exception:
            return []


def get_node_x(graph, node_type: str) -> Optional[torch.Tensor]:
    try:
        return graph[node_type].x
    except Exception:
        try:
            return graph.x_dict.get(node_type, None)
        except Exception:
            return None


def get_edge_index(graph, edge_type: EdgeType) -> Optional[torch.Tensor]:
    try:
        return graph[edge_type].edge_index
    except Exception:
        try:
            return graph.edge_index_dict.get(edge_type, None)
        except Exception:
            return None


def get_edge_attr(graph, edge_type: EdgeType) -> Optional[torch.Tensor]:
    try:
        return graph[edge_type].edge_attr
    except Exception:
        try:
            return graph.edge_attr_dict.get(edge_type, None)
        except Exception:
            return None


def get_api_names(graph) -> List[str]:
    """
    Try multiple possible places where API names may be stored.
    """
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
        if isinstance(c, np.ndarray):
            c = c.tolist()
        if isinstance(c, (list, tuple)):
            return [str(v) for v in c]
        if isinstance(c, str):
            return [c]

    return []


def get_api_timestamps(graph) -> Optional[torch.Tensor]:
    candidates = []

    try:
        candidates.append(graph["api"].timestamps)
    except Exception:
        pass

    try:
        candidates.append(graph["api"].timestamp)
    except Exception:
        pass

    try:
        candidates.append(graph.api_timestamps)
    except Exception:
        pass

    try:
        candidates.append(graph.timestamps)
    except Exception:
        pass

    for c in candidates:
        if c is None:
            continue
        if isinstance(c, torch.Tensor):
            return c.detach().float().view(-1)
        try:
            return torch.tensor(c, dtype=torch.float32).view(-1)
        except Exception:
            continue

    return None


def edge_type_to_str(edge_type: EdgeType) -> str:
    return f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cosine_counter(c1: Counter, c2: Counter) -> float:
    keys = set(c1) | set(c2)
    if not keys:
        return 1.0

    v1 = np.array([c1.get(k, 0.0) for k in keys], dtype=np.float64)
    v2 = np.array([c2.get(k, 0.0) for k in keys], dtype=np.float64)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 and n2 == 0:
        return 1.0
    if n1 == 0 or n2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (n1 * n2))


# ============================================================
# Per-graph extraction
# ============================================================

def summarize_graph(graph_path: Path, folder_name: str) -> Dict[str, Any]:
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    node_types = get_node_types(graph)
    edge_types = get_edge_types(graph)

    row: Dict[str, Any] = {
        "folder": folder_name,
        "sample_id": graph_path.stem,
        "graph_path": str(graph_path),
        "num_node_types": len(node_types),
        "num_edge_types": len(edge_types),
        "node_types": "|".join(node_types),
        "edge_types": "|".join(edge_type_to_str(et) for et in edge_types),
    }

    total_nodes = 0
    total_edges = 0

    # Node stats
    for nt in node_types:
        x = get_node_x(graph, nt)
        n = int(x.size(0)) if isinstance(x, torch.Tensor) and x.dim() >= 1 else 0
        d = int(x.size(-1)) if isinstance(x, torch.Tensor) and x.dim() >= 2 else None

        row[f"nodes_{nt}"] = n
        row[f"node_dim_{nt}"] = d
        total_nodes += n

        stats = tensor_stats(x, f"x_{nt}")
        row.update(stats)

    # Edge stats
    for et in edge_types:
        key = edge_type_to_str(et)
        edge_index = get_edge_index(graph, et)
        edge_attr = get_edge_attr(graph, et)

        e = int(edge_index.size(1)) if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 else 0
        total_edges += e

        row[f"edges_{key}"] = e

        if isinstance(edge_attr, torch.Tensor):
            row[f"edge_attr_dim_{key}"] = int(edge_attr.size(-1)) if edge_attr.dim() >= 2 else 1
        else:
            row[f"edge_attr_dim_{key}"] = None

        row.update(tensor_stats(edge_attr, f"edge_attr_{key}"))

    row["total_nodes"] = total_nodes
    row["total_edges"] = total_edges
    row["edge_node_ratio"] = float(total_edges / max(total_nodes, 1))

    # API names / token stats
    api_names = get_api_names(graph)
    api_counter = Counter(api_names)

    row["api_name_count"] = len(api_names)
    row["api_unique_count"] = len(api_counter)
    row["api_repetition_ratio"] = float(1.0 - (len(api_counter) / max(len(api_names), 1)))
    row["api_top_20"] = "|".join([f"{k}:{v}" for k, v in api_counter.most_common(20)])

    # Timestamp stats
    ts = get_api_timestamps(graph)
    if ts is not None and ts.numel() > 0:
        ts = ts[torch.isfinite(ts)]
        if ts.numel() > 0:
            diffs = torch.diff(torch.sort(ts).values) if ts.numel() > 1 else torch.tensor([], dtype=torch.float32)
            row["timestamp_present"] = True
            row["timestamp_min"] = float(ts.min().item())
            row["timestamp_max"] = float(ts.max().item())
            row["timestamp_span"] = float((ts.max() - ts.min()).item())
            row["timestamp_mean_gap"] = float(diffs.mean().item()) if diffs.numel() > 0 else 0.0
            row["timestamp_max_gap"] = float(diffs.max().item()) if diffs.numel() > 0 else 0.0
            row["timestamp_zero_gap_frac"] = float((diffs == 0).float().mean().item()) if diffs.numel() > 0 else 0.0
        else:
            row["timestamp_present"] = False
    else:
        row["timestamp_present"] = False

    return row


# ============================================================
# Folder-level summaries
# ============================================================

def collect_graph_summaries(old_dir: Path, all_dir: Path) -> pd.DataFrame:
    rows = []

    for folder_name, folder_path in [("old", old_dir), ("all", all_dir)]:
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        graph_files = sorted(folder_path.glob("*.pkl"))
        print(f"[INFO] {folder_name}: found {len(graph_files)} .pkl files in {folder_path}")

        for i, graph_path in enumerate(graph_files, start=1):
            try:
                rows.append(summarize_graph(graph_path, folder_name))
            except Exception as e:
                rows.append({
                    "folder": folder_name,
                    "sample_id": graph_path.stem,
                    "graph_path": str(graph_path),
                    "error": f"{type(e).__name__}: {e}",
                })

            if i % 100 == 0:
                print(f"[INFO] {folder_name}: processed {i}/{len(graph_files)}")

    return pd.DataFrame(rows)


def summarize_numeric_by_folder(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    out = {}
    for folder, g in df.groupby("folder"):
        out[folder] = {}
        for col in columns:
            if col not in g.columns:
                continue
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            out[folder][col] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "min": float(vals.min()),
                "p25": float(vals.quantile(0.25)),
                "median": float(vals.median()),
                "p75": float(vals.quantile(0.75)),
                "max": float(vals.max()),
            }
    return out


def build_api_frequency_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folder_counters = {"old": Counter(), "all": Counter()}

    for _, row in df.iterrows():
        folder = row.get("folder")
        top_raw = row.get("api_top_20", "")
        if folder not in folder_counters or not isinstance(top_raw, str):
            continue

        for item in top_raw.split("|"):
            if ":" not in item:
                continue
            name, count = item.rsplit(":", 1)
            try:
                folder_counters[folder][name] += int(count)
            except Exception:
                pass

    old_df = pd.DataFrame(folder_counters["old"].most_common(), columns=["api_name", "count_old"])
    all_df = pd.DataFrame(folder_counters["all"].most_common(), columns=["api_name", "count_all"])

    delta = pd.merge(old_df, all_df, on="api_name", how="outer").fillna(0)
    delta["count_old"] = delta["count_old"].astype(int)
    delta["count_all"] = delta["count_all"].astype(int)

    total_old = max(delta["count_old"].sum(), 1)
    total_all = max(delta["count_all"].sum(), 1)

    delta["freq_old"] = delta["count_old"] / total_old
    delta["freq_all"] = delta["count_all"] / total_all
    delta["freq_delta_all_minus_old"] = delta["freq_all"] - delta["freq_old"]
    delta["abs_freq_delta"] = delta["freq_delta_all_minus_old"].abs()
    delta = delta.sort_values("abs_freq_delta", ascending=False)

    return old_df, all_df, delta


def build_type_stats(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    cols = [c for c in df.columns if c.startswith(prefix)]

    for col in cols:
        for folder, g in df.groupby("folder"):
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue

            rows.append({
                "folder": folder,
                "metric": col,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "min": float(vals.min()),
                "median": float(vals.median()),
                "max": float(vals.max()),
            })

    return pd.DataFrame(rows)


# ============================================================
# Pairwise same-filename comparison
# ============================================================

def pairwise_same_filename_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    old = df[df["folder"] == "old"].set_index("sample_id")
    all_ = df[df["folder"] == "all"].set_index("sample_id")

    common = sorted(set(old.index) & set(all_.index))
    rows = []

    for sample_id in common:
        r_old = old.loc[sample_id]
        r_all = all_.loc[sample_id]

        old_api = parse_top_api_counter(r_old.get("api_top_20", ""))
        all_api = parse_top_api_counter(r_all.get("api_top_20", ""))

        row = {
            "sample_id": sample_id,
            "old_path": r_old.get("graph_path"),
            "all_path": r_all.get("graph_path"),
            "total_nodes_old": safe_float(r_old.get("total_nodes")),
            "total_nodes_all": safe_float(r_all.get("total_nodes")),
            "total_nodes_delta": safe_float(r_all.get("total_nodes")) - safe_float(r_old.get("total_nodes")),
            "total_edges_old": safe_float(r_old.get("total_edges")),
            "total_edges_all": safe_float(r_all.get("total_edges")),
            "total_edges_delta": safe_float(r_all.get("total_edges")) - safe_float(r_old.get("total_edges")),
            "api_name_count_old": safe_float(r_old.get("api_name_count")),
            "api_name_count_all": safe_float(r_all.get("api_name_count")),
            "api_name_count_delta": safe_float(r_all.get("api_name_count")) - safe_float(r_old.get("api_name_count")),
            "api_unique_old": safe_float(r_old.get("api_unique_count")),
            "api_unique_all": safe_float(r_all.get("api_unique_count")),
            "api_unique_delta": safe_float(r_all.get("api_unique_count")) - safe_float(r_old.get("api_unique_count")),
            "api_top20_jaccard": jaccard(old_api.keys(), all_api.keys()),
            "api_top20_cosine": cosine_counter(old_api, all_api),
            "timestamp_span_old": safe_float(r_old.get("timestamp_span")),
            "timestamp_span_all": safe_float(r_all.get("timestamp_span")),
            "timestamp_span_delta": safe_float(r_all.get("timestamp_span")) - safe_float(r_old.get("timestamp_span")),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def parse_top_api_counter(raw: Any) -> Counter:
    c = Counter()
    if not isinstance(raw, str):
        return c

    for item in raw.split("|"):
        if ":" not in item:
            continue
        name, count = item.rsplit(":", 1)
        try:
            c[name] += int(count)
        except Exception:
            pass

    return c




# ============================================================
# Textual summary
# ============================================================

def _folder_metric_mean(folder_summary: Dict[str, Any], folder: str, metric: str) -> Optional[float]:
    try:
        return folder_summary["numeric_summary"][folder][metric]["mean"]
    except Exception:
        return None


def _fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{float(x):.{digits}f}"


def _pct_change(old_value: Optional[float], new_value: Optional[float]) -> Optional[float]:
    if old_value is None or new_value is None:
        return None
    if abs(old_value) < 1e-12:
        return None
    return 100.0 * (new_value - old_value) / abs(old_value)


def _direction_word(delta: Optional[float]) -> str:
    if delta is None:
        return "changed"
    if delta > 0:
        return "higher"
    if delta < 0:
        return "lower"
    return "unchanged"


def _top_metric_differences(stats_df: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    """
    Convert node/edge/feature stats to old-vs-all deltas.
    Expects rows with columns: folder, metric, mean.
    """
    if stats_df is None or stats_df.empty:
        return pd.DataFrame()
    required = {"folder", "metric", "mean"}
    if not required.issubset(stats_df.columns):
        return pd.DataFrame()

    pivot = stats_df.pivot_table(index="metric", columns="folder", values="mean", aggfunc="mean")
    if "old" not in pivot.columns or "all" not in pivot.columns:
        return pd.DataFrame()

    pivot = pivot.reset_index()
    pivot["delta_all_minus_old"] = pivot["all"] - pivot["old"]
    pivot["abs_delta"] = pivot["delta_all_minus_old"].abs()
    pivot = pivot.sort_values("abs_delta", ascending=False)
    return pivot.head(top_k)


def build_textual_summary(
    df: pd.DataFrame,
    folder_summary: Dict[str, Any],
    api_delta_df: pd.DataFrame,
    node_stats_df: pd.DataFrame,
    edge_stats_df: pd.DataFrame,
    feature_summary: pd.DataFrame,
    pair_df: pd.DataFrame,
    top_k: int = 10,
) -> str:
    """
    Build a plain-English summary of the main old-vs-all graph differences.
    """
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("ZERO-DAY GRAPH FOLDER COMPARISON SUMMARY")
    lines.append("=" * 80)

    num_graphs = folder_summary.get("num_graphs", {})
    old_n = int(num_graphs.get("old", 0))
    all_n = int(num_graphs.get("all", 0))
    lines.append(f"Graphs compared: old={old_n}, all={all_n}")
    lines.append(f"Error count: {folder_summary.get('error_count', 0)}")

    lines.append("\n[1] High-level graph size / temporal shift")
    key_metrics = [
        ("total_nodes", "Total nodes"),
        ("total_edges", "Total edges"),
        ("edge_node_ratio", "Edge/node ratio"),
        ("api_name_count", "API sequence length"),
        ("api_unique_count", "Unique API count"),
        ("api_repetition_ratio", "API repetition ratio"),
        ("timestamp_span", "Timestamp span"),
        ("timestamp_mean_gap", "Mean timestamp gap"),
        ("timestamp_max_gap", "Max timestamp gap"),
    ]

    for metric, label in key_metrics:
        old_mean = _folder_metric_mean(folder_summary, "old", metric)
        all_mean = _folder_metric_mean(folder_summary, "all", metric)
        pct = _pct_change(old_mean, all_mean)
        direction = _direction_word(None if old_mean is None or all_mean is None else all_mean - old_mean)
        pct_text = "n/a" if pct is None else f"{pct:+.2f}%"
        lines.append(
            f"  - {label:<24}: old={_fmt_num(old_mean)} | all={_fmt_num(all_mean)} "
            f"| all is {direction} ({pct_text})"
        )

    lines.append("\n[2] Structural shift by node type")
    node_diff = _top_metric_differences(node_stats_df, top_k=top_k)
    if node_diff.empty:
        lines.append("  - No node-type shift summary available.")
    else:
        for _, r in node_diff.iterrows():
            lines.append(
                f"  - {r['metric']:<28}: old={_fmt_num(r['old'])} | all={_fmt_num(r['all'])} "
                f"| delta={_fmt_num(r['delta_all_minus_old'])}"
            )

    lines.append("\n[3] Structural shift by edge type")
    edge_diff = _top_metric_differences(edge_stats_df, top_k=top_k)
    if edge_diff.empty:
        lines.append("  - No edge-type shift summary available.")
    else:
        for _, r in edge_diff.iterrows():
            lines.append(
                f"  - {r['metric']:<40}: old={_fmt_num(r['old'])} | all={_fmt_num(r['all'])} "
                f"| delta={_fmt_num(r['delta_all_minus_old'])}"
            )

    lines.append("\n[4] API token / behavior shift")
    if api_delta_df is None or api_delta_df.empty:
        lines.append("  - No API frequency delta summary available.")
    else:
        top_api = api_delta_df.sort_values("abs_freq_delta", ascending=False).head(top_k)
        for _, r in top_api.iterrows():
            direction = "more common in all" if r["freq_delta_all_minus_old"] > 0 else "less common in all"
            lines.append(
                f"  - {str(r['api_name'])[:45]:<45} "
                f"old_freq={r['freq_old']:.5f} all_freq={r['freq_all']:.5f} "
                f"delta={r['freq_delta_all_minus_old']:+.5f} ({direction})"
            )

    lines.append("\n[5] Feature distribution shift")
    feat_diff = _top_metric_differences(feature_summary, top_k=top_k)
    if feat_diff.empty:
        lines.append("  - No feature-distribution shift summary available.")
    else:
        for _, r in feat_diff.iterrows():
            lines.append(
                f"  - {str(r['metric'])[:55]:<55}: old={_fmt_num(r['old'], 4)} | "
                f"all={_fmt_num(r['all'], 4)} | delta={_fmt_num(r['delta_all_minus_old'], 4)}"
            )

    lines.append("\n[6] Same-filename pairwise comparison")
    if pair_df is None or pair_df.empty:
        lines.append("  - No same-filename pairs found. This is expected if old/all contain different samples.")
    else:
        lines.append(f"  - Same-filename pairs found: {len(pair_df)}")
        for metric in ["total_nodes_delta", "total_edges_delta", "api_name_count_delta", "api_unique_delta", "timestamp_span_delta"]:
            if metric in pair_df.columns:
                vals = pd.to_numeric(pair_df[metric], errors="coerce").dropna()
                if len(vals) > 0:
                    lines.append(
                        f"  - {metric:<24}: mean={vals.mean():+.2f}, median={vals.median():+.2f}, "
                        f"min={vals.min():+.2f}, max={vals.max():+.2f}"
                    )
        if "api_top20_cosine" in pair_df.columns:
            vals = pd.to_numeric(pair_df["api_top20_cosine"], errors="coerce").dropna()
            if len(vals) > 0:
                lines.append(f"  - API top-20 cosine similarity: mean={vals.mean():.4f}, median={vals.median():.4f}")

    lines.append("\n[7] Interpretation for missed unseen threats")
    lines.append("  - If 'all' has much larger API length/node/edge counts, the unseen graphs may be longer or structurally denser than training-like graphs.")
    lines.append("  - If API token deltas are large, the unseen family may use different API vocabulary or different behavior frequency patterns.")
    lines.append("  - If edge-type counts differ strongly, SAGE/GAT may receive a different neighborhood structure than it learned on old graphs.")
    lines.append("  - If feature means/std/zero fractions differ, the model may face feature-scale/domain shift even if node/edge types match.")
    lines.append("  - If temporal gaps/spans differ, the LSTM temporal branch may be seeing a different execution rhythm.")

    lines.append("\n[8] What to inspect next")
    lines.append("  - Open api_token_frequency_delta.csv and check top APIs overrepresented in all.")
    lines.append("  - Open edge_type_stats_by_folder.csv and see whether temporal/process/file/registry/network edges shift.")
    lines.append("  - Open feature_distribution_by_folder.csv and look for large x_api_* or edge_attr_* shifts.")
    lines.append("  - Cross-check missed samples from zero-day predictions CSV against graph_level_summary.csv by sample_id.")
    lines.append("=" * 80)

    return "\n".join(lines)


def print_and_save_textual_summary(summary_text: str, out_dir: Path) -> Path:
    summary_path = out_dir / "textual_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)
    print(f"\n[INFO] Textual summary saved: {summary_path}")
    return summary_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compare zero-day .pkl graph folders.")
    parser.add_argument(
        "--old-dir",
        type=str,
        default=str(config.CACHE_DIR / "zero_day_unseen" / "old"),
        help="Folder containing older/same-family .pkl graphs.",
    )
    parser.add_argument(
        "--all-dir",
        type=str,
        default=str(config.CACHE_DIR / "zero_day_unseen" / "all"),
        help="Folder containing all/unseen .pkl graphs.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(config.RESULTS_DIR / "zero_day_compare"),
        help="Output directory for CSV/JSON reports.",
    )
    args = parser.parse_args()

    old_dir = Path(args.old_dir)
    all_dir = Path(args.all_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Comparing graph folders")
    print(f"       old_dir: {old_dir}")
    print(f"       all_dir: {all_dir}")
    print(f"       out_dir: {out_dir}")

    df = collect_graph_summaries(old_dir=old_dir, all_dir=all_dir)

    graph_summary_path = out_dir / "graph_level_summary.csv"
    df.to_csv(graph_summary_path, index=False)

    numeric_cols = [
        "total_nodes",
        "total_edges",
        "edge_node_ratio",
        "api_name_count",
        "api_unique_count",
        "api_repetition_ratio",
        "timestamp_span",
        "timestamp_mean_gap",
        "timestamp_max_gap",
        "timestamp_zero_gap_frac",
    ]
    folder_summary = {
        "old_dir": str(old_dir),
        "all_dir": str(all_dir),
        "num_graphs": {
            k: int(v) for k, v in df["folder"].value_counts().to_dict().items()
        },
        "numeric_summary": summarize_numeric_by_folder(df, numeric_cols),
        "error_count": int(df["error"].notna().sum()) if "error" in df.columns else 0,
    }

    folder_summary_path = out_dir / "folder_level_summary.json"
    with open(folder_summary_path, "w", encoding="utf-8") as f:
        json.dump(folder_summary, f, indent=2)

    pair_df = pairwise_same_filename_comparison(df)
    pair_path = out_dir / "pairwise_same_filename_comparison.csv"
    pair_df.to_csv(pair_path, index=False)

    old_api_df, all_api_df, api_delta_df = build_api_frequency_tables(df)
    old_api_path = out_dir / "api_token_frequency_old.csv"
    all_api_path = out_dir / "api_token_frequency_all.csv"
    api_delta_path = out_dir / "api_token_frequency_delta.csv"

    old_api_df.to_csv(old_api_path, index=False)
    all_api_df.to_csv(all_api_path, index=False)
    api_delta_df.to_csv(api_delta_path, index=False)

    node_stats_df = build_type_stats(df, prefix="nodes_")
    node_stats_path = out_dir / "node_type_stats_by_folder.csv"
    node_stats_df.to_csv(node_stats_path, index=False)

    edge_stats_df = build_type_stats(df, prefix="edges_")
    edge_stats_path = out_dir / "edge_type_stats_by_folder.csv"
    edge_stats_df.to_csv(edge_stats_path, index=False)

    feature_cols = [
        c for c in df.columns
        if (
            c.endswith("_mean")
            or c.endswith("_std")
            or c.endswith("_zero_frac")
            or c.endswith("_nan_frac")
        )
        and (c.startswith("x_") or c.startswith("edge_attr_"))
    ]
    feature_summary = build_type_stats(df, prefix="")
    if feature_cols:
        rows = []
        for col in feature_cols:
            for folder, g in df.groupby("folder"):
                vals = pd.to_numeric(g[col], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                rows.append({
                    "folder": folder,
                    "metric": col,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=0)),
                    "min": float(vals.min()),
                    "median": float(vals.median()),
                    "max": float(vals.max()),
                })
        feature_summary = pd.DataFrame(rows)

    feature_stats_path = out_dir / "feature_distribution_by_folder.csv"
    feature_summary.to_csv(feature_stats_path, index=False)

    print("\n[INFO] Done. Saved reports:")
    print(f"  Graph-level summary      : {graph_summary_path}")
    print(f"  Folder-level summary     : {folder_summary_path}")
    print(f"  Pairwise same-filename   : {pair_path}")
    print(f"  API frequency old        : {old_api_path}")
    print(f"  API frequency all        : {all_api_path}")
    print(f"  API frequency delta      : {api_delta_path}")
    print(f"  Node type stats          : {node_stats_path}")
    print(f"  Edge type stats          : {edge_stats_path}")
    print(f"  Feature distribution     : {feature_stats_path}")
    print(f"  Textual summary          : {folder_summary_path}")

    print("\n[INFO] Quick interpretation guide:")
    print("  - api_token_frequency_delta.csv shows APIs over/under-represented in all vs old.")
    print("  - graph_level_summary.csv shows per-graph node/edge/API/timestamp differences.")
    print("  - pairwise_same_filename_comparison.csv only helps if both folders share filenames.")
    print("  - feature_distribution_by_folder.csv helps detect feature-scale/domain shift.")
    print("  - node_type_stats_by_folder.csv and edge_type_stats_by_folder.csv show structural shift.")


if __name__ == "__main__":
    main()
