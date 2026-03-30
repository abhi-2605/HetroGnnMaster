"""
Analyze API node counts in enriched TGNN graphs.

Reports:
- total graph count
- mean / median / min / max API node counts
- how often the max occurs
- list of graphs with max API count
- list of graphs with min API count
- top-K largest graphs
- percentile summary
- per-split summaries if train/val/test folders exist

Usage:
    python -m scripts.analyze_api_node_counts

Optional:
    python -m scripts.analyze_api_node_counts --root /path/to/cache/enriched
    python -m scripts.analyze_api_node_counts --topk 25
"""

from __future__ import annotations

import argparse
import math
import pickle
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import config


def get_api_node_count(graph) -> int:
    """
    Safely extract api node count from a HeteroData graph.
    Returns 0 if api.x is missing or malformed.
    """
    try:
        if "api" not in graph.node_types:
            return 0
        if not hasattr(graph["api"], "x"):
            return 0
        x = graph["api"].x
        if x is None:
            return 0
        return int(x.shape[0])
    except Exception:
        return 0


def scan_split(split_dir: Path) -> List[Tuple[str, int, Path]]:
    """
    Return list of (graph_id, api_node_count, path) for one split directory.
    """
    rows: List[Tuple[str, int, Path]] = []
    pkl_files = sorted(split_dir.glob("*.pkl"))

    for pkl_path in pkl_files:
        try:
            with open(pkl_path, "rb") as f:
                graph = pickle.load(f)
            count = get_api_node_count(graph)
            rows.append((pkl_path.stem, count, pkl_path))
        except Exception as e:
            print(f"⚠️ Failed to read {pkl_path}: {e}")

    return rows


def percentile_safe(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def summarize_rows(rows: List[Tuple[str, int, Path]], title: str, topk: int = 20) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    if not rows:
        print("No graphs found.")
        return

    counts = [count for _, count, _ in rows]
    total = len(counts)

    mean_v = statistics.mean(counts)
    median_v = statistics.median(counts)
    min_v = min(counts)
    max_v = max(counts)
    std_v = statistics.pstdev(counts) if len(counts) > 1 else 0.0

    max_rows = [(gid, c, p) for gid, c, p in rows if c == max_v]
    min_rows = [(gid, c, p) for gid, c, p in rows if c == min_v]

    print(f"Total graphs              : {total}")
    print(f"Average api node count    : {mean_v:.2f}")
    print(f"Median api node count     : {median_v}")
    print(f"Std dev                   : {std_v:.2f}")
    print(f"Minimum api node count    : {min_v}")
    print(f"Maximum api node count    : {max_v}")
    print(f"Max count occurrences     : {len(max_rows)}")
    print(f"Min count occurrences     : {len(min_rows)}")

    print("\nPercentiles")
    print(f"  p50  : {percentile_safe(counts, 50):.2f}")
    print(f"  p75  : {percentile_safe(counts, 75):.2f}")
    print(f"  p90  : {percentile_safe(counts, 90):.2f}")
    print(f"  p95  : {percentile_safe(counts, 95):.2f}")
    print(f"  p99  : {percentile_safe(counts, 99):.2f}")
    print(f"  p99.5: {percentile_safe(counts, 99.5):.2f}")
    print(f"  p99.9: {percentile_safe(counts, 99.9):.2f}")

    print("\nGraphs with MAX api node count")
    for gid, c, p in max_rows[:topk]:
        print(f"  {gid}  | api_nodes={c} | {p}")
    if len(max_rows) > topk:
        print(f"  ... and {len(max_rows) - topk} more")

    print("\nGraphs with MIN api node count")
    for gid, c, p in min_rows[:topk]:
        print(f"  {gid}  | api_nodes={c} | {p}")
    if len(min_rows) > topk:
        print(f"  ... and {len(min_rows) - topk} more")

    largest = sorted(rows, key=lambda x: x[1], reverse=True)[:topk]
    print(f"\nTop {topk} largest graphs by api node count")
    for rank, (gid, c, p) in enumerate(largest, start=1):
        print(f"  #{rank:02d}  {gid}  | api_nodes={c} | {p}")

    freq = Counter(counts)
    most_common_sizes = freq.most_common(15)
    print("\nMost common api node counts")
    for value, occurrences in most_common_sizes:
        print(f"  api_nodes={value:<8} -> {occurrences} graphs")


def gather_all(root: Path) -> Dict[str, List[Tuple[str, int, Path]]]:
    """
    If train/val/test folders exist, scan them separately.
    Otherwise scan root directly.
    """
    split_names = ["train", "val", "test"]
    has_split_dirs = any((root / s).exists() and (root / s).is_dir() for s in split_names)

    results: Dict[str, List[Tuple[str, int, Path]]] = {}

    if has_split_dirs:
        for split in split_names:
            split_dir = root / split
            if split_dir.exists() and split_dir.is_dir():
                print(f"Scanning split: {split_dir}")
                results[split] = scan_split(split_dir)
    else:
        print(f"Scanning flat directory: {root}")
        results["all"] = scan_split(root)

    return results


def flatten_results(results: Dict[str, List[Tuple[str, int, Path]]]) -> List[Tuple[str, int, Path]]:
    merged: List[Tuple[str, int, Path]] = []
    for rows in results.values():
        merged.extend(rows)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=str(config.ACTIVE_GRAPH_DIR),
        help="Root enriched graph directory (default: config.ACTIVE_GRAPH_DIR)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="How many largest/min/max graphs to print",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Graph root not found: {root}")

    print("=" * 90)
    print("API NODE COUNT ANALYSIS")
    print("=" * 90)
    print(f"Graph root: {root}")

    results = gather_all(root)

    for split_name, rows in results.items():
        summarize_rows(rows, title=f"SPLIT: {split_name.upper()}", topk=args.topk)

    merged = flatten_results(results)
    if len(results) > 1:
        summarize_rows(merged, title="OVERALL SUMMARY", topk=args.topk)


if __name__ == "__main__":
    main()