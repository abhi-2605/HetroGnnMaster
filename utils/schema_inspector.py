"""
utils/schema_inspector.py

Inspect enriched HeteroData graphs for the currently active graph source
and emit a schema report used by the schema-aware training pipeline.

Usage:
    python -m utils.schema_inspector

Outputs:
    config.ACTIVE_SCHEMA_PATH
    config.ACTIVE_SCHEMA_SUMMARY_PATH
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

import config


@dataclass
class NodeTypeStats:
    graphs_present: int = 0
    zero_node_graphs: int = 0
    total_nodes: int = 0
    min_nodes: Optional[int] = None
    max_nodes: Optional[int] = None
    feature_dims: Optional[List[int]] = None


@dataclass
class EdgeTypeStats:
    graphs_present: int = 0
    zero_edge_graphs: int = 0
    total_edges: int = 0
    min_edges: Optional[int] = None
    max_edges: Optional[int] = None
    has_edge_attr: bool = False
    edge_attr_dims: Optional[List[int]] = None


class SchemaInspector:
    def __init__(
        self,
        graph_root: Optional[Path] = None,
        output_json: Optional[Path] = None,
        output_summary: Optional[Path] = None,
        source_name: Optional[str] = None,
    ):
        self.graph_root = Path(graph_root) if graph_root else config.ACTIVE_GRAPH_DIR
        self.output_json = Path(output_json) if output_json else config.ACTIVE_SCHEMA_PATH
        self.output_summary = (
            Path(output_summary) if output_summary else config.ACTIVE_SCHEMA_SUMMARY_PATH
        )
        self.source_name = source_name or config.ACTIVE_GRAPH_SOURCE

        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        self.output_summary.parent.mkdir(parents=True, exist_ok=True)

        self.splits = ["train", "val", "test"]

        self.global_stats = {
            "total_graphs": 0,
            "graphs_with_label_tuple": 0,
            "graphs_without_label_tuple": 0,
            "failed_files": [],
        }

        self.split_stats: Dict[str, Dict[str, Any]] = {}
        self.node_stats: Dict[str, NodeTypeStats] = {}
        self.edge_stats: Dict[str, EdgeTypeStats] = {}

        self.label_counter = Counter()
        self.graph_object_types = Counter()

    def inspect(self) -> Dict[str, Any]:
        print("=" * 80)
        print("SCHEMA INSPECTION")
        print("=" * 80)
        print(f"Active source     : {self.source_name}")
        print(f"Graph root        : {self.graph_root}")
        print(f"Schema JSON out   : {self.output_json}")
        print(f"Schema summary out: {self.output_summary}")

        if not self.graph_root.exists():
            raise FileNotFoundError(
                f"Graph root does not exist: {self.graph_root}\n"
                f"Check config.ACTIVE_GRAPH_SOURCE / config.ACTIVE_GRAPH_DIR"
            )

        for split in self.splits:
            split_dir = self.graph_root / split
            split_files = sorted(split_dir.glob("*.pkl"))

            if not split_dir.exists():
                print(f"\n⚠ Missing split directory: {split_dir}")
                self.split_stats[split] = {
                    "num_files": 0,
                    "missing": True,
                    "failed_files": [],
                }
                continue

            print(f"\n[{split.upper()}] {len(split_files)} graph files found")
            split_summary = {
                "num_files": len(split_files),
                "missing": False,
                "failed_files": [],
            }

            for file_path in split_files:
                try:
                    obj = self._load_pickle(file_path)
                    graph, label, storage_type = self._unpack_graph(obj)

                    self.global_stats["total_graphs"] += 1
                    self.graph_object_types[storage_type] += 1

                    if storage_type == "tuple_graph_label":
                        self.global_stats["graphs_with_label_tuple"] += 1
                    else:
                        self.global_stats["graphs_without_label_tuple"] += 1

                    if label is not None:
                        self.label_counter[str(label)] += 1

                    self._inspect_graph(graph)

                except Exception as e:
                    msg = f"{file_path}: {type(e).__name__}: {e}"
                    split_summary["failed_files"].append(msg)
                    self.global_stats["failed_files"].append(msg)

            self.split_stats[split] = split_summary

        schema = self._build_schema_dict()
        self._save_outputs(schema)
        self._print_summary(schema)
        return schema

    def _load_pickle(self, path: Path) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

    def _unpack_graph(self, obj: Any) -> Tuple[Any, Optional[Any], str]:
        """
        Returns:
            graph, label, storage_type
        """
        if isinstance(obj, tuple):
            if len(obj) == 2:
                graph, label = obj
                return graph, label, "tuple_graph_label"
            if len(obj) == 1:
                return obj[0], None, "tuple_graph_only"
        return obj, None, "graph_only"

    def _inspect_graph(self, graph: Any) -> None:
        self.graph_object_types[type(graph).__name__] += 1

        node_types = list(getattr(graph, "node_types", []))
        edge_types = list(getattr(graph, "edge_types", []))

        for node_type in node_types:
            if node_type not in self.node_stats:
                self.node_stats[node_type] = NodeTypeStats()

            nstats = self.node_stats[node_type]
            nstats.graphs_present += 1

            store = graph[node_type]
            x = getattr(store, "x", None)

            if x is None:
                num_nodes = int(getattr(store, "num_nodes", 0) or 0)
                feat_dim = None
            else:
                num_nodes = int(x.size(0))
                feat_dim = int(x.size(1)) if x.dim() == 2 else None

            if num_nodes == 0:
                nstats.zero_node_graphs += 1

            nstats.total_nodes += num_nodes
            self._update_min_max(nstats, num_nodes, "min_nodes", "max_nodes")

            if feat_dim is not None:
                if nstats.feature_dims is None:
                    nstats.feature_dims = []
                if feat_dim not in nstats.feature_dims:
                    nstats.feature_dims.append(feat_dim)

        for edge_type in edge_types:
            edge_key = self._edge_type_to_str(edge_type)

            if edge_key not in self.edge_stats:
                self.edge_stats[edge_key] = EdgeTypeStats()

            estats = self.edge_stats[edge_key]
            estats.graphs_present += 1

            store = graph[edge_type]
            edge_index = getattr(store, "edge_index", None)
            num_edges = int(edge_index.size(1)) if edge_index is not None else 0

            if num_edges == 0:
                estats.zero_edge_graphs += 1

            estats.total_edges += num_edges
            self._update_min_max(estats, num_edges, "min_edges", "max_edges")

            edge_attr = getattr(store, "edge_attr", None)
            if edge_attr is not None:
                estats.has_edge_attr = True
                attr_dim = None
                if edge_attr.dim() == 2:
                    attr_dim = int(edge_attr.size(1))
                elif edge_attr.dim() == 1:
                    attr_dim = 1

                if attr_dim is not None:
                    if estats.edge_attr_dims is None:
                        estats.edge_attr_dims = []
                    if attr_dim not in estats.edge_attr_dims:
                        estats.edge_attr_dims.append(attr_dim)

    def _update_min_max(self, stats_obj: Any, value: int, min_attr: str, max_attr: str) -> None:
        current_min = getattr(stats_obj, min_attr)
        current_max = getattr(stats_obj, max_attr)

        if current_min is None or value < current_min:
            setattr(stats_obj, min_attr, value)
        if current_max is None or value > current_max:
            setattr(stats_obj, max_attr, value)

    def _edge_type_to_str(self, edge_type: Tuple[str, str, str]) -> str:
        src, rel, dst = edge_type
        return f"{src}::{rel}::{dst}"

    def _build_schema_dict(self) -> Dict[str, Any]:
        node_types_sorted = sorted(self.node_stats.keys())
        edge_types_sorted = sorted(self.edge_stats.keys())

        expected_edges = [
            self._edge_type_to_str(et) for et in getattr(config, "EDGE_TYPES", [])
        ]

        schema = {
            "source": {
                "name": self.source_name,
                "graph_root": str(self.graph_root),
            },
            "paths": {
                "project_root": str(config.PROJECT_ROOT),
                "cache_dir": str(config.CACHE_DIR),
                "results_dir": str(config.RESULTS_DIR),
                "schema_json": str(self.output_json),
                "schema_summary": str(self.output_summary),
            },
            "global": {
                **self.global_stats,
                "graph_object_types": dict(self.graph_object_types),
                "label_distribution_detected": dict(self.label_counter),
            },
            "splits": self.split_stats,
            "schema_summary": {
                "node_types_found": node_types_sorted,
                "edge_types_found": edge_types_sorted,
                "num_node_types_found": len(node_types_sorted),
                "num_edge_types_found": len(edge_types_sorted),
            },
            "node_types": {
                nt: self._finalize_node_stats(self.node_stats[nt])
                for nt in node_types_sorted
            },
            "edge_types": {
                et: self._finalize_edge_stats(self.edge_stats[et])
                for et in edge_types_sorted
            },
            "expected_from_config": {
                "node_types": list(getattr(config, "NODE_TYPES", [])),
                "edge_types": expected_edges,
                "default_node_type_order": list(getattr(config, "DEFAULT_NODE_TYPE_ORDER", [])),
                "default_edge_type_order": [
                    self._edge_type_to_str(et)
                    for et in getattr(config, "DEFAULT_EDGE_TYPE_ORDER", [])
                ],
                "api_node_dim": getattr(config, "API_NODE_DIM", None),
                "resource_node_dim": getattr(config, "RESOURCE_NODE_DIM", None),
                "temporal_edge_dim": getattr(config, "TEMPORAL_EDGE_DIM", None),
            },
            "compatibility_checks": self._build_compatibility_checks(),
        }
        return schema

    def _finalize_node_stats(self, stats: NodeTypeStats) -> Dict[str, Any]:
        d = asdict(stats)
        if d["feature_dims"] is not None:
            d["feature_dims"] = sorted(d["feature_dims"])
        return d

    def _finalize_edge_stats(self, stats: EdgeTypeStats) -> Dict[str, Any]:
        d = asdict(stats)
        if d["edge_attr_dims"] is not None:
            d["edge_attr_dims"] = sorted(d["edge_attr_dims"])
        return d

    def _build_compatibility_checks(self) -> Dict[str, Any]:
        checks = {
            "api_feature_dim_ok": None,
            "resource_feature_dims_ok": {},
            "temporal_edge_dim_ok": None,
            "missing_expected_node_types": [],
            "missing_expected_edge_types": [],
            "unexpected_node_types": [],
            "unexpected_edge_types": [],
        }

        expected_nodes = set(getattr(config, "NODE_TYPES", []))
        found_nodes = set(self.node_stats.keys())

        expected_edges = {
            self._edge_type_to_str(et) for et in getattr(config, "EDGE_TYPES", [])
        }
        found_edges = set(self.edge_stats.keys())

        checks["missing_expected_node_types"] = sorted(expected_nodes - found_nodes)
        checks["unexpected_node_types"] = sorted(found_nodes - expected_nodes)

        checks["missing_expected_edge_types"] = sorted(expected_edges - found_edges)
        checks["unexpected_edge_types"] = sorted(found_edges - expected_edges)

        if "api" in self.node_stats:
            api_dims = self.node_stats["api"].feature_dims or []
            checks["api_feature_dim_ok"] = (
                len(api_dims) == 1 and api_dims[0] == getattr(config, "API_NODE_DIM", None)
            )

        for node_type in ["file", "registry", "process", "network"]:
            dims = self.node_stats.get(node_type, NodeTypeStats()).feature_dims or []
            checks["resource_feature_dims_ok"][node_type] = (
                len(dims) == 1 and dims[0] == getattr(config, "RESOURCE_NODE_DIM", None)
            )

        temporal_key = "api::temporal::api"
        if temporal_key in self.edge_stats:
            edge_dims = self.edge_stats[temporal_key].edge_attr_dims or []
            checks["temporal_edge_dim_ok"] = (
                self.edge_stats[temporal_key].has_edge_attr
                and len(edge_dims) == 1
                and edge_dims[0] == getattr(config, "TEMPORAL_EDGE_DIM", None)
            )

        return checks

    def _save_outputs(self, schema: Dict[str, Any]) -> None:
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

        with open(self.output_summary, "w", encoding="utf-8") as f:
            f.write(self._build_readable_summary(schema))

        print(f"\n✓ Saved schema JSON    : {self.output_json}")
        print(f"✓ Saved schema summary : {self.output_summary}")

    def _build_readable_summary(self, schema: Dict[str, Any]) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("ENRICHED GRAPH SCHEMA SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Source name : {schema['source']['name']}")
        lines.append(f"Graph root  : {schema['source']['graph_root']}")
        lines.append(f"Total graphs: {schema['global']['total_graphs']}")
        lines.append("")

        lines.append("NODE TYPES FOUND:")
        for nt, stats in schema["node_types"].items():
            lines.append(
                f"  - {nt}: dims={stats['feature_dims']} "
                f"graphs_present={stats['graphs_present']} "
                f"zero_node_graphs={stats['zero_node_graphs']} "
                f"min_nodes={stats['min_nodes']} max_nodes={stats['max_nodes']}"
            )

        lines.append("")
        lines.append("EDGE TYPES FOUND:")
        for et, stats in schema["edge_types"].items():
            lines.append(
                f"  - {et}: edge_attr={stats['has_edge_attr']} "
                f"edge_attr_dims={stats['edge_attr_dims']} "
                f"graphs_present={stats['graphs_present']} "
                f"zero_edge_graphs={stats['zero_edge_graphs']} "
                f"min_edges={stats['min_edges']} max_edges={stats['max_edges']}"
            )

        lines.append("")
        lines.append("COMPATIBILITY CHECKS:")
        for k, v in schema["compatibility_checks"].items():
            lines.append(f"  - {k}: {v}")

        lines.append("")
        lines.append("SPLITS:")
        for split, info in schema["splits"].items():
            lines.append(
                f"  - {split}: num_files={info['num_files']} "
                f"missing={info['missing']} "
                f"failed_files={len(info['failed_files'])}"
            )

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)

    def _print_summary(self, schema: Dict[str, Any]) -> None:
        print("\n" + "=" * 80)
        print("INSPECTION COMPLETE")
        print("=" * 80)
        print(f"Source       : {schema['source']['name']}")
        print(f"Total graphs : {schema['global']['total_graphs']}")
        print(f"Node types   : {schema['schema_summary']['node_types_found']}")
        print(f"Edge types   : {schema['schema_summary']['edge_types_found']}")
        print(f"Failed files : {len(schema['global']['failed_files'])}")


def main() -> None:
    inspector = SchemaInspector()
    inspector.inspect()


if __name__ == "__main__":
    main()