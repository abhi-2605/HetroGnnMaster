"""
build_graphs_v2.py

Build the initial heterogeneous graph for the new parallel pipeline.

Purpose
-------
Convert one canonical adapted report (from report_adapter_v2.py) into a
PyTorch Geometric HeteroData graph in a single pass.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import torch
from torch_geometric.data import HeteroData

from pipeline_v2 import pipeline_config_v2 as cfg


class GraphBuilderV2:
    """
    Build initial raw graphs from adapted reports.

    Output graph notes
    ------------------
    - api.x                : [num_api, 7]
    - process.x            : [num_process, 6]
    - file.x               : [num_file, 6]
    - registry.x           : [num_registry, 6]
    - network.x            : [num_network, 6]
    - ('api','temporal','api').edge_attr : [num_temporal_edges, 3]
    """

    API_RAW_DIM = 7
    TEMPORAL_BASIC_EDGE_DIM = 3

    def __init__(self):
        self.file_path_re = re.compile(r"(?i)\b(?:[A-Z]:\\|\\\\)[^\s\"']+")
        self.registry_re = re.compile(
            r"(?i)(HKEY_[A-Z_]+\\[^\s\"']+|HKLM\\[^\s\"']+|HKCU\\[^\s\"']+|HKCR\\[^\s\"']+|HKU\\[^\s\"']+|\\REGISTRY\\[^\s\"']+)"
        )
        self.ip_re = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
        self.domain_re = re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[A-Za-z]{2,}\b")
        self.url_re = re.compile(r"(?i)\b(?:https?|ftp)://[^\s\"']+")

    # ============================================================
    # PUBLIC API
    # ============================================================
    def build_graph(self, adapted: Dict[str, Any]) -> HeteroData:
        graph = HeteroData()

        label = adapted.get("label")
        graph.graph_id = adapted.get("report_id", "unknown")
        graph.source_format = adapted.get("source_format", "unknown")
        graph.regime = adapted.get("regime", "unknown")
        graph.label_name = adapted.get("label_name")
        graph.y = torch.tensor([-1 if label is None else int(label)], dtype=torch.long)

        activity = adapted.get("activity", {})
        graph.activity_features = torch.tensor(
            self._build_activity_feature_vector(adapted, activity),
            dtype=torch.float32
        ).view(1, -1)

        timing = adapted.get("timing", {})
        graph.analysis_duration = torch.tensor(
            [self._safe_float(timing.get("duration"), default=0.0)],
            dtype=torch.float32
        )

        processes = adapted.get("processes", [])
        processtree = adapted.get("processtree", [])
        process_nodes, process_key_to_idx = self._build_process_nodes(processes, processtree)

        (
            api_nodes,
            process_executes_edges,
            temporal_edges,
            temporal_edge_attr,
        ) = self._build_api_and_temporal(processes, process_key_to_idx)

        if len(api_nodes) == 0:
            api_nodes = [self._build_placeholder_api_node(adapted)]
            temporal_edges = []
            temporal_edge_attr = []
            process_executes_edges = self._attach_placeholder_api_to_context(
                process_nodes=process_nodes
            )

        summary = adapted.get("summary", {}) or {}
        raw_network = adapted.get("raw_network", {}) or {}

        (
            file_nodes,
            registry_nodes,
            network_nodes,
            api_file_edges,
            api_registry_edges,
            api_network_edges,
        ) = self._build_resources_and_edges(
            adapted=adapted,
            api_nodes=api_nodes,
            summary=summary,
            raw_network=raw_network,
        )

        self._materialize_api_nodes(graph, api_nodes)
        self._materialize_process_nodes(graph, process_nodes)
        self._materialize_file_nodes(graph, file_nodes)
        self._materialize_registry_nodes(graph, registry_nodes)
        self._materialize_network_nodes(graph, network_nodes)

        self._materialize_edge_index(
            graph,
            ("process", "executes", "api"),
            process_executes_edges,
        )

        self._materialize_edge_index_with_attr(
            graph,
            ("api", "temporal", "api"),
            temporal_edges,
            temporal_edge_attr,
            self.TEMPORAL_BASIC_EDGE_DIM,
        )

        self._materialize_edge_index(
            graph,
            ("api", "operates_on", "file"),
            api_file_edges,
        )

        self._materialize_edge_index(
            graph,
            ("api", "modifies", "registry"),
            api_registry_edges,
        )

        self._materialize_edge_index(
            graph,
            ("api", "connects_to", "network"),
            api_network_edges,
        )

        graph.num_api_nodes = torch.tensor([len(api_nodes)], dtype=torch.long)
        graph.num_process_nodes = torch.tensor([len(process_nodes)], dtype=torch.long)
        graph.num_file_nodes = torch.tensor([len(file_nodes)], dtype=torch.long)
        graph.num_registry_nodes = torch.tensor([len(registry_nodes)], dtype=torch.long)
        graph.num_network_nodes = torch.tensor([len(network_nodes)], dtype=torch.long)

        graph.target_sha256 = adapted.get("target", {}).get("sha256")
        graph.time_fallback_modes = self._collect_time_fallback_modes(api_nodes)

        return graph

    # ============================================================
    # PROCESS NODES
    # ============================================================
    def _build_process_nodes(
        self,
        processes: List[Dict[str, Any]],
        processtree: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[Tuple[Any, ...], int]]:
        seen: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

        for proc in processes:
            first_seen_raw = proc.get("first_seen")
            key = self._make_process_key(
                pid=proc.get("pid"),
                path=proc.get("path"),
                first_seen=first_seen_raw,
            )
            seen[key] = {
                "pid": proc.get("pid"),
                "ppid": proc.get("ppid"),
                "name": proc.get("name") or "",
                "path": proc.get("path") or "",
                "first_seen": self._safe_float(first_seen_raw, default=0.0),
                "_first_seen_raw": first_seen_raw,   # critical fix
                "is_tracked": bool(proc.get("is_tracked")),
                "has_calls": len(proc.get("calls", [])) > 0,
            }

        def walk_tree(nodes: List[Dict[str, Any]]):
            for node in nodes:
                first_seen_raw = node.get("first_seen")
                key = self._make_process_key(
                    pid=node.get("pid"),
                    path=node.get("path"),
                    first_seen=first_seen_raw,
                )
                if key not in seen:
                    seen[key] = {
                        "pid": node.get("pid"),
                        "ppid": node.get("ppid"),
                        "name": node.get("name") or "",
                        "path": node.get("path") or "",
                        "first_seen": self._safe_float(first_seen_raw, default=0.0),
                        "_first_seen_raw": first_seen_raw,   # critical fix
                        "is_tracked": False,
                        "has_calls": False,
                    }
                children = node.get("children", []) or []
                walk_tree(children)

        walk_tree(processtree)

        process_nodes = list(seen.values())
        process_nodes.sort(
            key=lambda p: (
                self._none_last_int(p.get("pid")),
                p.get("first_seen", 0.0),
                p.get("name", ""),
                p.get("path", ""),
            )
        )

        process_key_to_idx = {}
        for idx, proc in enumerate(process_nodes):
            key = self._make_process_key(
                pid=proc.get("pid"),
                path=proc.get("path"),
                first_seen=proc.get("_first_seen_raw"),   # critical fix
            )
            proc["global_process_index"] = idx
            process_key_to_idx[key] = idx

        return process_nodes, process_key_to_idx

    # ============================================================
    # API NODES + TEMPORAL
    # ============================================================
    def _build_api_and_temporal(
        self,
        processes: List[Dict[str, Any]],
        process_key_to_idx: Dict[Tuple[Any, ...], int],
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]], List[Tuple[int, int]], List[List[float]]]:
        api_nodes: List[Dict[str, Any]] = []
        process_executes_edges: List[Tuple[int, int]] = []
        temporal_edges: List[Tuple[int, int]] = []
        temporal_edge_attr: List[List[float]] = []

        global_api_idx = 0
        all_times: List[float] = []

        for proc in processes:
            for call in proc.get("calls", []):
                t = call.get("time")
                if t is not None and math.isfinite(float(t)):
                    all_times.append(float(t))

        min_time = min(all_times) if all_times else 0.0
        max_time = max(all_times) if all_times else 1.0
        time_span = max(max_time - min_time, 1e-6)

        for proc in processes:
            proc_key = self._make_process_key(
                pid=proc.get("pid"),
                path=proc.get("path"),
                first_seen=proc.get("first_seen"),
            )
            process_idx = process_key_to_idx.get(proc_key)

            calls = proc.get("calls", []) or []
            local_api_indices: List[int] = []

            for call in calls:
                t = call.get("time")
                t_float = self._safe_float(t, default=0.0)
                time_norm = (t_float - min_time) / time_span if all_times else 0.0

                api_node = {
                    "global_api_index": global_api_idx,
                    "process_key": proc_key,
                    "process_index": process_idx,
                    "pid": proc.get("pid"),
                    "tid": call.get("tid"),
                    "api_name": call.get("api_name") or "",
                    "category": call.get("category") or "",
                    "status": call.get("status") or "",
                    "time": t_float,
                    "time_source": call.get("time_source") or "missing",
                    "time_norm": float(max(0.0, min(1.0, time_norm))),
                    "event_index": call.get("event_index", 0),
                    "stable_event_index": call.get("stable_event_index", 0),
                    "arguments": call.get("arguments", []) or [],
                    "arguments_text": call.get("arguments_text", "") or "",
                    "is_placeholder": False,
                }
                api_nodes.append(api_node)

                if process_idx is not None:
                    process_executes_edges.append((process_idx, global_api_idx))

                local_api_indices.append(global_api_idx)
                global_api_idx += 1

            if len(local_api_indices) >= 2:
                for left, right in zip(local_api_indices[:-1], local_api_indices[1:]):
                    src = api_nodes[left]
                    dst = api_nodes[right]

                    dt = max(0.0, float(dst["time"]) - float(src["time"]))
                    same_process = 1.0
                    is_burst = 1.0 if dt < cfg.BURST_DELTA_THRESHOLD else 0.0

                    temporal_edges.append((left, right))
                    temporal_edge_attr.append([dt, same_process, is_burst])

        return api_nodes, process_executes_edges, temporal_edges, temporal_edge_attr

    def _build_placeholder_api_node(self, adapted: Dict[str, Any]) -> Dict[str, Any]:
        regime = adapted.get("regime", "unknown")
        return {
            "global_api_index": 0,
            "process_key": None,
            "process_index": None,
            "pid": None,
            "tid": None,
            "api_name": "NO_API_ACTIVITY",
            "category": "placeholder",
            "status": regime,
            "time": 0.0,
            "time_source": "placeholder",
            "time_norm": 0.0,
            "event_index": 0,
            "stable_event_index": 0,
            "arguments": [],
            "arguments_text": "",
            "is_placeholder": True,
        }

    def _attach_placeholder_api_to_context(
        self,
        process_nodes: List[Dict[str, Any]],
    ) -> List[Tuple[int, int]]:
        if not process_nodes:
            return []

        chosen_idx = None
        for idx, proc in enumerate(process_nodes):
            if proc.get("is_tracked") or proc.get("has_calls"):
                chosen_idx = idx
                break
        if chosen_idx is None:
            chosen_idx = 0

        return [(chosen_idx, 0)]

    # ============================================================
    # RESOURCES
    # ============================================================
    def _build_resources_and_edges(
        self,
        adapted: Dict[str, Any],
        api_nodes: List[Dict[str, Any]],
        summary: Dict[str, Any],
        raw_network: Dict[str, Any],
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Tuple[int, int]],
        List[Tuple[int, int]],
        List[Tuple[int, int]],
    ]:
        file_nodes: List[Dict[str, Any]] = []
        registry_nodes: List[Dict[str, Any]] = []
        network_nodes: List[Dict[str, Any]] = []

        file_key_to_idx: Dict[str, int] = {}
        registry_key_to_idx: Dict[str, int] = {}
        network_key_to_idx: Dict[str, int] = {}

        api_file_edges: Set[Tuple[int, int]] = set()
        api_registry_edges: Set[Tuple[int, int]] = set()
        api_network_edges: Set[Tuple[int, int]] = set()

        for fpath in summary.get("files", []) or []:
            norm = self._normalize_file_path(fpath)
            if norm:
                self._ensure_file_node(norm, file_nodes, file_key_to_idx, summary_only=True)

        for rkey in summary.get("registry", []) or []:
            norm = self._normalize_registry_key(rkey)
            if norm:
                self._ensure_registry_node(norm, registry_nodes, registry_key_to_idx, summary_only=True)

        for endpoint in self._extract_summary_network_endpoints(summary.get("network", {}), raw_network):
            norm = self._normalize_network_endpoint(endpoint)
            if norm:
                self._ensure_network_node(norm, network_nodes, network_key_to_idx, summary_only=True)

        for api in api_nodes:
            api_idx = api["global_api_index"]
            args_text = api.get("arguments_text", "")

            for fpath in self._extract_file_paths_from_text(args_text):
                norm = self._normalize_file_path(fpath)
                if not norm:
                    continue
                file_idx = self._ensure_file_node(norm, file_nodes, file_key_to_idx, summary_only=False)
                api_file_edges.add((api_idx, file_idx))

            registry_candidates = self._extract_registry_keys_from_text(args_text)
            if self._api_is_registry_relevant(api):
                for rkey in registry_candidates:
                    norm = self._normalize_registry_key(rkey)
                    if not norm:
                        continue
                    reg_idx = self._ensure_registry_node(norm, registry_nodes, registry_key_to_idx, summary_only=False)
                    api_registry_edges.add((api_idx, reg_idx))

            if self._api_is_network_relevant(api):
                endpoints = self._extract_network_endpoints_from_text(args_text)
                for endpoint in endpoints:
                    norm = self._normalize_network_endpoint(endpoint)
                    if not norm:
                        continue
                    if cfg.FILTER_BACKGROUND_NETWORK and self._is_background_network(norm):
                        continue
                    net_idx = self._ensure_network_node(norm, network_nodes, network_key_to_idx, summary_only=False)
                    api_network_edges.add((api_idx, net_idx))

        return (
            file_nodes,
            registry_nodes,
            network_nodes,
            sorted(api_file_edges),
            sorted(api_registry_edges),
            sorted(api_network_edges),
        )

    # ============================================================
    # NODE MATERIALIZATION
    # ============================================================
    def _materialize_api_nodes(self, graph: HeteroData, api_nodes: List[Dict[str, Any]]) -> None:
        x = []
        timestamps = []
        time_sources = []
        api_names = []
        arguments_texts = []
        categories = []
        statuses = []
        pids = []

        for node in api_nodes:
            api_names.append(node["api_name"])
            arguments_texts.append(node["arguments_text"])
            categories.append(node["category"])
            statuses.append(node["status"])
            time_sources.append(node["time_source"])
            pids.append(self._safe_float(node.get("pid"), default=0.0))

            timestamps.append(float(node["time"]))

            feat = [
                float(node["time_norm"]),
                self._norm_small_int(node.get("stable_event_index")),
                self._norm_small_int(node.get("event_index")),
                self._norm_tid(node.get("tid")),
                self._category_to_scalar(node.get("category", "")),
                1.0 if str(node.get("status", "")).lower() in {"success", "ok"} else 0.0,
                1.0 if node.get("is_placeholder") else 0.0,
            ]
            x.append(feat)

        graph["api"].x = self._tensor_2d(x, self.API_RAW_DIM)
        graph["api"].timestamps = torch.tensor(timestamps, dtype=torch.float32)
        graph["api"].pids = torch.tensor(pids, dtype=torch.float32)
        graph["api"].api_names = api_names
        graph["api"].arguments_text = arguments_texts
        graph["api"].categories = categories
        graph["api"].statuses = statuses
        graph["api"].time_sources = time_sources

    def _materialize_process_nodes(self, graph: HeteroData, process_nodes: List[Dict[str, Any]]) -> None:
        x = []
        names = []
        paths = []
        pids = []

        first_seen_vals = [self._safe_float(p.get("first_seen"), 0.0) for p in process_nodes]
        min_fs = min(first_seen_vals) if first_seen_vals else 0.0
        max_fs = max(first_seen_vals) if first_seen_vals else 1.0
        span_fs = max(max_fs - min_fs, 1e-6)

        for proc in process_nodes:
            names.append(proc.get("name", ""))
            paths.append(proc.get("path", ""))
            pids.append(proc.get("pid"))

            first_seen = self._safe_float(proc.get("first_seen"), 0.0)
            first_seen_norm = (first_seen - min_fs) / span_fs if first_seen_vals else 0.0

            feat = [
                1.0 if proc.get("is_tracked") else 0.0,
                1.0 if proc.get("has_calls") else 0.0,
                1.0 if proc.get("ppid") is not None else 0.0,
                float(max(0.0, min(1.0, first_seen_norm))),
                self._norm_small_int(len(proc.get("name", ""))),
                self._norm_small_int(len(proc.get("path", ""))),
            ]
            x.append(feat)

        graph["process"].x = self._tensor_2d(x, cfg.PROCESS_NODE_DIM)
        graph["process"].process_names = names
        graph["process"].process_paths = paths
        graph["process"].pids = pids

    def _materialize_file_nodes(self, graph: HeteroData, file_nodes: List[Dict[str, Any]]) -> None:
        x = []
        paths = []

        for node in file_nodes:
            path = node["path"]
            paths.append(path)

            ext = Path(path).suffix.lower()
            feat = [
                1.0 if path.lower().startswith("c:\\windows") else 0.0,
                1.0 if "\\temp\\" in path.lower() or path.lower().startswith("%temp%") else 0.0,
                1.0 if ext in {".exe", ".dll"} else 0.0,
                self._norm_small_int(path.count("\\")),
                self._norm_small_int(len(path)),
                1.0 if node.get("summary_only") else 0.0,
            ]
            x.append(feat)

        graph["file"].x = self._tensor_2d(x, cfg.FILE_NODE_DIM)
        graph["file"].paths = paths

    def _materialize_registry_nodes(self, graph: HeteroData, registry_nodes: List[Dict[str, Any]]) -> None:
        x = []
        keys = []

        for node in registry_nodes:
            key = node["key"]
            keys.append(key)
            upper = key.upper()

            feat = [
                1.0 if upper.startswith("HKCU\\") or upper.startswith("HKEY_CURRENT_USER\\") else 0.0,
                1.0 if upper.startswith("HKLM\\") or upper.startswith("HKEY_LOCAL_MACHINE\\") else 0.0,
                1.0 if upper.startswith("HKCR\\") or upper.startswith("HKEY_CLASSES_ROOT\\") else 0.0,
                self._norm_small_int(key.count("\\")),
                self._norm_small_int(len(key)),
                1.0 if node.get("summary_only") else 0.0,
            ]
            x.append(feat)

        graph["registry"].x = self._tensor_2d(x, cfg.REGISTRY_NODE_DIM)
        graph["registry"].keys = keys

    def _materialize_network_nodes(self, graph: HeteroData, network_nodes: List[Dict[str, Any]]) -> None:
        x = []
        endpoints = []

        for node in network_nodes:
            ep = node["endpoint"]
            endpoints.append(ep)

            is_ip = 1.0 if self.ip_re.fullmatch(ep) else 0.0
            is_domain = 1.0 if (not is_ip and self.domain_re.fullmatch(ep)) else 0.0
            is_http_like = 1.0 if ep.lower().startswith(("http://", "https://")) else 0.0
            is_multicast_or_broadcast = 1.0 if ep in cfg.BACKGROUND_IPS else 0.0

            feat = [
                is_ip,
                is_domain,
                is_http_like,
                is_multicast_or_broadcast,
                self._norm_small_int(len(ep)),
                1.0 if node.get("summary_only") else 0.0,
            ]
            x.append(feat)

        graph["network"].x = self._tensor_2d(x, cfg.NETWORK_NODE_DIM)
        graph["network"].endpoints = endpoints

    # ============================================================
    # EDGE MATERIALIZATION
    # ============================================================
    def _materialize_edge_index(
        self,
        graph: HeteroData,
        edge_type: Tuple[str, str, str],
        edges: List[Tuple[int, int]],
    ) -> None:
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        graph[edge_type].edge_index = edge_index

    def _materialize_edge_index_with_attr(
        self,
        graph: HeteroData,
        edge_type: Tuple[str, str, str],
        edges: List[Tuple[int, int]],
        edge_attr: List[List[float]],
        attr_dim: int,
    ) -> None:
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr_t = torch.empty((0, attr_dim), dtype=torch.float32)

        graph[edge_type].edge_index = edge_index
        graph[edge_type].edge_attr = edge_attr_t

    # ============================================================
    # RESOURCE HELPERS
    # ============================================================
    def _ensure_file_node(
        self,
        path: str,
        nodes: List[Dict[str, Any]],
        key_to_idx: Dict[str, int],
        summary_only: bool,
    ) -> int:
        if path not in key_to_idx:
            idx = len(nodes)
            key_to_idx[path] = idx
            nodes.append({"path": path, "summary_only": summary_only})
        else:
            idx = key_to_idx[path]
            if not summary_only:
                nodes[idx]["summary_only"] = False
        return idx

    def _ensure_registry_node(
        self,
        key: str,
        nodes: List[Dict[str, Any]],
        key_to_idx: Dict[str, int],
        summary_only: bool,
    ) -> int:
        if key not in key_to_idx:
            idx = len(nodes)
            key_to_idx[key] = idx
            nodes.append({"key": key, "summary_only": summary_only})
        else:
            idx = key_to_idx[key]
            if not summary_only:
                nodes[idx]["summary_only"] = False
        return idx

    def _ensure_network_node(
        self,
        endpoint: str,
        nodes: List[Dict[str, Any]],
        key_to_idx: Dict[str, int],
        summary_only: bool,
    ) -> int:
        if endpoint not in key_to_idx:
            idx = len(nodes)
            key_to_idx[endpoint] = idx
            nodes.append({"endpoint": endpoint, "summary_only": summary_only})
        else:
            idx = key_to_idx[endpoint]
            if not summary_only:
                nodes[idx]["summary_only"] = False
        return idx

    def _normalize_file_path(self, value: str) -> Optional[str]:
        if not value:
            return None
        path = str(value).strip().strip('"').strip("'")
        path = path.replace("/", "\\")
        path = re.sub(r"\\{2,}", r"\\", path)
        return path if path else None

    def _normalize_registry_key(self, value: str) -> Optional[str]:
        if not value:
            return None
        key = str(value).strip().strip('"').strip("'")
        key = key.replace("/", "\\")
        key = re.sub(r"\\{2,}", r"\\", key)
        return key.upper()

    def _normalize_network_endpoint(self, value: str) -> Optional[str]:
        if not value:
            return None
        ep = str(value).strip().strip('"').strip("'")
        return ep.lower()

    def _extract_file_paths_from_text(self, text: str) -> List[str]:
        found = self.file_path_re.findall(text or "")
        return self._dedupe_preserve_order(found)

    def _extract_registry_keys_from_text(self, text: str) -> List[str]:
        found = self.registry_re.findall(text or "")
        return self._dedupe_preserve_order(found)

    def _extract_network_endpoints_from_text(self, text: str) -> List[str]:
        text = text or ""
        found: List[str] = []
        found.extend(self.url_re.findall(text))
        found.extend(self.ip_re.findall(text))
        found.extend(self.domain_re.findall(text))
        return self._dedupe_preserve_order(found)

    def _extract_summary_network_endpoints(self, summary_network: Dict[str, Any], raw_network: Dict[str, Any]) -> List[str]:
        endpoints: List[str] = []

        for item in (summary_network.get("domains", []) or []):
            if isinstance(item, dict):
                val = item.get("domain") or item.get("hostname")
                if val:
                    endpoints.append(str(val))
            else:
                endpoints.append(str(item))

        for item in (summary_network.get("hosts", []) or []):
            if isinstance(item, dict):
                val = item.get("ip") or item.get("hostname")
                if val:
                    endpoints.append(str(val))
            else:
                endpoints.append(str(item))

        for proto_key in ("udp", "tcp"):
            for item in (summary_network.get(proto_key, []) or []):
                if isinstance(item, dict):
                    for k in ("dst", "host", "ip", "domain"):
                        if item.get(k):
                            endpoints.append(str(item[k]))

        for item in (raw_network.get("domains", []) or []):
            if isinstance(item, dict) and item.get("domain"):
                endpoints.append(str(item["domain"]))
        for item in (raw_network.get("hosts", []) or []):
            if isinstance(item, dict):
                if item.get("ip"):
                    endpoints.append(str(item["ip"]))
                if item.get("hostname"):
                    endpoints.append(str(item["hostname"]))

        return self._dedupe_preserve_order(endpoints)

    def _api_is_registry_relevant(self, api: Dict[str, Any]) -> bool:
        name = (api.get("api_name") or "").lower()
        cat = (api.get("category") or "").lower()
        return ("reg" in name or "key" in name or "registry" in cat)

    def _api_is_network_relevant(self, api: Dict[str, Any]) -> bool:
        name = (api.get("api_name") or "").lower()
        cat = (api.get("category") or "").lower()
        args_text = (api.get("arguments_text") or "").lower()

        return (
            "socket" in name
            or "connect" in name
            or "send" in name
            or "recv" in name
            or "dns" in name
            or "winhttp" in name
            or "ws2_" in name
            or "internet" in name
            or cat in {"network", "internet", "socket"}
            or bool(self.ip_re.search(args_text))
            or bool(self.domain_re.search(args_text))
            or bool(self.url_re.search(args_text))
        )

    def _is_background_network(self, endpoint: str) -> bool:
        ep = endpoint.lower()
        return ep in {x.lower() for x in cfg.BACKGROUND_IPS} or ep in {x.lower() for x in cfg.BACKGROUND_HOSTS}

    # ============================================================
    # GRAPH-LEVEL ACTIVITY FEATURES
    # ============================================================
    def _build_activity_feature_vector(self, adapted: Dict[str, Any], activity: Dict[str, Any]) -> List[float]:
        regime = adapted.get("regime", "unknown")
        return [
            math.log1p(self._safe_float(activity.get("activity_score"), 0.0)),
            1.0 if regime == "rich" else 0.0,
            1.0 if regime == "low_activity" else 0.0,
            1.0 if regime == "very_sparse" else 0.0,
            math.log1p(self._safe_float(activity.get("num_api_calls"), 0.0)),
            math.log1p(self._safe_float(activity.get("num_processes"), 0.0)),
            math.log1p(self._safe_float(activity.get("num_file_events"), 0.0)),
            math.log1p(self._safe_float(activity.get("num_registry_events"), 0.0)),
            math.log1p(self._safe_float(activity.get("num_network_events"), 0.0)),
        ]

    def _collect_time_fallback_modes(self, api_nodes: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = defaultdict(int)
        for node in api_nodes:
            counts[node.get("time_source", "missing")] += 1
        return dict(counts)

    # ============================================================
    # UTILS
    # ============================================================
    def save_graph(self, graph: HeteroData, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _tensor_2d(self, rows: List[List[float]], dim: int) -> torch.Tensor:
        if not rows:
            return torch.empty((0, dim), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    def _make_process_key(
        self,
        pid: Optional[int],
        path: Optional[str],
        first_seen: Optional[float],
    ) -> Tuple[Any, ...]:
        return (
            pid if cfg.USE_PID_AS_PRIMARY_PROCESS_KEY else None,
            (path or "").lower() if cfg.USE_PROCESS_PATH_FOR_DISAMBIGUATION else "",
            round(float(first_seen), 6) if (cfg.USE_FIRST_SEEN_FOR_DISAMBIGUATION and first_seen is not None) else None,
        )

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _none_last_int(self, value: Any) -> int:
        if value is None:
            return 10**12
        try:
            return int(value)
        except (TypeError, ValueError):
            return 10**12

    def _norm_small_int(self, value: Any, cap: float = 10000.0) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        v = max(0.0, min(v, cap))
        return v / cap

    def _norm_tid(self, value: Any) -> float:
        return self._norm_small_int(value, cap=100000.0)

    def _category_to_scalar(self, category: str) -> float:
        category = (category or "").strip().lower()
        mapping = {
            "filesystem": 0.2,
            "file": 0.2,
            "registry": 0.4,
            "network": 0.6,
            "process": 0.8,
            "system": 1.0,
            "memory": 0.5,
            "": 0.0,
        }
        return mapping.get(category, 0.1)

    def _dedupe_preserve_order(self, values: List[str]) -> List[str]:
        seen = set()
        out = []
        for v in values:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out


if __name__ == "__main__":
    builder = GraphBuilderV2()
    files = sorted(cfg.ADAPTED_REPORTS_DIR.glob("*.json"))

    if cfg.DEBUG_MAX_REPORTS is not None:
        files = files[: cfg.DEBUG_MAX_REPORTS]

    print(f"[build_graphs_v2] Found {len(files)} adapted report(s) in {cfg.ADAPTED_REPORTS_DIR}")

    for i, path in enumerate(files, start=1):
        with open(path, "r", encoding="utf-8") as f:
            adapted = json.load(f)

        graph = builder.build_graph(adapted)

        if i <= 3:
            print(f"\n--- Sample {i}: {path.name} ---")
            print(f"graph_id: {graph.graph_id}")
            print(f"regime: {graph.regime}")
            print(f"api nodes: {graph['api'].x.shape[0]}")
            print(f"process nodes: {graph['process'].x.shape[0]}")
            print(f"file nodes: {graph['file'].x.shape[0]}")
            print(f"registry nodes: {graph['registry'].x.shape[0]}")
            print(f"network nodes: {graph['network'].x.shape[0]}")
            print(f"temporal edges: {graph['api', 'temporal', 'api'].edge_index.shape[1]}")
            print(f"file edges: {graph['api', 'operates_on', 'file'].edge_index.shape[1]}")
            print(f"registry edges: {graph['api', 'modifies', 'registry'].edge_index.shape[1]}")
            print(f"network edges: {graph['api', 'connects_to', 'network'].edge_index.shape[1]}")



# 	1.	pipeline_config_v2.py — Defines the new raw input path, new enriched output source name, thresholds, and v2-specific constants without touching the current pipeline.
# Keeps: the current config idea of source-based switching, parallel graph roots, and schema-per-source handling.
# Improves: isolates the new pipeline cleanly, adds explicit thresholds for low-activity handling, timing fallback policy, and stable resource-dimension rules so the old inconsistencies are not repeated.
# 	2.	report_adapter_v2.py — Reads the new raw sandbox JSONs and converts their fields into one canonical internal representation for graph construction.
# Keeps: the current idea of parsing sandbox reports into a graph-friendly intermediate form before graph creation.
# Improves: explicitly normalizes the new raw report style, preserves per-call time and event order, unifies argument formats, and handles richer benign reports plus sparse benign reports without losing temporal information. --- done 
# 	3.	build_graphs_v2.py — Builds the initial heterogeneous graph in one pass, including API/process/file/registry/network nodes, per-process temporal order, and all base edges.
# Keeps: the current graph families the model already expects: api, process, file, registry, network, plus the core relations already used by the model.
# Improves: fixes current shortcomings by enforcing per-process temporal sequencing, creating real api::modifies::registry edges, creating real api::connects_to::network edges where attribution is valid, stabilizing resource-node construction, and explicitly tagging low-activity graphs instead of letting them silently look broken. -- done - will check - bug fixed
# 	4.	enrich_graphs_v2.py — Adds node embeddings and temporal edge features while enforcing the existing contracts of api=775 and temporal edge attrs =14.
# Keeps: the current enrichment contract the model already works with, especially 775-dimensional API node features and 14-dimensional temporal edge features.
# Improves: ensures resource-node dimensions are made stable per type, preserves time-derived edge semantics more carefully, and enriches the improved graph structure rather than decorating a structurally weaker graph. -- done -- bug fixed
# 	5.	run_pipeline_v2.py — Orchestrates the full new flow: adapt raw report, build graph, enrich graph, and save into the new enriched source folder.
# Keeps: the current end-to-end pipeline philosophy of structured preprocessing before training.
# Improves: gives you one clean entry point for the new parallel source, making it easy to generate the new enriched cache without touching the existing pipeline.
 

# So next -- enrich_graphs_v2.py create a bug free code and take everything to context and add all the needs checks 