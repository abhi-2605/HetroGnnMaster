"""
report_adapter_v2.py

Adapter for new raw sandbox JSON reports.

Purpose
-------
This module converts the new raw sandbox-report style into a single
canonical internal representation that downstream graph construction
can use consistently.

It is intentionally upstream of graph building.

Keeps from current pipeline
---------------------------
- parse report into a graph-friendly intermediate structure first
- preserve the core behavioral families: process / api / file / registry / network

Improves over current pipeline
------------------------------
- explicitly normalizes raw Cuckoo/CAPE-style field differences
- preserves per-call time and deterministic event order
- supports both rich and sparse benign reports
- unifies argument formats
- attaches regime/activity metadata early
- records timestamp fallback mode for later temporal feature handling

Important contract for build_graphs_v2.py
-----------------------------------------
- Process nodes must be created even when a process has no calls.
- `is_tracked` only means "has API calls to contribute", not "exists as a process".
- `stable_event_index` is local within a process stream.
- Graph builder must assign separate global API node indices across all processes.

Canonical representation
------------------------
Returned dict shape:

{
    "report_id": str,
    "label": Optional[int],
    "label_name": Optional[str],
    "source_format": str,
    "regime": str,
    "activity": {...},
    "timing": {...},
    "target": {...},
    "processes": [
        {
            "pid": int,
            "ppid": Optional[int],
            "name": str,
            "path": str,
            "first_seen": Optional[float],
            "is_tracked": bool,
            "calls": [
                {
                    "event_index": int,          # original local index before sorting
                    "stable_event_index": int,   # local per-process order after sorting
                    "api_name": str,
                    "category": str,
                    "status": str,
                    "tid": Optional[int],
                    "time": Optional[float],
                    "time_source": str,
                    "arguments": List[Dict[str, Any]],
                    "arguments_text": str,
                },
                ...
            ]
        },
        ...
    ],
    "processtree": [...],
    "summary": {
        "files": [...],
        "registry": [...],
        "network": {...}
    },
    "raw_network": {...},
    "metadata": {...}
}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline_v2 import pipeline_config_v2 as cfg


class ReportAdapterV2:
    """Convert new raw sandbox JSON into a canonical representation."""

    def __init__(self, labels_map: Optional[Dict[str, int]] = None):
        self.labels_map = labels_map or dict(cfg.LABEL_NAME_TO_ID)

    # ============================================================
    # PUBLIC API
    # ============================================================
    def load_json(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def adapt_file(
        self,
        path: Path,
        label_name: Optional[str] = None,
        label_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        raw = self.load_json(path)
        return self.adapt_report(
            raw=raw,
            report_id=path.stem,
            label_name=label_name,
            label_id=label_id,
            source_path=str(path),
        )

    def adapt_report(
        self,
        raw: Dict[str, Any],
        report_id: str,
        label_name: Optional[str] = None,
        label_id: Optional[int] = None,
        source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        source_format = self.detect_source_format(raw)

        if label_id is None and label_name is not None:
            label_id = self.labels_map.get(label_name.lower())

        info = raw.get("info", {}) if isinstance(raw.get("info"), dict) else {}
        target = raw.get("target", {}) if isinstance(raw.get("target"), dict) else {}
        behavior = raw.get("behavior", {}) if isinstance(behavior := raw.get("behavior"), dict) else {}
        raw_summary = behavior.get("summary", {}) if isinstance(behavior.get("summary"), dict) else {}
        raw_processes = behavior.get("processes", []) if isinstance(behavior.get("processes"), list) else []
        raw_processtree = behavior.get("processtree", []) if isinstance(behavior.get("processtree"), list) else []
        raw_network = raw.get("network", {}) if isinstance(raw.get("network"), dict) else {}

        timing = self._extract_report_timing(info)
        processes = self._adapt_processes(raw_processes)
        processtree = self._adapt_processtree(raw_processtree)
        summary = self._adapt_summary(raw_summary, raw_network)
        activity = self._compute_activity(processes, processtree, summary, timing)
        regime = self._assign_regime(activity)

        adapted = {
            "report_id": report_id,
            "source_path": source_path,
            "label": label_id,
            "label_name": label_name,
            "source_format": source_format,
            "regime": regime,
            "activity": activity,
            "timing": timing,
            "target": self._adapt_target(target),
            "processes": processes,
            "processtree": processtree,
            "summary": summary,
            "raw_network": raw_network,
            "metadata": {
                "score": info.get("score"),
                "duration": timing.get("duration"),
                "machine": info.get("machine", {}),
            },
        }

        return adapted

    # ============================================================
    # DETECTION
    # ============================================================
    def detect_source_format(self, raw: Dict[str, Any]) -> str:
        """Lightweight format detector."""
        behavior = raw.get("behavior", {})
        processes = behavior.get("processes", []) if isinstance(behavior, dict) else []

        if processes:
            sample_proc = processes[0]
            if "process_id" in sample_proc or "parent_id" in sample_proc:
                return "graph_ready_style"
            if "pid" in sample_proc or "ppid" in sample_proc:
                return "cuckoo_cape_raw_style"

        if "info" in raw and "target" in raw:
            return "cuckoo_cape_raw_style"

        return "unknown"

    # ============================================================
    # TARGET / TIMING
    # ============================================================
    def _adapt_target(self, target: Dict[str, Any]) -> Dict[str, Any]:
        file_info = target.get("file", {}) if isinstance(target.get("file"), dict) else {}
        return {
            "name": file_info.get("name"),
            "path": file_info.get("path"),
            "size": file_info.get("size"),
            "md5": file_info.get("md5"),
            "sha1": file_info.get("sha1"),
            "sha256": file_info.get("sha256"),
            "type": file_info.get("type"),
        }

    def _extract_report_timing(self, info: Dict[str, Any]) -> Dict[str, Any]:
        started = info.get("started")
        ended = info.get("ended")
        duration = info.get("duration")
        machine = info.get("machine", {}) if isinstance(info.get("machine"), dict) else {}

        return {
            "analysis_started": started,
            "analysis_ended": ended,
            "duration": duration,
            "machine_started_on": machine.get("started_on"),
            "machine_shutdown_on": machine.get("shutdown_on"),
        }

    # ============================================================
    # PROCESSES / CALLS
    # ============================================================
    def _adapt_processes(self, raw_processes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        adapted = []

        for proc in raw_processes:
            pid = self._to_int(proc.get("pid", proc.get("process_id")))
            ppid = self._to_int(proc.get("ppid", proc.get("parent_id")))

            name = proc.get("process_name") or proc.get("name") or ""
            path = proc.get("process_path") or proc.get("module_path") or ""
            first_seen = self._coerce_float(proc.get("first_seen"))

            raw_calls = proc.get("calls", [])
            calls = self._adapt_calls(raw_calls, first_seen=first_seen)

            adapted.append(
                {
                    "pid": pid,
                    "ppid": ppid,
                    "name": name,
                    "path": path,
                    "first_seen": first_seen,
                    "command_line": proc.get("command_line"),
                    # NOTE:
                    # A process with no calls must still survive into graph building.
                    # `is_tracked` only tells downstream code whether this process
                    # contributes API nodes to the temporal backbone.
                    "is_tracked": bool(raw_calls),
                    "calls": calls,
                }
            )

        return adapted

    def _adapt_calls(
        self,
        raw_calls: List[Dict[str, Any]],
        first_seen: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        adapted_calls: List[Dict[str, Any]] = []

        for idx, call in enumerate(raw_calls):
            api_name = call.get("api") or call.get("name") or ""
            category = str(call.get("category", "") or "")
            status = str(call.get("status", "") or "")
            tid = self._to_int(call.get("tid"))
            raw_time = call.get("time", call.get("timestamp"))

            time_value, time_source = self._normalize_call_time(
                raw_time=raw_time,
                first_seen=first_seen,
                event_index=idx,
            )

            normalized_args = self._normalize_arguments(call.get("arguments"))
            arguments_text = self._arguments_to_text(normalized_args)

            adapted_calls.append(
                {
                    "event_index": idx,  # original local index before sort
                    "api_name": api_name,
                    "category": category,
                    "status": status,
                    "tid": tid,
                    "time": time_value,
                    "time_source": time_source,
                    "arguments": normalized_args,
                    "arguments_text": arguments_text,
                }
            )

        # Deterministic local per-process ordering.
        adapted_calls.sort(
            key=lambda x: (
                float("inf") if x["time"] is None else x["time"],
                x["event_index"],
            )
        )

        # IMPORTANT:
        # stable_event_index is LOCAL to this process only.
        # build_graphs_v2.py must assign separate GLOBAL api node ids.
        for stable_idx, call in enumerate(adapted_calls):
            call["stable_event_index"] = stable_idx

        return adapted_calls

    def _normalize_call_time(
    self,
    raw_time: Any,
    first_seen: Optional[float],
    event_index: int,
) -> Tuple[Optional[float], str]:
        """
        Priority:
        1. per-call numeric time
        2. process first_seen + monotonic index
        3. pure event_index
        """
        if cfg.USE_CALL_TIME_IF_PRESENT:
            time_val = self._coerce_float(raw_time)
            if time_val is not None:
                return time_val, "call_time"
            if raw_time is not None and raw_time != "": 
                # distinguish "present but unparseable" from "missing"
                if cfg.USE_PROCESS_FIRST_SEEN_FALLBACK and first_seen is not None:
                    return float(first_seen) + (event_index * 1e-6), "call_time_unparseable_first_seen_plus_index"
                if cfg.USE_EVENT_INDEX_FALLBACK:
                    return float(event_index), "call_time_unparseable_event_index"
                return None, "call_time_unparseable"

        if cfg.USE_PROCESS_FIRST_SEEN_FALLBACK and first_seen is not None:
            return float(first_seen) + (event_index * 1e-6), "first_seen_plus_index"

        if cfg.USE_EVENT_INDEX_FALLBACK:
            return float(event_index), "event_index"

        return None, "missing"

    # ============================================================
    # ARGUMENTS
    # ============================================================
    def _normalize_arguments(self, raw_args: Any) -> List[Dict[str, Any]]:
        """
        Normalize arguments to:
        [
            {"name": <str>, "value": <serializable value>},
            ...
        ]
        """
        if raw_args is None:
            return []

        if isinstance(raw_args, list):
            normalized = []
            for item in raw_args:
                if isinstance(item, dict):
                    name = str(item.get("name", "arg"))
                    value = item.get("value")
                    normalized.append({"name": name, "value": value})
                else:
                    normalized.append({"name": "arg", "value": item})
            return normalized

        if isinstance(raw_args, dict):
            items = []
            for k in sorted(raw_args.keys(), key=str):
                items.append({"name": str(k), "value": raw_args[k]})
            return items

        return [{"name": "arg", "value": raw_args}]

    def _arguments_to_text(self, args: List[Dict[str, Any]]) -> str:
        parts = []
        for item in args:
            name = str(item.get("name", "arg"))
            value = self._safe_string(item.get("value"))
            parts.append(f"{name}={value}")
        return " | ".join(parts)

    # ============================================================
    # PROCESS TREE
    # ============================================================
    def _adapt_processtree(self, raw_tree: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def normalize_node(node: Dict[str, Any]) -> Dict[str, Any]:
            children = node.get("children", []) if isinstance(node.get("children"), list) else []
            return {
                "pid": self._to_int(node.get("pid", node.get("process_id"))),
                "ppid": self._to_int(node.get("ppid", node.get("parent_id"))),
                "name": node.get("name") or node.get("process_name") or "",
                "path": node.get("process_path") or node.get("module_path") or "",
                "first_seen": self._coerce_float(node.get("first_seen")),
                "children": [normalize_node(child) for child in children if isinstance(child, dict)],
            }

        return [normalize_node(node) for node in raw_tree if isinstance(node, dict)]

    # ============================================================
    # SUMMARY / RESOURCES
    # ============================================================
    def _adapt_summary(self, raw_summary: Dict[str, Any], raw_network: Dict[str, Any]) -> Dict[str, Any]:
        file_items = []
        registry_items = []

        for key in [
            "files",
            "file_read",
            "file_written",
            "file_opened",
            "file_created",
            "file_exists",
            "read_files",
            "write_files",
            "delete_files",
        ]:
            values = raw_summary.get(key, [])
            if isinstance(values, list):
                file_items.extend(values)

        for key in [
            "keys",
            "regkey_opened",
            "regkey_read",
            "regkey_written",
            "regkey_deleted",
            "read_keys",
            "write_keys",
            "delete_keys",
        ]:
            values = raw_summary.get(key, [])
            if isinstance(values, list):
                registry_items.extend(values)

        file_items = self._dedupe_preserve_order([self._safe_string(x) for x in file_items if x is not None])
        registry_items = self._dedupe_preserve_order([self._safe_string(x) for x in registry_items if x is not None])

        return {
            "files": file_items,
            "registry": registry_items,
            "network": self._summarize_network(raw_network),
        }

    def _summarize_network(self, raw_network: Dict[str, Any]) -> Dict[str, Any]:
        udp = raw_network.get("udp", []) if isinstance(raw_network.get("udp"), list) else []
        tcp = raw_network.get("tcp", []) if isinstance(raw_network.get("tcp"), list) else []
        dns = raw_network.get("dns", []) if isinstance(raw_network.get("dns"), list) else []
        http = raw_network.get("http", []) if isinstance(raw_network.get("http"), list) else []
        hosts = raw_network.get("hosts", []) if isinstance(raw_network.get("hosts"), list) else []
        domains = raw_network.get("domains", []) if isinstance(raw_network.get("domains"), list) else []

        return {
            "udp_count": len(udp),
            "tcp_count": len(tcp),
            "dns_count": len(dns),
            "http_count": len(http),
            "host_count": len(hosts),
            "domain_count": len(domains),
            "udp": udp,
            "tcp": tcp,
            "dns": dns,
            "http": http,
            "hosts": hosts,
            "domains": domains,
        }

    # ============================================================
    # ACTIVITY / REGIME
    # ============================================================
    def _compute_activity(
        self,
        processes: List[Dict[str, Any]],
        processtree: List[Dict[str, Any]],
        summary: Dict[str, Any],
        timing: Dict[str, Any],
    ) -> Dict[str, Any]:
        num_processes = len(processes)
        num_process_tree_nodes = self._count_tree_nodes(processtree)
        num_api_calls = sum(len(proc.get("calls", [])) for proc in processes)
        num_file_events = len(summary.get("files", []))
        num_registry_events = len(summary.get("registry", []))
        net = summary.get("network", {})
        num_network_events = (
            int(net.get("udp_count", 0))
            + int(net.get("tcp_count", 0))
            + int(net.get("dns_count", 0))
            + int(net.get("http_count", 0))
        )

        activity_score = (
            num_api_calls
            + num_file_events
            + num_registry_events
            + num_network_events
            + num_process_tree_nodes
        )

        return {
            "num_processes": num_processes,
            "num_process_tree_nodes": num_process_tree_nodes,
            "num_api_calls": num_api_calls,
            "num_file_events": num_file_events,
            "num_registry_events": num_registry_events,
            "num_network_events": num_network_events,
            "duration": timing.get("duration"),
            "activity_score": activity_score,
        }

    def _assign_regime(self, activity: Dict[str, Any]) -> str:
        """
        Invalid if either structural indicator is too weak,
        unless API evidence is strong enough to compensate.
        """
        num_processes = activity["num_processes"]
        num_tree = activity["num_process_tree_nodes"]
        num_api = activity["num_api_calls"]
        num_resources = (
            activity["num_file_events"]
            + activity["num_registry_events"]
            + activity["num_network_events"]
        )

        if (
            num_processes < cfg.MIN_PROCESSES_FOR_VALID
            or num_tree < cfg.MIN_PROCESS_TREE_NODES_FOR_VALID
        ):
            if num_api < cfg.MIN_API_CALLS_LOW_ACTIVITY:
                return "invalid"

        if num_api >= cfg.MIN_API_CALLS_RICH or num_resources >= cfg.MIN_RESOURCE_EVENTS_RICH:
            return "rich"

        if num_api >= cfg.MIN_API_CALLS_LOW_ACTIVITY or num_resources >= cfg.MIN_RESOURCE_EVENTS_LOW_ACTIVITY:
            return "low_activity"

        return "very_sparse" if cfg.ALLOW_VERY_SPARSE_REPORTS else "invalid"

    # ============================================================
    # HELPERS
    # ============================================================
    def _count_tree_nodes(self, nodes: List[Dict[str, Any]]) -> int:
        total = 0
        for node in nodes:
            total += 1
            children = node.get("children", [])
            if children:
                total += self._count_tree_nodes(children)
        return total

    def _to_int(self, value: Any) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_float(self, value: Any) -> Optional[float]:
        """
        Handles:
        - numeric values
        - numeric strings
        - non-numeric timestamps -> None
        """
        if value is None or value == "":
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            v = value.strip()
            if re.fullmatch(r"[-+]?\d+(\.\d+)?", v):
                try:
                    return float(v)
                except ValueError:
                    return None

        return None

    def _safe_string(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        except TypeError:
            return str(value)

    def _dedupe_preserve_order(self, values: List[str]) -> List[str]:
        seen = set()
        out = []
        for v in values:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out


if __name__ == "__main__":
    adapter = ReportAdapterV2()
    sample_dir = cfg.RAW_REPORTS_DIR

    files = sorted(sample_dir.glob("*.json"))
    if cfg.DEBUG_MAX_REPORTS is not None:
        files = files[: cfg.DEBUG_MAX_REPORTS]

    print(f"[report_adapter_v2] Found {len(files)} report(s) in {sample_dir}")

    for i, path in enumerate(files, start=1):
        adapted = adapter.adapt_file(path)
        if i <= 3:
            print(f"\n--- Sample {i}: {path.name} ---")
            print(json.dumps({
                "report_id": adapted["report_id"],
                "source_format": adapted["source_format"],
                "regime": adapted["regime"],
                "activity": adapted["activity"],
                "num_processes": len(adapted["processes"]),
                "num_tree_roots": len(adapted["processtree"]),
            }, indent=2))