"""
run_pipeline_v2.py

End-to-end runner for the new parallel raw-report -> graph -> enriched-graph pipeline.

Purpose
-------
This script orchestrates the full v2 flow:

1. discover raw sandbox JSON reports from cfg.RAW_REPORTS_DIR
2. create labels CSV on the fly if needed
3. skip reports already listed in an external CSV md5 column
4. adapt reports into a canonical representation
5. optionally save adapted reports for debugging
6. build initial raw heterogeneous graphs
7. optionally save raw graphs
8. enrich graphs into final schema-ready graphs
9. optionally run schema inspection on the new enriched source

Important notes
---------------
- Raw sample locations come from pipeline_config_v2.py
- This runner does NOT train the model; it prepares the new enriched source
- After successful completion, point ACTIVE_GRAPH_SOURCE to the new source
  in the main config and use the existing training entry point
- Reports are processed one at a time to keep memory usage low
"""

from __future__ import annotations

import csv
import gc
import json
import pickle
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from pipeline_v2 import pipeline_config_v2 as cfg
from pipeline_v2.report_adapter_v2 import ReportAdapterV2
from pipeline_v2.build_graphs_v2 import GraphBuilderV2
from pipeline_v2.enrich_graphs_v2 import GraphEnricherV2


@dataclass
class PipelineStatsV2:
    raw_reports_found: int = 0
    raw_reports_processed: int = 0
    raw_reports_failed: int = 0
    raw_reports_skipped: int = 0

    skipped_already_present: int = 0
    skipped_invalid_reports: int = 0
    skipped_memory_issues: int = 0
    skipped_low_activity: int = 0

    adapted_saved: int = 0
    raw_graphs_saved: int = 0
    enriched_graphs_saved: int = 0

    rich_reports: int = 0
    low_activity_reports: int = 0
    very_sparse_reports: int = 0
    invalid_reports: int = 0

    placeholder_graphs: int = 0

    graphs_with_registry_edges: int = 0
    graphs_with_network_edges: int = 0
    graphs_with_zero_temporal_edges: int = 0

    label_ransomware: int = 0
    label_benign: int = 0
    label_malware: int = 0
    label_unknown: int = 0

    labels_csv_created: int = 0
    schema_inspection_ran: int = 0


class PipelineRunnerV2:
    def __init__(self):
        # Correct adapter mapping: label_name -> label_id
        self.adapter = ReportAdapterV2(labels_map=dict(cfg.LABEL_NAME_TO_ID))
        self.builder = GraphBuilderV2()
        self.enricher = GraphEnricherV2(
            raw_graphs_dir=cfg.RAW_GRAPHS_DIR,
            enriched_graphs_dir=cfg.ENRICHED_GRAPHS_DIR,
            embedding_batch_size=128,
            burst_threshold=getattr(cfg, "BURST_DELTA_THRESHOLD", 0.1),
            window_size=5,
        )
        self.stats = PipelineStatsV2()

        # Cached once, not reloaded for every sample
        self._label_lookup: Dict[str, str] = {}

        # Controlled fully from config
        self.max_successful_graphs: Optional[int] = getattr(cfg, "MAX_SUCCESSFUL_GRAPHS", None)

        # External CSV used to skip already-converted samples
        self.skip_existing_md5_csv: Optional[Path] = self._get_skip_csv_path()
        self._existing_md5s: Set[str] = set()

    # ============================================================
    # PUBLIC ENTRY POINT
    # ============================================================
    def run(self) -> None:
        self._print_header()
        self._validate_inputs()

        raw_files = self._discover_raw_reports()
        self.stats.raw_reports_found = len(raw_files)

        if cfg.VERBOSE:
            print(f"[run_pipeline_v2] Found {len(raw_files)} raw report(s) under {cfg.RAW_REPORTS_DIR}")
            if self.max_successful_graphs is not None:
                print(f"[run_pipeline_v2] Target successful graphs: {self.max_successful_graphs}")

        if len(raw_files) == 0:
            raise FileNotFoundError(f"No .json reports found under {cfg.RAW_REPORTS_DIR}")

        # Load md5 skip list once
        self._existing_md5s = self._load_existing_md5s()

        # Create/recreate labels CSV on the fly, then cache lookup once
        self._ensure_labels_csv(raw_files)
        self._label_lookup = self._load_label_map()

        for idx, raw_path in enumerate(raw_files, start=1):
            if self._target_reached():
                if cfg.VERBOSE:
                    print(
                        f"[run_pipeline_v2] Reached target of "
                        f"{self.max_successful_graphs} successful graph(s). Stopping."
                    )
                break

            adapted = None
            graph = None
            enriched = None

            try:
                # Skip if already present in external md5 CSV
                if self._should_skip_existing(raw_path):
                    self.stats.raw_reports_skipped += 1
                    self.stats.skipped_already_present += 1
                    print(
                        f"[run_pipeline_v2] Skipping {raw_path.name}: "
                        f"already present in external md5 CSV"
                    )
                    continue

                label_name, label_id = self._resolve_label(raw_path)

                adapted = self.adapter.adapt_file(
                    path=raw_path,
                    label_name=label_name,
                    label_id=label_id,
                )

                self._update_report_stats(adapted)

                if cfg.SAVE_ADAPTED_REPORTS:
                    self._save_adapted_report(adapted)

                if adapted.get("regime") == "invalid" and cfg.SKIP_INVALID_REPORTS:
                    self.stats.raw_reports_skipped += 1
                    self.stats.skipped_invalid_reports += 1
                    print(f"[run_pipeline_v2] Skipping {raw_path.name}: invalid report")
                    del adapted
                    adapted = None
                    continue

                graph = self.builder.build_graph(adapted)

                # Skip low-activity / failed-detonation graphs (no behaviour to
                # detect) BEFORE the expensive enrichment, so they are never
                # produced and the run moves on to the next report.
                if getattr(cfg, "SKIP_LOW_ACTIVITY_GRAPHS", False):
                    n_api = int(graph["api"].x.shape[0]) if "api" in graph.node_types else 0
                    if n_api <= int(getattr(cfg, "MIN_API_NODES_TO_BUILD", 10)):
                        self.stats.raw_reports_skipped += 1
                        self.stats.skipped_low_activity += 1
                        self._record_low_activity(adapted.get("report_id", raw_path.stem), n_api)
                        print(f"[run_pipeline_v2] Skipping {raw_path.name}: low activity ({n_api} api nodes)")
                        continue

                if cfg.SAVE_RAW_GRAPHS:
                    self._save_raw_graph(graph, raw_path)

                enriched = self.enricher.enrich_graph(graph)

                if cfg.SAVE_ENRICHED_GRAPHS:
                    self._save_enriched_graph(enriched, adapted, raw_path)

                self._update_graph_stats(enriched)
                self.stats.raw_reports_processed += 1

                if cfg.VERBOSE and (idx % max(1, cfg.DEBUG_PRINT_EVERY) == 0):
                    if self.max_successful_graphs is not None:
                        print(
                            f"[run_pipeline_v2] scanned {idx}/{len(raw_files)} | "
                            f"successful graphs: {self.stats.raw_reports_processed}/{self.max_successful_graphs}"
                        )
                    else:
                        print(f"[run_pipeline_v2] processed {idx}/{len(raw_files)}")

            except Exception as e:
                if self._is_memory_or_semaphore_error(e):
                    self.stats.raw_reports_skipped += 1
                    self.stats.skipped_memory_issues += 1
                    print(
                        f"[run_pipeline_v2] Skipping {raw_path.name}: "
                        f"memory/semaphore issue -> {e}"
                    )
                    if cfg.VERBOSE:
                        traceback.print_exc()
                    self._cleanup_memory()
                    continue

                # A malformed/corrupt report (bad JSON, bad unicode escape) must
                # be SKIPPED, not fatal — only genuine dim/contract mismatches in
                # graph building should abort the run under FAIL_ON_DIM_MISMATCH.
                if isinstance(e, (json.JSONDecodeError, UnicodeDecodeError)):
                    self.stats.raw_reports_skipped += 1
                    self.stats.skipped_invalid_reports += 1
                    print(f"[run_pipeline_v2] Skipping {raw_path.name}: malformed report -> {e}")
                    self._cleanup_memory()
                    continue

                self.stats.raw_reports_failed += 1
                print(f"[run_pipeline_v2] FAILED {raw_path.name}: {e}")
                if cfg.VERBOSE:
                    traceback.print_exc()

                if cfg.FAIL_ON_DIM_MISMATCH and isinstance(e, (ValueError, RuntimeError)):
                    self._save_pipeline_stats()
                    raise

            finally:
                del adapted, graph, enriched
                self._cleanup_memory()

        if cfg.RUN_SCHEMA_INSPECTION_AFTER_BUILD:
            self._run_schema_inspection()

        self._save_pipeline_stats()
        self._print_summary()

    # ============================================================
    # INPUT / DISCOVERY
    # ============================================================
    def _validate_inputs(self) -> None:
        if not cfg.RAW_REPORTS_DIR.exists():
            raise FileNotFoundError(f"RAW_REPORTS_DIR does not exist: {cfg.RAW_REPORTS_DIR}")

        if self.skip_existing_md5_csv is not None and not self.skip_existing_md5_csv.exists():
            raise FileNotFoundError(
                f"SKIP_EXISTING_MD5_CSV does not exist: {self.skip_existing_md5_csv}"
            )

    def _discover_raw_reports(self) -> List[Path]:
        # Recursive because reports may be under benign/ malware/ ransomware folders
        return sorted(cfg.RAW_REPORTS_DIR.rglob("*.json"))

    def _target_reached(self) -> bool:
        return (
            self.max_successful_graphs is not None
            and self.stats.raw_reports_processed >= self.max_successful_graphs
        )

    # ============================================================
    # SKIP-CSV LOGIC
    # ============================================================
    def _get_skip_csv_path(self) -> Optional[Path]:
        raw_value = getattr(cfg, "DATA_METADATA_PATH", None)
        if not raw_value:
            return None
        return Path(raw_value)

    def _load_existing_md5s(self) -> Set[str]:
        if self.skip_existing_md5_csv is None:
            if cfg.VERBOSE:
                print("[run_pipeline_v2] No SKIP_EXISTING_MD5_CSV configured. No md5-based skipping.")
            return set()

        csv_path = self.skip_existing_md5_csv
        md5s: Set[str] = set()

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {csv_path}")

            cols = {c.lower().strip(): c for c in reader.fieldnames}
            md5_col = cols.get("md5")
            if md5_col is None:
                raise ValueError(
                    f"CSV must contain an 'md5' column. Found columns: {reader.fieldnames}"
                )

            for row in reader:
                md5_val = str(row.get(md5_col, "")).strip().lower()
                if md5_val:
                    md5s.add(md5_val)

        if cfg.VERBOSE:
            print(
                f"[run_pipeline_v2] Loaded {len(md5s)} existing md5 value(s) "
                f"from {csv_path}"
            )

        return md5s

    def _should_skip_existing(self, raw_path: Path) -> bool:
        if not self._existing_md5s:
            return False
        return raw_path.stem.strip().lower() in self._existing_md5s

    # ============================================================
    # LABEL CSV CREATION / LOADING
    # ============================================================
    def _ensure_labels_csv(self, raw_files: List[Path]) -> None:
        """
        Create labels CSV on the fly if missing or stale.

        Important:
        - safe for debug runs followed by full runs
        - regenerates if discovered files are not fully represented
        - writes rows one at a time (streaming-friendly)
        """
        csv_path = Path(cfg.RAW_LABELS_CSV)

        if csv_path.exists() and csv_path.stat().st_size > 0:
            existing_lookup = self._load_label_map()
            if all((p.stem in existing_lookup or p.name in existing_lookup) for p in raw_files):
                return
            if cfg.VERBOSE:
                print("[run_pipeline_v2] Re-creating labels CSV because file membership changed.")

        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["report_id", "filename", "label"])
            writer.writeheader()

            for path in raw_files:
                writer.writerow(
                    {
                        "report_id": path.stem,
                        "filename": path.name,
                        "label": self._infer_label_from_path(path),
                    }
                )

        self.stats.labels_csv_created = 1

        if cfg.VERBOSE:
            print(f"[run_pipeline_v2] Created/Re-created labels CSV: {csv_path}")

    def _infer_label_from_path(self, path: Path) -> str:
        parts = {p.lower() for p in path.parts}

        if {"benign", "ben_reports", "benignv2", "ben"} & parts:
            return "benign"

        if {"ransomware", "ransom", "ransomware_reports"} & parts:
            return "ransomware"

        if {"malware", "mal_reports", "mal"} & parts:
            return "malware"

        raise ValueError(f"Could not infer label from path: {path}")

    def _load_label_map(self) -> Dict[str, str]:
        """
        Returns:
            Dict[report_id_or_filename -> label_name]
        """
        label_map: Dict[str, str] = {}

        csv_path = Path(cfg.RAW_LABELS_CSV)
        if not csv_path.exists():
            return label_map

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in (reader.fieldnames or [])}

            id_col = None
            fname_col = None
            label_col = None

            for candidate in ("report_id", "sha256", "stem"):
                if candidate in cols:
                    id_col = cols[candidate]
                    break

            for candidate in ("filename", "file"):
                if candidate in cols:
                    fname_col = cols[candidate]
                    break

            for candidate in ("label", "label_name", "class"):
                if candidate in cols:
                    label_col = cols[candidate]
                    break

            if label_col is None:
                print("[run_pipeline_v2] WARNING: labels CSV has no label column.")
                return {}

            if id_col is None and fname_col is None:
                print(
                    "[run_pipeline_v2] WARNING: labels CSV has no recognised ID column "
                    "(expected one of: report_id, sha256, stem, filename, file). "
                    "All labels will resolve as unknown."
                )
                return {}

            for row in reader:
                label_val = str(row.get(label_col, "")).strip().lower()
                if not label_val:
                    continue

                if id_col:
                    key = str(row.get(id_col, "")).strip()
                    if key:
                        label_map[key] = label_val

                if fname_col:
                    key = str(row.get(fname_col, "")).strip()
                    if key:
                        label_map[key] = label_val

        return label_map

    def _resolve_label(self, raw_path: Path) -> Tuple[Optional[str], Optional[int]]:
        label_name = self._label_lookup.get(raw_path.stem) or self._label_lookup.get(raw_path.name)
        if label_name is None:
            return None, None

        label_id = cfg.LABEL_NAME_TO_ID.get(label_name.lower())
        return label_name, label_id

    # ============================================================
    # SAVE HELPERS
    # ============================================================
    def _make_unique_id(self, raw_path: Path, report_id: str) -> str:
        # Clean "<md5>" names so they match metadata md5s and the eval's
        # disjoint-leakage check (which compares bare filename stems).
        if getattr(cfg, "STRIP_REPORT_PREFIX_ON_SAVE", True):
            return report_id
        parent_name = raw_path.parent.name
        return f"{parent_name}__{report_id}"

    def _save_adapted_report(self, adapted: Dict[str, Any]) -> None:
        out = cfg.ADAPTED_REPORTS_DIR / f"{adapted['report_id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(adapted, f, indent=2)
        self.stats.adapted_saved += 1

    def _save_raw_graph(self, graph: Any, raw_path: Path) -> None:
        unique_id = self._make_unique_id(raw_path, graph.graph_id)
        out = cfg.RAW_GRAPHS_DIR / "all" / f"{unique_id}.pkl"
        out.parent.mkdir(parents=True, exist_ok=True)
        self.builder.save_graph(graph, out)
        self.stats.raw_graphs_saved += 1

    def _record_low_activity(self, md5: str, n_api: int) -> None:
        """Append an excluded low-activity sample to the log (idempotent header)."""
        path = Path(getattr(cfg, "LOW_ACTIVITY_EXCLUDED_PATH", cfg.DATA_DIR / "low_activity_excluded_md5s.csv"))
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["md5", "api_nodes", "reason"])
            w.writerow([md5, n_api, "low_activity_<=MIN_API_NODES_TO_BUILD"])

    def _conform_to_training_schema(self, graph: Any) -> Any:
        """Strip v2-only build-time attributes so the saved graph's attribute
        set matches the training graphs exactly (required for PyG batching with
        train/val/test/old graphs). Safe: called only at save time, after
        enrichment has already consumed timestamps/pids/api_names/etc."""
        for nt in list(graph.node_types):
            allow = {"x", "api_names", "num_nodes"} if nt == "api" else {"x", "num_nodes"}
            for k in list(graph[nt].keys()):
                if k not in allow:
                    del graph[nt][k]
        for et in list(graph.edge_types):
            for k in list(graph[et].keys()):
                if k not in {"edge_index", "edge_attr"}:
                    del graph[et][k]
        gs = graph._global_store
        keep_global = {"graph_features", "num_apis", "num_files",
                       "num_network", "num_processes", "num_registry"}
        for k in list(gs.keys()):
            if k not in keep_global:
                del gs[k]
        return graph

    def _save_enriched_graph(self, graph: Any, adapted: Dict[str, Any], raw_path: Path) -> None:
        unique_id = self._make_unique_id(raw_path, adapted["report_id"])
        if getattr(cfg, "CONFORM_TO_TRAINING_SCHEMA", True):
            graph = self._conform_to_training_schema(graph)
        # Write straight into the zero-day bucket the model's eval reads from.
        out = cfg.ZERO_DAY_OUTPUT_DIR / f"{unique_id}.pkl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.stats.enriched_graphs_saved += 1

    # ============================================================
    # ERROR / MEMORY HANDLING
    # ============================================================
    def _is_memory_or_semaphore_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(
            token in msg
            for token in (
                "semaphore",
                "sem_lock",
                "semlock",
                "cannot allocate memory",
                "out of memory",
                "cuda out of memory",
                "oom",
                "resource temporarily unavailable",
                "killed",
                "bus error",
            )
        )

    def _cleanup_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ============================================================
    # STATS
    # ============================================================
    def _update_report_stats(self, adapted: Dict[str, Any]) -> None:
        regime = adapted.get("regime", "unknown")
        if regime == "rich":
            self.stats.rich_reports += 1
        elif regime == "low_activity":
            self.stats.low_activity_reports += 1
        elif regime == "very_sparse":
            self.stats.very_sparse_reports += 1
        elif regime == "invalid":
            self.stats.invalid_reports += 1

        label_name = str(adapted.get("label_name") or "").strip().lower()
        if label_name == "ransomware":
            self.stats.label_ransomware += 1
        elif label_name == "benign":
            self.stats.label_benign += 1
        elif label_name == "malware":
            self.stats.label_malware += 1
        else:
            self.stats.label_unknown += 1

    def _update_graph_stats(self, graph: Any) -> None:
        placeholder_count = int(
            getattr(graph["api"], "placeholder_count", torch.tensor([0]))[0].item()
        )
        if placeholder_count > 0:
            self.stats.placeholder_graphs += 1

        if ("api", "modifies", "registry") in graph.edge_types:
            if graph["api", "modifies", "registry"].edge_index.shape[1] > 0:
                self.stats.graphs_with_registry_edges += 1

        if ("api", "connects_to", "network") in graph.edge_types:
            if graph["api", "connects_to", "network"].edge_index.shape[1] > 0:
                self.stats.graphs_with_network_edges += 1

        if ("api", "temporal", "api") in graph.edge_types:
            if graph["api", "temporal", "api"].edge_index.shape[1] == 0:
                self.stats.graphs_with_zero_temporal_edges += 1

    # ============================================================
    # SCHEMA
    # ============================================================
    def _run_schema_inspection(self) -> None:
        try:
            from utils.schema_manager import ensure_active_schema
        except Exception as e:
            print(f"[run_pipeline_v2] Could not import schema manager: {e}")
            return

        try:
            ensure_active_schema(
                auto_refresh=True,
                graph_root=cfg.ZERO_DAY_OUTPUT_DIR,
                schema_path=cfg.SCHEMA_JSON_PATH,
                schema_summary_path=cfg.SCHEMA_SUMMARY_PATH,
                source_name=cfg.ENRICHED_SOURCE_NAME,
            )
            self.stats.schema_inspection_ran = 1
        except Exception as e:
            print(f"[run_pipeline_v2] Schema inspection failed: {e}")
            if cfg.VERBOSE:
                traceback.print_exc()
            if cfg.FAIL_ON_DIM_MISMATCH:
                self._save_pipeline_stats()
                raise

    # ============================================================
    # AUDIT / LOGGING
    # ============================================================
    def _save_pipeline_stats(self) -> None:
        out = cfg.PIPELINE_AUDIT_DIR / "run_pipeline_stats_v2.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(asdict(self.stats), f, indent=2)

    def _print_header(self) -> None:
        print("=" * 80)
        print("RUN PIPELINE V2")
        print("=" * 80)
        print(f"RAW_REPORTS_DIR         : {cfg.RAW_REPORTS_DIR}")
        print(f"RAW_LABELS_CSV          : {cfg.RAW_LABELS_CSV}")
        print(f"SKIP_EXISTING_MD5_CSV   : {getattr(cfg, 'SKIP_EXISTING_MD5_CSV', None)}")
        print(f"ADAPTED_REPORTS_DIR     : {cfg.ADAPTED_REPORTS_DIR}")
        print(f"RAW_GRAPHS_DIR          : {cfg.RAW_GRAPHS_DIR}")
        print(f"ENRICHED_GRAPHS_DIR     : {cfg.ENRICHED_GRAPHS_DIR}")
        print(f"ZERO_DAY_OUTPUT_DIR     : {cfg.ZERO_DAY_OUTPUT_DIR}  (variant={cfg.ZERO_DAY_VARIANT})")
        print(f"SCHEMA_JSON_PATH        : {cfg.SCHEMA_JSON_PATH}")
        print(f"MAX_SUCCESSFUL_GRAPHS   : {getattr(cfg, 'MAX_SUCCESSFUL_GRAPHS', None)}")
        print("=" * 80)

    def _print_summary(self) -> None:
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)
        print(f"Raw reports found           : {self.stats.raw_reports_found}")
        print(f"Raw reports processed       : {self.stats.raw_reports_processed}")
        print(f"Raw reports skipped         : {self.stats.raw_reports_skipped}")
        print(f"  - already present         : {self.stats.skipped_already_present}")
        print(f"  - invalid reports         : {self.stats.skipped_invalid_reports}")
        print(f"  - memory/semaphore issues : {self.stats.skipped_memory_issues}")
        print(f"  - low activity            : {self.stats.skipped_low_activity}")
        print(f"Raw reports failed          : {self.stats.raw_reports_failed}")
        print(f"Adapted reports saved       : {self.stats.adapted_saved}")
        print(f"Raw graphs saved            : {self.stats.raw_graphs_saved}")
        print(f"Enriched graphs saved       : {self.stats.enriched_graphs_saved}")
        print(f"Labels CSV created          : {self.stats.labels_csv_created}")
        print(f"Rich reports                : {self.stats.rich_reports}")
        print(f"Low activity reports        : {self.stats.low_activity_reports}")
        print(f"Very sparse reports         : {self.stats.very_sparse_reports}")
        print(f"Invalid reports             : {self.stats.invalid_reports}")
        print(f"Placeholder graphs          : {self.stats.placeholder_graphs}")
        print(f"Graphs with reg edges       : {self.stats.graphs_with_registry_edges}")
        print(f"Graphs with net edges       : {self.stats.graphs_with_network_edges}")
        print(f"Graphs with 0 temp edges    : {self.stats.graphs_with_zero_temporal_edges}")
        print(f"Label ransomware            : {self.stats.label_ransomware}")
        print(f"Label benign                : {self.stats.label_benign}")
        print(f"Label malware               : {self.stats.label_malware}")
        print(f"Label unknown               : {self.stats.label_unknown}")
        print(f"Schema inspection ran       : {self.stats.schema_inspection_ran}")
        print("=" * 80)


if __name__ == "__main__":
    runner = PipelineRunnerV2()
    runner.run()