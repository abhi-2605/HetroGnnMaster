"""
run_pipeline_v2.py

End-to-end runner for the new parallel raw-report -> graph -> enriched-graph pipeline.

Purpose
-------
This script orchestrates the full v2 flow:

1. discover raw sandbox JSON reports from cfg.RAW_REPORTS_DIR
2. create labels CSV on the fly if needed
3. adapt reports into a canonical representation
4. optionally save adapted reports for debugging
5. build initial raw heterogeneous graphs
6. optionally save raw graphs
7. enrich graphs into final schema-ready graphs
8. optionally run schema inspection on the new enriched source

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
import json
import pickle
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

        if len(raw_files) == 0:
            raise FileNotFoundError(f"No .json reports found under {cfg.RAW_REPORTS_DIR}")

        # Create/recreate labels CSV on the fly, then cache lookup once
        self._ensure_labels_csv(raw_files)
        self._label_lookup = self._load_label_map()

        for idx, raw_path in enumerate(raw_files, start=1):
            adapted = None
            graph = None
            enriched = None

            try:
                label_name, label_id = self._resolve_label(raw_path)

                adapted = self.adapter.adapt_file(
                    path=raw_path,
                    label_name=label_name,
                    label_id=label_id,
                )

                self._update_report_stats(adapted)

                if cfg.SAVE_ADAPTED_REPORTS:
                    self._save_adapted_report(adapted)

                if adapted["regime"] == "invalid" and cfg.SKIP_INVALID_REPORTS:
                    if cfg.VERBOSE:
                        print(f"[run_pipeline_v2] Skipping invalid report: {raw_path.name}")
                    self.stats.raw_reports_skipped += 1
                    del adapted
                    continue

                graph = self.builder.build_graph(adapted)

                if cfg.SAVE_RAW_GRAPHS:
                    self._save_raw_graph(graph, raw_path)

                enriched = self.enricher.enrich_graph(graph)

                if cfg.SAVE_ENRICHED_GRAPHS:
                    self._save_enriched_graph(enriched, adapted, raw_path)

                self._update_graph_stats(enriched)

                self.stats.raw_reports_processed += 1

                if cfg.VERBOSE and (idx % max(1, cfg.DEBUG_PRINT_EVERY) == 0):
                    print(f"[run_pipeline_v2] processed {idx}/{len(raw_files)}")

            except Exception as e:
                self.stats.raw_reports_failed += 1
                print(f"[run_pipeline_v2] FAILED {raw_path.name}: {e}")
                if cfg.VERBOSE:
                    traceback.print_exc()

                if cfg.FAIL_ON_DIM_MISMATCH and isinstance(e, (ValueError, RuntimeError)):
                    self._save_pipeline_stats()
                    raise

            finally:
                # Explicit release for memory-heavy processing
                del adapted, graph, enriched

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

    def _discover_raw_reports(self) -> List[Path]:
        # Recursive because reports may be under benign/ malware/ ransomware folders
        files = sorted(cfg.RAW_REPORTS_DIR.rglob("*.json"))
        if cfg.DEBUG_MAX_REPORTS is not None:
            files = files[: cfg.DEBUG_MAX_REPORTS]
        return files

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

        # safer to raise than silently mislabel
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
        """
        Resolve label from cached label lookup:
        - stem first
        - filename second
        """
        label_name = self._label_lookup.get(raw_path.stem) or self._label_lookup.get(raw_path.name)
        if label_name is None:
            return None, None

        label_id = cfg.LABEL_NAME_TO_ID.get(label_name.lower())
        return label_name, label_id

    # ============================================================
    # SAVE HELPERS
    # ============================================================
    def _make_unique_id(self, raw_path: Path, report_id: str) -> str:
        """
        Avoid collisions when two reports in different subfolders share the same stem.
        Example:
            benign/abc123.json      -> benign__abc123
            malware/abc123.json     -> malware__abc123
        """
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

    def _save_enriched_graph(self, graph: Any, adapted: Dict[str, Any], raw_path: Path) -> None:
        unique_id = self._make_unique_id(raw_path, adapted["report_id"])
        out = cfg.ENRICHED_GRAPHS_DIR / "all" / f"{unique_id}.pkl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.stats.enriched_graphs_saved += 1

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
        """
        Reuse existing schema stack after enriched graphs are written.
        """
        try:
            from utils.schema_manager import ensure_active_schema
        except Exception as e:
            print(f"[run_pipeline_v2] Could not import schema manager: {e}")
            return

        try:
            ensure_active_schema(
                auto_refresh=True,
                graph_root=cfg.ENRICHED_GRAPHS_DIR,
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
        print(f"RAW_REPORTS_DIR     : {cfg.RAW_REPORTS_DIR}")
        print(f"RAW_LABELS_CSV      : {cfg.RAW_LABELS_CSV}")
        print(f"ADAPTED_REPORTS_DIR : {cfg.ADAPTED_REPORTS_DIR}")
        print(f"RAW_GRAPHS_DIR      : {cfg.RAW_GRAPHS_DIR}")
        print(f"ENRICHED_GRAPHS_DIR : {cfg.ENRICHED_GRAPHS_DIR}")
        print(f"SCHEMA_JSON_PATH    : {cfg.SCHEMA_JSON_PATH}")
        print("=" * 80)

    def _print_summary(self) -> None:
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)
        print(f"Raw reports found        : {self.stats.raw_reports_found}")
        print(f"Raw reports processed    : {self.stats.raw_reports_processed}")
        print(f"Raw reports skipped      : {self.stats.raw_reports_skipped}")
        print(f"Raw reports failed       : {self.stats.raw_reports_failed}")
        print(f"Adapted reports saved    : {self.stats.adapted_saved}")
        print(f"Raw graphs saved         : {self.stats.raw_graphs_saved}")
        print(f"Enriched graphs saved    : {self.stats.enriched_graphs_saved}")
        print(f"Labels CSV created       : {self.stats.labels_csv_created}")
        print(f"Rich reports             : {self.stats.rich_reports}")
        print(f"Low activity reports     : {self.stats.low_activity_reports}")
        print(f"Very sparse reports      : {self.stats.very_sparse_reports}")
        print(f"Invalid reports          : {self.stats.invalid_reports}")
        print(f"Placeholder graphs       : {self.stats.placeholder_graphs}")
        print(f"Graphs with reg edges    : {self.stats.graphs_with_registry_edges}")
        print(f"Graphs with net edges    : {self.stats.graphs_with_network_edges}")
        print(f"Graphs with 0 temp edges : {self.stats.graphs_with_zero_temporal_edges}")
        print(f"Label ransomware         : {self.stats.label_ransomware}")
        print(f"Label benign             : {self.stats.label_benign}")
        print(f"Label malware            : {self.stats.label_malware}")
        print(f"Label unknown            : {self.stats.label_unknown}")
        print(f"Schema inspection ran    : {self.stats.schema_inspection_ran}")
        print("=" * 80)


if __name__ == "__main__":
    runner = PipelineRunnerV2()
    runner.run()