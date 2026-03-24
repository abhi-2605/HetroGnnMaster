"""
enrich_graphs_v2.py

Enrich raw graphs produced by build_graphs_v2.py.

Purpose
-------
This module converts raw structural graphs into final enriched graphs that are
compatible with the existing schema-aware training pipeline.

Keeps from current pipeline
---------------------------
- API node feature contract: 775 dims
- temporal edge feature contract: 14 dims
- resource node families remain file / registry / process / network
- only temporal edges carry edge attributes

Improves over current pipeline
------------------------------
- enforces stable resource dims per type
- validates final dimensions before save
- keeps sparse / placeholder graphs valid but auditable
- records enrichment statistics for later schema/debug inspection
- expands 3-dim temporal edges into the 14-dim contract used previously

Expected raw graph inputs
-------------------------
- api.x: [N, 7] raw structural API features
- api.timestamps: [N]
- api.api_names / arguments_text / categories / statuses
- temporal edge_attr: [E, 3]  = [time_delta, same_process, is_burst]
- resource nodes already exist with 6-dim raw features (or are normalizable to 6)

Output enriched graph contracts
-------------------------------
- api.x: [N, 775] = [768 semantic embedding | 7 raw structural features]
- ('api','temporal','api').edge_attr: [E, 14]
- process/file/registry/network.x: stable 6 dims each

Notes
-----
- This file is intentionally self-contained.
- This pipeline requires sentence-transformers/all-mpnet-base-v2.
- If the semantic model cannot be loaded, enrichment stops with a hard error.
- No fallback embedding is allowed because semantic API features are mandatory.
"""

from __future__ import annotations

import json
import pickle
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from pipeline_v2 import pipeline_config_v2 as cfg


# ============================================================
# EMBEDDING BACKEND
# ============================================================
class APIEmbedder:
    """
    Semantic embedder for API nodes.

    Required backend:
        sentence-transformers/all-mpnet-base-v2  -> 768 dims

    No fallback is allowed, because semantic embeddings are a required
    part of the final API feature contract.
    """

    def __init__(self, batch_size: int = 128):
        self.batch_size = batch_size
        self.embedding_dim = 768
        self.backend = "sentence_transformer"
        self.model = None

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        except Exception as e:
            raise RuntimeError(
                "Required semantic embedding model could not be loaded.\n"
                "This pipeline requires sentence-transformers with "
                "'sentence-transformers/all-mpnet-base-v2' available.\n"
                "No fallback embedding is allowed because API semantic "
                "embeddings are mandatory for enrichment.\n"
                f"Original error: {e}"
            ) from e


    def _coerce_text(self, x) -> str:
        import json
        import math
        import numpy as np

        if x is None:
            return ""

        if isinstance(x, str):
            return x

        if isinstance(x, bytes):
            try:
                return x.decode("utf-8", errors="replace")
            except Exception:
                return str(x)

        if isinstance(x, np.generic):
            return str(x.item())

        if isinstance(x, np.ndarray):
            try:
                return json.dumps(x.tolist(), ensure_ascii=False, sort_keys=True, default=str)
            except Exception:
                return str(x.tolist())

        if isinstance(x, (list, tuple, dict)):
            try:
                return json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)
            except Exception:
                return str(x)

        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return ""
            return str(x)

        return str(x)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        import json
        import numpy as np

        cleaned: List[str] = []
        bad_precheck = []

        for i, t in enumerate(texts):
            try:
                ct = self._coerce_text(t)
            except Exception as e:
                bad_precheck.append(
                    {
                        "index": i,
                        "stage": "coerce_text",
                        "type": str(type(t)),
                        "repr": repr(t)[:1000],
                        "error": str(e),
                    }
                )
                ct = ""

            if not isinstance(ct, str):
                bad_precheck.append(
                    {
                        "index": i,
                        "stage": "post_coerce_not_str",
                        "type": str(type(ct)),
                        "repr": repr(ct)[:1000],
                    }
                )
                ct = str(ct)

            cleaned.append(ct)

        if bad_precheck:
            raise TypeError(
                "Bad embedding inputs detected before model.encode():\n"
                + json.dumps(bad_precheck[:10], indent=2)
            )

        if not cleaned:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        try:
            vecs = self.model.encode(
                cleaned,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except Exception as batch_error:
            # isolate exact offending item
            isolated = []
            for i, t in enumerate(cleaned):
                try:
                    _ = self.model.encode(
                        [t],
                        batch_size=1,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                    )
                except Exception as single_error:
                    isolated.append(
                        {
                            "index": i,
                            "type": str(type(t)),
                            "length": len(t),
                            "repr": repr(t)[:2000],
                            "error": str(single_error),
                        }
                    )
                    if len(isolated) >= 5:
                        break

            raise TypeError(
                "Batch embedding failed. Isolated problematic text entries:\n"
                + json.dumps(isolated, indent=2)
                + f"\nOriginal batch error: {batch_error}"
            ) from batch_error

        vecs = np.asarray(vecs, dtype=np.float32)

        if vecs.ndim != 2 or vecs.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Unexpected embedding shape from backend={self.backend}: {vecs.shape}, "
                f"expected (*, {self.embedding_dim})"
            )

        return vecs


# ============================================================
# STATS
# ============================================================
@dataclass
class EnrichmentStats:
    total_graphs_seen: int = 0
    total_graphs_saved: int = 0
    failed_graphs: int = 0

    api_nodes_total: int = 0
    placeholder_api_nodes_total: int = 0

    temporal_edges_total: int = 0
    graphs_with_zero_temporal_edges: int = 0

    graphs_with_registry_edges: int = 0
    graphs_with_network_edges: int = 0

    graphs_with_call_time: int = 0
    graphs_with_time_fallback: int = 0

    backend: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_graphs_seen": self.total_graphs_seen,
            "total_graphs_saved": self.total_graphs_saved,
            "failed_graphs": self.failed_graphs,
            "api_nodes_total": self.api_nodes_total,
            "placeholder_api_nodes_total": self.placeholder_api_nodes_total,
            "temporal_edges_total": self.temporal_edges_total,
            "graphs_with_zero_temporal_edges": self.graphs_with_zero_temporal_edges,
            "graphs_with_registry_edges": self.graphs_with_registry_edges,
            "graphs_with_network_edges": self.graphs_with_network_edges,
            "graphs_with_call_time": self.graphs_with_call_time,
            "graphs_with_time_fallback": self.graphs_with_time_fallback,
            "backend": self.backend,
        }


# ============================================================
# MAIN ENRICHER
# ============================================================
class GraphEnricherV2:
    def __init__(
        self,
        raw_graphs_dir: Optional[Path] = None,
        enriched_graphs_dir: Optional[Path] = None,
        embedding_batch_size: int = 128,
        burst_threshold: float = 0.1,
        window_size: int = 5,
    ):
        self.raw_graphs_dir = Path(raw_graphs_dir or cfg.RAW_GRAPHS_DIR)
        self.enriched_graphs_dir = Path(enriched_graphs_dir or cfg.ENRICHED_GRAPHS_DIR)

        self.embedder = APIEmbedder(batch_size=embedding_batch_size)
        self.burst_threshold = burst_threshold
        self.window_size = window_size

        self.stats = EnrichmentStats(backend=self.embedder.backend)
    def _sanitize_for_embedding(self, s: str, max_len: int = 512) -> str:
        """
        Make text safe and semantically useful for sentence-transformer encoding.

        - removes null bytes
        - removes private-use / control-like binary junk
        - collapses escaped binary payloads
        - truncates long strings
        """
        import re
        import unicodedata

        if s is None:
            return ""

        s = self.embedder._coerce_text(s)

        # First collapse obvious binary payload fields like value=...
        s = self._collapse_binary_value_field(s)

        # Remove null bytes
        s = s.replace("\x00", " ")

        cleaned_chars = []
        weird_count = 0

        for ch in s:
            code = ord(ch)
            cat = unicodedata.category(ch)

            # Remove most control characters except basic whitespace
            if cat.startswith("C") and ch not in ("\t", "\n", "\r", " "):
                weird_count += 1
                continue

            # Remove private use characters
            if 0xE000 <= code <= 0xF8FF:
                weird_count += 1
                continue

            # Remove explicit noncharacters
            if code in (0xFFFE, 0xFFFF):
                weird_count += 1
                continue

            cleaned_chars.append(ch)

        s = "".join(cleaned_chars)
        s = re.sub(r"\s+", " ", s).strip()

        # If it still looks like escaped binary, collapse it
        if s.count("\\x") > 20:
            return f"<escaped_binary_blob len={len(s)}>"

        # If very weird overall, replace with summary
        if weird_count > 20:
            return f"<binary_like_text len={len(s)}>"

        if len(s) > max_len:
            s = s[:max_len] + " ...<truncated>"

        return s
    
    def _collapse_binary_value_field(self, s: str) -> str:
        """
        Replace huge binary-like registry value dumps with a compact placeholder.
        """
        import re

        if not isinstance(s, str):
            s = str(s)

        # Common bad case: value=<lots of escaped binary junk>
        if "value=" in s and s.count("\\x") > 20:
            s = re.sub(r"value=[^|]*", "value=<binary_blob>", s)

        return s
    
    def _build_api_embedding_text(
    self,
    api_name: str,
    category: str,
    status: str,
    args: str,
) -> str:
        """
        Build the exact text that will be embedded for one API node.
        """
        api_name = self._sanitize_for_embedding(self._safe_field(api_name), max_len=128)
        category = self._sanitize_for_embedding(self._safe_field(category), max_len=64)
        status = self._sanitize_for_embedding(self._safe_field(status), max_len=64)
        args = self._sanitize_for_embedding(self._safe_field(args), max_len=512)

        return f"api={api_name} | category={category} | status={status} | args={args}"
    # ========================================================
    # PUBLIC API
    # ========================================================
    def enrich_graph(self, graph: HeteroData) -> HeteroData:
        """
        Enrich a single raw graph into final training-ready form.
        """
        graph = self._clone_graph(graph)

        self.stats.total_graphs_seen += 1

        self._normalize_resource_dims(graph)
        self._enrich_api_nodes(graph)
        self._enrich_temporal_edges(graph)
        self._validate_graph_contract(graph)
        self._update_stats(graph)

        return graph

    def run_directory(self) -> None:
        """
        Run enrichment over raw graph directories.
        Supports split subfolders if present.
        """
        split_dirs = self._discover_split_dirs(self.raw_graphs_dir)
        if not split_dirs:
            raise FileNotFoundError(f"No raw graph .pkl files found under: {self.raw_graphs_dir}")

        for split_name, in_dir in split_dirs:
            out_dir = self.enriched_graphs_dir / split_name
            out_dir.mkdir(parents=True, exist_ok=True)

            graph_files = sorted(in_dir.glob("*.pkl"))
            if cfg.DEBUG_MAX_REPORTS is not None:
                graph_files = graph_files[: cfg.DEBUG_MAX_REPORTS]

            if cfg.VERBOSE:
                print(f"[enrich_graphs_v2] Split={split_name} files={len(graph_files)}")

            for i, path in enumerate(graph_files, start=1):
                try:
                    graph = self._load_graph(path)
                    enriched = self.enrich_graph(graph)
                    self._save_graph(enriched, out_dir / path.name)
                    self.stats.total_graphs_saved += 1

                    if cfg.VERBOSE and (i % max(1, cfg.DEBUG_PRINT_EVERY) == 0):
                        print(f"  processed {i}/{len(graph_files)}")

                except Exception as e:
                    self.stats.failed_graphs += 1
                    print(f"[enrich_graphs_v2] FAILED {path.name}: {e}")
                    if cfg.VERBOSE:
                        traceback.print_exc()

                    # Flush partial stats before any hard abort
                    if cfg.FAIL_ON_DIM_MISMATCH and isinstance(e, (ValueError, RuntimeError)):
                        self._save_stats()
                        raise

        self._save_stats()

    # ========================================================
    # API NODE ENRICHMENT
    # ========================================================

    def _safe_field(self, x) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, bytes):
            try:
                return x.decode("utf-8", errors="replace")
            except Exception:
                return str(x)
        return str(x)

    def _enrich_api_nodes(self, graph: HeteroData) -> None:
        if "api" not in graph.node_types:
            raise ValueError("Graph missing required node type: api")

        raw_x = graph["api"].x
        if raw_x.ndim != 2:
            raise ValueError(f"api.x must be 2D, got shape={tuple(raw_x.shape)}")
        if raw_x.shape[1] != 7:
            raise ValueError(
                f"Raw api.x must have 7 dims before enrichment, got {raw_x.shape[1]}"
            )

        api_names = list(getattr(graph["api"], "api_names", []))
        arg_texts = list(getattr(graph["api"], "arguments_text", []))
        categories = list(getattr(graph["api"], "categories", []))
        statuses = list(getattr(graph["api"], "statuses", []))

        num_api = raw_x.shape[0]
        texts = []
        debug_rows = []
        placeholder_count = 0

        for i in range(num_api):
            raw_feat = raw_x[i]
            is_placeholder = bool(raw_feat[6].item() > 0)

            if is_placeholder:
                placeholder_count += 1
                text = "placeholder api node"
            else:
                api_name = api_names[i] if i < len(api_names) else ""
                args = arg_texts[i] if i < len(arg_texts) else ""
                cat = categories[i] if i < len(categories) else ""
                stat = statuses[i] if i < len(statuses) else ""


                text = self._build_api_embedding_text(
                            api_name=api_name,
                            category=cat,
                            status=stat,
                            args=args,
                        )

            texts.append(text)
            debug_rows.append(
                {
                    "index": i,
                    "is_placeholder": is_placeholder,
                    "api_name_type": str(type(api_names[i])) if i < len(api_names) else "missing",
                    "args_type": str(type(arg_texts[i])) if i < len(arg_texts) else "missing",
                    "category_type": str(type(categories[i])) if i < len(categories) else "missing",
                    "status_type": str(type(statuses[i])) if i < len(statuses) else "missing",
                    "text_repr": repr(text)[:1000],
                }
            )

        try:
            sem = self.embedder.encode(texts)
        except Exception as e:
            import json
            raise TypeError(
                "Embedding failed in _enrich_api_nodes. Sample debug rows:\n"
                + json.dumps(debug_rows[:10], indent=2)
                + f"\nOriginal error: {e}"
            ) from e

        sem_t = torch.tensor(sem, dtype=torch.float32)

        enriched = torch.cat([sem_t, raw_x.float()], dim=1)

        if enriched.shape[1] != cfg.API_NODE_DIM:
            raise ValueError(
                f"Enriched api.x dim mismatch: got {enriched.shape[1]}, expected {cfg.API_NODE_DIM}"
            )

        graph["api"].x = enriched
        graph["api"].placeholder_count = torch.tensor([placeholder_count], dtype=torch.long)



    

    # ========================================================
    # TEMPORAL EDGE ENRICHMENT
    # ========================================================
    def _enrich_temporal_edges(self, graph: HeteroData) -> None:
        edge_type = ("api", "temporal", "api")
        if edge_type not in graph.edge_types:
            graph[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
            graph[edge_type].edge_attr = torch.empty((0, cfg.TEMPORAL_EDGE_DIM), dtype=torch.float32)
            return

        edge_store = graph[edge_type]
        edge_index = edge_store.edge_index
        edge_attr = getattr(edge_store, "edge_attr", None)

        if edge_attr is None:
            raise ValueError("Temporal edges exist but edge_attr is missing")

        if edge_attr.ndim != 2:
            raise ValueError(f"Temporal edge_attr must be 2D, got shape={tuple(edge_attr.shape)}")

        if edge_attr.shape[1] == cfg.TEMPORAL_EDGE_DIM:
            return

        if edge_attr.shape[1] != 3:
            raise ValueError(
                f"Raw temporal edge_attr must have 3 dims before enrichment, got {edge_attr.shape[1]}"
            )

        num_edges = edge_attr.shape[0]
        if num_edges == 0:
            edge_store.edge_attr = torch.empty((0, cfg.TEMPORAL_EDGE_DIM), dtype=torch.float32)
            return

        base = edge_attr.cpu().numpy().astype(np.float32)
        time_deltas = base[:, 0]
        same_process = base[:, 1]
        is_burst = base[:, 2]

        if not hasattr(graph["api"], "timestamps"):
            raise ValueError("api.timestamps missing; needed for temporal edge enrichment")

        timestamps = graph["api"].timestamps.cpu().numpy().astype(np.float32)
        src_nodes = edge_index[0].cpu().numpy()

        # Bounds check for clearer diagnostics on malformed graphs
        if len(src_nodes) > 0 and int(np.max(src_nodes)) >= len(timestamps):
            raise ValueError(
                f"edge_index src node {int(np.max(src_nodes))} out of range "
                f"for api.timestamps length {len(timestamps)}"
            )

        edge_timestamps = timestamps[src_nodes]

        process_start = float(np.min(timestamps)) if len(timestamps) else 0.0
        process_end = float(np.max(timestamps)) if len(timestamps) else 1.0
        process_duration = max(process_end - process_start, 1e-6)

        speeds = self._compute_speed(time_deltas)
        accelerations = self._compute_acceleration(speeds)
        burst_intensities = self._compute_burst_intensity(edge_timestamps, self.burst_threshold)
        local_burst_ratios = self._compute_local_burst_ratio(time_deltas, self.burst_threshold, self.window_size)
        time_from_start = (edge_timestamps - process_start) / process_duration
        time_to_end = (process_end - edge_timestamps) / process_duration
        seq_positions = (
            np.arange(num_edges, dtype=np.float32) / max(num_edges - 1, 1)
            if num_edges > 1
            else np.zeros(num_edges, dtype=np.float32)
        )
        window_means, window_stds = self._compute_window_statistics(time_deltas, self.window_size)
        periodicity_scores = self._compute_periodicity(time_deltas)
        sudden_changes = self._compute_sudden_change(time_deltas)

        new_features = np.column_stack(
            [
                time_deltas,        # 0
                same_process,       # 1
                is_burst,           # 2
                speeds,             # 3
                accelerations,      # 4
                burst_intensities,  # 5
                local_burst_ratios, # 6
                time_from_start,    # 7
                time_to_end,        # 8
                seq_positions,      # 9
                window_means,       # 10
                window_stds,        # 11
                periodicity_scores, # 12
                sudden_changes,     # 13
            ]
        ).astype(np.float32)

        if new_features.shape[1] != cfg.TEMPORAL_EDGE_DIM:
            raise ValueError(
                f"Enriched temporal edge dim mismatch: got {new_features.shape[1]}, "
                f"expected {cfg.TEMPORAL_EDGE_DIM}"
            )

        edge_store.edge_attr = torch.tensor(new_features, dtype=torch.float32)

    # ========================================================
    # RESOURCE NORMALIZATION
    # ========================================================
    def _normalize_resource_dims(self, graph: HeteroData) -> None:
        self._normalize_node_dim(graph, "process", cfg.PROCESS_NODE_DIM)
        self._normalize_node_dim(graph, "file", cfg.FILE_NODE_DIM)
        self._normalize_node_dim(graph, "registry", cfg.REGISTRY_NODE_DIM)
        self._normalize_node_dim(graph, "network", cfg.NETWORK_NODE_DIM)

    def _normalize_node_dim(self, graph: HeteroData, node_type: str, expected_dim: int) -> None:
        if node_type not in graph.node_types:
            graph[node_type].x = torch.empty((0, expected_dim), dtype=torch.float32)
            return

        x = graph[node_type].x
        if x.ndim != 2:
            raise ValueError(f"{node_type}.x must be 2D, got shape={tuple(x.shape)}")

        current_dim = x.shape[1]
        if current_dim == expected_dim:
            return

        if current_dim < expected_dim:
            pad = torch.zeros((x.shape[0], expected_dim - current_dim), dtype=x.dtype, device=x.device)
            graph[node_type].x = torch.cat([x, pad], dim=1)
            return

        if cfg.FAIL_ON_DIM_MISMATCH and not getattr(cfg, "ALLOW_FEATURE_TRUNCATION", True):
            raise ValueError(
                f"{node_type}.x dim {current_dim} exceeds expected {expected_dim} and truncation is disabled"
            )
        graph[node_type].x = x[:, :expected_dim]

    # ========================================================
    # VALIDATION
    # ========================================================
    def _validate_graph_contract(self, graph: HeteroData) -> None:
        if "api" not in graph.node_types:
            raise ValueError("Final graph missing api node type")
        if graph["api"].x.shape[1] != cfg.API_NODE_DIM:
            raise ValueError(
                f"Final api.x dim mismatch: got {graph['api'].x.shape[1]}, expected {cfg.API_NODE_DIM}"
            )

        expected_resource_dims = {
            "process": cfg.PROCESS_NODE_DIM,
            "file": cfg.FILE_NODE_DIM,
            "registry": cfg.REGISTRY_NODE_DIM,
            "network": cfg.NETWORK_NODE_DIM,
        }
        for nt, dim in expected_resource_dims.items():
            if nt in graph.node_types and graph[nt].x.shape[1] != dim:
                raise ValueError(
                    f"Final {nt}.x dim mismatch: got {graph[nt].x.shape[1]}, expected {dim}"
                )

        edge_type = ("api", "temporal", "api")
        if edge_type not in graph.edge_types:
            raise ValueError("Final graph missing temporal edge type")

        edge_attr = graph[edge_type].edge_attr
        if edge_attr.ndim != 2 or edge_attr.shape[1] != cfg.TEMPORAL_EDGE_DIM:
            raise ValueError(
                f"Final temporal edge_attr dim mismatch: got {tuple(edge_attr.shape)}, "
                f"expected (*, {cfg.TEMPORAL_EDGE_DIM})"
            )

        if not hasattr(graph["api"], "timestamps"):
            raise ValueError("Final graph missing api.timestamps")
        if len(graph["api"].timestamps) != graph["api"].x.shape[0]:
            raise ValueError("api.timestamps length does not match number of api nodes")

    # ========================================================
    # STATS
    # ========================================================
    def _update_stats(self, graph: HeteroData) -> None:
        self.stats.api_nodes_total += int(graph["api"].x.shape[0])

        placeholder_count = int(getattr(graph["api"], "placeholder_count", torch.tensor([0]))[0].item())
        self.stats.placeholder_api_nodes_total += placeholder_count

        temporal_edges = graph["api", "temporal", "api"].edge_index.shape[1]
        self.stats.temporal_edges_total += int(temporal_edges)
        if temporal_edges == 0:
            self.stats.graphs_with_zero_temporal_edges += 1

        if ("api", "modifies", "registry") in graph.edge_types:
            if graph["api", "modifies", "registry"].edge_index.shape[1] > 0:
                self.stats.graphs_with_registry_edges += 1

        if ("api", "connects_to", "network") in graph.edge_types:
            if graph["api", "connects_to", "network"].edge_index.shape[1] > 0:
                self.stats.graphs_with_network_edges += 1

        time_sources = list(getattr(graph["api"], "time_sources", []))

        # Intentionally not mutually exclusive:
        # one graph can contain both true call_time nodes and fallback-timed nodes
        if any(ts == "call_time" for ts in time_sources):
            self.stats.graphs_with_call_time += 1
        if any(ts != "call_time" for ts in time_sources):
            self.stats.graphs_with_time_fallback += 1

    # ========================================================
    # TEMPORAL FEATURE HELPERS
    # ========================================================
    def _compute_speed(self, time_deltas: np.ndarray) -> np.ndarray:
        speeds = np.full(len(time_deltas), 1e6, dtype=np.float32)
        mask = time_deltas > 1e-6
        speeds[mask] = 1.0 / time_deltas[mask]
        return speeds

    def _compute_acceleration(self, speeds: np.ndarray) -> np.ndarray:
        if len(speeds) < 2:
            return np.zeros(len(speeds), dtype=np.float32)
        acc = np.zeros(len(speeds), dtype=np.float32)
        acc[1:] = np.diff(speeds)
        return acc

    def _compute_burst_intensity(self, timestamps: np.ndarray, threshold: float) -> np.ndarray:
        """
        Optimized from O(N^2) to O(N log N) using searchsorted.
        """
        if len(timestamps) == 0:
            return np.zeros(0, dtype=np.float32)

        order = np.argsort(timestamps)
        sorted_ts = timestamps[order]

        left = np.searchsorted(sorted_ts, sorted_ts - threshold, side="left")
        right = np.searchsorted(sorted_ts, sorted_ts + threshold, side="right")
        sorted_counts = (right - left - 1).astype(np.float32)

        intensities = np.zeros(len(timestamps), dtype=np.float32)
        intensities[order] = sorted_counts
        return intensities

    def _compute_local_burst_ratio(self, time_deltas: np.ndarray, threshold: float, window: int) -> np.ndarray:
        ratios = np.zeros(len(time_deltas), dtype=np.float32)
        is_burst = time_deltas < threshold
        for i in range(len(time_deltas)):
            start = max(0, i - window // 2)
            end = min(len(time_deltas), i + window // 2 + 1)
            window_total = max(end - start, 1)
            ratios[i] = float(np.sum(is_burst[start:end]) / window_total)
        return ratios

    def _compute_window_statistics(self, time_deltas: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        means = np.zeros(len(time_deltas), dtype=np.float32)
        stds = np.zeros(len(time_deltas), dtype=np.float32)
        for i in range(len(time_deltas)):
            start = max(0, i - window + 1)
            vals = time_deltas[start : i + 1]
            means[i] = float(np.mean(vals))
            stds[i] = float(np.std(vals)) if len(vals) > 1 else 0.0
        return means, stds

    def _compute_periodicity(self, time_deltas: np.ndarray, min_length: int = 10) -> np.ndarray:
        if len(time_deltas) < min_length:
            return np.zeros(len(time_deltas), dtype=np.float32)
        try:
            fft_vals = np.abs(np.fft.fft(time_deltas))
            usable = fft_vals[1 : len(fft_vals) // 2]
            if len(usable) == 0:
                return np.zeros(len(time_deltas), dtype=np.float32)
            peak = float(np.max(usable))
            mean = float(np.mean(usable))
            score = peak / (mean + 1e-6)
            return np.full(len(time_deltas), score, dtype=np.float32)
        except Exception:
            return np.zeros(len(time_deltas), dtype=np.float32)

    def _compute_sudden_change(self, time_deltas: np.ndarray, threshold_factor: float = 2.0) -> np.ndarray:
        if len(time_deltas) < 2:
            return np.zeros(len(time_deltas), dtype=np.float32)

        diffs = np.zeros(len(time_deltas), dtype=np.float32)
        diffs[1:] = np.abs(np.diff(time_deltas))

        std = float(np.std(diffs[1:]))
        if std == 0.0:
            return np.zeros(len(time_deltas), dtype=np.float32)

        threshold = threshold_factor * std
        return (diffs > threshold).astype(np.float32)

    # ========================================================
    # I/O
    # ========================================================
    def _discover_split_dirs(self, root: Path) -> List[Tuple[str, Path]]:
        split_names = ["train", "val", "test"]
        found = []

        for split in split_names:
            d = root / split
            if d.exists() and any(d.glob("*.pkl")):
                found.append((split, d))

        if found:
            return found

        if any(root.glob("*.pkl")):
            return [("all", root)]

        return []

    def _load_graph(self, path: Path) -> HeteroData:
        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, HeteroData):
            return data

        if isinstance(data, tuple):
            for item in data:
                if isinstance(item, HeteroData):
                    return item

        raise TypeError(f"Unsupported graph payload type in {path}: {type(data)}")

    def _save_graph(self, graph: HeteroData, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_stats(self) -> None:
        out = cfg.PIPELINE_AUDIT_DIR / "enrichment_stats_v2.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.stats.to_dict(), f, indent=2)

        if cfg.VERBOSE:
            print(f"[enrich_graphs_v2] stats written to {out}")
            print(json.dumps(self.stats.to_dict(), indent=2))

    # ========================================================
    # UTILS
    # ========================================================
    def _clone_graph(self, graph: HeteroData) -> HeteroData:
        return pickle.loads(pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL))


if __name__ == "__main__":
    enricher = GraphEnricherV2(
        raw_graphs_dir=cfg.RAW_GRAPHS_DIR,
        enriched_graphs_dir=cfg.ENRICHED_GRAPHS_DIR,
        embedding_batch_size=128,
        burst_threshold=0.1,
        window_size=5,
    )
    enricher.run_directory()