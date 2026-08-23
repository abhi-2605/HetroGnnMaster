"""
webapp/service.py

Backend inference + explanation service for the behavioural threat detection framework
(a heterogeneous graph neural network with semantic and temporal features).

It reuses the EXISTING project code end to end:
    raw JSON  ->  ReportAdapterV2 -> GraphBuilderV2 -> GraphEnricherV2  (enriched graph)
    enriched graph  ->  explain_one_graph()  (prediction + branch attribution)

The heavy objects (schema, vocab, model checkpoint, MPNet encoder inside the
enricher) are loaded ONCE when Analyzer() is constructed and then reused for
every request.

Public API:
    analyzer = Analyzer()                 # loads everything once
    result   = analyzer.analyze_json(path)      # full pipeline from a raw report
    result   = analyzer.analyze_graph(graph, sample_id)   # from an enriched graph object
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

import config  # noqa: E402
from utils.schema_manager import ensure_active_schema  # noqa: E402
from graph_construction.vocabulary import load_vocab  # noqa: E402
from scripts.explain_rq4_behavioral_decisions import (  # noqa: E402
    load_model,
    explain_one_graph,
)
from pipeline_v2 import pipeline_config_v2 as pcfg  # noqa: E402
from pipeline_v2.report_adapter_v2 import ReportAdapterV2  # noqa: E402
from pipeline_v2.build_graphs_v2 import GraphBuilderV2  # noqa: E402
from pipeline_v2.enrich_graphs_v2 import GraphEnricherV2  # noqa: E402

DEFAULT_CHECKPOINT = "best_model_1.pt" 

class AnalysisError(Exception):
    """Raised when a report cannot be turned into a scored graph."""


class Analyzer:
    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, top_k: int = 8, window: int = 5):
        self.top_k = top_k
        self.window = window
        self.device = torch.device(config.DEVICE)

        # --- heavy, load-once state ---
        self.schema = ensure_active_schema()
        self.vocab = load_vocab(config.RESULTS_DIR / "api_vocabulary.json")
        ckpt_path = config.CHECKPOINTS_DIR / checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        self.model, _ = load_model(checkpoint_path=ckpt_path, schema=self.schema, device=self.device)
        self.checkpoint_name = checkpoint

        # pipeline components (enricher lazily loads MPNet on first enrich)
        self.adapter = ReportAdapterV2(labels_map=dict(pcfg.LABEL_NAME_TO_ID))
        self.builder = GraphBuilderV2()
        self.enricher = GraphEnricherV2()

    # ------------------------------------------------------------------ #
    # entry points
    # ------------------------------------------------------------------ #
    def analyze_json(self, json_path: str | Path) -> Dict[str, Any]:
        """Full path: raw sandbox JSON report -> enriched graph -> scored report."""
        json_path = Path(json_path)
        t0 = time.time()
        try:
            adapted = self.adapter.adapt_file(json_path, label_name="benign", label_id=1)
        except Exception as e:  # noqa: BLE001
            raise AnalysisError(f"Could not adapt report: {e}") from e

        if adapted is None:
            raise AnalysisError("Report adapter returned nothing (unreadable report).")
        if adapted.get("regime") == "invalid":
            raise AnalysisError(
                "Report has too little behavioural activity to analyse "
                f"(min {config.MIN_API_CALLS} API calls required)."
            )

        try:
            graph = self.builder.build_graph(adapted)
            enriched = self.enricher.enrich_graph(graph)
        except Exception as e:  # noqa: BLE001
            raise AnalysisError(f"Graph construction/enrichment failed: {e}") from e

        sample_id = adapted.get("report_id") or json_path.stem
        result = self.analyze_graph(enriched, sample_id=str(sample_id))
        result["timing_seconds"] = round(time.time() - t0, 2)
        return result

    def analyze_graph(self, graph, sample_id: str = "uploaded_sample") -> Dict[str, Any]:
        """Score + explain an already-enriched graph object."""
        report = explain_one_graph(
            graph=graph,
            label=1,  # placeholder; the upload is unlabelled, correctness fields are dropped
            sample_id=str(sample_id),
            model=self.model,
            schema=self.schema,
            vocab=self.vocab,
            device=self.device,
            top_k=self.top_k,
            window=self.window,
            include_metadata_inventory=False,
        )
        return self._compress(report, sample_id)

    # ------------------------------------------------------------------ #
    # RQ4 report -> compact 4-panel structure
    # ------------------------------------------------------------------ #
    _PROB_DISPLAY = {
        "ransomware": "Ransomware",
        "benign": "Benign",
        "other_malware": "Other Malware",
    }
    _BURST_THRESHOLD = 0.1  # seconds; matches the enricher's burst-detection window

    @staticmethod
    def _collapse_seq(names, limit: int = 14) -> str:
        """['A','A','B','A'] -> 'A x2 -> B -> A' (readable local behavioural pattern)."""
        out = []
        for n in names:
            if not n:
                continue
            if out and out[-1][0] == n:
                out[-1][1] += 1
            else:
                out.append([n, 1])
        parts = [f"{n} x{c}" if c > 1 else n for n, c in out[:limit]]
        return " -> ".join(parts)

    def _build_graph(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Small salient sub-graph for visualisation: the high-saliency API nodes plus the
        process / file / registry / network nodes they touch, and the edges between them."""
        neigh = report.get("salient_behavioral_neighborhoods") or {}
        rel_summaries = neigh.get("relation_summaries", []) if isinstance(neigh, dict) else []
        nodes: Dict[str, Any] = {}
        edges = []

        def add(nd, salient=False, saliency=None):
            if not isinstance(nd, dict) or nd.get("node_index") is None:
                return None
            nid = f"{nd.get('node_type')}:{nd.get('node_index')}"
            if nid not in nodes:
                nodes[nid] = {
                    "id": nid,
                    "label": nd.get("display_name") or nid,
                    "type": nd.get("node_type"),
                    "salient": salient,
                }
                if saliency is not None:
                    nodes[nid]["saliency"] = saliency
            elif salient:
                nodes[nid]["salient"] = True
            return nid

        for node in (report.get("top_salient_api_nodes") or [])[:10]:
            api = node.get("salient_api") or {}
            add(api, salient=True, saliency=_round(node.get("saliency")))

        seen_edges = set()
        for rs in rel_summaries:
            for ex in (rs.get("sampled_examples") or [])[:6]:
                s = add(ex.get("source"))
                d = add(ex.get("destination"))
                if s and d:
                    key = (s, d, ex.get("relation"))
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    edges.append({"from": s, "to": d, "relation": ex.get("relation") or rs.get("edge_type")})

        return {"nodes": list(nodes.values()), "edges": edges}

    @classmethod
    def _timing_stats(cls, gaps) -> Dict[str, Any]:
        vals = sorted(g for g in gaps if isinstance(g, (int, float)))
        if not vals:
            return {}
        mx = vals[-1]
        med = vals[len(vals) // 2]
        stats = {"median_gap_s": round(med, 4), "max_gap_s": round(mx, 4)}
        if mx == 0.0:
            stats["pattern"] = "second-level timestamps; temporal branch relies on call ORDER here"
        elif mx <= cls._BURST_THRESHOLD:
            stats["pattern"] = f"burst: all calls fire within {cls._BURST_THRESHOLD}s"
        else:
            stats["pattern"] = "spread out in time"
        return stats

    def _compress(self, report: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
        probs = report.get("probabilities") or {}
        class_probs = {self._PROB_DISPLAY.get(k, k): round(float(v), 4) for k, v in probs.items()}
        benign_p = float(probs.get("benign", 0.0))
        threat_p = round(1.0 - benign_p, 4)
        top_class = max(probs, key=probs.get) if probs else None
        predicted_class = self._PROB_DISPLAY.get(top_class, report.get("pred_name"))

        branch = report.get("branch_gate_weights") or {}
        neigh = report.get("salient_behavioral_neighborhoods") or {}
        rel_summaries = neigh.get("relation_summaries", []) if isinstance(neigh, dict) else []

        # --- semantic: highest-saliency API calls (deduped by name) ---
        semantic_calls, seen = [], set()
        for node in (report.get("top_salient_api_nodes") or []):
            api = node.get("salient_api") or {}
            name = api.get("display_name") or f"api#{api.get('node_index')}"
            if name in seen:
                continue
            seen.add(name)
            semantic_calls.append(
                {"call": name, "saliency": _round(node.get("saliency")), "t": _round(node.get("timestamp"), 2)}
            )
            if len(semantic_calls) >= self.top_k:
                break

        # --- temporal (api->api ordering) vs structural (other relations) ---
        temporal_signals, structural_relations = [], []
        for rs in rel_summaries:
            examples = [ex.get("plain_english") for ex in (rs.get("sampled_examples") or []) if ex.get("plain_english")]
            if rs.get("edge_type") == "api__temporal__api":
                for e in examples:
                    if e not in temporal_signals:
                        temporal_signals.append(e)
            else:
                structural_relations.append(
                    {
                        "relation": rs.get("relation_meaning", rs.get("edge_type")),
                        "edges_touching_salient": rs.get("num_edges_touching_salient_api_nodes"),
                        "examples": examples[:3],
                    }
                )

        # --- temporal FEATURE windows: the local ordered pattern + timing/burst
        #     the temporal branch reads around each high-saliency API node ---
        temporal_windows = []
        for node in (report.get("top_salient_api_nodes") or [])[:3]:
            win = node.get("local_sequence_window") or []
            temporal_windows.append(
                {
                    "focus": (node.get("salient_api") or {}).get("display_name"),
                    "saliency": _round(node.get("saliency")),
                    "local_pattern": self._collapse_seq([e.get("display_name") for e in win]),
                    "timing": self._timing_stats([e.get("time_gap_from_previous_api") for e in win]),
                }
            )

        connected = neigh.get("connected_resource_summary", {}) if isinstance(neigh, dict) else {}
        rel_counts = neigh.get("global_relation_counts_touching_salient_api_nodes", {}) if isinstance(neigh, dict) else {}

        stats = report.get("graph_basic_stats") or {}
        nc = stats.get("node_counts", {}) or {}
        graph_summary = {
            "api": nc.get("api"),
            "file": nc.get("file"),
            "registry": nc.get("registry"),
            "process": nc.get("process"),
            "network": nc.get("network"),
            "total_edges": stats.get("total_edges"),
        }

        return {
            "sample_id": sample_id,
            "checkpoint": self.checkpoint_name,
            "verdict": report.get("pred_binary"),          # Threat / Benign
            "predicted_class": predicted_class,            # Ransomware / Benign / Other Malware
            "threat_probability": threat_p,
            "class_probabilities": class_probs,
            "dominant_branch": branch.get("dominant"),
            "graph_summary": graph_summary,
            "graph": self._build_graph(report),
            "rationale": report.get("analyst_rationale"),
            "drivers": {
                "temporal": {
                    "gate_weight": _round(branch.get("temporal")),
                    "signals": temporal_signals[: self.top_k],
                    "feature_windows": temporal_windows,
                },
                "structural": {
                    "gate_weight": _round(branch.get("structural")),
                    "relations": structural_relations[: self.top_k],
                    "relation_counts": rel_counts,
                    "connected_resources": connected,
                },
                "semantic": {
                    "gate_weight": _round(branch.get("semantic")),
                    "top_calls": semantic_calls,
                },
            },
        }


def _round(x, n: int = 3):
    return round(float(x), n) if isinstance(x, (int, float)) else x
