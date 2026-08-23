# Threat Detection Using Heterogeneous Graph Neural Networks with Semantic and Temporal Features — local web app

Local FastAPI service that scores a sandbox JSON report with the trained framework
— a heterogeneous graph neural network with semantic and temporal features
(`best_model_1.pt`, the reported BiLSTM backbone) — and returns
the verdict together with the temporal / structural / semantic evidence behind
the decision (a compressed form of the RQ4 attribution).

It **reuses the existing project code** — nothing is duplicated:

```
raw JSON ─▶ ReportAdapterV2 ─▶ GraphBuilderV2 ─▶ GraphEnricherV2 ─▶ enriched graph
enriched graph ─▶ explain_one_graph()  ─▶ prediction + branch attribution
```

## Run

From the **project root** (so imports resolve), using the project venv:

```bash
venv/bin/uvicorn webapp.app:app --host 127.0.0.1 --port 8000
```

The model, schema, vocabulary, and MPNet encoder load **once** at startup
(~8 s). Then open the built-in interactive UI:

- **http://127.0.0.1:8000/docs** — Swagger. Expand `POST /analyze`, click
  *Try it out*, choose a `.json` report, *Execute*, and read the response.
- `http://127.0.0.1:8000/health` — readiness probe.

Or with curl:

```bash
curl -s -F "file=@/path/to/report.json" http://127.0.0.1:8000/analyze | python -m json.tool
```

## Response shape

```json
{
  "verdict": "Threat",
  "predicted_class": "Other Malware",
  "threat_probability": 0.98,
  "class_probabilities": {"Ransomware": .., "Benign": .., "Other Malware": ..},
  "dominant_branch": "temporal",
  "graph_summary": {"api": .., "file": .., "registry": .., "process": .., "network": .., "total_edges": ..},
  "rationale": ["...analyst sentences..."],
  "drivers": {
    "temporal":   {"gate_weight": .., "signals": ["A was followed by B ...", ...]},
    "structural": {"gate_weight": .., "relations": [...], "relation_counts": {...}, "connected_resources": {...}},
    "semantic":   {"gate_weight": .., "top_calls": [{"call": "...", "saliency": ..}, ...]}
  }
}
```

## Notes

- Input is a CAPE-style sandbox report (`behavior.processes[].calls[]` with
  `api`/`name`, `category`, `time`). Reports with fewer than
  `config.MIN_API_CALLS` API calls are rejected (HTTP 422).
- The checkpoint is `results/checkpoints/best_model_1.pt`. Change it via the
  `checkpoint=` argument to `Analyzer(...)` in `service.py`.
- Part 2 will add a polished UI on top of this same `/analyze` endpoint.
