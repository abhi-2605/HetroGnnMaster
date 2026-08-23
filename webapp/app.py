"""
webapp/app.py

Local FastAPI server for the behavioural threat detection framework
(a heterogeneous graph neural network with semantic and temporal features).

Run:
    cd <project root>
    venv/bin/uvicorn webapp.app:app --host 127.0.0.1 --port 8000

Then open the built-in interactive UI at:
    http://127.0.0.1:8000/docs      (Swagger  - upload a report via "Try it out")
    http://127.0.0.1:8000/redoc     (ReDoc)

No hand-written frontend in Part 1 - the polished UI comes in Part 2.

Endpoints:
    GET  /            -> redirects to /docs
    GET  /health      -> readiness probe
    POST /analyze     -> multipart JSON upload -> structured detection + attribution
"""

from __future__ import annotations

import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, FileResponse  # noqa: E402

from webapp.service import Analyzer, AnalysisError  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Load model / schema / vocab / MPNet ONCE at startup and reuse for every request.
analyzer: Analyzer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    print("[webapp] loading model + schema + vocab (once)...")
    analyzer = Analyzer()
    print(f"[webapp] ready. checkpoint = {analyzer.checkpoint_name}")
    yield
    # (no teardown needed)


app = FastAPI(
    title="Threat Detection Using Heterogeneous Graph Neural Networks with Semantic and Temporal Features",
    version="0.1",
    description="Upload a sandbox JSON report; it is converted to a heterogeneous graph and "
    "scored by best_model_1.pt, returning the verdict plus the temporal, structural, and "
    "semantic evidence behind the decision (RQ4 attribution, compressed).",
    lifespan=lifespan,
)

# Allow the page to be served from a separate dev origin (e.g. VS Code Live Server
# on :5500) and still call this backend. Same-origin use (opening :8000 directly)
# needs none of this; it is purely a dev convenience. Local app -> permissive.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ready" if analyzer is not None else "loading",
        "checkpoint": getattr(analyzer, "checkpoint_name", None),
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(..., description="Sandbox JSON report (.json)")) -> JSONResponse:
    """Convert an uploaded sandbox report to a graph, score it, and return the structured
    detection report with per-branch (temporal / structural / semantic) attribution."""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again in a moment.")
    if not (file.filename or "").lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Please upload a .json sandbox report.")

    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        try:
            result = analyzer.analyze_json(tmp.name)
        except AnalysisError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    result["source_filename"] = file.filename
    return JSONResponse(result)

