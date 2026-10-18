"""The waiter: serves artifacts/results.json to the dashboard. It never trains or calculates.
Run: uv run fastapi dev api/main.py   (dashboard: http://127.0.0.1:8000/, test page: /docs)"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "artifacts" / "results.json"
DASHBOARD_PATH = ROOT / "frontend" / "index.html"

app = FastAPI(title="HavenRL API")

# Let a webpage on another address (the dashboard) read our answers.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])


def load_results():
    """Read the file fresh on every request, so a new backtest shows up without a restart."""
    if not RESULTS_PATH.exists():
        raise HTTPException(404, "results.json not found. Run: uv run python -m scripts.backtest")
    return json.loads(RESULTS_PATH.read_text())


@app.get("/", include_in_schema=False)
def dashboard():
    """The waiter also hands over the menu card: the dashboard page itself."""
    return FileResponse(DASHBOARD_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/results")
def results():
    return load_results()


@app.get("/summary")
def summary():
    r = load_results()
    return {"test_period": r["test_period"], "agent": r["agent"], "baseline": r["baseline"]}
