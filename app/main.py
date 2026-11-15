"""FastAPI service: score network flows, explain alerts, serve the analyst console.

Run:  uvicorn app.main:app --reload      (from the project root)
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import BASE_FEATURES, CLASSES  # noqa: E402
from explain import shap_array  # noqa: E402

ART = ROOT / "artifacts"
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Network Intrusion Detection API", version="1.0")

_bundle = joblib.load(ART / "model.joblib")
MODEL = _bundle["model"]
PRE = _bundle["preprocessor"]
MODEL_NAME = _bundle["name"]
EXPLAINER = shap.TreeExplainer(MODEL)
RESULTS = json.loads((ART / "results.json").read_text())
REPLAY = json.loads((ART / "replay_sample.json").read_text())
ATTACK_IDX = [i for i, c in enumerate(CLASSES) if c != "normal"]


class ScoreRequest(BaseModel):
    flows: list[dict[str, Any]] = Field(..., min_length=1, max_length=2000,
                                        description="Raw NSL-KDD connection records (41 fields each).")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Alert when P(attack) >= threshold.")
    explain: bool = Field(True, description="Attach SHAP reasons to alerts.")


def _frame(flows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(flows)
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(422, detail=f"Missing fields: {missing}")
    try:
        numeric = [c for c in BASE_FEATURES if c not in ("protocol_type", "service", "flag")]
        df[numeric] = df[numeric].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, detail=f"Non-numeric value in a numeric field: {exc}") from exc
    return df


def score_flows(flows: list[dict[str, Any]], threshold: float, explain: bool) -> list[dict[str, Any]]:
    df = _frame(flows)
    X = PRE.transform(df)
    proba = MODEL.predict_proba(X)
    attack_score = 1.0 - proba[:, 0]
    out = []
    for i in range(len(df)):
        alert = bool(attack_score[i] >= threshold)
        # Name the alert after the most likely *attack* class, even if "normal" has the top vote.
        attack_class = int(ATTACK_IDX[int(np.argmax(proba[i, ATTACK_IDX]))])
        label_idx = attack_class if alert else 0
        item = {
            "index": i,
            "alert": alert,
            "predicted": CLASSES[label_idx],
            "attack_score": round(float(attack_score[i]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, proba[i])},
        }
        if alert and explain:
            sv = shap_array(EXPLAINER, X.iloc[[i]])[0, :, attack_class]
            order = np.argsort(-np.abs(sv))[:5]
            item["reasons"] = [{"feature": X.columns[j], "value": float(X.iloc[i, j]),
                                "impact": round(float(sv[j]), 4)} for j in order]
        out.append(item)
    return out


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_NAME, "classes": CLASSES}


@app.get("/api/results")
def results() -> dict[str, Any]:
    """Offline evaluation numbers produced by src/train.py."""
    return RESULTS


@app.post("/api/score")
def score(req: ScoreRequest) -> dict[str, Any]:
    return {"model": MODEL_NAME, "threshold": req.threshold,
            "results": score_flows(req.flows, req.threshold, req.explain)}


@app.get("/api/replay")
def replay(n: int = Query(150, ge=1, le=400), seed: int | None = None,
           threshold: float = Query(0.5, ge=0.0, le=1.0)) -> dict[str, Any]:
    """Score a random slice of held-out KDDTest+ traffic (labels included for comparison)."""
    rng = random.Random(seed)
    rows = rng.sample(REPLAY, min(n, len(REPLAY)))
    flows = [{k: r[k] for k in BASE_FEATURES} for r in rows]
    scored = score_flows(flows, threshold, explain=True)
    for s, r in zip(scored, rows):
        s["true_family"] = r["true_family"]
        s["true_attack"] = r["true_attack"]
        s["flow"] = {k: r[k] for k in ("protocol_type", "service", "flag", "duration",
                                         "src_bytes", "dst_bytes", "count", "srv_count",
                                         "serror_rate", "same_srv_rate", "dst_host_count")}
    return {"model": MODEL_NAME, "threshold": threshold, "results": scored}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/artifacts/{name}")
def artifact_image(name: str) -> FileResponse:
    """Serve evaluation charts only (never the model file)."""
    if name not in {"confusion_matrix.png", "shap_global.png", "model_comparison.png"}:
        raise HTTPException(404, detail="Not found")
    path = ART / name
    if not path.exists():
        raise HTTPException(404, detail="Run src/train.py and src/explain.py first")
    return FileResponse(path)
