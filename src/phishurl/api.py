"""FastAPI service: the model as an HTTP endpoint.

The public demo runs entirely in the browser, so this service is not what backs
it. It exists because "a model behind an API" is the shape the model would take
inside an organisation — in a mail gateway, a SOC enrichment step, a proxy — and
because it is the reference implementation the browser port is checked against.

Explanations here come from ``shap`` itself rather than from a re-implementation,
which is what makes the parity tests meaningful: they compare the JavaScript
against this, not against another copy of the same idea.

Run::

    uv run uvicorn phishurl.api:app --reload
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from phishurl import __version__
from phishurl.canonicalize import MAX_INPUT_LENGTH, canonical_host
from phishurl.datasets import ROOT
from phishurl.features import FEATURE_NAMES, extract
from phishurl.psl import split_host

MAX_BATCH = 100

_state: dict = {}


def _load() -> None:
    """Load the model, the explainer and the operating threshold once."""
    import shap

    model_path = ROOT / "models" / "model.txt"
    if not model_path.exists():
        raise RuntimeError(
            "models/model.txt not found — run `uv run python -m phishurl.train` first"
        )

    booster = lgb.Booster(model_file=str(model_path))
    _state["booster"] = booster
    _state["explainer"] = shap.TreeExplainer(booster)

    # The default threshold is the one chosen on the out-of-distribution set at a
    # 1% false-positive rate, not 0.5: a false positive blocks a legitimate site.
    metrics_path = ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        _state["threshold"] = float(metrics["ood"]["threshold_at_fpr_1pct"])
        _state["metrics"] = metrics
    else:  # pragma: no cover - only when metrics were never produced
        _state["threshold"] = 0.5
        _state["metrics"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield
    _state.clear()


app = FastAPI(
    title="Phishing URL detector",
    version=__version__,
    summary="Scores a host name for phishing and explains the score with SHAP.",
    lifespan=lifespan,
)


class ScoreRequest(BaseModel):
    url: Annotated[str, Field(min_length=1, max_length=MAX_INPUT_LENGTH)]


class BatchRequest(BaseModel):
    urls: Annotated[list[str], Field(min_length=1, max_length=MAX_BATCH)]


class Contribution(BaseModel):
    feature: str
    value: float
    shap: float


class ScoreResponse(BaseModel):
    url: str
    host: str
    registrable_domain: str
    probability: float
    verdict: str
    threshold: float
    model_version: str
    explanation: list[Contribution]


def _score(url: str, with_explanation: bool = True) -> ScoreResponse:
    host = canonical_host(url)
    if not host:
        raise HTTPException(status_code=400, detail=f"No host name could be parsed from {url!r}")

    features = np.asarray([extract(host)], dtype=np.float64)
    probability = float(_state["booster"].predict(features)[0])
    threshold = _state["threshold"]

    explanation: list[Contribution] = []
    if with_explanation:
        values = np.asarray(_state["explainer"].shap_values(features))
        if values.ndim == 3:
            values = values[:, :, -1]
        explanation = [
            Contribution(feature=name, value=float(features[0][index]), shap=float(values[0][index]))
            for index, name in enumerate(FEATURE_NAMES)
        ]
        explanation.sort(key=lambda item: abs(item.shap), reverse=True)

    return ScoreResponse(
        url=url,
        host=host,
        registrable_domain=split_host(host)[1],
        probability=probability,
        verdict="phishing" if probability >= threshold else "no_phishing_signal",
        threshold=threshold,
        model_version=__version__,
        explanation=explanation,
    )


@app.post("/v1/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    """Score a single URL and return its full SHAP decomposition."""
    return _score(request.url)


@app.post("/v1/score/batch", response_model=list[ScoreResponse])
def score_batch(request: BatchRequest) -> list[ScoreResponse]:
    """Score up to 100 URLs. Explanations are omitted to keep the payload small."""
    return [_score(url, with_explanation=False) for url in request.urls]


@app.get("/v1/model")
def model_info() -> dict:
    """Model metadata: features, threshold, and the metrics behind that threshold."""
    metrics = _state.get("metrics")
    return {
        "version": __version__,
        "n_trees": _state["booster"].num_trees(),
        "n_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "threshold": _state["threshold"],
        "threshold_rationale": "recall at a 1% false-positive rate on the OOD set",
        "metrics": {
            "internal": metrics["internal"] if metrics else None,
            "ood": metrics["ood"] if metrics else None,
        },
        "limitations": (
            "Host names only. Cannot resolve shorteners, cannot inspect page content, "
            "and will miss phishing hosted on compromised legitimate domains."
        ),
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model_loaded": "booster" in _state}
