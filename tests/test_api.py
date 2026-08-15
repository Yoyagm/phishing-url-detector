"""FastAPI service: happy paths, every error code, and the invariants that matter.

The explanation returned by the service must be an exact decomposition of the
score, not a ranked list of plausible reasons — so that is asserted here rather
than assumed.
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from phishurl.api import MAX_BATCH, app
from phishurl.canonicalize import MAX_INPUT_LENGTH
from phishurl.datasets import ROOT

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client():
    if not (ROOT / "models" / "model.txt").exists():
        pytest.skip("models/model.txt missing — run `uv run python -m phishurl.train`")
    with TestClient(app) as test_client:
        yield test_client


def test_healthz(client) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_score_a_phishing_shaped_host(client) -> None:
    response = client.post(
        "/v1/score", json={"url": "https://appleid.apple.com.secure-login-verify.xyz/signin"}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["host"] == "appleid.apple.com.secure-login-verify.xyz"
    assert body["registrable_domain"] == "secure-login-verify.xyz"
    assert body["verdict"] == "phishing"
    assert 0.0 <= body["probability"] <= 1.0
    assert body["explanation"], "an explanation must be returned"


def test_score_a_real_site(client) -> None:
    response = client.post("/v1/score", json={"url": "https://www.bbc.co.uk"})
    assert response.status_code == 200
    body = response.json()
    assert body["host"] == "bbc.co.uk"
    assert body["verdict"] == "no_phishing_signal"


def test_explanation_is_sorted_and_complete(client) -> None:
    from phishurl.features import FEATURE_NAMES

    body = client.post("/v1/score", json={"url": "https://login-secure.tk"}).json()
    explanation = body["explanation"]

    assert len(explanation) == len(FEATURE_NAMES)
    magnitudes = [abs(item["shap"]) for item in explanation]
    assert magnitudes == sorted(magnitudes, reverse=True), "not ordered by importance"


def test_explanation_is_an_exact_decomposition(client) -> None:
    """base + Σ shap must reconstruct the log-odds of the returned probability."""
    from phishurl.api import _state

    body = client.post("/v1/score", json={"url": "https://paypal.com.evil.tk"}).json()
    total = sum(item["shap"] for item in body["explanation"])
    base = float(_state["explainer"].expected_value)

    probability = body["probability"]
    raw = math.log(probability / (1 - probability))
    assert abs(base + total - raw) < 1e-9


def test_url_that_has_no_host_is_a_400(client) -> None:
    for bad in ["https://", "   ", "@"]:
        response = client.post("/v1/score", json={"url": bad})
        assert response.status_code == 400, f"{bad!r} should be a 400"
        assert "host" in response.json()["detail"].lower()


def test_missing_or_malformed_body_is_a_422(client) -> None:
    assert client.post("/v1/score", json={}).status_code == 422
    assert client.post("/v1/score", json={"url": ""}).status_code == 422
    assert client.post("/v1/score", json={"url": 42}).status_code == 422


def test_url_over_the_length_limit_is_rejected(client) -> None:
    response = client.post("/v1/score", json={"url": "h" * (MAX_INPUT_LENGTH + 1)})
    assert response.status_code == 422


def test_batch_scoring(client) -> None:
    urls = ["https://www.bbc.co.uk", "https://login-verify-account.tk", "example.com"]
    response = client.post("/v1/score/batch", json={"urls": urls})
    assert response.status_code == 200

    body = response.json()
    assert len(body) == len(urls)
    assert [item["host"] for item in body] == ["bbc.co.uk", "login-verify-account.tk", "example.com"]
    # Batch responses drop the explanation on purpose.
    assert all(item["explanation"] == [] for item in body)


def test_batch_over_the_limit_is_a_422(client) -> None:
    response = client.post("/v1/score/batch", json={"urls": ["example.com"] * (MAX_BATCH + 1)})
    assert response.status_code == 422


def test_batch_stops_on_an_unparseable_member(client) -> None:
    response = client.post("/v1/score/batch", json={"urls": ["example.com", "https://"]})
    assert response.status_code == 400


def test_model_metadata_exposes_the_threshold_and_its_reason(client) -> None:
    body = client.get("/v1/model").json()
    assert body["n_trees"] == 300
    assert body["n_features"] == len(body["feature_names"])
    assert 0.0 < body["threshold"] < 1.0
    assert "false-positive" in body["threshold_rationale"]
    assert "shorteners" in body["limitations"]


def test_threshold_is_not_the_naive_half(client) -> None:
    """0.5 would be an unconsidered default; the service must use a chosen point."""
    assert client.get("/v1/model").json()["threshold"] != 0.5
