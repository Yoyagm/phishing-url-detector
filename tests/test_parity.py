"""Python ↔ JavaScript parity.

The demo page scores URLs in the browser with no backend, which only works if the
JavaScript engine is genuinely the same computation as the Python package rather
than a plausible re-write. These tests run one corpus through both and compare
three things at increasing depth:

1. the canonical host,
2. all 51 feature values,
3. the raw score and every per-feature SHAP attribution.

Tolerance is 1e-12. That is not arbitrary: ``shap``'s own additivity
(``expected_value + Σ shap == raw_score``) was measured at 5e-16 on this stack, so
1e-12 leaves four orders of magnitude of headroom over floating-point noise while
still catching any real divergence.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from phishurl.canonicalize import canonical_host
from phishurl.datasets import ROOT
from phishurl.features import FEATURE_NAMES, SUSPICIOUS_KEYWORDS, TOP_SUFFIXES, extract
from phishurl.export_model import feature_spec_hash

TOLERANCE = 1e-12
CORPUS = Path(__file__).parent / "parity_corpus.txt"
RUNNER = Path(__file__).parent / "parity_runner.mjs"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def corpus() -> list[str]:
    text = CORPUS.read_text(encoding="utf-8")
    return text.split("\n")[:-1]


@pytest.fixture(scope="module")
def javascript(tmp_path_factory) -> dict:
    """Run the real ``docs/engine.js`` under Node and collect its output."""
    if not (ROOT / "docs" / "model.json").exists():
        pytest.skip("docs/model.json missing — run `uv run python -m phishurl.export_model`")

    node = _find_node()
    output = tmp_path_factory.mktemp("parity") / "js.json"
    completed = subprocess.run(
        [node, str(RUNNER), str(CORPUS.relative_to(ROOT)), str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    if completed.returncode != 0:
        pytest.fail(f"parity runner failed:\n{completed.stdout}\n{completed.stderr}")
    return json.loads(output.read_text())


def _find_node() -> str:
    import shutil

    for candidate in ("node", str(Path.home() / ".local" / "node" / "bin" / "node")):
        found = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if found:
            return found
    pytest.skip("node not available")


def test_corpus_is_substantial(corpus: list[str]) -> None:
    assert len(corpus) >= 500, "the corpus should be large enough to exercise real hosts"


def test_feature_spec_hash_matches(javascript: dict) -> None:
    """The artifact the browser loads must describe the features Python builds."""
    assert javascript["feature_spec_sha256"] == feature_spec_hash()


def test_canonical_host_parity(corpus: list[str], javascript: dict) -> None:
    mismatches = [
        (entry["url"], canonical_host(entry["url"]), entry["host"])
        for entry in javascript["results"]
        if canonical_host(entry["url"]) != entry["host"]
    ]
    assert not mismatches, f"{len(mismatches)} host mismatches, first 5: {mismatches[:5]}"


def test_feature_vector_parity(javascript: dict) -> None:
    worst_delta = 0.0
    worst_where = None
    mismatches = 0

    for entry in javascript["results"]:
        python_vector = np.asarray(extract(entry["host"]), dtype=np.float64)
        js_vector = np.asarray(entry["features"], dtype=np.float64)

        assert len(python_vector) == len(js_vector) == len(FEATURE_NAMES)
        delta = np.abs(python_vector - js_vector)
        if delta.max() > worst_delta:
            worst_delta = float(delta.max())
            worst_where = (entry["url"], FEATURE_NAMES[int(delta.argmax())])
        if delta.max() > TOLERANCE:
            mismatches += 1

    assert mismatches == 0, (
        f"{mismatches} URLs have diverging features; worst Δ={worst_delta:.3e} "
        f"at {worst_where}"
    )


def test_raw_score_parity(javascript: dict) -> None:
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(ROOT / "models" / "model.txt"))
    matrix = np.asarray([extract(entry["host"]) for entry in javascript["results"]])
    python_raw = booster.predict(matrix, raw_score=True)
    js_raw = np.asarray([entry["raw_score"] for entry in javascript["results"]])

    delta = np.abs(python_raw - js_raw)
    assert delta.max() < TOLERANCE, (
        f"raw score diverges by up to {delta.max():.3e} "
        f"(worst: {javascript['results'][int(delta.argmax())]['url']!r})"
    )


def test_shap_parity(javascript: dict) -> None:
    """Every per-feature attribution, not just their sum.

    A sum-only check would pass for an implementation that gets the total right
    and the distribution across features wrong — which is precisely what the demo
    page shows the user.
    """
    import lightgbm as lgb
    import shap

    booster = lgb.Booster(model_file=str(ROOT / "models" / "model.txt"))
    explainer = shap.TreeExplainer(booster)
    matrix = np.asarray([extract(entry["host"]) for entry in javascript["results"]])

    python_shap = np.asarray(explainer.shap_values(matrix))
    if python_shap.ndim == 3:
        python_shap = python_shap[:, :, -1]
    js_shap = np.asarray([entry["shap"] for entry in javascript["results"]])

    assert python_shap.shape == js_shap.shape
    delta = np.abs(python_shap - js_shap)
    row, column = np.unravel_index(int(delta.argmax()), delta.shape)
    assert delta.max() < TOLERANCE, (
        f"SHAP diverges by up to {delta.max():.3e} on feature "
        f"{FEATURE_NAMES[column]!r} for {javascript['results'][row]['url']!r}"
    )


def test_shap_additivity_in_javascript(javascript: dict) -> None:
    """``base_score + Σ shap == raw_score`` must hold on the JavaScript side too."""
    base = javascript["base_score"]
    worst = 0.0
    for entry in javascript["results"]:
        reconstructed = base + sum(entry["shap"])
        worst = max(worst, abs(reconstructed - entry["raw_score"]))
    assert worst < TOLERANCE, f"additivity broken in JS by up to {worst:.3e}"


def test_vocabulary_travels_with_the_model(javascript: dict) -> None:
    model = json.loads((ROOT / "docs" / "model.json").read_text())
    assert tuple(model["vocab"]["top_suffixes"]) == TOP_SUFFIXES
    assert tuple(model["vocab"]["suspicious_keywords"]) == SUSPICIOUS_KEYWORDS
