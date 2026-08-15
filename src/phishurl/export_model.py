"""Export the trained model into a form the browser can score and explain.

The demo has no backend, so everything the page needs travels in two static JSON
files: the pruned tree ensemble and the Public Suffix List. ``dump_model()`` is
kept only where TreeSHAP needs it — per node the split feature, the threshold,
the child pointers and the *cover* (sample count) that gives the algorithm its
proportions — and rewritten as flat arrays, which are both smaller than nested
objects and faster to walk in JavaScript.

Two safeguards travel with the artifact:

* ``base_score`` is the ``expected_value`` taken from ``shap.TreeExplainer``, so the
  browser's attributions are anchored to the same baseline as the Python reference
  rather than to a re-derived one.
* ``feature_spec_sha256`` hashes the feature definition. The page refuses to score
  when its own extractor disagrees, turning train/serve skew into a visible
  failure instead of a silently wrong probability.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from phishurl.datasets import ROOT
from phishurl.features import (
    FEATURE_NAMES,
    SUSPICIOUS_KEYWORDS,
    TOP_SUFFIXES,
    load_brand_tokens,
)
from phishurl.psl import DEFAULT_PSL_PATH, load_rules

MODELS = ROOT / "models"
DOCS = ROOT / "docs"


def feature_spec_hash() -> str:
    """Hash the parts of the feature definition both languages must agree on."""
    payload = json.dumps(
        {
            "feature_names": list(FEATURE_NAMES),
            "top_suffixes": list(TOP_SUFFIXES),
            "suspicious_keywords": list(SUSPICIOUS_KEYWORDS),
            "brand_tokens": sorted(load_brand_tokens()),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def flatten_tree(structure: dict) -> dict:
    """Depth-first flatten of one LightGBM tree into parallel arrays.

    Leaves are marked by ``feature = -1``. ``count`` carries ``internal_count`` /
    ``leaf_count``, which is the cover TreeSHAP uses to weight the two branches
    of a split when the feature is treated as absent.
    """
    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    value: list[float] = []
    count: list[float] = []
    default_left: list[int] = []

    def visit(node: dict) -> int:
        index = len(feature)

        if "leaf_value" in node:
            feature.append(-1)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            value.append(float(node["leaf_value"]))
            count.append(float(node.get("leaf_count", 1)))
            default_left.append(0)
            return index

        if node["decision_type"] != "<=":
            raise ValueError(
                f"Unsupported split type {node['decision_type']!r}. The browser scorer "
                "implements numerical splits only; categorical splits would need a "
                "bitset decoder on both sides."
            )

        feature.append(int(node["split_feature"]))
        threshold.append(float(node["threshold"]))
        left.append(-1)
        right.append(-1)
        value.append(float(node["internal_value"]))
        count.append(float(node["internal_count"]))
        default_left.append(1 if node.get("default_left") else 0)

        left[index] = visit(node["left_child"])
        right[index] = visit(node["right_child"])
        return index

    visit(structure)
    return {
        "feature": feature, "threshold": threshold, "left": left, "right": right,
        "value": value, "count": count, "default_left": default_left,
    }


def export_psl(destination: Path) -> dict[str, int]:
    """Write the PSL as one ordered rule array, ICANN rules first.

    Storing a single array with a split point rather than two arrays keeps the
    file roughly half the size while still letting the browser reconstruct both
    rule sets — it needs both, since grouping uses ICANN and one feature uses
    the private section.
    """
    icann, everything = load_rules(DEFAULT_PSL_PATH)
    icann_sorted = sorted(icann)
    private_sorted = sorted(everything - icann)
    payload = {"icann_count": len(icann_sorted), "rules": icann_sorted + private_sorted}
    destination.write_text(json.dumps(payload, separators=(",", ":")))
    return {"icann": len(icann_sorted), "private": len(private_sorted)}


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    model_path = MODELS / "model.txt"
    if not model_path.exists():
        raise SystemExit("models/model.txt not found — run `uv run python -m phishurl.train` first")

    booster = lgb.Booster(model_file=str(model_path))
    dump = booster.dump_model()

    # Anchor the browser's attributions to the reference baseline rather than to a
    # value re-derived from the trees.
    import shap

    explainer = shap.TreeExplainer(booster)
    base_score = float(np.asarray(explainer.expected_value).ravel()[-1])

    trees = [flatten_tree(tree["tree_structure"]) for tree in dump["tree_info"]]
    n_nodes = sum(len(tree["feature"]) for tree in trees)

    artifact = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "objective": "binary_sigmoid",
        "feature_names": list(FEATURE_NAMES),
        "feature_spec_sha256": feature_spec_hash(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "base_score": base_score,
        "vocab": {
            "top_suffixes": list(TOP_SUFFIXES),
            "suspicious_keywords": list(SUSPICIOUS_KEYWORDS),
            "brand_tokens": sorted(load_brand_tokens()),
        },
        "trees": trees,
    }

    model_json = DOCS / "model.json"
    model_json.write_text(json.dumps(artifact, separators=(",", ":")))
    psl_counts = export_psl(DOCS / "psl.json")

    print(f"trees            {len(trees):,}")
    print(f"nodes            {n_nodes:,}")
    print(f"features         {len(FEATURE_NAMES)}")
    print(f"base_score       {base_score:.10f}")
    print(f"feature spec     {artifact['feature_spec_sha256'][:16]}…")
    print(f"docs/model.json  {model_json.stat().st_size / 1e6:.2f} MB")
    print(f"docs/psl.json    {(DOCS / 'psl.json').stat().st_size / 1e6:.2f} MB "
          f"({psl_counts['icann']:,} ICANN + {psl_counts['private']:,} private rules)")


if __name__ == "__main__":
    main()
