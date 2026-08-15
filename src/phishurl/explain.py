"""Global explainability: which features drive the model across the whole set.

The demo explains one host at a time; this answers the other question a reviewer
asks — what did the model learn in general. Mean absolute SHAP value per feature,
computed on a sample of the out-of-distribution set so the answer describes
behaviour on data the model never saw.

Run::

    uv run python -m phishurl.explain
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np

from phishurl import viz
from phishurl.datasets import ROOT, build_ood_frame, build_training_frame
from phishurl.features import FEATURE_NAMES, extract_many

SAMPLE = 4000
TOP_N = 15


def pretty(name: str) -> str:
    """Readable axis labels; the 26 suffix one-hots collapse into one entry."""
    labels = {
        "n_subdomain_labels": "Subdomain depth",
        "subdomain_length": "Subdomain length",
        "host_length": "Host length",
        "entropy_host": "Character entropy (host)",
        "entropy_registrable": "Character entropy (domain)",
        "registrable_length": "Registrable domain length",
        "is_multi_level_suffix": "Multi-level suffix (.co.uk)",
        "has_suspicious_keyword": "Login/verify keyword",
        "brand_in_subdomain_only": "Brand in subdomain, not domain",
        "brand_token_present": "Known brand present",
        "is_platform_hosted": "Platform-hosted subdomain",
        "is_ip_literal": "Raw IP as host",
        "has_punycode": "Punycode (xn--)",
        "max_consecutive_consonants": "Longest consonant run",
        "max_consecutive_digits": "Longest digit run",
        "digits_in_registrable": "Digits in domain",
        "mean_label_length": "Mean label length",
        "max_label_length": "Longest label",
        "suffix_length": "Suffix length",
        "n_labels": "Number of labels",
        "digit_count": "Digits",
        "digit_ratio": "Digit ratio",
        "hyphen_count": "Hyphens",
        "hyphen_ratio": "Hyphen ratio",
        "vowel_ratio": "Vowel ratio",
    }
    return labels.get(name, name)


def main() -> None:
    import shap

    viz.apply_style()

    booster = lgb.Booster(model_file=str(ROOT / "models" / "model.txt"))
    train_hosts = set(build_training_frame()["host"])
    ood = build_ood_frame(train_hosts)

    rng = np.random.default_rng(42)
    index = rng.choice(len(ood.frame), size=min(SAMPLE, len(ood.frame)), replace=False)
    hosts = ood.frame["host"].to_numpy()[index]
    matrix = np.asarray(extract_many(list(hosts)))

    values = np.asarray(shap.TreeExplainer(booster).shap_values(matrix))
    if values.ndim == 3:
        values = values[:, :, -1]

    # Sum the suffix one-hots into a single "Top-level domain" entry: they encode
    # one decision, and summing SHAP values is exact.
    importance: dict[str, float] = {}
    suffix_total = np.zeros(len(matrix))
    for position, name in enumerate(FEATURE_NAMES):
        if name.startswith("suffix_is_"):
            suffix_total += values[:, position]
        else:
            importance[pretty(name)] = float(np.abs(values[:, position]).mean())
    importance["Top-level domain"] = float(np.abs(suffix_total).mean())

    ranked = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:TOP_N]
    names = [name for name, _ in ranked][::-1]
    scores = [score for _, score in ranked][::-1]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    bars = ax.barh(names, scores, color=viz.BLUE, height=0.62)
    for bar, score in zip(bars, scores):
        ax.text(score + max(scores) * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", color=viz.INK_SECONDARY, fontsize=8)

    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title(f"What the model uses — {len(hosts):,} out-of-distribution hosts")
    ax.set_xlim(0, max(scores) * 1.14)
    viz.strip_chrome(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=viz.GRID, linewidth=0.7)
    viz.save(fig, "shap_global_importance")

    output = ROOT / "reports" / "shap_importance.json"
    output.write_text(json.dumps(dict(ranked[::-1]), indent=2))

    print(f"Sampled {len(hosts):,} out-of-distribution hosts")
    for name, score in ranked[::-1]:
        print(f"  {name:34s} {score:.4f}")
    print("\nWrote paper/figures/shap_global_importance.{pdf,png} and reports/shap_importance.json")


if __name__ == "__main__":
    main()
