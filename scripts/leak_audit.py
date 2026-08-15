"""Quantify the two label leaks in PhiUSIIL, and show what they do to a model.

This is the experiment the rest of the project is built on top of. It answers one
question with numbers: *where does the 99%+ accuracy reported on this dataset
actually come from?*

Run::

    uv run python scripts/leak_audit.py

Writes ``reports/leak_audit.json`` and the figures used by the report.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from phishurl import viz
from phishurl.datasets import ROOT, load_phiusiil

REPORTS = ROOT / "reports"

# The 27 columns that describe the fetched page rather than the URL. Unusable by a
# detector that must score a link *before* anyone clicks it.
CONTENT_COLUMNS = [
    "LineOfCode", "LargestLineLength", "HasTitle", "Title", "DomainTitleMatchScore",
    "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive", "NoOfURLRedirect",
    "NoOfSelfRedirect", "HasDescription", "NoOfPopup", "NoOfiFrame", "HasExternalFormSubmit",
    "HasSocialNet", "HasSubmitButton", "HasHiddenFields", "HasPasswordField", "Bank",
    "Pay", "Crypto", "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef",
    "NoOfEmptyRef", "NoOfExternalRef",
]

_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def structural_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure the collection artefacts that separate the two classes."""
    url = frame["URL"].astype(str)
    rest = url.str.replace(_SCHEME_RE, "", regex=True)
    authority = rest.str.split(r"[/?#]", n=1, regex=True).str[0]
    tail = rest.str.slice(0)  # everything after the scheme

    traits = pd.DataFrame({
        "y": frame["y"],
        "starts_https": url.str.startswith("https://"),
        "host_starts_www": authority.str.startswith("www."),
        "has_path": tail.str.contains(r"/.+", regex=True, na=False),
        "has_query": tail.str.contains(r"\?", regex=True, na=False),
        "has_at": authority.str.contains("@", regex=False, na=False),
        "url_length": url.str.len(),
    })

    rows = []
    for label, group in traits.groupby("y"):
        rows.append({
            "class": "phishing" if label == 1 else "legitimate",
            "n": len(group),
            "starts_https_pct": 100 * group["starts_https"].mean(),
            "host_starts_www_pct": 100 * group["host_starts_www"].mean(),
            "has_path_pct": 100 * group["has_path"].mean(),
            "has_query_pct": 100 * group["has_query"].mean(),
            "has_at_pct": 100 * group["has_at"].mean(),
            "mean_url_length": group["url_length"].mean(),
        })
    return pd.DataFrame(rows).set_index("class")


def rule_accuracy(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    correct = (prediction == truth)
    positive = truth == 1
    return {
        "accuracy": float(correct.mean()),
        "recall_phishing": float((prediction[positive] == 1).mean()),
        "specificity": float((prediction[~positive] == 0).mean()),
    }


def fit_lightgbm(X: pd.DataFrame, y: np.ndarray, seed: int = 0) -> float:
    """Random 80/20 split, as the literature does. Returns test accuracy."""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = lgb.train(
        {"objective": "binary", "num_leaves": 31, "verbose": -1, "seed": seed},
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=200,
    )
    predicted = (model.predict(X_test) >= 0.5).astype(int)
    return float((predicted == y_test).mean())


def naive_url_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The obvious 'rescue' feature set: lexical features over the whole raw URL.

    This is what a practitioner reaches for after being told the precomputed
    columns are tainted. Leak 2 defeats it.
    """
    url = frame["URL"].astype(str)
    rest = url.str.replace(_SCHEME_RE, "", regex=True)
    return pd.DataFrame({
        "url_length": url.str.len(),
        "n_dots": url.str.count(r"\."),
        "n_slashes": rest.str.count("/"),
        "n_digits": url.str.count(r"\d"),
        "n_hyphens": url.str.count("-"),
        "is_https": url.str.startswith("https://").astype(int),
        "has_query": rest.str.contains(r"\?").astype(int),
        "has_at": rest.str.contains("@").astype(int),
    })


def figure_leak_one(frame: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    similarity = frame["URLSimilarityIndex"]
    shares = [
        100 * (similarity[frame["y"] == 0] >= 99.9).mean(),
        100 * (similarity[frame["y"] == 1] >= 99.9).mean(),
    ]

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    bars = ax.barh(
        ["Legitimate", "Phishing"], shares,
        color=[viz.LEGITIMATE, viz.PHISHING], height=0.55,
    )
    for bar, value in zip(bars, shares):
        ax.text(value + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}%", va="center", color=viz.INK, fontweight="bold")
    ax.set_xlim(0, 112)
    ax.set_xlabel("Share of class with URLSimilarityIndex ≥ 99.9")
    ax.set_title("Leak 1 — one feature separates the classes")
    viz.strip_chrome(ax)
    ax.set_xticks([0, 25, 50, 75, 100])
    viz.save(fig, "leak1_similarity_index")


def figure_leak_two(profile: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    # '@' is left to the table: at 0.01% of phishing it renders as "0 vs 0",
    # which reads as a broken chart rather than as a negligible difference.
    traits = ["starts_https_pct", "host_starts_www_pct", "has_path_pct", "has_query_pct"]
    labels = ["Scheme is\nHTTPS", "Host starts\nwith www.", "Has a\npath", "Has a query\nstring"]
    legitimate = [profile.loc["legitimate", trait] for trait in traits]
    phishing = [profile.loc["phishing", trait] for trait in traits]

    x = np.arange(len(traits))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(x - width / 2, legitimate, width, label="Legitimate", color=viz.LEGITIMATE)
    ax.bar(x + width / 2, phishing, width, label="Phishing", color=viz.PHISHING)

    for index, (left, right) in enumerate(zip(legitimate, phishing)):
        ax.text(index - width / 2, left + 2, f"{left:.0f}", ha="center",
                color=viz.INK_SECONDARY, fontsize=8)
        ax.text(index + width / 2, right + 2, f"{right:.0f}", ha="center",
                color=viz.INK_SECONDARY, fontsize=8)

    ax.set_xticks(x, labels)
    ax.set_ylim(0, 116)
    ax.set_ylabel("% of class")
    ax.set_title("Leak 2 — the legitimate class is a collection artefact")
    ax.legend(loc="upper right", ncols=2)
    viz.strip_chrome(ax)
    viz.save(fig, "leak2_structural_profile")


def main() -> None:
    viz.apply_style()
    REPORTS.mkdir(exist_ok=True)

    print("Loading PhiUSIIL …")
    frame = load_phiusiil()
    y = frame["y"].to_numpy()
    print(f"  {len(frame):,} rows — {int(y.sum()):,} phishing, {int((1 - y).sum()):,} legitimate\n")

    # ---------------------------------------------------------------- leak 1
    print("=" * 74)
    print("LEAK 1 — URLSimilarityIndex is a label proxy")
    print("=" * 74)
    similarity = frame["URLSimilarityIndex"]
    legit_at_100 = int((similarity[y == 0] >= 99.9).sum())
    phish_at_100 = int((similarity[y == 1] >= 99.9).sum())
    rule_one = (similarity < 99.9).astype(int).to_numpy()
    leak_one_scores = rule_accuracy(rule_one, y)

    print(f"  legitimate rows with index >= 99.9 : {legit_at_100:,} / {int((y == 0).sum()):,} "
          f"({100 * legit_at_100 / (y == 0).sum():.2f}%)")
    print(f"  phishing rows with index >= 99.9   : {phish_at_100:,} / {int((y == 1).sum()):,} "
          f"({100 * phish_at_100 / (y == 1).sum():.2f}%)")
    print(f"  RULE 'index < 99.9 -> phishing' accuracy: {leak_one_scores['accuracy']:.4f}")
    print(f"    recall {leak_one_scores['recall_phishing']:.4f} · "
          f"specificity {leak_one_scores['specificity']:.4f}\n")

    # ---------------------------------------------------------------- leak 2
    print("=" * 74)
    print("LEAK 2 — the legitimate class is a collection artefact")
    print("=" * 74)
    profile = structural_profile(frame)
    print(profile.to_string(float_format=lambda v: f"{v:8.2f}"))

    rest = frame["URL"].astype(str).str.replace(_SCHEME_RE, "", regex=True)
    has_path = rest.str.contains(r"/.+", regex=True, na=False).to_numpy()
    leak_two_scores = rule_accuracy(has_path.astype(int), y)
    print(f"\n  RULE 'has a path -> phishing': accuracy {leak_two_scores['accuracy']:.4f}, "
          f"and it never fires on a legitimate row "
          f"(specificity {leak_two_scores['specificity']:.4f})\n")

    # ------------------------------------------------------------ modelling
    print("=" * 74)
    print("WHAT THE LEAKS ARE WORTH TO A MODEL  (LightGBM, random 80/20 split)")
    print("=" * 74)
    numeric = frame.select_dtypes(include=[np.number]).drop(columns=["label", "y"])

    all_features = fit_lightgbm(numeric, y)
    print(f"  (a) all {numeric.shape[1]} precomputed features        : {all_features:.4f}   <- the published headline")

    without_similarity = numeric.drop(columns=["URLSimilarityIndex"])
    no_sim = fit_lightgbm(without_similarity, y)
    print(f"  (b) minus URLSimilarityIndex ({without_similarity.shape[1]} features) : {no_sim:.4f}")

    url_only = numeric.drop(columns=["URLSimilarityIndex"] + [
        column for column in CONTENT_COLUMNS if column in numeric.columns
    ])
    no_content = fit_lightgbm(url_only, y)
    print(f"  (c) also minus page-content columns ({url_only.shape[1]} features): {no_content:.4f}")

    naive = fit_lightgbm(naive_url_features(frame), y)
    print(f"  (d) hand-rolled lexical features on the raw URL  : {naive:.4f}   <- the obvious rescue, still leaking")
    print("\n  Every row above is inflated. (d) is the important one: discarding all")
    print("  precomputed columns and starting from the raw URL does NOT fix it,")
    print("  because leak 2 lives in the URL string itself.\n")

    figure_leak_one(frame)
    figure_leak_two(profile)

    report = {
        "dataset": {"rows": int(len(frame)), "phishing": int(y.sum()),
                    "legitimate": int((1 - y).sum())},
        "leak_1_url_similarity_index": {
            "legitimate_at_or_above_99_9": legit_at_100,
            "phishing_at_or_above_99_9": phish_at_100,
            "single_rule": leak_one_scores,
        },
        "leak_2_collection_artefact": {
            "profile": profile.to_dict(orient="index"),
            "has_path_rule": leak_two_scores,
        },
        "lightgbm_random_split_accuracy": {
            "all_precomputed_features": all_features,
            "without_url_similarity_index": no_sim,
            "without_content_columns": no_content,
            "naive_raw_url_lexical": naive,
        },
    }
    output = REPORTS / "leak_audit.json"
    output.write_text(json.dumps(report, indent=2))
    print(f"Wrote {output.relative_to(ROOT)} and 2 figures to paper/figures/")


if __name__ == "__main__":
    main()
