"""Train the host-name classifier and evaluate it honestly.

Two decisions here carry most of the weight:

* **The split is grouped by registrable domain**, so no eTLD+1 appears on both
  sides. Without it the model memorises domains and the held-out score is a
  measure of memory, not of generalisation.
* **The out-of-distribution set is the real test.** It pairs Tranco hosts the
  model has never seen with PhishTank hosts submitted *after* the training data
  was collected. Performance is expected to fall there; that fall is the result.

Run::

    uv run python -m phishurl.train
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from phishurl import evaluate, viz
from phishurl.datasets import ROOT, build_ood_frame, build_training_frame
from phishurl.features import FEATURE_NAMES, extract_many

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

PARAMS = {
    "objective": "binary",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 50,
    "verbose": -1,
    "seed": 42,
    "num_threads": 4,
}
NUM_ROUNDS = 300
SEED = 42


def build_matrix(hosts: list[str]) -> np.ndarray:
    return np.asarray(extract_many(hosts), dtype=np.float64)


def main() -> None:
    viz.apply_style()
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    print("Building the training frame from PhiUSIIL …")
    frame = build_training_frame()
    conflicts = frame.attrs.get("n_conflicting_hosts", 0)
    print(f"  {len(frame):,} unique hosts "
          f"({int(frame['y'].sum()):,} phishing, {int((1 - frame['y']).sum()):,} legitimate)")
    print(f"  dropped {conflicts:,} hosts that carried both labels")
    print(f"  {frame['registrable_domain'].nunique():,} distinct registrable domains (split groups)")

    X = build_matrix(frame["host"].tolist())
    y = frame["y"].to_numpy()
    groups = frame["registrable_domain"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_index, test_index = next(splitter.split(X, y, groups))
    print(f"\nGrouped split: {len(train_index):,} train / {len(test_index):,} test")

    overlap = set(groups[train_index]) & set(groups[test_index])
    assert not overlap, f"{len(overlap)} registrable domains leaked across the split"
    print("  no registrable domain appears on both sides")

    dataset = lgb.Dataset(X[train_index], label=y[train_index], feature_name=list(FEATURE_NAMES))
    print(f"\nTraining LightGBM — {NUM_ROUNDS} rounds, {len(FEATURE_NAMES)} features …")
    model = lgb.train(PARAMS, dataset, num_boost_round=NUM_ROUNDS)
    model.save_model(str(MODELS / "model.txt"))

    internal = evaluate.compute("Internal hold-out", y[test_index], model.predict(X[test_index]))

    print("\nBuilding the out-of-distribution / temporal set …")
    train_hosts = set(frame["host"])
    ood = build_ood_frame(train_hosts)
    for key, value in ood.stats.items():
        print(f"  {key:38s} {value:>9,}")

    ood_scores = model.predict(build_matrix(ood.frame["host"].tolist()))
    out_of_distribution = evaluate.compute("OOD + temporal", ood.frame["y"].to_numpy(), ood_scores)

    # OpenPhish is phishing-only, so ROC is undefined; report detection rate at the
    # threshold chosen on the OOD set instead.
    openphish_scores = model.predict(build_matrix(ood.openphish["host"].tolist()))
    openphish_detected = float((openphish_scores >= out_of_distribution.threshold_at_fpr_1pct).mean())

    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)
    header = f"{'':22s} {'ROC-AUC':>9s} {'PR-AUC':>9s} {'R@FPR1%':>9s} {'R@FPR0.1%':>10s}"
    print(header)
    for metrics in (internal, out_of_distribution):
        print(f"{metrics.name:22s} {metrics.roc_auc:9.4f} {metrics.pr_auc:9.4f} "
              f"{metrics.recall_at_fpr_1pct:9.4f} {metrics.recall_at_fpr_01pct:10.4f}")
    print(f"\nOpenPhish (live, {len(openphish_scores)} hosts, phishing only): "
          f"{openphish_detected:.1%} detected at the OOD operating point")

    drop = internal.recall_at_fpr_1pct - out_of_distribution.recall_at_fpr_1pct
    print(f"\nRecall drop from internal to out-of-distribution: {drop:+.4f}")
    print("That drop is the honest cost of leaving the benchmark's own distribution.")

    evaluate.figure_curves(
        [("Internal hold-out", y[test_index], model.predict(X[test_index])),
         ("OOD + temporal", ood.frame["y"].to_numpy(), ood_scores)],
        "roc_pr_curves",
    )
    evaluate.figure_confusion(out_of_distribution, "confusion_ood")
    evaluate.figure_confusion(internal, "confusion_internal")

    report = {
        "training": {
            "unique_hosts": int(len(frame)),
            "dropped_conflicting_hosts": int(conflicts),
            "registrable_domains": int(frame["registrable_domain"].nunique()),
            "n_features": len(FEATURE_NAMES),
            "params": PARAMS,
            "num_boost_round": NUM_ROUNDS,
        },
        "internal": internal.to_dict(),
        "ood": out_of_distribution.to_dict(),
        "ood_composition": ood.stats,
        "openphish_detection_rate_at_ood_threshold": openphish_detected,
        "recall_drop_internal_to_ood": float(drop),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {(REPORTS / 'metrics.json').relative_to(ROOT)}, "
          f"{(MODELS / 'model.txt').relative_to(ROOT)} and 3 figures.")


if __name__ == "__main__":
    main()
