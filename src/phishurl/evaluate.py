"""Metrics and figures.

The primary metric is **recall at a fixed false-positive rate**, not accuracy.
A false positive here blocks a legitimate site, which is the expensive error for
anyone who would actually deploy this, and accuracy on a near-balanced training
set says nothing about behaviour at the operating point that matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from phishurl import viz


@dataclass
class Metrics:
    """Everything reported about one model on one evaluation set."""

    name: str
    n: int
    n_phishing: int
    roc_auc: float
    pr_auc: float
    recall_at_fpr_1pct: float
    threshold_at_fpr_1pct: float
    recall_at_fpr_01pct: float
    threshold_at_fpr_01pct: float
    accuracy_at_operating_point: float
    precision_at_operating_point: float
    f1_at_operating_point: float
    confusion: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "n": self.n, "n_phishing": self.n_phishing,
            "roc_auc": self.roc_auc, "pr_auc": self.pr_auc,
            "recall_at_fpr_1pct": self.recall_at_fpr_1pct,
            "threshold_at_fpr_1pct": self.threshold_at_fpr_1pct,
            "recall_at_fpr_01pct": self.recall_at_fpr_01pct,
            "threshold_at_fpr_01pct": self.threshold_at_fpr_01pct,
            "accuracy_at_operating_point": self.accuracy_at_operating_point,
            "precision_at_operating_point": self.precision_at_operating_point,
            "f1_at_operating_point": self.f1_at_operating_point,
            "confusion": self.confusion,
        }


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """Highest recall reachable without exceeding ``target_fpr``.

    Returns ``(recall, threshold)``. If no threshold keeps the false-positive rate
    under the target, returns ``(0.0, 1.0)`` — the honest answer, rather than the
    nearest achievable point dressed up as a hit.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    allowed = fpr <= target_fpr
    if not allowed.any():
        return 0.0, 1.0
    index = int(np.max(np.where(allowed)[0]))
    return float(tpr[index]), float(thresholds[index])


def compute(name: str, y_true: np.ndarray, scores: np.ndarray) -> Metrics:
    """Score one model on one set, with the operating point fixed at FPR ≤ 1 %."""
    recall_1, threshold_1 = recall_at_fpr(y_true, scores, 0.01)
    recall_01, threshold_01 = recall_at_fpr(y_true, scores, 0.001)

    predicted = (scores >= threshold_1).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return Metrics(
        name=name,
        n=int(len(y_true)),
        n_phishing=int(y_true.sum()),
        roc_auc=float(roc_auc_score(y_true, scores)),
        pr_auc=float(average_precision_score(y_true, scores)),
        recall_at_fpr_1pct=recall_1,
        threshold_at_fpr_1pct=threshold_1,
        recall_at_fpr_01pct=recall_01,
        threshold_at_fpr_01pct=threshold_01,
        accuracy_at_operating_point=float((predicted == y_true).mean()),
        precision_at_operating_point=float(precision),
        f1_at_operating_point=float(2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0,
        confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    )


def figure_curves(curves: list[tuple[str, np.ndarray, np.ndarray]], name: str) -> None:
    """ROC and precision-recall side by side, one line per evaluation set."""
    import matplotlib.pyplot as plt

    colours = [viz.BLUE, viz.ORANGE, viz.AQUA]
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(8.6, 3.6))

    for (label, y_true, scores), colour in zip(curves, colours):
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        roc_ax.plot(fpr, tpr, color=colour, label=f"{label} (AUC {auc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        pr_ax.plot(recall, precision, color=colour, label=f"{label} (AP {ap:.3f})")

    roc_ax.plot([0, 1], [0, 1], color=viz.MUTED, linewidth=1, linestyle=(0, (4, 3)))
    roc_ax.axvline(0.01, color=viz.MUTED, linewidth=1, linestyle=(0, (2, 2)))
    roc_ax.text(0.012, 0.06, "operating point\nFPR 1%", color=viz.MUTED, fontsize=7.5)
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")
    roc_ax.set_title("ROC")
    roc_ax.set_xlim(0, 1)
    roc_ax.set_ylim(0, 1.02)
    roc_ax.legend(loc="lower right")

    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Precision–recall")
    pr_ax.set_xlim(0, 1)
    pr_ax.set_ylim(0, 1.02)
    pr_ax.legend(loc="lower left")

    for axis in (roc_ax, pr_ax):
        viz.strip_chrome(axis)
        axis.xaxis.grid(True, color=viz.GRID, linewidth=0.7)

    viz.save(fig, name)


def figure_confusion(metrics: Metrics, name: str) -> None:
    """Confusion matrix at the operating point, as counts on a sequential ramp."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    counts = np.array([[metrics.confusion["tn"], metrics.confusion["fp"]],
                       [metrics.confusion["fn"], metrics.confusion["tp"]]], dtype=float)
    shares = counts / counts.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("seq_blue", viz.SEQUENTIAL)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.imshow(shares, cmap=cmap, vmin=0, vmax=1)

    for row in range(2):
        for column in range(2):
            # Ink flips to white on the dark end of the ramp so labels stay legible.
            colour = "#ffffff" if shares[row, column] > 0.55 else viz.INK
            ax.text(column, row, f"{int(counts[row, column]):,}\n{shares[row, column]:.1%}",
                    ha="center", va="center", color=colour, fontsize=9.5, fontweight="bold")

    ax.set_xticks([0, 1], ["Predicted\nlegitimate", "Predicted\nphishing"])
    ax.set_yticks([0, 1], ["Actually\nlegitimate", "Actually\nphishing"])
    ax.set_title(f"{metrics.name} — at FPR ≤ 1 %")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    viz.save(fig, name)
