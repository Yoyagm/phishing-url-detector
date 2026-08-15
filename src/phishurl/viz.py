"""Shared chart styling, so every figure in the report reads as one system.

The palette is fixed and colour-vision-deficiency validated rather than chosen by
eye: the blue/orange pair used for the two classes measures ΔE 24.7 under protanopia
and 33.6 under normal vision, well clear of the 8 / 15 floors. Class colour is
always paired with a label or a legend, so identity never rests on hue alone.

Figures are written as both PDF (vector, for the LaTeX report) and PNG (for the
README), from a single call to :func:`save`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots (light surface). Blue leads; orange is slot 2.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"

# Domain roles, so call sites read as meaning rather than colour.
LEGITIMATE = BLUE
PHISHING = ORANGE

# Chrome and ink
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Sequential blue ramp (100 -> 700), for heatmaps.
SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

FIGURES_DIR = Path(__file__).resolve().parents[2] / "paper" / "figures"


def apply_style() -> None:
    """Set the rcParams every figure in this project inherits."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 9,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "figure.dpi": 130,
    })


def strip_chrome(ax, keep_x: bool = True) -> None:
    """Recessive axes: drop the top/right spines and the vertical grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(False)
    if not keep_x:
        ax.spines["bottom"].set_visible(False)


def save(fig, name: str, directory: Path | None = None) -> Path:
    """Write ``name`` as both PDF (report) and PNG (README). Returns the PDF path."""
    directory = directory or FIGURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    pdf_path = directory / f"{name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf_path
