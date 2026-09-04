#!/usr/bin/env python3
"""
Taxonomy granularity figure for SwissCrop25.

Shows mIoU and mF1 vs. taxonomy level (coarse → fine) for U-TAE, TSViT, and
Galileo-nano, illustrating how performance gaps between architectures widen as
the classification task becomes more fine-grained.

Data source: results/tables/perclass_metrics.csv (produced by analysis/compute_perclass_metrics.py)

Usage:
  python scripts/analysis/taxonomy_granularity_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).parents[2]
CSV_PATH = ROOT / "results/tables/perclass_metrics.csv"

MODELS = [
    ("U-TAE",        "UTAE"),
    ("TSViT",        "TSViT"),
    ("Galileo-nano", "Galileo-nano"),
]
MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# coarse → fine, matching the paper's hierarchy description
LEVELS      = ["lv3",  "lv2", "lv1", "leaf"]
LEVEL_NAMES = ["lv3\n(3)", "lv2\n(16)", "lv1\n(49)", "leaf\n(65)"]

OUTPUTS = [
    ROOT / "results/figures/taxonomy_granularity.pdf",
    ROOT / "Lauber_2026_SwissCrop_ECCV26_TerraBytes/figures/taxonomy_granularity.pdf",
]


def load_data():
    df = pd.read_csv(CSV_PATH)
    # crop-only means: for leaf level use is_ag=True rows; higher levels are already ag-only
    df_ag = df[df["is_ag"] == True]
    summary = (
        df_ag.groupby(["model", "level"])[["IoU", "F1"]]
        .mean()
        .reset_index()
    )
    return summary


def make_figure(summary):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), sharey=False)

    xs = np.arange(len(LEVELS))

    for ax, metric, ylabel in zip(
        axes,
        ["IoU", "F1"],
        ["mIoU (%)", "mF1 (%)"],
    ):
        for (label, model_key), color in zip(MODELS, MODEL_COLORS):
            ys = []
            for level in LEVELS:
                row = summary[(summary["model"] == model_key) & (summary["level"] == level)]
                ys.append(float(row[metric].values[0]) if len(row) else np.nan)

            ax.plot(xs, ys, color=color, linewidth=1.8, marker="o",
                    markersize=5, label=label)

        ax.set_xticks(xs)
        ax.set_xticklabels(LEVEL_NAMES, fontsize=8)
        ax.set_xlabel("Taxonomy level (# classes)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(-0.3, len(LEVELS) - 0.7)
        ax.set_ylim(bottom=0)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    axes[0].legend(fontsize=8, framealpha=0.8)
    fig.tight_layout()
    return fig


def main():
    summary = load_data()

    # print summary table
    print(f"\n{'Model':<16}  " + "  ".join(f"{n:<12}" for n in LEVELS))
    print("-" * 72)
    for label, model_key in MODELS:
        row_vals = []
        for level in LEVELS:
            row = summary[(summary["model"] == model_key) & (summary["level"] == level)]
            iou = float(row["IoU"].values[0]) if len(row) else float("nan")
            f1  = float(row["F1"].values[0])  if len(row) else float("nan")
            row_vals.append(f"{iou:.1f}/{f1:.1f}")
        print(f"{label:<16}  " + "  ".join(f"{v:<12}" for v in row_vals))
    print("(IoU/F1 per level, coarse→fine)")

    fig = make_figure(summary)
    for path in OUTPUTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Written: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
