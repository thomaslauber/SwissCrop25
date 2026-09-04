#!/usr/bin/env python3
"""Poster version of taxonomy granularity figure — single mIoU panel, larger fonts."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.size": 14})
import numpy as np
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).parents[2]
CSV_PATH = ROOT / "results/tables/perclass_metrics.csv"
OUTPUT   = ROOT / "documentation/poster_taxonomy_granularity.pdf"

MODELS = [
    ("U-TAE",        "UTAE"),
    ("TSViT",        "TSViT"),
    ("Galileo-nano", "Galileo-nano"),
]
MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

LEVELS      = ["lv3",  "lv2", "lv1", "leaf"]
LEVEL_NAMES = ["lv3\n(3)", "lv2\n(16)", "lv1\n(49)", "leaf\n(65)"]


def main():
    df = pd.read_csv(CSV_PATH)
    df_ag = df[df["is_ag"] == True]
    summary = (
        df_ag.groupby(["model", "level"])[["IoU", "F1"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    xs = np.arange(len(LEVELS))

    for (label, model_key), color in zip(MODELS, MODEL_COLORS):
        ys = []
        for level in LEVELS:
            row = summary[(summary["model"] == model_key) & (summary["level"] == level)]
            ys.append(float(row["IoU"].values[0]) if len(row) else np.nan)
        ax.plot(xs, ys, color=color, linewidth=2.5, marker="o", markersize=8, label=label)

    ax.set_xticks(xs)
    ax.set_xticklabels(LEVEL_NAMES)
    ax.set_xlabel("Taxonomy level (# classes)")
    ax.set_ylabel("mIoU (%)")
    ax.set_xlim(-0.3, len(LEVELS) - 0.7)
    ax.set_ylim(bottom=0)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=13, framealpha=0.8)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Written: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
