#!/usr/bin/env python3
"""Poster version of winter cereal confusion matrix — auto-scaled Blues colormap."""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.size": 14})
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).parents[2]
STORAGE = ROOT / "storage"
OUTPUT  = ROOT / "documentation/poster_confmat_winter_cereals.pdf"

CONFIGS = [
    ("Without phen. alignment", "tsvit_cloudsub_S4", "test_metrics.pkl"),
    ("With phen. alignment", "tsvit_gddsub_gddpe_S4", "test_metrics.pkl"),
]

LV2_WINTER   = "Winter Cereals"
AG_LV3       = ["Arable Land", "Permanent", "Grassland"]
EXCLUDE_ROWS = {"Spelt"}


def load_class_info():
    df = pd.read_excel(ROOT / "SwissCrop25.xlsx", sheet_name="label_sheet")
    df = df[df["Exclude"] != True]
    df = df[df["Crop_Label"].notna()]
    df = df.drop_duplicates(subset="Crop_Label", keep="first").reset_index(drop=True)
    df["class_idx"] = range(1, len(df) + 1)
    return df


def make_panel_matrix(cm, row_idx, col_idx):
    """Rows = filtered winter cereals; columns = all winter cereals. Row-normalised."""
    mat = cm[np.ix_(row_idx, col_idx)].astype(np.float64)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mat / row_sums * 100


def main():
    df = load_class_info()
    wc_all = df[df["Crop_Label_lv2"] == LV2_WINTER].sort_values("class_idx")
    ag     = df[df["Crop_Label_lv3"].isin(AG_LV3)].sort_values("class_idx")
    wc     = wc_all[~wc_all["Crop_Label"].isin(EXCLUDE_ROWS)]
    wc_idx    = wc["class_idx"].tolist()
    wc_all_idx = wc_all["class_idx"].tolist()
    ag_idx    = ag["class_idx"].tolist()
    wc_labels = wc["Crop_Label"].tolist()
    col_labels = wc_labels

    # Load all matrices first to get shared colour scale
    matrices = []
    for _, folder, fname in CONFIGS:
        path = STORAGE / folder / fname
        if path.exists():
            cm = pickle.load(open(path, "rb")).astype(np.float64)
            matrices.append(make_panel_matrix(cm, wc_idx, wc_idx))
        else:
            matrices.append(None)
    vmax = 75

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    last_im = None
    for i, (ax, (title, _, __), mat) in enumerate(zip(axes, CONFIGS, matrices)):
        if mat is None:
            continue
        n_rows, n_cols = mat.shape
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=vmax,
                       interpolation="nearest")
        last_im = im

        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="gray", linewidth=0.5, alpha=0.4)
        ax.tick_params(which="minor", length=0)

        for r in range(n_rows):
            for c in range(n_cols):
                val = mat[r, c]
                color = "white" if val > vmax * 0.55 else "black"
                ax.text(c, r, f"{val:.0f}", ha="center", va="center",
                        fontsize=13, color=color)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=15)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(wc_labels if i == 0 else [], fontsize=15)
        if i == 0:
            ax.set_ylabel("Ground truth", fontsize=15)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes[1], fraction=0.046, pad=0.04,
                            orientation="vertical")
        cbar.set_label("Recall (%)", fontsize=14)
        cbar.ax.tick_params(labelsize=13)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    fig.text(0.5, -0.02, "Predicted", ha="center", fontsize=15)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Written: {OUTPUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
