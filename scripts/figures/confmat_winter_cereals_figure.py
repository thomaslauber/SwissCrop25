#!/usr/bin/env python3
"""
Winter cereal confusion matrix figure for SwissCrop25 supplementary.

Two-panel figure comparing TSViT on the 2024 test split (S4):
  (a) Baseline: DOY sampling with cloud filtering (tsvit_cloudsub_S4)
  (b) T3S + TPE: GDD-bin reindexing + thermal positional encoding
      (tsvit_gddsub_gddpe_S4)

Rows = 5 winter cereal classes; columns = 5 winter cereals + "Other"
(aggregated predictions outside the group). Row-normalised (recall %).

Usage:
  python scripts/figures/confmat_winter_cereals_figure.py
"""

import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).parents[2]
STORAGE = ROOT / "storage"

CONFIGS = [
    ("(a) Without phenological alignment", "tsvit_cloudsub_S4",                    "test_metrics.pkl"),
    ("(b) With phenological alignment ($T^{3}S$ + TPE)", "tsvit_gddsub_gddpe_S4", "test_metrics.pkl"),
]

OUTPUTS = [
    ROOT / "results/figures/confmat_winter_cereals_2024.pdf",
    ROOT / "Lauber_2026_SwissCrop_ECCV26_TerraBytes/figures/confmat_winter_cereals_2024.pdf",
]

LV2_WINTER = "Winter Cereals"
AG_LV3     = ["Arable Land", "Permanent", "Grassland"]


def load_class_info():
    df = pd.read_excel(ROOT / "SwissCrop25.xlsx", sheet_name="label_sheet")
    df = df[df["Exclude"] != True]
    df = df[df["Crop_Label"].notna()]
    df = df.drop_duplicates(subset="Crop_Label", keep="first").reset_index(drop=True)
    df["class_idx"] = range(1, len(df) + 1)
    return df


def make_panel_matrix(cm, wc_idx, ag_idx):
    """
    Extract rows for winter cereals; columns = winter cereals + Other (ag only).
    Returns (n_wc x (n_wc+1)) row-normalised matrix (recall %).
    """
    rows = cm[np.ix_(wc_idx, ag_idx)]               # (5, n_ag)
    # within ag columns, split winter cereal vs other ag
    wc_set = set(wc_idx)
    wc_pos  = [i for i, idx in enumerate(ag_idx) if idx in wc_set]
    other_pos = [i for i, idx in enumerate(ag_idx) if idx not in wc_set]
    wc_cols   = rows[:, wc_pos]                      # (5, 5)
    other_col = rows[:, other_pos].sum(axis=1, keepdims=True)  # (5, 1)
    mat = np.hstack([wc_cols, other_col])            # (5, 6)

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mat / row_sums * 100


def main():
    df = load_class_info()
    wc = df[df["Crop_Label_lv2"] == LV2_WINTER].sort_values("class_idx")
    ag = df[df["Crop_Label_lv3"].isin(AG_LV3)].sort_values("class_idx")
    wc_idx    = wc["class_idx"].tolist()
    ag_idx    = ag["class_idx"].tolist()
    wc_labels = wc["Crop_Label"].tolist()
    col_labels = wc_labels + ["Other (ag)"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("TSViT — 2024 Test Split", fontsize=16, fontweight="bold", y=1.01)

    last_im = None
    for i, (ax, (title, folder, fname)) in enumerate(zip(axes, CONFIGS)):
        path = STORAGE / folder / fname
        if not path.exists():
            print(f"WARNING: not found: {path}")
            continue
        cm = pickle.load(open(path, "rb")).astype(np.float64)
        mat = make_panel_matrix(cm, wc_idx, ag_idx)

        n_rows, n_cols = mat.shape
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=100,
                       interpolation="nearest")
        last_im = im

        # Grid
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="gray", linewidth=0.5, alpha=0.4)
        ax.tick_params(which="minor", length=0)

        # Dashed separator before "Other" column
        ax.axvline(n_cols - 1 - 0.5, color="#888888", linewidth=1.2,
                   linestyle="--", alpha=0.8)

        # Cell annotations
        for r in range(n_rows):
            for c in range(n_cols):
                val = mat[r, c]
                color = "white" if val > 55 else "black"
                ax.text(c, r, f"{val:.0f}", ha="center", va="center",
                        fontsize=13, color=color)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=15)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(wc_labels if i == 0 else [], fontsize=15)
        ax.set_xlabel("Predicted", fontsize=15)
        if i == 0:
            ax.set_ylabel("Ground truth", fontsize=15)
        ax.set_title(title, fontsize=15, pad=10)

    # Single colorbar on the right
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes[1], fraction=0.046, pad=0.04,
                            orientation="vertical")
        cbar.set_label("Recall (%)", fontsize=14)
        cbar.ax.tick_params(labelsize=13)

    fig.tight_layout(w_pad=0.0)

    for path in OUTPUTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Written: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
