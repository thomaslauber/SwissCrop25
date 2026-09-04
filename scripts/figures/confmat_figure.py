#!/usr/bin/env python3
"""
Pooled confusion matrix figure for SwissCrop25 (SM).

Pools conf_mat across all 5 LOYO splits per model, row-normalises,
and plots a 70×70 heatmap sorted by taxonomy group.
Three subplots (U-TAE, TSViT, Galileo-nano) arranged side by side.

Usage:
  python scripts/analysis/confmat_figure.py
"""

import sys
import pickle
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parents[2]
STORAGE = ROOT / "storage"
SPLITS  = ["S5", "S4", "S3", "S2", "S1"]

MODELS = [
    ("U-TAE",         "utae_gddsub_gddpe",   "test_metrics.pkl"),
    ("TSViT",         "tsvit_gddsub_gddpe",   "test_metrics.pkl"),
    ("Galileo-nano",  "galileo_nano_gddsub",  "conf_mat.pkl"),
]

OUTPUTS = [
    ROOT / "results/figures/confmat.pdf",
    ROOT / "Lauber_2026_SwissCrop_ECCV26_TerraBytes/figures/confmat.pdf",
]

MODEL_SLUGS = ["utae", "tsvit", "galileo"]

# taxonomy group → display order and colour
GROUP_ORDER  = ["Arable Land", "Grassland", "Permanent", "Forest",
                "Water", "Unproductive Area", "Built-up", "Wetland"]
GROUP_COLORS = {
    "Arable Land":      "#4e9a50",
    "Grassland":        "#a8d08d",
    "Permanent":        "#f4a460",
    "Forest":           "#8b6914",
    "Water":            "#5b9bd5",
    "Unproductive Area":"#bfbfbf",
    "Built-up":         "#e06c75",
    "Wetland":          "#85c1e9",
}


def load_class_info():
    df = pd.read_excel(ROOT / "SwissCrop25.xlsx", sheet_name="label_sheet")
    df = df[df["Exclude"] != True]
    df = df[df["Crop_Label"].notna()]
    # one row per unique label (take first occurrence)
    df = df.drop_duplicates(subset="Crop_Label", keep="first").reset_index(drop=True)
    # assign 1-based class index (matches conf_mat)
    df["class_idx"] = range(1, len(df) + 1)
    # normalise lv3 for landscape classes
    df["lv3"] = df["Crop_Label_lv3"].fillna(df["Crop_Label_lv2"])
    return df


def sort_order(df):
    """Return list of class indices sorted by lv3 (GROUP_ORDER) → lv2 → lv1 → LNF_code."""
    rows = []
    for grp in GROUP_ORDER:
        sub = df[df["lv3"] == grp].sort_values(
            ["Crop_Label_lv2", "Crop_Label_lv1", "LNF_code"]
        )
        rows.append(sub)
    ordered = pd.concat(rows, ignore_index=True)
    return ordered["class_idx"].tolist(), ordered


def pool_confmat(prefix, fname):
    cm = None
    for split in SPLITS:
        p = STORAGE / f"{prefix}_{split}" / fname
        if not p.exists():
            continue
        c = pickle.load(open(p, "rb")).astype(np.float64)
        cm = c if cm is None else cm + c
    return cm  # shape (71, 71) — row/col 0 = background


def row_normalise(cm, idx):
    """Extract submatrix for idx (1-based) and row-normalise."""
    sub = cm[np.ix_(idx, idx)]
    row_sums = sub.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return sub / row_sums * 100


def group_boundaries(ordered_df):
    """Return list of (start, end, group_name) for divider lines."""
    bounds = []
    prev_grp, start = None, 0
    for i, grp in enumerate(ordered_df["lv3"]):
        if grp != prev_grp:
            if prev_grp is not None:
                bounds.append((start, i, prev_grp))
            start = i
            prev_grp = grp
    bounds.append((start, len(ordered_df), prev_grp))
    return bounds


TICK_FS   = 12.0  # class label fontsize
TITLE_FS  = 16
AXIS_FS   = 13


def _draw_matrix(ax, cm_n, name, class_labels, bounds):
    """Draw heatmap with grid, group bands, and tick labels. Returns AxesImage."""
    n = len(class_labels)
    im = ax.imshow(cm_n, aspect="equal", interpolation="nearest",
                   cmap="Blues", vmin=0, vmax=100)

    # cell-level grid
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.2, alpha=0.3)
    ax.tick_params(which="minor", length=0)

    # thick group boundary lines
    for start, end, grp in bounds:
        for pos in [start - 0.5, end - 0.5]:
            ax.axhline(pos, color="white", linewidth=1.2, alpha=0.85)
            ax.axvline(pos, color="white", linewidth=1.2, alpha=0.85)

    # colour bands on left spine
    for start, end, grp in bounds:
        ax.add_patch(mpatches.Rectangle(
            (-3.0, start - 0.5), 1.5, end - start,
            transform=ax.transData, clip_on=False,
            color=GROUP_COLORS.get(grp, "#cccccc"), linewidth=0))

    ax.set_title(f"{name} — Confusion Matrix", fontsize=TITLE_FS, fontweight="bold", pad=7)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    tick_labels = [l[:16] for l in class_labels]
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=TICK_FS, ha="center", va="top")
    ax.set_yticklabels(tick_labels, fontsize=TICK_FS)

    ax.set_xlabel("Predicted", fontsize=AXIS_FS, labelpad=4)
    ax.set_ylabel("Ground truth", fontsize=AXIS_FS, labelpad=4)

    return im


def make_single_figure(cm_n, name, ordered_df, class_labels):
    """Individual full-width figure for one model."""
    bounds = group_boundaries(ordered_df)

    # figsize chosen so axes are square:
    # ax_width_in  = 0.80 × 18 = 14.40
    # ax_height_in = 0.740 × 19 = 14.06  → no right-side whitespace
    # left/bottom margins larger to fit 12pt tick labels
    fig = plt.figure(figsize=(18, 19))
    ax = fig.add_axes([0.15, 0.23, 0.80, 0.740])

    im = _draw_matrix(ax, cm_n, name, class_labels, bounds)

    # narrow colorbar bottom-left; leave visible gap below the matrix
    cax = fig.add_axes([0.15, 0.11, 0.24, 0.018])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Recall (%)", fontsize=AXIS_FS)
    cbar.ax.tick_params(labelsize=11)

    # group legend to the right of the colorbar
    legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=g)
                      for g in GROUP_ORDER if g in GROUP_COLORS]
    fig.legend(handles=legend_patches, fontsize=13,
               bbox_to_anchor=(0.96, 0.135), loc="upper right",
               ncol=2, framealpha=0.85, handlelength=1.4, columnspacing=1.2)

    return fig


def main():
    df = load_class_info()
    idx_order, ordered_df = sort_order(df)
    class_labels = ordered_df["Crop_Label"].tolist()

    cms_norm = []
    for name, prefix, fname in MODELS:
        cm = pool_confmat(prefix, fname)
        if cm is None:
            print(f"WARNING: no data for {name}")
            cms_norm.append(np.zeros((len(idx_order), len(idx_order))))
            continue
        cm_n = row_normalise(cm, idx_order)
        cms_norm.append(cm_n)
        acc = np.diag(cm_n).mean()
        print(f"{name}: mean recall = {acc:.1f}%  (pooled over available splits)")

    # individual figures
    for (name, _, __), cm_n, slug in zip(MODELS, cms_norm, MODEL_SLUGS):
        fig = make_single_figure(cm_n, name, ordered_df, class_labels)
        for base in [ROOT / "results/figures", ROOT / "Lauber_2026_SwissCrop_ECCV26_TerraBytes/figures"]:
            base.mkdir(parents=True, exist_ok=True)
            path = base / f"confmat_{slug}.pdf"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Written: {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
