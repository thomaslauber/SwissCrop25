#!/usr/bin/env python3
# longtail_figure.py
# Crop mIoU above frequency threshold figure for SwissCrop25 supplementary.
# Non-agricultural classes (Forest, Water, Built-up, Unproductive Area, Wetland)
# are excluded so the analysis reflects crop classification difficulty only.
#
# Usage:
#   python scripts/figures/longtail_figure.py

import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "code" / "analysis"))
from eval_utils import get_ag_indices, restrict_to_ag

STORAGE = ROOT / "storage"
SPLITS  = ["S5", "S4", "S3", "S2", "S1"]

MODELS = [
    ("U-TAE",                 "utae_gddsub_gddpe"),
    ("TSViT",                 "tsvit_gddsub_gddpe"),
    ("Galileo-nano",          "galileo_nano_gddsub"),
    ("Galileo-nano (frozen)", "galileo_nano_frozen_gddsub"),
    ("Galileo-base (frozen)", "galileo_base_frozen_gddsub"),
]

OUTPUTS = [
    ROOT / "results/figures/longtail_scatter.pdf",
    ROOT / "Lauber_2026_SwissCrop_ECCV26_TerraBytes/figures/longtail_scatter.pdf",
]

MODEL_COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MODEL_MARKERS = ["o", "s", "^", "D", "v"]


def load_cm_ag(folder, ag_idx):
    p = STORAGE / folder / "conf_mat.pkl"
    if not p.exists():
        p = STORAGE / folder / "test_metrics.pkl"
    if not p.exists():
        return None
    cm = pickle.load(open(p, "rb")).astype(np.float64)
    return restrict_to_ag(cm, ag_idx)


def per_class_iou_freq(cm):
    eps  = 1e-10
    tp   = np.diag(cm)
    fp   = cm.sum(axis=0) - tp
    fn   = cm.sum(axis=1) - tp
    iou  = tp / (tp + fp + fn + eps)
    freq = cm.sum(axis=1)
    has_gt = freq > 0
    return iou[has_gt], freq[has_gt]


def collect(ag_idx):
    THRESHOLDS = list(range(0, 100, 10))
    curve_vals = {name: {t: [] for t in THRESHOLDS} for name, _ in MODELS}
    freq_global = np.zeros(len(ag_idx))

    for split in SPLITS:
        for name, prefix in MODELS:
            cm = load_cm_ag(f"{prefix}_{split}", ag_idx)
            if cm is None:
                continue
            iou, freq = per_class_iou_freq(cm)
            freq_global[:len(freq)] += freq
            for t in THRESHOLDS:
                thresh = np.percentile(freq, t)
                mask   = freq >= thresh
                if mask.sum() > 0:
                    curve_vals[name][t].append(iou[mask].mean() * 100)

    curves = {name: {t: np.mean(v) if v else None for t, v in curve_vals[name].items()}
              for name, _ in MODELS}
    n_values = [int((freq_global >= np.percentile(freq_global, t)).sum())
                for t in THRESHOLDS]
    return curves, THRESHOLDS, n_values


def make_figure(curves, thresholds, n_values):
    fig, ax = plt.subplots(figsize=(6, 4))
    for (name, _), color, marker in zip(MODELS, MODEL_COLORS, MODEL_MARKERS):
        c  = curves[name]
        xs = [t for t in thresholds if c[t] is not None]
        ys = [c[t] for t in thresholds if c[t] is not None]
        if not xs:
            continue
        ax.plot(xs, ys, color=color, linewidth=1.8, marker=marker, markersize=5, label=name)
    ax.set_xlabel("Frequency percentile threshold (crop classes only)", labelpad=8)
    ax.set_ylabel("mIoU (%)")
    ax.set_xlim(0, 90)
    ax.set_xticks(thresholds)
    tick_labels = [f"{t}\n$n$={n}" for t, n in zip(thresholds, n_values)]
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.legend(fontsize=7, framealpha=0.7)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    arrow_y = -0.30
    ax.annotate("", xy=(1.0, arrow_y), xytext=(0.0, arrow_y),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.0))
    ax.text(0.0, arrow_y - 0.05, "all classes (incl. rare)",
            ha="left", va="top", transform=ax.transAxes, fontsize=9, color="dimgray")
    ax.text(1.0, arrow_y - 0.05, "frequent classes only",
            ha="right", va="top", transform=ax.transAxes, fontsize=9, color="dimgray")

    fig.tight_layout()
    return fig


def main():
    ag_idx, _ = get_ag_indices()
    curves, thresholds, n_values = collect(ag_idx)
    fig = make_figure(curves, thresholds, n_values)
    for path in OUTPUTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Written: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
