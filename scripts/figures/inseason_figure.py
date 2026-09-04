#!/usr/bin/env python3
"""
In-season mIoU figure for SwissCrop25.

Loads test_metrics_month{1-12}.json for U-TAE, TSViT, Galileo-nano across
five LOYO splits, plots mean (± std) mIoU vs month and saves the figure.

Usage:
  python scripts/analysis/inseason_figure.py
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.size": 14})

ROOT    = Path(__file__).parents[2]
STORAGE = ROOT / "storage"
sys.path.insert(0, str(ROOT / "code" / "analysis"))
from eval_utils import get_ag_indices, crop_metrics_from_cm, load_cm

_AG_IDX, _ = get_ag_indices()
SPLITS  = ["S5", "S4", "S3", "S2", "S1"]
MONTHS  = list(range(1, 13))
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

MODELS = [
    ("U-TAE",        "utae_gddsub_gddpe"),
    ("TSViT",        "tsvit_gddsub_gddpe"),
    ("Galileo-nano", "galileo_nano_gddsub"),
]

MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

OUTPUTS = [
    ROOT / "results/figures/inseason_miou_oa.pdf",
    ROOT / "Lauber_2026_SwissCrop_ECCV26_TerraBytes/figures/inseason_miou_oa.pdf",
]


def load_metrics(res_dir, month=None):
    """Load ag-restricted OA and mIoU from pkl. Returns (oa, miou) or (None, None) if missing."""
    res_dir = Path(res_dir)
    try:
        cm = load_cm(res_dir.parent, res_dir.name, month=month)
        oa, _, miou, _ = crop_metrics_from_cm(cm, _AG_IDX)
        return oa, miou
    except FileNotFoundError:
        return None, None


def collect():
    data = {}
    for name, prefix in MODELS:
        miou_curves, oa_curves = [], []
        full_miou, full_oa = [], []
        for split in SPLITS:
            res_dir = STORAGE / f"{prefix}_{split}"
            miou_row, oa_row = [], []
            for m in MONTHS:
                oa, miou = load_metrics(res_dir, m)
                miou_row.append(miou)
                oa_row.append(oa)
            full_oa_v, full_miou_v = load_metrics(res_dir, month=None)
            if any(v is not None for v in miou_row):
                miou_curves.append(miou_row)
                oa_curves.append(oa_row)
            if full_miou_v is not None:
                full_miou.append(full_miou_v)
            if full_oa_v is not None:
                full_oa.append(full_oa_v)
        data[name] = {
            "miou_curves": miou_curves, "full_miou": full_miou,
            "oa_curves":   oa_curves,   "full_oa":   full_oa,
        }
    return data


EVERY2_LABELS = [l if i % 2 == 0 else "" for i, l in enumerate(MONTH_LABELS)]


def _plot_panel(ax, data, curve_key, full_key, ylabel, auc_title, ylim_top=None):
    xs = np.arange(1, 13)
    handles = []
    auc_lines = []
    for (name, _), color in zip(MODELS, MODEL_COLORS):
        curves = np.array(data[name][curve_key], dtype=float)
        mean = np.nanmean(curves, axis=0)
        std  = np.nanstd(curves, axis=0)
        auc  = float(np.nanmean(mean))
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)
        line, = ax.plot(xs, mean, color=color, linewidth=1.8, marker="o", markersize=4, label=name)
        handles.append(line)
        dec_val = mean[-1]
        if not np.isnan(dec_val):
            ax.axhline(dec_val, color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        if not np.all(np.isnan(mean)):
            auc_lines.append(f"{name}: {auc:.1f}%")
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels(EVERY2_LABELS, fontsize=13)
    ax.set_xlim(1, 12)
    ax.set_ylim(bottom=0, top=ylim_top)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.text(0.03, 0.97, f"$\\mathbf{{{auc_title}}}$\n" + "\n".join(auc_lines),
            transform=ax.transAxes, fontsize=12, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.8))
    return handles


def make_figure(data):
    fig, (ax_oa, ax_miou) = plt.subplots(1, 2, figsize=(10, 3.4))
    handles = _plot_panel(ax_oa,   data, "oa_curves",   "full_oa",   "OA (%)",    "AUC-OA",   ylim_top=100)
    _plot_panel(ax_miou, data, "miou_curves", "full_miou", "mIoU (%)", "AUC-mIoU")
    fig.legend(handles, [name for name, _ in MODELS],
               loc="lower center", ncol=3, fontsize=13, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


def main():
    data = collect()

    print(f"{'Model':<16}" + "".join(f"{m:>6}" for m in MONTH_LABELS) + f"{'AUC':>7}" + f"{'Full':>7}")
    print("-" * 100)
    for name, _ in MODELS:
        curves = np.array(data[name]["miou_curves"], dtype=float)
        mean   = np.nanmean(curves, axis=0)
        auc    = float(np.nanmean(mean))
        full   = np.mean(data[name]["full_miou"])
        row    = f"{name:<16}" + "".join(f"{v:6.1f}" for v in mean) + f"{auc:7.1f}" + f"{full:7.1f}"
        print(row)

    fig = make_figure(data)
    for path in OUTPUTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Written: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
