#!/usr/bin/env python3
"""
Three-panel supplementary figure: in-season mIoU split by class frequency.

Panel A  –  frequent crop classes (top-quartile GT pixel count), all three models
Panel B  –  rare crop classes (bottom-quartile GT pixel count), all three models
Panel C  –  TSViT − U-TAE mIoU advantage vs frequency percentile threshold,
            one curve per key month + full season (same x-axis convention as
            longtail_analysis.py: at threshold T, (100-T)% most frequent classes)

Panels A/B use the same quartile split fixed from global pooled frequency.

Usage:
  python scripts/figures/inseason_classfreq_figure.py
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.rcParams.update({"font.size": 11})

ROOT    = Path(__file__).parents[2]
STORAGE = ROOT / "storage"
sys.path.insert(0, str(ROOT / "code" / "analysis"))
from eval_utils import get_ag_indices, restrict_to_ag, load_cm

_AG_IDX, _ = get_ag_indices()
SPLITS  = ["S5", "S4", "S3", "S2", "S1"]
MONTHS  = list(range(1, 13))
MONTH_LABELS_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

UTAE_PREFIX    = "utae_gddsub_gddpe"
TSVIT_PREFIX   = "tsvit_gddsub_gddpe"
GALILEO_PREFIX = "galileo_nano_gddsub"

MODEL_COLORS = {"U-TAE": "#1f77b4", "TSViT": "#ff7f0e", "Galileo-nano": "#2ca02c"}

# Panel C: months to compare + full season
PANEL_C_MONTHS = [5, 6, 7, 8, 9, None]
PANEL_C_MONTH_LABEL = {
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep",
    None: "Full season",
}
# tab10 colours — maximally distinguishable
PANEL_C_COLORS = {
    5:    "#1f77b4",
    6:    "#2ca02c",
    7:    "#ff7f0e",
    8:    "#d62728",
    9:    "#9467bd",
    None: "black",
}

THRESHOLDS = list(range(0, 100, 10))  # [0, 10, …, 90]

OUTPUTS = [
    ROOT / "results/figures/inseason_classfreq.pdf",
    ROOT / "Lauber_2026_SwissCrop_ECCV26_TerraBytes/figures/inseason_classfreq.pdf",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def miou_subset(cm_ag, local_idx):
    eps = 1e-10
    tp = np.diag(cm_ag)[local_idx]
    fp = (cm_ag.sum(axis=0) - np.diag(cm_ag))[local_idx]
    fn = (cm_ag.sum(axis=1) - np.diag(cm_ag))[local_idx]
    has_gt = cm_ag.sum(axis=1)[local_idx] > 0
    iou = tp / (tp + fp + fn + eps)
    return float(iou[has_gt].mean() * 100) if has_gt.sum() > 0 else np.nan


def global_class_freq():
    total = np.zeros(len(_AG_IDX))
    for split in SPLITS:
        for prefix in (UTAE_PREFIX, TSVIT_PREFIX):
            try:
                cm_ag = restrict_to_ag(load_cm(STORAGE, f"{prefix}_{split}"), _AG_IDX)
                total += cm_ag.sum(axis=1)
            except FileNotFoundError:
                pass
    return total


# ── data collection ───────────────────────────────────────────────────────────

def collect_ab(freq_global):
    """
    Panels A & B: per-model, per-month mIoU for frequent and rare class subsets.
    Quartile split is fixed from the global pooled frequency.
    Returns dict: model_name -> {"frequent": array(n_splits, 12), "rare": ...}
    """
    q25 = np.percentile(freq_global, 25)
    q75 = np.percentile(freq_global, 75)
    rare_idx     = np.where(freq_global <= q25)[0]
    frequent_idx = np.where(freq_global >= q75)[0]

    result = {}
    for name, prefix in [("U-TAE", UTAE_PREFIX), ("TSViT", TSVIT_PREFIX),
                         ("Galileo-nano", GALILEO_PREFIX)]:
        freq_curves, rare_curves = [], []
        for split in SPLITS:
            folder = f"{prefix}_{split}"
            freq_row, rare_row = [], []
            for m in MONTHS:
                try:
                    cm_ag = restrict_to_ag(load_cm(STORAGE, folder, month=m), _AG_IDX)
                    freq_row.append(miou_subset(cm_ag, frequent_idx))
                    rare_row.append(miou_subset(cm_ag, rare_idx))
                except FileNotFoundError:
                    freq_row.append(np.nan)
                    rare_row.append(np.nan)
            if any(v == v for v in freq_row):
                freq_curves.append(freq_row)
                rare_curves.append(rare_row)
        result[name] = {
            "frequent": np.array(freq_curves, dtype=float),
            "rare":     np.array(rare_curves, dtype=float),
        }
    return result, frequent_idx, rare_idx


def collect_c(freq_global):
    """
    Panel C: TSViT − U-TAE mIoU difference at each frequency threshold and month.
    Class selection at each threshold uses the split-specific frequency (same
    computation as longtail_analysis.py) for split-accurate thresholding.
    Returns diff[threshold][month] = list of per-split differences.
    """
    diff = {t: {m: [] for m in PANEL_C_MONTHS} for t in THRESHOLDS}

    for split in SPLITS:
        # split-specific freq (averaged over both models)
        freqs = []
        for prefix in (UTAE_PREFIX, TSVIT_PREFIX):
            try:
                cm_ag = restrict_to_ag(load_cm(STORAGE, f"{prefix}_{split}"), _AG_IDX)
                freqs.append(cm_ag.sum(axis=1))
            except FileNotFoundError:
                pass
        if not freqs:
            continue
        freq_split = np.stack(freqs).mean(axis=0)

        try:
            cm_u_full = restrict_to_ag(load_cm(STORAGE, f"{UTAE_PREFIX}_{split}"),  _AG_IDX)
            cm_t_full = restrict_to_ag(load_cm(STORAGE, f"{TSVIT_PREFIX}_{split}"), _AG_IDX)
        except FileNotFoundError:
            continue

        for month in PANEL_C_MONTHS:
            if month is None:
                cm_u, cm_t = cm_u_full, cm_t_full
            else:
                try:
                    cm_u = restrict_to_ag(
                        load_cm(STORAGE, f"{UTAE_PREFIX}_{split}",  month=month), _AG_IDX)
                    cm_t = restrict_to_ag(
                        load_cm(STORAGE, f"{TSVIT_PREFIX}_{split}", month=month), _AG_IDX)
                except FileNotFoundError:
                    continue

            for t in THRESHOLDS:
                idx = np.where(freq_split >= np.percentile(freq_split, t))[0]
                if len(idx) == 0:
                    continue
                mu, mt = miou_subset(cm_u, idx), miou_subset(cm_t, idx)
                if not (np.isnan(mu) or np.isnan(mt)):
                    diff[t][month].append(mt - mu)

    return diff


# ── figure ────────────────────────────────────────────────────────────────────

def _plot_ab_panel(ax, data_ab, freq_key, title):
    xs = np.arange(1, 13)
    for name in ("U-TAE", "TSViT", "Galileo-nano"):
        curves = data_ab[name][freq_key]
        mean = np.nanmean(curves, axis=0)
        std  = np.nanstd(curves, axis=0)
        color = MODEL_COLORS[name]
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)
        ax.plot(xs, mean, color=color, linewidth=1.8, marker="o", markersize=4,
                label=name)
    ax.set_xticks(xs)
    every2 = [l if i % 2 == 0 else "" for i, l in enumerate(MONTH_LABELS_SHORT)]
    ax.set_xticklabels(every2, fontsize=11)
    ax.set_xlim(1, 12)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Month")
    ax.set_ylabel("mIoU (%)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(True, linewidth=0.4, alpha=0.5)


def _plot_c_panel(ax, diff, n_values):
    for m in PANEL_C_MONTHS:
        ys = [np.mean(diff[t][m]) if diff[t][m] else np.nan for t in THRESHOLDS]
        color   = PANEL_C_COLORS[m]
        ls      = "--" if m is None else "-"
        lw      = 2.2 if m is None else 1.8
        marker  = "s" if m is None else "o"
        ax.plot(THRESHOLDS, ys, color=color, linewidth=lw, linestyle=ls,
                marker=marker, markersize=4, label=PANEL_C_MONTH_LABEL[m])

    ax.axhline(0, color="black", linewidth=1.4, linestyle="--", alpha=0.5)
    ax.set_ylabel("TSViT $-$ U-TAE mIoU (pp)")
    ax.set_xlim(0, 90)
    ax.legend(fontsize=10, framealpha=0.8, loc="upper right", ncol=2)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_title(r"(c) $\Delta$mIoU (TSViT $-$ U-TAE)",
                 fontsize=11, fontweight="bold")

    # x-tick labels: threshold + n= count
    tick_labels = [f"{t}\n$n$={n}" for t, n in zip(THRESHOLDS, n_values)]
    ax.set_xticks(THRESHOLDS)
    ax.set_xticklabels(tick_labels, fontsize=10)

    # x-axis title above the tick labels
    ax.set_xlabel("Frequency percentile threshold (crop classes only)", labelpad=8)

    # Right-margin labels: place symmetrically above/below the zero line
    ymin, ymax = ax.get_ylim()
    zero_frac = (0 - ymin) / (ymax - ymin)
    ax.text(1.02, zero_frac + 0.26, "TSViT leads", transform=ax.transAxes,
            rotation=90, va="center", ha="left", fontsize=11, color="dimgray")
    ax.text(1.02, zero_frac - 0.19, "U-TAE leads", transform=ax.transAxes,
            rotation=90, va="center", ha="left", fontsize=11, color="dimgray")

    # Direction arrow + end labels below ticks (axes-fraction coords)
    arrow_y = -0.28
    ax.annotate("", xy=(1.0, arrow_y), xytext=(0.0, arrow_y),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.0))
    ax.text(0.0, arrow_y - 0.04, "all classes (incl. rare)",
            ha="left", va="top", transform=ax.transAxes, fontsize=11, color="dimgray")
    ax.text(1.0, arrow_y - 0.04, "frequent classes only",
            ha="right", va="top", transform=ax.transAxes, fontsize=11, color="dimgray")


def make_figure(data_ab, diff, n_values, frequent_idx, rare_idx, freq_global):
    fig = plt.figure(figsize=(9, 7.0))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[1, 1.15],
                            hspace=0.38, wspace=0.42,
                            bottom=0.15, top=0.97, left=0.08, right=0.97)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    n_freq = len(frequent_idx)
    n_rare = len(rare_idx)
    max_freq_px = freq_global[frequent_idx].min()   # min of top-Q = lower bound
    max_rare_px = freq_global[rare_idx].max()        # max of bottom-Q = upper bound

    _plot_ab_panel(ax_a, data_ab, "frequent",
                   f"(a) Most frequent crop classes\n(top quartile, $n$={n_freq})")
    _plot_ab_panel(ax_b, data_ab, "rare",
                   f"(b) Least frequent crop classes\n(bottom quartile, $n$={n_rare})")

    _plot_c_panel(ax_c, diff, n_values)

    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    freq_global = global_class_freq()
    n_values    = [int((freq_global >= np.percentile(freq_global, t)).sum())
                   for t in THRESHOLDS]

    print("Collecting Panels A & B …")
    data_ab, frequent_idx, rare_idx = collect_ab(freq_global)

    print("Collecting Panel C …")
    diff = collect_c(freq_global)

    # Summary printout
    for freq_key in ("frequent", "rare"):
        print(f"\n--- {freq_key} ---")
        print(f"{'Model':<10}" + "".join(f"{m:>6}" for m in MONTH_LABELS_SHORT))
        for name in ("U-TAE", "TSViT", "Galileo-nano"):
            mean = np.nanmean(data_ab[name][freq_key], axis=0)
            print(f"{name:<14}" + "".join(f"{v:6.1f}" for v in mean))

    print(f"\n{'Month':<14}" + "".join(f"  T={t:2d}" for t in THRESHOLDS))
    print("-" * 90)
    for m in PANEL_C_MONTHS:
        lbl = PANEL_C_MONTH_LABEL[m]
        row = f"{lbl:<14}" + "".join(
            f"  {np.mean(diff[t][m]):+5.1f}" if diff[t][m] else "    ---"
            for t in THRESHOLDS)
        print(row)
    print("n= classes   " + "".join(f"  {n:5d}" for n in n_values))

    fig = make_figure(data_ab, diff, n_values, frequent_idx, rare_idx, freq_global)
    for path in OUTPUTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Written: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
