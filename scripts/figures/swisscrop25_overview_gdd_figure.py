#!/usr/bin/env python3
"""
Cumulative GDD time series for tile 379100_5224620, year 2022. Output: SVG.

Used as a panel in the SwissCrop25 dataset overview figure
(results/figures/SwissCrop_overview/gdd_2022.svg).

Usage:
  python scripts/figures/swisscrop25_overview_gdd_figure.py
"""

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ZARR_PATH = Path("/mnt/eo-nas1/eoa-share/projects/020_crop1990/data/CGDD_Sentinel/2022_sentinel/S2_379100_5224620_20220101_20221231.zarr")
ROOT      = Path(__file__).parents[2]
OUT       = ROOT / "results/figures/SwissCrop_overview/gdd_2022.svg"

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]


def main():
    zz   = zarr.open(str(ZARR_PATH), mode="r")
    cgdd = zz["CGDD_mean"][:].astype(np.float32)  # (365,)
    doy  = np.arange(1, len(cgdd) + 1)

    fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(doy, cgdd, color="#E30311", linewidth=2)
    ax.fill_between(doy, cgdd, alpha=0.12, color="#E30311")

    ax.set_xlim(1, 365)
    ax.set_ylim(0, cgdd.max() * 1.05)
    ax.set_xticks(MONTH_STARTS)
    ax.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax.set_ylabel("Cumul. GDD (°C·days)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="0.88", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="svg", bbox_inches="tight")
    print(f"Saved: {OUT}  (max GDD: {cgdd.max():.0f}°C·days)")

    # Second version: shape + axes only, no labels
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(doy, cgdd, color="#E30311", linewidth=3.5)
    ax2.fill_between(doy, cgdd, alpha=0.25, color="#E30311")
    ax2.set_xlim(1, 365)
    ax2.set_ylim(0, cgdd.max() * 1.05)
    ax2.set_xticks(MONTH_STARTS)
    ax2.set_xticklabels([])
    ax2.set_yticks([])
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.yaxis.grid(False)
    fig2.patch.set_alpha(0)
    ax2.patch.set_alpha(0)
    fig2.tight_layout()
    out2 = OUT.parent / "gdd_2022_nolabels.svg"
    fig2.savefig(out2, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig2)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
