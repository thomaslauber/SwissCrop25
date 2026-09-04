#!/usr/bin/env python3
"""
Label distribution figure for SwissCrop25.

Shows pixel-equivalent area per agricultural class (averaged over 2021--2025,
the five complete national-coverage years), sorted descending within each lv3
taxonomy group. Classes are coloured by lv2 subgroup.

Data source: SwissCrop25.xlsx (label_sheet), columns *_Area_m2.

Usage:
  python scripts/analysis/label_dist_figure.py
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

ROOT       = Path(__file__).parents[2]
LABEL_XLSX = ROOT / "SwissCrop25.xlsx"

OUTPUTS = [
    ROOT / "results/figures/label_dist.pdf",
    ROOT / "Lauber_2026_SwissCrop_ECCV26_TerraBytes/figures/label_dist.pdf",
]

# 5 landscape classes (non-agricultural, from swissTLM3D)
LANDSCAPE_LABELS = {"Forest", "Water", "Built-up", "Unproductive Area", "Wetland"}

# Complete national-coverage years only
COMPLETE_YEARS = [2021, 2022, 2023, 2024, 2025]

# lv3 display order (top to bottom in the plot)
LV3_ORDER = ["Arable Land", "Grassland", "Permanent", "Non-crop"]

# Color per lv2 subgroup — warm/cool hue families track lv3 membership
LV2_COLORS = {
    # Arable Land — blues & warm tones
    "Winter Cereals":         "#1f4e79",
    "Spring Cereals":         "#2e75b6",
    "Summer Cereals":         "#9dc3e6",
    "Root Crops":             "#7f3f00",
    "Oilseeds":               "#ed7d31",
    "Pulses":                 "#ffd966",
    "Annual Horticulture":    "#c55a11",
    "Temporary Grassland":    "#a8d08d",
    "Rice":                   "#bdd7ee",
    # Grassland — greens
    "Meadow":                 "#375623",
    "Pasture":                "#548235",
    "Alpine Pasture":         "#a9d18e",
    # Permanent — purples
    "Orchard":                "#7030a0",
    "Perennial Horticulture": "#bf86d9",
    "Nursery":                "#d9b8f0",
    "Greenhouses":            "#e2d3ed",
    # Woody Crops (Chestnut, Christmas Trees, Hedges) — browns
    "Woody Crops":            "#833c00",
    # Non-crop landscape classes — greys
    "Non-crop":               "#aaaaaa",
}


def load_classes():
    df = pd.read_excel(LABEL_XLSX, sheet_name="label_sheet")
    df = df[df["Exclude"] != True]
    df = df[df["Crop_Label"].notna()]
    # tag landscape classes with a synthetic lv2 for colouring
    df = df.copy()
    df.loc[df["Crop_Label"].isin(LANDSCAPE_LABELS), "Crop_Label_lv2"] = "Non-crop"
    df.loc[df["Crop_Label"].isin(LANDSCAPE_LABELS), "Crop_Label_lv3"] = "Non-crop"
    return df


def area_col(year: int) -> str:
    """Return the xlsx column name for a given year."""
    # Columns are named e.g. '2019_Area_m2', '2021_Area_m22', '2022_Area_m23' ...
    cols = pd.read_excel(LABEL_XLSX, sheet_name="label_sheet", nrows=0).columns.tolist()
    matches = [c for c in cols if c.startswith(str(year))]
    assert len(matches) == 1, f"Could not find unique column for {year}: {matches}"
    return matches[0]


def compute_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each Crop_Label, compute mean area (ha) averaged over COMPLETE_YEARS,
    summing over all LNF codes that map to the same label.
    """
    year_cols = [area_col(y) for y in COMPLETE_YEARS]

    df = df.copy()
    df["mean_area_m2"] = df[year_cols].mean(axis=1)

    agg = (
        df.groupby(["Crop_Label", "Crop_Label_lv2", "Crop_Label_lv3"])
        ["mean_area_m2"]
        .sum()
        .reset_index()
    )
    agg["mean_area_ha"] = agg["mean_area_m2"] / 1e4
    agg["mean_pixels"] = agg["mean_area_m2"] / 100  # at 10 m resolution
    return agg


def sort_classes(agg: pd.DataFrame) -> pd.DataFrame:
    """Sort within each lv3 group by area descending; lv3 groups in LV3_ORDER."""
    # classes with lv3 not in LV3_ORDER (e.g. Forest ag) go last
    def lv3_rank(lv3):
        try:
            return LV3_ORDER.index(lv3)
        except ValueError:
            return len(LV3_ORDER)

    agg = agg.copy()
    agg["_lv3_rank"] = agg["Crop_Label_lv3"].map(lv3_rank)
    agg = agg.sort_values(["_lv3_rank", "mean_pixels"], ascending=[True, False])
    agg = agg.drop(columns="_lv3_rank").reset_index(drop=True)
    return agg


def make_figure(agg: pd.DataFrame):
    n = len(agg)
    fig, ax = plt.subplots(figsize=(7.0, 0.115 * n + 0.6))

    ys = np.arange(n)
    colors = [LV2_COLORS.get(lv2, "#999999") for lv2 in agg["Crop_Label_lv2"]]

    ax.barh(ys, agg["mean_pixels"], color=colors, height=0.7, linewidth=0)
    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels(agg["Crop_Label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_ylim(n - 0.5, -0.5)  # remove top/bottom whitespace
    ax.set_xlabel("Mean pixel count (2021–2025, log scale)", fontsize=9)
    ax.grid(True, axis="x", linewidth=0.25, alpha=0.35, which="major")
    ax.tick_params(axis="x", labelsize=8)

    LV2_DISPLAY = {}

    # --- legend: one patch per lv2 group, placed below axes ---
    legend_elements = []
    seen = set()
    for lv3 in LV3_ORDER + [g for g in agg["Crop_Label_lv3"].unique() if g not in LV3_ORDER]:
        subset = agg[agg["Crop_Label_lv3"] == lv3]
        if subset.empty:
            continue
        for lv2 in subset["Crop_Label_lv2"].unique():
            if lv2 in seen:
                continue
            seen.add(lv2)
            color = LV2_COLORS.get(lv2, "#999999")
            label = LV2_DISPLAY.get(lv2, lv2)
            legend_elements.append(
                mpatches.Patch(facecolor=color, label=label, linewidth=0)
            )

    ax.legend(
        handles=legend_elements,
        fontsize=5.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        framealpha=0.9,
        ncol=4,
        handlelength=1.0,
        handleheight=0.8,
        columnspacing=1.0,
    )

    # draw horizontal dividers between lv3 groups
    prev_lv3 = None
    for i, row in agg.iterrows():
        if prev_lv3 is not None and row["Crop_Label_lv3"] != prev_lv3:
            ax.axhline(i - 0.5, color="black", linewidth=0.5, alpha=0.35)
        prev_lv3 = row["Crop_Label_lv3"]

    fig.tight_layout()
    return fig


def main():
    df = load_classes()
    agg = compute_distribution(df)
    agg = sort_classes(agg)

    print(f"Classes: {len(agg)}")
    print(f"Pixel range: {agg['mean_pixels'].min():.0f} -- {agg['mean_pixels'].max():.2e}")
    print(f"Imbalance ratio: {agg['mean_pixels'].max() / agg['mean_pixels'].min():.0f}:1")
    print()
    print(agg[["Crop_Label", "Crop_Label_lv2", "Crop_Label_lv3", "mean_area_ha"]].to_string(index=False))

    fig = make_figure(agg)
    for path in OUTPUTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Written: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
