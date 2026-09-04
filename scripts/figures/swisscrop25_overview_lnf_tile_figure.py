#!/usr/bin/env python3
"""
Plot a single grid tile from the LNF GeoPackage for a given year.

Used as a panel in the SwissCrop25 dataset overview figure.

Usage:
  python scripts/figures/swisscrop25_overview_lnf_tile_figure.py
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import box

ROOT       = Path(__file__).parents[2]
GT_DIR     = Path("/mnt/eo-nas1/eoa-share/projects/020_crop1990/data/GTs")
COLOR_FILE = Path("/mnt/eo-nas1/eoa-share/projects/020_crop1990/data/crop1990_colors.txt.txt")
GRID_SHP   = ROOT / "results/figures/SwissCrop_overview/gridface_s2tiles.shp"
XLSX       = ROOT / "SwissCrop25.xlsx"

TILE_ID = 16165
YEAR    = 2022
OUT     = ROOT / f"results/figures/SwissCrop_overview/lnf_tile_{TILE_ID}_{YEAR}.png"


def load_colors():
    colors = {}
    with open(COLOR_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            _, r, g, b, name = parts
            colors[name.strip()] = (int(r) / 255, int(g) / 255, int(b) / 255)
    return colors


def load_lnf_mapping():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(XLSX, sheet_name="label_sheet")
    df = df[df["LNF_code"].notna() & df["Crop_Label"].notna()]
    return dict(zip(df["LNF_code"].astype(int), df["Crop_Label"]))


def main():
    colors = load_colors()
    lnf_to_label = load_lnf_mapping()

    # Work entirely in EPSG:32632 to match S2 data (no rotation artefact)
    grid = gpd.read_file(GRID_SHP)  # already EPSG:32632
    tile = grid[grid["id"] == TILE_ID]
    tile_geom = tile.geometry.iloc[0]
    minx, miny, maxx, maxy = tile_geom.bounds

    # Load LNF parcels (EPSG:2056), reproject to 32632, then clip
    gpkg = GT_DIR / f"LNF_swissTLM3D_{YEAR}.gpkg"
    tile_2056 = tile.to_crs("EPSG:2056")
    bbox_2056 = tile_2056.total_bounds  # (minx, miny, maxx, maxy)
    gdf = gpd.read_file(gpkg, bbox=tuple(bbox_2056), columns=["lnf_code"])
    gdf = gdf.to_crs("EPSG:32632")
    gdf = gdf.clip(tile_geom)
    gdf["Crop_Label"] = gdf["lnf_code"].map(lnf_to_label)
    face_colors = [colors.get(lbl, (0.85, 0.85, 0.85)) if pd.notna(lbl) else (0.85, 0.85, 0.85)
                   for lbl in gdf["Crop_Label"]]

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")
    ax.patch.set_facecolor("white")
    gdf.plot(ax=ax, color=face_colors, linewidth=0.2, edgecolor="none")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_axis_off()
    ax.margins(0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
