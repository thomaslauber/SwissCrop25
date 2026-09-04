#!/usr/bin/env python3
"""
Plot LNF crop maps for Switzerland, one PNG per year (2019-2025).

Rasterizes the LNF GeoPackage at 50m resolution using rasterio, maps lnf_code
to RGB via crop1990_colors.txt.txt (with Crop_Label remapping via SwissCrop25.xlsx),
and overlays the Swiss national boundary.

Used as a panel in the SwissCrop25 dataset overview figure.

Usage:
  python scripts/figures/swisscrop25_overview_lnf_map_figure.py
"""

import warnings
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.features
import rasterio.transform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT       = Path(__file__).parents[2]
GT_DIR     = Path("/mnt/eo-nas1/eoa-share/projects/020_crop1990/data/GTs")
COLOR_FILE = Path("/mnt/eo-nas1/eoa-share/projects/020_crop1990/data/crop1990_colors.txt.txt")
SWISS_BND  = ROOT / "results/figures/SwissCrop_overview/swiss_boundary.shp"
XLSX       = ROOT / "SwissCrop25.xlsx"
YEARS      = list(range(2019, 2026))
OUT_DIR    = ROOT / "results/figures/SwissCrop_overview"
RESOLUTION = 50  # metres

# Switzerland bounds in EPSG:2056 (with small margin)
XMIN, YMIN, XMAX, YMAX = 2484000, 1074000, 2834000, 1296000



def load_colors():
    """Parse color file: index R G B Name → {Name: (r, g, b)} normalised to [0,1]."""
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
    """Return {lnf_code (int): Crop_Label (str)} from SwissCrop25.xlsx label_sheet."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(XLSX, sheet_name="label_sheet")
    df = df[df["LNF_code"].notna() & df["Crop_Label"].notna()]
    return dict(zip(df["LNF_code"].astype(int), df["Crop_Label"]))


def build_lnf_to_rgba(lnf_to_label, colors):
    """Build array mapping lnf_code integer → (R, G, B, A) uint8; nodata/unmapped → transparent."""
    max_code = max(lnf_to_label.keys()) + 1
    lut = np.zeros((max_code, 4), dtype=np.uint8)  # default: transparent
    for code, label in lnf_to_label.items():
        if label in colors:
            r, g, b = colors[label]
            lut[code] = (int(r * 255), int(g * 255), int(b * 255), 255)
        # unmapped labels stay transparent (alpha=0)
    return lut


def rasterize_year(gpkg_path, resolution=RESOLUTION):
    """Rasterize lnf_code from gpkg to a national grid. Returns (lnf_raster, transform)."""
    width  = int((XMAX - XMIN) / resolution)
    height = int((YMAX - YMIN) / resolution)
    transform = rasterio.transform.from_bounds(XMIN, YMIN, XMAX, YMAX, width, height)

    gdf = gpd.read_file(gpkg_path, columns=["lnf_code"])
    gdf = gdf[gdf["lnf_code"].notna()]

    shapes = ((geom, int(code)) for geom, code in zip(gdf.geometry, gdf["lnf_code"]))
    raster = rasterio.features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32,
    )
    return raster, transform


def raster_to_rgba(raster, lut):
    """Map integer lnf_code raster to H×W×4 uint8 RGBA image; code=0 and unmapped → transparent."""
    h, w = raster.shape
    flat = raster.ravel()
    rgba_flat = lut[np.clip(flat, 0, len(lut) - 1)]
    rgba_flat[flat == 0] = 0  # nodata → fully transparent
    rgba_flat[flat >= len(lut)] = 0
    return rgba_flat.reshape(h, w, 4)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = load_colors()
    lnf_to_label = load_lnf_mapping()
    lut = build_lnf_to_rgba(lnf_to_label, colors)

    print("Loading Swiss outline...")
    swiss = gpd.read_file(SWISS_BND)

    for year in YEARS:
        gpkg = GT_DIR / f"LNF_swissTLM3D_{year}.gpkg"
        t0 = time.time()
        print(f"[{year}] Rasterizing...", end=" ", flush=True)

        raster, transform = rasterize_year(gpkg)
        rgba = raster_to_rgba(raster, lut)

        print(f"done in {time.time()-t0:.1f}s. Plotting...", end=" ", flush=True)

        extent = [XMIN, XMAX, YMIN, YMAX]
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("white")
        ax.patch.set_facecolor("white")
        ax.imshow(rgba, extent=extent, origin="upper", interpolation="nearest")
        swiss.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.6)
        ax.set_axis_off()
        fig.tight_layout(pad=0.3)

        out = OUT_DIR / f"lnf_map_{year}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"saved {out.name} ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
