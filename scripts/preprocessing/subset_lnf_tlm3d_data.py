"""
subset_lnf_tlm3d_data.py
==========================
Subset the LNF and TLM3D data to only cantons with sufficient coverage for that year.
Backs up each original file with a .bak suffix before overwriting.
"""

import argparse
import shutil
from pathlib import Path
import geopandas as gpd
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
GT_DIR = Path("/mnt/eo-nas1/eoa-share/projects/020_crop1990/data/GTs")
CANTONAL_COVERAGE_FILE = Path(
    "/mnt/eo-nas1/eoa-share/projects/020_crop1990/Crop1990/results/tables/lnf_canton_area_timeseries_pct.csv"
)

COVERAGE_THRESHOLD = 90.0


def main():
    coverage = pd.read_csv(CANTONAL_COVERAGE_FILE, index_col="year")

    for year in coverage.index:
        gt_path = GT_DIR / f"LNF_swissTLM3D_{year}.gpkg"
        if not gt_path.exists():
            print(f"[SKIP] {year}: no GT file found")
            continue

        row = coverage.loc[year]
        excluded = row[row < COVERAGE_THRESHOLD].index.tolist()
        if excluded:
            print(f"{year}: excluding cantons with insufficient coverage: {excluded}")

        valid_cantons = set(row[row >= COVERAGE_THRESHOLD].index.tolist())

        gdf = gpd.read_file(gt_path)
        before = len(gdf)

        dedup_before = len(gdf)
        gdf = gdf.drop_duplicates(subset=["year", "geometry"])
        print(f"  Deduped by year+geometry: {dedup_before - len(gdf):,} rows removed")

        gdf = gdf[gdf["canton"].isin(valid_cantons)]
        after = len(gdf)
        print(f"{year}: {before:,} → {after:,} rows ({before - after:,} removed)")

        bak_path = gt_path.with_suffix(".gpkg.bak")
        shutil.copy2(gt_path, bak_path)
        print(f"  Backup: {bak_path}")

        gt_path.unlink()
        gdf.to_file(gt_path, driver="GPKG", mode="w")
        print(f"  Saved: {gt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subset LNF/TLM3D GeoPackages to cantons with sufficient coverage.")
    parser.add_argument("--gt-dir", default=str(GT_DIR),
                        help=f"Directory containing LNF_swissTLM3D_{{year}}.gpkg files (default: {GT_DIR}).")
    parser.add_argument("--coverage-file", default=str(CANTONAL_COVERAGE_FILE),
                        help=f"Path to cantonal coverage PCT CSV (default: {CANTONAL_COVERAGE_FILE}).")
    parser.add_argument("--threshold", type=float, default=COVERAGE_THRESHOLD,
                        help=f"Minimum coverage percentage to keep a canton (default: {COVERAGE_THRESHOLD}).")
    args = parser.parse_args()

    GT_DIR = Path(args.gt_dir)
    CANTONAL_COVERAGE_FILE = Path(args.coverage_file)
    COVERAGE_THRESHOLD = args.threshold

    main()
