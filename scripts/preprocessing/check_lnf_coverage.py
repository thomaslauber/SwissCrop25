"""
check_lnf_coverage.py
==========================
Check the coverage of LNF parcels across years per canton,
split into land-use groups based on LNF classification.

Uses the pre-processed GT GeoPackages (LNF_swissTLM3D_{year}.gpkg) which already
contain `canton` and `year` columns, avoiding the need for spatial joins or deduplication.

Flags:
  --latex   Skip recomputation; read existing _pct CSV and print LaTeX table rows.
"""

import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
GT_DIR = Path("/mnt/eo-nas1/eoa-share/projects/020_crop1990/data/GTs")

CLASS_FILE = Path(
    "/mnt/eo-nas1/eoa-share/projects/020_crop1990/data/LNF_code_classification_20260217.xlsx"
)

OUTFILE = Path(
    "/mnt/eo-nas1/eoa-share/projects/020_crop1990/Crop1990/results/tables/lnf_canton_area_timeseries.csv"
)

YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

COVERAGE_THRESHOLD = 90.0

# English display names and abbreviations, keyed by the DataFrame column name
CANTON_LABELS = {
    "Aargau":                    "Aargau (AG)",
    "Appenzell Innerrhoden":     "Appenzell Inner Rhodes (AI)",
    "Appenzell Ausserrhoden":    "Appenzell Outer Rhodes (AR)",
    "Basel-Landschaft":          "Basel-Land (BL)",
    "Basel-Stadt":               "Basel-Stadt (BS)",
    "Bern":                      "Bern (BE)",
    "Fribourg":                  "Fribourg (FR)",
    "Genève":                    "Geneva (GE)",
    "Glarus":                    "Glarus (GL)",
    "Graubünden":                "Grisons (GR)",
    "Jura":                      "Jura (JU)",
    "Luzern":                    "Lucerne (LU)",
    "Neuchâtel":                 "Neuch\\^{a}tel (NE)",
    "Nidwalden":                 "Nidwalden (NW)",
    "Obwalden":                  "Obwalden (OW)",
    "Schaffhausen":              "Schaffhausen (SH)",
    "Schwyz":                    "Schwyz (SZ)",
    "Solothurn":                 "Solothurn (SO)",
    "St. Gallen":                "St.~Gallen (SG)",
    "Thurgau":                   "Thurgau (TG)",
    "Ticino":                    "Ticino (TI)",
    "Uri":                       "Uri (UR)",
    "Valais":                    "Valais (VS)",
    "Vaud":                      "Vaud (VD)",
    "Zug":                       "Zug (ZG)",
    "Zürich":                    "Z\\\"urich (ZH)",
}

# Special cell annotations: (data_column_name, year) -> LaTeX suffix added inside the cell
CELL_ANNOTATIONS = {
    ("Ticino", 2021): r"$^{\dagger}$",
}


def generate_latex(outfile: Path = None):
    # Recompute pct directly from the raw CSV
    crop_area = pd.read_csv(OUTFILE)
    groups_to_sum = ["Grassland", "Arable_And_Permanent_Land"]
    crop_areas = (
        crop_area[crop_area["group"].isin(groups_to_sum)]
        .groupby(["canton", "year"], as_index=False)["area_m2"]
        .sum()
    )
    wide = crop_areas.pivot_table(index="year", columns="canton", values="area_m2", fill_value=0)
    pct = wide.divide(wide.max(axis=0), axis=1) * 100

    # Canton names and sort order come from the CSV; CANTON_LABELS provides English display names
    cantons = sorted(pct.columns.tolist(), key=lambda c: CANTON_LABELS.get(c, c))
    years = sorted(pct.index.tolist())
    lines = []

    for canton in cantons:
        label = CANTON_LABELS.get(canton, canton)
        cells = []
        for y in years:
            val = pct.loc[y, canton]
            annotation = CELL_ANNOTATIONS.get((canton, y), "")
            if val < COVERAGE_THRESHOLD:
                cells.append("{\\textbf{" + f"{val:.1f}{annotation}" + "}}")
            else:
                cells.append("{" + f"{val:.1f}{annotation}" + "}")

        row = f"{label:<40} & " + " & ".join(cells) + r" \\"
        lines.append(row)

    tex_body = "\n".join(lines)
    print(tex_body)

    tex_out = OUTFILE.with_name("appendix_lnf_coverage_body.tex")
    tex_out.write_text(tex_body + "\n")
    print(f"\nSaved: {tex_out}")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_classification():
    df = pd.read_excel(CLASS_FILE)
    df = df.rename(columns={
        "LNF_code": "lnf_code",
        "Crop_Label_lv3": "lv3",
        "Crop_Label_lv2": "lv2",
    })
    df = df[df["Exclude"] != True]
    return df[["lnf_code", "lv2", "lv3"]].copy()


# ── Processing ────────────────────────────────────────────────────────────────

def process_year(year, class_df):
    print(f"Processing {year}...")

    gt_path = GT_DIR / f"LNF_swissTLM3D_{year}.gpkg"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing file: {gt_path}")

    # Load only LNF rows — TLM rows have no agricultural lnf_code
    gdf = gpd.read_file(gt_path, where="\"layer\" = 'LNF'")
    print(f"  {len(gdf):,} LNF rows")

    # ── Dedup by year + geometry ───────────────────────────────────────────
    # Ticino 2021 contains genuine duplicate parcels; identical geometry+year
    # is a reliable signal across all years without relying on uuid.
    before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["year", "geometry"])
    print(f"  Deduped by year+geometry: {before - len(gdf):,} rows removed")

    # ── Merge classification ────────────────────────────────────────────────
    gdf["lnf_code"] = gdf["lnf_code"].astype(int)
    class_df = class_df.copy()
    class_df["lnf_code"] = class_df["lnf_code"].astype(int)
    gdf = gdf.merge(class_df, on="lnf_code", how="left")

    # ── Define land-use groups ─────────────────────────────────────────────
    gdf["group"] = None

    gdf.loc[gdf["lv3"].isin(["Arable Land", "Permanent"]), "group"] = "Arable_And_Permanent_Land"
    gdf.loc[(gdf["lv3"] == "Grassland") & (gdf["lv2"] != "Alpine Pasture"), "group"] = "Grassland"
    gdf.loc[gdf["lv2"] == "Alpine Pasture", "group"] = "Alpine_Pasture"

    gdf = gdf[gdf["group"].notna()].copy()

    # ── Compute area and aggregate ─────────────────────────────────────────
    gdf["area_m2"] = gdf.geometry.area

    agg = (
        gdf.groupby(["canton", "group"])["area_m2"]
        .sum()
        .reset_index()
    )
    agg["year"] = year
    return agg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    class_df = load_classification()

    results = []
    for year in YEARS:
        try:
            res = process_year(year, class_df)
            results.append(res)
        except Exception as e:
            print(f"[ERROR] Year {year}: {e}")

    final = pd.concat(results, ignore_index=True)
    final.to_csv(OUTFILE, index=False)
    print(f"Saved: {OUTFILE}")


def save_pct_csv():
    crop_area = pd.read_csv(OUTFILE)
    groups_to_sum = ["Grassland", "Arable_And_Permanent_Land"]
    crop_areas = (
        crop_area[crop_area["group"].isin(groups_to_sum)]
        .groupby(["canton", "year"], as_index=False)["area_m2"]
        .sum()
    )
    wide = crop_areas.pivot_table(
        index="year",
        columns="canton",
        values="area_m2",
        fill_value=0
    )
    pct = wide.divide(wide.max(axis=0), axis=1) * 100
    pct_path = OUTFILE.with_name(OUTFILE.stem + "_pct" + OUTFILE.suffix)
    pct.to_csv(pct_path)
    print(f"Saved: {pct_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true",
                        help="Read existing raw CSV and print LaTeX table rows (no recomputation).")
    parser.add_argument("--gt-dir", default=str(GT_DIR),
                        help=f"Directory containing LNF_swissTLM3D_{{year}}.gpkg files (default: {GT_DIR}).")
    parser.add_argument("--class-file", default=str(CLASS_FILE),
                        help=f"Path to LNF classification Excel file (default: {CLASS_FILE}).")
    parser.add_argument("--outfile", default=str(OUTFILE),
                        help=f"Output CSV path (default: {OUTFILE}).")
    args = parser.parse_args()

    GT_DIR = Path(args.gt_dir)
    CLASS_FILE = Path(args.class_file)
    OUTFILE = Path(args.outfile)

    if args.latex:
        generate_latex()
    else:
        main()
        save_pct_csv()
