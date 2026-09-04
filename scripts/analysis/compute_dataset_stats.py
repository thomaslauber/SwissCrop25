"""
Compute summary statistics from SwissCrop25.xlsx for use in the paper.

TLM3D additions are identified by LNF_code >= 1900 (25 rows, all with null
HCATv2_Identifier). Codes < 1900 are official LNF codes.

Run with:
    ~/.virtualenvs/020_crop1990/bin/python compute_dataset_stats.py
"""

import pandas as pd
from pathlib import Path

LNF_FILE = Path("/mnt/eo-nas1/eoa-share/projects/020_crop1990/SwissCrop25/SwissCrop25.xlsx")
OUT_FILE = Path("/mnt/eo-nas1/eoa-share/projects/020_crop1990/SwissCrop25/results/tables/dataset_stats.txt")

YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
AREA_COLS = {y: c for y, c in zip(YEARS, [
    "2019_Area_m2", "2020_Area_m2", "2021_Area_m22",
    "2022_Area_m23", "2023_Area_m24", "2024_Area_m25", "2025_Area_m26"
])}
M2_TO_HA = 1e-4
M2_TO_KM2 = 1e-6
CH_AREA_KM2 = 41285
CLASS_LEVELS = ["LNF_code", "Crop_Label", "Crop_Label_lv1", "Crop_Label_lv2", "Crop_Label_lv3", "HCATv2_Identifier"]


def area_summary(df, label):
    lines = [f"\nCOVERAGE — {label}"]
    year_areas = {}
    for year, col in AREA_COLS.items():
        km2 = df[col].sum() * M2_TO_KM2
        year_areas[year] = km2
        lines.append(f"  {year}: {km2:>8,.0f} km²  ({km2 / CH_AREA_KM2 * 100:.1f}% of CH)")
    mean_km2 = sum(year_areas.values()) / len(year_areas)
    lines.append(f"  Mean: {mean_km2:>8,.0f} km²  ({mean_km2 / CH_AREA_KM2 * 100:.1f}% of CH)")
    return "\n".join(lines), mean_km2


def class_counts(df, label):
    lines = [f"\nCLASS COUNTS — {label}"]
    for level in CLASS_LEVELS:
        if level in df.columns:
            n = df[level].nunique()
            lines.append(f"  {level:<25}: {n}")
    return "\n".join(lines)


def hierarchy_counts(df_all, df_included):
    """Show pre- and post-exclusion hierarchy counts for paper macros."""
    ag_lv3 = ["Arable Land", "Permanent", "Grassland"]
    lines = ["\nHIERARCHY COUNTS (for paper macros)"]
    lines.append(f"  {'Level':<30} {'Pre-excl':>10} {'Post-excl':>10}")
    lines.append(f"  {'-'*52}")
    for col, label in [
        ("Crop_Label",      "Leaf labels  (nrLabelsRaw)"),
        ("Crop_Label_lv1",  "lv1 intermediate (nrHierInterm)"),
        ("Crop_Label_lv2",  "lv2 crop-type groups (nrHierGroups)"),
        ("Crop_Label_lv3",  "lv3 top-level (nrHierTop)"),
    ]:
        pre  = df_all[col].nunique()
        post = df_included[col].nunique()
        lines.append(f"  {label:<30} {pre:>10} {post:>10}")

    # Ag-only breakdown (post-exclusion, for reference)
    lines.append("\n  Post-exclusion lv2 breakdown:")
    ag_inc = df_included[df_included["Crop_Label_lv3"].isin(ag_lv3)]
    non_ag_inc = df_included[~df_included["Crop_Label_lv3"].isin(ag_lv3)]
    lines.append(f"    ag lv2  : {ag_inc['Crop_Label_lv2'].nunique()}")
    lines.append(f"    non-ag lv2: {non_ag_inc['Crop_Label_lv2'].nunique()}")
    lines.append(f"    total   : {df_included['Crop_Label_lv2'].nunique()}")
    return "\n".join(lines)


def main():
    df = pd.read_excel(LNF_FILE, sheet_name="label_sheet")
    included = df[df["Exclude"].isna()].copy()
    lnf_only = included[included["LNF_code"] < 1900].copy()
    tlm_only = included[included["LNF_code"] >= 1900].copy()

    lines = []
    ag_lv3 = ["Arable Land", "Permanent", "Grassland"]
    lnf_all = df[df["LNF_code"] < 1900]
    lnf_ag = lnf_all[lnf_all["Crop_Label_lv3"].isin(ag_lv3)]

    lines.append("=" * 60)
    lines.append("CODE COUNTS")
    lines.append(f"  Total rows in file:        {len(df)}")
    lines.append(f"  Excluded:                  {df['Exclude'].notna().sum()}")
    lines.append(f"  Included total:            {len(included)}")
    lines.append(f"    of which LNF (<1900):    {len(lnf_only)}")
    lines.append(f"    of which TLM3D (>=1900): {len(tlm_only)}")
    lines.append(f"")
    lines.append(f"  Agricultural LNF codes (lv3 = ag, nrLNFCodes): {len(lnf_ag)}")

    # Hierarchy counts (pre- and post-exclusion)
    lines.append(hierarchy_counts(df, included))

    # Class counts
    lines.append(class_counts(included, "LNF + TLM3D"))
    lines.append(class_counts(lnf_only, "LNF only"))
    lines.append(class_counts(tlm_only, "TLM3D only"))

    # Coverage
    s, _ = area_summary(included, "LNF + TLM3D")
    lines.append(s)
    s, mean_lnf = area_summary(lnf_only, "LNF only")
    lines.append(s)

    # Area per output class (avg across years, LNF + TLM3D)
    lines.append("\nAREA PER Crop_Label — avg across all years (ha), LNF + TLM3D")
    for col in AREA_COLS.values():
        included[col + "_ha"] = included[col] * M2_TO_HA
    ha_cols = [c + "_ha" for c in AREA_COLS.values()]
    included["avg_ha"] = included[ha_cols].mean(axis=1)
    by_class = included.groupby("Crop_Label")["avg_ha"].sum().sort_values(ascending=False)
    for cls, ha in by_class.items():
        lines.append(f"  {cls:<40} {ha:>10,.0f} ha")

    # Class imbalance ratio (ag leaf classes only)
    ag_inc = included[included["Crop_Label_lv3"].isin(ag_lv3)]
    ag_by_class = ag_inc.groupby("Crop_Label")["avg_ha"].sum().sort_values(ascending=False)
    max_cls, max_ha = ag_by_class.index[0], ag_by_class.iloc[0]
    min_cls, min_ha = ag_by_class.index[-1], ag_by_class.iloc[-1]
    ratio = max_ha / min_ha
    lines.append("\nCLASS IMBALANCE — ag leaf classes only (avg area across years)")
    lines.append(f"  Most frequent : {max_cls:<40} {max_ha:>10,.0f} ha")
    lines.append(f"  Least frequent: {min_cls:<40} {min_ha:>10,.0f} ha")
    lines.append(f"  Imbalance ratio (max/min): {ratio:,.0f}x")

    output = "\n".join(lines)
    print(output)
    OUT_FILE.write_text(output)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
