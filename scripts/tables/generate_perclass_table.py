# generate_perclass_table.py
# Per-class IoU table for SwissCrop25 supplementary material.
#
# Reads results/tables/perclass_metrics.csv and produces a longtable body with:
#   - Full 4-level hierarchy: lv3 > lv2 > lv1 (where multi-leaf) > leaf
#   - Bold aggregate row at end of each group (lv1/lv2/lv3)
#   - lv1 aggregate rows only shown when lv1 contains >1 leaf class
#   - Summary block at the bottom: mIoU / mF1 at lv3, lv2, lv1, leaf levels
#
# Sorting: within each lv2, lv1 groups are ordered by lv1 aggregate mIoU
# (descending); within each lv1, leaves are ordered by leaf mean IoU
# (descending).  This keeps each lv1 group contiguous.
#
# Usage:
#   python scripts/tables/generate_perclass_table.py

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).parents[2]
CSV_PATH = ROOT / "results/tables/perclass_metrics.csv"

sys.path.insert(0, str(ROOT / "scripts/analysis"))
from eval_utils import get_ag_indices, crop_metrics_from_cm

MODELS = ["UTAE", "TSViT", "Galileo-nano"]

SPLITS = ["S5", "S4", "S3", "S2", "S1"]

MODEL_DIRS = {
    "UTAE":         "utae_gddsub_gddpe",
    "TSViT":        "tsvit_gddsub_gddpe",
    "Galileo-nano": "galileo_nano_gddsub",
}

STORAGE = ROOT / "storage"

LV3_ORDER = ["Arable Land", "Permanent", "Grassland"]

# lv2 display order, grouped by lv3 (agronomically coherent)
LV2_ORDER = [
    # Arable Land
    "Winter Cereals", "Spring Cereals", "Summer Cereals",
    "Oilseeds", "Root Crops", "Pulses",
    "Annual Horticulture", "Temporary Grassland", "Rice",
    # Permanent
    "Perennial Horticulture", "Orchard", "Nursery", "Greenhouses", "Woody Crops",
    # Grassland
    "Meadow", "Pasture", "Alpine Pasture",
]

LEVEL_LABELS = {
    "lv3":  "lv3 (3 groups)",
    "lv2":  "lv2 (16 groups)",
    "lv1":  "lv1 (49 groups)",
    "leaf": "leaf (65 classes)",
}

OUTPUTS = [
    ROOT / "results/tables/tab_perclass_body.tex",
    ROOT / "Lauber_2026_SwissCrop_ECCV26_TerraBytes/tables/tab_perclass_body.tex",
]


def pivot_wide(df_level, index_col):
    wide = df_level.pivot_table(
        index=index_col, columns="model", values=["IoU", "F1"], aggfunc="first"
    ).reset_index()
    wide.columns = [index_col] + [
        f"{metric}_{model}" for metric, model in wide.columns[1:]
    ]
    wide["mean_IoU"] = np.mean([wide[f"IoU_{m}"] for m in MODELS], axis=0)
    wide["best_model"] = (
        wide[[f"IoU_{m}" for m in MODELS]].idxmax(axis=1).str.replace("IoU_", "")
    )
    return wide


def load_split_miou():
    """Compute per-split mean mIoU for each model (matching the main results table)."""
    import pickle
    ag_idx, _ = get_ag_indices()
    result = {}
    for model_name, dir_prefix in MODEL_DIRS.items():
        per_split = []
        for split in SPLITS:
            folder = STORAGE / f"{dir_prefix}_{split}"
            for fname in ("conf_mat.pkl", "test_metrics.pkl"):
                p = folder / fname
                if p.exists():
                    with open(p, "rb") as f:
                        cm = pickle.load(f).astype(float)
                    _, _, miou, _ = crop_metrics_from_cm(cm, ag_idx)
                    per_split.append(miou)
                    break
        result[model_name] = float(np.mean(per_split)) if per_split else None
    return result


def load_data():
    df = pd.read_csv(CSV_PATH)

    # Leaf level (ag only), with lv1/lv2/lv3 for grouping
    leaf_raw = df[(df["level"] == "leaf") & (df["is_ag"] == True)].copy()
    leaf = leaf_raw.pivot_table(
        index=["group", "lv1", "lv2", "lv3"], columns="model",
        values=["IoU", "F1"], aggfunc="first"
    ).reset_index()
    leaf.columns = ["group", "lv1", "lv2", "lv3"] + [
        f"{metric}_{model}" for metric, model in leaf.columns[4:]
    ]
    leaf["mean_IoU"] = np.mean([leaf[f"IoU_{m}"] for m in MODELS], axis=0)
    leaf["best_model"] = (
        leaf[[f"IoU_{m}" for m in MODELS]].idxmax(axis=1).str.replace("IoU_", "")
    )

    # Aggregated levels
    agg = {}
    for level in ["lv2", "lv1", "lv3"]:
        sub = df[(df["level"] == level) & (df["is_ag"] == True)].copy()
        agg[level] = pivot_wide(sub, "group")

    # Leaf-level means (already ag-only)
    leaf_means = {
        m: leaf_raw[leaf_raw["model"] == m]["IoU"].mean()
        for m in MODELS
    }

    return leaf, agg, leaf_means


def fmt_cell(val, bold=False):
    s = f"{val:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def header_row(name, agg_wide, indent="", bold=True, italic=False):
    """Return a LaTeX row showing the group name + its aggregate IoU values."""
    row = agg_wide[agg_wide["group"] == name]
    if row.empty:
        return None
    row = row.iloc[0]
    label = name
    if bold and italic:
        label = f"\\textit{{\\textbf{{{name}}}}}"
    elif bold:
        label = f"\\textbf{{{name}}}"
    elif italic:
        label = f"\\textit{{{name}}}"
    cells = [f"{indent}{label}"]
    for m in MODELS:
        best = row["best_model"] == m
        cells.append(fmt_cell(row[f"IoU_{m}"], best))
    return " & ".join(cells) + r" \\"


def make_body(leaf, agg, leaf_means, split_miou):
    def lv2_key(lv2):
        try:
            return LV2_ORDER.index(lv2)
        except ValueError:
            return len(LV2_ORDER)

    def lv3_key(lv3):
        try:
            return LV3_ORDER.index(lv3)
        except ValueError:
            return len(LV3_ORDER)

    leaf = leaf.copy()
    leaf["lv2_order"] = leaf["lv2"].apply(lv2_key)
    leaf["lv3_order"] = leaf["lv3"].apply(lv3_key)

    # Attach lv1 aggregate mean IoU so we can sort lv1 groups within each lv2.
    lv1_agg_iou = (
        agg["lv1"][["group", "mean_IoU"]]
        .rename(columns={"group": "lv1", "mean_IoU": "lv1_agg_IoU"})
    )
    leaf = leaf.merge(lv1_agg_iou, on="lv1", how="left")
    leaf["lv1_agg_IoU"] = leaf["lv1_agg_IoU"].fillna(leaf["mean_IoU"])

    leaf = leaf.sort_values(
        ["lv3_order", "lv2_order", "lv1_agg_IoU", "mean_IoU"],
        ascending=[True, True, False, False]
    )

    # lv1 groups that contain >1 leaf — these get their own header row + values
    lv1_counts = leaf.groupby(["lv2", "lv1"])["group"].count().reset_index()
    multi_lv1 = set(
        zip(lv1_counts[lv1_counts["group"] > 1]["lv2"],
            lv1_counts[lv1_counts["group"] > 1]["lv1"])
    )

    lines = ["% auto-generated by generate_perclass_table.py — do not edit manually"]

    current_lv3 = None
    current_lv2 = None
    current_lv1 = None

    for _, row in leaf.iterrows():
        lv3 = row["lv3"]
        lv2 = row["lv2"]
        lv1 = row["lv1"]
        # Suppress inner headers when names collapse (e.g. Greenhouses lv3=lv2=lv1)
        skip_lv2_header = (lv2 == lv3)
        skip_lv1_header = (lv1 == lv2)
        is_multi_lv1 = (lv2, lv1) in multi_lv1

        # ── lv3 group change ─────────────────────────────────────────────────
        if lv3 != current_lv3:
            if current_lv3 is not None:
                lines.append("\\addlinespace[6pt]")
            r = header_row(lv3, agg["lv3"], indent="", bold=True, italic=True)
            if r:
                lines.append(r)
            lines.append("\\midrule")
            current_lv3 = lv3
            current_lv2 = None
            current_lv1 = None

        # ── lv2 group change ─────────────────────────────────────────────────
        if lv2 != current_lv2:
            if current_lv2 is not None:
                lines.append("\\addlinespace[3pt]")
            if not skip_lv2_header:
                r = header_row(lv2, agg["lv2"], indent="~~", bold=True)
                if r:
                    lines.append(r)
                lines.append("\\midrule")
            current_lv2 = lv2
            current_lv1 = None

        # ── lv1 group change (only when multi-leaf) ──────────────────────────
        if lv1 != current_lv1:
            if current_lv1 is not None and (current_lv2, current_lv1) in multi_lv1:
                lines.append("\\addlinespace[2pt]")
            if is_multi_lv1 and not skip_lv1_header:
                indent = "~~" if skip_lv2_header else "~~~~"
                r = header_row(lv1, agg["lv1"], indent=indent, bold=False, italic=True)
                if r:
                    lines.append(r)
            current_lv1 = lv1

        # ── Leaf class row ───────────────────────────────────────────────────
        if skip_lv2_header:
            leaf_indent = "~~~~"
        elif is_multi_lv1 and not skip_lv1_header:
            leaf_indent = "~~~~~~~~"
        else:
            leaf_indent = "~~~~~~"

        cells = [f"{leaf_indent}{row['group']}"]
        for m in MODELS:
            bold = row["best_model"] == m
            cells.append(fmt_cell(row[f"IoU_{m}"], bold))
        lines.append(" & ".join(cells) + r" \\")

    # ── Summary block: mIoU / mF1 at each taxonomy level ─────────────────────
    lines.append("\\addlinespace[6pt]")
    lines.append("\\midrule")
    lines.append("\\midrule")
    lines.append(
        "\\multicolumn{4}{l}{\\textit{Mean IoU by taxonomy level}} \\\\"
    )
    lines.append("\\midrule")

    for level in ["lv3", "lv2", "lv1", "leaf"]:
        label = LEVEL_LABELS[level]
        cells = [f"\\textbf{{{label}}}"]

        if level == "leaf":
            # Use per-split averaged mIoU (matching the main results table exactly).
            iou_vals = [split_miou[m] for m in MODELS]
        else:
            iou_vals = [agg[level][f"IoU_{m}"].mean() for m in MODELS]

        best_idx = int(np.argmax(iou_vals))
        for i, iou in enumerate(iou_vals):
            cells.append(fmt_cell(iou, i == best_idx))

        lines.append(" & ".join(cells) + r" \\")

    if lines[-1].endswith(r" \\"):
        lines[-1] = lines[-1][:-3]
    return "\n".join(lines)


def main():
    leaf, agg, leaf_means = load_data()
    split_miou = load_split_miou()
    body = make_body(leaf, agg, leaf_means, split_miou)

    for path in OUTPUTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        print(f"Written: {path}")


if __name__ == "__main__":
    main()
