#!/usr/bin/env python3
"""
Per-class and hierarchical taxonomy F1 / IoU analysis.

Loads conf_mat.pkl for each model × split, averages across splits, then reports:
  - Per leaf class  (Crop_Label)      F1 and IoU
  - Per lv1 group   (Crop_Label_lv1)  F1 and IoU  (aggregated from raw counts)
  - Per lv2 group   (Crop_Label_lv2)  F1 and IoU
  - Per lv3 group   (Crop_Label_lv3)  F1 and IoU

Aggregation is done correctly at the confusion-matrix level (sum TP/FP/FN, then
compute metrics) rather than averaging per-class scores.

Outputs:
  - results/tables/perclass_metrics.csv   — full per-class table, all models
  - stdout summary tables

Usage:
  python analyze_taxonomy.py
  python analyze_taxonomy.py --storage ./storage --label-sheet ./SwissCrop25.xlsx
"""

import argparse
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from eval_utils import get_ag_indices, restrict_to_ag

SPLITS = ["S5", "S4", "S3", "S2", "S1"]

MODELS = {
    "UTAE":         "utae_gddsub_gddpe",
    "TSViT":        "tsvit_gddsub_gddpe",
    "Galileo-nano": "galileo_nano_gddsub",
}

TAXONOMY_LEVELS = [
    ("Crop_Label",     "leaf"),
    ("Crop_Label_lv1", "lv1"),
    ("Crop_Label_lv2", "lv2"),
    ("Crop_Label_lv3", "lv3"),
]


def build_class_mapping(label_sheet_path):
    """
    Replicate the dataset's class-index assignment:
    - filter Exclude rows
    - assign index i+1 in order of first appearance (.unique()) for Crop_Label
    Returns:
      idx_to_label : list of length 70, idx_to_label[i] = name for class index i+1
      label_to_idx : dict name -> 1-based index
      hierarchy    : dict Crop_Label -> {lv1, lv2, lv3}
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sheet = pd.read_excel(label_sheet_path, sheet_name="label_sheet")
    sheet = sheet[sheet["Exclude"] != True].copy()

    unique_labels = list(sheet["Crop_Label"].unique())
    label_to_idx = {name: i + 1 for i, name in enumerate(unique_labels)}
    idx_to_label = unique_labels  # 0-based list, maps to 1-based indices

    # Build hierarchy: one row per unique Crop_Label (take first occurrence)
    hier_cols = ["Crop_Label", "Crop_Label_lv1", "Crop_Label_lv2", "Crop_Label_lv3"]
    hier_df = sheet[hier_cols].drop_duplicates(subset="Crop_Label").set_index("Crop_Label")
    hierarchy = hier_df.to_dict(orient="index")

    return idx_to_label, label_to_idx, hierarchy


def load_conf_mat(storage_dir, model_prefix, split):
    folder = os.path.join(storage_dir, f"{model_prefix}_{split}")
    for name in ("conf_mat.pkl", "test_metrics.pkl"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f).astype(np.float64)
    return None


def per_class_metrics(cm_valid):
    """
    Given a confusion matrix with ignore class removed (shape [n_cls, n_cls]),
    return arrays of IoU and F1 per class (length n_cls).
    """
    TP = np.diag(cm_valid)
    FP = cm_valid.sum(axis=0) - TP
    FN = cm_valid.sum(axis=1) - TP
    iou = TP / (TP + FP + FN + 1e-10)
    f1_denom = 2 * TP + FP + FN
    f1 = np.where(f1_denom > 0, 2 * TP / f1_denom, 0.0)
    return iou, f1


def aggregate_to_level(cm_valid, idx_to_label, hierarchy, level_col, ag_mask):
    """
    Collapse the fine-grained confusion matrix to a coarser level by summing
    rows and columns that belong to the same group.

    Only agricultural classes (ag_mask[i]=True) are included; landscape classes
    are excluded from both the coarse CM construction and the group means.

    Returns:
      group_names : list of group names (sorted, ag-only groups)
      iou         : array of IoU per group
      f1          : array of F1 per group
    """
    n = len(idx_to_label)
    fine_to_group = []
    for i, label in enumerate(idx_to_label):
        if not ag_mask[i]:
            fine_to_group.append(None)  # exclude landscape from coarse CM
        else:
            grp = hierarchy.get(label, {}).get(level_col, None)
            fine_to_group.append(grp)

    group_names = sorted(set(g for g in fine_to_group if g is not None))
    group_to_coarse_idx = {g: i for i, g in enumerate(group_names)}
    n_groups = len(group_names)

    coarse_cm = np.zeros((n_groups, n_groups), dtype=np.float64)
    for fi in range(n):
        gi = fine_to_group[fi]
        if gi is None:
            continue
        ci = group_to_coarse_idx[gi]
        for fj in range(n):
            gj = fine_to_group[fj]
            if gj is None:
                continue
            cj = group_to_coarse_idx[gj]
            coarse_cm[ci, cj] += cm_valid[fi, fj]

    iou, f1 = per_class_metrics(coarse_cm)
    return group_names, iou, f1


def print_level_table(model_name, group_names, iou_arr, f1_arr, title):
    miou = iou_arr.mean() * 100
    mf1  = f1_arr.mean() * 100
    print(f"\n  [{model_name}]  {title}   mIoU={miou:.1f}%  mF1={mf1:.1f}%")
    print(f"  {'Class':<30}  {'IoU':>7}  {'F1':>7}")
    print("  " + "-" * 48)
    order = np.argsort(iou_arr)[::-1]
    for i in order:
        print(f"  {group_names[i]:<30}  {iou_arr[i]*100:7.1f}  {f1_arr[i]*100:7.1f}")
    print(f"  {'MEAN':<30}  {miou:7.1f}  {mf1:7.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage",     default="./storage")
    parser.add_argument("--label-sheet", default="./SwissCrop25.xlsx")
    parser.add_argument("--out-csv",     default="./results/tables/perclass_metrics.csv")
    parser.add_argument("--levels",      nargs="+",
                        default=["leaf", "lv1", "lv2", "lv3"],
                        help="Which taxonomy levels to report")
    args = parser.parse_args()

    idx_to_label, label_to_idx, hierarchy = build_class_mapping(args.label_sheet)

    # Agricultural mask: True for the 65 ag classes, False for the 5 landscape classes.
    # ag_idx from ag_classes is 1-based; cm_valid is 0-based (class i → cm_valid row i-1).
    ag_idx_1based, _ = get_ag_indices()
    ag_idx_0based = set(i - 1 for i in ag_idx_1based)
    ag_mask = np.array([i in ag_idx_0based for i in range(len(idx_to_label))])
    ag_labels_ordered = [l for i, l in enumerate(idx_to_label) if ag_mask[i]]

    all_rows = []  # for CSV output

    for model_name, dir_prefix in MODELS.items():
        print("\n" + "=" * 70)
        print(f"  MODEL: {model_name}")
        print("=" * 70)

        # ── Load confusion matrices per split ─────────────────────────────────
        cms = []
        for split in SPLITS:
            cm = load_conf_mat(args.storage, dir_prefix, split)
            if cm is not None:
                cms.append(cm)

        if not cms:
            print(f"  No conf_mat.pkl found for {model_name}, skipping.")
            continue

        # Per-split averaging (consistent with main results table methodology):
        # compute metrics independently for each split, then average across splits.
        # Classes absent from a split (has_gt=False) contribute NaN for that split
        # and are excluded from that split's average via nanmean.

        # ── Leaf level (agricultural classes only) ────────────────────────────
        # Use ag-restricted CM (65×65) to match crop_metrics_from_cm in the main
        # results table, which excludes non-crop FP/FN from the IoU calculation.
        if "leaf" in args.levels:
            per_split_iou = []
            per_split_f1  = []
            for cm in cms:
                cm_ag = restrict_to_ag(cm, ag_idx_1based)  # 65×65
                iou_s, f1_s = per_class_metrics(cm_ag)
                has_gt = cm_ag.sum(axis=1) > 0
                iou_s = np.where(has_gt, iou_s, np.nan)
                f1_s  = np.where(has_gt, f1_s,  np.nan)
                per_split_iou.append(iou_s)
                per_split_f1.append(f1_s)
            ag_iou = np.nanmean(per_split_iou, axis=0)  # 65 values
            ag_f1  = np.nanmean(per_split_f1,  axis=0)

            print_level_table(model_name, ag_labels_ordered, ag_iou, ag_f1,
                              "LEAF — crop classes only (Crop_Label, 65 ag)")
            for j, label in enumerate(ag_labels_ordered):
                hier = hierarchy.get(label, {})
                all_rows.append({
                    "model":        model_name,
                    "level":        "leaf",
                    "group":        label,
                    "is_ag":        True,
                    "lv1":          hier.get("Crop_Label_lv1"),
                    "lv2":          hier.get("Crop_Label_lv2"),
                    "lv3":          hier.get("Crop_Label_lv3"),
                    "IoU":          round(float(ag_iou[j]) * 100, 2),
                    "F1":           round(float(ag_f1[j]) * 100, 2),
                    "n_splits":     len(cms),
                })

        # ── Hierarchical levels (ag classes only) ─────────────────────────────
        for col, level_tag in [
            ("Crop_Label_lv1", "lv1"),
            ("Crop_Label_lv2", "lv2"),
            ("Crop_Label_lv3", "lv3"),
        ]:
            if level_tag not in args.levels:
                continue
            per_split_iou_grp = []
            per_split_f1_grp  = []
            ref_grp_names = None
            for cm in cms:
                cm_valid = cm[1:, 1:]
                grp_names_s, iou_grp_s, f1_grp_s = aggregate_to_level(
                    cm_valid, idx_to_label, hierarchy, col, ag_mask
                )
                if ref_grp_names is None:
                    ref_grp_names = grp_names_s
                per_split_iou_grp.append(iou_grp_s)
                per_split_f1_grp.append(f1_grp_s)
            grp_names = ref_grp_names
            iou_grp = np.nanmean(per_split_iou_grp, axis=0)
            f1_grp  = np.nanmean(per_split_f1_grp,  axis=0)

            print_level_table(model_name, grp_names, iou_grp, f1_grp,
                              f"{level_tag.upper()} ({col}) — ag only")
            for i, grp in enumerate(grp_names):
                all_rows.append({
                    "model":    model_name,
                    "level":    level_tag,
                    "group":    grp,
                    "is_ag":    True,
                    "lv1":      grp if level_tag == "lv1" else None,
                    "lv2":      grp if level_tag == "lv2" else None,
                    "lv3":      grp if level_tag == "lv3" else None,
                    "IoU":      round(float(iou_grp[i]) * 100, 2),
                    "F1":       round(float(f1_grp[i]) * 100, 2),
                    "n_splits": len(cms),
                })

    # ── CSV output ────────────────────────────────────────────────────────────
    if all_rows:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df = pd.DataFrame(all_rows)
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved: {args.out_csv}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
