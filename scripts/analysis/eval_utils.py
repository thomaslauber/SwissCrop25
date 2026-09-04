# ag_classes.py
# Shared helper: derive agricultural vs non-agricultural class indices from SwissCrop25.xlsx.
# Returns 1-based class indices matching the conf_mat rows/cols (background=0 excluded).

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).parents[2]
LABEL_SHEET = ROOT / "SwissCrop25.xlsx"

# The 5 landscape classes added from swissTLM3D (not LNF agricultural codes).
# All other non-excluded Crop_Label values are agricultural classes (65 total).
LANDSCAPE_LABELS = {"Forest", "Water", "Built-up", "Unproductive Area", "Wetland"}


def get_ag_indices():
    """
    Returns (ag_idx, nag_idx) — 1-based class indices (matching conf_mat before bg strip).
    Agricultural: all 65 non-excluded LNF-derived classes (incl. Christmas Trees, Chestnut, Hedges).
    Non-agricultural: the 5 swissTLM3D landscape classes only.
    Background (class 0) is not included in either list.
    """
    df = pd.read_excel(LABEL_SHEET, sheet_name="label_sheet")
    df = df[df["Exclude"] != True]
    df = df[df["Crop_Label"].notna()].reset_index(drop=True)

    unique_labels = list(df["Crop_Label"].unique())
    mapping = {name: i + 1 for i, name in enumerate(unique_labels)}

    ag_idx  = sorted(idx for name, idx in mapping.items() if name not in LANDSCAPE_LABELS)
    nag_idx = sorted(idx for name, idx in mapping.items() if name in LANDSCAPE_LABELS)
    return ag_idx, nag_idx


def restrict_to_ag(cm, ag_idx):
    """
    Given a full conf_mat (with background at row/col 0),
    return submatrix restricted to agricultural classes only.
    ag_idx: 1-based indices.
    """
    return cm[np.ix_(ag_idx, ag_idx)]


def crop_metrics_from_cm(cm, ag_idx):
    """
    Compute OA, GIoU, mIoU, mF1 all restricted to the ag-class submatrix.
    cm: full confusion matrix, shape (71, 71), background at index 0.
    ag_idx: 1-based agricultural class indices.
    Returns (oa, giou, miou, mf1) as percentages.
    """
    eps = 1e-10
    cm_ag = restrict_to_ag(cm, ag_idx)
    tp = np.diag(cm_ag)
    fp = cm_ag.sum(axis=0) - tp
    fn = cm_ag.sum(axis=1) - tp
    has_gt = cm_ag.sum(axis=1) > 0

    iou = tp / (tp + fp + fn + eps)
    f1  = 2 * tp / (2 * tp + fp + fn + eps)
    miou = float(iou[has_gt].mean() * 100)
    mf1  = float(f1[has_gt].mean() * 100)

    total_tp = float(tp.sum())
    total_px = float(cm_ag.sum())
    oa   = total_tp / (total_px + eps) * 100
    giou = total_tp / (2 * total_px - total_tp + eps) * 100

    return oa, giou, miou, mf1


def load_cm(storage_dir, folder, month=None):
    """
    Load a confusion matrix pkl from storage_dir/folder.
    Priority: conf_mat.pkl > test_metrics.pkl (or monthly variants).
    month: int or None. If set, looks for test_metrics_month{month}.pkl.
    Raises FileNotFoundError if no pkl exists.
    All pkl files are 71x71 (background at index 0).
    """
    from pathlib import Path
    import pickle
    d = Path(storage_dir) / folder
    if month is not None:
        candidates = [d / f"test_metrics_month{month}.pkl"]
    else:
        candidates = [d / "conf_mat.pkl", d / "test_metrics.pkl"]
    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f).astype(np.float64)
    raise FileNotFoundError(f"No confusion matrix pkl found in {d}" +
                            (f" for month {month}" if month else ""))
