#!/usr/bin/env python3
"""
Compute two temporal baselines for the SwissCrop25 LOYO benchmark.

  1. previous-year  — predict each pixel's label using the previous year's GT
  2. majority-vote  — predict using majority across 3 non-test years from {2022-2025}

Evaluated on S1-S4 only (test years 2022-2025).  S5 (test=2021) is skipped:
2020 has incomplete Swiss coverage, and a 4-year pool gives ties.

Majority pool is always exactly 3 years (odd → no ties).

Memory / IO model: iterate over spatial keys once. For each tile, load all
needed years, compute every split's CM contribution in one pass, then discard.
Each year's tile data is read exactly once per tile — no redundant IO, no
full-year arrays in RAM.

Outputs per split (skipped if already present → safe to rerun):
  storage/prev_year_baseline_S{N}/conf_mat.pkl        (71×71 int64)
  storage/prev_year_baseline_S{N}/test_metrics.json
  storage/majority_baseline_S{N}/conf_mat.pkl
  storage/majority_baseline_S{N}/test_metrics.json
"""

import sys
import re
import json
import pickle
import threading
import numpy as np
import pandas as pd
import fsspec
import zarr
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from ag_classes import get_ag_indices, crop_metrics_from_cm

ROOT        = Path(__file__).parents[2]
GT_ROOT     = Path("/capstor/scratch/cscs/tlauber/020_crop1990/data/GTs_Sentinel")
STORAGE     = ROOT / "storage"
XLSX        = ROOT / "SwissCrop25.xlsx"
NUM_CLASSES = 71
NUM_WORKERS = 16

SPLITS = {
    "S5": {"test": 2025, "prev": 2024, "majority": [2022, 2023, 2024]},
    "S4": {"test": 2024, "prev": 2023, "majority": [2022, 2023, 2025]},
    "S3": {"test": 2023, "prev": 2022, "majority": [2022, 2024, 2025]},
    "S2": {"test": 2022, "prev": 2021, "majority": [2023, 2024, 2025]},
}


def build_label_lookup(xlsx_path):
    """Return {lnf_code_int: target_class_int} — same mapping as dataset's target_mapping[0]."""
    df = pd.read_excel(xlsx_path, sheet_name="label_sheet")
    df = df[df["Exclude"] != True]
    df = df[df["Crop_Label"].notna()].reset_index(drop=True)
    class_mapping = {name: i + 1 for i, name in enumerate(df["Crop_Label"].unique())}
    target_mapping = {}
    for _, row in df.iterrows():
        lnf_code = int(row["LNF_code"])
        target_mapping[lnf_code] = class_mapping.get(row["Crop_Label"], 0)
    return target_mapping


def open_year(year):
    """
    Open zarr reference filesystem for `year`.
    Returns (root, key_to_name) where key_to_name maps
      spatial_key → full tile name (strips trailing _YYYYMMDD_YYYYMMDD).
    Only reference metadata is loaded; tile arrays stay on disk.
    """
    path = str(GT_ROOT / f"{year}.tar.json")
    tar_path = path[:-5]
    fs = fsspec.filesystem("reference", fo=path, remote_options={"fo": tar_path})
    mapper = fs.get_mapper("")
    root = zarr.open_group(mapper, mode="r")
    key_to_name = {}
    for tile_name in root.keys():
        spatial_key = re.sub(r'_\d{8}_\d{8}(\.zarr)?$', '', tile_name)
        key_to_name[spatial_key] = tile_name
    print(f"  {year}: {len(key_to_name)} tiles", flush=True)
    return root, key_to_name


def load_tile(root, tile_name, mapping):
    """
    Load one tile and return a (H, W) int16 label array.
"""
    tile       = root[tile_name]
    data_arr   = tile["lnf_code"][:]              # (num_bands, H, W) float32
    lnf_codes  = tile["band"][:].astype(int)      # (num_bands,) LNF codes

    H, W = data_arr.shape[1], data_arr.shape[2]
    aggregated = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)

    for band_idx, lnf_code in enumerate(lnf_codes):
        target_class = mapping.get(int(lnf_code), 0)
        aggregated[target_class] += data_arr[band_idx]

    return np.argmax(aggregated, axis=0).astype(np.int16)


def majority_vote(arrays):
    stack = np.stack(arrays, axis=0)            # (N, H, W)
    n_years, H, W = stack.shape
    n_pixels = H * W
    flat = stack.reshape(n_years, n_pixels).astype(np.int32)
    counts = np.zeros((NUM_CLASSES, n_pixels), dtype=np.int32)
    col_idx = np.arange(n_pixels)
    for i in range(n_years):
        counts[flat[i], col_idx] += 1
    return np.argmax(counts, axis=0).reshape(H, W).astype(np.int16)


def compute_metrics(cm, ag_idx):
    oa, giou, miou, mf1 = crop_metrics_from_cm(cm, ag_idx)
    print(f"    OA={oa:.2f}%  GIoU={giou:.2f}%  mIoU={miou:.2f}%  mF1={mf1:.2f}%",
          flush=True)
    return {
        "test_accuracy":    round(oa,   4),
        "test_global_IoU":  round(giou, 4),
        "test_IoU":         round(miou, 4),
        "test_F1":          round(mf1,  4),
    }


def save_results(folder_name, cm, metrics):
    out = STORAGE / folder_name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "conf_mat.pkl", "wb") as f:
        pickle.dump(cm, f)
    with open(out / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    Saved → {out}", flush=True)


def already_done(folder_name):
    out = STORAGE / folder_name
    return (out / "conf_mat.pkl").exists() and (out / "test_metrics.json").exists()


def main():
    print("Building label mapping...")
    mapping = build_label_lookup(XLSX)
    ag_idx, _ = get_ag_indices()
    print(f"  {len(ag_idx)} agricultural classes, {len(mapping)} LNF codes mapped\n")

    # Determine which (split, baseline) pairs still need computing
    needed = {}  # split_name → {"need_prev": bool, "need_maj": bool, **cfg}
    for split_name, cfg in SPLITS.items():
        need_prev = not already_done(f"prev_year_baseline_{split_name}")
        need_maj  = not already_done(f"majority_baseline_{split_name}")
        if need_prev or need_maj:
            needed[split_name] = {**cfg, "need_prev": need_prev, "need_maj": need_maj}

    if not needed:
        print("All splits already done.")
        return

    # Collect exactly the years we actually need
    years_needed = set()
    for split_name, cfg in needed.items():
        years_needed.add(cfg["test"])
        if cfg["need_prev"]:
            years_needed.add(cfg["prev"])
        if cfg["need_maj"]:
            years_needed.update(cfg["majority"])

    print(f"Opening zarr for years: {sorted(years_needed)}")
    handles = {}  # year → (root, key_to_name)
    for year in sorted(years_needed):
        handles[year] = open_year(year)

    # Union of all spatial keys across all open years
    all_keys = set()
    for root, key_to_name in handles.values():
        all_keys |= set(key_to_name)
    print(f"\n{len(all_keys)} unique spatial keys across all years")
    print(f"Processing {len(needed)} split(s): {list(needed)}\n")

    # Shared CMs — one per (split, baseline) that needs computing
    cms_prev = {s: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                for s, cfg in needed.items() if cfg["need_prev"]}
    cms_maj  = {s: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                for s, cfg in needed.items() if cfg["need_maj"]}
    cms_lock = threading.Lock()
    error_count = [0]
    error_lock  = threading.Lock()
    MAX_ERRORS  = 100  # abort if this many tile reads fail (avoids silent empty CM)

    def process_tile(spatial_key):
        # Load this tile for every year that has it — exactly one read per year per tile
        year_arrays = {}
        for year, (root, key_to_name) in handles.items():
            if spatial_key not in key_to_name:
                continue
            try:
                year_arrays[year] = load_tile(root, key_to_name[spatial_key], mapping)
            except Exception as e:
                print(f"\n    Warning: skipped {year}/{spatial_key}: {e}", flush=True)
                with error_lock:
                    error_count[0] += 1
                    if error_count[0] >= MAX_ERRORS:
                        raise RuntimeError(
                            f"Aborting: {error_count[0]} tile read errors — "
                            "check zarr paths and fsspec reference files."
                        )

        # Accumulate into local CMs (avoid lock contention per-pixel)
        local_prev = {s: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                      for s in cms_prev}
        local_maj  = {s: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                      for s in cms_maj}

        for split_name, cfg in needed.items():
            test_year = cfg["test"]
            if test_year not in year_arrays:
                continue
            t = year_arrays[test_year].ravel().astype(np.int32)

            if split_name in local_prev:
                prev_year = cfg["prev"]
                if prev_year in year_arrays:
                    p = year_arrays[prev_year].ravel().astype(np.int32)
                    np.add.at(local_prev[split_name], (t, p), 1)

            if split_name in local_maj:
                maj_years = cfg["majority"]
                if all(y in year_arrays for y in maj_years):
                    pred = majority_vote([year_arrays[y] for y in maj_years])
                    p = pred.ravel().astype(np.int32)
                    np.add.at(local_maj[split_name], (t, p), 1)

        with cms_lock:
            for s in local_prev:
                cms_prev[s] += local_prev[s]
            for s in local_maj:
                cms_maj[s] += local_maj[s]

        del year_arrays, local_prev, local_maj

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = [ex.submit(process_tile, k) for k in sorted(all_keys)]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="tiles", unit="tile"):
            pass

    print("\nSaving results...")
    for split_name in needed:
        if split_name in cms_prev:
            print(f"\n  prev_year_baseline_{split_name}")
            metrics = compute_metrics(cms_prev[split_name], ag_idx)
            save_results(f"prev_year_baseline_{split_name}", cms_prev[split_name], metrics)
        if split_name in cms_maj:
            print(f"\n  majority_baseline_{split_name}")
            metrics = compute_metrics(cms_maj[split_name], ag_idx)
            save_results(f"majority_baseline_{split_name}", cms_maj[split_name], metrics)

    print("\nAll baselines complete.")


if __name__ == "__main__":
    main()
