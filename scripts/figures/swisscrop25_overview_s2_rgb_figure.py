#!/usr/bin/env python3
"""
Generate 6 RGB Sentinel-2 images for tile 16165 (379100_5224620), year 2022.

Used as a panel in the SwissCrop25 dataset overview figure.

Picks 6 low-cloud timesteps spread across the calendar year.
SCL classes 4 (vegetation), 5 (bare soil), 6 (water), 7 (unclassified) = valid;
clouds = SCL 8,9,10,11. Cloud fraction per scene used for selection.

Usage:
  python scripts/figures/swisscrop25_overview_s2_rgb_figure.py
"""

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

ZARR_PATH = Path("/mnt/eo-nas1/data/satellite/sentinel2/raw/CH/S2_379100_5224620_20220104_20221230.zarr")
ROOT      = Path(__file__).parents[2]
OUT_DIR   = ROOT / "results/figures/SwissCrop_overview"

# SCL cloud classes
CLOUD_SCL = {8, 9, 10, 11}

# Stretch percentiles for RGB
P_LOW, P_HIGH = 2, 98


def load_zarr():
    return zarr.open(str(ZARR_PATH), mode="r")


def cloud_fraction(scl):
    """Fraction of pixels with cloud SCL values."""
    cloud_mask = np.isin(scl, list(CLOUD_SCL))
    return cloud_mask.mean()


def pick_timesteps(times, scl_stack, n=6):
    """
    Pick n low-cloud timesteps spread evenly across the year.
    Divides the year into n windows, picks the clearest scene in each.
    """
    n_times = len(times)
    window_size = n_times / n
    selected = []
    for i in range(n):
        start = int(i * window_size)
        end   = int((i + 1) * window_size)
        window_times = range(start, end)
        best_t = min(window_times, key=lambda t: cloud_fraction(scl_stack[t]))
        cf = cloud_fraction(scl_stack[best_t])
        selected.append((best_t, cf))
        print(f"  Window {i+1}: t={best_t}, cloud={cf:.1%}")
    return selected


def compute_global_stretch(b04, b03, b02, timesteps):
    """Compute global lo/hi percentiles across all selected scenes."""
    all_pixels = np.concatenate([
        np.stack([b04[t], b03[t], b02[t]], axis=-1).ravel()
        for t, _ in timesteps
    ])
    lo = np.percentile(all_pixels, P_LOW)
    hi = np.percentile(all_pixels, P_HIGH)
    return lo, hi


def to_rgb(r, g, b, lo, hi):
    """Stack and stretch bands to uint8 RGB using pre-computed global limits."""
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading zarr...")
    zz = load_zarr()
    times   = zz["time"][:]          # ms since epoch
    scl     = zz["s2_SCL"][:]        # (T, 128, 128)
    b04     = zz["s2_B04"][:]        # Red
    b03     = zz["s2_B03"][:]        # Green
    b02     = zz["s2_B02"][:]        # Blue

    uris    = zz["product_uri"][:]
    # Parse acquisition date from product_uri: S2X_MSIL2A_YYYYMMDDTHHMMSS_...
    dates = [datetime.strptime(u.split("_")[2][:8], "%Y%m%d") for u in uris]

    print(f"Timesteps: {len(times)}, date range: {dates[0].date()} – {dates[-1].date()}")
    print("Selecting 6 low-cloud scenes spread across year...")
    selected = pick_timesteps(times, scl, n=6)

    lo, hi = compute_global_stretch(b04, b03, b02, selected)
    print(f"Global stretch: lo={lo:.0f}, hi={hi:.0f}")

    for idx, (t, cf) in enumerate(selected):
        label = dates[t].strftime("%Y-%m-%d")

        rgb = to_rgb(b04[t], b03[t], b02[t], lo, hi)

        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_axis_off()
        ax.margins(0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        out = OUT_DIR / f"s2_rgb_{label}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close(fig)
        print(f"  Saved: {out.name}")


if __name__ == "__main__":
    main()
