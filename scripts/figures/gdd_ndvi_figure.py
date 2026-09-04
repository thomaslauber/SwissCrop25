#!/usr/bin/env python3
"""
Combined climate anomaly figure for SwissCrop25.

Two-panel figure:
  Left:  Cumulative GDD Jan–May (national spatial mean from MeteoSwiss TabsD),
         all 7 dataset years (2019–2025).  Full-year GDD shown as an inset in
         the top-left corner of the panel.
  Right: Median NDVI of winter wheat pixels (Swiss Mittelland, Bern area),
         Jan–May, for years where GT data is available.

2024 is highlighted in red (warm-winter anomaly).
2019/2020 are drawn dashed (partial national LNF coverage).
Annotation marks the +53 % cumulative GDD gap at April 1.

Usage:
  python scripts/analysis/combined_climate_figure.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import io
import re
import tarfile
import tempfile
import datetime
import numpy as np
import pandas as pd
import zarr
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.size": 12.5})
import matplotlib.dates as mdates
import matplotlib.lines as mlines
from pathlib import Path
from scipy.signal import savgol_filter

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parents[2]
METEO_DIR  = Path("/mnt/Data-Raw-RE/27_Natural_Resources-RE/99_Meteo_Public/"
                  "MeteoSwiss_netCDF/__griddedData/lv95updated")
S2_DIR     = Path("/mnt/eo-nas1/data/satellite/sentinel2/raw/CH")
GT_TAR_DIR = Path("/mnt/eo-nas1/eoa-share/share/Dominik/GTs")
GT_ZARR_2023 = Path("/mnt/eo-nas1/eoa-share/projects/020_crop1990/data/GTs_Sentinel/2023.zarr")

CACHE_GDD  = ROOT / "results/cache/cgdd"
CACHE_NDVI = ROOT / "results/cache/ndvi_wheat"

OUTPUTS = [
    ROOT / "results/figures/gdd_ndvi.pdf",
    ROOT / "Lauber_2026_SwissCrop_ECCV26_TerraBytes/figures/gdd_ndvi.pdf",
]

# ── Year config ───────────────────────────────────────────────────────────────

ALL_YEARS     = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
PARTIAL_YEARS = {2019, 2020}   # dashed: incomplete national LNF coverage
T_BASE        = 0.0

# 7-colour sequential palette; 2024 breaks the sequence in red
YEAR_COLORS = {
    2019: "#b5cee0",   # light blue  (partial)
    2020: "#88b0cc",   # medium blue (partial)
    2021: "#5fa3c4",   # blue
    2022: "#7dc1a0",   # green
    2023: "#e8a838",   # amber
    2024: "#d62728",   # red — warm-winter anomaly
    2025: "#9467bd",   # purple
}
YEAR_LW = {y: (2.2 if y == 2024 else 1.4) for y in ALL_YEARS}

# ── NDVI config (Bern Mittelland, UTM32N) ─────────────────────────────────────

WHEAT_LNF_CODES = [507, 513]
BBOX_E = (370_000, 410_000)
BBOX_N = (5_193_000, 5_220_000)
VALID_SCL = {4, 5, 6, 7}

ZOOM_END_MONTH = 5   # main panels show Jan through end of May


# ══════════════════════════════════════════════════════════════════════════════
# GDD helpers (MeteoSwiss national mean)
# ══════════════════════════════════════════════════════════════════════════════

def _tabsd_path(year: int) -> Path:
    if year >= 2025:
        return METEO_DIR / (
            f"ogd-surface-derived-grid-archive."
            f"tabsd_ch01r.swiss.lv95_{year}0101000000_{year}1231000000.nc"
        )
    return METEO_DIR / f"TabsD_ch01r.swiss.lv95_{year}01010000_{year}12310000.nc"


def _load_spatial_mean(year: int):
    """Return (time_array, spatial_mean_degC) for a year, using disk cache."""
    CACHE_GDD.mkdir(parents=True, exist_ok=True)
    cache = CACHE_GDD / f"tabsd_mean_{year}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["time"], d["mean"]
    path = _tabsd_path(year)
    if not path.exists():
        print(f"  GDD [{year}]: file not found: {path.name}")
        return None, None
    print(f"  GDD [{year}]: loading…")
    for engine in ("scipy", "h5netcdf"):
        try:
            ds = xr.open_dataset(path, engine=engine)
            break
        except Exception:
            ds = None
    if ds is None:
        print(f"  GDD [{year}]: could not open with any engine")
        return None, None
    rename = {k: {"E": "x", "N": "y"}.get(k, k) for k in ds.dims if k in ("E", "N")}
    if rename:
        ds = ds.rename(rename)
    da = ds["TabsD"]
    sp_mean = da.mean(dim=("y", "x"), skipna=True).values
    time_vals = ds["time"].values
    np.savez(cache, time=time_vals, mean=sp_mean)
    return time_vals, sp_mean


def _to_ref_dates(time_array, ref_year: int = 2000):
    """Map calendar dates to a fixed reference year (leap day → Feb 28)."""
    out = []
    for t in time_array:
        s = str(t)
        m, d = int(s[5:7]), int(s[8:10])
        if m == 2 and d == 29:
            d = 28
        out.append(datetime.datetime(ref_year, m, d))
    return out


def load_gdd(year: int):
    """Return (list[datetime], np.ndarray) of cumulative GDD for ref year 2000."""
    tv, sp = _load_spatial_mean(year)
    if tv is None:
        return None, None
    cgdd = np.cumsum(np.maximum(0.0, sp - T_BASE))
    dates = _to_ref_dates(tv)
    return dates, cgdd


# ══════════════════════════════════════════════════════════════════════════════
# NDVI helpers (winter wheat, Bern Mittelland)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_tile(name: str):
    m = re.search(r"S2_(\d+)_(\d+)_(\d+)_", name)
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)[:4])


def _in_bbox(e: int, n: int) -> bool:
    return BBOX_E[0] <= e <= BBOX_E[1] and (BBOX_N[0] - 1280) <= n <= BBOX_N[1]


_s2_index: dict = {}


def _build_s2_index():
    if _s2_index:
        return
    print("  Building S2 tile index…")
    for name in os.listdir(S2_DIR):
        e, n, y = _parse_tile(name)
        if e is not None:
            _s2_index[(e, n, y)] = S2_DIR / name
    print(f"  S2 index: {len(_s2_index)} tiles")


_tar_cache: dict = {}
_tar_index: dict = {}


def _get_tar_index(year: int):
    if year in _tar_index:
        return _tar_index[year]
    tar_path = GT_TAR_DIR / f"{year}.tar"
    if not tar_path.exists():
        return {}
    tf = tarfile.open(tar_path)
    _tar_cache[year] = tf
    idx = {}
    for m in tf.getmembers():
        match = re.match(r"S2_(\d+)_(\d+)_", m.name)
        if match:
            idx.setdefault(f"{match.group(1)}_{match.group(2)}", m)
    _tar_index[year] = idx
    print(f"  NDVI [{year}]: tar index: {len(idx)} tiles")
    return idx


def _get_wheat_mask(e: int, n: int, year: int):
    if year == 2023:
        tile_dirs = [d for d in GT_ZARR_2023.iterdir()
                     if d.name.startswith(f"S2_{e}_{n}_")]
        if not tile_dirs:
            return None
        z = zarr.open(str(tile_dirs[0]), "r")
        return np.isin(z["lnf_code"][:], WHEAT_LNF_CODES)

    idx = _get_tar_index(year)
    if not idx:
        return None
    member = idx.get(f"{e}_{n}")
    if member is None:
        return None
    tar = _tar_cache[year]
    fobj = tar.extractfile(member)
    if fobj is None:
        return None
    raw = fobj.read()
    with tempfile.NamedTemporaryFile(suffix=".zarr.zip", delete=False) as tf:
        tf.write(raw)
        tmp_path = tf.name
    try:
        store = zarr.ZipStore(tmp_path, mode="r")
        z = zarr.open(store, "r")
        band = z["band"][:]
        wheat_idx = np.where(np.isin(band, WHEAT_LNF_CODES))[0]
        if len(wheat_idx) == 0:
            return None
        lnf_wheat = z["lnf_code"].oindex[wheat_idx, :, :]
        store.close()
    finally:
        os.unlink(tmp_path)
    return (lnf_wheat > 0.5).any(axis=0)


def _tile_ndvi(zpath: Path, wheat_mask, year: int) -> pd.DataFrame:
    z = zarr.open(str(zpath), "r")
    t_raw = z["time"][:]
    dates = pd.to_datetime(t_raw, unit="D", origin=pd.Timestamp(f"{year}-01-01"))
    doys  = dates.day_of_year.values
    b04 = z["s2_B04"][:]
    b08 = z["s2_B08"][:]
    scl = z["s2_SCL"][:]
    records = []
    for i, doy in enumerate(doys):
        b4, b8, sc = b04[i].astype(np.float32), b08[i].astype(np.float32), scl[i]
        valid = np.isin(sc, list(VALID_SCL)) & wheat_mask
        valid &= (b4 > 0) & (b8 > 0) & (b4 < 65000) & (b8 < 65000)
        if valid.sum() < 5:
            continue
        ndvi = (b8[valid] - b4[valid]) / (b8[valid] + b4[valid] + 1e-6)
        records.append({"doy": int(doy), "ndvi": float(np.median(ndvi))})
    return pd.DataFrame(records)


def load_ndvi(year: int) -> pd.DataFrame:
    """Return median NDVI per DOY for wheat pixels (cached)."""
    CACHE_NDVI.mkdir(parents=True, exist_ok=True)
    cache = CACHE_NDVI / f"ndvi_wheat_{year}.parquet"
    if cache.exists():
        print(f"  NDVI [{year}]: from cache")
        return pd.read_parquet(cache)

    _build_s2_index()

    if year == 2023:
        if not GT_ZARR_2023.exists():
            print(f"  NDVI [{year}]: GT zarr not found")
            return pd.DataFrame()
        bbox_tiles = []
        for d in GT_ZARR_2023.iterdir():
            e, n, y = _parse_tile(d.name)
            if e is not None and _in_bbox(e, n):
                bbox_tiles.append((e, n))
    else:
        idx = _get_tar_index(year)
        if not idx:
            print(f"  NDVI [{year}]: no GT tar, skipping")
            return pd.DataFrame()
        bbox_tiles = []
        for key in idx:
            parts = key.split("_")
            e, n = int(parts[0]), int(parts[1])
            if _in_bbox(e, n):
                bbox_tiles.append((e, n))

    print(f"  NDVI [{year}]: {len(bbox_tiles)} tiles in bbox")
    all_records = []
    for e, n in bbox_tiles:
        zpath = _s2_index.get((e, n, year))
        if zpath is None:
            continue
        try:
            wmask = _get_wheat_mask(e, n, year)
            if wmask is None or wmask.sum() < 20:
                continue
            df = _tile_ndvi(zpath, wmask, year)
            all_records.append(df)
        except Exception as ex:
            print(f"    Warning {e}_{n}: {ex}")

    if not all_records:
        print(f"  NDVI [{year}]: no data")
        return pd.DataFrame()

    result = (pd.concat(all_records, ignore_index=True)
              .groupby("doy")["ndvi"].median()
              .reset_index()
              .sort_values("doy")
              .reset_index(drop=True))
    result.to_parquet(cache, index=False)
    print(f"  NDVI [{year}]: {len(result)} DOY entries")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════

REF_YEAR   = 2000   # all GDD dates mapped here for visual alignment
ZOOM_START = datetime.datetime(REF_YEAR, 1, 1)
ZOOM_END   = datetime.datetime(REF_YEAR, ZOOM_END_MONTH, 31)   # May 31


def _year_style(year: int):
    """Return (color, linewidth, linestyle, zorder)."""
    color = YEAR_COLORS[year]
    lw    = YEAR_LW[year]
    ls    = "--" if year in PARTIAL_YEARS else "-"
    zo    = 5 if year == 2024 else 2
    return color, lw, ls, zo


def _plot_gdd_panel(ax, gdd_data: dict, add_inset: bool = True):
    """Draw Jan–May GDD curves on ax; optionally add full-year inset."""

    for year in ALL_YEARS:
        if year not in gdd_data:
            continue
        dates, cgdd = gdd_data[year]
        mask = [d <= ZOOM_END for d in dates]
        pd_ = [d for d, m in zip(dates, mask) if m]
        cg_ = cgdd[[i for i, m in enumerate(mask) if m]]
        c, lw, _ls, zo = _year_style(year)
        ax.plot(pd_, cg_, color=c, linewidth=lw, linestyle="-", zorder=zo)

    ax.set_xlim(ZOOM_START, ZOOM_END)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Cumulative GDD (°C·days)")
    ax.tick_params(labelsize=11.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="0.9", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # ── Inset: full year with zoom-box indicator ───────────────────────────────
    if add_inset:
        axins = ax.inset_axes([0.13, 0.46, 0.42, 0.52])   # top-left
        full_start = datetime.datetime(REF_YEAR, 1, 1)
        full_end   = datetime.datetime(REF_YEAR, 12, 31)
        for year in ALL_YEARS:
            if year not in gdd_data:
                continue
            dates, cgdd = gdd_data[year]
            c, lw, _ls, zo = _year_style(year)
            axins.plot(dates, cgdd, color=c, linewidth=max(lw * 0.65, 0.7),
                       linestyle="-", zorder=zo)
        axins.set_xlim(full_start, full_end)
        axins.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axins.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        axins.yaxis.set_major_locator(plt.MultipleLocator(1000))
        axins.tick_params(labelsize=9, pad=1)
        axins.spines["top"].set_visible(False)
        axins.spines["right"].set_visible(False)
        axins.yaxis.grid(True, color="0.9", linewidth=0.4, zorder=0)
        axins.set_axisbelow(True)

        # Zoom box: x = Jan–May, y = 0 to main panel's ylim top (~700)
        import matplotlib.patches as mpatches
        y_box_top = ax.get_ylim()[1]   # matches the left (zoom) panel
        y_box_bot = axins.get_ylim()[0]
        zoom_width = mdates.date2num(ZOOM_END) - mdates.date2num(ZOOM_START)
        axins.add_patch(mpatches.Rectangle(
            (mdates.date2num(ZOOM_START), y_box_bot),
            zoom_width, y_box_top - y_box_bot,
            linewidth=0, facecolor="#333333", alpha=0.13,
            transform=axins.transData, zorder=0, clip_on=True,
        ))
        axins.add_patch(mpatches.Rectangle(
            (mdates.date2num(ZOOM_START), y_box_bot),
            zoom_width, y_box_top - y_box_bot,
            linewidth=0.9, edgecolor="#444444", facecolor="none",
            transform=axins.transData, zorder=5, clip_on=True,
        ))


def _plot_ndvi_panel(ax, ndvi_data: dict):
    """Draw smoothed median NDVI Jan–Aug wheat curves on ax."""
    DOY_END = 227   # August 15

    for year in ALL_YEARS:
        if year not in ndvi_data or ndvi_data[year].empty:
            continue
        df   = ndvi_data[year]
        mask = df["doy"] <= DOY_END
        doy  = df.loc[mask, "doy"].values
        ndvi = df.loc[mask, "ndvi"].values
        if len(ndvi) < 5:
            continue
        wl = min(31, len(ndvi) // 2 * 2 - 1)
        ndvi_s = savgol_filter(ndvi, window_length=max(wl, 5), polyorder=3)
        c, lw, ls, zo = _year_style(year)
        ax.plot(doy, ndvi_s, color=c, linewidth=lw, linestyle=ls, zorder=zo)

    month_starts = [1, 32, 60, 91, 121, 152, 182, 213]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels, fontsize=11.5)
    ax.set_xlim(1, DOY_END)
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel("Median NDVI")
    ax.tick_params(labelsize=11.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="0.9", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def main():
    # ── Load GDD ──────────────────────────────────────────────────────────────
    print("Loading GDD data…")
    gdd_data = {}
    for year in ALL_YEARS:
        dates, cgdd = load_gdd(year)
        if dates is not None:
            gdd_data[year] = (dates, cgdd)

    # ── Load NDVI ─────────────────────────────────────────────────────────────
    print("\nLoading NDVI data…")
    ndvi_data = {}
    for year in ALL_YEARS:
        df = load_ndvi(year)
        if not df.empty:
            ndvi_data[year] = df

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, (ax_gdd, ax_ndvi) = plt.subplots(1, 2, figsize=(9.0, 3.6))

    _plot_gdd_panel(ax_gdd,  gdd_data,  add_inset=True)
    _plot_ndvi_panel(ax_ndvi, ndvi_data)

    ax_gdd.set_title("Cumulative GDD", pad=10, fontsize=12.5)
    ax_ndvi.set_title("Winter Wheat NDVI (Swiss Mittelland)", pad=10, fontsize=12.5)

    # ── Shared legend (below both panels) ────────────────────────────────────
    legend_handles = []
    for year in ALL_YEARS:
        if year not in gdd_data and year not in ndvi_data:
            continue
        c, lw, ls, _ = _year_style(year)
        # GDD lines are always solid; only NDVI uses dashes for partial years
        ls_legend = ls if year in ndvi_data else "-"
        legend_handles.append(
            mlines.Line2D([], [], color=c, linewidth=lw, linestyle=ls_legend,
                          label=str(year))
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=11.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    for out in OUTPUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
