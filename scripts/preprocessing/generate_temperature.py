import argparse
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings(
    "ignore",
    message="angle from rectified to skew grid parameter lost in conversion to CF",
    category=UserWarning,
    module="pyproj.crs._cf1x8"
)

import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc
import geopandas as gpd
import rioxarray
from tqdm import tqdm
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
import glob


def load_data(args):
    if args.year >= 2025:
        var_lower = args.var.lower()
        path = os.path.join(args.meteo_dir, f"ogd-surface-derived-grid-archive.{var_lower}_ch01r.swiss.lv95_{args.year}0101000000_{args.year}1231000000.nc")
    else:
        path = os.path.join(args.meteo_dir, f"{args.var}_ch01r.swiss.lv95_{args.year}01010000_{args.year}12310000.nc")
    return xr.open_dataset(path).rename({"E": "x", "N": "y"})


def compute_cumulative_gdd(da, t_base, dim='time'):
    mask = ~da.isnull()
    GDD_daily = xr.where(da > t_base, da - t_base, 0)
    GDD = GDD_daily.cumsum(dim=dim)
    GDD = xr.where(mask, GDD, np.nan)
    return GDD


def clip_to_tile(ds, tile, pixel_size, pad=0):
    minx, miny, maxx, maxy = tile.geometry.bounds
    minx -= (pixel_size/2 + pad * pixel_size)
    miny -= (pixel_size/2 + pad * pixel_size)
    maxx += (pixel_size/2 + pad * pixel_size)
    maxy += (pixel_size/2 + pad * pixel_size)
    x_idx = ds.x[(ds.x >= minx) & (ds.x <= maxx)]
    y_idx = ds.y[(ds.y >= miny) & (ds.y <= maxy)]
    return ds.sel({"x": x_idx, "y": y_idx})


def fill_nans_across_border(ds, dims=('y', 'x')):
    filled = ds.copy()
    for dim in dims:
        filled = filled.ffill(dim=dim)
        filled = filled.bfill(dim=dim)
    return filled


def subset_tiles(ds, grid, pixel_size=1000, pad=3):
    tiles = []
    for idx, tile in grid.iterrows():
        ds_tile = clip_to_tile(ds, tile, pixel_size, pad)
        if len(ds_tile.x) == 0 or len(ds_tile.y) == 0:
            tiles.append({})
            continue
        tiles.append({'tile_id': idx, 'tile': tile, 'array': ds_tile})
    return tiles


def generate_mask(da, da_fine):
    xs = da_fine.x.values
    ys = da_fine.y.values
    valid_mask = ~da.isnull()
    if 'time' in da.dims:
        valid_mask = valid_mask.any(dim='time')
    valid_num = valid_mask.astype('uint8')
    valid_mask_fine = valid_num.interp(
        x=(('x', xs)),
        y=(('y', ys)),
        method="nearest"
    )
    return valid_mask_fine.data.astype(bool)


def process_tile(tile, args):
    tile_id, tile, ds = tile.values()

    target_x_res = abs(tile["left"] - tile["right"]) / 128
    target_y_res = abs(tile["bottom"] - tile["top"]) / 128
    assert abs(target_x_res - target_y_res) < 1e-6

    da = ds[args.var].rio.write_crs(args.crs)
    da_filled = fill_nans_across_border(da)

    factor = 10
    da_intermediate = da_filled.rio.reproject(
        da_filled.rio.crs,
        shape=(int(da_filled.shape[1] * factor), int(da_filled.shape[2] * factor)),
        resampling=getattr(Resampling, args.resampling)
    )

    x_res = abs(float(ds.x[1] - ds.x[0]))
    y_res = abs(float(ds.y[1] - ds.y[0]))
    assert abs(x_res - y_res) < 1e-6
    da_intermediate_clipped = clip_to_tile(da_intermediate, tile, pixel_size=int(x_res/factor), pad=args.pad)

    current_res = x_res / factor
    factor = current_res / target_x_res
    da_fine = da_intermediate_clipped.rio.reproject(
        da_intermediate_clipped.rio.crs,
        shape=(int(da_intermediate_clipped.shape[1] * factor), int(da_intermediate_clipped.shape[2] * factor)),
        resampling=getattr(Resampling, args.resampling)
    )

    mask = generate_mask(da, da_fine)
    da_fine = da_fine.where(mask)

    transform = from_bounds(tile["left"], tile["bottom"], tile["right"], tile["top"], 128, 128)
    da_output = da_fine.rio.reproject(
        "EPSG:32632",
        shape=(128, 128),
        transform=transform,
        resampling=getattr(Resampling, args.resampling)
    )

    mean = da_output.mean(axis=(1, 2))

    x_res_out = abs(float(da_output.x[1] - da_output.x[0]))
    y_res_out = abs(float(da_output.y[1] - da_output.y[0]))
    x = da_output.x - x_res_out / 2
    y = da_output.y + y_res_out / 2

    fill_value = np.nan
    var = "CGDD"

    var_mean_da = xr.DataArray(mean, dims=("time",), coords={"time": da.time}).rio.write_nodata(fill_value, inplace=True)

    if args.save_spatial:
        var_da = xr.DataArray(da_output.values, dims=("time", "lat", "lon"), coords={"time": da.time, "lat": y.values, "lon": x.values}).rio.write_nodata(fill_value, inplace=True)
        ds_output = xr.Dataset({
            var: var_da.chunk({'time': -1, 'lat': -1, 'lon': len(x.values)//2}),
            f"{var}_mean": var_mean_da
        })
    else:
        ds_output = xr.Dataset({f"{var}_mean": var_mean_da})

    attributes = args.original_attributes.copy()
    attributes["grid"] = 'EPSG:32632. Coordinates are upper left corners of pixels'
    attributes["history"] += f", \n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - reprojected and regridded by {os.getenv('USER')}"
    attributes["base temperature"] = args.t_base
    attributes["average temperature"] = "Tabs" if "Tabs" in args.mean_var else "(Tmin + Tmax) / 2"
    ds_output.attrs = attributes

    tile_id_str = f"{int(tile.left)}_{int(tile.top)}"
    path = f"S2_{tile_id_str}_{args.year}0101_{args.year}1231.zarr.zip"
    full_path = os.path.join(args.out_dir, path)

    if os.path.exists(full_path) and not args.overwrite:
        return
    compressor = Blosc(cname='zstd', clevel=3, shuffle=2)
    store = zarr.ZipStore(full_path, mode='w')
    ds_output.to_zarr(
        store, consolidated=True, mode='w', zarr_format=2,
        encoding={v: {'compressor': compressor} for v in ds_output.data_vars}
    )
    store.close()


def main(args):
    print(f"LOADING DATA")
    if not any(x in args.mean_var for x in ["Tmin", "Tmax"]):
        ds = load_data(args).load()
        T = args.var[-1]
        ds = ds.rename({f"Tabs{T}": "Tmean"})
        args.var = "Tmean"
    else:
        T = args.var[-1]
        args.var = f"Tmax{T}"
        da_max = load_data(args).load()[args.var]
        args.var = f"Tmin{T}"
        ds = load_data(args).load()
        da_min = ds[args.var]
        ds["Tmean"] = (da_max + da_min) / 2
        ds = ds.drop_vars(args.var)
        args.var = "Tmean"

    args.original_attributes = ds.attrs
    args.crs = ds[args.var].attrs["esri_pe_string"]

    x_res = float((ds[args.var].x[1] - ds[args.var].x[0]).values)
    y_res = float((ds[args.var].y[1] - ds[args.var].y[0]).values)
    assert abs(x_res - y_res) < 1e-6
    args.pad = 3

    print(f"COMPUTE CUMULATIVE GDD (base temperature {args.t_base}°C)")
    CGDD = compute_cumulative_gdd(ds, args.t_base)
    ds = CGDD

    sentinel_grid = "/mnt/eo-nas1/eoa-share/projects/012_EO_dataInfrastructure/Project layers/gridface_s2tiles_CH.shp"
    grid = gpd.read_file(sentinel_grid)
    grid_lv95 = grid.to_crs(args.crs)

    print(f"SPLIT DATA INTO TILES")
    tiles_to_process = subset_tiles(ds, grid_lv95, pixel_size=float(x_res), pad=args.pad)

    processed_tiles = set(glob.glob(os.path.join(args.out_dir, "*.zarr.zip")))
    tiles_to_do = []
    for tile in tiles_to_process:
        _, t, _ = tile.values()
        tile_id = f"_{int(t['left'])}_{int(t['top'])}_"
        if args.overwrite or all(tile_id not in p for p in processed_tiles):
            tiles_to_do.append(tile)

    def batched(iterable, n):
        from itertools import islice
        it = iter(iterable)
        while batch := list(islice(it, n)):
            yield batch

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        with tqdm(total=len(tiles_to_do), desc=f"Processing year {args.year}") as pbar:
            for batch in batched(tiles_to_do, 50):
                futures = [executor.submit(process_tile, tile, args) for tile in batch]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print("Tile processing failed:", e)
                    pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cumulative Growing Degree Day (CGDD) time series from MeteoSwiss data, aligned to the Sentinel-2 tile grid.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--year", type=int, required=True, help="Year to process.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for Zarr tiles.")
    parser.add_argument("--meteo_dir", type=str,
                        default="/mnt/Data-Raw-RE/27_Natural_Resources-RE/99_Meteo_Public/MeteoSwiss_netCDF/__griddedData/lv95updated/",
                        help="Directory containing MeteoSwiss NetCDF files.")
    parser.add_argument("--mean_var", type=str, default="TabsD",
                        choices=["TabsD", "TabsM", "TminD", "TminM", "TmaxD", "TmaxM"],
                        help="Temperature variable for daily mean. Use 'Tabs' (default) or 'Tmin'/'Tmax' to compute mean as (Tmax+Tmin)/2.")
    parser.add_argument("--t_base", type=float, default=0,
                        help="Base temperature for GDD accumulation (°C). Default: 0.")
    parser.add_argument("--resampling", type=str, default="nearest",
                        choices=[r.name for r in Resampling],
                        help="Resampling method. Default: nearest.")
    parser.add_argument("--save_spatial", action="store_true",
                        help="Also save the full 128x128 spatial CGDD array per tile (in addition to the spatial mean).")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Zarr stores.")
    args = parser.parse_args()

    args.var = args.mean_var
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
