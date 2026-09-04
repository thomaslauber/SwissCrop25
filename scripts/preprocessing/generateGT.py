import argparse
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from concurrent.futures import ProcessPoolExecutor, as_completed
from sys import stdout
import tempfile
import subprocess
import re
import glob
from datetime import datetime
from functools import partial
import shutil
import zipfile
from collections import Counter

import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import xarray as xr
import rioxarray
import rasterio
import numpy as np
import zarr
import fsspec

# Band name mapping: sensor-specific → generic names (Landsat only)
LANDSAT_BAND_MAP = {
    "OLI_B1": "Coastal", "OLI_B2": "Blue",  "OLI_B3": "Green",
    "OLI_B4": "Red",     "OLI_B5": "NIR",   "OLI_B6": "SWIR1", "OLI_B7": "SWIR2",
    "ETM_B1": "Blue",    "ETM_B2": "Green",  "ETM_B3": "Red",
    "ETM_B4": "NIR",     "ETM_B5": "SWIR1",  "ETM_B7": "SWIR2",
}

def get_spatial_key(tile_name):
  """Strip date range from tile name to get a sensor-agnostic spatial key.
  E.g. 'LS_262230_5113980_20190104_20191230' -> 'LS_262230_5113980'
  Falls back to full tile name if it doesn't match the expected pattern."""
  parts = tile_name.split('_')
  if len(parts) >= 3:
    return '_'.join(parts[:3])
  return tile_name

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(folder_path):
            for f in files:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, folder_path)
                zf.write(abs_path, arcname=rel_path)
    shutil.rmtree(folder_path)

def compute_image_stats(arr, file_name, tier1_scene_ids=None):
  """ Compute mean, stddev and count from the input image array.
      For Landsat data, band names are automatically renamed to generic names
      (Blue, Green, Red, NIR, SWIR1, SWIR2) and only Tier-1 scenes are included
      in the statistics (when tier1_scene_ids is provided)."""
  keys = list(arr.keys())
  is_landsat = any(k.startswith(("OLI_", "ETM_")) for k in keys)
  band_map = LANDSAT_BAND_MAP if is_landsat else {}

  # Build Tier-1 time mask for Landsat data
  time_mask = None
  if is_landsat and tier1_scene_ids is not None and "scene_id" in keys:
    scene_ids = arr["scene_id"][:]
    time_mask = np.array([sid in tier1_scene_ids for sid in scene_ids])

  # Load QA_PIXEL once for re-use across bands (apply time mask if needed)
  qa_full = None
  if "QA_PIXEL" in keys:
    qa_full = arr["QA_PIXEL"][:]
    if time_mask is not None and qa_full.ndim == 3:
      qa_full = qa_full[time_mask]

  stats = {}
  for var in keys:
    data = arr[var][:]

    # Skip non-numeric data
    if not np.issubdtype(data.dtype, np.number):
      continue

    # Apply Tier-1 time filter (Landsat only, 3-D arrays have a time dimension)
    if time_mask is not None and data.ndim == 3:
      data = data[time_mask]

    # Get mask of valid data
    mask = np.ones_like(data, dtype=bool)
    if "missing data fill value" in arr.attrs:
      fill_value = arr.attrs["missing data fill value"]
      mask = mask & (data != fill_value)
    if data.ndim == 3 and qa_full is not None:
        # mask nodata values using bit 0 == 1
        nodata_flag = (qa_full >> 0) & 1
        valid_qa = nodata_flag == 0  # Keep only bit0 == 0 (good data)
        mask = mask & valid_qa

    # Compute statistics
    count = mask.sum()
    mean = data[mask].mean() if count > 0 else np.nan
    std = data[mask].std() if count > 0 else np.nan

    stats[var] = {
      "count": int(count),
      "mean": float(mean) if not np.isnan(mean) else None,
      "std": float(std) if not np.isnan(std) else None
    }

  # Rename Landsat band keys to generic names
  return {band_map.get(k, k): v for k, v in stats.items()}

def generate_class_weights(path, args):
  """ Generate class weights from the given property in the input arr.
      For coverage_fractions, returns summed coverage fractions (fractional pixel counts).
  """
  tile_path = path
  prop = args.property
  try:
      arr = load_zarr(tile_path)

      # Check if multi-band (coverage_fractions) FIRST by checking for 'band' dimension
      if 'band' in arr and prop in arr:
          # Multi-band coverage_fractions format
          data_arr = arr[prop]
          # Get band coordinate values (these are the lnf_codes)
          band_coords = arr['band'][:]

          # Vectorized sum: sum along spatial dimensions (height, width) for all bands at once
          # This is MUCH faster than looping through each band
          sums = np.sum(data_arr[:], axis=(1, 2))  # Sum along dimensions 1 and 2 (height, width)
          counts = dict(zip(band_coords.astype(int).tolist(), sums.tolist()))

          return counts
      elif prop in arr:
          # Single-band mode output (old format for backward compatibility)
          data = arr[prop][:]
          unique, counts = np.unique(data, return_counts=True)
          return dict(zip(unique.tolist(), counts.tolist()))
      else:
          return {"_error": "Property not found in zarr file"}

  except Exception as e:
       return {"_error": str(e)}

def generate_GTs(arr, file_name, args):
  """ Generate Ground Truths from polygons as given by args.polygon_path and args.year and
      save them in the form of the input args.image_dir."""

  # Early-exit: if a GT for this spatial key already exists, skip generation
  # (handles duplicate spatial locations from multiple sensor sources)
  year_dir = os.path.join(args.output_dir, str(args.year))
  candidate = os.path.join(year_dir, file_name + ".zarr")
  zip_candidate = candidate + ".zip"
  if os.path.exists(zip_candidate):
    return zip_candidate, None
  if os.path.exists(candidate):
    return candidate, None

  # Get coordinates (these are top-left corner coordinates)
  if "lat" in arr and "lon" in arr:
    lat_corners, lon_corners = arr["lat"][:], arr["lon"][:]
    coords = "lat", "lon"
  elif "latitude" in arr and "longitude" in arr:
    lat_corners, lon_corners = arr["latitude"][:], arr["longitude"][:]
    coords = "latitude", "longitude"
  elif "y" in arr and "x" in arr:
    lat_corners, lon_corners = arr["y"][:], arr["x"][:]
    coords = "y", "x"
  elif "Y" in arr and "X" in arr:
    lat_corners, lon_corners = arr["Y"][:], arr["X"][:]
    coords = "Y", "X"
  else:
    raise KeyError("No latitude/longitude coordinates found in dataset.")

  # Get crs
  crs = "EPSG:32632"

  # Calculate pixel spacing (assuming regular grid)
  lat_spacing = np.mean(np.diff(lat_corners))
  lon_spacing = np.mean(np.diff(lon_corners))

  # Round to avoid floating point drift for integer spacings (e.g., 10m, 30m)
  lat_spacing = np.round(lat_spacing, decimals=6)
  lon_spacing = np.round(lon_spacing, decimals=6)

  # Convert corner coordinates to center coordinates
  # For top-left corners: center = corner + spacing/2
  lat_centers = lat_corners + lat_spacing / 2
  lon_centers = lon_corners + lon_spacing / 2

  # Generate binary mask with CENTER coordinates
  mask = xr.DataArray(
    np.ones((len(lat_centers), len(lon_centers)), dtype="uint8"),
    coords={"lat": lat_centers, "lon": lon_centers},
    dims=("lat", "lon"),
  ).rio.write_crs(crs).rio.set_spatial_dims(x_dim="lon", y_dim="lat")

  # Create a temporary GeoTIFF
  with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmpfile:
    tmp_path = tmpfile.name

  # Save to temporary file 
  mask.rio.to_raster(tmp_path)

  # Call reduceToImage and overwrite the tif
  cmd = [
    "Rscript",
    os.path.join(_SCRIPT_DIR, "reduceToImage.R"),
    "--image_path", tmp_path,
    "--polygon_path", args.polygon_file,
    "--output_path", tmp_path,
    "--property", args.property,
    "--reducer", args.reducer
  ]

  # For coverage_fractions, pass all lnf_codes to avoid loading full polygon file
  if args.reducer == "coverage_fractions" and hasattr(args, 'all_lnf_codes') and args.all_lnf_codes:
    cmd.extend(["--all_lnf_codes", ",".join(map(str, args.all_lnf_codes))])

  result = subprocess.run(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    check=False
  )
  stdout = result.stdout
  returncode = result.returncode
  if returncode != 0:
    # Log warning and skip this tile (often due to corrupt geometries in polygon file)
    print(f"WARNING: R reduceToImage failed for {file_name} - skipping tile")
    print(f"  Error: {result.stderr[:200]}...")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return None, None
  if not stdout.strip() or stdout.strip() == "NULL":
    # Skip tile silently - count will be reported at the end
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return None, None

  # Read back the processed raster
  result = rioxarray.open_rasterio(tmp_path)

  # For multi-band rasters, read band names from R output and set as coordinates
  if result.sizes.get("band", 1) > 1:
    # Read band descriptions from the GeoTIFF (R sets band names there)
    with rasterio.open(tmp_path) as src:
      band_names = [src.descriptions[i] or str(i+1) for i in range(src.count)]

    # Convert band names to integers (they should be lnf_codes)
    try:
      band_codes = [int(name) for name in band_names]
      # Update the band coordinate with actual lnf_codes
      result = result.assign_coords(band=band_codes)
    except ValueError:
      # If band names aren't integers, keep sequential indices
      pass

  # Fill NAs
  result = result.fillna(args.fill_value)

  # Delete temporary file
  os.remove(tmp_path)

  # Convert center coordinates back to top-left corner coordinates
  # Extract the center coordinates that came from the GeoTIFF
  result_lat_centers = result.coords["y"].values
  result_lon_centers = result.coords["x"].values

  # Calculate pixel spacing from center coordinates
  lat_spacing = np.mean(np.diff(result_lat_centers))
  lon_spacing = np.mean(np.diff(result_lon_centers))

  # Round to avoid floating point drift for integer spacings (e.g., 10m, 30m)
  lat_spacing = np.round(lat_spacing, decimals=6)
  lon_spacing = np.round(lon_spacing, decimals=6)

  # Convert centers back to top-left corners: corner = center - spacing/2
  result_lat_corners = result_lat_centers - lat_spacing / 2
  result_lon_corners = result_lon_centers - lon_spacing / 2

  # Assign corner coordinates to the result
  result = result.assign_coords({
      "y": result_lat_corners,
      "x": result_lon_corners
  })

  # Save to zarr in year subdirectory
  result = result.drop_vars("spatial_ref")

  # Only squeeze band dimension if it's single-band (mode reducer)
  # Multi-band (coverage_fractions) should keep the band dimension
  if result.sizes.get("band", 1) == 1:
    result = result.squeeze("band", drop=True)

  # Set name for both single-band and multi-band
  result.name = args.property

  result = result.rename({"x": coords[1], "y": coords[0]})
  result.attrs = {}
  result.attrs["grid"] = crs
  result.attrs["history"] = f"Created by {os.environ.get('USER')} on {datetime.now().strftime('%Y-%m-%d %H:%M')} using {os.path.dirname(os.path.abspath(__file__))}."
  result.attrs["missing data fill value"] = args.fill_value
  
  # Create year subdirectory
  year_dir = os.path.join(args.output_dir, str(args.year))
  os.makedirs(year_dir, exist_ok=True)
  
  output_file_name = file_name + ".zarr" if not file_name.endswith(".zarr") else file_name
  output_path = os.path.join(year_dir, output_file_name)

  # Compute class weights from in-memory result BEFORE saving to disk
  # This avoids having to reload the file from disk later
  class_weights = None
  if not args.no_class_weights:
    # Check if multi-band (coverage_fractions) by checking for 'band' dimension
    if 'band' in result.dims and result.sizes.get("band", 1) > 1:
      # Multi-band coverage_fractions format
      data_arr = result.values  # Get numpy array
      band_coords = result.coords['band'].values  # Get band coordinate values (lnf_codes)

      # Vectorized sum along spatial dimensions for all bands at once
      sums = np.sum(data_arr, axis=(1, 2))
      class_weights = dict(zip(band_coords.astype(int).tolist(), sums.tolist()))
    else:
      # Single-band mode
      data = result.values
      unique, counts = np.unique(data, return_counts=True)
      class_weights = dict(zip(unique.tolist(), counts.tolist()))

  # Save as zarr
  result.to_zarr(output_path, mode="w", zarr_format=2)

  # Optionally zip the output and delete the zarr folder
  if args.zip:
    zip_folder(output_path, output_path + ".zip")
    output_path = output_path + ".zip"

  return output_path, class_weights


# A function to open and load the zarr data once
zarr_cache = {}
def load_zarr(path):
  # Case 1: zipped zarr file
  if '.zarr.zip/' in path or path.endswith('.zarr.zip'):
    # Get store and keys from the path
    store_path, sep, group_key = path.partition('.zarr.zip')
    store_path += '.zarr.zip'
    group_key = group_key.lstrip('/')
    if store_path not in zarr_cache:
       store = zarr.storage.ZipStore(store_path, mode='r')
       try:
           zarr_group = zarr.open_consolidated(store, mode='r')
       except Exception:
           zarr_group = zarr.open(store, mode='r')
       zarr_cache[store_path] = zarr_group
    return zarr_cache[store_path] if group_key == "" else zarr_cache[store_path][group_key]

  # Case 2: JSON reference
  elif path.endswith(".json") or ".json/" in path:
    store_path, sep, group_key = path.partition(".json")
    store_path += ".json"
    group_key = group_key.lstrip("/")
    # Cache the fs to avoid re-reading the large JSON for every tile
    fs_key = "__fs__" + store_path
    if fs_key not in zarr_cache:
        import json as _json
        with open(store_path) as _f:
            _ref = _json.load(_f)
        # Resolve relative template paths against the JSON's own directory
        # so the JSON is portable (not tied to an absolute path on one machine)
        if "templates" in _ref:
            _base = os.path.dirname(os.path.abspath(store_path))
            _ref["templates"] = {
                k: os.path.join(_base, v) if not os.path.isabs(v) and "://" not in v else v
                for k, v in _ref["templates"].items()
            }
        zarr_cache[fs_key] = fsspec.filesystem("reference", fo=_ref)
    fs = zarr_cache[fs_key]
    if group_key == "":
        if store_path not in zarr_cache:
            try:
                zarr_cache[store_path] = zarr.open_consolidated(fs.get_mapper(""), mode="r")
            except Exception:
                zarr_cache[store_path] = zarr.open_group(fs.get_mapper(""), mode="r", zarr_format=2)
        return zarr_cache[store_path]
    else:
        # Open with a mapper rooted at the tile so zarr can enumerate band arrays via keys()
        return zarr.open_group(fs.get_mapper(group_key), mode="r", zarr_format=2)
  
  # Case 3: local zarr file
  elif '.zarr/' in path or path.endswith('.zarr'):
    # Get store and keys from the path
    store_path, sep, group_key = path.partition('.zarr')
    store_path += '.zarr'
    group_key = group_key.lstrip('/')
    if store_path not in zarr_cache:
        store = zarr.storage.LocalStore(store_path)
        try:
            zarr_group = zarr.open_consolidated(store, mode='r')
        except Exception:
            zarr_group = zarr.open(store, mode='r')
        if group_key != "" or any(zarr_group.group_keys()):
            zarr_cache[store_path] = zarr_group
    if group_key == "":
        return zarr_group
    else:
        return zarr_cache[store_path][group_key]
 
  else:
      raise ValueError(f"Unsupported zarr path format: {path}")

def combine_image_stats(all_stats):
    """
    Combine per-image statistics into dataset-wide mean and std.
    all_stats: list of dicts like {band: {'count':..., 'mean':..., 'std':...}, ...}
    Returns: dict {band: {'count':..., 'mean':..., 'std':...}}
    """
    # Get union of all band names (tiles may differ, e.g. OLI has Coastal, ETM does not)
    bands = set().union(*(s.keys() for s in all_stats))
    # Loop over band names
    combined = {}
    for band in bands:
        counts = np.array([s[band]['count'] if band in s else 0 for s in all_stats])
        means = np.array([s[band]['mean'] if band in s and s[band]['mean'] is not None else np.nan for s in all_stats])
        stds = np.array([s[band]['std'] if band in s and s[band]['std'] is not None else np.nan for s in all_stats])
        
        # Remove empty images
        mask = counts > 0
        counts = counts[mask]
        means = means[mask]
        stds = stds[mask]
        if counts.size == 0:
            combined[band] = {'count': 0, 'mean': None, 'std': None}
            continue

        # Compute weighted global mean
        global_mean = np.sum(means * counts) / np.sum(counts)

        # Compute weighted variance
        global_var = (
            np.sum((counts - 1) * stds**2 + counts * (means - global_mean)**2)
            / (np.sum(counts) - 1)
        )
        global_std = np.sqrt(global_var)
        
        combined[band] = {
            'count': int(np.sum(counts)),
            'mean': float(global_mean),
            'std': float(global_std)
        }
    return combined

def process_image(path, args):
  """ Process the image. 
      This includes (i) computing statistics of the original image, (ii) generating GTs, and (iii) generating class weights."""

  # Load the data
  tile_name = os.path.basename(path).replace('.zarr.zip', '').replace('.zarr', '')
  file_name = get_spatial_key(tile_name)  # sensor-agnostic spatial key (strips date range)
  arr = load_zarr(path)

  # Compute statistics from image (with Landsat band renaming + Tier-1 filtering)
  stats = None
  if not args.no_stats:
     stats = compute_image_stats(arr, file_name, tier1_scene_ids=getattr(args, "tier1_scene_ids", None))
  
  # Generate GTs or determine GT path
  output_path = None
  class_weights = None
  if not args.no_gts:
     # Generate new GTs - they will be saved in year subdirectory
     # generate_GTs now returns both path and class_weights (computed from in-memory data)
     output_path, class_weights = generate_GTs(arr, file_name, args)
  else:
     # GTs already exist - determine where they are (using spatial key as file name)
     year_dir = os.path.join(args.output_dir, str(args.year))

     # Try different possible locations
     possible_paths = [
         # In year subdirectory as .zarr.zip
         os.path.join(year_dir, file_name + ".zarr.zip"),
         # In year subdirectory as .zarr
         os.path.join(year_dir, file_name + ".zarr"),
         # In tar.json reference
         os.path.join(args.output_dir, f"{args.year}.tar.json", file_name),
         # In tar.json reference with .zarr extension
         os.path.join(args.output_dir, f"{args.year}.tar.json", file_name + ".zarr"),
         # In tar.json reference with .zarr.zip extension
         os.path.join(args.output_dir, f"{args.year}.tar.json", file_name + ".zarr.zip"),
     ]

     for possible_path in possible_paths:
         try:
             # Try to load to verify it exists
             _ = load_zarr(possible_path)
             output_path = possible_path
             break
         except:
             continue

     # Only load from disk if GTs already exist (not generated in this run)
     if not args.no_class_weights and output_path is not None:
         class_weights = generate_class_weights(output_path, args)

  return (file_name, stats, class_weights)


def main(args):
  os.makedirs(args.output_dir, exist_ok=True)

  # Print configuration
  print("=" * 60)
  print("Ground Truth Generation - Configuration")
  print("=" * 60)
  print(f"Output directory:      {args.output_dir}")
  print(f"Year:                  {args.year}")
  for i, d in enumerate(args.image_dir):
    label = "Image dir(s):" if i == 0 else "              "
    print(f"{label}         {d}")
  print(f"Tier-1 lookup:         {args.tier_lookup}")
  print(f"Polygon path:          {args.polygon_path}")
  print(f"Property:              {args.property}")
  print(f"Reducer:               {args.reducer}")
  print(f"Number of workers:     {args.num_workers}")
  print(f"Chunk size:            200")
  print(f"Fill value:            {args.fill_value}")
  print(f"Zip output:            {args.zip}")
  print(f"Generate GTs:          {not args.no_gts}")
  print(f"Compute stats:         {not args.no_stats}")
  print(f"Compute class weights: {not args.no_class_weights}")
  print("=" * 60)
  print()

  # For coverage_fractions reducer, read all lnf_codes once from polygon file
  # This avoids each worker loading the full polygon file
  if args.reducer == "coverage_fractions" and not args.no_gts:
    print(f"Reading all lnf_codes from polygon file (this may take a moment)...")
    args.polygon_file = os.path.join(
        args.polygon_path,
        args.polygon_file_template.format(year=args.year)
    )
    polygon_file = args.polygon_file

    # Use geopandas to extract unique lnf_codes
    polygons_df = gpd.read_file(polygon_file, columns=[args.property])
    all_lnf_codes = sorted(polygons_df[args.property].astype(int).unique().tolist())
    args.all_lnf_codes = all_lnf_codes
    print(f"Found {len(all_lnf_codes)} unique lnf_codes")

    # Explicitly free memory from polygon file
    del polygons_df
  else:
    args.all_lnf_codes = None

  # Load Tier-1 scene IDs once (Landsat only — auto-applied when bands match OLI_/ETM_)
  args.tier1_scene_ids = None
  if os.path.exists(args.tier_lookup):
    print(f"Loading Tier-1 scene list from {args.tier_lookup}...")
    tier1_df = pd.read_csv(args.tier_lookup, usecols=["scene_id", "collection_category"])
    args.tier1_scene_ids = set(tier1_df.loc[tier1_df["collection_category"] == "T1", "scene_id"])
    print(f"  {len(args.tier1_scene_ids)} Tier-1 scenes loaded")
    del tier1_df
  else:
    print(f"WARNING: Tier-1 lookup file not found at {args.tier_lookup} — no Tier-1 filtering applied")

  # Collect image paths from all sources
  all_image_paths = []
  for image_dir in args.image_dir:
    if not os.path.exists(image_dir):
      print(f"WARNING: image_dir not found, skipping: {image_dir}")
      continue
    if not image_dir.endswith((".zarr", ".zarr.zip", ".zarr.zip.json", ".json")):
      paths = glob.glob(f"{image_dir}/*_*_*_{args.year}*_{args.year}*")
    else:
      zarr_group = load_zarr(image_dir)
      paths = [os.path.join(image_dir, key) for key in zarr_group.keys()]
    all_image_paths.extend(paths)

  if not all_image_paths:
    print("ERROR: no tiles found across all image_dirs — nothing to process. Exiting.")
    return

  # Split into primary (unique spatial key → will generate GT) and secondary
  # (duplicate spatial key → GT already exists, only compute stats).
  # Processing primaries first eliminates race conditions: all GTs are written
  # before any secondary tile checks for them, with no loss of parallelism.
  seen_spatial_keys = set()
  primary_paths, secondary_paths = [], []
  for path in all_image_paths:
    tile_name = os.path.basename(path).replace('.zarr.zip', '').replace('.zarr', '')
    sk = get_spatial_key(tile_name)
    if sk not in seen_spatial_keys:
      seen_spatial_keys.add(sk)
      primary_paths.append(path)
    else:
      secondary_paths.append(path)

  n_dup = len(secondary_paths)
  print(f"Found {len(all_image_paths)} tiles ({len(primary_paths)} unique spatial keys"
        + (f", {n_dup} duplicates from secondary sources)" if n_dup else ")"))

  def run_parallel(paths, pbar):
    results = []
    for i in range(0, len(paths), chunk_size):
      chunk = paths[i:i + chunk_size]
      with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(worker_func, p) for p in chunk]
        for fut in as_completed(futures):
          results.append(fut.result())
          pbar.update(1)
    return results

  worker_func = partial(process_image, args=args)
  chunk_size = 200  # Reduced to limit memory usage
  all_results = []
  with tqdm(total=len(all_image_paths), miniters=10) as pbar:
    # Pass 1: primary tiles — generates GTs + stats
    all_results.extend(run_parallel(primary_paths, pbar))
    # Pass 2: secondary tiles — GTs already written, stats only (no race condition)
    if secondary_paths:
      all_results.extend(run_parallel(secondary_paths, pbar))

  # Print summary
  if not args.no_gts:
      # Count successful, empty, and error tiles based on class_weights (res[2])
      successful_gts = sum(1 for res in all_results if res[2] is not None and isinstance(res[2], dict) and "_error" not in res[2])
      empty_gts = sum(1 for res in all_results if res[2] is None)
      error_gts = sum(1 for res in all_results if res[2] is not None and isinstance(res[2], dict) and "_error" in res[2])
      print(f"\nProcessing complete:")
      print(f"  {successful_gts} tiles succeeded")
      if empty_gts > 0:
          print(f"  {empty_gts} tiles skipped (no polygon coverage)")
      if error_gts > 0:
          print(f"  {error_gts} tiles had errors")
  else:
      print(f"\nProcessing complete: {len(all_results)} tiles processed")

  # ==================== Save Pixel Counts ====================
  if not args.no_class_weights:
      print("Processing pixel counts...")
      # Use long format for efficient parquet storage
      total_counts = Counter()
      pixel_count_rows = []

      for res in all_results:
          if isinstance(res[2], dict) and "_error" not in res[2]:
              tile_name = res[0]
              # Expand dict to long format rows immediately
              for lnf_code, count in res[2].items():
                  pixel_count_rows.append({
                      'tile': tile_name,
                      args.property: int(lnf_code),
                      'count': count
                  })
              total_counts.update(res[2])

      if pixel_count_rows:
          # Save per-tile counts as parquet in long format (MUCH faster)
          out_file_per_tile = os.path.join(args.output_dir, f"{args.year}_pixelCounts.parquet")
          df_per_tile = pd.DataFrame(pixel_count_rows)
          df_per_tile.to_parquet(out_file_per_tile, index=False)
          print(f"Saved per-tile pixel counts to {out_file_per_tile}")
          del df_per_tile
          del pixel_count_rows

          # Save global counts for backward compatibility
          df_global = pd.DataFrame(list(total_counts.items()), columns=[args.property, "count"])
          df_global = df_global.sort_values(args.property).reset_index(drop=True)
          out_file_global = os.path.join(args.output_dir, f"{args.year}_pixelCounts.csv")
          df_global.to_csv(out_file_global, index=False)
          print(f"Saved global pixel counts to {out_file_global}")
          del df_global
          del total_counts

  # ==================== Save Statistics ====================
  if not args.no_stats:
      print("Processing statistics...")
      # Process in a single pass to avoid duplicate data structures
      all_stats = []
      stats_rows = []

      for file_name, stats, class_weights in all_results:
          if stats is not None and stats:  # All tiles with computed stats (GT may have been skipped as duplicate)
              all_stats.append(stats)
              # Flatten stats dict to long format rows
              for band, band_stats in stats.items():
                  stats_rows.append({
                      'tile': file_name,
                      'band': band,
                      'count': band_stats['count'],
                      'mean': band_stats['mean'],
                      'std': band_stats['std']
                  })

      print(f"Collected stats from {len(all_stats)} tiles")
      if all_stats:
          # Save per-tile stats as parquet in long format
          df_stats_per_tile = pd.DataFrame(stats_rows)
          out_file_per_tile = os.path.join(args.output_dir, f"{args.year}_stats.parquet")
          df_stats_per_tile.to_parquet(out_file_per_tile, index=False)
          print(f"Saved per-tile statistics to {out_file_per_tile}")
          del df_stats_per_tile
          del stats_rows

          # Compute and save global statistics
          print("Computing global statistics...")
          combined_stats = combine_image_stats(all_stats)
          del all_stats  # Free memory before creating next dataframe

          stats_global = []
          for band, s in combined_stats.items():
              stats_global.append({
                  "band": band,
                  "count": s["count"],
                  "mean": s["mean"],
                  "std": s["std"]
              })
          df_stats_global = pd.DataFrame(stats_global)
          stats_global_file = os.path.join(args.output_dir, f"{args.year}_stats_global.csv")
          df_stats_global.to_csv(stats_global_file, index=False)
          print(f"Saved global statistics to {stats_global_file}")
          del df_stats_global
          del stats_global
          del combined_stats

          print("Statistics processing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description="""
      Generate Ground Truths from polygons and input images. \n
      !! Warning: Do not zip the output files directly when working on a network drive like the NAS.""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-o", "--output_dir", required=True, help="The output directory where outputs will be saved.")
    parser.add_argument("-y", "--year", type=int, required=True, help="Year of the data to process.")
    parser.add_argument("-p", "--polygon_path", default="/capstor/store/cscs/2go/go57/data/landuse/", help="Path to the directory holding the polygons (default: '/mnt/eo-nas1/data/landuse/raw').")
    parser.add_argument("--polygon_file_template", default="LNF_swissTLM3D_{year}.gpkg", help="Template for polygon filename inside polygon_path (default: LNF_swissTLM3D_{year}.gpkg)")
    parser.add_argument("-i", "--image_dir", nargs="+", required=True, help="Path(s) to the image source(s). Accepts one or more zarr, zarr.zip, or JSON reference paths. Multiple paths are combined into one flat tile list (e.g. for merging Landsat 7 and 8/9).")
    parser.add_argument("--tier_lookup", default="/capstor/store/cscs/2go/go57/data/satellite/landsat/scene_tier_lookup.csv", help="Path to scene_tier_lookup.csv for Tier-1 filtering. Applied automatically when Landsat bands (OLI_/ETM_) are detected (default: /capstor/store/cscs/2go/go57/data/satellite/landsat/scene_tier_lookup.csv).")
    parser.add_argument("--property", default="lnf_code", help="The property to map (default: 'lnf_code').")
    parser.add_argument("--reducer", default="coverage_fractions", help="The reducer to use. Needs to be one of 'mode', 'centroid', or 'coverage_fractions' (default 'coverage_fractions').")
    parser.add_argument("--zip", type=bool, default=True, help="If the output should be stored in zip files (default: True).")
    parser.add_argument("--num_workers", type=int, default=2, help="The number of workers (default: 2).")
    parser.add_argument("--fill_value", type=int, default=0, help="The fill_value for the background pixels (default: 0).")
    parser.add_argument("--no_gts", action="store_true", help="If specified, no ground truths are generated from the input images.")
    parser.add_argument("--no_stats", action="store_true", help="If specified, no statistics like mean, stddev and count are computed from the input images.")
    parser.add_argument("--no_class_weights", action="store_true", help="If specified, no class weights are generated from the ground truths.")

    args = parser.parse_args()

    main(args)