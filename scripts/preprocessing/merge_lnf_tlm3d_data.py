import geopandas as gpd
import numpy as np
from shapely.geometry import box
from pyproj import Transformer
from shapely.ops import transform as shp_transform
from pathlib import Path
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
import shapely.ops as ops
from shapely.strtree import STRtree
from multiprocessing import Pool, cpu_count
import warnings
import tempfile
import shutil
from shapely import make_valid
import argparse
from shapely.ops import transform
import fcntl
import time

CANTON_FILE = Path(
    "/mnt/eo-nas1/eoa-share/projects/020_crop1990/data/BFS"
    "/swissboundaries3d_2019-01_2056_5728.gpkg"
)



def snap_coordinates(geom, precision=1.0):
    """
    Snap coordinates to a grid with specified precision (in meters).
    This eliminates floating-point precision issues.
    Also removes Z coordinates if present.
    """
    if geom is None or geom.is_empty:
        return geom
    
    def round_coords(x, y, z=None):
        # Always return 2D coordinates (ignore z)
        x_round = round(x / precision) * precision
        y_round = round(y / precision) * precision
        return (x_round, y_round)
    
    return transform(round_coords, geom)


def clean_geo(gdf, snap_precision=1.0):
    """
    Clean and validate geometries with coordinate snapping.
    """
    gdf["geometry"] = gdf.geometry.apply(make_valid)
    gdf["geometry"] = gdf.geometry.apply(lambda g: snap_coordinates(g, snap_precision))
    gdf["geometry"] = gdf.geometry.buffer(0)  # Clean up after snapping
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    return gdf


def get_file_bounds(file_path):
    """
    Get bounding box of a GeoPackage file without loading all data.
    """
    # Read just one row to get CRS and bounds
    gdf_sample = gpd.read_file(file_path, rows=1)
    crs = gdf_sample.crs
    
    # Get full bounds efficiently using SQL
    import sqlite3
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()
    
    # GeoPackage stores bounds in gpkg_contents table
    cursor.execute("SELECT min_x, min_y, max_x, max_y FROM gpkg_contents LIMIT 1")
    bounds = cursor.fetchone()
    conn.close()
    
    if bounds:
        return bounds, crs
    else:
        # Fallback: read all geometries (slower)
        gdf = gpd.read_file(file_path)
        return gdf.total_bounds, gdf.crs


def create_tiles(bounds, tile_size=5000):
    """Create a grid of tiles covering the given bounds."""
    x_tiles = np.arange(bounds[0], bounds[2], tile_size)
    y_tiles = np.arange(bounds[1], bounds[3], tile_size)
    
    tiles = []
    for i, x in enumerate(x_tiles):
        for j, y in enumerate(y_tiles):
            tile_geom = box(x, y, x + tile_size, y + tile_size)
            tiles.append({
                'tile_id': f"{i}_{j}",
                'geometry': tile_geom,
                'x_idx': i,
                'y_idx': j,
                'bounds': (x, y, x + tile_size, y + tile_size)
            })
    
    return tiles


def extract_polygons(geom):
    """
    Extract only Polygon geometries from any geometry type.
    Handles GeometryCollections, MultiPolygons, etc.
    Filters out LineStrings, Points, etc.
    """
    if geom is None or geom.is_empty:
        return []
    
    if isinstance(geom, Polygon):
        return [geom]
    elif isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    elif isinstance(geom, GeometryCollection):
        # Recursively extract polygons from collection
        polygons = []
        for g in geom.geoms:
            polygons.extend(extract_polygons(g))
        return polygons
    else:
        # LineString, Point, MultiLineString, etc. - ignore
        return []


def remove_overlaps_by_priority_fast(gdf, priority_list, code_col="lnf_code"):
    """
    Remove overlaps with robust geometry handling.
    PRESERVES ALL COLUMNS from input GeoDataFrame.
    """
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()]
    
    # Make all geometries valid using shapely's make_valid (better than buffer(0))
    gdf["geometry"] = gdf.geometry.apply(lambda g: make_valid(g) if not g.is_valid else g)

    result = []
    occupied = []

    for code in priority_list:
        subset = gdf[gdf[code_col] == code]
        if subset.empty:
            continue

        rows_to_keep = []

        if occupied:
            higher_geoms = occupied
            tree = STRtree(higher_geoms)
        else:
            tree = None

        for idx, row in subset.iterrows():
            geom = row.geometry
            if geom.is_empty:
                continue

            try:
                if tree:
                    idxs = tree.query(geom)
                    if idxs.size > 0:
                        local_union = ops.unary_union(
                            [higher_geoms[i] for i in idxs]
                        )
                        geom = geom.difference(local_union)
                        if not geom.is_empty and not geom.is_valid:
                            geom = make_valid(geom)
                
                if geom.is_empty:
                    continue
                
                # Extract only polygons (handles GeometryCollections)
                polygons = extract_polygons(geom)
                
                # For each polygon, preserve ALL row attributes
                for poly in polygons:
                    row_dict = row.to_dict()
                    row_dict['geometry'] = poly
                    rows_to_keep.append(row_dict)
                    occupied.append(poly)
                
            except Exception as e:
                # Skip problematic geometries with topology errors
                print(f"Skipping geometry in code {code}: {e}")
                continue

        if not rows_to_keep:
            continue

        result.append(gpd.GeoDataFrame(rows_to_keep, crs=gdf.crs))

    return (
        gpd.GeoDataFrame(pd.concat(result, ignore_index=True), crs=gdf.crs)
        if result else
        gpd.GeoDataFrame(columns=gdf.columns.tolist(), crs=gdf.crs)
    )


def read_tile_data(file_path, tile_bounds, target_crs):
    """
    Read data from file that intersects with tile bounds.
    Uses GeoPackage spatial index for efficient filtering.
    """
    try:
        # Get file CRS
        sample = gpd.read_file(file_path, rows=1)
        file_crs = sample.crs

        # Reproject bbox to file CRS if needed
        if file_crs != target_crs:
            # Create a box geometry in target CRS
            minx, miny, maxx, maxy = tile_bounds
            tile_box = box(minx, miny, maxx, maxy)
            
            # Reproject to file CRS
            tile_gdf = gpd.GeoDataFrame(geometry=[tile_box], crs=target_crs)
            tile_gdf_file_crs = tile_gdf.to_crs(file_crs)
            
            # Get bounds in file CRS and convert to tuple
            bbox = tuple(tile_gdf_file_crs.total_bounds)
        else:
            bbox = tile_bounds
        
        # Read with spatial filter (uses GeoPackage R-tree index)
        gdf = gpd.read_file(file_path, bbox=bbox)
        
        if gdf.empty:
            return gdf
        
        # Ensure correct CRS
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
        
        return gdf
        
    except Exception as e:
        print(f"Error reading tile data from {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return gpd.GeoDataFrame()


def process_single_tile(args):
    """
    Process a single tile by reading data from files on-demand.
    Writes result to individual tile file.
    """
    (tile_idx, tile_geom, tile_bounds, lnf_path, tlm_path, 
     priority_list, code_col, target_crs, temp_dir,
     snap_precision, lnf_columns) = args
    
    warnings.filterwarnings('ignore')
    
    try:
        # Read LNF data for this tile
        lnf_tile = read_tile_data(lnf_path, tile_bounds, target_crs)
        
        # Read TLM data for this tile
        tlm_tile = read_tile_data(tlm_path, tile_bounds, target_crs)
        
        # If both are empty, skip
        if lnf_tile.empty and tlm_tile.empty:
            return tile_idx, 0, 0.0, None
        
        # Standardize LNF columns
        if not lnf_tile.empty:
            # Convert lnf_code to integer if it's string
            if lnf_tile['lnf_code'].dtype == 'object':
                lnf_tile['lnf_code'] = pd.to_numeric(lnf_tile['lnf_code'], errors='coerce').astype('Int64')
                lnf_tile['lnf_code'] = lnf_tile['lnf_code'].astype(int)
            
            lnf_tile['layer'] = 'LNF'
            lnf_tile['year'] = year
            
            # Ensure all columns exist
            for col in lnf_columns:
                if col not in lnf_tile.columns and col != 'geometry':
                    lnf_tile[col] = None
            # Reorder to match schema
            lnf_tile = lnf_tile[lnf_columns]
        
        # Standardize TLM columns
        if not tlm_tile.empty:
            # Convert lnf_code to integer if needed
            if 'lnf_code' in tlm_tile.columns and tlm_tile['lnf_code'].dtype == 'object':
                tlm_tile['lnf_code'] = pd.to_numeric(tlm_tile['lnf_code'], errors='coerce').astype('Int64')
                tlm_tile['lnf_code'] = tlm_tile['lnf_code'].astype(int)
            
            tlm_tile['layer'] = 'TLM'
            tlm_tile['year'] = year
            
            # Ensure all columns exist
            for col in lnf_columns:
                if col not in tlm_tile.columns and col != 'geometry':
                    tlm_tile[col] = None
            # Reorder to match schema
            tlm_tile = tlm_tile[lnf_columns]
        
        # Combine the data
        if not lnf_tile.empty and not tlm_tile.empty:
            tile_data = pd.concat([lnf_tile, tlm_tile], ignore_index=True)
            tile_data = gpd.GeoDataFrame(tile_data, crs=target_crs)
        elif not lnf_tile.empty:
            tile_data = lnf_tile
        else:
            tile_data = tlm_tile
        
        if tile_data.empty:
            return tile_idx, 0, 0.0, None

        # Clip to tile boundary
        try:
            tile_data["geometry"] = tile_data.geometry.apply(
                lambda g: g.intersection(tile_geom) if g is not None else None
            )
        except Exception as e:
            print(f"Tile {tile_idx}: Error during intersection - {e}")
            tile_data["geometry"] = tile_data.geometry.apply(
                lambda g: make_valid(g).intersection(tile_geom) if g is not None else None
            )
        
        tile_data = clean_geo(tile_data, snap_precision=snap_precision)
        
        if tile_data.empty:
            return tile_idx, 0, 0.0, None
        
        # Run priority algorithm
        result = remove_overlaps_by_priority_fast(tile_data, priority_list, code_col)
        result = clean_geo(result, snap_precision=snap_precision)
        
        if result.empty:
            return tile_idx, 0, 0.0, None
        
        # Final cleanup: ensure only Polygons and MultiPolygons
        rows_to_keep = []
        for idx, row in result.iterrows():
            polygons = extract_polygons(row.geometry)
            if polygons:
                if len(polygons) == 1:
                    row_dict = row.to_dict()
                    row_dict['geometry'] = polygons[0]
                    rows_to_keep.append(row_dict)
                else:
                    row_dict = row.to_dict()
                    for poly in polygons:
                        new_row = row_dict.copy()
                        new_row['geometry'] = poly
                        rows_to_keep.append(new_row)
        
        if not rows_to_keep:
            return tile_idx, 0, 0.0, None
        
        # Reconstruct result preserving ALL columns
        result = gpd.GeoDataFrame(rows_to_keep, crs=target_crs)
        
        if result.empty:
            return tile_idx, 0, 0.0, None
        
        # Write to individual tile file
        tile_output = Path(temp_dir) / f"tile_{tile_idx}.gpkg"
        result.to_file(tile_output, driver='GPKG')
        
        tile_area = result.geometry.area.sum()
        
        return tile_idx, len(result), tile_area, str(tile_output)
        
    except Exception as e:
        print(f"Error processing tile {tile_idx}: {e}")
        import traceback
        traceback.print_exc()
        return tile_idx, 0, 0.0, None


def process_with_tiles_parallel(lnf_path, tlm_path, priority_list, code_col="lnf_code", 
                                tile_size=5000, output_path=None, n_jobs=None,
                                snap_precision=1.0, target_crs="EPSG:2056"):
    """
    Main function to process large datasets with tile-by-tile reading.
    Only reads bounding boxes initially, then reads tile data on-demand.
    """
    if n_jobs is None:
        n_jobs = cpu_count()
    
    print(f"Using {n_jobs} cores")
    print(f"Coordinate snapping: {snap_precision}m precision")
    
    # Get bounds from both files without loading all data
    print("\nReading file bounds...")
    lnf_bounds, lnf_crs = get_file_bounds(lnf_path)
    tlm_bounds, tlm_crs = get_file_bounds(tlm_path)
    
    print(f"LNF bounds: {lnf_bounds}, CRS: {lnf_crs}")
    print(f"TLM bounds: {tlm_bounds}, CRS: {tlm_crs}")
    
    # Reproject bounds to target CRS if needed
    if lnf_crs != target_crs:
        lnf_bounds_gdf = gpd.GeoDataFrame(
            geometry=[box(*lnf_bounds)], crs=lnf_crs
        ).to_crs(target_crs)
        lnf_bounds = lnf_bounds_gdf.total_bounds
    
    if tlm_crs != target_crs:
        tlm_bounds_gdf = gpd.GeoDataFrame(
            geometry=[box(*tlm_bounds)], crs=tlm_crs
        ).to_crs(target_crs)
        tlm_bounds = tlm_bounds_gdf.total_bounds
    
    # Combined bounds
    combined_bounds = (
        min(lnf_bounds[0], tlm_bounds[0]),
        min(lnf_bounds[1], tlm_bounds[1]),
        max(lnf_bounds[2], tlm_bounds[2]),
        max(lnf_bounds[3], tlm_bounds[3])
    )
    
    print(f"\nCombined bounds: {combined_bounds}")
    
    # Read LNF columns schema (read just one row)
    print("Reading LNF schema...")
    lnf_sample = gpd.read_file(lnf_path, rows=1)
    lnf_sample['layer'] = 'LNF'
    lnf_sample['year'] = year
    lnf_columns = lnf_sample.columns.tolist()
    print(f"LNF columns: {lnf_columns}")
    
    # Create tiles
    print(f"\nCreating tiles (size={tile_size}m)...")
    tiles = create_tiles(combined_bounds, tile_size)
    print(f"Created {len(tiles)} tiles")
    
    # Create temp directory in /tmp
    temp_dir = tempfile.mkdtemp(prefix="tiles_", dir="/tmp")
    print(f"Temporary directory: {temp_dir}")
    
    # Prepare tile arguments
    print("Preparing tile arguments...")
    tile_args = []
    for tile in tiles:
        tile_args.append((
            tile['tile_id'],
            tile['geometry'],
            tile['bounds'],
            str(lnf_path),
            str(tlm_path),
            priority_list,
            code_col,
            target_crs,
            str(temp_dir),
            snap_precision,
            lnf_columns
        ))
    
    print(f"Processing {len(tiles)} tiles in parallel...")
    
    total_geometries = 0
    non_empty_tiles = 0
    total_area_from_tiles = 0.0
    tile_files = []
    
    with Pool(processes=n_jobs) as pool:
        for i, (tile_idx, n_geoms, tile_area, tile_file) in enumerate(pool.imap_unordered(process_single_tile, tile_args)):
            if (i + 1) % 100 == 0:
                print(f"Completed {i+1}/{len(tiles)} tiles ({non_empty_tiles} non-empty, {total_geometries} geometries, {total_area_from_tiles:.2f} m² area)")
            
            if n_geoms > 0 and tile_file is not None:
                non_empty_tiles += 1
                total_geometries += n_geoms
                total_area_from_tiles += tile_area
                tile_files.append(tile_file)
    
    print(f"\nProcessing complete!")
    print(f"  Non-empty tiles: {non_empty_tiles}/{len(tiles)}")
    print(f"  Total geometries: {total_geometries}")
    print(f"  Total area: {total_area_from_tiles:.2f} m²")
    print(f"  Tile files created: {len(tile_files)}")
    
    if not tile_files:
        print("No results generated!")
        shutil.rmtree(temp_dir)
        return None
    
    # Merge all tile files (single-threaded)
    print(f"\nMerging {len(tile_files)} tile files...")
    temp_output = Path(temp_dir) / "output.gpkg"
    
    # Read and concatenate all tiles
    print("  Reading tile files...")
    all_tiles = []
    for i, tile_file in enumerate(tile_files):
        if (i + 1) % 100 == 0:
            print(f"  Read {i+1}/{len(tile_files)} tiles")
        all_tiles.append(gpd.read_file(tile_file))
    
    print("  Concatenating all tiles...")
    final_result = gpd.GeoDataFrame(pd.concat(all_tiles, ignore_index=True), crs=target_crs)
    
    print("  Writing merged output...")
    final_result.to_file(temp_output, driver='GPKG')
    
    final_area = final_result.geometry.area.sum()
    print(f"\nMerge complete!")
    print(f"  Final geometries: {len(final_result)}")
    print(f"  Final area: {final_area:.2f} m²")
    
    # Verify geometry types
    print("\nGeometry types in result:")
    print(final_result.geometry.geom_type.value_counts())
    
    if output_path:
        # Prepare target path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove existing file if present
        if output_path.exists():
            print(f"\nRemoving existing file: {output_path}")
            output_path.unlink()
        
        # Also remove any GeoPackage auxiliary files
        for ext in ['-wal', '-shm', '-journal']:
            aux_file = Path(str(output_path) + ext)
            if aux_file.exists():
                aux_file.unlink()
        
        # Copy from temp to final location
        print(f"\nCopying file from {temp_output} to {output_path}...")
        shutil.copy2(str(temp_output), str(output_path))
        print(f"Successfully copied to {output_path}")
    
    print("\nCleaning up temporary directory...")
    shutil.rmtree(temp_dir)
    
    return final_result


# ============= USAGE =============
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine LNF and SwissTLM3D data with priority-based overlap removal')
    
    parser.add_argument('--year', type=str, required=True,
                       help='Year of data to process (e.g., 2020)')
    parser.add_argument('--lnf-path', type=str, default=None,
                       help='Path to LNF data file (default: /mnt/eo-nas1/data/landuse/raw/lnf{year}.gpkg)')
    parser.add_argument('--tlm-path', type=str, default=None,
                       help='Path to TLM data file (default: /mnt/eo-nas1/eoa-share/projects/020_crop1990/data/SwissTopo/swissTLM3D/swissTLM3D_{year}.gpkg)')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Output file path (default: /mnt/eo-nas1/eoa-share/projects/020_crop1990/data/GTs/LNF_swissTLM3D_{year}.gpkg)')
    parser.add_argument('--tile-size', type=int, default=5000,
                       help='Tile size in meters (default: 5000)')
    parser.add_argument('--n-jobs', type=int, default=32,
                       help='Number of parallel jobs (default: 32)')
    parser.add_argument('--snap-precision', type=float, default=1.0,
                       help='Coordinate snapping precision in meters (default: 1.0)')
    
    args = parser.parse_args()
    
    year = args.year
    
    # Set default paths if not provided
    lnf_path = args.lnf_path or f"/mnt/eo-nas1/data/landuse/raw/lnf{year}.gpkg"
    tlm_path = args.tlm_path or f"/mnt/eo-nas1/eoa-share/projects/020_crop1990/data/SwissTopo/swissTLM3D/swissTLM3D_{year}.gpkg"
    year_tlm = "2020" if year == "2019" else year
    tlm_path_actual = tlm_path.replace(year, year_tlm)
    output_path = args.output_path or f"/mnt/eo-nas1/eoa-share/projects/020_crop1990/data/GTs/LNF_swissTLM3D_{year}.gpkg"
    
    print(f"Configuration:")
    print(f"  Year: {year}")
    print(f"  LNF path: {lnf_path}")
    print(f"  TLM path: {tlm_path_actual}")
    print(f"  Output path: {output_path}")
    print(f"  Tile size: {args.tile_size}m")
    print(f"  Parallel jobs: {args.n_jobs}")
    print(f"  Snap precision: {args.snap_precision}m")
    print()
    
    # Read just one row from LNF to get lnf_codes
    print("Reading LNF metadata...")
    lnf_sample = gpd.read_file(lnf_path, rows=1)
    
    # Convert lnf_code to integer if needed
    if 'lnf_code' not in lnf_sample.columns:
        print("ERROR: LNF data doesn't have 'lnf_code' column.")
        exit(1)
    
    # Get unique LNF codes by reading full dataset (only lnf_code column)
    print("Reading unique LNF codes...")
    lnf_codes_df = gpd.read_file(lnf_path, columns=['lnf_code'])
    
    # Convert to integer if needed
    if lnf_codes_df['lnf_code'].dtype == 'object':
        print("Converting LNF lnf_code from string to integer...")
        lnf_codes_df['lnf_code'] = pd.to_numeric(lnf_codes_df['lnf_code'], errors='coerce').astype('Int64')
    lnf_codes = sorted([int(x) for x in lnf_codes_df['lnf_code'].dropna().unique()])
    print(f"LNF codes found: {lnf_codes}")
        
    # Split LNF codes: all except 'Sömmerungsweiden' (930) and 'Übrige Flächen ausserhalb der LN und SF' (998)
    lnf_codes_high = [int(code) for code in lnf_codes if code not in [930, 998]]
    lnf_code_SW = [930] if 930 in lnf_codes else []
    lnf_code_UF = [998] if 998 in lnf_codes else []
    
    # Define priority order
    tlm_codes = [
        1915,  # Wald
        1914,  # Wald offen
        1913,  # Gebueschwald
        1912,  # Gehoelzflaeche
        1909,  # Stehende Gewaesser
        1910,  # Fliessgewaesser
        1911,  # Feuchtgebiet
        1907,  # Gletscher
        1908,  # Schneefeld Toteis
        1928,  # Rock extraction areas
        1906,  # Lockergestein locker
        1905,  # Lockergestein
        1904,  # Felsbloecke locker
        1903,  # Felsbloecke
        1902,  # Fels locker
        1901,  # Fels
        1923,  # Buildings
        1921,  # Streets
        1922,  # Railways
        1930,  # Traffic areas
        1926,  # Transport facilities
        1925,  # Dams
        1924,  # Sport facilities
        1927,  # Built areas
        1929,  # Allotments
    ]
    
    # Build final priority list
    priority_list = lnf_codes_high + tlm_codes + lnf_code_SW + lnf_code_UF
    
    print(f"\nPriority order (first = highest priority):")
    print(f"  1. LNF codes (except 930, 998): {lnf_codes_high}")
    print(f"  2. TLM codes: {tlm_codes}")
    print(f"  3. LNF 930 (Sömmerungsweiden): {lnf_code_SW}")
    print(f"  4. Lowest priority - LNF 998 (Übrige Flächen): {lnf_code_UF}")
    print(f"\nTotal priority levels: {len(priority_list)}")
    
    # Run processing
    result = process_with_tiles_parallel(
        lnf_path=lnf_path,
        tlm_path=tlm_path_actual,
        priority_list=priority_list,
        code_col="lnf_code",
        tile_size=args.tile_size,
        output_path=output_path,
        n_jobs=args.n_jobs,
        snap_precision=args.snap_precision
    )
    
    if result is not None:
        print(f"\nFinal result: {len(result)} geometries")
        print(f"Final columns: {result.columns.tolist()}")

        # Show breakdown by source
        if 'layer' in result.columns:
            result_lnf = result[result['layer'] == 'LNF']
            result_lnf_area = result_lnf.geometry.area.sum()
            result_tlm = result[result['layer'] != 'LNF']
            result_tlm_area = result_tlm.geometry.area.sum()
            print(f"  From LNF: {len(result_lnf)} ({100 * (len(result_lnf))/(len(result_lnf)+len(result_tlm)):.2f}%) geometries")
            print(f"  From LNF: {result_lnf_area} ({100 * (result_lnf_area)/(result_lnf_area+result_tlm_area):.2f}%) m²")
            print(f"  From TLM: {len(result_tlm)} ({100 * (len(result_tlm))/(len(result_lnf)+len(result_tlm)):.2f}%) geometries")
            print(f"  From TLM: {result_tlm_area} ({100 * (result_tlm_area)/(result_lnf_area+result_tlm_area):.2f}%) m²")
        else:
            print("Warning: 'layer' column not found in result")

        if output_path:
            # Base file already written safely above.
            # Add canton columns to the in-memory result, write to tmp, then replace.
            print("\nAdding canton columns …")
            canton_bnd = gpd.read_file(CANTON_FILE, layer="tlm_kantonsgebiet")[["name", "geometry"]]
            result = result.rename(columns={"kanton": "canton_lnf"})
            cents = result[["geometry"]].copy()
            cents["geometry"] = cents.geometry.centroid
            joined = gpd.sjoin(cents, canton_bnd[["name", "geometry"]], how="left", predicate="within")
            joined = joined[~joined.index.duplicated(keep="first")]
            result["canton"] = joined["name"]
            tmp = Path(output_path).parent / (Path(output_path).stem + "_canton_tmp.gpkg")
            result.to_file(tmp, driver="GPKG", layer=Path(output_path).stem)
            shutil.move(str(tmp), str(output_path))
            print(f"  Canton columns added → {Path(output_path).name}")
    else:
        print("\nNo results generated!")

