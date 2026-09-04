#!/usr/bin/env python3
"""
Check for and fix corrupted geometries in polygon files.

Usage:
    python check_fix_geometries.py <input_file> [--fix] [--output <output_file>]

Example:
    # Check only (no changes):
    python check_fix_geometries.py /path/to/polygons.gpkg

    # Check and fix in-place:
    python check_fix_geometries.py /path/to/polygons.gpkg --fix

    # Check and save fixed version to new file:
    python check_fix_geometries.py /path/to/polygons.gpkg --fix --output /path/to/fixed.gpkg
"""

import argparse
import geopandas as gpd
from shapely.validation import explain_validity
import warnings
warnings.filterwarnings('ignore')


def check_and_fix_geometries(input_file, fix=False, output_file=None):
    """
    Check for corrupted/invalid geometries and optionally fix them.

    Args:
        input_file: Path to input polygon file
        fix: If True, attempt to fix invalid geometries
        output_file: Path to save fixed file (if None, overwrites input when fix=True)
    """
    print(f"Loading polygon file: {input_file}")
    gdf = gpd.read_file(input_file)

    print(f"Total features: {len(gdf)}")
    print(f"CRS: {gdf.crs}")
    print(f"Geometry type: {gdf.geom_type.unique()}")

    # Check for None/null geometries
    null_geoms = gdf.geometry.isna()
    if null_geoms.any():
        print(f"\n⚠️  WARNING: Found {null_geoms.sum()} features with NULL geometries")
        print(f"   Indices: {gdf[null_geoms].index.tolist()[:10]}...")
        if fix:
            print(f"   → Removing {null_geoms.sum()} NULL geometries")
            gdf = gdf[~null_geoms].copy()

    # Check for empty geometries
    empty_geoms = gdf.geometry.is_empty
    if empty_geoms.any():
        print(f"\n⚠️  WARNING: Found {empty_geoms.sum()} empty geometries")
        print(f"   Indices: {gdf[empty_geoms].index.tolist()[:10]}...")
        if fix:
            print(f"   → Removing {empty_geoms.sum()} empty geometries")
            gdf = gdf[~empty_geoms].copy()

    # Check for invalid geometries
    print("\nChecking geometry validity...")
    invalid_mask = ~gdf.geometry.is_valid

    if invalid_mask.any():
        num_invalid = invalid_mask.sum()
        print(f"\n❌ Found {num_invalid} invalid geometries!")

        # Show examples of what's wrong
        invalid_gdf = gdf[invalid_mask].head(5)
        print("\nExamples of invalid geometries:")
        for idx, row in invalid_gdf.iterrows():
            reason = explain_validity(row.geometry)
            print(f"  Index {idx}: {reason}")

        if fix:
            print(f"\n🔧 Attempting to fix {num_invalid} invalid geometries...")

            # Try to fix using buffer(0) trick
            def fix_geometry(geom):
                if geom is None or geom.is_empty:
                    return None
                if not geom.is_valid:
                    try:
                        # buffer(0) often fixes self-intersections and other issues
                        fixed = geom.buffer(0)
                        if fixed.is_valid:
                            return fixed
                        # If buffer(0) didn't work, try make_valid (requires shapely >= 1.8)
                        try:
                            from shapely import make_valid
                            return make_valid(geom)
                        except ImportError:
                            return fixed
                    except Exception as e:
                        print(f"     Failed to fix geometry: {e}")
                        return None
                return geom

            # Apply fixes
            gdf['geometry'] = gdf['geometry'].apply(fix_geometry)

            # Remove any that couldn't be fixed
            still_invalid = ~gdf.geometry.is_valid
            could_not_fix = (still_invalid | gdf.geometry.isna()).sum()

            if could_not_fix > 0:
                print(f"   ⚠️  Could not fix {could_not_fix} geometries - removing them")
                gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.isna()].copy()

            fixed_count = num_invalid - could_not_fix
            print(f"   ✅ Fixed {fixed_count} geometries")
            print(f"   ✅ Total valid geometries: {len(gdf)}")
    else:
        print("✅ All geometries are valid!")

    # Check for very small/degenerate geometries
    print("\nChecking for tiny/degenerate geometries...")
    areas = gdf.geometry.area
    very_small = areas < 1e-10  # Essentially zero area
    if very_small.any():
        print(f"⚠️  WARNING: Found {very_small.sum()} geometries with area < 1e-10")
        if fix:
            print(f"   → Removing {very_small.sum()} degenerate geometries")
            gdf = gdf[~very_small].copy()

    # Final validation
    print("\n" + "="*60)
    print("FINAL SUMMARY:")
    print("="*60)
    print(f"Total features: {len(gdf)}")
    print(f"Valid geometries: {gdf.geometry.is_valid.sum()}")
    print(f"Invalid geometries: {(~gdf.geometry.is_valid).sum()}")
    print(f"NULL geometries: {gdf.geometry.isna().sum()}")
    print(f"Empty geometries: {gdf.geometry.is_empty.sum()}")

    if fix:
        # Save the fixed file
        output_path = output_file if output_file else input_file
        print(f"\n💾 Saving fixed geometries to: {output_path}")
        gdf.to_file(output_path, driver="GPKG")
        print("✅ Done!")

        # Verify the saved file
        print("\nVerifying saved file...")
        test_gdf = gpd.read_file(output_path)
        test_invalid = (~test_gdf.geometry.is_valid).sum()
        if test_invalid == 0:
            print("✅ Verification successful - all geometries valid in saved file")
        else:
            print(f"⚠️  WARNING: {test_invalid} invalid geometries still present in saved file")

    return gdf


def main():
    parser = argparse.ArgumentParser(
        description="Check for and fix corrupted geometries in polygon files",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_file", help="Path to input polygon file (gpkg, shp, geojson, etc.)")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix invalid geometries")
    parser.add_argument("--output", help="Output file path (if not specified, overwrites input when --fix is used)")

    args = parser.parse_args()

    check_and_fix_geometries(args.input_file, fix=args.fix, output_file=args.output)


if __name__ == "__main__":
    main()
