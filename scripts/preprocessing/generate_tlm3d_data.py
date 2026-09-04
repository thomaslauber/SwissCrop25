import geopandas as gpd
from pathlib import Path
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, box
import pandas as pd
from shapely.strtree import STRtree
import shapely.ops as ops
from shapely.geometry import GeometryCollection
from shapely import make_valid
import numpy as np
import shapely
import networkx as nx
from multiprocessing import Pool, cpu_count
import argparse

FILE_PATH_DIR = {
    "2020": "swisstlm3d_2020-03_2056_5728.gdb/2020_SWISSTLM3D_FGDB101_CHLV95_LN02/SWISSTLM3D_CHLV95_LN02.gdb",
    "2021": "swisstlm3d_2021-04_2056_5728.gdb/2021_SWISSTLM3D_FGDB101_CHLV95_LN02/SWISSTLM3D_2021_LV95_LN02.gdb",
    "2022": "swisstlm3d_2022-03_2056_5728.gdb/SWISSTLM3D_2022_LV95_LN02.gdb",
    "2023": "swisstlm3d_2023-03_2056_5728.gpkg/SWISSTLM3D_2023_LV95_LN02.gpkg",
    "2024": "swisstlm3d_2024-03_2056_5728.gpkg/SWISSTLM3D_2024_LV95_LN02.gpkg",
    "2025": "swisstlm3d_2025-03_2056_5728.gpkg/SWISSTLM3D_2025.gpkg"
}

DEFAULT_INPUT_BASE = "/mnt/eo-nas1/eoa-share/projects/020_crop1990/data/SwissTopo/swissTLM3D/raw"
DEFAULT_OUTPUT_DIR = "/mnt/eo-nas1/eoa-share/projects/020_crop1990/data/SwissTopo/swissTLM3D"


def clean_geo(gdf):
    gdf["geometry"] = gdf.geometry.apply(make_valid)
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    return gdf

def lines_to_polygons(gdf, width_column):
    """Convert LineString or MultiLineString geometries to polygons using buffer."""
    new_geoms = []
    for geom, width in zip(gdf.geometry, gdf[width_column]):
        if geom is None or pd.isna(width):
            new_geoms.append(None)
            continue
        if isinstance(geom, (MultiLineString, LineString)):
            buffered = geom.buffer(width / 2)
            new_geoms.append(buffered)
        elif isinstance(geom, (Polygon, MultiPolygon)):
            new_geoms.append(geom)
        else:
            new_geoms.append(geom)

    gdf = gdf.copy()
    gdf["geometry"] = new_geoms
    return gdf

def create_tiles(gdf, tile_size=5000):
    """Create a grid of tiles covering the data extent."""
    bounds = gdf.total_bounds
    x_tiles = np.arange(bounds[0], bounds[2], tile_size)
    y_tiles = np.arange(bounds[1], bounds[3], tile_size)

    tiles = []
    for i, x in enumerate(x_tiles):
        for j, y in enumerate(y_tiles):
            tile_geom = box(x, y, x + tile_size, y + tile_size)
            tiles.append({
                'tile_id': f"{i}_{j}",
                'geometry': tile_geom
            })

    return gpd.GeoDataFrame(tiles, crs=gdf.crs)

def union_tile(args):
    """
    Union all geometries within a single tile robustly.

    Args:
        args: (tile_geom, gdf_subset, crs)
            - tile_geom: shapely Polygon representing the tile
            - gdf_subset: GeoDataFrame of geometries potentially intersecting the tile
            - crs: coordinate reference system (not used but kept for signature)

    Returns:
        shapely Geometry (unioned) or None
    """
    tile_geom, gdf_subset, crs = args

    try:
        tile_data = gdf_subset[gdf_subset.geometry.intersects(tile_geom)].copy()
        if tile_data.empty:
            return None

        clipped = [geom.intersection(tile_geom) for geom in tile_data.geometry]
        clipped = [make_valid(geom).buffer(0) for geom in clipped
                   if geom is not None and not geom.is_empty]

        if not clipped:
            return None

        unioned = shapely.ops.unary_union(clipped)
        unioned = unioned.intersection(tile_geom)

        if unioned.is_empty:
            return None

        return unioned

    except Exception as e:
        print(f"Error in tile: {e}")
        return None

def union_overlapping_tiled(gdf, tile_size=5000, n_jobs=None):
    """
    Union overlapping geometries using a tiled approach.
    Much faster than graph-based approach, but may split features at tile boundaries.
    """
    if n_jobs is None:
        n_jobs = cpu_count()

    print(f"    > Creating tiles (size={tile_size}m)...")
    tiles = create_tiles(gdf, tile_size)
    print(f"    > Created {len(tiles)} tiles")

    gdf.sindex

    tile_args = []
    for idx, tile_row in tiles.iterrows():
        candidates_idx = gdf.sindex.query(tile_row.geometry, predicate='intersects')
        if len(candidates_idx) > 0:
            gdf_subset = gdf.iloc[candidates_idx]
            tile_args.append((tile_row.geometry, gdf_subset, gdf.crs))

    print(f"    > Unioning {len(tile_args)} non-empty tiles using {n_jobs} cores...")

    with Pool(processes=n_jobs) as pool:
        results = pool.map(union_tile, tile_args)

    new_geoms = [g for g in results if g is not None]

    print(f"    > Result: {len(new_geoms)} geometries")
    return gpd.GeoDataFrame(geometry=new_geoms, crs=gdf.crs)


def main(year, file_path, output_path):
    # Land cover
    layer = "tlm_bb_bodenbedeckung" if year >= "2023" else "TLM_BODENBEDECKUNG"
    print(f"Processing '{layer}'")
    landcover = gpd.read_file(file_path, layer=layer)
    if year < "2023":
        landcover.rename(columns={'OBJEKTART': 'objektart'}, inplace=True)
        mappingToName = {
            1: "Fels",
            2: "Fels locker",
            3: "Felsbloecke",
            4: "Felsbloecke locker",
            5: "Fliessgewaesser",
            6: "Gebueschwald",
            7: "Lockergestein",
            8: "Lockergestein locker",
            9: "Gletscher",
            10: "Stehende Gewaesser",
            11: "Feuchtgebiet",
            12: "Wald",
            13: "Wald offen",
            14: "Gehoelzflaeche",
            15: "Schneefeld Toteis"
        }
        landcover["objektart"] = landcover["objektart"].map(mappingToName)
    mapping = {
        'Fels': 1901,
        'Fels locker': 1902,
        'Felsbloecke': 1903,
        'Felsbloecke locker': 1904,
        'Lockergestein': 1905,
        'Lockergestein locker': 1906,
        'Gletscher': 1907,
        'Schneefeld Toteis': 1908,
        'Stehende Gewaesser': 1909,
        'Fliessgewaesser': 1910,
        'Feuchtgebiet': 1911,
        'Gehoelzflaeche': 1912,
        'Gebueschwald': 1913,
        'Wald offen': 1914,
        'Wald': 1915
    }
    landcover["lnf_code"] = landcover["objektart"].map(mapping)
    landcover["layer"] = layer
    landcover = landcover[["lnf_code", "layer", "geometry"]]

    # Streets
    layer = "tlm_strassen_strasse" if year >= "2023" else "TLM_STRASSE"
    print(f"Processing '{layer}'")
    streets = gpd.read_file(file_path, layer=layer)
    if year < "2023":
        streets.rename(columns={'OBJEKTART': 'objektart', 'KUNSTBAUTE': 'kunstbaute'}, inplace=True)
        mappingToName = {
            0: "Ausfahrt",
            1: "Einfahrt",
            2: "Autobahn",
            5: "Zufahrt",
            6: "Dienstzufahrt",
            8: "10m Strasse",
            9: "6m Strasse",
            10: "4m Strasse",
            11: "3m Strasse",
            15: "2m Weg",
            16: "1m Weg",
            17: "1m Wegfragment",
            18: "2m Wegfragment",
            20: "8m Strasse",
            21: "Autostrasse"
        }
        streets["objektart"] = streets["objektart"].map(mappingToName)
        streets = streets[~streets["objektart"].isna()]
        mappingToName = {
            100: "Keine",
            200: "Bruecke",
            300: "Bruecke mit Galerie",
            400: "Gedeckte Bruecke",
            450: "Bruecke mit Treppe",
            500: "Staudamm",
            600: "Steg",
            700: "Galerie",
            800: "Staumauer, Wehr",
            900: "Treppe",
            1000: "Tunnel",
            1100: "Unterfuehrung",
            1200: "Unterfuehrung mit Treppe",
            1300: "Furt",
            1400: "in/auf Gebaeude",
            999997: "ub",
            999998: "k_W"
        }
        streets["kunstbaute"] = streets["kunstbaute"].map(mappingToName)
    else:
        streets = streets[streets["belagsart"] != "k_W"]
    streets = streets[~streets["kunstbaute"].isin(['Tunnel', 'Galerie', 'Unterfuehrung', 'Unterfuehrung mit Treppe', 'in/auf Gebaeude', 'Furt'])]
    widths = {
        '10m Strasse': 10,
        '1m Weg': 1,
        '1m Wegfragment': 1,
        '2m Weg': 2,
        '2m Wegfragment': 2,
        '3m Strasse': 3,
        '4m Strasse': 4,
        '6m Strasse': 6,
        '8m Strasse': 8,
        'Ausfahrt': 5,
        'Autobahn': 30,
        'Autostrasse': 15,
        'Dienstzufahrt': 5,
        'Einfahrt': 5,
        'Zufahrt': 5
    }
    streets["widths"] = streets["objektart"].map(widths)
    streets = lines_to_polygons(streets, "widths")
    print("    > union overlapping polygons")
    streets = union_overlapping_tiled(streets, tile_size=5000, n_jobs=32)
    streets["lnf_code"] = 1921
    streets["layer"] = layer
    streets = streets[["lnf_code", "layer", "geometry"]]

    # Railways
    layer = "tlm_oev_eisenbahn" if year >= "2023" else "TLM_EISENBAHN"
    print(f"Processing '{layer}'")
    railways = gpd.read_file(file_path, layer=layer)
    if year < "2023":
        railways.rename(columns={'KUNSTBAUTE': 'kunstbaute', 'ANZAHL_SPUREN':'anzahl_spuren'}, inplace=True)
        mappingToName = {
            100: "Keine",
            200: "Bruecke",
            300: "Bruecke mit Galerie",
            400: "Galerie",
            500: "Gedeckte Bruecke",
            600: "Staudamm",
            700: "Staumauer / Wehr",
            800: "Tunnel",
            900: "Unterfuehrung",
            1000: "in/auf Gebaeude",
            999997: "ub",
            999998: "k_W"
        }
        railways["kunstbaute"] = railways["kunstbaute"].map(mappingToName)
    railways = railways[~railways["kunstbaute"].isin(['Tunnel', 'Galerie', 'Unterfuehrung', 'in/auf Gebaeude'])]
    railways["widths"] = railways["anzahl_spuren"].apply(lambda nr: 13 if nr=="2" else 8)
    railways = lines_to_polygons(railways, "widths")
    print("    > union overlapping polygons")
    railways = union_overlapping_tiled(railways, tile_size=5000, n_jobs=32)
    railways["lnf_code"] = 1922
    railways["layer"] = layer
    railways = railways[["lnf_code", "layer", "geometry"]]

    # Buildings
    layer = "tlm_bauten_gebaeude_footprint" if year >= "2023" else "TLM_GEBAEUDE_FOOTPRINT"
    print(f"Processing '{layer}'")
    buildings = gpd.read_file(file_path, layer=layer)
    if year < "2023":
        buildings.rename(columns={'OBJEKTART': 'objektart'}, inplace=True)
        mappingToName = {
            0: "Gebaeude",
            2: "Hochhaus",
            3: "Hochkamin",
            4: "Turm",
            5: "Kuehlturm",
            6: "Lagertank",
            7: "Lueftungsschacht",
            8: "Offenes Gebaeude",
            9: "Treibhaus",
            10: "Im Bau",
            11: "Kapelle",
            12: "Sakraler Turm",
            13: "Sakrales Gebaeude",
            15: "Flugdach",
            16: "Unterirdisches Gebaeude",
            17: "Mauer gross",
            18: "Mauer gross gedeckt",
            19: "Historische Baute",
            22: "Verbindungsbruecke",
            24: "Einhausung"
        }
        buildings["objektart"] = buildings["objektart"].map(mappingToName)
    buildings = buildings[~buildings["objektart"].isin(['Unterirdisches Gebaeude', 'Treibhaus'])]
    buildings["lnf_code"] = 1923
    buildings["layer"] = layer
    buildings = buildings[["lnf_code", "layer", "geometry"]]

    # Sport facilities
    layer = "tlm_bauten_sportbaute_ply" if year >= "2023" else "TLM_SPORTBAUTE_PLY"
    print(f"Processing '{layer}'")
    sport = gpd.read_file(file_path, layer=layer)
    sport["lnf_code"] = 1924
    sport["layer"] = layer
    sport = sport[["lnf_code", "layer", "geometry"]]

    # Dams and weirs
    layer = "tlm_bauten_staubaute" if year >= "2023" else "TLM_STAUBAUTE"
    print(f"Processing '{layer}'")
    dams = gpd.read_file(file_path, layer=layer)
    if year < "2023":
        dams.rename(columns={'OBJEKTART': 'objektart'}, inplace=True)
        mappingToName = {
            0: "Staumauer",
            1: "Staudamm",
            2: "Wasserbecken",
            3: "Wehr",
            4: "Schutzdamm"
        }
        dams["objektart"] = dams["objektart"].map(mappingToName)
    dams = dams[~dams["objektart"].isin(['Schutzdamm'])]
    dams["lnf_code"] = 1925
    dams["layer"] = layer
    dams = dams[["lnf_code", "layer", "geometry"]]

    # Transport facilities
    layer = "tlm_bauten_verkehrsbaute_ply" if year >= "2023" else "TLM_VERKEHRSBAUTE_PLY"
    print(f"Processing '{layer}'")
    transport = gpd.read_file(file_path, layer=layer)
    transport["lnf_code"] = 1926
    transport["layer"] = layer
    transport = transport[["lnf_code", "layer", "geometry"]]

    # Special use areas
    layer = "tlm_areale_nutzungsareal" if year >= "2023" else "TLM_NUTZUNGSAREAL"
    print(f"Processing '{layer}'")
    areas = gpd.read_file(file_path, layer=layer)
    if year < "2023":
        areas.rename(columns={'OBJEKTART': 'objektart'}, inplace=True)
        mappingToName = {
            0: "Abwasserreinigungsareal",
            2: "Antennenareal",
            3: "Baumschule",
            4: "Deponieareal",
            5: "Kraftwerkareal",
            6: "Friedhof",
            7: "Historisches Areal",
            9: "Kehrichtverbrennungsareal",
            10: "Abbauareal",
            11: "Klosterareal",
            13: "Massnahmenvollzugsanstaltsareal",
            14: "Messeareal",
            15: "Obstanlage",
            16: "Oeffentliches Parkareal",
            17: "Reben",
            18: "Schrebergartenareal",
            19: "Schul- und Hochschulareal",
            21: "Spitalareal",
            23: "Unterwerkareal",
            24: "Wald nicht bestockt",
            25: "Truppenuebungsplatz"
        }
        areas["objektart"] = areas["objektart"].map(mappingToName)
    built_areas = areas[areas["objektart"].isin(['Abwasserreinigungsareal', 'Antennenareal', 'Deponieareal', 'Kehrichtverbrennungsareal',
        'Kraftwerkareal', 'Massnahmenvollzugsanstaltsareal', 'Messeareal', 'Unterwerkareal'])].copy()
    rock_areas = areas[areas["objektart"].isin(['Kiesabbauareal', 'Lehmabbauareal', 'Steinbruchareal'])].copy()
    allotments = areas[areas["objektart"].isin(['Schrebergartenareal'])].copy()
    built_areas["lnf_code"] = 1927
    rock_areas["lnf_code"] = 1928
    allotments["lnf_code"] = 1929
    areas = pd.concat([built_areas, rock_areas, allotments])
    areas["layer"] = layer
    areas = areas[["lnf_code", "layer", "geometry"]]

    # Traffic areas
    layer = "tlm_areale_verkehrsareal" if year >= "2023" else "TLM_VERKEHRSAREAL"
    print(f"Processing '{layer}'")
    traffic = gpd.read_file(file_path, layer=layer)
    if year < "2023":
        traffic.rename(columns={'OBJEKTART': 'objektart'}, inplace=True)
        mappingToName = {
            0: "Flughafenareal",
            1: "Flugplatzareal",
            2: "Flugfeldareal",
            3: "Gleisareal",
            4: "Heliport",
            5: "Oeffentliches Parkplatzareal",
            6: "Rastplatzareal",
            7: "Privates Fahrareal",
            8: "Verkehrsflaeche",
            10: "Privates Parkplatzareal"
        }
        traffic["objektart"] = traffic["objektart"].map(mappingToName)
    traffic = traffic[traffic["objektart"].isin(['Gleisareal', 'Oeffentliches Parkplatzareal', 'Privates Fahrareal', 'Privates Parkplatzareal', 'Rastplatzareal', 'Verkehrsflaeche'])]
    traffic["lnf_code"] = 1930
    traffic["layer"] = layer
    traffic = traffic[["lnf_code", "layer", "geometry"]]

    # Combine all layers and save
    print("\nCombining all layers...")
    gdf = pd.concat([
        landcover,
        streets,
        railways,
        buildings,
        sport,
        dams,
        transport,
        areas,
        traffic
    ], ignore_index=True)
    gdf["year"] = year

    print(f"Total geometries: {len(gdf)}")
    print(f"Columns: {gdf.columns.tolist()}")
    print("\nSaving to file...")
    gdf.to_file(output_path)
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process SwissTLM3D data for a specific year')
    parser.add_argument('--year', type=str, required=True,
                        choices=list(FILE_PATH_DIR.keys()),
                        help='Year of SwissTLM3D data to process')
    parser.add_argument('--input-base-dir', type=str, default=DEFAULT_INPUT_BASE,
                        help=f'Base directory containing raw TLM3D files (default: {DEFAULT_INPUT_BASE})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory for the processed GeoPackage (default: {DEFAULT_OUTPUT_DIR})')
    args = parser.parse_args()

    year = args.year
    file_path = str(Path(args.input_base_dir) / FILE_PATH_DIR[year])
    output_path = str(Path(args.output_dir) / f"swissTLM3D_{year}.gpkg")

    main(year, file_path, output_path)
