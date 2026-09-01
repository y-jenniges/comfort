"""Water-mass classification, distance to coast and province assignment
from ``comfort.geo``.

Requires:
    COMFORT_DB_PATH - path to the COMFORT SQLite database

Optional:
    LONGHURST_SHP_PATH - path to the Longhurst provinces shapefile (marineregions.org)
    JENNIGES_CSV_PATH  - path to cluster_set.csv (Zenodo record 15201767)

Install extras:
    pip install "comfort-db[geo]"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import comfort
from comfort.geo import (
    classify_from_grid,
    distance_to_coast,
    load_jenniges_provinces,
    load_longhurst,
    water_mass_masks,
    water_mass_statistics,
)

load_dotenv()

if __name__ == "__main__":
    # Get DB path from .env
    db_path = os.environ.get("COMFORT_DB_PATH", "")
    if not db_path:
        sys.exit("COMFORT_DB_PATH is not set. Point it to your COMFORT SQLite database.")

    # Load nitrate with spatial metadata
    print("Loading NITRATE (QC_GOOD, North Atlantic)...")
    with comfort.connect(db_path) as conn:
        dfs = comfort.load_comfort(
            conn, parameters=["NITRATE"],
            quality_flags=comfort.QC_GOOD,
            lat_min=20, lat_max=65, lon_min=-80, lon_max=0,
            depth_max=100,
            as_xarray=False,
            convert_units=True,
        )
    df = dfs["NITRATE"]
    print(f"  {len(df)} rows, {df['PROFILE_NUMBER'].nunique()} profiles\n")

    # --- Water-mass classification with user-defined regions ---
    print("=== water_mass_masks (user-defined regions) ===")
    regions = {
        "Subpolar": [(-60, 45), (0, 45), (0, 65), (-60, 65)],
        "Subtropical": [(-80, 20), (0, 20), (0, 44), (-80, 44)],
    }
    df_m = water_mass_masks(df, regions)
    print(df_m["region"].value_counts().to_string(), "\n")

    # Per-region statistics
    print("=== water_mass_statistics ===")
    stats = water_mass_statistics(df_m, param_cols=["NITRATE"])
    print(stats.to_string(), "\n")

    # --- Distance to coast ---
    print("=== distance_to_coast ===")
    dist = distance_to_coast(df)
    df_d = df.copy()
    df_d["dist_km"] = dist.round(1)
    print(df_d[["LATITUDE", "LONGITUDE", "dist_km"]]
          .drop_duplicates(subset=["LATITUDE", "LONGITUDE"])
          .head(10).to_string(index=False))

    # --- Longhurst provinces (requires shapefile from marineregions.org) ---
    print("\n=== Longhurst provinces ===")
    longhurst_path = os.environ.get("LONGHURST_SHP_PATH")
    if longhurst_path and Path(longhurst_path).exists():
        gdf = load_longhurst(longhurst_path)
        print(f"Loaded {len(gdf)} Longhurst provinces")
        print(gdf[["province_code", "province_name"]].head(10)
              .to_string(index=False), "\n")

        df_lh = water_mass_masks(df, gdf, region_name_col="province_code")
        print(df_lh["region"].value_counts().head(10).to_string())
    else:
        print("Skipped - set LONGHURST_SHP_PATH to the Longhurst .shp file to run.")

    # --- Jenniges provinces (requires cluster_set.csv from Zenodo 15201767) ---
    print("\n=== Jenniges provinces ===")
    jenniges_path = os.environ.get("JENNIGES_CSV_PATH")
    if jenniges_path and Path(jenniges_path).exists():
        grid = load_jenniges_provinces(jenniges_path, depth=0)
        print(f"Loaded {len(grid)} grid cells at depth "
              f"{grid['LEV_M'].iloc[0]:.0f} m\n")

        df_j = classify_from_grid(df, grid)
        print(df_j["label"].value_counts().head(10).to_string())
    else:
        print("Skipped - set JENNIGES_CSV_PATH to cluster_set.csv to run.")
