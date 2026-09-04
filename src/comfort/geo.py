"""Geospatial utility functions for COMFORT data."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
import gsw
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd


# Mean Earth radius for haversine distance calculations
_EARTH_RADIUS_KM = 6371.0


def distance_to_coast(df: pd.DataFrame, lat_col: str = "LATITUDE",
                      lon_col: str = "LONGITUDE", resolution: str = "110m",
                      coastline_geom: object = None) -> pd.Series:
    """Approximates the distance of each observation to the nearest coastline.

    Uses Natural Earth coastlines (cached via cartopy) unless a custom
    geometry is provided. Coastline vertices are indexed with a KDTree
    for fast approximate nearest-neighbour search, followed by exact
    haversine distance for the top-5 candidates. Fully vectorised -
    suitable for large DataFrames.

    Args:
        df (pandas.DataFrame): DataFrame with latitude and longitude columns.
        lat_col (str): Latitude column [°N].
        lon_col (str): Longitude column [°E].
        resolution (str): Natural Earth resolution (``'110m'``, ``'50m'``
            or ``'10m'``). Ignored when ``coastline_geom`` is provided.
        coastline_geom: Pre-loaded coastline as a shapely geometry, list
            of geometries or geopandas GeoSeries/GeoDataFrame.
    Returns:
        pandas.Series: Distance to nearest coastline [km], same index as df.
    """
    from scipy.spatial import cKDTree

    # Load coastline vertices
    coast_pts = _get_coast_points(resolution, coastline_geom)
    if coast_pts is None or len(coast_pts) == 0:
        logging.error("distance_to_coast: no coastline points could be loaded")
        return pd.Series(np.nan, index=df.index)

    # Convert to radians for haversine
    coast_lon_r = np.radians(coast_pts[:, 0])
    coast_lat_r = np.radians(coast_pts[:, 1])

    obs_lon_r = np.radians(df[lon_col].values.astype(float))
    obs_lat_r = np.radians(df[lat_col].values.astype(float))

    # Build KDTree for fast approximate nearest-neighbour lookup
    tree = cKDTree(np.column_stack([coast_lon_r, coast_lat_r]))
    k = min(5, len(coast_pts))
    _, idxs = tree.query(np.column_stack([obs_lon_r, obs_lat_r]), k=k)

    if idxs.ndim == 1:
        idxs = idxs[:, np.newaxis]

    # Exact haversine for top-k candidates
    cand_lat = coast_lat_r[idxs]
    cand_lon = coast_lon_r[idxs]
    obs_lat_2d = obs_lat_r[:, np.newaxis]
    obs_lon_2d = obs_lon_r[:, np.newaxis]

    dlat = cand_lat - obs_lat_2d
    dlon = cand_lon - obs_lon_2d
    a = (np.sin(dlat / 2) ** 2
         + np.cos(obs_lat_2d) * np.cos(cand_lat) * np.sin(dlon / 2) ** 2)
    dist_km = 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    return pd.Series(dist_km.min(axis=1), index=df.index)


def along_track_distance(df: pd.DataFrame, lat_col: str = "LATITUDE",
                         lon_col: str = "LONGITUDE", order_col: str | None = None) -> pd.Series:
    """Cumulative great-circle distance along a track [km].

    Collapses to unique ``(lat_col, lon_col)`` stations, orders them by
    *order_col* (or by first appearance in *df* when ``None``) and sums
    ``gsw.distance`` between consecutive stations. Rows sharing a station
    (e.g. different depth levels of one profile) get the same value.

    Args:
        df (pandas.DataFrame): DataFrame with latitude/longitude columns.
        lat_col (str): Latitude column [°N].
        lon_col (str): Longitude column [°E].
        order_col (str): Column to sort stations by (e.g. ``DATEANDTIME``
            or ``PROFILE_NUMBER``). Defaults to first-appearance order.
    Returns:
        pandas.Series: Cumulative distance [km] from the first station,
        same index as df.
    """
    # Reduce to one row per station, ordered along the track
    cols = [lat_col, lon_col] if order_col is None else [lat_col, lon_col, order_col]
    stations = df[cols].drop_duplicates([lat_col, lon_col])
    if order_col is not None:
        stations = stations.sort_values(order_col)

    # Great-circle distance between consecutive stations
    if len(stations) < 2:
        cum_km = np.zeros(len(stations))
    else:
        step_km = gsw.distance(
            stations[lon_col].values.astype(float), stations[lat_col].values.astype(float)
        ) / 1000.0
        cum_km = np.concatenate([[0.0], np.cumsum(step_km)])

    # Broadcast each station's cumulative distance back onto every row
    # sharing its (lat, lon), e.g. multiple depth levels of one profile
    lookup = pd.Series(cum_km, index=pd.MultiIndex.from_arrays(
        [stations[lat_col].values, stations[lon_col].values]
    ))
    keys = pd.MultiIndex.from_arrays([df[lat_col].values, df[lon_col].values])
    return pd.Series(lookup.loc[keys].values, index=df.index)


def _get_coast_points(resolution, coastline_geom):
    """Return an (N, 2) float array of [lon, lat] coastline vertices."""
    coords = []

    if coastline_geom is None:
        # Load cartopy coastlines
        try:
            import cartopy.io.shapereader as shpreader
        except ImportError:
            raise ImportError(
                "cartopy is required for distance_to_coast; "
                "install it with: pip install cartopy"
            )
        path = shpreader.natural_earth(
            resolution=resolution, category="physical", name="coastline"
        )
        for geom in shpreader.Reader(path).geometries():
            _extract_coords(geom, coords)

    else:
        # Load custom coastlines
        try:
            import geopandas as gpd
            if isinstance(coastline_geom, (gpd.GeoDataFrame, gpd.GeoSeries)):
                geoms = (coastline_geom.geometry
                         if isinstance(coastline_geom, gpd.GeoDataFrame)
                         else coastline_geom)
                for geom in geoms:
                    _extract_coords(geom, coords)
                return np.array(coords) if coords else None
        except ImportError:
            pass

        # Extracting coordinates
        if hasattr(coastline_geom, "__iter__") and not hasattr(coastline_geom, "geoms"):
            for geom in coastline_geom:
                _extract_coords(geom, coords)
        else:
            _extract_coords(coastline_geom, coords)

    return np.array(coords) if coords else None


def _extract_coords(geom, coords):
    """Recursively extract (lon, lat) coordinates from a shapely geometry."""
    if hasattr(geom, "geoms"):          # Multi* or GeometryCollection
        for part in geom.geoms:
            _extract_coords(part, coords)
    elif hasattr(geom, "exterior"):     # Polygon
        coords.extend(geom.exterior.coords)
    elif hasattr(geom, "coords"):       # LineString / Point
        coords.extend(geom.coords)


def load_longhurst(path: str | Path, code_col: str | None = None,
                   name_col: str | None = None) -> gpd.GeoDataFrame:
    """Load Longhurst province boundaries from a shapefile.

    Returns a GeoDataFrame with columns ``province_code``,
    ``province_name`` and ``geometry``. VLIZ column names
    (``ProvCode`` / ``ProvDescr``) are detected automatically.

    Requires geopandas (``pip install "comfort-db[geo]"``).

    Args:
        path (str or Path): Path to the Longhurst ``.shp`` file.
        code_col (str): Province code column. Auto-detected when ``None``.
        name_col (str): Province description column. Auto-detected when ``None``.
    Returns:
        geopandas.GeoDataFrame: Standardised province boundaries.
    """
    # Check if geopandas is installed
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "geopandas is required for load_longhurst; "
            "install it with: pip install \"comfort-db[geo]\""
        )

    # Read shapefile
    gdf = gpd.read_file(path)

    _CODE_CANDIDATES = ["ProvCode", "PROVCODE", "code", "CODE"]
    _NAME_CANDIDATES = ["ProvDescr", "PROVDESCR", "LongName", "name", "NAME"]

    # Define province code and name columns
    if code_col is None:
        code_col = next((c for c in _CODE_CANDIDATES if c in gdf.columns), None)
    if name_col is None:
        name_col = next((c for c in _NAME_CANDIDATES if c in gdf.columns), None)

    # Ensure that a code and name columns were found
    if code_col is None or name_col is None:
        raise ValueError(
            f"Could not detect code/name columns in {list(gdf.columns)}. "
            "Pass code_col and name_col explicitly."
        )

    return gdf[[code_col, name_col, "geometry"]].rename(
        columns={code_col: "province_code", name_col: "province_name"}
    )


def water_mass_masks(df: pd.DataFrame, regions: dict | str | Path | gpd.GeoDataFrame,
                     lat_col: str = "LATITUDE", lon_col: str = "LONGITUDE",
                     region_name_col: str = "name",
                     output_col: str = "region") -> pd.DataFrame:
    """Assign each observation to a 2d geographic region.

    For overlapping regions the first match wins (dict order /
    GeoDataFrame row order). Observations outside every region get ``NaN``.

    ``regions`` may be any of:

    * **dict** ``{name: [(lon, lat), ...]}`` or ``{name: shapely.Polygon}``
      - no extra dependencies required.
    * **CSV path** with columns ``name`` (or *region_name_col*) and
      ``geometry`` (WKT) - requires ``shapely``.
    * **Shapefile path** (``.shp``) - requires ``geopandas``.
    * **GeoDataFrame** / **GeoSeries** - requires ``geopandas``.

    Args:
        df (pandas.DataFrame): DataFrame with latitude and longitude columns.
        regions: Region definitions in any format listed above.
        lat_col (str): Latitude column [°N].
        lon_col (str): Longitude column [°E].
        region_name_col (str): Region name column in GeoDataFrame / CSV.
            Ignored for dict input. Default ``"name"``.
        output_col (str): Name of the new column. Default ``"region"``.
    Returns:
        pandas.DataFrame: Copy of df with an added ``output_col`` column.
    """
    # Get regions and their polygons
    named_regions = _parse_regions(regions, region_name_col)

    # Fast path: geopandas spatial join
    try:
        import geopandas as gpd

        # Init the observations geo df
        obs_gdf = gpd.GeoDataFrame(
            index=range(len(df)),
            geometry=gpd.points_from_xy(
                df[lon_col].values, df[lat_col].values
            ),
            crs="EPSG:4326",
        )

        # Init regions geo df
        regions_gdf = gpd.GeoDataFrame(
            {"_rname": [n for n, _ in named_regions],
             "geometry": [g for _, g in named_regions]},
            crs="EPSG:4326",
        )

        # Join
        joined = gpd.sjoin(obs_gdf, regions_gdf, how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]
        region_series = joined["_rname"].reindex(range(len(df)))

        # Assemble final df
        result = df.copy()
        result[output_col] = region_series.values
        return result

    except ImportError:
        pass

    # Fallback: pure shapely containment test
    from shapely.geometry import Point

    # Iterate over observations
    labels = []
    for _, row in df.iterrows():
        # Containment test
        pt = Point(row[lon_col], row[lat_col])
        match = next((name for name, poly in named_regions if poly.contains(pt)), None)
        labels.append(match)

    # Assemble final df
    result = df.copy()
    result[output_col] = labels
    return result


def water_mass_statistics(df: pd.DataFrame, param_cols: list[str] | str,
                          region_col: str = "region") -> pd.DataFrame:
    """Compute per-region descriptive statistics for one or more parameters.

    Args:
        df (pandas.DataFrame): DataFrame with region assignments
            (from :func:`water_mass_masks`) and parameter columns.
        param_cols (list[str]): Parameter columns to aggregate.
        region_col (str): Region column. Default ``"region"``.
    Returns:
        pandas.DataFrame: MultiIndex columns ``(parameter, statistic)``
            with count, mean, std, min, max. Rows are regions.
    """
    if isinstance(param_cols, str):
        param_cols = [param_cols]
    return df.groupby(region_col)[param_cols].agg(["count", "mean", "std", "min", "max"])


def _parse_regions(regions, region_name_col):
    """Return a list of ``(name, shapely.Polygon)`` tuples from any input format."""
    from shapely.geometry import Polygon

    # Regions as dict
    if isinstance(regions, dict):
        result = []
        for name, geom in regions.items():
            poly = geom if hasattr(geom, "contains") else Polygon(geom)
            result.append((name, poly))
        return result

    # Regions as CSV
    path = Path(regions) if isinstance(regions, (str, Path)) else None
    if path is not None and path.suffix.lower() == ".csv":
        from shapely import wkt
        csv_df = pd.read_csv(path)
        name_col = region_name_col if region_name_col in csv_df.columns else "name"
        return [
            (row[name_col], wkt.loads(row["geometry"]))
            for _, row in csv_df.iterrows()
        ]

    # Regions as shapefile
    if path is not None and path.suffix.lower() == ".shp":
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required to read shapefiles; "
                "install it with: pip install \"comfort-db[geo]\""
            )
        gdf = gpd.read_file(path)
        return [(row[region_name_col], row.geometry) for _, row in gdf.iterrows()]

    # Return Series or df
    try:
        import geopandas as gpd
        if isinstance(regions, gpd.GeoSeries):
            return list(regions.items())
        if isinstance(regions, gpd.GeoDataFrame):
            return [(row[region_name_col], row.geometry) for _, row in regions.iterrows()]
    except ImportError:
        pass

    raise ValueError(
        f"Cannot parse regions of type {type(regions).__name__}. "
        "Expected dict, CSV path, shapefile path, or GeoDataFrame."
    )


def load_jenniges_provinces(path: str | Path, depth: float | None = None) -> pd.DataFrame:
    """Load Jenniges biogeochemical provinces from a cluster CSV file.

    The CSV must have columns ``LATITUDE``, ``LONGITUDE``,
    ``LEV_M`` and ``label``. The standard file is ``cluster_set.csv``
    from `Zenodo record 15201767 <https://zenodo.org/records/15201767>`_.

    Args:
        path (str or Path): Path to the cluster CSV file.
        depth (float): Depth level [m] to select. When ``None`` all depths
            are returned. Uses closest available level if exact match
            is unavailable.
    Returns:
        pandas.DataFrame
    """
    # Check if file exists
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Jenniges provinces file not found: {path}")

    # Load Jenniges regions
    df = pd.read_csv(path)
    for col in ("LATITUDE", "LONGITUDE", "label"):
        if col not in df.columns:
            raise ValueError(f"Expected column {col!r} not found in {path.name}")
    if depth is not None and "LEV_M" in df.columns:
        available = df["LEV_M"].unique()
        closest = float(available[np.abs(available - depth).argmin()])
        if closest != depth:
            logging.info(
                "load_jenniges_provinces: requested depth %.1f m, "
                "using closest available %.1f m", depth, closest,
            )
        df = df[df["LEV_M"] == closest].copy()
    return df


def classify_from_grid(df: pd.DataFrame, grid_df: pd.DataFrame,
                       lat_col: str = "LATITUDE", lon_col: str = "LONGITUDE",
                       grid_lat_col: str = "LATITUDE",
                       grid_lon_col: str = "LONGITUDE",
                       label_col: str = "label",
                       output_col: str = "label") -> pd.DataFrame:
    """Classify observations by nearest-neighbour lookup against a labelled grid.

    Finds the closest grid point (Euclidean distance on radian-converted
    coordinates) and copies its label.

    Args:
        df (pandas.DataFrame): Observation DataFrame with lat/lon columns.
        grid_df (pandas.DataFrame): Labelled grid (e.g. from
            :func:`load_jenniges_provinces`).
        lat_col (str): Latitude column in *df*.
        lon_col (str): Longitude column in *df*.
        grid_lat_col (str): Latitude column in *grid_df*.
        grid_lon_col (str): Longitude column in *grid_df*.
        label_col (str): Label column in *grid_df*.
        output_col (str): Name of the new column added to the result.
    Returns:
        pandas.DataFrame: Copy of *df* with an added *output_col* column.
    """
    from scipy.spatial import cKDTree

    # Define grid coordinates in radians
    grid_coords = np.column_stack([
        np.radians(grid_df[grid_lon_col].values.astype(float)),
        np.radians(grid_df[grid_lat_col].values.astype(float)),
    ])

    # Define observations coordinates in radians
    obs_coords = np.column_stack([
        np.radians(df[lon_col].values.astype(float)),
        np.radians(df[lat_col].values.astype(float)),
    ])

    # Compute closest grid point
    tree = cKDTree(grid_coords)
    _, idxs = tree.query(obs_coords, k=1)

    # Assemble result
    result = df.copy()
    result[output_col] = grid_df[label_col].values[idxs]
    return result
