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
    if hasattr(geom, "geoms"):  # Multi* or GeometryCollection
        for part in geom.geoms:
            _extract_coords(part, coords)
    elif hasattr(geom, "exterior"):  # Polygon
        coords.extend(geom.exterior.coords)
    elif hasattr(geom, "coords"):  # LineString / Point
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


def load_basins(path: str | Path, name_col: str | None = None) -> gpd.GeoDataFrame:
    """Load "Global Oceans and Seas" basin boundaries from a shapefile
    (Flanders Marine Institute (2021). Global Oceans and Seas, version 1.
    Available online at https://www.marineregions.org/. https://doi.org/10.14284/542).

    Requires geopandas (``pip install "comfort-db[geo]"``).

    Args:
        path (str or Path): Path to the basins ``.shp`` file.
        name_col (str): Basin name column. Auto-detected when ``None``.
    Returns:
        geopandas.GeoDataFrame: Basin boundaries with columns ``basin_name`` and ``geometry``.
    """
    # Check if geopandas is installed
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "geopandas is required for load_basins. "
            "Install it with: pip install \"comfort-db[geo]\""
        )

    # Read shapefile
    gdf = gpd.read_file(path)

    # Define basin name column (case-insensitive match)
    if name_col is None:
        cols_lower = {c.lower(): c for c in gdf.columns}
        name_col = cols_lower.get("name")
    if name_col is None:
        raise ValueError(f"Could not detect name column in {list(gdf.columns)}. Pass name_col explicitly.")

    return gdf[[name_col, "geometry"]].rename(columns={name_col: "basin_name"})


def load_biomes(path: str | Path, biome_var: str) -> pd.DataFrame:
    """Load Fay and McKinley (2014) global ocean biomes from a gridded NetCDF file.

     Reference: Fay, A. R. and McKinley, G. A.: Global open-ocean biomes:
     mean and temporal variability, Earth Syst. Sci. Data, 6, 273–284,
     https://doi.org/10.5194/essd-6-273-2014, 2014.

    Requires xarray (``pip install "comfort-db[grid]"``).

    Args:
        path (str or Path): Path to the biomes NetCDF file.
        biome_var (str): Biome-ID variable (``"MeanBiomes"``, ``"CoreBiomes"``or
            ``"TimeVaryingBiomes"``).
    Returns:
        pandas.DataFrame: Columns ``LATITUDE``, ``LONGITUDE``, ``label``
        (and ``YEAR`` if biome_var=``"TimeVaryingBiomes"``).
    """
    # Check if xarray is installed
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray is required for load_biomes; "
            "install it with: pip install \"comfort-db[grid]\""
        )

    # Read NetCDF file
    ds = xr.open_dataset(path)

    # Filter for the requested biome data
    if biome_var not in ds.data_vars:
        raise ValueError(
            f"biome_var {biome_var!r} not found in {list(ds.data_vars)}. "
            "Pass one of these explicitly."
        )
    da = ds[biome_var]

    # Extract time dimension
    extra_dims = [d for d in da.dims if d not in ("lat", "lon")]
    time_dim = extra_dims[0] if extra_dims else None

    # Formatting
    df = da.to_dataframe(name="label").reset_index()  # to df
    df = df.rename(columns={"lat": "LATITUDE", "lon": "LONGITUDE"})  # rename to library convention
    df = df.dropna(subset=["label"])  # drop land/no-biome cells
    out_cols = ["LATITUDE", "LONGITUDE", "label"]
    if time_dim is not None:
        # Add time dimension if present
        df = df.rename(columns={time_dim: time_dim.upper()})
        out_cols = ["LATITUDE", "LONGITUDE", time_dim.upper(), "label"]
    return df[out_cols].reset_index(drop=True)


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


def region_bbox(regions: dict | str | Path | gpd.GeoDataFrame,
                region_name_col: str = "name") -> tuple[float, float, float, float]:
    """Return the lat/lon bounding box of one or more regions.

    Intended to derive a bounding box for a cheap SQL query before an exact polygon refined region search,
    see ``region=`` on :func:`comfort.io.load_comfort`/:func:`~comfort.io.read_parameter`/:func:`~comfort.io.subset_region`.

    A region whose polygon crosses the antimeridian (+-180°) is inidcted by returning a lon_min that is
    bigger than lon_max.

    Args:
        regions: Region definition(s), same formats as water_mass_masks.
        region_name_col (str): Region name column. See water_mass_masks.
    Returns:
        tuple[float, float, float, float]: ``(lat_min, lat_max, lon_min, lon_max)``.
    """
    # Get a dict of (region_name, polygon)
    named_regions = _parse_regions(regions, region_name_col)

    # Collect every vertex longitude/latitude per region (handles Multi* parts too)
    per_region_lons = []
    all_lats = []
    for _, poly in named_regions:
        coords = []
        _extract_coords(poly, coords)  # get coords of the polygon
        per_region_lons.append([c[0] for c in coords])  # store lons as a list per region
        all_lats.extend(c[1] for c in coords)  # store all lats as a simple list

    # A vertex span > 180 degrees signals the polygon includes the antimeridian
    includes_antimeridian = any(max(lons) - min(lons) > 180 for lons in per_region_lons)
    if not includes_antimeridian:
        # Take min/max longitude first per region, then across regions
        lon_min = min(min(lons) for lons in per_region_lons)
        lon_max = max(max(lons) for lons in per_region_lons)
    else:
        # Shift negative longitudes to [0,360] before taking min/max
        shifted = [lon + 360 if lon < 0 else lon for lons in per_region_lons for lon in lons]
        s_min, s_max = min(shifted), max(shifted)
        lon_min = s_min if s_min <= 180 else s_min - 360  # project back to [-180, 180]
        lon_max = s_max if s_max <= 180 else s_max - 360

    return min(all_lats), max(all_lats), lon_min, lon_max


def select_regions(regions: dict | str | Path | gpd.GeoDataFrame,
                   names: str | list[str], region_name_col: str = "name") -> dict:
    """Filter a water_mass_masks-compatible regions source for named region(s).

    Args:
        regions: Region definition(s), same format as water_mass_masks.
        names (str or list[str]): Region name(s) to keep.
        region_name_col (str): Region name column. See water_mass_masks.
    Returns:
        dict[str, shapely.Polygon]: Only the requested region(s).
    Raises:
        ValueError: If any requested name is not found.
    """
    # Ensure that region names are a list
    if isinstance(names, str):
        names = [names]

    # Get (region-name, polygon) dict
    named_regions = _parse_regions(regions, region_name_col)

    # Filter for the requested regions
    selected = {name: poly for name, poly in named_regions if name in names}

    # Raise if a region name was not found
    missing = set(names) - set(selected)
    if missing:
        raise ValueError(f"Region name(s) not found: {sorted(missing)}")

    return selected


def _resolve_name_col(columns, region_name_col):
    """Return region_name_col, raising an error if it is not in columns."""
    if region_name_col not in columns:
        raise ValueError(f"Could not find region name column {region_name_col!r} in {list(columns)}.")
    return region_name_col


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
        name_col = _resolve_name_col(csv_df.columns, region_name_col)
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
        name_col = _resolve_name_col(gdf.columns, region_name_col)
        return [(row[name_col], row.geometry) for _, row in gdf.iterrows()]

    # Regions as gpd
    try:
        import geopandas as gpd
        if isinstance(regions, gpd.GeoSeries):
            return list(regions.items())
        if isinstance(regions, gpd.GeoDataFrame):
            name_col = _resolve_name_col(regions.columns, region_name_col)
            return [(row[name_col], row.geometry) for _, row in regions.iterrows()]
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
                       output_col: str = "label",
                       time_col: str | None = None,
                       grid_time_col: str | None = None) -> pd.DataFrame:
    """Classify observations against a labelled grid by nearest-neighbour lookup.

    Finds the closest grid point (Euclidean distance on radian-converted
    coordinates) and copies its label.

    If *time_col* is given, each sample is first matched to its nearest
    time step in *grid_time_col*. Then, spatial neighbours are searched
    only in that time step.

    Args:
        df (pandas.DataFrame): Observation DataFrame with lat/lon columns.
        grid_df (pandas.DataFrame): Labelled grid (e.g. from
            :func:`load_jenniges_provinces` or :func:`load_biomes`).
        lat_col (str): Latitude column in *df*.
        lon_col (str): Longitude column in *df*.
        grid_lat_col (str): Latitude column in *grid_df*.
        grid_lon_col (str): Longitude column in *grid_df*.
        label_col (str): Label column in *grid_df*.
        output_col (str): Name of the new column added to the result.
        time_col (str): Time column in *df* (e.g. year) to match against
            *grid_time_col*. Omit for a purely spatial grid.
        grid_time_col (str): Time column in *grid_df*. Defaults to
            *time_col* if *time_col* is given.
    Returns:
        pandas.DataFrame: Copy of *df* with an added *output_col* column.
    """
    result = df.copy()

    # Spatial nearest neighbours (if no time_col is given)
    if time_col is None:
        result[output_col] = _nearest_labels(
            obs_lon=df[lon_col].values,
            obs_lat=df[lat_col].values,
            grid_lon=grid_df[grid_lon_col].values,
            grid_lat=grid_df[grid_lat_col].values,
            grid_labels=grid_df[label_col].values,
        )
        return result

    # Temporal nearest neighbours
    grid_time_col = grid_time_col or time_col
    grid_times = np.sort(grid_df[grid_time_col].unique().astype(float))  # all sorted grid times
    obs_times = df[time_col].values.astype(float)  # all observation times
    nearest_times = grid_times[np.abs(obs_times[:, None] - grid_times[None, :]).argmin(axis=1)]

    # Compute spatial nearest neighbours per time step
    labels = np.empty(len(df), dtype=grid_df[label_col].values.dtype)
    for t in grid_times:
        # Check if current time step has data
        mask = nearest_times == t
        if not mask.any():
            continue

        # Filter for current time step
        grid_slice = grid_df[grid_df[grid_time_col] == t]

        # Spatial nearest neighbours
        labels[mask] = _nearest_labels(
            obs_lon=df.loc[mask, lon_col].values,
            obs_lat=df.loc[mask, lat_col].values,
            grid_lon=grid_slice[grid_lon_col].values,
            grid_lat=grid_slice[grid_lat_col].values,
            grid_labels=grid_slice[label_col].values,
        )
    result[output_col] = labels
    return result


def _nearest_labels(obs_lon, obs_lat, grid_lon, grid_lat, grid_labels):
    """Nearest-neighbour label lookup (Euclidean distance on radian-converted coordinates)."""
    from scipy.spatial import cKDTree

    # Convert to radians
    grid_coords = np.column_stack([np.radians(grid_lon.astype(float)), np.radians(grid_lat.astype(float))])
    obs_coords = np.column_stack([np.radians(obs_lon.astype(float)), np.radians(obs_lat.astype(float))])

    # Construct neighbourhood graph
    tree = cKDTree(grid_coords)

    # Retrun the one nearest neighbour for each observation
    _, idxs = tree.query(obs_coords, k=1)
    return grid_labels[idxs]
