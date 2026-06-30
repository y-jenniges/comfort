"""Classes and functions for spatiotemporal gridding of parameter values.

Adapted from Jenniges (2025), doi:10.5281/zenodo.15827777
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from .util import sqlite_utils
from .util.sqlite_utils import validate_identifier
from .database.information import does_table_exist, get_table_as_df, get_names_of_all_parameter_tables
from .database.structure import remove_tables_like


class _BaseGrid:
    """Shared grid logic for Grid and SpaceGrid."""

    def generate_grid_id(self) -> int:
        """Return a grid ID derived from the current Unix timestamp (seconds)."""
        return round(time.time())

    def _open_bathymetry(self, bathymetry_grid_path: str, lat_variable: str,
                         lon_variable: str, depth_variable: str):
        """Open, normalise and load a bathymetry NetCDF file into memory, then close it."""
        # Check xarray import
        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "xarray is required for gridding; "
                "install it with: pip install \"comfort-db[grid]\""
            )

        # Check if filepath is valid
        if not os.path.isfile(bathymetry_grid_path):
            raise FileNotFoundError(
                f"Bathymetry file not found: {bathymetry_grid_path}. "
                "Pass bathymetry_grid_path to the constructor."
            )

        # Load bathymetry dataset
        with xr.open_dataset(bathymetry_grid_path) as ds:
            ds = ds.rename({lat_variable: "LATITUDE", lon_variable: "LONGITUDE", depth_variable: "LEV_M"})
            ds["LEV_M"] = ds["LEV_M"] * -1  # Positive z points downward (COMFORT convention)
            ds.load()
        return ds

    def _check_spatial_input(self) -> None:
        """Validate spatial grid parameters; raise ValueError on invalid input."""
        required = {
            "lat_min": self.lat_min, "lat_max": self.lat_max, "dlat": self.dlat,
            "lon_min": self.lon_min, "lon_max": self.lon_max, "dlon": self.dlon,
            "z_min": self.z_min, "z_max": self.z_max,
        }

        # Check if all params are specified
        for name, val in required.items():
            if val is None:
                raise ValueError(f"Grid parameter {name!r} must not be None")

        # Check min/max values for latitude and longitude
        if not (-90 <= self.lat_min <= 90) or not (-90 <= self.lat_max <= 90):
            raise ValueError("lat_min and lat_max must be in [-90, 90]")
        if not (-180 <= self.lon_min <= 180) or not (-180 <= self.lon_max <= 180):
            raise ValueError("lon_min and lon_max must be in [-180, 180]")

        # Check min/max values for depth
        if not (0 <= self.z_min <= 12000) or not (0 <= self.z_max <= 12000):
            raise ValueError("z_min and z_max must be in [0, 12000]")

        # Check depth (specified as z_array)
        if self.z_array is None:
            if self.dz is None:
                raise ValueError("dz must be provided when z_array is None")

            # For a regular grid, we need complete grid cells
            if (abs(self.lat_min) + abs(self.lat_max)) % self.dlat != 0:
                raise ValueError("Latitude range must be evenly divisible by dlat")
            if (abs(self.lon_min) + abs(self.lon_max)) % self.dlon != 0:
                raise ValueError("Longitude range must be evenly divisible by dlon")
            if (abs(self.z_min) + abs(self.z_max)) % self.dz != 0:
                raise ValueError("Depth range must be evenly divisible by dz")
        else:
            if self.dz is not None:
                raise ValueError("dz must be None when z_array is provided")

    def _make_z_arr(self) -> np.ndarray:
        """Return the depth array to use for the water mask (excluding the last boundary node)."""
        if self.z_array is None:
            return np.arange(self.z_min, self.z_max, self.dz)
        return self.z_array[:-1]

    def _water_mask(self, ds, lat, lon):
        """Return a xarray DataArray that is True where there is water (depth > 0)."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "xarray is required for gridding; "
                "install it with: pip install \"comfort-db[grid]\""
            )
        # Wrap depth levels into xarray
        z_arr = self._make_z_arr()
        z = xr.DataArray(z_arr, dims="LEV_M", coords={"LEV_M": z_arr}, name="LEV_M")

        # Assign
        z_at_grid = ds.LEV_M.sel(LONGITUDE=lon, LATITUDE=lat, method="nearest")
        z_at_grid.coords["LATITUDE"] = lat
        z_at_grid.coords["LONGITUDE"] = lon
        return (z <= z_at_grid).rename("water")

    def _map_lat_lon_depth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bin LATITUDE, LONGITUDE and LEV_M columns in df to grid cell centres."""
        # Map latitude values into grid cells (labels = cell centres)
        lat_bins = np.arange(self.lat_min, self.lat_max + self.dlat, self.dlat, dtype=float)
        lat_bins[-1] += 0.001 * (1 if lat_bins[-1] >= 0 else -1)
        df["LATITUDE"] = pd.cut(df["LATITUDE"], bins=lat_bins, right=False,
                                labels=lat_bins[:-1] + self.dlat / 2)

        # Map longitude values
        lon_bins = np.arange(self.lon_min, self.lon_max + self.dlon, self.dlon, dtype=float)
        lon_bins[-1] += 0.001 * (1 if lon_bins[-1] >= 0 else -1)
        df["LONGITUDE"] = pd.cut(df["LONGITUDE"], bins=lon_bins, right=False,
                                 labels=lon_bins[:-1] + self.dlon / 2)

        # Map depth values
        if self.z_array is not None:
            z_bins = self.z_array.astype(float)
        else:
            z_bins = np.arange(self.z_min, self.z_max + self.dz, self.dz, dtype=float)

        if len(z_bins) == 1:
            df["LEV_M"] = z_bins[0]
        else:
            z_bins[-1] += 0.001
            df["LEV_M"] = pd.cut(df["LEV_M"], bins=z_bins, right=False, labels=z_bins[:-1])

        return df


class GridManager:
    """Persists Grid objects in the database and retrieves them by ID."""

    def __init__(self, db_path: str, grid_info_table: str = "grid_info") -> None:
        # Validate identifier
        validate_identifier(grid_info_table)

        # Set paths
        self.db_path = db_path
        self.grid_info_table = grid_info_table

        # Connect to db
        self.connection = sqlite3.connect(db_path)

        # Create grid information table if it does not yet exist
        if not does_table_exist(self.connection, self.grid_info_table, "table"):
            self.connection.execute(
                f"CREATE TABLE {self.grid_info_table}("
                f"  grid_id PRIMARY KEY, "
                f"  lat_min REAL, lat_max REAL, dlat REAL, "
                f"  lon_min REAL, lon_max REAL, dlon REAL, "
                f"  z_min TEXT, z_max TEXT, dz TEXT, z_array TEXT, "
                f"  time_min TEXT, time_max TEXT, mode TEXT, dtime INTEGER, selection TEXT);"
            )

    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()

    def __enter__(self) -> GridManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def does_grid_exist(self, grid_id: int) -> bool:
        """Check whether a grid with the given ID exists in the grid_info table."""
        cur = self.connection.execute(
            f"SELECT 1 FROM {self.grid_info_table} WHERE grid_id=?;",
            (grid_id,),
        )
        return cur.fetchone() is not None

    def add_grid(self, grid: Grid) -> None:
        """Add a Grid in the grid_info table and write its data table."""
        z_min_val = "None" if grid.z_array is not None else grid.z_min
        z_max_val = "None" if grid.z_array is not None else grid.z_max
        dz_val = "None" if grid.z_array is not None else grid.dz
        z_array_val = (json.dumps(grid.z_array.tolist())
                       if grid.z_array is not None else "None")
        sel_val = (json.dumps(grid.selection.tolist())
                   if grid.selection is not None else "None")

        # Insert into grid_info table
        self.connection.execute(
            f"INSERT INTO {self.grid_info_table} "
            f"(grid_id, lat_min, lat_max, dlat, lon_min, lon_max, dlon, "
            f"z_min, z_max, dz, z_array, time_min, time_max, mode, dtime, selection) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (grid.grid_id, grid.lat_min, grid.lat_max, grid.dlat,
             grid.lon_min, grid.lon_max, grid.dlon,
             z_min_val, z_max_val, dz_val, z_array_val,
             str(grid.time_min), str(grid.time_max), str(grid.mode),
             grid.dtime, sel_val),
        )
        self.connection.commit()

        # Create grid table
        grid.grid.to_sql(grid.grid_name, self.connection, if_exists="replace",
                         index=True, index_label="idx")

    def remove_grid(self, grid_id: int) -> None:
        """Remove a grid and all its mapped parameter tables from the database."""
        # Check if grid_id exists
        if not does_table_exist(self.connection, f"grid_{grid_id}"):
            return

        # Load all grids and drop the one to remove from grid information table
        grids = get_table_as_df(self.connection, "grid_info")
        grids = grids.drop(grids[grids.grid_id.astype(str) == str(grid_id)].index)

        # Drop all mapped parameter tables and the grid_table
        remove_tables_like(self.connection, like_pattern=f"%|_{grid_id}",
                           escape_char="|", table_type="table")
        remove_tables_like(self.connection, like_pattern=f"%|_{grid_id}|_%",
                           escape_char="|", table_type="table")

        # Write new grid information table
        grids.to_sql("grid_info", self.connection, if_exists="replace", index=False)

    def remove_all_grids(self) -> None:
        """Remove every grid from the database."""
        # Get all grid_id entries
        df_grid_info = get_table_as_df(self.connection, "grid_info")

        # Remove all grids
        for grid_id in df_grid_info["grid_id"].values:
            self.remove_grid(grid_id)

    def load_grid(self, grid_id: int, bathymetry_grid_path: str, lat_variable: str = "lat",
                  lon_variable: str = "lon", depth_variable: str = "elevation") -> Grid | None:
        """Load a previously saved Grid by its ID.

        Args:
            grid_id: Grid ID to load.
            bathymetry_grid_path (str): Path to the bathymetry NetCDF file.
            lat_variable (str): Latitude variable name in the NetCDF file.
            lon_variable (str): Longitude variable name in the NetCDF file.
            depth_variable (str): Depth/elevation variable name in the NetCDF file.
        Returns:
            Grid or None.
        """
        # Check if grid exists
        if not self.does_grid_exist(grid_id):
            logging.warning(f"Grid with ID {grid_id} does not exist")
            return None

        # Get grid info
        cur = self.connection.execute(
            f"SELECT * FROM {self.grid_info_table} WHERE grid_id=?;", (grid_id,)
        )
        info = pd.DataFrame(cur.fetchall(), columns=[x[0] for x in cur.description])

        # Helper to set grid parameters
        def _val(col):
            v = info[col].values[0]
            return None if str(v) == "None" else v

        return Grid(
            lat_min=_val("lat_min"), lat_max=_val("lat_max"), dlat=_val("dlat"),
            lon_min=_val("lon_min"), lon_max=_val("lon_max"), dlon=_val("dlon"),
            z_min=float(_val("z_min")) if _val("z_min") is not None else None,
            z_max=float(_val("z_max")) if _val("z_max") is not None else None,
            dz=float(_val("dz")) if _val("dz") is not None else None,
            z_array=(json.loads(info["z_array"].values[0])
                     if info["z_array"].values[0] != "None" else None),
            time_min=_val("time_min"), time_max=_val("time_max"),
            mode=_val("mode"), dtime=_val("dtime"),
            selection=(json.loads(info["selection"].values[0])
                       if info["selection"].values[0] != "None" else None),
            bathymetry_grid_path=bathymetry_grid_path,
            lat_variable=lat_variable, lon_variable=lon_variable,
            depth_variable=depth_variable, grid_id=grid_id,
        )

    def create_grid(self, lat_min, lat_max, dlat, lon_min, lon_max, dlon,
                    z_min=None, z_max=None, dz=None, z_array=None,
                    time_min=None, time_max=None, mode=None, dtime=None, selection=None,
                    bathymetry_grid_path=None,
                    lat_variable="lat", lon_variable="lon",
                    depth_variable="elevation") -> Grid:
        """Create a new Grid (or load an existing one with the same parameters).

        Args:
            bathymetry_grid_path (str): Path to the bathymetry NetCDF file. Required.
        Returns:
            Grid
        """
        # Try to find existing grid with same params
        grid_id = self.get_grid_id_from_params(
            lat_min, lat_max, dlat, lon_min, lon_max, dlon,
            z_min, z_max, dz, z_array, time_min, time_max, mode, dtime, selection,
        )

        # If the grid already exists, only load it
        if grid_id is None:
            grid = Grid(
                lat_min=lat_min, lat_max=lat_max, dlat=dlat,
                lon_min=lon_min, lon_max=lon_max, dlon=dlon,
                z_min=z_min, z_max=z_max, dz=dz, z_array=z_array,
                time_min=time_min, time_max=time_max, mode=mode, dtime=dtime,
                selection=selection, bathymetry_grid_path=bathymetry_grid_path,
                lat_variable=lat_variable, lon_variable=lon_variable,
                depth_variable=depth_variable,
            )
            self.add_grid(grid)
        else:
            logging.info(f"Grid already exists with ID {grid_id}; loading")
            grid = self.load_grid(grid_id, bathymetry_grid_path=bathymetry_grid_path,
                                  lat_variable=lat_variable, lon_variable=lon_variable,
                                  depth_variable=depth_variable)
        return grid

    def get_grid_id_from_params(self, lat_min, lat_max, dlat, lon_min, lon_max, dlon,
                                z_min=None, z_max=None, dz=None, z_array=None,
                                time_min=None, time_max=None, mode=None, dtime=None,
                                selection=None) -> int | None:
        """Return the grid_id of an existing grid matching these parameters, or None."""
        # Define params
        params = [lat_min, lat_max, dlat, lon_min, lon_max, dlon]

        # Add depth params
        if z_array is None:
            params.extend([z_min, z_max, dz, "None"])
        else:
            params.extend(["None", "None", "None",
                           json.dumps(np.array(z_array).tolist())])

        # Add  time selection
        sel_val = ("None" if selection is None
                   else json.dumps(np.array(selection).tolist()))
        params.extend([str(time_min), str(time_max), str(mode), dtime, sel_val])

        # Get grid_id from db
        query = (
            f"SELECT grid_id FROM {self.grid_info_table} WHERE "
            f"lat_min=? AND lat_max=? AND dlat=? AND "
            f"lon_min=? AND lon_max=? AND dlon=? AND "
            f"z_min=? AND z_max=? AND dz=? AND z_array=? AND "
            f"time_min=? AND time_max=? AND mode=? AND dtime=? AND selection=?"
        )
        res = self.connection.execute(query, params).fetchall()
        return res[0][0] if res else None


class Grid(_BaseGrid):
    """A spatio-temporal grid based on a bathymetry file.
    Lat/lon are cell-centred, time and depth are node-centred.
    """

    # Time modes
    _VALID_MODES = [None, "Y", "M", "D", "YM", "YD", "MD"]

    def __init__(self, lat_min=-90, lat_max=90, dlat=1, lon_min=-180, lon_max=180, dlon=1,
                 bathymetry_grid_path=None,
                 z_min=None, z_max=None, dz=None, z_array=None,
                 time_min=None, time_max=None, mode=None, dtime=None, selection=None,
                 lat_variable="lat", lon_variable="lon", depth_variable="elevation",
                 grid_id=None):
        """
        Args:
            lat_min, lat_max, dlat: Latitude range and step [°].
            lon_min, lon_max, dlon: Longitude range and step [°].
            bathymetry_grid_path (str): Path to a bathymetry NetCDF file. Required.
            z_min, z_max, dz: Depth range and step [m]. Provide either dz or z_array.
            z_array (array-like): Explicit depth nodes [m]. Mutually exclusive with dz.
            time_min, time_max (str): Date strings bounding the time dimension.
            mode (str): Time grouping mode. One of None, 'Y', 'M', 'D', 'YM', 'YD', 'MD'.
            dtime (int): Number of time units per cell.
            selection (array-like): Boolean mask to sub-select the time array.
            lat_variable, lon_variable, depth_variable: Variable names in the NetCDF file.
            grid_id: Existing ID to use instead of generating a new one.
        """
        # Grid specs
        self.grid_id = grid_id if grid_id else self.generate_grid_id()
        self.grid_name = f"grid_{self.grid_id}"

        # Init grid params
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.dlat = dlat
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.dlon = dlon
        self.z_array = np.sort(np.array(z_array)) if z_array is not None else None
        self.z_min = z_min if z_array is None else self.z_array.min()
        self.z_max = z_max if z_array is None else self.z_array.max()
        self.dz = dz
        self.time_min = time_min
        self.time_max = time_max
        self.mode = mode
        self.dtime = dtime
        self.selection = np.array(selection) if selection is not None else None
        self.bathymetry_grid_path = bathymetry_grid_path

        self._check_spatial_input()
        self._check_time_input()

        # Create grid
        self.time_array = self._create_time_array()
        self.grid = self._create_grid(lat_variable, lon_variable, depth_variable)

    def _check_time_input(self) -> None:
        """Consistency checks for time input."""
        # Not None
        for name, val in [("time_min", self.time_min), ("time_max", self.time_max),
                          ("dtime", self.dtime)]:
            if val is None:
                raise ValueError(f"Grid parameter {name!r} must not be None")

        # Time step >=1
        if self.dtime < 1:
            raise ValueError("dtime must be >= 1")

        # Valid modes
        if self.mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {self._VALID_MODES}")

        # Type of selection object
        if self.selection is not None and not isinstance(self.selection, (np.ndarray, list)):
            raise ValueError("selection must be a list or numpy array")

    @staticmethod
    def _iter_range(start, end, step):
        """Yield datetimes from start to end (exclusive) by relativedelta step."""
        from dateutil.parser import parse as _parse_dt
        current = _parse_dt(start) if isinstance(start, str) else start
        end_dt = _parse_dt(end) if isinstance(end, str) else end
        while current < end_dt:
            yield current
            current += step

    def _create_time_array(self) -> np.ndarray:
        """Create the time array containing the time steps for the grid."""
        time_array = []

        # Determine all valid dates for the desired time range
        if self.mode == "Y":
            time_array = [v.strftime("%Y-%m-%d %H:%M:%S")
                          for v in self._iter_range(self.time_min, self.time_max,
                                                    relativedelta(years=+self.dtime))]
        elif self.mode == "YM":
            time_array = [v.strftime("%Y-%m-%d %H:%M:%S")
                          for v in self._iter_range(self.time_min, self.time_max,
                                                    relativedelta(months=+self.dtime))]
        elif self.mode == "YD":
            time_array = [v.strftime("%Y-%m-%d %H:%M:%S")
                          for v in self._iter_range(self.time_min, self.time_max,
                                                    relativedelta(days=+int(self.dtime)))]
        elif self.mode == "M":
            time_array = np.char.zfill(np.arange(1, 13, self.dtime).astype(str), 2)
        elif self.mode == "MD":
            time_array = [v.strftime("%m-%d %H:%M:%S")
                          for v in self._iter_range("2000-01-01", "2000-12-31",
                                                    relativedelta(days=+int(self.dtime)))]
        elif self.mode == "D":
            time_array = np.char.zfill(np.arange(1, 32, self.dtime).astype(str), 2)

        # Perform selection
        time_array = np.array(time_array)
        if self.selection is not None:
            if time_array.shape != self.selection.shape:
                raise ValueError("selection shape must match time_array shape")
            time_array = time_array[self.selection.astype(bool)]

        return time_array

    def _create_grid(self, lat_variable: str, lon_variable: str,
                      depth_variable: str) -> pd.DataFrame:
        """Creating a grid based on:
        https://github.com/willirath/geomar-open-hacky-hour-2021-04/blob/main/2022-06-23/etopo05_to_grid.ipynb
        grid must be nc file, format: latitude, longitude, depth
        """
        import xarray as xr

        # Load bathymetry
        ds = self._open_bathymetry(self.bathymetry_grid_path, lat_variable, lon_variable, depth_variable)

        # Define cell-centered lat/lon values
        lat = xr.DataArray(np.arange(self.lat_min + self.dlat / 2, self.lat_max, self.dlat),
                           dims="LATITUDE")
        lon = xr.DataArray(np.arange(self.lon_min + self.dlon / 2, self.lon_max, self.dlon),
                           dims="LONGITUDE")

        # Add water mask and time dimension
        water_filled = self._water_mask(ds, lat, lon)
        water_filled = water_filled.expand_dims(DATEANDTIME=self.time_array)

        # Convert to df
        df = water_filled.to_dataframe().reset_index()

        return df.astype({"DATEANDTIME": str, "LEV_M": float,
                          "LATITUDE": float, "LONGITUDE": float, "water": bool})

    def map_tables(self, connection: sqlite3.Connection, param_tables: list[str] | None = None,
                   replace_existing: bool = False, include_z_max: bool = True) -> list[str]:
        """Bin each parameter table into this grid and write the result to the database.

        Args:
            connection (sqlite3.Connection): Connection to the database.
            param_tables (list[str]): Tables to map. None maps all parameter tables.
            replace_existing (bool): Replace already-mapped tables without prompting. Default is False.
            include_z_max (bool): Include the maximum depth value. Default is True.
        Returns:
            list[str]: Names of the mapped tables in the database.
        """
        logging.info("Mapping tables to grid...")
        connection.create_aggregate("median", 1, sqlite_utils.Median)
        connection.create_aggregate("std", 1, sqlite_utils.Std)

        # Determine parameter table names
        if not param_tables:
            param_tables = get_names_of_all_parameter_tables(connection)

        mapped_tables = []
        for table in param_tables:
            # Validate table name
            validate_identifier(table)

            # Check if table already exists
            existing = does_table_exist(connection, f"{table}_{self.grid_id}", "table")

            # Check if table should be replaced
            if existing and not replace_existing:
                # Append existing table name
                logging.info(f"Skipping already-mapped table {table}_{self.grid_id}")
                mapped_tables.append(f"{table}_{self.grid_id}")
                continue
            elif existing:
                connection.execute(f"DROP TABLE {table}_{self.grid_id};")

            # Build SQL query to retrieve table data
            z_eq = "<=" if include_z_max else "<"
            q = (f"SELECT LATITUDE, LONGITUDE, LEV_M, DATEANDTIME, VAL "
                 f"FROM {table} WHERE "
                 f"LATITUDE >= ? AND LATITUDE <= ? AND "
                 f"LONGITUDE >= ? AND LONGITUDE <= ? AND "
                 f"LEV_M >= ? AND LEV_M {z_eq} ? AND "
                 f"DATEANDTIME >= ? AND DATEANDTIME <= ?")
            bind = [self.lat_min, self.lat_max, self.lon_min, self.lon_max,
                    self.z_min, self.z_max,
                    str(self.time_min), str(self.time_max)]
            logging.debug(q)

            # Fetch data as df
            cur = connection.execute(q, bind)
            df = pd.DataFrame(cur.fetchall(), columns=[x[0] for x in cur.description])

            # Map lat/lon/depth
            df = self._map_lat_lon_depth(df)

            # Map DATEANDTIME column according to the time mode
            if self.mode in ("Y", "YM", "YD"):
                time_bins = pd.DatetimeIndex(self.time_array)
                time_bins = time_bins.union([pd.Timestamp(self.time_max) + pd.Timedelta(nanoseconds=1)])
                df["DATEANDTIME"] = pd.cut(
                    pd.to_datetime(df["DATEANDTIME"], format="%Y-%m-%d %H:%M:%S"),
                    bins=time_bins, labels=time_bins[:-1], right=False,
                )
                df["DATEANDTIME"] = (df["DATEANDTIME"].astype("datetime64[ns]")
                                     .dt.strftime("%Y-%m-%d %H:%M:%S"))
            elif self.mode == "M":
                # Get month from date column
                df["DATEANDTIME"] = df["DATEANDTIME"].astype("datetime64[ns]").dt.strftime("%m")
            elif self.mode == "MD":
                # Get month-day from date column
                df["DATEANDTIME"] = df["DATEANDTIME"].astype("datetime64[ns]").dt.strftime("%m-%d %H:%M:%S")
            elif self.mode == "D":
                # Get day from date column
                df["DATEANDTIME"] = df["DATEANDTIME"].astype("datetime64[ns]").dt.strftime("%d")

            # Aggregate cells with same latitude, longitude, depth and time (mean, median, std, count)
            df_grouped = (
                df[["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME", "VAL"]]
                .groupby(["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"])
                .agg(["mean", "median", "std", "count"])
                .reset_index()
            )
            df_grouped.columns = [
                x[0] if x[0] != "VAL" else x[1]
                for x in df_grouped.columns.to_flat_index()
            ]
            df_grouped = df_grouped.rename(columns={"mean": "VAL"})

            # Check data types
            df_grouped = df_grouped.astype(
                {"DATEANDTIME": str, "LEV_M": float, "LATITUDE": float,
                 "LONGITUDE": float, "VAL": float}
            )

            grid_name = f"grid_{self.grid_id}"
            # Add grid to db (if not yet included)
            if not does_table_exist(connection, grid_name, "table"):
                self.grid.to_sql(grid_name, connection, if_exists="replace",
                                 index=True, index_label="idx")
                logging.warning(
                    "Grid was added to the database without a GridManager - "
                    "it will not appear in grid_info."
                )

            # Write mapped table to db
            df_grouped.to_sql(f"temp_{table}_{self.grid_id}", connection,
                              if_exists="replace", index=False)

            # Add grid index to the table
            connection.execute(
                f"CREATE TABLE {table}_{self.grid_id} AS "
                f"SELECT g.idx, "
                f"t.LATITUDE, t.LONGITUDE, t.LEV_M, t.DATEANDTIME, "
                f"t.VAL AS {table}, t.median, t.std, t.count "
                f"FROM temp_{table}_{self.grid_id} AS t "
                f"LEFT JOIN grid_{self.grid_id} AS g "
                f"USING(LATITUDE, LONGITUDE, LEV_M, DATEANDTIME);"
            )

            # Drop temporary mapped table
            connection.execute(f"DROP TABLE temp_{table}_{self.grid_id};")
            mapped_tables.append(f"{table}_{self.grid_id}")

        return mapped_tables


class SpaceGrid(_BaseGrid):
    """A geospatial grid (no time dimension).
    Individual observations are mapped but not time-averaged.
    """

    def __init__(self, lat_min, lat_max, dlat, lon_min, lon_max, dlon,
                 bathymetry_grid_path,
                 z_min=None, z_max=None, dz=None, z_array=None,
                 lat_variable="lat", lon_variable="lon", depth_variable="elevation",
                 grid_id=None):
        """
        Args:
            lat_min, lat_max, dlat: Latitude range and step [°].
            lon_min, lon_max, dlon: Longitude range and step [°].
            bathymetry_grid_path (str): Path to a bathymetry NetCDF file. Required.
            z_min, z_max, dz: Depth range and step [m].
            z_array (array-like): Explicit depth nodes [m].
            lat_variable, lon_variable, depth_variable: Variable names in the NetCDF file.
            grid_id: Existing ID to use instead of generating a new one.
        """
        # Grid specs
        self.grid_id = grid_id if grid_id else self.generate_grid_id()
        self.grid_name = f"grid_{self.grid_id}"

        # Init grid params
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.dlat = dlat
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.dlon = dlon
        self.z_array = np.sort(np.array(z_array)) if z_array is not None else None
        self.z_min = z_min if z_array is None else self.z_array.min()
        self.z_max = z_max if z_array is None else self.z_array.max()
        self.dz = dz
        self.bathymetry_grid_path = bathymetry_grid_path

        # Check input
        self._check_spatial_input()

        # Create grid
        self.grid = self._create_grid(lat_variable, lon_variable, depth_variable)

    def _create_grid(self, lat_variable: str, lon_variable: str,
                      depth_variable: str) -> pd.DataFrame:
        """Creating a grid based on:
        https://github.com/willirath/geomar-open-hacky-hour-2021-04/blob/main/2022-06-23/etopo05_to_grid.ipynb
        grid must be nc file, format: latitude, longitude, depth.
        """
        import xarray as xr

        # Load bathymetry
        ds = self._open_bathymetry(self.bathymetry_grid_path, lat_variable, lon_variable, depth_variable)

        # Define cell-centered lat/lon values
        lat = xr.DataArray(np.arange(self.lat_min + self.dlat / 2, self.lat_max, self.dlat),
                           dims="LATITUDE")
        lon = xr.DataArray(np.arange(self.lon_min + self.dlon / 2, self.lon_max, self.dlon),
                           dims="LONGITUDE")

        # Add water mask and time dimension
        water_filled = self._water_mask(ds, lat, lon)

        # Convert to df
        df = water_filled.to_dataframe().reset_index()
        return df.astype({"LEV_M": float, "LATITUDE": float, "LONGITUDE": float, "water": bool})


    def map_tables(self, connection: sqlite3.Connection, param_tables: list[str] | None = None,
                   include_z_max: bool = True, output_dir: str | None = None) -> None:
        """Bin each parameter table into this spatial grid and write CSV output.

        Args:
            connection (sqlite3.Connection): Connection to the database.
            param_tables (list[str]): Tables to map. None maps all parameter tables.
            include_z_max (bool): Include the maximum depth value. Default is True.
            output_dir (str): Directory to write CSV files. None uses the current
                working directory.
        """
        logging.info("Mapping tables to SpaceGrid...")

        # Determine parameter table names
        if not param_tables:
            param_tables = get_names_of_all_parameter_tables(connection)

        # Create output_dir if not existant yet
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        for table in param_tables:
            # Validate table name
            validate_identifier(table)

            # Build SQL query to retrieve table data
            z_eq = "<=" if include_z_max else "<"
            q = (f"SELECT LATITUDE, LONGITUDE, LEV_M, DATEANDTIME, VAL "
                 f"FROM {table} WHERE "
                 f"LATITUDE >= ? AND LATITUDE <= ? AND "
                 f"LONGITUDE >= ? AND LONGITUDE <= ? AND "
                 f"LEV_M >= ? AND LEV_M {z_eq} ?")
            bind = [self.lat_min, self.lat_max, self.lon_min, self.lon_max,
                    self.z_min, self.z_max]
            logging.debug(q)

            # Fetch data as df
            cur = connection.execute(q, bind)
            df = pd.DataFrame(cur.fetchall(), columns=[x[0] for x in cur.description])

            # Map lat/lon/depth
            df = self._map_lat_lon_depth(df)

            # Check data types
            df = df.astype(
                {"DATEANDTIME": str, "LEV_M": float, "LATITUDE": float,
                 "LONGITUDE": float, "VAL": float}
            )

            # Join grid and parameter table and store table
            joined = df.merge(self.grid, on=["LATITUDE", "LONGITUDE", "LEV_M"], how="outer")
            joined.rename(columns={"VAL": table}, inplace=True)
            csv_name = f"{table}_grid.csv"
            if output_dir is not None:
                csv_name = os.path.join(output_dir, csv_name)
            joined.to_csv(csv_name)


def drop_land_cells(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that are on land (water=False) and have no parameter value.

    A row is dropped only if it is on land AND all P_* columns are NaN.
    Rows with data on land (e.g. coastal grid cells) are kept.

    Args:
        df_wide (pandas.DataFrame): Wide grid table with a 'water' column.
    Returns:
        pandas.DataFrame
    """
    temp = df_wide.copy()

    # Get parameter columns
    param_tables = [x for x in temp.columns if x.startswith("P_")]

    # Drop points that are land and have no parameter value assigned to it
    # make sure that for each depth level, the grid looks the same for all times
    depths = temp["LEV_M"].value_counts().index.tolist()
    times = temp["DATEANDTIME"].value_counts().index.tolist()
    for d in depths:
        condition = True
        indexes = pd.DataFrame()
        for param in param_tables:
            for t in times:
                ddtt = temp[(temp["LEV_M"] == d) & (temp["DATEANDTIME"] == t)].reset_index()
                condition = condition & ~ddtt["water"] & ddtt[param].isna()
                indexes = pd.concat([indexes, pd.DataFrame({f"{param}_{t}": ddtt["index"]})], axis=1)
        indexes = indexes[condition]  # get indexes that can be dropped
        indexes = pd.concat(indexes[col] for col in indexes)  # flatten to get one list of indexes to drop
        temp = temp.drop(indexes)
    return temp


def create_wide_table_offline(mapped_tables: list[pd.DataFrame],
                              dropping_land_cells: bool = True) -> pd.DataFrame:
    """Merge a list of per-parameter grid DataFrames into a single wide table.

    Args:
        mapped_tables (list[pandas.DataFrame]): Grid frames, one per parameter.
        dropping_land_cells (bool): Drop land-only rows. Default is True.
    Returns:
        pandas.DataFrame
    """
    # Get parameter table names and idx
    param_tables = [[x for x in df.columns if x.startswith("P_")][0] for df in mapped_tables]
    param_idx = [(col, idx) for idx, df in enumerate(mapped_tables)
                 for col in df.columns if col in param_tables]

    # Merge params and grid into one wide table
    df_wide = mapped_tables[param_idx[0][1]][
        ["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME", "water", param_tables[0]]
    ]
    for col, idx in param_idx[1:]:
        df_wide = df_wide.merge(
            mapped_tables[idx][["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME", col]],
            how="left", on=["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"],
        )

    # Drop land cells
    if dropping_land_cells:
        df_wide = drop_land_cells(df_wide)
    return df_wide


def create_wide_table_online(connection: sqlite3.Connection, grid_id: int,
                             param_tables: list[str] | None = None) -> str:
    """Create a wide table in the database by joining all mapped parameter tables.

    Args:
        connection (sqlite3.Connection): Connection to the database.
        grid_id: Grid ID whose mapped parameter tables should be joined.
        param_tables (list[str]): Parameter table names. None uses all parameter tables.
    Returns:
        str: Name of the created wide table.
    """
    # If no parameter tables are specified, use all
    if not param_tables:
        param_tables = get_names_of_all_parameter_tables(connection)

    # Define SQL query parts
    vals = ", ".join([f"{p}_{grid_id}.{p}" for p in param_tables])
    joins = " ".join([f"LEFT JOIN {p}_{grid_id} USING(idx)" for p in param_tables])
    wide_table_name = f"wide_{grid_id}_{round(time.time())}"

    # Create table in db
    connection.execute(
        f"CREATE TABLE {wide_table_name} AS SELECT "
        f"grid_{grid_id}.idx, grid_{grid_id}.DATEANDTIME, grid_{grid_id}.LEV_M, "
        f"grid_{grid_id}.LATITUDE, grid_{grid_id}.LONGITUDE, grid_{grid_id}.water, {vals} "
        f"FROM grid_{grid_id} {joins};"
    )
    return wide_table_name


def load_wide_table(connection: sqlite3.Connection, wide_table_name: str,
                    dropping_land_cells: bool = True) -> pd.DataFrame:
    """Load an existing wide table from the database.

    Args:
        connection (sqlite3.Connection): Connection to the database.
        wide_table_name (str): Name of the wide table.
        dropping_land_cells (bool): Drop land-only rows. Default is True.
    Returns:
        pandas.DataFrame
    """
    df_wide = get_table_as_df(connection, wide_table_name)
    if dropping_land_cells:
        df_wide = drop_land_cells(df_wide)
    return df_wide


def get_missing_value_info_per_param(connection: sqlite3.Connection, wide_table_name: str,
                                     param_tables: list[str]) -> pd.DataFrame:
    """Compute the fraction of missing values per parameter in a wide table.

    Args:
        connection (sqlite3.Connection): Connection to the database.
        wide_table_name (str): Name of the wide table.
        param_tables (list[str]): Parameter column names.
    Returns:
        pandas.DataFrame with columns 'parameter', 'total', 'relative'.
    """
    # Define SQL query parts
    water_or_any = " or ".join(f"{p} is not null" for p in param_tables)
    all_not_null = " and ".join(f"{p} is not null" for p in param_tables)

    # Number of relevant grid cells
    num_grid_cells = connection.execute(
        f"SELECT COUNT(*) FROM {wide_table_name} WHERE {water_or_any} OR water=1;"
    ).fetchone()[0]

    # Count cells that have all values
    num_complete = connection.execute(
        f"SELECT COUNT(*) FROM {wide_table_name} WHERE {all_not_null};"
    ).fetchone()[0]

    # Count not-null values per parameter column
    null_expr = ", ".join(f"COUNT(*) - COUNT({p}) AS {p}" for p in param_tables)
    cur = connection.execute(
        f"SELECT {null_expr} FROM {wide_table_name} WHERE {water_or_any} OR water=1;"
    )

    # Assemble df
    num_nulls = pd.DataFrame(cur.fetchall(), columns=[x[0] for x in cur.description]).T.reset_index()
    num_nulls.columns = ["parameter", "total"]
    num_nulls = pd.concat(
        [num_nulls,
         pd.DataFrame({"parameter": ["all"], "total": [num_grid_cells - num_complete]})]
    )
    num_nulls["relative"] = num_nulls["total"] / num_grid_cells * 100
    return num_nulls.sort_values("relative")


def get_missing_value_info_offline_per_param(mapped_dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """Compute missing-value statistics from a list of per-parameter DataFrames.

    Args:
        mapped_dataframes (list[pandas.DataFrame]): One DataFrame per parameter.
    Returns:
        pandas.DataFrame with columns 'parameter', 'absolute', 'relative'.
    """
    # If no mapped dfs are passed, return empty df
    if not mapped_dataframes:
        return pd.DataFrame(columns=np.array(["parameter", "absolute", "relative"]))

    # Extract grid
    grid = mapped_dataframes[0][["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME", "water"]]
    grid = grid[grid["water"]]

    # Compute number of null values per mapped table
    num_nulls = pd.DataFrame(columns=np.array(["parameter", "complete"]))
    for df in mapped_dataframes:
        param_col = [x for x in df.columns if x.startswith("P_")][0]
        num_complete = df[~df[param_col].isna()][param_col].count()
        num_nulls = pd.concat(
            [num_nulls, pd.DataFrame([{"parameter": param_col, "complete": num_complete}])],
            ignore_index=True,
        )
        filled_land = df[~df["water"] & ~df[param_col].isna()]
        grid = pd.concat([grid, filled_land[["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"]]])

    # Assemble missingness info df
    grid = grid.drop_duplicates()
    num_grid_cells = len(grid)
    num_nulls["absolute"] = num_grid_cells - num_nulls["complete"]
    num_nulls["relative"] = num_nulls["absolute"] / num_grid_cells * 100
    return num_nulls.sort_values("relative")


def get_missing_value_info_offline(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Compute missing-value counts from a wide DataFrame.

    Args:
        df_wide (pandas.DataFrame): Wide grid table.
    Returns:
        pandas.DataFrame with columns 'parameter', 'absolute', 'relative'.
    """
    # Get parameter table names
    param_tables = [x for x in df_wide.columns if x.startswith("P_")]

    # Compute absolute missingness
    rows = [{"parameter": p, "absolute": df_wide[p].isna().sum()} for p in param_tables]
    num_nulls = pd.DataFrame(rows)

    # Compute relative missinngess
    num_grid_cells = len(df_wide)
    if num_grid_cells == 0:
        num_nulls["relative"] = 0.0
    else:
        num_nulls["relative"] = num_nulls["absolute"] / num_grid_cells * 100
        num_nulls = num_nulls.sort_values("relative")
    return num_nulls
