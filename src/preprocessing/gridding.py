""" Classes and functions to grid parameter values. """
import logging
import os
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
import json
import shutil
from datetimerange import DateTimeRange
from dateutil.relativedelta import relativedelta
from src.util import sqlite_util
from src.database.communication import create_connection
from src.database.information import does_table_exist, get_table_as_df, get_names_of_all_parameter_tables
from src.database.structure import remove_tables_like


class GridManager:
    def __init__(self, db_path, grid_info_table="grid_info"):
        self.db_path = db_path
        self.grid_info_table = grid_info_table

        # connect to db
        self.connection = create_connection(db_path)

        # create grid information table if it does not yet exist
        if not does_table_exist(self.connection, self.grid_info_table, "table"):
            q = f"create table {self.grid_info_table}(" \
                f"  grid_id primary key, " \
                f"  lat_min real, lat_max real, dlat real, " \
                f"  lon_min real, lon_max real, dlon real, " \
                f"  z_min text, z_max text, dz text, z_array text, " \
                f"  time_min text, time_max text, mode text, dtime integer, selection text);"
            self.connection.execute(q)

    def does_grid_exist(self, grid_id):
        q = f"select * from {self.grid_info_table} where grid_id={grid_id};"
        cur = self.connection.execute(q)
        res = cur.fetchall()
        grid_exists = len(res) == 1
        return grid_exists

    def add_grid(self, grid):
        """ Add grid to grid information table. """
        if grid.z_array is not None:
            z_input = f"'None', 'None', 'None', '{json.dumps(grid.z_array.tolist())}'"
        else:
            z_input = f"{grid.z_min}, {grid.z_max}, {grid.dz}, 'None'"

        if grid.selection is not None:
            sel_input = f"'{json.dumps(grid.selection.tolist())}'"
        else:
            sel_input = "'None'"

        # insert into existing grid info table
        q = f"insert into {self.grid_info_table}" \
            f"(grid_id, lat_min, lat_max, dlat, lon_min, lon_max, dlon, z_min, z_max, dz, z_array, " \
            f"time_min, time_max, mode, dtime, selection) " \
            f"values({grid.grid_id}, {grid.lat_min}, {grid.lat_max}, {grid.dlat}, " \
            f"{grid.lon_min}, {grid.lon_max}, {grid.dlon}, " \
            f"{z_input}, " \
            f"'{grid.time_min}', '{grid.time_max}', '{grid.mode}', {grid.dtime}, " \
            f"{sel_input});"
        self.connection.execute(q)
        self.connection.commit()

        # create grid table itself
        grid.grid.to_sql(grid.grid_name, self.connection, if_exists="replace", index=True, index_label="idx")

    def remove_grid(self, grid_id):
        # check if grid_id exists
        if does_table_exist(self.connection, "grid_" + str(grid_id)):
            # load all grids and drop the one to remove from grid information table
            grids = get_table_as_df(self.connection, "grid_info")
            grids = grids.drop(grids[grids.grid_id.astype(str) == str(grid_id)].index)

            # drop all mapped parameter tables and the grid_table
            remove_tables_like(self.connection, like_pattern=f"%|_{grid_id}", escape_char="|", table_type="table")
            remove_tables_like(self.connection, like_pattern=f"%|_{grid_id}|_%", escape_char="|", table_type="table")

            # write new grid information table
            grids.to_sql("grid_info", self.connection, if_exists="replace", index=False)

    def remove_all_grids(self):
        # get all grid_ids
        df_grid_info = get_table_as_df(self.connection, "grid_info")

        # remove all grids
        for grid_id in df_grid_info["grid_id"].values:
            self.remove_grid(grid_id)

    def load_grid(self, grid_id,
                  bathymetry_grid_path="../../data/bathymetry/gebco_2022_sub_ice_topo/gebco_2022_sub_ice_topo.nc",
                  lat_variable="lat", lon_variable="lon", depth_variable="elevation"):
        # check if grid exists
        grid_exists = self.does_grid_exist(grid_id)

        # if it exists, load it
        if grid_exists:
            # get grid info
            q = f"select * from grid_info where grid_id={grid_id};"
            cur = self.connection.execute(q)
            grid_info = pd.DataFrame(cur.fetchall(), columns=[x[0] for x in cur.description])

            # init grid parameters
            lat_min = grid_info["lat_min"].values[0] if grid_info["lat_min"].values[0] != "None" else None
            lat_max = grid_info["lat_max"].values[0] if grid_info["lat_max"].values[0] != "None" else None
            dlat = grid_info["dlat"].values[0] if grid_info["dlat"].values[0] != "None" else None
            lon_min = grid_info["lon_min"].values[0] if grid_info["lon_min"].values[0] != "None" else None
            lon_max = grid_info["lon_max"].values[0] if grid_info["lon_max"].values[0] != "None" else None
            dlon = grid_info["dlon"].values[0] if grid_info["dlon"].values[0] != "None" else None
            z_min = float(grid_info["z_min"].values[0]) if grid_info["z_min"].values[0] != "None" else None
            z_max = float(grid_info["z_max"].values[0]) if grid_info["z_max"].values[0] != "None" else None
            dz = float(grid_info["dz"].values[0]) if grid_info["dz"].values[0] != "None" else None
            z_array = json.loads(grid_info["z_array"].values[0]) \
                if grid_info["z_array"].values[0] != "None" else None
            time_min = grid_info["time_min"].values[0] if grid_info["time_min"].values[0] != "None" else None
            time_max = grid_info["time_max"].values[0] if grid_info["time_max"].values[0] != "None" else None
            mode = grid_info["mode"].values[0] if grid_info["mode"].values[0] != "None" else None
            dtime = grid_info["dtime"].values[0] if grid_info["dtime"].values[0] != "None" else None
            selection = json.loads(grid_info["selection"].values[0]) \
                if grid_info["selection"].values[0] != "None" else None

            grid = Grid(lat_min=lat_min, lat_max=lat_max, dlat=dlat,
                        lon_min=lon_min, lon_max=lon_max, dlon=dlon,
                        z_min=z_min, z_max=z_max, dz=dz, z_array=z_array,
                        time_min=time_min, time_max=time_max, mode=mode, dtime=dtime, selection=selection,
                        bathymetry_grid_path=bathymetry_grid_path,
                        lat_variable=lat_variable, lon_variable=lon_variable, depth_variable=depth_variable,
                        grid_id=grid_id)
            return grid
        else:
            print(f"Grid with ID {grid_id} does not exists.")
            return

    def create_grid(self, lat_min, lat_max, dlat, lon_min, lon_max, dlon,
                    z_min=None, z_max=None, dz=None, z_array=None,
                    time_min=None, time_max=None, mode=None, dtime=None, selection=None,
                    bathymetry_grid_path=None,
                    lat_variable="lat", lon_variable="lon", depth_variable="elevation"):

        grid_id = self.get_grid_id_from_params(lat_min, lat_max, dlat, lon_min, lon_max, dlon,
                                               z_min, z_max, dz, z_array,
                                               time_min, time_max, mode, dtime, selection)

        # if the grid already exists, only load it
        if not grid_id:
            grid = Grid(lat_min=lat_min, lat_max=lat_max, dlat=dlat,
                        lon_min=lon_min, lon_max=lon_max, dlon=dlon,
                        z_min=z_min, z_max=z_max, dz=dz, z_array=z_array,
                        time_min=time_min, time_max=time_max, mode=mode, dtime=dtime, selection=selection,
                        bathymetry_grid_path=bathymetry_grid_path,
                        lat_variable=lat_variable, lon_variable=lon_variable, depth_variable=depth_variable)
            self.add_grid(grid)
        else:
            print(f"gridding.GridManager.create_grid: Grid already exists with ID {grid_id}. Loading grid...")
            grid = self.load_grid(grid_id, bathymetry_grid_path=bathymetry_grid_path)

        return grid

    def get_grid_id_from_params(self, lat_min, lat_max, dlat, lon_min, lon_max, dlon,
                                z_min=None, z_max=None, dz=None, z_array=None,
                                time_min=None, time_max=None, mode=None, dtime=None, selection=None):
        # z array must be a numpy array
        if z_array is None:
            z_statement = f"z_min={z_min} and z_max={z_max} and dz={dz} and z_array='None'"
        else:
            z_statement = f"z_min='None' and z_max='None' and dz='None' and " \
                          f"z_array='{json.dumps(z_array.tolist())}'"
        if selection is None:
            sel_statement = "selection='None'"
        else:
            sel_statement = f"selection='{json.dumps(selection.tolist())}'"

        q = f"select grid_id from {self.grid_info_table} where " \
            f"lat_min={lat_min} and lat_max={lat_max} and dlat={dlat} and " \
            f"lon_min={lon_min} and lon_max={lon_max} and dlon={dlon} and " \
            f"{z_statement} and " \
            f"time_min='{time_min}' and time_max='{time_max}' and mode='{mode}' and dtime={dtime} and " \
            f"{sel_statement};"
        cur = self.connection.execute(q)
        res = cur.fetchall()

        if len(res) == 0:
            # print("Could not find an existing grid with the given parameters.")
            return

        grid_id = res[0][0]
        return grid_id


class Grid:
    def __init__(self,
                 lat_min=None, lat_max=None, dlat=None,
                 lon_min=None, lon_max=None, dlon=None,
                 z_min=None, z_max=None, dz=None, z_array=None,
                 time_min=None, time_max=None, mode=None, dtime=None, selection=None,
                 bathymetry_grid_path="../../data/bathymetry/gebco_2022_sub_ice_topo/gebco_2022_sub_ice_topo.nc",
                 lat_variable="lat", lon_variable="lon", depth_variable="elevation",
                 grid_id=None):
        # lat/lon are cell-centered, time and depth are node-centered
        # grid specifics
        if grid_id:
            self.grid_id = grid_id
            self.grid_name = "grid_" + str(grid_id)
        else:
            self.grid_id = self.generate_grid_id()
            self.grid_name = "grid_" + str(self.grid_id)

        # init grid parameters
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
        self.selection = selection
        self.bathymetry_grid_path = bathymetry_grid_path

        self.check_input()

        if self.selection is not None:
            self.selection = np.array(self.selection)

        # create grid
        self.time_array = self.create_time_array()
        self.grid = self.create_grid(lat_variable, lon_variable, depth_variable)

    def check_input(self):
        # consistency checks
        assert (self.lat_min is not None)
        assert (self.lat_max is not None)
        assert (self.dlat is not None)
        assert (self.lon_min is not None)
        assert (self.lon_max is not None)
        assert (self.dlon is not None)
        assert (self.time_min is not None)
        assert (self.time_max is not None)
        assert (self.dtime is not None)
        assert (self.z_min is not None)
        assert (self.z_max is not None)
        assert (self.z_min >= 0)
        assert (self.z_max <= 12000)  # 10977)

        if self.z_array is None:
            assert (self.dz is not None)
            # for a regular grid, we need complete grid cells
            assert ((abs(self.lat_min) + abs(self.lat_max)) % self.dlat == 0)
            assert ((abs(self.lon_min) + abs(self.lon_max)) % self.dlon == 0)
            assert ((abs(self.z_min) + abs(self.z_max)) % self.dz == 0)
        else:
            assert (self.dz is None)

        assert (self.lat_min >= -90)
        assert (self.lat_max <= 90)
        assert (self.lon_min >= -180)
        assert (self.lon_max <= 180)
        assert (self.dtime >= 1)

        if self.selection is not None:
            assert(isinstance(self.selection, np.ndarray) or isinstance(self.selection, list))

        # check for all possible time modes
        single_modes = ["Y", "M", "D"]  # not yet implemented: , "H", "m", "S"]
        all_modes = [None]
        for i in range(len(single_modes)):
            for j in range(i, len(single_modes)):
                if i == j:
                    all_modes.append(single_modes[i])
                else:
                    all_modes.append(single_modes[i] + single_modes[j])
        assert (self.mode in all_modes)

        # check if path is valid
        assert (os.path.isfile(self.bathymetry_grid_path))

    def generate_grid_id(self):
        grid_id = round(time.time())  # current timestamp in seconds
        return grid_id

    def create_time_array(self):
        time_range = DateTimeRange(self.time_min, self.time_max)
        time_array = []

        # determine all valid dates for the desired time range
        if self.mode == "Y":
            for value in time_range.range(relativedelta(years=+self.dtime)):
                time_array.append(value.strftime("%Y-%m-%d %H:%M:%S"))
        elif self.mode == "YM":
            for value in time_range.range(relativedelta(months=+self.dtime)):
                time_array.append(value.strftime("%Y-%m-%d %H:%M:%S"))
        elif self.mode == "YD":
            for value in time_range.range(relativedelta(days=+int(self.dtime))):
                time_array.append(value.strftime("%Y-%m-%d %H:%M:%S"))
        elif self.mode == "M":
            time_array = np.char.zfill(np.arange(1, 12 + 1, self.dtime).astype(str), 2)
        elif self.mode == "MD":
            for value in DateTimeRange("2000-01-01", "2000-12-31").range(relativedelta(days=+int(self.dtime))):
                time_array.append(value.strftime("%m-%d %H:%M:%S"))
        elif self.mode == "D":
            time_array = np.char.zfill(np.arange(1, 31 + 1, self.dtime).astype(str), 2)

        # perform selection
        time_array = np.array(time_array)
        if self.selection is not None:
            assert (np.equal(time_array.shape, self.selection.shape))
            time_array = time_array[self.selection.astype(bool)]

        return time_array

    def create_grid(self, lat_variable="lat", lon_variable="lon", depth_variable="elevation"):
        # bases on:
        # https://github.com/willirath/geomar-open-hacky-hour-2021-04/blob/main/2022-06-23/etopo05_to_grid.ipynb
        # grid must be nc file, format: latitude, longitude, depth
        grid_dataset = xr.open_dataset(self.bathymetry_grid_path)
        grid_dataset = grid_dataset.rename({
            lat_variable: "LATITUDE",
            lon_variable: "LONGITUDE",
            depth_variable: "LEV_M"
        }
        )

        # positive z points downwards (convention in COMFORT)
        grid_dataset["LEV_M"] = grid_dataset["LEV_M"] * -1

        # will be used as grid selectors (cell-centered)
        lat = xr.DataArray(np.arange(self.lat_min + self.dlat / 2, self.lat_max, self.dlat), dims="LATITUDE")
        lon = xr.DataArray(np.arange(self.lon_min + self.dlon / 2, self.lon_max, self.dlon), dims="LONGITUDE")

        # will be used to generate a boolean array which is True where water is present (cell-centered)
        if self.z_array is None:
            z_arr = np.arange(self.z_min, self.z_max, self.dz)
        else:
            z_arr = self.z_array[:-1]

        z = xr.DataArray(
            z_arr,
            dims="LEV_M",
            coords={"LEV_M": z_arr},
            name="LEV_M",
        )

        z_at_grid = grid_dataset.LEV_M.sel(
            LONGITUDE=lon,
            LATITUDE=lat,
            method="nearest"
        )

        # make sure we label with the selectors
        z_at_grid.coords["LATITUDE"] = lat
        z_at_grid.coords["LONGITUDE"] = lon

        # addd a dimension which is True if there is water
        water_filled = (z <= z_at_grid).rename("water")
        # water_filled.plot(col="LEV_M", col_wrap=3)

        # add time dimension
        water_filled = water_filled.expand_dims(DATEANDTIME=self.time_array)
        # water_filled[:, 0, :, :].plot(col="time")

        df = water_filled.to_dataframe().reset_index()
        # write grid to SQL database
        # df = df[df["water"] == 1].drop(["water"], axis=1)  # only keep grid points where there is water

        # have right dtypes
        df = df.astype({"DATEANDTIME": str, "LEV_M": float, "LATITUDE": float, "LONGITUDE": float, "water": bool})

        return df

    def map_tables(self, connection, param_tables=None, using_database=True, include_z_max=True):
        logging.info("Mapping tables to grid...")
        connection.create_aggregate('median', 1, sqlite_util.Median)
        connection.create_aggregate('std', 1, sqlite_util.Std)

        if not param_tables:
            param_tables = get_names_of_all_parameter_tables(connection)

        mapped_tables = []  # offline computations: contains grids; online computations: contains table names
        for table in param_tables:
            answer = "n"  # whether to replace table in database if existing
            # if mapped table already exists in database, only continue computations if they should be replaced
            if using_database and does_table_exist(conn=connection, table_name=f"{table}_{self.grid_id}",
                                                   table_type="table"):
                grid_name = f"grid_{self.grid_id}"
                # let user decide whether to replace existing table
                input_valid = False
                while not input_valid:
                    print("Replace table? y/n")
                    answer = input()
                    if answer == "y" or answer == "n":
                        input_valid = True
                    else:
                        print("Invalid input.")
                if answer == "n":
                    mapped_tables.append(f"{table}_{self.grid_id}")
                    continue

            # load data
            # where clause --> filter samples for latitude, longitude, depth, time range
            z_eq = "<=" if include_z_max else "<"
            q = f"select LATITUDE, LONGITUDE, LEV_M, DATEANDTIME, VAL " \
                f"from {table} where " \
                f"LATITUDE >= {self.lat_min} and LATITUDE <= {self.lat_max} and " \
                f"LONGITUDE >= {self.lon_min} and LONGITUDE <= {self.lon_max} and " \
                f"LEV_M >= {self.z_min} and LEV_M {z_eq} {self.z_max} and " \
                f"DATEANDTIME >= '{self.time_min}' and DATEANDTIME <= '{self.time_max}' " \
                f";"
            print(q)
            cur = connection.execute(q)
            df = pd.DataFrame(cur.fetchall(), columns=[x[0] for x in cur.description])

            # map latitude
            lat_bins = np.arange(self.lat_min, self.lat_max + self.dlat, self.dlat).astype(float)
            lat_bins[-1] = lat_bins[-1] + 0.001 * (1 if lat_bins[-1] >= 0 else -1)
            # df["LATITUDE_binned"] = pd.cut(df["LATITUDE"], bins=lat_bins, right=False, labels=lat_bins[:-1])
            df["LATITUDE"] = pd.cut(df["LATITUDE"], bins=lat_bins, right=False,
                                    labels=lat_bins[:-1] + self.dlat / 2)

            # map longitude
            lon_bins = np.arange(self.lon_min, self.lon_max + self.dlon, self.dlon).astype(float)
            lon_bins[-1] = lon_bins[-1] + 0.001 * (1 if lon_bins[-1] >= 0 else -1)
            df["LONGITUDE"] = pd.cut(df["LONGITUDE"], bins=lon_bins, right=False,
                                     labels=lon_bins[:-1] + self.dlon / 2)

            # map depth
            if self.z_array is not None:
                z_bins = self.z_array.astype(float)
            else:
                z_bins = np.arange(self.z_min, self.z_max + self.dz, self.dz).astype(float)

            if len(z_bins) == 1:
                df["LEV_M"] = z_bins[0]
            else:
                z_bins[-1] = z_bins[-1] + 0.001  # make sure, last values are included while cutting using pandas
                df["LEV_M"] = pd.cut(df["LEV_M"], bins=z_bins, right=False, labels=z_bins[:-1])

            # map time
            if self.mode in ["Y", "YM", "YD"]:
                time_bins = pd.DatetimeIndex(self.time_array)
                time_bins = time_bins.union([pd.Timestamp(self.time_max) + pd.Timedelta(nanoseconds=1)])
                df["DATEANDTIME"] = pd.cut(pd.to_datetime(df["DATEANDTIME"], format="%Y-%m-%d %H:%M:%S"),
                                           bins=time_bins, labels=time_bins[:-1], right=False)
                df["DATEANDTIME"] = df["DATEANDTIME"].astype(np.datetime64).dt.strftime("%Y-%m-%d %H:%M:%S")
            elif self.mode == "M":
                # get month from date column
                df["DATEANDTIME"] = df["DATEANDTIME"].astype(np.datetime64).dt.strftime("%m")
            elif self.mode == "MD":
                # get month-day from date column
                df["DATEANDTIME"] = df["DATEANDTIME"].astype(np.datetime64).dt.strftime("%m-%d %H:%M:%S")
            elif self.mode in ["D"]:
                # get day from date column
                df["DATEANDTIME"] = df["DATEANDTIME"].astype(np.datetime64).dt.strftime("%d")

            # averaging cells with same latitude, longitude, depth and time
            df_grouped = df[["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME", "VAL"]] \
                .groupby(by=["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"]) \
                .agg(["mean", "median", "std", "count"]).reset_index()
            df_grouped.columns = [x[0] if x[0] != "VAL" else x[1] for x in list(df_grouped.columns.to_flat_index())]
            df_grouped = df_grouped.rename(columns={"mean": "VAL"})

            # check data types
            df_grouped = df_grouped.astype({"DATEANDTIME": str, "LEV_M": float,
                                            "LATITUDE": float, "LONGITUDE": float, "VAL": float})

            if using_database:
                grid_name = f"grid_{self.grid_id}"
                # if grid was not yet added to database, add it
                if not does_table_exist(connection, grid_name, "table"):
                    self.grid.to_sql(grid_name, connection, if_exists="replace", index=True, index_label="idx")
                    print("Warning in gridding.map_tables: Grid was added to the database but not to the grid_info "
                          "table as no GridManager was used.")

                # if mapped table already exists and user wants to replace it, do that
                if answer == "y":
                    # drop existing table
                    q = f"drop table {table}_{self.grid_id};"
                    connection.execute(q)

                # write mapped table to database
                df_grouped.to_sql(f"temp_{table}_{self.grid_id}", connection, if_exists="replace", index=False)

                # adding the grid index to the table
                q = f"create table {table}_{self.grid_id} as " \
                    f"select g.idx, " \
                    f"t.LATITUDE, t.LONGITUDE, t.LEV_M, t.DATEANDTIME, t.VAL as {table}, t.median, t.std, t.count " \
                    f"from temp_{table}_{self.grid_id} as t left join grid_{self.grid_id} as g " \
                    f"using(LATITUDE, LONGITUDE, LEV_M, DATEANDTIME);"
                connection.execute(q)

                # drop temporary mapped table
                q = f"drop table temp_{table}_{self.grid_id};"
                connection.execute(q)

                # add grid name to return-list
                mapped_tables.append(f"{table}_{self.grid_id}")

            else:
                # join grid and parameter table and return table
                joined = self.grid.merge(df_grouped, on=["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"],
                                         how="left")
                joined.rename(columns={"VAL": table}, inplace=True)
                mapped_tables.append(joined)

        return mapped_tables


def drop_land_cells(df_wide):
    # get parameter columns
    param_tables = [x for x in df_wide.columns if x.startswith("P_")]

    # drop points that are land and have no parameter value assigned to it
    # make sure that for each depth level, the grid looks the same for all times
    depths = df_wide["LEV_M"].value_counts().index.tolist()
    times = df_wide["DATEANDTIME"].value_counts().index.tolist()
    for d in depths:
        condition = True
        indexes = pd.DataFrame()
        for param in param_tables:
            for t in times:
                ddtt = df_wide[(df_wide["LEV_M"] == d) & (df_wide["DATEANDTIME"] == t)].reset_index()
                condition = condition & ~ddtt["water"] & ddtt[param].isna()
                indexes = pd.concat([indexes, pd.DataFrame({f"{param}_{t}": ddtt["index"]})], axis=1)
        indexes = indexes[condition]  # get indexes that can be dropped
        indexes = pd.concat(indexes[col] for col in indexes)  # flatten to get one list of indexes to drop
        df_wide = df_wide.drop(indexes)
    return df_wide


def create_wide_table_offline(mapped_tables, dropping_land_cells=True):
    # get parameter columns
    param_tables = []
    for df in mapped_tables:
        param_tables.append([x for x in df.columns if x.startswith("P_")][0])

    # get dataframes
    param_idx = []
    for idx, mt in enumerate(mapped_tables):
        for col in mt.columns:
            if col in param_tables:
                # print(idx, mt.columns)
                param_idx.append([col, idx])

    idxs = [pai[1] for pai in param_idx]
    params = [pai[0] for pai in param_idx]

    # merge params and grid into one wide table
    df_wide = mapped_tables[idxs[0]][["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME", "water", param_tables[0]]]
    for i in range(1, len(idxs)):
        df_wide = df_wide.merge(mapped_tables[idxs[i]][["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME", params[i]]],
                                how="left", on=["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"])

    if dropping_land_cells:
        df_wide = drop_land_cells(df_wide)

    return df_wide


def create_wide_table_online(connection, grid_id, param_tables=None):
    """
    Creates step-wise a wide table with the columns:
    latitude, longitude, dateandtime, lev_m, param_0, ..., param_n (where n=[0, #param_tables - 1]).
    Given parameter tables need to be mapped already and stored in the database as paramTableName_gridId.

    Args:
        connection (sqlite.Connection): Connection to the database.
        grid_id (str): ID of the grid based on which to join the parameter tables.
        param_tables (list<str>): Names of the parameter tables to merge. If None, all parameters will be used.
        Default is None.
    Returns:
        new_path (str): Path to the new database.
        table_name (str): Name of the created wide table.
    """
    # if no parameter tables are specified, use all
    if not param_tables:
        param_tables = get_names_of_all_parameter_tables(connection)

    vals = ", ".join([f"{param_table}_{grid_id}.{param_table}" for param_table in param_tables])
    joins = " ".join([f"left join {param_table}_{grid_id} using(idx)" for param_table in param_tables])

    wide_table_name = f"wide_{grid_id}_{round(time.time())}"
    q = f"create table {wide_table_name} as select " \
        f"grid_{grid_id}.idx, grid_{grid_id}.DATEANDTIME, grid_{grid_id}.LEV_M, grid_{grid_id}.LATITUDE, " \
        f"grid_{grid_id}.LONGITUDE, grid_{grid_id}.water, {vals} " \
        f"from grid_{grid_id} " \
        f"{joins};"
    connection.execute(q)

    return wide_table_name


def load_wide_table(connection, wide_table_name, dropping_land_cells=True):
    df_wide = get_table_as_df(conn=connection, table_name=wide_table_name)
    if dropping_land_cells:
        df_wide = drop_land_cells(df_wide)
    return df_wide


def get_missing_value_info_per_param(connection, wide_table_name, param_tables):
    # number of relevant grid cells
    q = f"select count(*) from {wide_table_name} " \
        f"where {' or '.join([param_table + ' is not null' for param_table in param_tables])} " \
        f"or water=1;"
    num_grid_cells = connection.execute(q).fetchall()[0][0]

    # count cells that have all values
    q = f"select count(*) from {wide_table_name} " \
        f"where {' and '.join([param_table + ' is not null' for param_table in param_tables])};"
    num_complete_cells = connection.execute(q).fetchall()[0][0]

    # count not-null values per parameter column
    q = f"select {', '.join(f'count(*) - count({p}) as {p}' for p in param_tables)} from {wide_table_name} " \
        f"where {' or '.join([param_table + ' is not null' for param_table in param_tables])} " \
        f"or water=1;"
    cur = connection.execute(q)
    num_nulls = pd.DataFrame(cur.fetchall(), columns=[x[0] for x in cur.description]).T.reset_index()
    num_nulls.columns = ["parameter", "total"]
    num_nulls = pd.concat([num_nulls,
                           pd.DataFrame({"parameter": ["all"], "total": [num_grid_cells - num_complete_cells]})])
    num_nulls["relative"] = num_nulls["total"] / num_grid_cells * 100
    num_nulls = num_nulls.sort_values(by="relative")

    return num_nulls


def get_missing_value_info_offline_per_param(mapped_dataframes):
    if len(mapped_dataframes) == 0:
        return

    grid = mapped_dataframes[0][["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME", "water"]]
    grid = grid[grid["water"]]
    num_nulls = pd.DataFrame(columns=["parameter", "complete"])
    for df in mapped_dataframes:
        param_col = [x for x in df.columns if x.startswith("P_")][0]
        num_complete_cells = df[~df[param_col].isna()][param_col].count()
        num_nulls = pd.concat([num_nulls,
                               pd.DataFrame({"parameter": [param_col],
                                             "complete": [num_complete_cells]
                                             })]
                              )
        filled_land_cells = df[~df["water"] & ~df[param_col].isna()]
        grid = pd.concat([grid, filled_land_cells[["LATITUDE", "LONGITUDE", "LEV_M", "DATEANDTIME"]]])
    grid = grid.drop_duplicates()
    num_grid_cells = len(grid)
    num_nulls["absolute"] = num_grid_cells - num_nulls["complete"]
    num_nulls["relative"] = num_nulls["absolute"] / num_grid_cells * 100
    num_nulls = num_nulls.sort_values(by="relative")

    return num_nulls


def get_missing_value_info_offline(df_wide):
    # get parameter columns
    param_tables = [x for x in df_wide.columns if x.startswith("P_")]

    num_nulls = pd.DataFrame(columns=["parameter", "absolute"])
    for param in param_tables:
        num_nulls_param = df_wide[param].isna().sum()
        num_nulls = pd.concat([num_nulls,
                               pd.DataFrame({"parameter": [param],
                                             "absolute": [num_nulls_param]
                                             })]
                              )

    num_grid_cells = len(df_wide)
    if num_grid_cells == 0:
        num_nulls["relative"] = 0
    else:
        num_nulls["relative"] = num_nulls["absolute"] / num_grid_cells * 100
        num_nulls = num_nulls.sort_values(by="relative")

    return num_nulls


def plot_missing_value_info(num_nulls):
    ax = num_nulls.plot(kind="bar", title="Fraction of missing values", x="parameter", y="relative", legend=False,
                        ylabel="%", xlabel="", grid=True)
    for p in ax.patches:
        ax.annotate(str(round(p.get_height())), (p.get_x() * 1.01, p.get_height() * 1.01))
    plt.tight_layout()
    plt.show(block=True)
    plt.close()


def plot_missing_value_info_map(df_wide, param_tables, depths=None, times=None, savefig_folder=None):
    # df_wide needs LATITUDE, LONGITUDE, optionally: DATEANDTIME, LEV_M
    df_wide["DATEANDTIME"] = pd.to_datetime(df_wide["DATEANDTIME"])
    if not depths:
        depths = list(df_wide["LEV_M"].value_counts().index)
    if not times:
        times = list(df_wide["DATEANDTIME"].value_counts().index)

    cmap = cm.get_cmap("viridis").copy()
    cmap.set_bad("black", alpha=0.2)

    for param in param_tables:
        min_val = df_wide[param].min()
        max_val = df_wide[param].max()
        norm = plt.Normalize(min_val, max_val)  # LogNorm, PowerNorm
        for t in times:
            for depth in depths:
                dft = df_wide[(df_wide["LEV_M"] == depth) & (df_wide["DATEANDTIME"] == t)]

                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.set_global()
                ax.gridlines(draw_labels=True)
                sc = plt.scatter(dft["LONGITUDE"], dft["LATITUDE"], c=dft[param], cmap=cmap, norm=norm,
                                 s=0.5, marker="s", plotnonfinite=True)
                plt.colorbar(location="bottom", pad=0.05)  # .set_label(legend_title[2])
                plt.rcParams["figure.figsize"] = (10, 8)
                plt.title(f"{param} - {depth}m - {t.year}")
                # plt.clim(min_val, max_val)
                plt.tight_layout()
                if savefig_folder:
                    if not os.path.exists(savefig_folder):
                        os.makedirs(savefig_folder)
                    plt.savefig(f"{savefig_folder}/{param}_depth{depth}_time{t.year}.png")
                plt.show()  #block=True)
                plt.close()


def plot_missing_value_info_map_over_depth(df_wide, param_tables, times=None, savefig_folder=None, relative=True):
    for param in param_tables:
        if relative:
            col_name = "missing_relative"
        else:
            col_name = param

        df_wide["DATEANDTIME"] = pd.to_datetime(df_wide["DATEANDTIME"])
        if not times:
            times = list(df_wide["DATEANDTIME"].value_counts().index)

        # plot colors
        cmap = cm.get_cmap("viridis")
        norm = plt.Normalize(0, 100)

        for t in times:
            dft = df_wide[df_wide["DATEANDTIME"] == t]

            # compute how many cells in each water column are empty/missing values
            # (in percent as the number of depth cells varies per column)
            grouped = dft[["LATITUDE", "LONGITUDE", "LEV_M", param]]\
                .groupby(by=["LATITUDE", "LONGITUDE"]).agg("count").reset_index()
            grouped["missing_relative"] = 100 - grouped[param]/grouped["LEV_M"]*100

            # plotting
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.gridlines(draw_labels=True)
            sc = plt.scatter(grouped["LONGITUDE"], grouped["LATITUDE"], c=grouped[col_name], cmap=cmap, norm=norm,
                             marker="s", plotnonfinite=True)
            plt.colorbar(location="bottom", pad=0.05)
            plt.rcParams["figure.figsize"] = (10, 8)
            plt.title(f"% of missing values for {param}")
            plt.tight_layout()
            if savefig_folder:
                if not os.path.exists(savefig_folder):
                    os.makedirs(savefig_folder)
                plt.savefig(f"{savefig_folder}/{param}_time{t.year}.png")
            plt.show()


def plot_lat_lon_range(lat_min, lat_max, lon_min, lon_max):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_global()
    ax.gridlines(draw_labels=True)
    plt.plot([lon_min, lon_max], [lat_min, lat_min], color='blue', linewidth=2)  # horizontal line
    plt.plot([lon_min, lon_max], [lat_max, lat_max], color='blue', linewidth=2)  # horizontal line
    plt.plot([lon_min, lon_min], [lat_min, lat_max], color='blue', linewidth=2)  # vertical line
    plt.plot([lon_max, lon_max], [lat_min, lat_max], color='blue', linewidth=2)  # vertical line
    plt.show()
