import unittest

from project_configuration import globals
from src.database.communication import create_connection
from src.preprocessing.gridding import Grid, GridManager
import xarray as xr
import numpy as np
import pandas as pd
from datetimerange import DateTimeRange
from dateutil.relativedelta import relativedelta
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
import logging
import sys
import gc
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, MiniBatchKMeans
from matplotlib import cm
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.patch import geos_to_path
import itertools
from scipy.optimize import minimize
from scipy import optimize
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score
from project_configuration import globals
from src.database.communication import create_connection
from src.preprocessing.gridding import Grid, get_missing_value_info_offline, plot_missing_value_info, \
    create_wide_table_offline, plot_missing_value_info_map, GridManager, load_wide_table, \
    create_wide_table_online
from src.database.communication import create_connection
from src.database.information import get_table_as_df, does_table_exist, get_num_samples

db_path = "C:/users/yvjennig/pycharmprojects/data/4_comfort_averaged.sqlite"
test_grids = {"grid_A": {"param_tables": ["P_NITRATE"],
                         "lat_min": 0,
                         "lat_max": 50,
                         "dlat": 10,
                         "lon_min": 0,
                         "lon_max": 50,
                         "dlon": 10,
                         "z_min": None,
                         "z_max": None,
                         "dz": None,
                         "z_array": np.array([0, 1000, 5000]),
                         "time_min": "1980-01-01 00:00:00",
                         "time_max": "2020-07-08 04:45:00",
                         "mode": "Y",
                         "selection": None,
                         "dtime": 300,
                         "bathymetry_grid_path": "C:/users/yvjennig/pycharmprojects/data/bathymetry/"
                                                 "gebco_2022_sub_ice_topo/gebco_2022_sub_ice_topo.nc",
                         "lat_variable": "lat",
                         "lon_variable": "lon",
                         "depth_variable": "elevation",
                         "note": "Small area, 10 degree, zarray"},
              "grid_B": {"param_tables": ["P_TEMPERATURE"],
                         "lat_min": 0,
                         "lat_max": 50,
                         "dlat": 10,
                         "lon_min": 0,
                         "lon_max": 50,
                         "dlon": 10,
                         "z_min": 0,
                         "z_max": 100,
                         "dz": 100,
                         "z_array": None,
                         "time_min": "1980-01-01 00:00:00",
                         "time_max": "1982-03-12 00:00:00",
                         "mode": "YM",
                         "selection": None,
                         "dtime": 2,
                         "bathymetry_grid_path": "C:/users/yvjennig/pycharmprojects/data/bathymetry/"
                                                 "gebco_2022_sub_ice_topo/gebco_2022_sub_ice_topo.nc",
                         "lat_variable": "lat",
                         "lon_variable": "lon",
                         "depth_variable": "elevation",
                         "note": "Small area, 10 degree, z min max, time mode YM"},
              "grid_C": {"param_tables": ["P_TEMPERATURE"],
                         "lat_min": 0,
                         "lat_max": 50,
                         "dlat": 10,
                         "lon_min": 0,
                         "lon_max": 50,
                         "dlon": 10,
                         "z_min": 0,
                         "z_max": 100,
                         "dz": 100,
                         "z_array": None,
                         "time_min": "1980-01-01 00:00:00",
                         "time_max": "1980-02-02 00:00:00",
                         "mode": "YD",
                         "selection": np.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
                                                0, 1, 0, 1, 0, 0, 1, 1, 0]),
                         "dtime": 1,
                         "bathymetry_grid_path": "C:/users/yvjennig/pycharmprojects/data/bathymetry/"
                                                 "gebco_2022_sub_ice_topo/gebco_2022_sub_ice_topo.nc",
                         "lat_variable": "lat",
                         "lon_variable": "lon",
                         "depth_variable": "elevation",
                         "note": "Small area, 10 degree, z min max, time mode YD incl. selection"},
              "grid_D": {"param_tables": ["P_TEMPERATURE"],
                         "lat_min": 0,
                         "lat_max": 50,
                         "dlat": 10,
                         "lon_min": 0,
                         "lon_max": 50,
                         "dlon": 10,
                         "z_min": 0,
                         "z_max": 100,
                         "dz": 100,
                         "z_array": None,
                         "time_min": "1980-01-01 00:00:00",
                         "time_max": "1982-08-12 00:00:00",
                         "mode": "M",
                         "selection": None,
                         "dtime": 1,
                         "bathymetry_grid_path": "C:/users/yvjennig/pycharmprojects/data/bathymetry/"
                                                 "gebco_2022_sub_ice_topo/gebco_2022_sub_ice_topo.nc",
                         "lat_variable": "lat",
                         "lon_variable": "lon",
                         "depth_variable": "elevation",
                         "note": "Small area, 10 degree, z min max, time mode M"},
              "grid_E": {"param_tables": ["P_TEMPERATURE"],
                         "lat_min": 0,
                         "lat_max": 50,
                         "dlat": 10,
                         "lon_min": 0,
                         "lon_max": 50,
                         "dlon": 10,
                         "z_min": 0,
                         "z_max": 100,
                         "dz": 100,
                         "z_array": None,
                         "time_min": "1980-01-01 00:00:00",
                         "time_max": "1982-03-12 00:00:00",
                         "mode": "MD",
                         "selection": None,
                         "dtime": 1,
                         "bathymetry_grid_path": "C:/users/yvjennig/pycharmprojects/data/bathymetry/"
                                                 "gebco_2022_sub_ice_topo/gebco_2022_sub_ice_topo.nc",
                         "lat_variable": "lat",
                         "lon_variable": "lon",
                         "depth_variable": "elevation",
                         "note": "Small area, 10 degree, z min max, time mode MD"},
              "grid_F": {"param_tables": ["P_TEMPERATURE"],
                         "lat_min": 0,
                         "lat_max": 50,
                         "dlat": 10,
                         "lon_min": 0,
                         "lon_max": 50,
                         "dlon": 10,
                         "z_min": 0,
                         "z_max": 100,
                         "dz": 100,
                         "z_array": None,
                         "time_min": "1980-12-27 00:00:00",
                         "time_max": "1981-01-05 00:00:00",
                         "mode": "D",
                         "selection": None,
                         "dtime": 1,
                         "bathymetry_grid_path": "C:/users/yvjennig/pycharmprojects/data/bathymetry/"
                                                 "gebco_2022_sub_ice_topo/gebco_2022_sub_ice_topo.nc",
                         "lat_variable": "lat",
                         "lon_variable": "lon",
                         "depth_variable": "elevation",
                         "note": "Small area, 10 degree, z min max, time mode D"},
              "grid_G": {"param_tables": ["P_TEMPERATURE"],
                         "lat_min": -90,
                         "lat_max": 90,
                         "dlat": 10,
                         "lon_min": -180,
                         "lon_max": 180,
                         "dlon": 10,
                         "z_min": None,
                         "z_max": None,
                         "dz": None,
                         "z_array": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                              150, 200, 250, 300, 350, 400, 450, 500,
                                              600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
                                              1500, 1600, 1700, 1800, 1900, 2000,
                                              2500, 3000, 3500, 4000, 4500, 5000, 5500]),
                         "time_min": "1980-01-01 00:00:00",
                         "time_max": "2020-07-08 04:45:00",
                         "mode": "Y",
                         "selection": None,
                         "dtime": 300,
                         "bathymetry_grid_path": "C:/users/yvjennig/pycharmprojects/data/bathymetry/"
                                                 "gebco_2022_sub_ice_topo/gebco_2022_sub_ice_topo.nc",
                         "lat_variable": "lat",
                         "lon_variable": "lon",
                         "depth_variable": "elevation",
                         "note": "global grid, fine resolution in lat, lon, depth. Time average."}
              }


def load_grid_from_config(grid_name, use_database=False):
    grid = test_grids[grid_name]
    lat_min = grid["lat_min"]
    lat_max = grid["lat_max"]
    dlat = grid["dlat"]
    lon_min = grid["lon_min"]
    lon_max = grid["lon_max"]
    dlon = grid["dlon"]
    z_min = grid["z_min"]
    z_max = grid["z_max"]
    dz = grid["dz"]
    z_array = grid["z_array"]
    time_min = grid["time_min"]
    time_max = grid["time_max"]
    mode = grid["mode"]
    selection = grid["selection"]
    dtime = grid["dtime"]
    bathymetry_grid_path = grid["bathymetry_grid_path"]
    lat_variable = grid["lat_variable"]
    lon_variable = grid["lon_variable"]
    depth_variable = grid["depth_variable"]

    if use_database:
        gm = GridManager(db_path=db_path, grid_info_table="grid_info")
        grid = gm.create_grid(lat_min=lat_min, lat_max=lat_max, dlat=dlat,
                              lon_min=lon_min, lon_max=lon_max, dlon=dlon,
                              z_min=z_min, z_max=z_max, dz=dz, z_array=z_array,
                              time_min=time_min, time_max=time_max, dtime=dtime, mode=mode,
                              selection=selection,
                              bathymetry_grid_path=bathymetry_grid_path,
                              lat_variable=lat_variable, lon_variable=lon_variable,
                              depth_variable=depth_variable)
    else:
        grid = Grid(lat_min=lat_min, lat_max=lat_max, dlat=dlat,
                    lon_min=lon_min, lon_max=lon_max, dlon=dlon,
                    z_min=z_min, z_max=z_max, dz=dz, z_array=z_array,
                    time_min=time_min, time_max=time_max, dtime=dtime, mode=mode,
                    selection=selection,
                    bathymetry_grid_path=bathymetry_grid_path,
                    lat_variable=lat_variable, lon_variable=lon_variable,
                    depth_variable=depth_variable)
    return grid


class GridManagerTest(unittest.TestCase):

    def test_remove_grid(self):
        """ Checks if removing a non-existing grid does not yield an error. Also, it checks if a grid and associated
        mapped and wide tables are removed from the database.
        Prerequisite: map_tables, create_wide_table_online must work. """
        gm = GridManager(db_path=db_path, grid_info_table="grid_info")
        conn = create_connection(db_path)

        # try to remove a non-existing grid
        gm.remove_grid(grid_id="000")

        # create a grid, a mapped table and a wide table
        grid = load_grid_from_config("grid_A", use_database=True)
        mapped = grid.map_tables(connection=conn, param_tables=["P_ARGON", "P_NITRATE"], using_database=True)
        wide = create_wide_table_online(connection=conn, grid_id=grid.grid_id, param_tables=["P_ARGON", "P_NITRATE"])

        # remove existing grid
        gm.remove_grid(grid.grid_id)

        # check if grid and corresponding mapped and wide tables are removed from grid_info
        df_info = get_table_as_df(conn, "grid_info")
        self.assertFalse(grid.grid_id in df_info["grid_id"].to_list())

        for mapped_table in mapped:
            self.assertFalse(does_table_exist(conn=conn, table_name=mapped_table, table_type="table"))
        self.assertFalse(does_table_exist(conn=conn, table_name=wide, table_type="table"))

    def test_remove_all_grids(self):
        """ Check if all grids are removed from the database and grid_info table.
        Prerequisite: remove_grid must work. """
        gm = GridManager(db_path=db_path, grid_info_table="grid_info")
        conn = create_connection(db_path)

        # adding grids
        grid_a = load_grid_from_config("grid_A", use_database=True)
        grid_b = load_grid_from_config("grid_B", use_database=True)
        grid_c = load_grid_from_config("grid_C", use_database=True)

        # remove all grids
        gm.remove_all_grids()

        # check if grids are removed from database
        self.assertFalse(does_table_exist(conn=conn, table_name=grid_a.grid_name, table_type="table"))
        self.assertFalse(does_table_exist(conn=conn, table_name=grid_b.grid_name, table_type="table"))
        self.assertFalse(does_table_exist(conn=conn, table_name=grid_c.grid_name, table_type="table"))

        # check if grids are removed from grid_info
        df_info = get_table_as_df(conn, "grid_info")
        self.assertTrue(df_info.empty)

    def test_load_grid(self):
        """ Check if loading different grids works. Checks if loading a not-existing grid returns None. """
        gm = GridManager(db_path=db_path, grid_info_table="grid_info")

        # load existing grids
        for grid_name in ["grid_A", "grid_B", "grid_C"]:
            grid = load_grid_from_config(grid_name, use_database=True)  # create grid
            loaded = gm.load_grid(grid_id=grid.grid_id,  # load the new grid using the load-function in GridManager
                                  bathymetry_grid_path="C:/Users/yvjennig/PycharmProjects/data/bathymetry/"
                                                       "gebco_2022_sub_ice_topo/GEBCO_2022_sub_ice_topo.nc",
                                  lat_variable="lat", lon_variable="lon", depth_variable="elevation")
            self.assertIsNotNone(loaded)
            gm.remove_grid(grid_id=grid.grid_id)

        # load not-existing grid
        self.assertIsNone(gm.load_grid(grid_id="000"))

    def test_create_grid(self):
        """ Checks if different grids are created in database. Checks if existing grid is not re-created. """
        gm = GridManager(db_path=db_path, grid_info_table="grid_info")
        conn = create_connection(db_path)

        # check if grids are created successfully (function returns not None)
        grids = []
        for grid_name in ["grid_A", "grid_B", "grid_C"]:
            grid_config = test_grids[grid_name]
            grid = gm.create_grid(
                lat_min=grid_config["lat_min"], lat_max=grid_config["lat_max"], dlat=grid_config["dlat"],
                lon_min=grid_config["lon_min"], lon_max=grid_config["lon_max"], dlon=grid_config["dlon"],
                z_min=grid_config["z_min"], z_max=grid_config["z_max"], dz=grid_config["dz"],
                z_array=grid_config["z_array"],
                time_min=grid_config["time_min"], time_max=grid_config["time_max"], mode=grid_config["mode"],
                selection=grid_config["selection"], dtime=grid_config["dtime"],
                bathymetry_grid_path=grid_config["bathymetry_grid_path"], lat_variable=grid_config["lat_variable"],
                lon_variable=grid_config["lon_variable"], depth_variable=grid_config["depth_variable"])
            grids.append(grid)
            self.assertIsNotNone(grid)

        # check if existing grid is not re-created
        df_info_before = get_table_as_df(conn=conn, table_name="grid_info")
        grid_config = test_grids["grid_A"]
        grid = gm.create_grid(
            lat_min=grid_config["lat_min"], lat_max=grid_config["lat_max"], dlat=grid_config["dlat"],
            lon_min=grid_config["lon_min"], lon_max=grid_config["lon_max"], dlon=grid_config["dlon"],
            z_min=grid_config["z_min"], z_max=grid_config["z_max"], dz=grid_config["dz"],
            z_array=grid_config["z_array"],
            time_min=grid_config["time_min"], time_max=grid_config["time_max"], mode=grid_config["mode"],
            selection=grid_config["selection"], dtime=grid_config["dtime"],
            bathymetry_grid_path=grid_config["bathymetry_grid_path"], lat_variable=grid_config["lat_variable"],
            lon_variable=grid_config["lon_variable"], depth_variable=grid_config["depth_variable"])
        df_info_after = get_table_as_df(conn=conn, table_name="grid_info")
        self.assertEqual(len(df_info_before), len(df_info_after))  # check if no grid was added to grid_info table
        self.assertEqual(grids[0].grid_id, grid.grid_id)  # check if grid was loaded by comparing IDs

        # remove grids
        for grid in grids:
            gm.remove_grid(grid_id=grid.grid_id)

    def test_get_grid_id_from_params(self):
        """ Check if grids with different parameters are found by the function. Also, it checks if it returns None if
          the grid_id does not exists. """
        gm = GridManager(db_path=db_path, grid_info_table="grid_info")

        for grid_name in ["grid_A", "grid_B", "grid_C"]:
            # add grid
            grid = load_grid_from_config(grid_name, use_database=True)

            # check if grid_id is correct
            returned_id = gm.get_grid_id_from_params(lat_min=grid.lat_min, lat_max=grid.lat_max, dlat=grid.dlat,
                                                     lon_min=grid.lon_min, lon_max=grid.lon_max, dlon=grid.dlon,
                                                     z_min=grid.z_min, z_max=grid.z_max, dz=grid.dz,
                                                     z_array=grid.z_array,
                                                     time_min=grid.time_min, time_max=grid.time_max, mode=grid.mode,
                                                     dtime=grid.dtime, selection=grid.selection)
            self.assertEqual(grid.grid_id, returned_id)

            # remove grid
            gm.remove_grid(grid.grid_id)

            # if grid does not exist, expect None
            returned_id = gm.get_grid_id_from_params(lat_min=grid.lat_min, lat_max=grid.lat_max, dlat=grid.dlat,
                                                     lon_min=grid.lon_min, lon_max=grid.lon_max, dlon=grid.dlon,
                                                     z_min=grid.z_min, z_max=grid.z_max, dz=grid.dz,
                                                     z_array=grid.z_array,
                                                     time_min=grid.time_min, time_max=grid.time_max, mode=grid.mode,
                                                     dtime=grid.dtime, selection=grid.selection)
            self.assertIsNone(returned_id)

    def test_does_grid_exist(self):
        """ Checks if the function can recognize existence or non-existence of a grid.
        Prerequisite: Remove and create grid functions must work. """
        gm = GridManager(db_path=db_path, grid_info_table="grid_info")

        # existing grid (first create grid, then check if the function recognizes it)
        grid = load_grid_from_config("grid_A", use_database=True)
        self.assertTrue(gm.does_grid_exist(grid.grid_id))

        # not existing grid
        grid_id = "000"
        self.assertFalse(gm.does_grid_exist(grid_id))

    def test_add_grid(self):
        """ Check if new grid appears in grid_info (does_grid_exist must work for this) and if entries in grid_info
        correspond to initial grid specification (from the test_grids) """
        gm = GridManager(db_path=db_path, grid_info_table="grid_info")
        conn = create_connection(db_path)

        test_grid_names = ["grid_A", "grid_B"]  # one with z_array, one with z_min/z_max/dz
        for grid_name in test_grid_names:
            grid = load_grid_from_config(grid_name, use_database=True)  # create grid
            grid_config = test_grids[grid_name]

            # check if grids exist in grid_info
            self.assertTrue(gm.does_grid_exist(grid.grid_id))

            # check entries in grid_info table
            df_info = get_table_as_df(conn, "grid_info")
            entry = df_info[df_info["grid_id"] == grid.grid_id]

            if entry["z_min"].values[0] is None or entry["z_min"].values[0] == "None":
                self.assertIsNone(grid_config["z_min"])
            else:
                self.assertEqual(float(entry["z_min"].values[0]), grid_config["z_min"])

            if entry["z_max"].values[0] is None or entry["z_max"].values[0] == "None":
                self.assertIsNone(grid_config["z_max"])
            else:
                self.assertEqual(float(entry["z_max"].values[0]), grid_config["z_max"])

            if entry["dz"].values[0] is None or entry["dz"].values[0] == "None":
                self.assertIsNone(grid_config["dz"])
            else:
                self.assertEqual(float(entry["dz"].values[0]), grid_config["dz"])

            if entry["z_array"].values[0] is None or entry["z_array"].values[0] == "None":
                self.assertIsNone(grid_config["z_array"])
            else:
                z_array = np.array(entry["z_array"].values[0].replace("]", "").replace("[", "").split(sep=",")) \
                    .astype(float)
                np.testing.assert_array_equal(z_array, grid_config["z_array"])

            if entry["selection"].values[0] is None or entry["selection"].values[0] == "None":
                self.assertIsNone(grid_config["selection"])
            else:
                selection = np.array(entry["selection"].values[0].replace("]", "").replace("[", "").split(sep=",")) \
                    .astype(float)
                np.testing.assert_array_equal(selection, grid_config["selection"])

            self.assertEqual(entry["lat_min"].values[0], grid_config["lat_min"])
            self.assertEqual(entry["lat_max"].values[0], grid_config["lat_max"])
            self.assertEqual(entry["dlat"].values[0], grid_config["dlat"])
            self.assertEqual(entry["lon_min"].values[0], grid_config["lon_min"])
            self.assertEqual(entry["lon_max"].values[0], grid_config["lon_max"])
            self.assertEqual(entry["dlon"].values[0], grid_config["dlon"])
            self.assertEqual(entry["time_min"].values[0], grid_config["time_min"])
            self.assertEqual(entry["time_max"].values[0], grid_config["time_max"])
            self.assertEqual(entry["mode"].values[0], grid_config["mode"])
            self.assertEqual(entry["dtime"].values[0], grid_config["dtime"])

            # clean up
            gm.remove_grid(grid.grid_id)


class GridTest(unittest.TestCase):

    def test_generate_grid_id(self):
        """ Checks if grid_id is not None and an integer. """
        grid = load_grid_from_config("grid_A")
        self.assertIsNotNone(grid.grid_id)
        self.assertIsInstance(grid.grid_id, int)

    def test_create_time_array(self):
        """ Checks if the array is a numpy array, test examples for selection, step size and all different modes. """
        # check 1 time step, mode=year
        grid_a = load_grid_from_config("grid_A")
        self.assertIsInstance(grid_a.time_array, np.ndarray)
        self.assertEqual(grid_a.time_array, np.array(["1980-01-01 00:00:00"]))

        # check mode=year-month
        grid_b = load_grid_from_config("grid_B")
        self.assertIsInstance(grid_b.time_array, np.ndarray)
        np.testing.assert_array_equal(grid_b.time_array, np.array(["1980-01-01 00:00:00", "1980-03-01 00:00:00",
                                                                   "1980-05-01 00:00:00", "1980-07-01 00:00:00",
                                                                   "1980-09-01 00:00:00", "1980-11-01 00:00:00",
                                                                   "1981-01-01 00:00:00", "1981-03-01 00:00:00",
                                                                   "1981-05-01 00:00:00", "1981-07-01 00:00:00",
                                                                   "1981-09-01 00:00:00", "1981-11-01 00:00:00",
                                                                   "1982-01-01 00:00:00", "1982-03-01 00:00:00"]))

        # check mode=year-day with a selection
        grid_c = load_grid_from_config("grid_C")
        self.assertIsInstance(grid_c.time_array, np.ndarray)
        np.testing.assert_array_equal(grid_c.time_array,
                                      np.array(["1980-01-01 00:00:00", "1980-01-02 00:00:00", "1980-01-04 00:00:00",
                                                "1980-01-05 00:00:00", "1980-01-06 00:00:00", "1980-01-08 00:00:00",
                                                "1980-01-10 00:00:00", "1980-01-14 00:00:00", "1980-01-15 00:00:00",
                                                "1980-01-16 00:00:00", "1980-01-17 00:00:00", "1980-01-19 00:00:00",
                                                "1980-01-20 00:00:00", "1980-01-22 00:00:00", "1980-01-23 00:00:00",
                                                "1980-01-24 00:00:00", "1980-01-26 00:00:00", "1980-01-28 00:00:00",
                                                "1980-01-31 00:00:00", "1980-02-01 00:00:00"]))

        # check mode=month
        grid_d = load_grid_from_config("grid_D")
        self.assertIsInstance(grid_d.time_array, np.ndarray)
        np.testing.assert_array_equal(grid_d.time_array, np.array(["01", "02", "03", "04", "05", "06",
                                                                   "07", "08", "09", "10", "11", "12"]))

        # check mode=month-day
        grid_e = load_grid_from_config("grid_E")
        self.assertIsInstance(grid_e.time_array, np.ndarray)
        np.testing.assert_array_equal(grid_e.time_array, np.array(
            ['01-01 00:00:00', '01-02 00:00:00', '01-03 00:00:00', '01-04 00:00:00', '01-05 00:00:00', '01-06 00:00:00',
             '01-07 00:00:00', '01-08 00:00:00', '01-09 00:00:00', '01-10 00:00:00', '01-11 00:00:00', '01-12 00:00:00',
             '01-13 00:00:00', '01-14 00:00:00', '01-15 00:00:00', '01-16 00:00:00', '01-17 00:00:00', '01-18 00:00:00',
             '01-19 00:00:00', '01-20 00:00:00', '01-21 00:00:00', '01-22 00:00:00', '01-23 00:00:00', '01-24 00:00:00',
             '01-25 00:00:00', '01-26 00:00:00', '01-27 00:00:00', '01-28 00:00:00', '01-29 00:00:00', '01-30 00:00:00',
             '01-31 00:00:00', '02-01 00:00:00', '02-02 00:00:00', '02-03 00:00:00', '02-04 00:00:00', '02-05 00:00:00',
             '02-06 00:00:00', '02-07 00:00:00', '02-08 00:00:00', '02-09 00:00:00', '02-10 00:00:00', '02-11 00:00:00',
             '02-12 00:00:00', '02-13 00:00:00', '02-14 00:00:00', '02-15 00:00:00', '02-16 00:00:00', '02-17 00:00:00',
             '02-18 00:00:00', '02-19 00:00:00', '02-20 00:00:00', '02-21 00:00:00', '02-22 00:00:00', '02-23 00:00:00',
             '02-24 00:00:00', '02-25 00:00:00', '02-26 00:00:00', '02-27 00:00:00', '02-28 00:00:00', '02-29 00:00:00',
             '03-01 00:00:00', '03-02 00:00:00', '03-03 00:00:00', '03-04 00:00:00', '03-05 00:00:00', '03-06 00:00:00',
             '03-07 00:00:00', '03-08 00:00:00', '03-09 00:00:00', '03-10 00:00:00', '03-11 00:00:00', '03-12 00:00:00',
             '03-13 00:00:00', '03-14 00:00:00', '03-15 00:00:00', '03-16 00:00:00', '03-17 00:00:00', '03-18 00:00:00',
             '03-19 00:00:00', '03-20 00:00:00', '03-21 00:00:00', '03-22 00:00:00', '03-23 00:00:00', '03-24 00:00:00',
             '03-25 00:00:00', '03-26 00:00:00', '03-27 00:00:00', '03-28 00:00:00', '03-29 00:00:00', '03-30 00:00:00',
             '03-31 00:00:00', '04-01 00:00:00', '04-02 00:00:00', '04-03 00:00:00', '04-04 00:00:00', '04-05 00:00:00',
             '04-06 00:00:00', '04-07 00:00:00', '04-08 00:00:00', '04-09 00:00:00', '04-10 00:00:00', '04-11 00:00:00',
             '04-12 00:00:00', '04-13 00:00:00', '04-14 00:00:00', '04-15 00:00:00', '04-16 00:00:00', '04-17 00:00:00',
             '04-18 00:00:00', '04-19 00:00:00', '04-20 00:00:00', '04-21 00:00:00', '04-22 00:00:00', '04-23 00:00:00',
             '04-24 00:00:00', '04-25 00:00:00', '04-26 00:00:00', '04-27 00:00:00', '04-28 00:00:00', '04-29 00:00:00',
             '04-30 00:00:00', '05-01 00:00:00', '05-02 00:00:00', '05-03 00:00:00', '05-04 00:00:00', '05-05 00:00:00',
             '05-06 00:00:00', '05-07 00:00:00', '05-08 00:00:00', '05-09 00:00:00', '05-10 00:00:00', '05-11 00:00:00',
             '05-12 00:00:00', '05-13 00:00:00', '05-14 00:00:00', '05-15 00:00:00', '05-16 00:00:00', '05-17 00:00:00',
             '05-18 00:00:00', '05-19 00:00:00', '05-20 00:00:00', '05-21 00:00:00', '05-22 00:00:00', '05-23 00:00:00',
             '05-24 00:00:00', '05-25 00:00:00', '05-26 00:00:00', '05-27 00:00:00', '05-28 00:00:00', '05-29 00:00:00',
             '05-30 00:00:00', '05-31 00:00:00', '06-01 00:00:00', '06-02 00:00:00', '06-03 00:00:00', '06-04 00:00:00',
             '06-05 00:00:00', '06-06 00:00:00', '06-07 00:00:00', '06-08 00:00:00', '06-09 00:00:00', '06-10 00:00:00',
             '06-11 00:00:00', '06-12 00:00:00', '06-13 00:00:00', '06-14 00:00:00', '06-15 00:00:00', '06-16 00:00:00',
             '06-17 00:00:00', '06-18 00:00:00', '06-19 00:00:00', '06-20 00:00:00', '06-21 00:00:00', '06-22 00:00:00',
             '06-23 00:00:00', '06-24 00:00:00', '06-25 00:00:00', '06-26 00:00:00', '06-27 00:00:00', '06-28 00:00:00',
             '06-29 00:00:00', '06-30 00:00:00', '07-01 00:00:00', '07-02 00:00:00', '07-03 00:00:00', '07-04 00:00:00',
             '07-05 00:00:00', '07-06 00:00:00', '07-07 00:00:00', '07-08 00:00:00', '07-09 00:00:00', '07-10 00:00:00',
             '07-11 00:00:00', '07-12 00:00:00', '07-13 00:00:00', '07-14 00:00:00', '07-15 00:00:00', '07-16 00:00:00',
             '07-17 00:00:00', '07-18 00:00:00', '07-19 00:00:00', '07-20 00:00:00', '07-21 00:00:00', '07-22 00:00:00',
             '07-23 00:00:00', '07-24 00:00:00', '07-25 00:00:00', '07-26 00:00:00', '07-27 00:00:00', '07-28 00:00:00',
             '07-29 00:00:00', '07-30 00:00:00', '07-31 00:00:00', '08-01 00:00:00', '08-02 00:00:00', '08-03 00:00:00',
             '08-04 00:00:00', '08-05 00:00:00', '08-06 00:00:00', '08-07 00:00:00', '08-08 00:00:00', '08-09 00:00:00',
             '08-10 00:00:00', '08-11 00:00:00', '08-12 00:00:00', '08-13 00:00:00', '08-14 00:00:00', '08-15 00:00:00',
             '08-16 00:00:00', '08-17 00:00:00', '08-18 00:00:00', '08-19 00:00:00', '08-20 00:00:00', '08-21 00:00:00',
             '08-22 00:00:00', '08-23 00:00:00', '08-24 00:00:00', '08-25 00:00:00', '08-26 00:00:00', '08-27 00:00:00',
             '08-28 00:00:00', '08-29 00:00:00', '08-30 00:00:00', '08-31 00:00:00', '09-01 00:00:00', '09-02 00:00:00',
             '09-03 00:00:00', '09-04 00:00:00', '09-05 00:00:00', '09-06 00:00:00', '09-07 00:00:00', '09-08 00:00:00',
             '09-09 00:00:00', '09-10 00:00:00', '09-11 00:00:00', '09-12 00:00:00', '09-13 00:00:00', '09-14 00:00:00',
             '09-15 00:00:00', '09-16 00:00:00', '09-17 00:00:00', '09-18 00:00:00', '09-19 00:00:00', '09-20 00:00:00',
             '09-21 00:00:00', '09-22 00:00:00', '09-23 00:00:00', '09-24 00:00:00', '09-25 00:00:00', '09-26 00:00:00',
             '09-27 00:00:00', '09-28 00:00:00', '09-29 00:00:00', '09-30 00:00:00', '10-01 00:00:00', '10-02 00:00:00',
             '10-03 00:00:00', '10-04 00:00:00', '10-05 00:00:00', '10-06 00:00:00', '10-07 00:00:00', '10-08 00:00:00',
             '10-09 00:00:00', '10-10 00:00:00', '10-11 00:00:00', '10-12 00:00:00', '10-13 00:00:00', '10-14 00:00:00',
             '10-15 00:00:00', '10-16 00:00:00', '10-17 00:00:00', '10-18 00:00:00', '10-19 00:00:00', '10-20 00:00:00',
             '10-21 00:00:00', '10-22 00:00:00', '10-23 00:00:00', '10-24 00:00:00', '10-25 00:00:00', '10-26 00:00:00',
             '10-27 00:00:00', '10-28 00:00:00', '10-29 00:00:00', '10-30 00:00:00', '10-31 00:00:00', '11-01 00:00:00',
             '11-02 00:00:00', '11-03 00:00:00', '11-04 00:00:00', '11-05 00:00:00', '11-06 00:00:00', '11-07 00:00:00',
             '11-08 00:00:00', '11-09 00:00:00', '11-10 00:00:00', '11-11 00:00:00', '11-12 00:00:00', '11-13 00:00:00',
             '11-14 00:00:00', '11-15 00:00:00', '11-16 00:00:00', '11-17 00:00:00', '11-18 00:00:00', '11-19 00:00:00',
             '11-20 00:00:00', '11-21 00:00:00', '11-22 00:00:00', '11-23 00:00:00', '11-24 00:00:00', '11-25 00:00:00',
             '11-26 00:00:00', '11-27 00:00:00', '11-28 00:00:00', '11-29 00:00:00', '11-30 00:00:00', '12-01 00:00:00',
             '12-02 00:00:00', '12-03 00:00:00', '12-04 00:00:00', '12-05 00:00:00', '12-06 00:00:00', '12-07 00:00:00',
             '12-08 00:00:00', '12-09 00:00:00', '12-10 00:00:00', '12-11 00:00:00', '12-12 00:00:00', '12-13 00:00:00',
             '12-14 00:00:00', '12-15 00:00:00', '12-16 00:00:00', '12-17 00:00:00', '12-18 00:00:00', '12-19 00:00:00',
             '12-20 00:00:00', '12-21 00:00:00', '12-22 00:00:00', '12-23 00:00:00', '12-24 00:00:00', '12-25 00:00:00',
             '12-26 00:00:00', '12-27 00:00:00', '12-28 00:00:00', '12-29 00:00:00', '12-30 00:00:00',
             '12-31 00:00:00']))

        # check mode=day
        grid_f = load_grid_from_config("grid_F")
        self.assertIsInstance(grid_f.time_array, np.ndarray)
        np.testing.assert_array_equal(grid_f.time_array,
                                      np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
                                                '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
                                                '23', '24', '25', '26', '27', '28', '29', '30', '31']))

    def test_create_grid(self):
        """ Check number of lat, lon, depth, time cells for different grids. """
        for grid_name in test_grids.keys():
            grid = load_grid_from_config(grid_name)

            # get number of cells
            num_lat_cells = len(grid.grid["LATITUDE"].value_counts())
            num_lon_cells = len(grid.grid["LONGITUDE"].value_counts())
            num_z_vals = len(grid.grid["LEV_M"].value_counts())
            num_time_vals = len(grid.grid["DATEANDTIME"].value_counts())
            num_cells = len(grid.grid)

            # check number of lat, lon, depths, time cells
            if grid.z_array is not None:
                self.assertEqual(num_z_vals, len(grid.z_array))
            else:
                self.assertEqual(num_z_vals, (abs(grid.z_min) + abs(grid.z_max)) / grid.dz)
            self.assertEqual(num_lat_cells, (abs(grid.lat_min) + abs(grid.lat_max)) / grid.dlat)
            self.assertEqual(num_lon_cells, (abs(grid.lon_min) + abs(grid.lon_max)) / grid.dlon)
            self.assertEqual(num_time_vals, len(grid.time_array))
            self.assertEqual(num_cells, num_lat_cells * num_lon_cells * num_z_vals * num_time_vals)

    def test_map_tables(self):
        # connect to database
        conn = create_connection(db_path)

        # create a dummy parameter table
        table_name = "P_TEST"
        param_tables = [table_name]
        p_test = pd.DataFrame({"VAL": [1, 2, 3, 3.1, 3.2, 3.3, 4, 5, 6, 7, 8, 9, 10, 11, 11.1, 11.2],
                               "LEV_M": [0, 50, 0, 100.01, 100, 80, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35],
                               "LATITUDE": [0, 9, 22, 43, -0.01, 32, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35],
                               "LONGITUDE": [0, 9, 22, 2, 45, 50.01, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35],
                               "DATEANDTIME": ["1980-01-01 00:00:00", "1981-01-08 00:00:00", "1982-01-14 00:00:00",
                                               "1982-01-14 00:00:00", "1982-01-14 00:00:00", "1982-01-14 00:00:00",
                                               # January
                                               "1980-02-09 00:00:00", "1980-02-19 00:00:00",  # February
                                               "1980-07-22 00:00:00",  # July
                                               "1980-09-03 00:00:00", "1980-09-17 00:00:00", "1980-09-30 00:00:00",
                                               "1980-09-30 00:00:00",  # September
                                               "1980-11-10 00:00:00",  # November
                                               "1983-01-01 00:00:00", "1979-12-31 00:00:00"]
                               # should be filtered out due to min, max
                               })
        p_test.to_sql(name=table_name, con=conn, if_exists="replace")  # write to database

        # create grid
        grid_d = load_grid_from_config("grid_D")

        # --- map test table to grid not using database
        mapped_tables_online = grid_d.map_tables(connection=conn, param_tables=param_tables, using_database=False,
                                                 include_z_max=True)
        mapped_offline = mapped_tables_online[0]

        # --- map test table to grid using database
        mapped_tables_offline = grid_d.map_tables(connection=conn, param_tables=param_tables, using_database=True,
                                                  include_z_max=True)
        mapped_online = get_table_as_df(conn=conn, table_name=mapped_tables_offline[0])

        # --- perform checks on both options (with and without database usage)
        self.assertEqual(len(mapped_tables_online), 1)  # check if there is one mapped table
        self.assertEqual(len(mapped_tables_offline), 1)

        for mapped in [mapped_offline, mapped_online]:
            # check if the mapped table has the right amount of cells in the 4 dimensions
            self.assertEqual(len(mapped),
                             len(grid_d.grid["LATITUDE"].value_counts()) * len(
                                 grid_d.grid["LONGITUDE"].value_counts()) *
                             len(grid_d.grid["LEV_M"].value_counts()) * len(grid_d.grid["DATEANDTIME"].value_counts()))

            # check if the data points were mapped correctly
            jan0_idx = mapped[(mapped["DATEANDTIME"] == "01") & (mapped["LATITUDE"] == 5) &
                              (mapped["LONGITUDE"] == 5) & (mapped["LEV_M"] == 0)].index
            self.assertEqual(mapped.iloc[jan0_idx][table_name].values[0], np.mean([1, 2]))
            self.assertEqual(mapped.iloc[jan0_idx]["median"].values[0], np.median([1, 2]))
            self.assertEqual(mapped.iloc[jan0_idx]["std"].values[0], np.std([1, 2], ddof=1))
            self.assertEqual(mapped.iloc[jan0_idx]["count"].values[0], len([1, 2]))

            jan1_idx = mapped[(mapped["DATEANDTIME"] == "01") & (mapped["LATITUDE"] == 25) &
                              (mapped["LONGITUDE"] == 25) & (mapped["LEV_M"] == 0)].index
            self.assertEqual(mapped.iloc[jan1_idx][table_name].values[0], 3)
            self.assertEqual(mapped.iloc[jan1_idx]["median"].values[0], np.median([3]))
            self.assertTrue(
                np.isnan(mapped.iloc[jan1_idx]["std"].values[0]))  # must be None as there is only 1 sample in list
            self.assertEqual(mapped.iloc[jan1_idx]["count"].values[0], 1)

            feb_idx = mapped[(mapped["DATEANDTIME"] == "02") & (mapped["LATITUDE"] == 35) &
                             (mapped["LONGITUDE"] == 35) & (mapped["LEV_M"] == 0)].index
            self.assertEqual(mapped.iloc[feb_idx][table_name].values[0], np.mean([4, 5]))
            self.assertEqual(mapped.iloc[feb_idx]["median"].values[0], np.median([4, 5]))
            self.assertEqual(mapped.iloc[feb_idx]["std"].values[0], np.std([4, 5], ddof=1))
            self.assertEqual(mapped.iloc[feb_idx]["count"].values[0], len([4, 5]))

            jul_idx = mapped[(mapped["DATEANDTIME"] == "07") & (mapped["LATITUDE"] == 35) &
                             (mapped["LONGITUDE"] == 35) & (mapped["LEV_M"] == 0)].index
            self.assertEqual(mapped.iloc[jul_idx][table_name].values[0], 6)
            self.assertEqual(mapped.iloc[jul_idx]["median"].values[0], 6)
            self.assertTrue(np.isnan(mapped.iloc[jul_idx]["std"].values[0]))
            self.assertEqual(mapped.iloc[jul_idx]["count"].values[0], 1)

            sep_idx = mapped[(mapped["DATEANDTIME"] == "09") & (mapped["LATITUDE"] == 35) &
                             (mapped["LONGITUDE"] == 35) & (mapped["LEV_M"] == 0)].index
            self.assertEqual(mapped.iloc[sep_idx][table_name].values[0], np.mean([7, 8, 9, 10]))
            self.assertEqual(mapped.iloc[sep_idx]["median"].values[0], np.median([7, 8, 9, 10]))
            self.assertEqual(mapped.iloc[sep_idx]["std"].values[0], np.std([7, 8, 9, 10], ddof=1))
            self.assertEqual(mapped.iloc[sep_idx]["count"].values[0], len([7, 8, 9, 10]))

            nov_idx = mapped[(mapped["DATEANDTIME"] == "11") & (mapped["LATITUDE"] == 35) &
                             (mapped["LONGITUDE"] == 35) & (mapped["LEV_M"] == 0)].index
            self.assertEqual(mapped.iloc[nov_idx][table_name].values[0], 11)
            self.assertEqual(mapped.iloc[nov_idx]["median"].values[0], 11)
            self.assertTrue(np.isnan(mapped.iloc[nov_idx]["std"].values[0]))
            self.assertEqual(mapped.iloc[nov_idx]["count"].values[0], 1)

            # all other cells must be empty
            empty_idxs = mapped[mapped[table_name].isna()].index.to_list()
            all_but_filled_idxs = [x for x in mapped.index.to_list() if x not in [jan0_idx, jan1_idx, feb_idx,
                                                                                  jul_idx, sep_idx, nov_idx]]
            self.assertSequenceEqual(empty_idxs, all_but_filled_idxs)

        # clean up database
        to_drop = [table_name, f"grid_{grid_d.grid_id}", f"P_TEST_{grid_d.grid_id}"]
        for td in to_drop:
            q = f"drop table {td};"
            conn.execute(q)
        q = "vacuum;"
        conn.execute(q)


class FreeFunctionsTest(unittest.TestCase):
    def test_drop_land_cells(self):
        pass

    def test_create_wide_table_offline(self):
        """ Checks if the number of grid cells of the wide table is equal or smaller than the number of grid cells
        of the pure grid. """
        # @todo add specific test case, i.e. specific grid with mapped table and wide table example
        conn = create_connection(db_path)

        # create a grid and map tables to it
        grid = load_grid_from_config("grid_A", use_database=False)
        mapped = grid.map_tables(connection=conn, param_tables=["P_ARGON", "P_NITRATE"], using_database=False)

        # create a wide table
        wide_water_land = create_wide_table_offline(mapped_tables=mapped, dropping_land_cells=False)
        wide_water = create_wide_table_offline(mapped_tables=mapped, dropping_land_cells=True)

        # check if the number of entries equals the number of entries of the original grid
        self.assertEqual(len(wide_water_land), len(grid.grid))
        self.assertLessEqual(len(wide_water), len(grid.grid))

    def test_create_wide_table_online(self):
        """ Checks if the number of grid cells of the wide table is equal or smaller than the number of grid cells
        of the pure grid. """
        gm = GridManager(db_path=db_path, grid_info_table="grid_info")
        conn = create_connection(db_path)

        # define grid
        param_tables = ["P_ARGON", "P_NITRATE"]
        grid_a = load_grid_from_config("grid_A", use_database=True)

        # create wide table
        wide_table_name = create_wide_table_online(connection=conn, grid_id=grid_a.grid_id, param_tables=param_tables)
        len_wide = get_num_samples(conn=conn, table_name=wide_table_name, table_type="table")

        # check if length of wide table is same length as the original grid
        self.assertEqual(len_wide, len(grid_a))

        # remove grid
        gm.remove_grid(grid_id=grid_a.grid_id)

    def test_load_wide_table(self):
        pass

    def test_get_missing_value_info_per_param(self):
        pass

    def test_get_missing_value_info_offline_per_param(self):
        pass

    def test_get_missing_value_info_offline(self):
        pass
