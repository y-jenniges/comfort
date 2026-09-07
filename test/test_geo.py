"""Tests for comfort.geo, i.e. geospatial utility functions."""
import numpy as np
import pandas as pd
import pytest

from comfort.geo import (
    along_track_distance,
    classify_from_grid,
    distance_to_coast,
    load_jenniges_provinces,
    load_longhurst,
    water_mass_masks,
    water_mass_statistics,
)

_has_geopandas = True
try:
    import geopandas
except ImportError:
    _has_geopandas = False


class TestDistanceToCoast:
    @staticmethod
    def _vertical_coastline():
        """Synthetic coastline: A vertical line at lon=0, lat from -10 to 10."""
        from shapely.geometry import LineString
        return LineString([(0.0, lat) for lat in range(-10, 11)])

    # Output is a pandas Series
    def test_returns_series(self):
        df = pd.DataFrame({"LATITUDE": [0.0], "LONGITUDE": [0.0]})
        result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert isinstance(result, pd.Series)

    # One distance value per input row
    def test_length_matches_input(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 5.0], "LONGITUDE": [1.0, 2.0]})
        result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert len(result) == 2

    # Output index matches the input DataFrame's index
    def test_preserves_index(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 0.0], "LONGITUDE": [1.0, 2.0]},
                          index=[10, 20])
        result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert list(result.index) == [10, 20]

    def test_point_on_coast_near_zero(self):
        # A point at lon=0 is on the coastline, i.e. distance should be ~0 km
        df = pd.DataFrame({"LATITUDE": [0.0], "LONGITUDE": [0.0]})
        result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert float(result.iloc[0]) < 1.0

    # Points farther from the coastline get a larger distance
    def test_distance_increases_with_separation(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 0.0], "LONGITUDE": [1.0, 5.0]})
        result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert result.iloc[0] < result.iloc[1]

    # Distance to the equator should be about right
    def test_approximate_distance_at_equator(self):
        # 1° of longitude at the equator ≈ 111 km
        df = pd.DataFrame({"LATITUDE": [0.0], "LONGITUDE": [1.0]})
        result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert 100.0 < float(result.iloc[0]) < 125.0

    # A list of coastline segments is accepted, not just a single geometry
    def test_list_of_geometries(self):
        from shapely.geometry import LineString
        geoms = [LineString([(0.0, lat), (0.0, lat + 1)]) for lat in range(-10, 10)]
        df = pd.DataFrame({"LATITUDE": [0.0], "LONGITUDE": [1.0]})
        result = distance_to_coast(df, coastline_geom=geoms)
        assert 100.0 < float(result.iloc[0]) < 125.0

    # Distances are never negative
    def test_non_negative(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 5.0, -5.0], "LONGITUDE": [3.0, 1.0, 7.0]})
        result = distance_to_coast(df, coastline_geom=self._vertical_coastline())
        assert (result >= 0).all()


class TestAlongTrackDistance:
    # The first station along the track is the distance origin
    def test_first_station_is_zero(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 0.0], "LONGITUDE": [0.0, 1.0]})
        result = along_track_distance(df)
        assert result.iloc[0] == 0.0

    # Cumulative distance grows monotonically along a straight track
    def test_monotonically_increases_along_straight_track(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 0.0, 0.0], "LONGITUDE": [0.0, 1.0, 2.0]})
        result = along_track_distance(df)
        assert result.iloc[0] < result.iloc[1] < result.iloc[2]

    # The distance the equator is about right
    def test_approximate_distance_at_equator(self):
        # 1 degree of longitude at the equator is about 111 km
        df = pd.DataFrame({"LATITUDE": [0.0, 0.0], "LONGITUDE": [0.0, 1.0]})
        result = along_track_distance(df)
        assert 100.0 < float(result.iloc[1]) < 125.0

    # Two depth levels at the same station (repeated lat/lon) share one distance
    def test_repeated_station_gets_same_distance(self):
        df = pd.DataFrame({
            "LATITUDE": [0.0, 0.0, 1.0],
            "LONGITUDE": [0.0, 0.0, 0.0],
            "LEV_M": [0, 10, 0],
        })
        result = along_track_distance(df)
        assert result.iloc[0] == result.iloc[1]
        assert result.iloc[2] > result.iloc[0]

    # Distances of out-of-order rows
    def test_order_col_reorders_stations(self):
        # Out-of-order rows: station at lon=2 was visited before lon=1
        df = pd.DataFrame({
            "LATITUDE": [0.0, 0.0, 0.0],
            "LONGITUDE": [0.0, 2.0, 1.0],
            "SEQ": [0, 2, 1],
        })
        result = along_track_distance(df, order_col="SEQ")

        # Ordered by SEQ: lon 0 -> 1 -> 2, so the lon=1 station is closer to start than lon=2
        dist_by_lon = dict(zip(df["LONGITUDE"], result))
        assert dist_by_lon[1.0] < dist_by_lon[2.0]

    # Output index matches the input DataFrame's index
    def test_preserves_index(self):
        df = pd.DataFrame({"LATITUDE": [0.0, 1.0], "LONGITUDE": [0.0, 0.0]}, index=[10, 20])
        result = along_track_distance(df)
        assert list(result.index) == [10, 20]

    # Distances are never negative
    def test_non_negative(self):
        df = pd.DataFrame({"LATITUDE": [0.0, -3.0, 5.0], "LONGITUDE": [3.0, 1.0, -7.0]})
        result = along_track_distance(df)
        assert (result >= 0).all()


def _obs_df():
    """Three observations: one north, one south, one outside both boxes."""
    return pd.DataFrame({
        "LATITUDE": [15.0, -10.0, 50.0],
        "LONGITUDE": [5.0, 5.0, 5.0],
        "VAL": [1.0, 2.0, 3.0],
    })


class TestWaterMassMasks:
    @staticmethod
    def _box_regions():
        """Two non-overlapping rectangular regions as a dict of coordinate lists."""
        return {
            "north": [(0.0, 10.0), (10.0, 10.0), (10.0, 20.0), (0.0, 20.0)],
            "south": [(0.0, -20.0), (10.0, -20.0), (10.0, 0.0), (0.0, 0.0)],
        }

    # Output is a DataFrame
    def test_returns_dataframe(self):
        result = water_mass_masks(_obs_df(), self._box_regions())
        assert isinstance(result, pd.DataFrame)

    # A region column is added
    def test_adds_region_column(self):
        result = water_mass_masks(_obs_df(), self._box_regions())
        assert "region" in result.columns

    # Observations are assigned to the region containing them
    def test_correct_assignments(self):
        result = water_mass_masks(_obs_df(), self._box_regions())
        assert result["region"].iloc[0] == "north"
        assert result["region"].iloc[1] == "south"

    # An observation outside every region gets NaN, not a wrong label
    def test_outside_region_is_nan(self):
        result = water_mass_masks(_obs_df(), self._box_regions())
        assert pd.isna(result["region"].iloc[2])

    # output_col lets the caller rename the added column
    def test_custom_output_col(self):
        result = water_mass_masks(_obs_df(), self._box_regions(), output_col="zone")
        assert "zone" in result.columns
        assert "region" not in result.columns

    # The input DataFrame is not modified in place
    def test_does_not_mutate_input(self):
        df = _obs_df()
        water_mass_masks(df, self._box_regions())
        assert "region" not in df.columns

    # A dict of shapely Polygons (not just coordinate lists) is accepted directly
    def test_dict_with_shapely_polygons(self):
        from shapely.geometry import Polygon
        regions = {
            "box": Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        }
        df = pd.DataFrame({"LATITUDE": [5.0, 50.0], "LONGITUDE": [5.0, 50.0]})
        result = water_mass_masks(df, regions)
        assert result["region"].iloc[0] == "box"
        assert pd.isna(result["region"].iloc[1])

    # A CSV of WKT polygons is accepted
    def test_csv_input(self, tmp_path):
        """water_mass_masks accepts a CSV file with WKT geometry column."""
        csv_path = tmp_path / "regions.csv"
        csv_path.write_text(
            "name,geometry\n"
            "north,\"POLYGON ((0 10, 10 10, 10 20, 0 20, 0 10))\"\n"
            "south,\"POLYGON ((0 -20, 10 -20, 10 0, 0 0, 0 -20))\"\n"
        )
        result = water_mass_masks(_obs_df(), str(csv_path))
        assert result["region"].iloc[0] == "north"
        assert result["region"].iloc[1] == "south"

    # An empty region set leaves every observation unlabelled
    def test_empty_regions_all_nan(self):
        result = water_mass_masks(_obs_df(), {})
        assert result["region"].isna().all()

    # Row count is unchanged by labelling
    def test_length_preserved(self):
        result = water_mass_masks(_obs_df(), self._box_regions())
        assert len(result) == len(_obs_df())

    # Output index matches the input DataFrame's index
    def test_index_preserved(self):
        df = _obs_df()
        df.index = [10, 20, 30]
        result = water_mass_masks(df, self._box_regions())
        assert list(result.index) == [10, 20, 30]


class TestWaterMassStatistics:
    @staticmethod
    def _labelled_df():
        df = _obs_df().copy()
        df["region"] = ["north", "south", np.nan]
        df["VAL2"] = [10.0, 20.0, 30.0]
        return df

    # Output is a DataFrame
    def test_returns_dataframe(self):
        result = water_mass_statistics(self._labelled_df(), param_cols=["VAL"])
        assert isinstance(result, pd.DataFrame)

    # Rows are indexed by region name
    def test_rows_are_regions(self):
        result = water_mass_statistics(self._labelled_df(), param_cols=["VAL"])
        assert set(result.index) == {"north", "south"}

    # Unlabelled (NaN-region) observations do not get their own row
    def test_nan_region_excluded(self):
        result = water_mass_statistics(self._labelled_df(), param_cols=["VAL"])
        assert np.nan not in result.index

    # Standard descriptive statistics are all present
    def test_statistics_columns(self):
        result = water_mass_statistics(self._labelled_df(), param_cols=["VAL"])
        assert set(result.columns.get_level_values(1)) >= {"count", "mean", "std", "min", "max"}

    # Multiple parameter columns each get their own statistics block
    def test_multiple_params(self):
        result = water_mass_statistics(self._labelled_df(), param_cols=["VAL", "VAL2"])
        assert ("VAL", "mean") in result.columns
        assert ("VAL2", "mean") in result.columns

    # Each region's count matches its number of observations
    def test_count_correct(self):
        result = water_mass_statistics(self._labelled_df(), param_cols=["VAL"])
        assert result.loc["north", ("VAL", "count")] == 1
        assert result.loc["south", ("VAL", "count")] == 1

    # Per-region mean is computed correctly
    def test_mean_correct(self):
        result = water_mass_statistics(self._labelled_df(), param_cols=["VAL"])
        assert result.loc["north", ("VAL", "mean")] == pytest.approx(1.0)
        assert result.loc["south", ("VAL", "mean")] == pytest.approx(2.0)

    # Single parameter as string (not list) still works
    def test_string_param_col(self):
        result = water_mass_statistics(self._labelled_df(), param_cols="VAL")
        assert ("VAL", "mean") in result.columns


@pytest.mark.skipif(not _has_geopandas, reason="geopandas not installed ([geo] extra)")
class TestLoadLonghurst:
    @staticmethod
    def _write_shapefile(tmp_path, code_col="ProvCode", name_col="ProvDescr"):
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame({
            code_col: [1, 2],
            name_col: ["North Atlantic Drift", "Gulf Stream"],
            "geometry": [
                Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                Polygon([(10, 0), (20, 0), (20, 10), (10, 10)]),
            ],
        }, crs="EPSG:4326")
        path = tmp_path / "longhurst.shp"
        gdf.to_file(path)
        return path

    # Column names (ProvCode/ProvDescr) are auto-detected and renamed
    def test_auto_detects_columns(self, tmp_path):
        result = load_longhurst(self._write_shapefile(tmp_path))
        assert {"province_code", "province_name", "geometry"} <= set(result.columns)

    # Province names survive the read/rename process
    def test_values_preserved(self, tmp_path):
        result = load_longhurst(self._write_shapefile(tmp_path))
        assert set(result["province_name"]) == {"North Atlantic Drift", "Gulf Stream"}

    # code_col/name_col override auto-detection of names
    def test_explicit_column_names(self, tmp_path):
        path = self._write_shapefile(tmp_path, code_col="code", name_col="label")
        result = load_longhurst(path, code_col="code", name_col="label")
        assert "province_code" in result.columns
        assert "province_name" in result.columns

    # Neither auto-detection nor an explicit column name found is a hard error
    def test_raises_when_columns_not_found(self, tmp_path):
        path = self._write_shapefile(tmp_path, code_col="foo", name_col="bar")
        with pytest.raises(ValueError, match="Could not detect"):
            load_longhurst(path)


class TestLoadJennigesProvinces:
    @staticmethod
    def _jenniges_csv(tmp_path):
        """Write a minimal Jenniges-style CSV and return its path."""
        path = tmp_path / "cluster_set.csv"
        path.write_text(
            "LEV_M,LATITUDE,LONGITUDE,label,color\n"
            "0,50.0,-30.0,1,#ff0000\n"
            "0,55.0,-25.0,2,#00ff00\n"
            "0,60.0,-20.0,1,#ff0000\n"
            "100,50.0,-30.0,3,#0000ff\n"
            "100,55.0,-25.0,3,#0000ff\n"
        )
        return path

    # Output is a DataFrame
    def test_returns_dataframe(self, tmp_path):
        result = load_jenniges_provinces(self._jenniges_csv(tmp_path))
        assert isinstance(result, pd.DataFrame)

    # depth=None returns every row across all depth levels
    def test_all_depths_when_none(self, tmp_path):
        result = load_jenniges_provinces(self._jenniges_csv(tmp_path), depth=None)
        assert len(result) == 5

    # An exact depth match filters to just that level
    def test_filter_to_depth(self, tmp_path):
        result = load_jenniges_provinces(self._jenniges_csv(tmp_path), depth=0)
        assert len(result) == 3
        assert (result["LEV_M"] == 0).all()

    # A depth with no exact match falls back to the closest available level
    def test_closest_depth(self, tmp_path):
        result = load_jenniges_provinces(self._jenniges_csv(tmp_path), depth=10)
        assert len(result) == 3
        assert (result["LEV_M"] == 0).all()

    # A missing file is a FileNotFoundError, not a silent empty result
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_jenniges_provinces(tmp_path / "nonexistent.csv")

    # A required column absent from the CSV raises with its name in the message
    def test_missing_column(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("LATITUDE,LONGITUDE\n50,30\n")
        with pytest.raises(ValueError, match="label"):
            load_jenniges_provinces(path)


class TestClassifyFromGrid:
    @staticmethod
    def _grid():
        return pd.DataFrame({
            "LATITUDE":  [50.0, 55.0, 60.0],
            "LONGITUDE": [-30.0, -25.0, -20.0],
            "label":     [1, 2, 3],
        })

    # Output is a DataFrame
    def test_returns_dataframe(self):
        df = pd.DataFrame({"LATITUDE": [50.0], "LONGITUDE": [-30.0]})
        result = classify_from_grid(df, self._grid())
        assert isinstance(result, pd.DataFrame)

    # A point exactly on a grid node gets that node's label
    def test_exact_match(self):
        df = pd.DataFrame({"LATITUDE": [55.0], "LONGITUDE": [-25.0]})
        result = classify_from_grid(df, self._grid())
        assert result["label"].iloc[0] == 2

    # A point between grid nodes gets the nearest one's label
    def test_nearest_neighbour(self):
        df = pd.DataFrame({"LATITUDE": [51.0], "LONGITUDE": [-29.0]})
        result = classify_from_grid(df, self._grid())
        assert result["label"].iloc[0] == 1

    # output_col lets the caller rename the added label column
    def test_custom_output_col(self):
        df = pd.DataFrame({"LATITUDE": [50.0], "LONGITUDE": [-30.0]})
        result = classify_from_grid(df, self._grid(), output_col="cluster")
        assert "cluster" in result.columns
        assert "label" not in result.columns

    # The input DataFrame is not modified in place
    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"LATITUDE": [50.0], "LONGITUDE": [-30.0]})
        classify_from_grid(df, self._grid())
        assert "label" not in df.columns

    # Row count is unchanged by classification
    def test_length_preserved(self):
        df = pd.DataFrame({"LATITUDE": [50.0, 55.0, 60.0], "LONGITUDE": [-30.0, -25.0, -20.0]})
        result = classify_from_grid(df, self._grid())
        assert len(result) == 3

    # Output index matches the input DataFrame's index
    def test_index_preserved(self):
        df = pd.DataFrame({"LATITUDE": [50.0, 55.0], "LONGITUDE": [-30.0, -25.0]},
                          index=[10, 20])
        result = classify_from_grid(df, self._grid())
        assert list(result.index) == [10, 20]
