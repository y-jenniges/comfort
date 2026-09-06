"""Tests for comfort.scaling."""
import sqlite3
import pandas as pd
import pytest

from comfort.scaling import ParamScaler, DBParamScaler

_has_sklearn = True
try:
    import sklearn
except ImportError:
    _has_sklearn = False


@pytest.mark.skipif(not _has_sklearn, reason="scikit-learn not installed ([scale] extra)")
class TestParamScaler:
    # Default scaler_class is MinMaxScaler
    def test_default_scaler_is_minmax(self):
        scaler = ParamScaler()
        from sklearn.preprocessing import MinMaxScaler
        assert scaler.scaler_class is MinMaxScaler

    # scaler_class can be swapped for any sklearn-compatible scaler
    def test_custom_scaler_class(self):
        from sklearn.preprocessing import StandardScaler
        scaler = ParamScaler(scaler_class=StandardScaler)
        assert scaler.scaler_class is StandardScaler

    # scale() adds a VAL_SCALED column
    def test_scale_adds_val_scaled(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"VAL": [0.0, 5.0, 10.0], "PARAM_NAME": ["NITRATE"] * 3})
        result = scaler.scale(df)
        assert "VAL_SCALED" in result.columns

    # MinMax scaling maps the value range onto [0, 1]
    def test_scale_range_is_0_to_1(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"VAL": [0.0, 5.0, 10.0], "PARAM_NAME": ["NITRATE"] * 3})
        result = scaler.scale(df)
        assert result["VAL_SCALED"].min() == pytest.approx(0.0)
        assert result["VAL_SCALED"].max() == pytest.approx(1.0)

    # scale() returns a copy, the input DataFrame is untouched
    def test_scale_does_not_mutate_input(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"VAL": [0.0, 5.0, 10.0], "PARAM_NAME": ["NITRATE"] * 3})
        scaler.scale(df)
        assert "VAL_SCALED" not in df.columns

    # scale_columns() adds a _SCALED column per requested column
    def test_scale_columns(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"NITRATE": [0.0, 5.0, 10.0], "OXYGEN": [200.0, 250.0, 300.0]})
        result = scaler.scale_columns(["NITRATE", "OXYGEN"], df)
        assert "NITRATE_SCALED" in result.columns
        assert "OXYGEN_SCALED" in result.columns

    # scale_columns() also maps onto [0, 1] by default
    def test_scale_columns_range(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"NITRATE": [0.0, 5.0, 10.0]})
        result = scaler.scale_columns(["NITRATE"], df)
        assert result["NITRATE_SCALED"].min() == pytest.approx(0.0)
        assert result["NITRATE_SCALED"].max() == pytest.approx(1.0)

    # A requested column absent from df logs a warning instead of raising
    def test_scale_columns_missing_column_warns(self, caplog):
        scaler = ParamScaler()
        df = pd.DataFrame({"NITRATE": [0.0, 5.0, 10.0]})
        import logging
        with caplog.at_level(logging.WARNING):
            result = scaler.scale_columns(["MISSING"], df)
        assert "MISSING" in caplog.text
        assert "MISSING_SCALED" not in result.columns

    # scale_columns() returns a copy, the input DataFrame is untouched
    def test_scale_columns_does_not_mutate_input(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"NITRATE": [0.0, 5.0, 10.0]})
        scaler.scale_columns(["NITRATE"], df)
        assert "NITRATE_SCALED" not in df.columns

    # scale_value() applies a previously fitted column's scaler to a single value
    def test_scale_value_after_fit(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"NITRATE": [0.0, 10.0]})
        scaler.scale_columns(["NITRATE"], df)
        assert scaler.scale_value("NITRATE", 5.0) == pytest.approx(0.5)

    # scale_value() raises when the column has never been fitted
    def test_scale_value_unfitted_raises(self):
        scaler = ParamScaler()
        with pytest.raises(KeyError, match="No scaler fitted"):
            scaler.scale_value("NITRATE", 5.0)

    # Extra kwargs are forwarded to the underlying sklearn scaler
    def test_scaler_kwargs_forwarded(self):
        from sklearn.preprocessing import MinMaxScaler
        scaler = ParamScaler(scaler_class=MinMaxScaler, feature_range=(0, 100))
        df = pd.DataFrame({"NITRATE": [0.0, 5.0, 10.0]})
        result = scaler.scale_columns(["NITRATE"], df)
        assert result["NITRATE_SCALED"].max() == pytest.approx(100.0)

    # The fitted scaler persists across calls instead of refitting each time
    def test_scaler_reused_across_calls(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"VAL": [0.0, 10.0], "PARAM_NAME": ["A"] * 2})
        scaler.scale(df)
        # Second call with different values should use the same fitted scaler
        df2 = pd.DataFrame({"VAL": [5.0, 15.0], "PARAM_NAME": ["A"] * 2})
        result = scaler.scale(df2)
        # 5.0 scaled with [0, 10] range -> 0.5; 15.0 -> 1.5 (extrapolated)
        assert result["VAL_SCALED"].iloc[0] == pytest.approx(0.5)


class TestDBParamScaler:
    @pytest.fixture
    def db_conn(self):
        # Create dummy database
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE P_NITRATE (ID INTEGER, LEV_M REAL, VAL REAL, "
            "UNITS_ID INTEGER, PQF1 INTEGER)"
        )
        conn.executemany("INSERT INTO P_NITRATE VALUES (?,?,?,?,?)", [
            (1, 0.0, 0.0, 3, 1),
            (1, 100.0, 5.0, 3, 1),
            (1, 200.0, 10.0, 3, 1),
        ])
        conn.execute(
            "CREATE TABLE P_OXYGEN (ID INTEGER, LEV_M REAL, VAL REAL, "
            "UNITS_ID INTEGER, PQF1 INTEGER)"
        )
        conn.executemany("INSERT INTO P_OXYGEN VALUES (?,?,?,?,?)", [
            (1, 0.0, 200.0, 3, 1),
            (1, 100.0, 250.0, 3, 1),
            (1, 200.0, 300.0, 3, 1),
        ])
        conn.commit()
        yield conn
        conn.close()

    # get_min_max reads VAL bounds straight from the database
    def test_get_min_max(self, db_conn):
        scaler = DBParamScaler(db_conn)
        vmin, vmax = scaler.get_min_max("P_NITRATE", "VAL")
        assert vmin == 0.0
        assert vmax == 10.0

    # get_min_max_depth pools the depth range across multiple tables
    def test_get_min_max_depth(self, db_conn):
        scaler = DBParamScaler(db_conn)
        dmin, dmax = scaler.get_min_max_depth(["P_NITRATE", "P_OXYGEN"])
        assert dmin == 0.0
        assert dmax == 200.0

    # scale() writes one scaled_P_* table per requested parameter
    def test_scale_creates_tables(self, db_conn):
        scaler = DBParamScaler(db_conn)
        result = scaler.scale({"P_NITRATE": "NITRATE", "P_OXYGEN": "OXYGEN"})
        assert "scaled_P_NITRATE" in result
        assert "scaled_P_OXYGEN" in result
        # Verify tables exist in the database
        from comfort.database.information import does_table_exist
        assert does_table_exist(db_conn, "scaled_P_NITRATE")
        assert does_table_exist(db_conn, "scaled_P_OXYGEN")

    # Scaled VAL values fall within the default [0, 1] bounds
    def test_scale_values_in_range(self, db_conn):
        scaler = DBParamScaler(db_conn)
        scaler.scale({"P_NITRATE": "NITRATE"})
        cur = db_conn.execute("SELECT VAL FROM scaled_P_NITRATE;")
        vals = [row[0] for row in cur.fetchall()]
        assert min(vals) == pytest.approx(0.0)
        assert max(vals) == pytest.approx(1.0)

    # An empty parameter dict returns empty dict
    def test_scale_empty_dict(self, db_conn):
        scaler = DBParamScaler(db_conn)
        assert scaler.scale({}) == {}

    # lower/upper bounds override the default [0, 1] range
    def test_scale_custom_bounds(self, db_conn):
        scaler = DBParamScaler(db_conn, lower=-1, upper=1)
        scaler.scale({"P_NITRATE": "NITRATE"})
        cur = db_conn.execute("SELECT VAL FROM scaled_P_NITRATE;")
        vals = [row[0] for row in cur.fetchall()]
        assert min(vals) == pytest.approx(-1.0)
        assert max(vals) == pytest.approx(1.0)

    # A malicious/invalid table name is rejected before it reaches SQL
    def test_validate_identifier_rejects_bad_name(self, db_conn):
        scaler = DBParamScaler(db_conn)
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            scaler.get_min_max("DROP TABLE foo; --", "VAL")
