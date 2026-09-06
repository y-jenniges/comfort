"""Tests for comfort.database.information."""
import sqlite3
import pytest

from comfort.database.information import (
    get_min_max_dates,
    get_minmax,
    get_names_of_all_parameter_tables,
    get_num_samples,
)
from comfort.qc import QC_GOOD


@pytest.fixture
def comfort_conn():
    """In-memory COMFORT-like database with station, P_NITRATE and P_OXYGEN."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE station ("
        "  ID INTEGER PRIMARY KEY, LATITUDE REAL, LONGITUDE REAL, DATEANDTIME TEXT)"
    )
    conn.executemany("INSERT INTO station VALUES (?,?,?,?)", [
        (1, 45.0, -30.0, "2000-01-15"),
        (2, 55.0, -20.0, "2005-06-10"),
        (3, 60.0, -10.0, "1998-11-02"),
    ])

    _param_ddl = (
        "CREATE TABLE {tbl} ("
        "  ID INTEGER, LEV_DBAR REAL, LEV_M REAL, VAL REAL,"
        "  PQF1 INTEGER, PQF2 INTEGER, SQF INTEGER,"
        "  BOTTLE_NUMBER INTEGER, PROFILE_NUMBER INTEGER,"
        "  PROFILE_BEST INTEGER, UNITS_ID INTEGER, INSTRUMENT_ID INTEGER)"
    )
    for tbl in ("P_NITRATE", "P_OXYGEN"):
        conn.execute(_param_ddl.format(tbl=tbl))

    conn.executemany("INSERT INTO P_NITRATE VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        (1, 0.0, 0.0, 5.0, 1, 3, 0, 1, 1, 1, 3, 1),
        (1, 100.0, 100.0, 4.0, 1, 3, 0, 2, 1, 1, 3, 1),
        (2, 0.0, 0.0, 8.0, 0, 1, 0, 1, 2, 1, 3, 2),  # fails QC_GOOD
    ])
    conn.executemany("INSERT INTO P_OXYGEN VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        (1, 0.0, 0.0, 200.0, 1, 3, 0, 1, 1, 1, 3, 1),
        (2, 0.0, 0.0, 220.0, 1, 3, 0, 1, 2, 1, 3, 2),
    ])
    conn.commit()
    yield conn
    conn.close()


class TestGetMinMaxDates:
    # Fetch min/max DATEANDTIME across all station rows
    def test_returns_min_and_max(self, comfort_conn):
        min_date, max_date = get_min_max_dates(comfort_conn, table_name="station")
        assert str(min_date.date()) == "1998-11-02"
        assert str(max_date.date()) == "2005-06-10"

    # Use a non-default time column
    def test_custom_time_column(self, comfort_conn):
        comfort_conn.execute("ALTER TABLE station ADD COLUMN VISIT_DATE TEXT")
        comfort_conn.execute("UPDATE station SET VISIT_DATE = DATEANDTIME")
        comfort_conn.commit()
        min_date, _ = get_min_max_dates(comfort_conn, table_name="station", time_column="VISIT_DATE")
        assert str(min_date.date()) == "1998-11-02"


class TestGetNumSamples:
    # Count every row when no quality flags are given
    def test_counts_all_rows_without_flags(self, comfort_conn):
        assert get_num_samples(comfort_conn, "P_NITRATE") == 3

    # Drop the row that fails QC_GOOD
    def test_applies_quality_flags(self, comfort_conn):
        assert get_num_samples(comfort_conn, "P_NITRATE", quality_flags=QC_GOOD) == 2

    # Return 0 and log an error for a table that does not exist
    def test_nonexistent_table_returns_zero(self, comfort_conn, caplog):
        import logging
        with caplog.at_level(logging.ERROR):
            result = get_num_samples(comfort_conn, "P_DOESNOTEXIST")
        assert result == 0
        assert "P_DOESNOTEXIST" in caplog.text


class TestGetNamesOfAllParameterTables:
    # Match both parameter tables with the default LIKE pattern
    def test_matches_parameter_tables(self, comfort_conn):
        names = get_names_of_all_parameter_tables(comfort_conn)
        assert set(names) == {"P_NITRATE", "P_OXYGEN"}

    # Station is not a P_* table, must not be included
    def test_excludes_non_matching_tables(self, comfort_conn):
        names = get_names_of_all_parameter_tables(comfort_conn)
        assert "station" not in names

    # Grid-mapped tables (numeric suffix) are excluded by default
    def test_excludes_digit_suffixed_tables_by_default(self, comfort_conn):
        comfort_conn.execute(
            "CREATE TABLE P_NITRATE_123 (ID INTEGER, LEV_M REAL, VAL REAL)"
        )
        comfort_conn.commit()
        names = get_names_of_all_parameter_tables(comfort_conn)
        assert "P_NITRATE_123" not in names

    # include_digits=True brings grid-mapped tables back
    def test_includes_digit_suffixed_tables_when_requested(self, comfort_conn):
        comfort_conn.execute(
            "CREATE TABLE P_NITRATE_123 (ID INTEGER, LEV_M REAL, VAL REAL)"
        )
        comfort_conn.commit()
        names = get_names_of_all_parameter_tables(comfort_conn, include_digits=True)
        assert "P_NITRATE_123" in names

    # Same lookup works against views, not just tables
    def test_matches_views(self, comfort_conn):
        comfort_conn.execute("CREATE VIEW E_NITRATE AS SELECT * FROM P_NITRATE")
        comfort_conn.commit()
        names = get_names_of_all_parameter_tables(
            comfort_conn, like_pattern="E|_%", escape_char="|", table_type="view"
        )
        assert names == ["E_NITRATE"]


class TestGetMinMax:
    # Global min/max of VAL across two explicit parameter tables
    def test_global_min_max_across_tables(self, comfort_conn):
        lo, hi = get_minmax(comfort_conn, column="VAL", column_type="float",
                            param_tables=["P_NITRATE", "P_OXYGEN"])
        assert lo == pytest.approx(4.0)
        assert hi == pytest.approx(220.0)

    # param_tables=None auto-discovers all parameter tables
    def test_default_param_tables_uses_all_parameter_tables(self, comfort_conn):
        lo, hi = get_minmax(comfort_conn, column="VAL", column_type="float")
        assert lo == pytest.approx(4.0)
        assert hi == pytest.approx(220.0)

    # column_type="int" converts the result to Python int
    def test_int_column_type(self, comfort_conn):
        lo, hi = get_minmax(comfort_conn, column="ID", column_type="int",
                            param_tables=["P_NITRATE"])
        assert (lo, hi) == (1, 2)

    # Unsupported column_type logs an error and returns (None, None)
    def test_unsupported_column_type_returns_none(self, comfort_conn, caplog):
        import logging
        with caplog.at_level(logging.ERROR):
            result = get_minmax(comfort_conn, column="VAL", column_type="bogus",
                                param_tables=["P_NITRATE"])
        assert result == (None, None)
        assert "bogus" in caplog.text
