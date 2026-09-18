"""Tests for comfort.database."""
import sqlite3
import pytest

from comfort.database.information import does_table_exist, get_table_as_df
from comfort.database.structure import (
    create_combined_parameter_table,
    create_extended_parameter_tables,
    execute_sql_scripts,
    remove_tables_like,
)


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
        (2, 0.0, 0.0, 8.0, 1, 3, 0, 1, 2, 1, 3, 2),
    ])
    conn.executemany("INSERT INTO P_OXYGEN VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        (1, 0.0, 0.0, 200.0, 1, 3, 0, 1, 1, 1, 3, 1),
        (2, 0.0, 0.0, 220.0, 1, 3, 0, 1, 2, 1, 3, 2),
    ])

    # Temperature and salinity tables for extended view joins
    for tbl in ("P_TEMPERATURE", "P_SALINITY"):
        conn.execute(_param_ddl.format(tbl=tbl))
    conn.execute("INSERT INTO P_TEMPERATURE VALUES (1, 0.0, 0.0, 15.0, 1, 3, 0, 1, 1, 1, 1, 1)")
    conn.execute("INSERT INTO P_SALINITY VALUES (1, 0.0, 0.0, 35.0, 1, 3, 0, 1, 1, 1, 2, 1)")

    conn.commit()
    yield conn
    conn.close()


class TestCreateExtendedParameterTables:
    # Creates one E_* view per requested parameter
    def test_creates_views(self, comfort_conn):
        result = create_extended_parameter_tables(
            comfort_conn, table_type="view",
            parameters=["NITRATE"], add_temperature=False, add_salinity=False,
        )
        assert "E_NITRATE" in result
        assert does_table_exist(comfort_conn, "E_NITRATE", "view")

    # Station metadata is joined into the view
    def test_view_contains_station_columns(self, comfort_conn):
        create_extended_parameter_tables(
            comfort_conn, table_type="view",
            parameters=["NITRATE"], add_temperature=False, add_salinity=False,
        )
        df = get_table_as_df(comfort_conn, "E_NITRATE")
        assert "LATITUDE" in df.columns
        assert "LONGITUDE" in df.columns
        assert "DATEANDTIME" in df.columns

    # Row count matches the source parameter table
    def test_correct_row_count(self, comfort_conn):
        create_extended_parameter_tables(
            comfort_conn, table_type="view",
            parameters=["NITRATE"], add_temperature=False, add_salinity=False,
        )
        df = get_table_as_df(comfort_conn, "E_NITRATE")
        assert len(df) == 3

    # add_temperature=True joins in a temperature column
    def test_adds_temperature_column(self, comfort_conn):
        create_extended_parameter_tables(
            comfort_conn, table_type="view",
            parameters=["NITRATE"], add_temperature=True, add_salinity=False,
        )
        df = get_table_as_df(comfort_conn, "E_NITRATE")
        assert "temperature" in df.columns

    # add_salinity=True joins in a salinity column
    def test_adds_salinity_column(self, comfort_conn):
        create_extended_parameter_tables(
            comfort_conn, table_type="view",
            parameters=["NITRATE"], add_temperature=False, add_salinity=True,
        )
        df = get_table_as_df(comfort_conn, "E_NITRATE")
        assert "salinity" in df.columns

    # parameters=None discovers and builds a view for every P_* table
    def test_auto_detects_all_parameters(self, comfort_conn):
        result = create_extended_parameter_tables(
            comfort_conn, table_type="view",
            add_temperature=False, add_salinity=False,
        )
        # Should create views for all P_* tables
        assert "E_NITRATE" in result
        assert "E_OXYGEN" in result

    # A parameter with no matching P_* table yields None instead of raising
    def test_nonexistent_parameter_returns_none(self, comfort_conn):
        result = create_extended_parameter_tables(
            comfort_conn, table_type="view",
            parameters=["DOESNOTEXIST"],
            add_temperature=False, add_salinity=False,
        )
        assert result is None

    # A second temperature reading at the same (ID, LEV_M) is averaged
    def test_duplicate_temperature_does_not_multiply_rows(self, comfort_conn):
        comfort_conn.execute(
            "INSERT INTO P_TEMPERATURE VALUES (1, 0.0, 0.0, 17.0, 1, 3, 0, 2, 1, 1, 1, 1)"
        )
        comfort_conn.commit()
        create_extended_parameter_tables(
            comfort_conn, table_type="view",
            parameters=["NITRATE"], add_temperature=True, add_salinity=False,
        )
        df = get_table_as_df(comfort_conn, "E_NITRATE")
        assert len(df) == 3
        row = df[(df["ID"] == 1) & (df["LEV_M"] == 0.0)].iloc[0]
        assert row["temperature"] == pytest.approx(16.0)

    # A bad-quality (PQF2<=2) duplicate temperature reading must be
    # excluded from the average used for the QC_GOOD-filtered view
    def test_quality_flags_applied_to_joined_temperature(self, comfort_conn):
        from comfort.qc import QC_GOOD

        comfort_conn.execute(
            "INSERT INTO P_TEMPERATURE VALUES (1, 0.0, 0.0, 99.0, 1, 1, 0, 2, 1, 1, 1, 1)"
        )
        comfort_conn.commit()
        create_extended_parameter_tables(
            comfort_conn, table_type="view",
            parameters=["NITRATE"], add_temperature=True, add_salinity=False,
            quality_flags=QC_GOOD,
        )
        df = get_table_as_df(comfort_conn, "E_NITRATE")
        row = df[(df["ID"] == 1) & (df["LEV_M"] == 0.0)].iloc[0]
        assert row["temperature"] == pytest.approx(15.0)


class TestRemoveTablesLike:
    # Drop every view matching the LIKE pattern
    def test_removes_views(self, comfort_conn):
        # Create views first
        create_extended_parameter_tables(
            comfort_conn, table_type="view",
            parameters=["NITRATE", "OXYGEN"],
            add_temperature=False, add_salinity=False,
        )
        assert does_table_exist(comfort_conn, "E_NITRATE", "view")

        # Remove them
        remove_tables_like(comfort_conn, like_pattern="E|_%", escape_char="|", table_type="view")
        assert not does_table_exist(comfort_conn, "E_NITRATE", "view")
        assert not does_table_exist(comfort_conn, "E_OXYGEN", "view")

    # tables_except protects named views from removal
    def test_tables_except(self, comfort_conn):
        create_extended_parameter_tables(
            comfort_conn, table_type="view",
            parameters=["NITRATE", "OXYGEN"],
            add_temperature=False, add_salinity=False,
        )
        remove_tables_like(
            comfort_conn, like_pattern="E|_%", escape_char="|",
            table_type="view", tables_except=["E_NITRATE"],
        )
        # E_NITRATE should survive, E_OXYGEN should be dropped
        assert does_table_exist(comfort_conn, "E_NITRATE", "view")
        assert not does_table_exist(comfort_conn, "E_OXYGEN", "view")


class TestExecuteSqlScripts:
    # Scripts whose filename matches the prefix are executed
    def test_executes_matching_scripts(self, comfort_conn, tmp_path):
        (tmp_path / "create_view_nitrate.sql").write_text(
            "CREATE VIEW E_NITRATE AS SELECT * FROM P_NITRATE;"
        )
        execute_sql_scripts(comfort_conn, sql_folder=str(tmp_path))
        assert does_table_exist(comfort_conn, "E_NITRATE", "view")

    # Scripts with a non-matching prefix are left untouched
    def test_ignores_non_matching_prefix(self, comfort_conn, tmp_path):
        (tmp_path / "other_script.sql").write_text(
            "CREATE VIEW E_NITRATE AS SELECT * FROM P_NITRATE;"
        )
        execute_sql_scripts(comfort_conn, sql_folder=str(tmp_path), prefix="create_view_")
        assert not does_table_exist(comfort_conn, "E_NITRATE", "view")

    # Non-.sql files in the folder are silently skipped
    def test_ignores_non_sql_files(self, comfort_conn, tmp_path):
        (tmp_path / "create_view_notes.txt").write_text("not sql")
        # Must not raise despite the non-matching extension
        execute_sql_scripts(comfort_conn, sql_folder=str(tmp_path))

    # A missing sql_folder logs an error instead of raising
    def test_missing_directory_logs_error(self, comfort_conn, tmp_path, caplog):
        import logging
        with caplog.at_level(logging.ERROR):
            execute_sql_scripts(comfort_conn, sql_folder=str(tmp_path / "does_not_exist"))
        assert "does not exist" in caplog.text

    # A script with invalid SQL logs the failure and moves on
    def test_bad_sql_logs_error_without_raising(self, comfort_conn, tmp_path, caplog):
        import logging
        (tmp_path / "create_view_bad.sql").write_text("NOT VALID SQL;")
        with caplog.at_level(logging.ERROR):
            execute_sql_scripts(comfort_conn, sql_folder=str(tmp_path))
        assert "create_view_bad.sql" in caplog.text


class TestCreateCombinedParameterTable:
    # Builds the UNION ALL view across all requested parameters
    def test_creates_combined_view(self, comfort_conn):
        create_combined_parameter_table(comfort_conn, ["NITRATE", "OXYGEN"])
        assert does_table_exist(comfort_conn, "P_COMBINED", "view")

    # Adds a PARAM_NAME column identifying each source parameter
    def test_combined_has_param_name_column(self, comfort_conn):
        create_combined_parameter_table(comfort_conn, ["NITRATE", "OXYGEN"])
        df = get_table_as_df(comfort_conn, "P_COMBINED")
        assert "PARAM_NAME" in df.columns

    # Row count is the sum of rows across all source tables
    def test_combined_row_count(self, comfort_conn):
        create_combined_parameter_table(comfort_conn, ["NITRATE", "OXYGEN"])
        df = get_table_as_df(comfort_conn, "P_COMBINED")
        # 3 nitrate + 2 oxygen = 5
        assert len(df) == 5

    # PARAM_NAME values match the requested parameter list
    def test_combined_param_names(self, comfort_conn):
        create_combined_parameter_table(comfort_conn, ["NITRATE", "OXYGEN"])
        df = get_table_as_df(comfort_conn, "P_COMBINED")
        assert set(df["PARAM_NAME"].unique()) == {"NITRATE", "OXYGEN"}

    # A nonexistent parameter is skipped with a warning, not a hard failure
    def test_nonexistent_param_skipped(self, comfort_conn, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            create_combined_parameter_table(comfort_conn, ["NITRATE", "DOESNOTEXIST"])
        assert "DOESNOTEXIST" in caplog.text
        df = get_table_as_df(comfort_conn, "P_COMBINED")
        # Only NITRATE rows should be present
        assert set(df["PARAM_NAME"].unique()) == {"NITRATE"}
