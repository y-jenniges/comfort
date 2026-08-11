"""Functions to manage views and tables in the COMFORT database.

remove_tables_like adapted from Jenniges (2025), doi:10.5281/zenodo.15827777
"""
from __future__ import annotations

import logging
import os
import sqlite3

from .information import get_names_of_all_parameter_tables, does_table_exist
from ..qc import build_where_clause
from ..util.sqlite_utils import validate_identifier


def execute_sql_scripts(conn: sqlite3.Connection, sql_folder: str = "sql_scripts/",
                        prefix: str = "create_view_") -> None:
    """Execute all SQL scripts in a folder whose name starts with prefix.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        sql_folder (str): Directory containing the SQL files. Default is 'sql_scripts/'.
        prefix (str): Only execute files whose name starts with this string. Default is ``create_view_``.
    """
    # Connect to db and define prefix
    cur = conn.cursor()
    prefix = prefix or ""

    # Check if directory exists
    if not os.path.isdir(sql_folder):
        logging.error(f"execute_sql_scripts: directory {sql_folder} does not exist")
        return

    # Get all scripts to execute
    scripts = [f for f in os.listdir(sql_folder)
               if f.startswith(prefix) and f.endswith(".sql")]

    # Iterate over scripts
    for script in scripts:
        # Read SQL file
        sql = open(os.path.join(sql_folder, script)).read()

        # Execute script
        try:
            logging.info(f"Executing SQL script {script}")
            cur.executescript(sql)
        except sqlite3.Error as e:
            logging.error(f"Error executing {script}: {e}")


def create_extended_parameter_tables(conn: sqlite3.Connection, table_type: str = "view",
                                     parameters: list[str] | None = None,
                                     add_temperature: bool = True, add_salinity: bool = True,
                                     quality_flags: list | None = None) -> dict[str, str] | None:
    """Create extended table/view for each parameter with lat/lon/datetime.

    Optionally joins temperature and salinity columns.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        table_type (str): 'view' or 'table'. Default is 'view'.
        parameters (list[str]): Parameter names. None means all parameters.
        add_temperature (bool): Add a temperature column. Default is True.
        add_salinity (bool): Add a salinity column. Default is True.
        quality_flags (list[QCFilter] or list[tuple[str, str]]): Quality flag filters.
    Returns:
        dict: Maps new extended table names to parameter names.
    """
    # Connect to db
    cur = conn.cursor()

    # Define quality filters
    quality_statement = build_where_clause(quality_flags, table_alias="t")

    # Optionally join temperature and salinity
    ts_join = ""
    ts_select = ""
    if add_temperature:
        ts_join += "LEFT JOIN P_TEMPERATURE AS temp ON (t.LEV_M=temp.LEV_M AND temp.id=s.id) "
        ts_select += ", temp.VAL AS temperature"
    if add_salinity:
        ts_join += "LEFT JOIN P_SALINITY AS sal ON (t.LEV_M=sal.LEV_M AND sal.id=s.id) "
        ts_select += ", sal.VAL AS salinity"

    # Define parameter names
    if not parameters:
        parameters = [x[2:] for x in get_names_of_all_parameter_tables(conn)]

    #
    tables_params = {}
    for param_name in parameters:
        # Validate identifier
        validate_identifier(param_name)

        # Define table/view names
        new_name = "E_" + param_name.upper()
        table_name = "P_" + param_name.upper()

        # Check if table exists
        if not does_table_exist(conn, table_name, table_type="table"):
            logging.error(f"create_extended_parameter_tables: {table_name} does not exist")
            return

        # Assemble and execute SQL query
        query = (f"CREATE {table_type} IF NOT EXISTS {new_name} AS "
                 f"SELECT t.*, s.LATITUDE, s.LONGITUDE, s.DATEANDTIME {ts_select} "
                 f"FROM {table_name} AS t "
                 f"LEFT JOIN STATION AS s ON t.id=s.id "
                 f"{ts_join} "
                 f"{quality_statement};")
        logging.info(f"create_extended_parameter_tables: {query}")
        cur.execute(query)

        tables_params[new_name] = param_name.upper()

    conn.commit()
    return tables_params


def remove_tables_like(conn: sqlite3.Connection, like_pattern: str = "E|_%", escape_char: str = "|",
                       table_type: str = "view", tables_except: list[str] | None = None) -> None:
    """Drop views/tables whose names match the given LIKE pattern.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        like_pattern (str): SQLite LIKE pattern. Default is 'E|_%'.
        escape_char (str): Escape character. Default is '|'.
        table_type (str): 'view' or 'table'. Default is 'view'.
        tables_except (list[str]): Names to exclude from removal.
    """
    # Get views/tables to remove
    table_names = get_names_of_all_parameter_tables(conn, like_pattern, escape_char,
                                                    table_type=table_type, include_digits=True)
    if tables_except:
        table_names = [t for t in table_names if t not in tables_except]

    # Connect to db and drop the tables
    cur = conn.cursor()
    for name in table_names:
        query = f"DROP {table_type} {name};"
        logging.info(f"remove_tables_like: {query}")
        cur.execute(query)

    conn.commit()


def create_combined_parameter_table(conn: sqlite3.Connection, parameters: list[str],
                                    quality_flags: list | None = None,
                                    columns: list[str] | None = None,
                                    table_type: str = "view") -> None:
    """Create a UNION ALL table/view combining multiple parameters.

    Adds a PARAM_NAME column. Assumes P_* tables exist.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        parameters (list[str]): Parameter names to combine.
        quality_flags (list[QCFilter] or list[tuple[str, str]]): Quality flag filters.
        columns (list[str]): Columns to include. Defaults to the standard COMFORT parameter columns.
        table_type (str): 'view' or 'table'. Default is 'view'.
    """
    # Define columns to include
    if columns is None:
        columns = ["ID", "LEV_DBAR", "LEV_M", "VAL", "PQF1", "PQF2", "SQF", "BOTTLE_NUMBER",
                   "PROFILE_NUMBER", "PROFILE_BEST", "UNITS_ID", "INSTRUMENT_ID"]

    # Validate identifiers
    for p in parameters:
        validate_identifier(p)
    for col in columns:
        validate_identifier(col)

    # Build quality filter
    quality_statement = build_where_clause(quality_flags)

    # Iterate over parameters
    select_params = []
    for p in parameters:
        # Check if view/table exist
        view_exists = does_table_exist(conn, f"P_{p.upper()}", "view")
        table_exists = does_table_exist(conn, f"P_{p.upper()}", "table")

        # Build data-fetching query
        if view_exists or table_exists:
            select_params.append(
                f"SELECT '{p.upper()}' AS PARAM_NAME, {', '.join(columns)} "
                f"FROM P_{p.upper()} {quality_statement}"
            )
        else:
            logging.warning(f"create_combined_parameter_table: P_{p.upper()} does not exist - skipping")

    # Build combined query and execute
    query = (f"CREATE {table_type} IF NOT EXISTS P_COMBINED AS "
             f"SELECT * FROM ({' UNION ALL '.join(select_params)});")
    conn.cursor().execute(query)
    conn.commit()
