"""
Module containing functions to get information from the database/from tables.
"""
import logging
import sqlite3
import pandas as pd
import numpy as np


def does_table_exist(conn, table_name, table_type="table"):
    """
    Checks if a table/view with the given name exists. Warning: Make sure that the casing is correct. Function is
    case-sensitive!

    Args:
        conn (sqlite3.Connection): Connection to the database.
        table_name (str): Check if this table name is already in the database.
        table_type (str): Type of the database structure to check, e.g. 'view'. Default is 'table'.
    Returns:
        does_table_exist (bool): If the table/view exists in the database.
    """
    cur = conn.cursor()
    query = f"select name from sqlite_master where type='{table_type}' AND name='{table_name}';"
    result = cur.execute(query).fetchall()

    return True if result else False


def get_min_max_dates(conn, table_name="e_combined", time_column="DATEANDTIME"):
    """
    Connects to database and fetches minimum and maximum dates from a table.
    Args:
        conn (sqlite3.Connection): Connection to the database.
        table_name (str): Table or view to search the dates in.
        time_column (str): Name of the column containing the date and time information.
    Returns:
        min_date (pandas.Timestamp): Minimum date in the given table.
        max_date (pandas.Timestamp): Maximum date in the given table.
    """
    cur = conn.cursor()

    query_min = f"select min({time_column}) from {table_name};"
    min_date = pd.Timestamp(cur.execute(query_min).fetchall()[0][0])
    query_max = f"select max({time_column}) from {table_name};"
    max_date = pd.Timestamp(cur.execute(query_max).fetchall()[0][0])

    return min_date, max_date


def get_num_samples(conn, table_name, table_type="table", quality_flags=None):
    """
    Get the number of samples contained in the given table.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        table_name (str): Table or view to search the dates in.
        table_type (str): Type of the database structure to check, e.g. 'view'. Default is 'table'.
        quality_flags (list<list<str, str>>): List containing the quality flags and their value. Default is None.

    Returns:
        num_samples (int): Number of samples.
    """
    cur = conn.cursor()
    num_samples = 0

    # determine SQL expression to consider quality flags
    quality_statement = ""
    if quality_flags:
        quality_statement = f"where {' and '.join(x[0] + x[1] for x in quality_flags)}"

    # check if table exists
    if does_table_exist(conn, table_name, table_type):
        # get the number of samples
        query = f"select count(*) from {table_name} {quality_statement};"
        num_samples = cur.execute(query).fetchall()[0]
    else:
        logging.error(f"Error in viewsAndTables.get_num_samples: No such {table_type} {table_name}.")

    return num_samples


def get_table_as_df(conn, table_name, columns=None):
    """
    Get a table from the database as pandas dataframe.

    Args:
        table_name (str): Name of the table to fetch.
        conn (sqlite3.Connection): Connection to the database that stores the table.
        columns (list<str>): List of columns to query. If None, all columns will be queried. Default is None.
    Returns:
        df (pandas.DataFrame): Dataframe containing the table information.
    """
    col_selection = "*"
    if columns:
        col_selection = ", ".join(columns)

    query = f"select {col_selection} from {table_name};"
    cur = conn.cursor()
    ex = cur.execute(query)
    cols = [description[0] for description in cur.description]

    result = ex.fetchall()
    df = pd.DataFrame(result, columns=cols)
    return df


def get_names_of_all_parameter_tables(conn, like_pattern="P|_%", escape_char="|", include_digits=False,
                                      table_type="table"):
    """
    Get all tables conforming with the given like_pattern.

    Args:
        conn (sqlite3.Connection): Connection to the sqlite3 database.
        like_pattern (str): Table names should conform to this string pattern. If None, all table names are queried.
        Default is 'P|_%'
        escape_char (str): Escape char used for the like pattern. If None, sqlite3 default are used. Default is '|'.
        include_digits (bool): Weather tables names including numbers should be returned as well.
        table_type (str): The table type to look for (e.g. 'view'). Default is 'table'.
    Returns:
        param_table_names (list<str>): A list of table names that conform to the given pattern.
    """
    digits_filter = ""
    if not include_digits:
        digits_filter = "and name not glob '*_[0-9]*'"

    if like_pattern and escape_char:
        query = f"select name from sqlite_master " \
                f"where type='{table_type}' and name like '{like_pattern}' escape '{escape_char}' " \
                f"{digits_filter};"
    elif like_pattern:
        query = f"select name from sqlite_master " \
                f"where type='{table_type}' and name like '{like_pattern}' " \
                f"{digits_filter};"
    else:
        query = f"select name from sqlite_master " \
                f"where type='{table_type}' " \
                f"{digits_filter};"

    print(f"Information: {query}")
    ex = conn.cursor().execute(query)
    result = ex.fetchall()
    param_table_names = [entry[0] for entry in result]

    return param_table_names


def get_minmax(conn, column="LATITUDE", column_type="float", param_tables=None):
    column_minimum = None
    column_maximum = None

    if not param_tables:
        param_tables = get_names_of_all_parameter_tables(conn)

    for table in param_tables:
        q = f"select min({column}), max({column}) from {table};"
        cur = conn.execute(q)
        res = cur.fetchall()[0]

        # convert type
        if column_type == "float":
            res = [float(x) for x in res]
        elif column_type == "datetime64":
            res = [np.datetime64(x) for x in res]
        elif column_type == "int":
            res = [int(x) for x in res]
        else:
            print(f"Error: database.information.get_minmax: Conversion to {column_type} not implemented!")
            return

        if column_minimum is None or column_minimum > res[0]:
            column_minimum = res[0]

        if column_maximum is None or column_maximum < res[1]:
            column_maximum = res[1]

        print(table, res)
        print(column_minimum, column_maximum)
        print()

    return column_minimum, column_maximum
