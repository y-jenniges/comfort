""" Module containing functions to manage views and tables. """
import logging
import os
import sqlite3
from src.database.information import get_names_of_all_parameter_tables, does_table_exist


def execute_sql_scripts(conn, sql_folder="sql_scripts/", prefix="create_view_"):
    """
    Executes all scripts in the specified folder.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        sql_folder (str): Directory containing the SQL files to execute. Default is 'sql_scripts'.
        prefix (str): Optional. Execute SQL files in dir only if their name starts with this string. Default is None.
    """
    cur = conn.cursor()

    # get all scripts to execute
    prefix = "" if not prefix else prefix
    suffix = ".sql"
    if not os.path.isdir(sql_folder):
        logging.error(f"ERROR in viewsAndTables.execute_sql_scripts: Directory {sql_folder} does not exist.")
        return

    sql_scripts = [filename for filename in os.listdir(sql_folder)
                   if filename.startswith(prefix) and filename.endswith(suffix)]

    for script in sql_scripts:
        # read sql file
        sql_script = open(sql_folder + script).read()

        # execute script
        try:
            print(f"viewsAndTables: Execute SQL script {script}")
            cur.executescript(sql_script)
        except sqlite3.Error as e:
            print("ERROR in viewsAndTables: An error occurred while executing the SQL files:")
            print(f"    {e}")


def create_extended_parameter_tables(conn, table_type="view", parameters=None, add_temperature=True, add_salinity=True,
                                     quality_flags=None):
    """
    Creates an extended table (or view) for the specified parameters. The added features are dateandtime, latitude,
    longitude (optionally temperature and salinity). Optionally, quality flags can be filtered for.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        table_type (str): Whether to create views or tables. Default is 'view'.
        parameters (list<str>): Extended tables are created for these parameters. If None, a table is created for every
        parameter. Default is None.
        add_temperature (bool): Whether to add a column with temperature values from P_TEMPERATURE.
        add_salinity (bool): Whether to add a column with salinity values from P_SALINITY.
        quality_flags (list<list<str, str>>): List containing the quality flags and their value.
    Returns:
        tables_params (dict): Maps the parameter names to the newly created extended tables.
    """
    # create a cursor
    cur = conn.cursor()

    # determine SQL expression to consider quality flags
    quality_statement = ""
    if quality_flags:
        quality_statement = f"where {' and '.join('t.' + x[0] + x[1] for x in quality_flags)}"

    # determine SQL expression to consider temperature and salinity values
    ts_join_statement = ""
    ts_select_statement = ""
    if add_temperature:
        ts_join_statement = ts_join_statement + \
                            f"left join P_TEMPERATURE as temp on (t.LEV_M=temp.LEV_M and temp.id=s.id) "
        ts_select_statement = ts_select_statement + ", temp.VAL as temperature"
    if add_salinity:
        ts_join_statement = ts_join_statement + \
                            f"left join P_SALINITY as sal on (t.LEV_M=sal.LEV_M and sal.id=s.id) "
        ts_select_statement = ts_select_statement + ", sal.VAL as salinity"

    # get all parameter names
    if not parameters:
        parameters = [x[2:] for x in get_names_of_all_parameter_tables(conn)]

    # create one view per parameter (with according quality flags)
    tables_params = {}
    for param_name in parameters:
        new_name = "E_" + param_name.upper()  # E for extended
        table_name = "P_" + param_name.upper()

        # check if table exists
        if not does_table_exist(conn, table_name, table_type="table"):
            logging.error("ERROR in viewsAndTables.create_extended_parameter_tables: Parameter name is wrong because "
                          f"the {table_type} called {table_name} does not exist.")
            return

        # create extended table
        query = f"create {table_type} if not exists {new_name} as " \
                f"select t.*, s.latitude, s.longitude, s.dateandtime {ts_select_statement} " \
                f"from {table_name} as t " \
                f"left join station as s on t.id=s.id " \
                f"{ts_join_statement} " \
                f"{quality_statement};"
        print(f"Structure.create_extended_parameter_tables: {query}")
        cur.execute(query)

        tables_params[new_name] = param_name.upper()

    return tables_params


def remove_tables_like(conn, like_pattern="E|_%", escape_char="|", table_type="view", tables_except=None):
    """
    Removes views/tables whose name match the given like_pattern.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        like_pattern (str): Table names should conform to this string pattern. If None, all parameter tables are
        queried. Default is 'E|_%'
        escape_char (str): Escape char used for the like pattern. If None, sqlite3 default is used. Default is '|'.
        table_type (str): Type of the structure (view or table) to remove. Default is 'view'.
        tables_except (list<str>): Remove all tables matching the like pattern except the ones specified in this list.
        Default is None.
    """
    # get views/tables to remove
    table_names = get_names_of_all_parameter_tables(conn, like_pattern, escape_char, table_type=table_type,
                                                    include_digits=True)
    cur = conn.cursor()

    if tables_except:
        for te in tables_except:
            if te in table_names:
                table_names.remove(te)

    for vtn in table_names:
        query = f"drop {table_type} {vtn};"
        print(f"Structure.remove_tables_like: {query}")
        cur.execute(query)


def create_combined_parameter_table(conn, parameters, quality_flags=None, columns=None, table_type="view"):
    """
    Creates a table/view that combines multiple parameters. A column 'PARAM_NAME' is added. The function assumes that
    tables named P_% exist.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        parameters (list<str>): List of parameter names.
        quality_flags (list<list<str, str>>): List containing the quality flags and their value.
        columns (list<str>): List of desired columns.
        table_type (str): Whether to create views or tables. Default is 'view'.
    """
    if columns is None:
        columns = ["ID", "LEV_DBAR", "LEV_M", "VAL", "PQF1", "PQF2", "SQF", "BOTTLE_NUMBER",
                   "PROFILE_NUMBER", "PROFILE_BEST", "UNITS_ID", "INSTRUMENT_ID"]
        # , "LATITUDE", "LONGITUDE", "DATEANDTIME"]

    # SQL statement to filter for quality flags
    quality_statement = ""
    if quality_flags:
        quality_statement = f"where {' and '.join('t.' + x[0] + x[1] for x in quality_flags)}"

    # SQL statement to select the desired columns from each parameter table/view
    select_params = []
    for p in parameters:
        # check if required view/table exists
        if does_table_exist(conn, f"P_{p.upper()}", "view") or does_table_exist(conn, f"P_{p.upper()}", "table"):
            select_params.append("select '" + p.upper() + "' as PARAM_NAME, " + ", ".join(columns) + " from P_" +
                                 p.upper() + " " + quality_statement)
        else:
            logging.warning(f"WARNING in viewsAndTables.create_combined_parameter_table: Table or view P_{p.upper()} "
                            f"does not exist. Cannot add {p.upper()} to the combined table.")

    # create new combined parameter table
    query = f"create {table_type} if not exists P_COMBINED as " \
            f"select * from (" \
            f"{' UNION ALL '.join(select_params)}" \
            f");"
    cur = conn.cursor()
    cur.execute(query)


def create_wide_parameter_table(conn, parameters, quality_flags=None, table_type="table", table_name="wide"):
    print("WARNING: THIS FUNCTION MAY BE ERRONEOUS!!")
    cur = conn.cursor()

    # SQL statement to filter for quality flags
    quality_statement = ""
    if quality_flags:
        temp = []
        for i in range(len(parameters)):
            temp.append(f"{' and '.join(f'p{str(i)}.' + x[0] + x[1] for x in quality_flags)}")
        quality_statement = "where " + " and ".join(temp)

    value_statement = ", ".join([f"p{str(i)}.VAL as {param}" for i, param in enumerate(parameters)])
    join_statement = " ".join([f"left join p_{param} as p{str(i)} using(latitude, longitude, lev_m, dateandtime)"
                               for i, param in enumerate(parameters)][1:])

    # @todo filtering for quality here results in a weird output table!! bug! do not use the quality filter!
    # @todo the number of samples increases when joining, even without quality_statement
    q = f"create {table_type} if not exists {table_name} as " \
        f"select p0.LATITUDE, p0.LONGITUDE, p0.LEV_M, p0.DATEANDTIME, {value_statement} " \
        f"from p_{parameters[0]} as p0 " \
        f"{join_statement} " \
        f"{quality_statement};"

    cur.execute(q)
