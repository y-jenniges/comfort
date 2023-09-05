""" Classes and functions to scale parameter values. """
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class ParamScaler:
    """ Scales parameters to the range [0, 1] by providing one scaler per parameter. """

    def __init__(self):
        """
        Initializes the parameter scaler.

        Args:
            df (pandas.DataFrame): Dataframe containing the parameters to scale. There has to be a separate column for
            each parameter.
        """
        self.scalers = pd.DataFrame(columns=["scaler", "param_name"])

    def scale(self, df):
        temp = df.copy()

        param_name = temp["PARAM_NAME"].value_counts().index[0]  # .compute().index[0]
        #vals = temp["VAL"].to_dask_array(lengths=True).reshape(-1, 1)
        vals = np.array(temp["VAL"]).reshape(-1, 1)  # @todo possibly better convert it to dask?

        # check if a scaler is available for the given parameter
        if param_name in self.scalers["param_name"]:
            # if the scaler is already defined, use it
            xm = self.scalers[self.scalers["param_name"] == param_name]["scaler"].transform(vals)
        else:
            # if no scaler was defined yet, create one
            minmax_scaler = MinMaxScaler().fit(vals)
            xm = minmax_scaler.transform(vals)
            self.scalers.append({"scaler": minmax_scaler, "param_name": param_name}, ignore_index=True)

        temp["VAL_SCALED"] = xm
        # temp["VAL_SCALED"] = da.from_array(xm.ravel())
        return temp

    def scale_mixed_column(self, df):
        """
        Scales values in the column 'VAL'. Every different parameter (as defined by the column 'PARAM_NAME') is scaled
        separately.

        Args:
            df (pandas.DataFrame): Contains the values to scale in the column 'VAL'.
        Returns:
            scaled_df (pandas.DataFrame): Dataframe with scaled values in the column called 'VAL_SCALED'.
        """
        temp = df.copy()  # work on a copy of the dataframe

        # get columns and datatypes, add a row for the new metadata
        meta = temp.dtypes.reset_index()
        meta = meta.append({"index": "VAL_SCALED", 0: "float64"}, ignore_index=True)

        # convert meta data into a dict
        new_meta = {}
        for i in range(len(meta)):
            new_meta[meta["index"].iloc[i]] = meta[0].iloc[i]

        # group data by parameter name (shuffle to resort data along new index (param_name))
        # with dask.config.set(shuffle='tasks'):
        #     temp_grouped = temp.groupby("PARAM_NAME")
        temp_grouped = temp.groupby("PARAM_NAME")

        # apply scaling
        res = temp_grouped.apply(self.scale)
        # res = temp_grouped.apply(self.scale, meta=new_meta)

        return res

    def scale_columns(self, columns, df):
        """ Scales the given columns. They need to exist in the initially provided dataframe. Warning: One parameter
        only per column!

        Args:
            columns (list<str>): List of columns whose values should be scaled.
            df (pandas.DataFrame): Contains the values to scale. If None, the initially provided df will be used.
            Default is None.
        Returns:
            scaled_df (pandas.DataFrame): Dataframe with scaled values in a separate column.
        """
        temp = df.copy()

        for col in columns:
            # check if column is in dataframe
            if col not in df.columns:
                print("WARNING in scaling: Could not find the column {col} in the dataframe.")
                break

            # get the column values as numpy array
            vals = np.array(df[col]).reshape(-1, 1)  # df[col].to_dask_array(True).reshape(-1, 1)

            # check if a scaler is available for the given parameter
            if col in self.scalers["param_name"]:
                # if the scaler is already defined, use it
                xm = self.scalers[self.scalers["param_name"] == col]["scaler"].transform(vals)
            else:
                # if no scaler was defined yet, create one
                minmax_scaler = MinMaxScaler().fit(vals)
                xm = minmax_scaler.transform(vals)
                self.scalers.append({"scaler": minmax_scaler, "param_name": col}, ignore_index=True)

            # create a new column in the temporary dataframe with the scaled values
            temp[col + "_SCALED"] = xm
            # temp[col+"_SCALED"] = da.from_array(xm.ravel())

        return temp

    def scale_value(self, param_name, value):
        """ Scales a dingle value of a parameter that is in the dataframe. Before using this function, scale_columns has
        to be called to define the min and max values for that parameter (necessary for scaling).

        Args:
            param_name: Measured parameter name. It has to be present as a column in the dataframe used in the
            initialization.
            value (float): Value to scale.
        Returns:
            scaled_value (float): Input value scaled.
        """
        if param_name in self.scalers["param_name"]:
            scaled_value = self.scalers[self.scalers["param_name"] == param_name]["scaler"].transform(value)
            return scaled_value
        else:
            print(f"ERROR in scaler: Scaling failed, no scaler defined for {param_name}.")
            return None


class DBParamScaler:
    """
    Scales parameters to the range [0, 1] by operating directly on the database.

    Args:
        scalers (pandas.DataFrame): Contains the configuration of the scalers used for the different parameters. This
        can be used to e.g. transform the values back to their original range.
    """

    def __init__(self, connection, lower=0, upper=1, value_column="VAL", depth_column="LEV_M"):
        """
        Initializes the scaler.

        Args:
            connection (sqlite3.Connection): Connection to the database. It must hold one table per parameter.
            lower (float) : The new minimum value after scaling.
            upper (float) : The new maximum value after scaling.
            value_column (str): Name of the column containing the parameter values.
            depth_column (str): Name of the column containing the depth values.
        """
        self.connection = connection
        self.lower_limit = lower
        self.upper_limit = upper
        self.value_column = value_column
        self.depth_column = depth_column

        self.scalers = pd.DataFrame(columns=["param_name", "min", "max"])

    def get_min_max(self, table_name, column):
        query = f"select min({column}), max({column}) from {table_name};"
        res = self.connection.cursor().execute(query).fetchall()
        param_min = res[0][0]
        param_max = res[0][1]
        return param_min, param_max

    def get_min_max_depth(self, table_names):
        mins = []
        maxs = []
        for table in table_names:
            tmin, tmax = self.get_min_max(table, self.depth_column)
            mins.append(tmin)
            maxs.append(tmax)

        return min(mins), max(maxs)

    def min_max_scaling(self, table, column, param_min, param_max):
        formula = f"({table}.{column} - {param_min}) / ({param_max} - " \
                  f"{param_min}) * ({self.upper_limit} - {self.lower_limit}) + {self.lower_limit}"
        return formula

    def scale(self, tables_params):
        new_tables_params = {}
        cur = self.connection.cursor()

        # get min and max depth over all parameter tables
        param_tables = list(tables_params.keys())

        # only continue if the list is not empty
        if not param_tables:
            return

        min_depth, max_depth = self.get_min_max_depth(param_tables)
        self.scalers = self.scalers.append({"param_name": self.depth_column, "min": min_depth, "max": max_depth},
                                           ignore_index=True)

        # for every parameter table, scale the val column and the depth (lev_m) column
        for table, param in tables_params.items():
            print(f"DBScaler: Scaling {table, param}")
            param_min, param_max = self.get_min_max(table, self.value_column)
            val_scaling_formula = self.min_max_scaling(table, self.value_column, param_min, param_max)
            depth_scaling_formula = self.min_max_scaling(table, self.depth_column, min_depth, max_depth)
            self.scalers = self.scalers.append({"param_name": param, "min": param_min, "max": param_max},
                                               ignore_index=True)

            # get columns and define columns for the new table
            query = f"select * from {table} limit 1;"
            cur.execute(query)
            old_columns = [desc[0] for desc in cur.description]
            new_columns = [table + "." + x for x in old_columns
                           if x != self.value_column and x != "UNITS_ID" and x != self.depth_column]

            new_table_name = f"scaled_{table}"
            query = f"create table {new_table_name} as " \
                    f"select {', '.join(new_columns)}, " \
                    f"{val_scaling_formula} as {self.value_column}, " \
                    f"{depth_scaling_formula} as {self.depth_column} " \
                    f"from {table} " \
                    f";"
            self.connection.cursor().execute(query)

            # save table name
            new_tables_params[new_table_name] = param

        return new_tables_params
