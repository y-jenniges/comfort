"""
Module to plot sample counts, e.g. over time or over latitude and longitude.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
from scipy import stats
from src.database.information import get_names_of_all_parameter_tables, get_num_samples


def count_samples_over_time(df, date_column, time_mode="year"):
    """
    Count the number of samples for each of the 12 months or for every year.

    Args:
        df (pandas.DataFrame): Contains the data. Must have a column containing the dates.
        date_column (str): Column containing the dates
        time_mode (str): Over which timeframe to count the number of samples. Default is 'year'.
    Returns:
        Pandas.DataFrame containing the number of samples for the selected timeframe. First column is time, second is
        'COUNT'.
    """
    temp = df.copy()
    temp[date_column] = pd.to_datetime(temp[date_column])

    if time_mode == "year":
        min_year = temp[date_column].dt.year.min()
        max_year = temp[date_column].dt.year.max()
        year_range = np.array(range(min_year, max_year + 1))

        val_counts = temp[date_column].dt.year.value_counts()
        val_counts = val_counts.reindex(year_range, fill_value=0)
        val_counts = val_counts.reset_index()
        val_counts.columns = ["YEAR", "COUNT"]
        return val_counts

    elif time_mode == "month":
        month_range = np.array(range(1, 13))
        val_counts = temp[date_column].dt.month.value_counts()
        val_counts = val_counts.reindex(month_range, fill_value=0)
        val_counts = val_counts.reset_index()
        val_counts.columns = ["MONTH", "COUNT"]
        return val_counts

    else:
        print("Error: Please select either 'year' or 'month' to count the samples for.")
        return


def plot_count_over_time(df_count, time_mode="year", plot_title="Number of samples per year",
                         month_ticks=None, save_as=None):
    """
    Bar plot for the number of samples per year or month. First entry in df must be a collection of months/years,
    second is the count.

    Args:
        df_count (pandas.DataFrame): Dataframe containing the number of samples per year/month. First column must be
        month or year. Second column must be the sample count.
        time_mode (str): Whether to plot monthly or yearly counts. '' will ignore the time. Default ist 'year'.
        plot_title (str): Title of the plot. Default is 'Number of samples per year'.
        month_ticks (list<str>): A list containing 12 entries used as month labels (for plots in time_mode 'month'). If
        None, three-letter abbreviation of months will be used. Default is None.
        save_as (str): If specified, save the plot to this filename. Default is None.
    """
    if month_ticks is None:
        month_ticks = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if df_count is None or df_count.empty:
        print("Count: Passed dataframe is empty.")
        return

    cols = df_count.columns

    # plot
    x = df_count[cols[0]].map(str).to_list()
    y = df_count[cols[1]].to_list()

    plt.figure()
    plt.barh(x, y)
    for i in range(len(x)):
        plt.text(y[i], i, y[i], ha='left', va="center")
    plt.title(plot_title)
    plt.ylabel("Count")

    if time_mode == "year":
        min_year = df_count.iloc[0, 0]
        max_year = df_count.iloc[-1, 0]
        min_year_rounded = np.round(min_year, -1)
        min_tick = min_year_rounded if min_year_rounded >= min_year else min_year_rounded+10
        ticks = [str(min_year)] + [str(x) for x in range(min_tick, max_year, 10)] + [str(max_year)]
        plt.xticks(ticks, rotation=90)
    elif time_mode == "month":
        plt.xticks(df_count.iloc[:, 0]-1, month_ticks)

    # save plot if a path is specified
    if save_as:
        # create the directory if it does not exist yet
        save_dir = os.path.split(os.path.abspath(save_as))[0]
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_as)

    plt.tight_layout()
    plt.show(block=True)


def count_and_plot_over_time(conn, param_name, chunk_size=1000000, date_column="DATEANDTIME", time_mode="year",
                             plot_title="Number of samples per year", quality_flags=None, save_as=None):
    """
    Query data from a sqlite3 database chunk-wise, assemble the count and finally plot it. The function
    assumes that tables named P_% exist that contain information about latitude, longitude and dateandtime.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        param_name (str): Name of the parameter to count the samples of.
        chunk_size (int): Number of samples to query at once. Default is 1000000.
        date_column (str): Name of the column containing the dates. Default is 'DATEANDTIME'
        time_mode (str): Whether to plot monthly or yearly counts. Default ist 'year'.
        plot_title (str): Title of the plot. Default is 'Number of samples per year'.
        quality_flags (list<list<str, str>>): List of lists. Inner lists contain a quality parameter and their
        specification. Default is None.
        save_as (str): If specified, save the plot to this path. Default is None.
    Returns:
        df_count (pandas.DataFrame): Sample count per parameter.
    """
    # determine SQL expression to consider quality flags
    quality_statement = ""
    if quality_flags:
        quality_statement = f"where {' and '.join(x[0] + x[1] for x in quality_flags)}"

    # query data
    cursor = conn.cursor()
    query = f"select LATITUDE, LONGITUDE, DATEANDTIME from p_{param_name.lower()} {quality_statement};"
    ex = cursor.execute(query)
    cols = [description[0].upper() for description in cursor.description]
    df_count = pd.DataFrame(columns=["COUNT"])

    # counting number of samples chunk-wise
    while True:
        result = ex.fetchmany(chunk_size)
        if not result:
            print("Count: No data left to fetch.")
            break
        sample_count = count_samples_over_time(pd.DataFrame(result, columns=cols), date_column, time_mode)
        sample_count = sample_count.set_index(time_mode.upper())
        df_count = df_count.reindex(set(list(sample_count.index)+list(df_count.index)), fill_value=0)  # add new indices
        df_count = df_count.add(sample_count, axis=0, fill_value=0)

    # clean up dataframe and plot
    print(f"Count: Number of samples per chosen time is {df_count}")
    df_count = df_count.reset_index()
    df_count.columns = [time_mode.upper(), "COUNT"]
    plot_count_over_time(df_count, time_mode, plot_title, save_as)

    return df_count


def count_and_plot_multiple_params(conn, param_names, quality_flags=None, plot_title="Number of samples per parameter",
                                   save_as=None):
    """
    Counts number of samples per parameter directly in the sqlite database and plots it.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        param_names (list<str>): List of parameter names or None (then all parameters will be counted).
        quality_flags (list<list<str, str>>): List of lists. Inner lists contain a quality parameter and their
        specification. Default is None.
        plot_title(str): Title for the plot.
        save_as (str): If specified, save the plot using this path and name. Default is None.
    Returns:
        df_count (pandas.DataFrame): Frame containing the number of samples per parameter.
    """
    # get name of all parameter tables if no parameters are specified
    if not param_names:
        param_names = [x[2:] for x in get_names_of_all_parameter_tables(conn)]

    # count number of samples per parameter
    df_count = pd.DataFrame({"parameter": param_names, "count": 0})
    for i in range(len(param_names)):
        res = get_num_samples(conn, f"P_{param_names[i]}", table_type="table", quality_flags=quality_flags)
        df_count.loc[i, "count"] = int(res[0])

    # plot count
    plot_count_over_time(df_count.sort_values("count"), time_mode="", plot_title=plot_title, save_as=save_as)
    return df_count


def count_and_plot_over_lat_lon(conn, param_name, resolution=0.5, clim=None, quality_flags=None, save_as=None):
    """
    Counts the number of samples of one parameter at the same latitudes and longitudes. Plots the count on a map.
    (WARNING: The lat, lon values must fit into memory!). The function assumes that tables named P_% exist that contain
    latitude and longitude information.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        param_name: Parameter to count the samples of.
        resolution (float): Resolution of the world grid/bins. Default is 0.5.
        clim (list<float, float>): If specified, the parameter range will be cropped to these values. Default is None.
        quality_flags (list<list<str, str>>): List of lists. Inner lists contain a quality parameter and their
        specification. Default is None.
        save_as (str): If specified, save the plot to this path. Default is None.
    """
    cursor = conn.cursor()

    # determine SQL expression to consider quality flags
    quality_statement = ""
    if quality_flags:
        quality_statement = f"where {' and '.join(x[0] + x[1] for x in quality_flags)}"

    # get the number of values measured at the same position
    table_name = f"p_{param_name.lower()}"
    query = f"select latitude, longitude from {table_name} {quality_statement};"
    ex = cursor.execute(query)
    columns = [x[0] for x in cursor.description]
    res = ex.fetchall()
    df = pd.DataFrame(res, columns=columns)

    # basic plot resolutions
    dlatlon = resolution
    lonbins = np.arange(-180, 180, dlatlon)
    latbins = np.arange(-90, 90, dlatlon)

    # get latitude and longitude
    lat = df["LATITUDE"]
    lon = df["LONGITUDE"]

    # compute number of observations in each bin
    hist = stats.binned_statistic_2d(lat, lon, None, bins=[latbins, lonbins], statistic="count")
    hist.statistic[hist.statistic == 0] = np.nan  # sets all values of 0 to nan as log10(0) = -inf

    # plot the histogram
    fig = plt.figure()
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='grey')
    ax.gridlines(draw_labels=True)
    cmap = plt.cm.get_cmap("Spectral_r", 64)
    image = plt.pcolormesh(lonbins, latbins, hist.statistic, cmap=cmap, shading="flat",
                           transform=ccrs.PlateCarree())
    fig.colorbar(image, ax=ax, orientation='horizontal', fraction=0.1, aspect=40, pad=0.08,
                 label=f"Number of {param_name} samples")

    # limit value range if specified
    if clim:
        plt.clim(clim[0], clim[1])

    # save plot if a path is specified
    if save_as:
        # create the directory if it does not exist yet
        save_dir = os.path.split(os.path.abspath(save_as))[0]
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_as)

    plt.show(block=True)


def count_and_plot_negative_samples(conn, param_names, quality_flags=None, save_as=None):
    """
    Counts the number of negative samples per parameter and plots relative number of negative samples. The function
    assumes that tables named P_% exist that contain information on the value of samples (VAL column).

    Args:
        conn (sqlite3.Connection): Connection to the database.
        param_names (list<str>): Parameters to count the negative samples of.
        quality_flags (list<list<str, str>>): List of lists. Inner lists contain a quality parameter and their
        specification. Default is None.
        save_as (str): If specified, save the plot to this path. Default is None.

    Returns:
        df (pandas.DataFrame): Contains total sample count, total and relative negative sample count per parameter.

    """
    cursor = conn.cursor()

    # determine SQL expression to consider quality flags
    quality_statement = ""
    if quality_flags:
        quality_statement = f"and {' and '.join(x[0] + x[1] for x in quality_flags)}"

    df = pd.DataFrame(columns=["parameter", "number of samples", "number of negative samples"])
    for param_name in param_names:
        print(param_name)
        # get number of negative samples
        q = f"select count(*) from P_{param_name} where VAL<0 {quality_statement};"
        ex = cursor.execute(q)
        negative_count = ex.fetchall()[0][0]

        # get total number of samples
        q = f"select count(*) from P_{param_name};"
        ex = cursor.execute(q)
        count = ex.fetchall()[0][0]

        df = df.append({"parameter": param_name, "number of samples": count,
                        "number of negative samples": negative_count}, ignore_index=True)

    # plot
    df["proportion of negative samples"] = df["number of negative samples"] / df["number of samples"] * 100
    df = df.sort_values("proportion of negative samples")
    ax = sns.barplot(x="proportion of negative samples", y="parameter", data=df)
    ax.bar_label(ax.containers[0], fmt="%.2f")
    if save_as:
        plt.savefig(save_as)
    plt.show(block=True)

    return df


def count_and_plot_spatiotemporal_duplicates(conn, param_names, quality_flags=None, save_as_prefix=None):
    """
    Different plots to investigate statistics and distribution of spatiotemporal duplicates. The function
    assumes that tables named P_% exist that contain VAL, ID and LEV_M columns.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        param_names (list<str>): Parameters to count the negative samples of.
        quality_flags (list<list<str, str>>): List of lists. Inner lists contain a quality parameter and their
        specification. Default is None.
        save_as_prefix (str): If specified, save the plot to this path. A suffix for each plot name is already
        specified. Default is None.
    """
    cursor = conn.cursor()

    # determine SQL expression to consider quality flags
    quality_statement = ""
    if quality_flags:
        quality_statement = f"where {' and '.join(x[0] + x[1] for x in quality_flags)}"

    df = pd.DataFrame(columns=["parameter", "number of samples", "number of row duplicates",
                               "number of time loc duplicates"])
    for param_name in param_names:
        print(param_name)

        # get data
        q = f"select * from P_{param_name} {quality_statement};"
        ex = cursor.execute(q)
        df_param = pd.DataFrame(ex.fetchall(), columns=[x[0] for x in cursor.description])

        # duplicate values in time and location
        df_dups_time_loc = df_param.groupby(["ID", "LEV_M"], as_index=False, dropna=False).agg(size=('VAL', 'size'),
                                                                                               mean=("VAL", "mean"),
                                                                                               std=('VAL', 'std'))
        df_dups_time_loc = df_dups_time_loc[df_dups_time_loc["size"] > 1]

        if len(df_dups_time_loc) < 1:
            print("    no duplicates")
            continue

        # completely duplicate rows
        df_dups_row = df_param.groupby(df_param.columns.tolist(), as_index=False, dropna=False).size()
        df_dups_row = df_dups_row[df_dups_row["size"] > 1]

        df = df.append({"parameter": param_name, "number of samples": len(df_param),
                        "number of row duplicates": len(df_dups_row),
                        "number of time loc duplicates": len(df_dups_time_loc)}, ignore_index=True)

        # plot duplicates on scatter plot - goal: check location
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
        sns.scatterplot(ax=axs[0], data=df_dups_time_loc, x="ID", y="LEV_M", hue="size", size="size")
        axs[0].set_title(f"{param_name} - Number of duplicates in time and location")

        sns.scatterplot(ax=axs[1], data=df_dups_row, x="ID", y="LEV_M", hue="size", size="size")
        axs[1].set_title(f"{param_name} - Number of duplicates in all columns")

        plt.xticks(rotation=90)
        plt.tight_layout()
        if save_as_prefix:
            plt.savefig(f"{save_as_prefix}_{param_name}_number_of_duplicates.png")
        plt.show(block=True)
        plt.close()

        # plot - goal: investigate mean, std and size/count of duplicates (time/loc duplicates)
        sns.scatterplot(data=df_dups_time_loc, x="size", y="mean", size="std", hue="std")
        plt.title(f"{param_name} - Time/loc duplicate statistics")
        plt.tight_layout()
        if save_as_prefix:
            plt.savefig(f"{save_as_prefix}_{param_name}_duplicates_statistics.png")
        plt.show(block=True)
        plt.close()

        # plot - goal: further investigate standard deviation
        sns.boxplot(data=df_dups_time_loc, x="size", y="std")
        plt.title(f"{param_name} - Time/loc duplicate standard deviation")
        plt.tight_layout()
        if save_as_prefix:
            plt.savefig(f"{save_as_prefix}_{param_name}_duplicates_std.png")
        plt.show(block=True)
        plt.close()

    # plot number of duplicates
    df["proportion of row duplicates"] = df["number of row duplicates"] / df["number of samples"] * 100
    df["proportion of time loc duplicates"] = df["number of time loc duplicates"] / df["number of samples"] * 100

    width = 0.35
    x = np.arange(len(df))
    fig, ax = plt.subplots()
    b0 = ax.bar(x - width/2, df["proportion of row duplicates"], width, label="Row duplicates")
    b1 = ax.bar(x + width/2, df["proportion of time loc duplicates"], width, label="Time/location duplicates")

    ax.set_ylabel("%")
    ax.set_title("Proportion of duplicates per parameter")

    ax.set_xticks(x)
    ax.set_xticklabels(df["parameter"], rotation=90)

    ax.bar_label(b0, padding=3)
    ax.bar_label(b1, padding=3)

    ax.legend()
    plt.tight_layout()
    plt.savefig(f"ORIGINAL_duplicates.png")
    plt.show(block=True)


# # example usage (copy to a different file)
# from src.database import structure, communication as comm
# from src.plot.count import count_and_plot_over_time, count_and_plot_multiple_params, count_and_plot_over_lat_lon
# from project_configuration import globals
# # connect to db
# conn = comm.create_connection(globals.db_path)
# chunk_size = 1000
# date_column = "DATEANDTIME"
# time_mode = "year"
#
# count_and_plot_over_time(conn, "argon", chunk_size, date_column, time_mode,
#                          plot_title="Number of argon samples per year")
# count_and_plot_over_time(conn, "don", chunk_size, date_column, time_mode,
#                          plot_title="Number of DON samples per year")
# count_and_plot_multiple_params(conn, param_names=["argon", "don"])
# count_and_plot_over_lat_lon(conn, "argon", resolution=1)
# count_and_plot_negative_samples(conn, ["ARGON", "ALKALINITY"])
# count_and_plot_spatiotemporal_duplicates(conn, ["ARGON", "ALKALINITY"])
