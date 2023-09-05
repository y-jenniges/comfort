import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
import numpy as np

# [["pqf1", ">0"], ["pqf2", ">2"], ["sqf", ">=-1"]]


def plot_monthly_hists(conn, param_names, quality_flags=None, save_as_prefix=None):
    """
    Plots monthly histograms for the given parameters.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        param_names (list<str>): Parameters to count the negative samples of.
        quality_flags (list<list<str, str>>): List of lists. Inner lists contain a quality parameter and their
        specification. Default is None.
        save_as_prefix (str): If specified, save the plot to this path. A suffix for each plot name is already
        specified. Default is None.
    """
    cur = conn.cursor()

    # determine SQL expression to consider quality flags
    quality_statement = ""
    if quality_flags:
        quality_statement = f"where {' and '.join(x[0] + x[1] for x in quality_flags)}"

    for param in param_names:
        print(param)
        # fetch all values for histogram
        q = f"select VAL, DATEANDTIME from p_{param} {quality_statement};"
        ex = cur.execute(q)
        df = pd.DataFrame(ex.fetchall(), columns=[param, "DATEANDTIME"])
        df["DATEANDTIME"] = pd.to_datetime(df["DATEANDTIME"])

        # plot
        sns.set(style="darkgrid")
        fig, axs = plt.subplots(3, 4, figsize=(6, 8))

        month = 0
        while month < 12:
            for i in range(3):
                for j in range(4):
                    month = month + 1
                    data = df[df["DATEANDTIME"].dt.month == month]
                    month_name = datetime.datetime.strptime(str(month), "%m").strftime("%B")

                    h = sns.histplot(data=data, x=param, kde=True, color="teal", ax=axs[i, j])
                    h.set(title=month_name)
                    h.set(xlabel=None, ylabel=None)
        plt.get_current_fig_manager().full_screen_toggle()
        fig.supylabel("Count")
        fig.supxlabel(param)
        plt.tight_layout()
        if save_as_prefix:
            plt.savefig(f"{save_as_prefix}_{param}_hist_monthly.png")
        plt.show(block=True)
        plt.close()


def plot_hists(conn, param_names, quality_flags=None, kind="hist", save_as_prefix=None):
    """
    Plot histograms for the given parameters.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        param_names (list<str>): Parameters to count the negative samples of.
        quality_flags (list<list<str, str>>): List of lists. Inner lists contain a quality parameter and their
        specification. Default is None.
        save_as_prefix (str): If specified, save the plot to this path. A suffix for each plot name is already
        specified. Default is None.
    """
    cur = conn.cursor()

    # determine SQL expression to consider quality flags
    quality_statement = ""
    if quality_flags:
        quality_statement = f"where {' and '.join(x[0] + x[1] for x in quality_flags)}"

    for param in param_names:
        print(param, count)

        # fetch all values for histogram
        q = f"select VAL from p_{param} {quality_statement};"
        ex = cur.execute(q)
        df = pd.DataFrame(ex.fetchall(), columns=[param.upper()], dtype=float)

        df.plot(kind=kind, title=f"Number of {param.upper()} samples", legend=False)
        if save_as_prefix:
            plt.savefig(f"{save_as_prefix}_{param}_{kind}.png")

        # # plot
        # sns.set(style="darkgrid")
        # fig, axs = plt.subplots(1, 1, figsize=(7, 7))
        # sns.histplot(data=df, x=param.upper(), kde=True, color="teal", bins=10)
        # plt.title(f"Number of {param.upper()} samples")
        # plt.xticks(rotation=90)
        # plt.tight_layout()
        # if save_as_prefix:
        #     plt.savefig(f"{save_as_prefix}_{param}_hist.png")
        # plt.show(block=True)
        # plt.close()


def plot_joint_plots(conn, param_names, quality_flags=None, save_as_prefix=None):
    """
    Plot joint plots for all combinations of the given parameters.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        param_names (list<str>): Parameters to count the negative samples of.
        quality_flags (list<list<str, str>>): List of lists. Inner lists contain a quality parameter and their
        specification. Default is None.
        save_as_prefix (str): If specified, save the plot to this path. A suffix for each plot name is already
        specified. Default is None.
    """
    cur = conn.cursor()

    # determine SQL expression to consider quality flags
    quality_statement = ""
    if quality_flags:
        quality_statement = f"where {' and '.join('p.' + x[0] + x[1] for x in quality_flags)} " \
                            f"and {' and '.join('q.' + x[0] + x[1] for x in quality_flags)}"

    # define all parameter combinations to plot
    combinations = []
    i = 0
    for p in param_names:
        temp = [(p, x) for x in param_names[i:] if p != x]
        combinations = combinations + temp
        i = i + 1

    for x, y in combinations:
        print(x, y)
        # get the data to plot
        q = f"select p.VAL, q.VAL from p_{x} as p left join p_{y} as q using(ID, LEV_M) " \
            f"{quality_statement};"
        ex = cur.execute(q)
        res = ex.fetchall()
        df = pd.DataFrame(res, columns=[x, y])

        # 2D histogram plot
        sns.jointplot(x=x, y=y, data=df, kind="hist")
        plt.tight_layout()
        if save_as_prefix:
            plt.savefig(f"{save_as_prefix}_{x}_{y}_jointplot.png")
        plt.show(block=True)
        plt.close()


def plot_correlation(conn, param_names, quality_flags=None, save_as_prefix=None):
    pass


import os
import src.database.communication as comm
from src.database.information import get_names_of_all_parameter_tables
from project_configuration import globals
# example usage
# connect to DB and plot
dir_path = os.path.dirname(os.path.abspath(globals.db_path))
db_path = os.path.join(dir_path, "5_comfort_averaged.sqlite")
conn = comm.create_connection(db_path)
cur = conn.cursor()

# get all parameter tables and plot a histogram per parameter
params = [x[2:] for x in get_names_of_all_parameter_tables(conn)]
plot_hists(conn, params)

# get all parameter tables and plot monthly histograms per parameter
params = [x[2:] for x in get_names_of_all_parameter_tables(conn)]
plot_monthly_hists(conn, params)

# only plot selected parameters in joint plot
params = ["nitrate", "silicate", "oxygen", "phosphate", "temperature", "salinity"]
plot_joint_plots(conn, params)


# create_wide_parameter_table(conn, parameters=["AOU", "ALKALINITY", "NITRATE"], quality_flags=None, table_type="table", table_name="wide")

# plot correlation matrix for all parameters
params = ["silicate", "nitrate"]
q = "select n.val as nitrate, s.val as silicate, o.val as oxygen, sa.val as salinity, t.val as temperature " \
    "from v_nitrate as n " \
    "left join v_silicate as s using(latitude, longitude, lev_m) " \
    "left join v_oxygen as o using(latitude, longitude, lev_m) " \
    "left join v_salinity as sa using(latitude, longitude, lev_m) " \
    "left join v_temperature as t using(latitude, longitude, lev_m);"
ex = cur.execute(q)
df = pd.DataFrame(ex.fetchall(), columns=["nitrate", "silicate", "oxygen", "salinity", "temperature"])
c = df.corr(method="pearson")
