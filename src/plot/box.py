import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def boxplot_per_parameter(conn, parameters, add_violin=True, save_as_prefix="auto", quality_flags=None):
    """
    Draws a boxplot of the values for each of the given parameters. Optionally adds a violinplot on top.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        parameters: Parameters for which to draw the boxplot.
        add_violin (bool): Default is True.
        save_as_prefix: If specified, the plot will be saved using this prefix_{param}.png. If 'auto', it will be
        f'5_comfort_potT__violinboxplot__{param}.png'. Default is 'auto'.
        quality_flags: If specified, filter for these quality flags.
    """
    cursor = conn.cursor()
    for param in parameters:
        print(param)
        # define SQL statement to filter for quality flags
        quality_statement = ""
        if quality_flags:
            quality_statement = f"where {' and '.join(x[0] + x[1] for x in quality_flags)}"

        # get data
        q = f"select VAL from P_{param} {quality_statement};"
        ex = cursor.execute(q)
        df = pd.DataFrame(ex.fetchall(), columns=[x[0] for x in cursor.description])

        # boxplot
        ax0 = sns.boxplot(x="VAL", data=df)
        if add_violin: ax1 = sns.violinplot(x="VAL", data=df, color="red")
        plt.title(f"{param}")
        plt.setp(ax0.collections, alpha=.3)

        if save_as_prefix:
            if save_as_prefix == "auto":
                plt.savefig(f"5_comfort_averaged__violinboxplot__{param}.png")
            else:
                plt.savefig(f"{save_as_prefix}_{param}.png")

        plt.tight_layout()
        plt.show()
        plt.close()


# # example usage
# import os
# import src.database.communication as comm
# from src.database.information import get_names_of_all_parameter_tables
# from project_configuration import globals
# # connect to DB and plot
# dir_path = os.path.dirname(os.path.abspath(globals.db_path))
# db_path = os.path.join(dir_path, "5_comfort_potT.sqlite")
# conn = comm.create_connection(db_path)
# cur = conn.cursor()
#
# # get all parameter tables and plot a histogram per parameter
# parameters = [x[2:] for x in get_names_of_all_parameter_tables(conn)]
# parameters.sort()
#
# boxplot_per_parameter(conn, parameters, add_violin=True, quality_flags=None, save_as="auto")
