from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import os
import shutil
import time
import numpy as np
import pandas as pd
import datetimerange
import hvplot.pandas
import holoviews as hv
from project_configuration import globals
from src.database.communication import create_connection
from src.preprocessing.gridding import Grid, GridManager, create_wide_table, get_missing_value_info
from src.database.information import get_num_samples, get_names_of_all_parameter_tables

app = Dash(__name__)

colors = {
    'background': '#111111',
    'text': 'white'
}

# define grid parameters
dlat = 1
lat_min = 0
lat_max = 30
dlon = 1
lon_min = 120
lon_max = 180
dz = None  # 1000
z_min = None  # 0
z_max = None  # 6000
z_array = np.array([0, 500, 3000, 6000])
time_min = "1772-12-01 00:00:00"
time_max = "2020-11-01 00:00:00"
mode = "Y"
dtime = 300
selection = None
bathymetry_grid_path = "C:/Users/yvjennig/PycharmProjects/data/bathymetry/gebco_2022_sub_ice_topo/gebco_2022_sub_ice_topo.nc"
lat_variable = "lat"
lon_variable = "lon"
depth_variable = "elevation"

connection = create_connection(globals.db_path_preprocessed)
grid_manager = GridManager(globals.db_path_preprocessed, "grid_info")

param_tables = ["P_DOC", "P_TEMPERATURE", "P_SALINITY", "P_NITRATE"]
param_tables = get_names_of_all_parameter_tables(connection, include_digits=False)


def recompute(lat_min, lat_max, dlat,
              lon_min, lon_max, dlon,
              z_min, z_max, dz, z_array,
              time_min, time_max, mode, dtime, selection,
              bathymetry_grid_path, lat_variable, lon_variable, depth_variable):
    connection = create_connection(globals.db_path_preprocessed)
    grid_manager = GridManager(globals.db_path_preprocessed, "grid_info")
    grid = grid_manager.create_grid(lat_min=lat_min, lat_max=lat_max, dlat=dlat,
                                    lon_min=lon_min, lon_max=lon_max, dlon=dlon,
                                    z_min=z_min, z_max=z_max, dz=dz, z_array=z_array,
                                    time_min=time_min, time_max=time_max, mode=mode, dtime=dtime, selection=selection,
                                    bathymetry_grid_path=bathymetry_grid_path,
                                    lat_variable=lat_variable, lon_variable=lon_variable, depth_variable=depth_variable)
    grid.map_tables(connection, param_tables)

    # create wide table with nan values
    wide_table_name = create_wide_table(connection=connection, grid_id=grid.grid_id, param_tables=param_tables,
                                        copy=False)

    # compute fraction of missing values
    missing_value_info = get_missing_value_info(connection, wide_table_name, param_tables)

    connection.close()
    return missing_value_info


missing_value_info = recompute(lat_min, lat_max, dlat,
                               lon_min, lon_max, dlon,
                               z_min, z_max, dz, z_array,
                               time_min, time_max, mode, dtime, selection,
                               bathymetry_grid_path, lat_variable, lon_variable, depth_variable)

fig = px.bar(missing_value_info, x="parameter", y="relative", labels={"relative": "%", "parameter": ""})

# # assume you have a "long-form" data frame
# # see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })
#
# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group",
#              color_discrete_map={"SF": "#119dff", "Montreal": "#66c2a5"})
fig.update_layout(
    plot_bgcolor=colors["background"],
    paper_bgcolor=colors["background"],
    font_color=colors["text"]
)

lat_min_output = html.Div()

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[html.H1(children="Missing values",
                      style={"textAlign": "center", "color": colors["text"]}
                      ),
              html.Div(children='''Investigate missing value fractions based on the given grid.''',
                       style={"textAlign": "center", "color": colors["text"]}
                       ),
              dcc.Graph(
                  id='missing_value_bar_chart',
                  figure=fig
              ),
              html.Div(
                  children=["Latitude_min", dcc.Input(id="lat_min", type="number", min=-90, max=90, value=0, debounce=True),
                            "Latitude_max", dcc.Input(id="lat_max", type="number", min=-90, max=90, value=90, debounce=True),
                            "dlat", dcc.Input(id="dlat", type="number", min=0.0000001, max=180, value=1),
                            lat_min_output],
                  style={"textAlign": "left", "color": colors["text"]}),
              html.Div(
                  children=["Longitude_min", dcc.Input(id="lon_min", type="number", min=-180, max=180, value=0, debounce=True),
                            "Longitude_max", dcc.Input(id="lon_max", type="number", min=-180, max=180, value=90, debounce=True),
                            "dlon", dcc.Input(id="dlon", type="number", min=0.0000001, max=180, value=1)],
                  style={"textAlign": "left", "color": colors["text"]})
              ]
)


@app.callback(
    Output("missing_value_bar_chart", "figure"),
    Input("lat_min", "value")
)
def update_lat_min(new_lat_min):
    if new_lat_min is not None:
        print(f"New lat_min: {new_lat_min}")
        missing_value_info = recompute(new_lat_min, lat_max, dlat,
                                       lon_min, lon_max, dlon,
                                       z_min, z_max, dz, z_array,
                                       time_min, time_max, mode, dtime, selection,
                                       bathymetry_grid_path, lat_variable, lon_variable, depth_variable)
        fig = px.bar(missing_value_info, x="parameter", y="relative", labels={"relative": "%", "parameter": ""})
        fig.update_layout(
            plot_bgcolor=colors["background"],
            paper_bgcolor=colors["background"],
            font_color=colors["text"]
        )
        print("recomputation complete")
        return fig


if __name__ == '__main__':
    app.run_server(debug=True)
