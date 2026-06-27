"""comfort-db - helper library for the COMFORT oceanographic dataset."""

__version__ = "0.1.0"

from .io import connect, load_comfort, describe_variables, subset_region, list_parameters, read_parameter
from .qc import QC_ALL, QC_GOOD, QCFilter, apply_qc_flags
from .database.information import get_table_as_df, does_table_exist
from .analysis import detect_depth_col
from .gridding import Grid, GridManager, SpaceGrid
from .units import UnitsConverter
from .scaling import ParamScaler
from .util.sqlite_utils import vacuum

__all__ = [
    "connect",
    "load_comfort",
    "describe_variables",
    "subset_region",
    "list_parameters",
    "read_parameter",
    "QC_ALL",
    "QC_GOOD",
    "QCFilter",
    "apply_qc_flags",
    "get_table_as_df",
    "does_table_exist",
    "Grid",
    "GridManager",
    "SpaceGrid",
    "UnitsConverter",
    "ParamScaler",
    "detect_depth_col",
    "vacuum",
]
