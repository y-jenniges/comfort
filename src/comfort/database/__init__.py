"""Database communication, querying and schema management for COMFORT."""
from __future__ import annotations

from .information import (
    does_table_exist,
    get_min_max_dates,
    get_minmax,
    get_names_of_all_parameter_tables,
    get_num_samples,
    get_table_as_df,
)
from .structure import (
    create_combined_parameter_table,
    create_extended_parameter_tables,
    execute_sql_scripts,
    remove_tables_like,
)
