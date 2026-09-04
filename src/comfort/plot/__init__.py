"""Visualisation functions for COMFORT data.

Requires:
    pip install "comfort-db[plot]"
"""
from __future__ import annotations
try:
    import matplotlib
except ImportError:
    raise ImportError(
        "matplotlib is required for comfort.plot; "
        'install it with: pip install "comfort-db[plot]"'
    )

# --- profiles & sections: how does this vary with depth? ---
from .profile import plot_profile, plot_section

# --- relationships: how do two variables relate to each other? ---
from .relationships import plot_correlation, plot_joint, plot_ts_diagram

# --- distributions: what's the spread of values? ---
from .distribution import (
    boxplot,
    compare_stats,
    plot_histogram,
    plot_monthly_histogram,
)

# --- spatial: where is the data, and where is it missing? ---
from .spatial import (
    plot_lat_lon_range,
    plot_missing_value_info_map,
    plot_missing_value_info_map_over_depth,
    plot_spatial_distribution,
)

# --- coverage: how much data is there, and how complete is it? ---
from .coverage import (
    count_and_plot_negative_samples,
    count_and_plot_samples_over_time,
    count_and_plot_samples_per_parameter,
    count_negative_samples,
    count_samples_over_time,
    count_samples_over_time_from_db,
    count_samples_per_parameter,
    detect_and_plot_spatiotemporal_duplicates,
    plot_annual_coverage,
    plot_counts_bar,
    plot_missing_value_info,
)

__all__ = [
    # profiles & sections
    "plot_profile",
    "plot_section",
    # relationships
    "plot_correlation",
    "plot_joint",
    "plot_ts_diagram",
    # distributions
    "boxplot",
    "compare_stats",
    "plot_histogram",
    "plot_monthly_histogram",
    # spatial
    "plot_spatial_distribution",
    "plot_lat_lon_range",
    "plot_missing_value_info_map",
    "plot_missing_value_info_map_over_depth",
    # coverage
    "count_samples_over_time",
    "count_samples_over_time_from_db",
    "count_samples_per_parameter",
    "count_negative_samples",
    "plot_annual_coverage",
    "plot_counts_bar",
    "plot_missing_value_info",
    "count_and_plot_samples_over_time",
    "count_and_plot_samples_per_parameter",
    "count_and_plot_negative_samples",
    "detect_and_plot_spatiotemporal_duplicates",
]
