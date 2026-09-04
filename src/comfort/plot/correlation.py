"""Correlation and joint-distribution plots for COMFORT parameter data."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..qc import build_where_clause
from ..util.sqlite_utils import validate_identifier
from ._helpers import _finish, _get_or_create_ax, _save_fig

if TYPE_CHECKING:
    import sqlite3
    import matplotlib.axes
    from ..qc import QCFilter

logger = logging.getLogger(__name__)


def plot_correlation(
    df: pd.DataFrame | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    *,
    color_col: str | None = None,
    cmap: str = "viridis",
    kind: str = "scatter",
    gridsize: int = 40,
    regression: bool = True,
    annotate_r2: bool = True,
    marker_size: float = 8,
    alpha: float = 0.4,
    xlabel: str | None = None,
    ylabel: str | None = None,
    conn: sqlite3.Connection | None = None,
    parameters: tuple[str, str] | None = None,
    quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes | None:
    """Scatter plot of two variables with optional regression line and r2.

    Accepts either a *df* with two named columns, or a database *conn* with
    a *(param_x, param_y)* tuple to join two parameter tables.

    Args:
        df: DataFrame containing *x_col* and *y_col*.
        x_col: Column name for the x-axis.
        y_col: Column name for the y-axis.
        color_col: Optional column for colour coding.
        cmap: Matplotlib colormap (used when *color_col* is set).
        kind: ``"scatter"`` (default) or ``"hexbin"``.
        gridsize: Number of hexagons across the x-axis when ``kind="hexbin"``.
        regression: Overlay a linear regression line.
        annotate_r2: Annotate the plot with the r2 value.
        marker_size: Scatter marker size.
        alpha: Marker opacity.
        xlabel: x-axis label (defaults to *x_col*).
        ylabel: y-axis label (defaults to *y_col*).
        conn: Database connection (alternative to *df*).
        parameters: ``(param_x, param_y)`` names (without ``P_`` prefix).
        quality_flags: QC filters forwarded to the SQL query.
        ax: Axes to draw into.
        save_as: File path to save the figure.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib axes or ``None`` when the data is empty.
    """
    # Load from database when no DataFrame is given
    if df is None:
        df, x_col, y_col = _load_pair(conn, parameters, quality_flags)

    if x_col is None or y_col is None:
        raise ValueError("x_col and y_col are required")

    # Drop rows with missing values
    valid = df[[x_col, y_col]].dropna()
    if color_col and color_col in df.columns:
        # Ensure color_col is present
        valid = df[[x_col, y_col, color_col]].dropna()
    if valid.empty:
        logger.warning("plot_correlation: No valid data after dropping NaNs")
        return None

    ax, standalone = _get_or_create_ax(ax)

    if kind == "hexbin":
        # Hexbin plot
        has_color = color_col and color_col in valid.columns
        hb = ax.hexbin(
            valid[x_col], valid[y_col],
            C=valid[color_col] if has_color else None,
            reduce_C_function=np.mean, gridsize=gridsize, cmap=cmap, mincnt=1,
        )

        # Colorbar
        ax.figure.colorbar(hb, ax=ax, label=color_col if has_color else "count")
    else:
        # Scatter plot with optional colour coding
        scatter_kw = dict(s=marker_size, alpha=alpha, edgecolors="none")
        if color_col and color_col in valid.columns:
            sc = ax.scatter(valid[x_col], valid[y_col], c=valid[color_col],
                            cmap=cmap, **scatter_kw)
            ax.figure.colorbar(sc, ax=ax, label=color_col)
        else:
            ax.scatter(valid[x_col], valid[y_col], **scatter_kw)

    # Linear regression and coefficient of determination (R2)
    if regression or annotate_r2:
        x = valid[x_col].values.astype(float)
        y = valid[y_col].values.astype(float)
        coeffs = np.polyfit(x, y, 1)
        r2 = np.corrcoef(x, y)[0, 1] ** 2

        if regression:
            # Draw regression line
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, np.polyval(coeffs, x_line), color="red",
                    linewidth=1.5, label=f"y = {coeffs[0]:.3g}x + {coeffs[1]:.3g}")
        if annotate_r2:
            # Add R2 annotation
            ax.annotate(
                f"$r^2$ = {r2:.3f}",
                xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    # Axes labels
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)

    # Legend
    if regression:
        ax.legend(loc="lower right")

    return _finish(ax, standalone, save_as, dpi)


def plot_joint(
    df: pd.DataFrame | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    *,
    kind: str = "hist",
    conn: sqlite3.Connection | None = None,
    parameters: tuple[str, str] | None = None,
    quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> sns.JointGrid | None:
    """Joint plot of two variables (scatter with marginal distributions).

    Accepts either a *df* or a database *conn* with *(param_x, param_y)*.

    Args:
        df: DataFrame containing *x_col* and *y_col*.
        x_col: Column for x-axis.
        y_col: Column for y-axis.
        kind: Joint-plot kind (``'scatter'``, ``'hist'``, ``'kde'``, ``'hex'``, ``'reg'``).
        conn: Database connection (alternative to *df*).
        parameters: ``(param_x, param_y)`` names (without ``P_`` prefix).
        quality_flags: QC filters.
        save_as: File path to save the figure.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        :class:`seaborn.JointGrid`, or ``None`` when the data is empty.
    """
    # Load from database when no DataFrame is given
    if df is None:
        df, x_col, y_col = _load_pair(conn, parameters, quality_flags)

    if x_col is None or y_col is None:
        raise ValueError("x_col and y_col are required")

    # Drop rows with missing values
    valid = df[[x_col, y_col]].dropna()
    if valid.empty:
        logger.warning("plot_joint: no valid data after dropping NaNs")
        return None

    # Joint plot
    g = sns.jointplot(x=x_col, y=y_col, data=valid, kind=kind)
    plt.tight_layout()
    _save_fig(save_as, dpi)
    return g


def _load_pair(conn, parameters, quality_flags):
    """Join two parameter tables and return (df, x_col, y_col)."""
    if conn is None or parameters is None or len(parameters) != 2:
        raise ValueError(
            "Provide either a DataFrame (df) or conn + parameters=(param_x, param_y)"
        )
    x_name, y_name = parameters

    # Validate table identifiers
    validate_identifier(f"P_{x_name}")
    validate_identifier(f"P_{y_name}")

    # Build quality-flag clauses for both tables
    where = build_where_clause(quality_flags, table_alias="p")
    and_clause = ""
    if quality_flags:
        and_clause = " " + build_where_clause(quality_flags, table_alias="q", keyword="AND")

    # Join the two parameter tables on station ID and depth
    sql = (
        f"SELECT p.VAL, q.VAL FROM P_{x_name} AS p "
        f"LEFT JOIN P_{y_name} AS q USING(ID, LEV_M) "
        f"{where}{and_clause};"
    )
    cur = conn.cursor()
    rows = cur.execute(sql).fetchall()
    df = pd.DataFrame(rows, columns=np.array([x_name, y_name]))
    return df, x_name, y_name
