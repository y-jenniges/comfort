"""Temperature-Salinity (T-S) diagram with isopycnal contours.

The plot is adapted from:

    Yvonne Jenniges (2025).
    y-jenniges/ocean_clustering_and_validation:
    Biogeochemical Ocean Regions - Code Base (v1.0.1).
    Zenodo. https://doi.org/10.5281/zenodo.15827777
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import gsw
import numpy as np

from ._helpers import _finish, _get_or_create_ax

if TYPE_CHECKING:
    import matplotlib.axes
    import pandas as pd

logger = logging.getLogger(__name__)


_SAL_LABELS: dict[str, str] = {
    "SA": r"$S_A$ [g kg$^{-1}$]",
    "SP": r"$S_P$ [1]",
}
_TEMP_LABELS: dict[str, str] = {
    "CT": r"$\Theta$ [°C]",
    "pt": r"$\theta$ [°C]",
    "pt0": r"$\theta$ [°C]",
    "t": r"$T$ [°C]",
}
_TEOS10_SAL = {"SA"}
_TEOS10_TEMP = {"CT"}


def plot_ts_diagram(
    df: pd.DataFrame,
    temp_col: str,
    sal_col: str,
    *,
    temp_type: str = "CT",
    sal_type: str = "SA",
    color_col: str | None = None,
    depth_col: str = "LEV_M",
    cmap: str = "viridis_r",
    colorbar_label: str | None = None,
    density_contours: bool = True,
    contour_levels: int | list[float] | None = None,
    kind: str = "scatter",
    gridsize: int = 40,
    marker_size: float = 9,
    alpha: float = 0.6,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    save_as: str | None = None,
    dpi: int = 300,
) -> matplotlib.axes.Axes | None:
    """Temperature-Salinity diagram with optional isopycnal contours.
    Density contours are only accurate when ``sal_type="SA"`` and ``temp_type="CT"``.

    Axis labels are derived from *sal_type* / *temp_type* when *xlabel* /
    *ylabel* are not supplied explicitly:

    * ``sal_type="SA"``  -> :math:`S_A` [g kg-1]  - Absolute Salinity (TEOS-10)
    * ``sal_type="SP"``  -> :math:`S_P` [1]  - Practical Salinity (dimensionless)
    * ``temp_type="CT"`` -> :math:`\\Theta` [°C] - Conservative Temperature (TEOS-10)
    * ``temp_type="pt"`` / ``"pt0"`` -> :math:`\\theta` [°C] - Potential Temperature
    * ``temp_type="t"``  -> :math:`T` [°C] - In-situ Temperature

    Args:
        df: DataFrame containing temperature and salinity.
        temp_col: Column with temperature values.
        sal_col: Column with salinity values.
        temp_type: Variable type for *temp_col* - ``"CT"``, ``"pt"``,
            ``"pt0"``, or ``"t"``.  Controls the y-axis label and
            density contour accuracy.
        sal_type: Variable type for *sal_col* - ``"SA"`` or ``"SP"``.
            Controls the x-axis label and density contour accuracy.
        color_col: Column to colour the scatter by. Falls back to
            *depth_col* when ``None``.
        depth_col: Depth column used as default colour variable.
        cmap: Matplotlib colormap name.
        colorbar_label: Colour-bar label (defaults to the colour column).
        density_contours: Overlay sigma0 contour lines.
        contour_levels: Explicit contour levels or number of levels.
        kind: ``"scatter"`` (default) or ``"hexbin"``.
        gridsize: Number of hexagons across the x-axis when ``kind="hexbin"``.
        marker_size: Scatter marker size.
        alpha: Scatter marker opacity.
        xlabel: x-axis label (defaults to label derived from *sal_type*).
        ylabel: y-axis label (defaults to label derived from *temp_type*).
        xlim: Salinity axis limits.
        ylim: Temperature axis limits.
        ax: Axes to draw into.
        save_as: File path to save the figure.
        dpi: Resolution (dots per inch) used when saving.

    Returns:
        The matplotlib Axes, or ``None`` when the data is empty.
    """
    # Determine colour column
    if color_col is not None:
        # Use passed color_col
        use_color = color_col
    elif kind != "hexbin":
        # Default (for scatter) is depth observation count
        use_color = depth_col
    else:
        # Default (for hexbin) is observation count
        use_color = None

    # Define which columns need full data
    needed = [temp_col, sal_col]
    if use_color in df.columns:
        needed.append(use_color)

    # Extract valid data
    valid = df[needed].dropna()
    if valid.empty:
        logger.warning("plot_ts_diagram: No valid data after dropping NaNs.")
        return None

    # Warning if density contours need different sal/temp to be accurate
    if density_contours and (sal_type not in _TEOS10_SAL or temp_type not in _TEOS10_TEMP):
        logger.warning(
            "Density contours use gsw.rho(SA, CT, 0) but sal_type=%r, temp_type=%r - "
            "contours will be inaccurate. Pass SA and CT for correct isopycnals.",
            sal_type, temp_type,
        )

    ax, standalone = _get_or_create_ax(ax)

    # Isopycnal contour grid
    if density_contours:
        s_min = xlim[0] if xlim else valid[sal_col].min()
        s_max = xlim[1] if xlim else valid[sal_col].max()
        t_min = ylim[0] if ylim else valid[temp_col].min()
        t_max = ylim[1] if ylim else valid[temp_col].max()

        s_pad = 0.01 * abs(s_max - s_min) or 0.5
        t_pad = 0.1 * abs(t_max - t_min) or 0.5
        s_min -= s_pad
        s_max += s_pad
        t_min -= t_pad
        t_max += t_pad

        # Build an S-T grid and compute sigma-0 at each node
        si = np.linspace(s_min, s_max, max(int(round((s_max - s_min) / 0.1)) + 1, 10))
        ti = np.linspace(t_min, t_max, max(int(round((t_max - t_min) / 0.1)) + 1, 10))
        s_grid, t_grid = np.meshgrid(si, ti)
        sigma0 = gsw.rho(s_grid, t_grid, 0) - 1000

        # Overlay dashed isopycnal lines
        levels = contour_levels if contour_levels is not None else 12
        cs = ax.contour(si, ti, sigma0, levels=levels, linestyles="dashed", colors="k", alpha=0.5)
        contour_labels = ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
        # Labels near the edge of the padded grid otherwise get sliced by
        # the axes border instead of drawn into the surrounding margin
        for label in contour_labels:
            label.set_clip_on(False)

    # Plot observations on top of the contours
    has_color_data = use_color in valid.columns
    if kind == "hexbin":
        hb = ax.hexbin(valid[sal_col], valid[temp_col], C=valid[use_color] if has_color_data else None, reduce_C_function=np.mean, gridsize=gridsize, cmap=cmap, mincnt=1)
        ax.figure.colorbar(hb, ax=ax, label=colorbar_label or use_color if has_color_data else "count")
    else:
        scatter_kw = dict(s=marker_size, alpha=alpha, marker=".", edgecolors="none")
        if has_color_data:
            sc = ax.scatter(valid[sal_col], valid[temp_col], c=valid[use_color], cmap=cmap, **scatter_kw)
            ax.figure.colorbar(sc, ax=ax, label=colorbar_label or use_color)
        else:
            ax.scatter(valid[sal_col], valid[temp_col], **scatter_kw)

    # Axes labels and limits
    ax.set_xlabel(xlabel or _SAL_LABELS.get(sal_type, sal_col))
    ax.set_ylabel(ylabel or _TEMP_LABELS.get(temp_type, temp_col))
    ax.set_title("T-S diagram")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    return _finish(ax, standalone, save_as, dpi)
