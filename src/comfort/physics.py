"""TEOS-10 physical oceanography functions for COMFORT data."""
from __future__ import annotations

import logging
import gsw
import numpy as np
import pandas as pd


def _to1d(x):
    """Convert x to a 1D float64 numpy array."""
    return np.atleast_1d(np.asarray(x, dtype=float))


def _find_col(df, col):
    """Return the actual column name in *df* matching *col* (case-insensitive).

    Raises KeyError if no match is found.
    """
    # Return the col
    if col in df.columns:
        return col

    # Check if lower-case version is in df.columns
    lower = col.lower()
    for c in df.columns:
        if c.lower() == lower:
            return c

    # Column not found
    raise KeyError(
        f"Column {col!r} not found in DataFrame. "
        f"Available columns: {list(df.columns)}"
    )


def convert_salinity(sp: float | np.ndarray | pd.Series,
                     pressure: float | np.ndarray | pd.Series,
                     longitude: float | np.ndarray | pd.Series,
                     latitude: float | np.ndarray | pd.Series) -> pd.Series:
    """Convert Practical Salinity to Absolute Salinity (TEOS-10).

    Args:
        sp (array-like): Practical Salinity [PSS-78].
        pressure (array-like): Sea pressure [dbar] (0 dbar at the ocean surface).
        longitude (array-like): Longitude [degrees East].
        latitude (array-like): Latitude [degrees North].
    Returns:
        pandas.Series: Absolute Salinity SA [g/kg].
    """
    return pd.Series(gsw.SA_from_SP(_to1d(sp), _to1d(pressure),
                                    _to1d(longitude), _to1d(latitude)))


def convert_temperature(t_insitu: float | np.ndarray | pd.Series,
                        sa: float | np.ndarray | pd.Series,
                        pressure: float | np.ndarray | pd.Series,
                        to: str = "CT") -> pd.Series:
    """Convert in-situ temperature to a TEOS-10 temperature variable.

    Args:
        t_insitu (array-like): In-situ temperature [deg C, ITS-90].
        sa (array-like): Absolute Salinity [g/kg]
            (from :func:`convert_salinity`).
        pressure (array-like): Sea pressure [dbar].
        to (str): Target variable:
            ``"CT"`` - Conservative Temperature [deg C];
            ``"pt0"`` - potential temperature ref. 0 dbar [deg C].
    Returns:
        pandas.Series
    Raises:
        ValueError: If ``to`` is not ``'CT'`` or ``'pt0'``.
    """
    # Prepare variables
    t = _to1d(t_insitu)
    sa_ = _to1d(sa)
    p = _to1d(pressure)

    # Convert
    if to == "CT":
        return pd.Series(gsw.CT_from_t(sa_, t, p))
    if to == "pt0":
        return pd.Series(gsw.pt0_from_t(sa_, t, p))
    raise ValueError(f"Unknown target {to!r}; choose 'CT' or 'pt0'")


def compute_density(sa: float | np.ndarray | pd.Series,
                    ct: float | np.ndarray | pd.Series,
                    pressure: float | np.ndarray | pd.Series | None = None,
                    quantity: str = "sigma0") -> pd.Series:
    """Compute seawater density or potential density anomaly (TEOS-10).

    Args:
        sa (array-like): Absolute Salinity [g/kg].
        ct (array-like): Conservative Temperature [deg C].
        pressure (array-like): Sea pressure [dbar]. Required only for
            ``quantity='rho'``, ignored otherwise.
        quantity (str): What to compute:
            ``"sigma0"`` - potential density anomaly [kg/m3], ref. 0 dbar;
            ``"sigma1"`` - [kg/m3], ref. 1000 dbar;
            ``"sigma2"`` - [kg/m3], ref. 2000 dbar;
            ``"rho"`` - in-situ density [kg/m3].
    Returns:
        pandas.Series
    Raises:
        ValueError: If ``quantity`` is unknown or pressure is missing
            for ``'rho'``.
    """
    # Prepare variables
    sa_ = _to1d(sa)
    ct_ = _to1d(ct)

    # Compute density based on sigma type
    if quantity == "sigma0":
        return pd.Series(gsw.sigma0(sa_, ct_))
    if quantity == "sigma1":
        return pd.Series(gsw.sigma1(sa_, ct_))
    if quantity == "sigma2":
        return pd.Series(gsw.sigma2(sa_, ct_))

    # Compute density from pressure
    if quantity == "rho":
        if pressure is None:
            raise ValueError("pressure is required for quantity='rho'")
        return pd.Series(gsw.rho(sa_, ct_, _to1d(pressure)))
    raise ValueError(
        f"Unknown quantity {quantity!r}; choose 'sigma0', 'sigma1', 'sigma2' or 'rho'"
    )


def compute_buoyancy_frequency(df: pd.DataFrame, sa_col: str, ct_col: str,
                               pressure_col: str = "LEV_DBAR",
                               profile_col: str = "PROFILE_NUMBER",
                               lat_col: str = "LATITUDE") -> pd.DataFrame:
    """Compute Brunt-Vaisala (buoyancy) frequency N2 for each profile.

    N2 > 0 indicates stable stratification; N2 < 0 indicates gravitational
    instability. Values are computed at mid-depths between consecutive
    levels using :func:`gsw.Nsquared`.

    Requires temperature and salinity on the same row - use a wide-format
    DataFrame (e.g. from :func:`add_teos10_variables`).

    Args:
        df (pandas.DataFrame): Wide-format DataFrame with SA and CT columns.
        sa_col (str): Column with Absolute Salinity [g/kg].
        ct_col (str): Column with Conservative Temperature [deg C].
        pressure_col (str): Column with sea pressure [dbar].
        profile_col (str): Column identifying individual profiles.
        lat_col (str): Latitude column [degrees N] for gravitational
            acceleration correction. Falls back to 0 if absent.
    Returns:
        pandas.DataFrame: One row per consecutive level pair per profile
            with columns [profile_col, ``'pressure_mid'``, ``'N2'``]
            (N2 in 1/s2).
    """
    lat_available = lat_col in df.columns
    rows = []

    # Iterate over profiles
    for pid, group in df.groupby(profile_col):
        # Check if profile has min. 2 levels
        cols = [pressure_col, sa_col, ct_col] + ([lat_col] if lat_available else [])
        valid = group[cols].dropna().sort_values(pressure_col)
        if len(valid) < 2:
            logging.debug("compute_buoyancy_frequency: profile %s has < 2 levels, skipped", pid)
            continue

        # Fall back to equatorial gravity if latitude not available
        lat = valid[lat_col].values if lat_available else np.zeros(len(valid))

        # N2 is computed at mid-depth pressure levels between consecutive pairs
        n2, p_mid = gsw.Nsquared(
            valid[sa_col].values, valid[ct_col].values,
            valid[pressure_col].values, lat=lat,
        )
        for val, pmid in zip(n2, p_mid):
            rows.append({profile_col: pid, "pressure_mid": float(pmid), "N2": float(val)})

    return pd.DataFrame(rows)


def compute_spiciness(sa: float | np.ndarray | pd.Series,
                      ct: float | np.ndarray | pd.Series,
                      reference_pressure: int = 0) -> pd.Series:
    """Compute oceanographic spiciness as a water-mass tracer (TEOS-10).

    Spiciness increases with temperature and salinity along isopycnals:
    high = warm and salty; low = cold and fresh at the same density.
    Use alongside sigma0 in T-S space to identify water masses.

    Args:
        sa (array-like): Absolute Salinity [g/kg].
        ct (array-like): Conservative Temperature [deg C].
        reference_pressure (int): Reference pressure in dbar:
            ``0`` - upper ocean (spiciness0);
            ``2000`` - deep ocean (spiciness2).
    Returns:
        pandas.Series: Spiciness [kg/m3].
    Raises:
        ValueError: If ``reference_pressure`` is not 0 or 2000.
    """
    # Prepare variables
    sa_ = _to1d(sa)
    ct_ = _to1d(ct)

    # Return spiciness according to reference pressure
    if reference_pressure == 0:
        return pd.Series(gsw.spiciness0(sa_, ct_))
    if reference_pressure == 2000:
        return pd.Series(gsw.spiciness2(sa_, ct_))
    raise ValueError(
        f"reference_pressure must be 0 or 2000; got {reference_pressure}"
    )


def add_teos10_variables(df: pd.DataFrame, sp_col: str, t_col: str,
                         pressure_col: str = "LEV_DBAR",
                         lon_col: str = "LONGITUDE",
                         lat_col: str = "LATITUDE") -> pd.DataFrame:
    """Add SA, CT and sigma0 columns to a DataFrame in a single call.

    Convenience wrapper around :func:`convert_salinity`,
    :func:`convert_temperature` and :func:`compute_density`.

    Args:
        df (pandas.DataFrame): DataFrame with Practical Salinity, in-situ
            temperature, sea pressure, longitude and latitude columns.
        sp_col (str): Column with Practical Salinity [PSS-78].
        t_col (str): Column with in-situ temperature [deg C, ITS-90].
        pressure_col (str): Column with sea pressure [dbar].
        lon_col (str): Column with longitude [degrees East].
        lat_col (str): Column with latitude [degrees North].
    Returns:
        pandas.DataFrame: Copy of *df* with three added columns:
            ``SA`` [g/kg], ``CT`` [deg C], ``sigma0`` [kg/m3].
    """
    out = df.copy()

    # Resolve column names (case-insensitive matching)
    sp = _find_col(out, sp_col)
    t = _find_col(out, t_col)
    p = _find_col(out, pressure_col)
    lon = _find_col(out, lon_col)
    lat = _find_col(out, lat_col)

    # Convert step by step: SP -> SA -> CT -> sigma0
    out["SA"] = convert_salinity(out[sp], out[p], out[lon], out[lat]).values
    out["CT"] = convert_temperature(out[t], out["SA"], out[p], to="CT").values
    out["sigma0"] = compute_density(out["SA"], out["CT"], quantity="sigma0").values
    return out
