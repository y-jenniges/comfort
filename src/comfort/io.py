"""High-level I/O helpers for reading and subsetting COMFORT parameter data."""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import xarray as xr

from .profile_analysis import interpolate_depth_levels
from .database.information import get_names_of_all_parameter_tables
from .qc import QCFilter, build_where_clause
from .util.sqlite_utils import validate_identifier


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a connection to the COMFORT SQLite database and close it on exit.

    Args:
        db_path (str or Path): Path to the COMFORT SQLite database.
    Yields:
        sqlite3.Connection
    """
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


# Depth levels (similar to Jenniges et al. 2026 (https://doi.org/10.1016/j.ecoinf.2025.103390)
DEPTH_INTERVALS = np.array([0, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000])


def info(db_path_or_conn: str | Path | sqlite3.Connection) -> pd.DataFrame:
    """Print a quick summary of the COMFORT database without loading data.

    Shows parameter names, row counts, depth range and date range.

    Args:
        db_path_or_conn (str, Path or sqlite3.Connection): Path to the
            COMFORT SQLite database or an existing open connection.
    Returns:
        pandas.DataFrame: Summary table (one row per parameter).
    """
    # Open connection if path given
    own_conn = not isinstance(db_path_or_conn, sqlite3.Connection)
    conn = sqlite3.connect(db_path_or_conn) if own_conn else db_path_or_conn
    try:
        params = list_parameters(conn)
        if not params:
            logging.warning("info: no parameter tables found")
            return pd.DataFrame()

        # Collect sample count and depth/date range per parameter
        rows = []
        for param in params:
            validate_identifier(param)
            table = f"P_{param}"
            cur = conn.cursor()
            cur.execute(
                f"SELECT COUNT(*), MIN(LEV_M), MAX(LEV_M) FROM {table};"
            )
            n, dmin, dmax = cur.fetchone()

            # Get date range via station join
            try:
                cur.execute(
                    f"SELECT MIN(s.DATEANDTIME), MAX(s.DATEANDTIME) "
                    f"FROM {table} p JOIN station s ON p.ID = s.ID;"
                )
                date_min, date_max = cur.fetchone()
            except Exception:
                date_min, date_max = None, None

            rows.append({
                "parameter": param,
                "n_samples": n,
                "depth_min_m": dmin,
                "depth_max_m": dmax,
                "date_min": date_min,
                "date_max": date_max,
            })

        result = pd.DataFrame(rows)
        logging.info(result.to_string(index=False))
        return result
    finally:
        if own_conn:
            conn.close()


def list_parameters(conn: sqlite3.Connection) -> list[str]:
    """Return the names of all ``P_*`` parameter tables (without the ``P_`` prefix).

    Args:
        conn (sqlite3.Connection): Connection to the database.
    Returns:
        list[str]: Parameter names, e.g. ``['NITRATE', 'OXYGEN', ...]``.
    """
    return [name[2:] for name in get_names_of_all_parameter_tables(conn)]


def read_parameter(conn: sqlite3.Connection, param_name: str,
                   quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
                   profiles: list[int] | None = None,
                   lat_min: float | None = None, lat_max: float | None = None,
                   lon_min: float | None = None, lon_max: float | None = None,
                   depth_min: float | None = None, depth_max: float | None = None,
                   date_min: str | None = None, date_max: str | None = None,
                   limit: int | None = None) -> pd.DataFrame:
    """Read a P_* parameter table as a DataFrame with optional QC filtering.

    When any spatial or temporal filter is supplied the ``station`` table is
    joined automatically and ``LATITUDE``, ``LONGITUDE`` and ``DATEANDTIME``
    columns are included.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        param_name (str): Parameter name (without ``'P_'`` prefix).
        quality_flags (list[QCFilter] or list[tuple[str, str]]): QC filters,
            e.g. ``QC_GOOD``.
        profiles (list[int]): Profile numbers to load. ``None`` loads all.
        lat_min (float): Minimum latitude [degrees N].
        lat_max (float): Maximum latitude [degrees N].
        lon_min (float): Minimum longitude [degrees E].
        lon_max (float): Maximum longitude [degrees E].
        depth_min (float): Minimum depth [m].
        depth_max (float): Maximum depth [m].
        date_min (str or datetime-like): Start date (inclusive).
        date_max (str or datetime-like): End date (inclusive).
        limit (int): Maximum number of rows to return. ``None`` returns all.
            Useful as a safeguard for large tables.
    Returns:
        pandas.DataFrame
    """
    # Validate identifier
    validate_identifier(param_name)

    # Check if geo filters are requested
    has_geo = any(v is not None for v in (
        lat_min, lat_max, lon_min, lon_max,
        depth_min, depth_max, date_min, date_max,
    ))
    params: list = []

    if has_geo:
        # Build query with geo, profile and quality filters
        conditions = [f"p.{col}{cond}" for col, cond in (quality_flags or [])]
        if profiles is not None:
            placeholders = ",".join("?" * len(profiles))
            conditions.append(f"p.PROFILE_NUMBER IN ({placeholders})")
            params.extend(profiles)
        _append_geo_conditions(
            conditions, params, "s.", "s.", "p.", "s.",
            lat_min, lat_max, lon_min, lon_max,
            depth_min, depth_max, date_min, date_max,
        )
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = (
            f"SELECT p.*, s.LATITUDE, s.LONGITUDE, "
            f"strftime('%Y-%m-%d %H:%M:%S', s.DATEANDTIME) AS DATEANDTIME "
            f"FROM P_{param_name.upper()} p "
            f"LEFT JOIN station s ON p.ID = s.ID {where}"
        )
    else:
        # Build query with quality and profile filters
        conditions = [f"{col}{cond}" for col, cond in (quality_flags or [])]
        if profiles is not None:
            placeholders = ",".join("?" * len(profiles))
            conditions.append(f"PROFILE_NUMBER IN ({placeholders})")
            params.extend(profiles)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT * FROM P_{param_name.upper()} {where}"

    # Add optional limit to query
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    # Fetch data directly into DataFrame
    return pd.read_sql_query(sql + ";", conn, params=params)


def read_station(conn: sqlite3.Connection) -> pd.DataFrame:
    """Read the station metadata table as a DataFrame.

    Args:
        conn (sqlite3.Connection): Connection to the database.
    Returns:
        pandas.DataFrame
    """
    df = pd.read_sql_query("SELECT * FROM STATION;", conn)
    if "DATEANDTIME" in df.columns:
        df["DATEANDTIME"] = df["DATEANDTIME"].str.slice(0, 19)
    return df


def read_extended(conn: sqlite3.Connection, param_name: str,
                  quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
                  limit: int | None = None) -> pd.DataFrame:
    """Read an E_* extended view (parameter + lat/lon/time) as a DataFrame.

    Low-level helper. Prefer :func:`load_comfort` for most use cases.
    The E_* views must be created first via
    :func:`comfort.database.structure.create_extended_parameter_tables`.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        param_name (str): Parameter name (without ``'E_'`` prefix).
        quality_flags (list[QCFilter] or list[tuple[str, str]]): QC filters,
            e.g. ``QC_GOOD``.
        limit (int): Maximum number of rows to return. ``None`` returns all.
            Useful as a safeguard for large tables.
    Returns:
        pandas.DataFrame
    """
    # Validate identifier
    validate_identifier(param_name)

    # Build quality filter
    where = build_where_clause(quality_flags)

    # Fetch data directly into DataFrame
    sql = f"SELECT * FROM E_{param_name.upper()} {where}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return pd.read_sql_query(sql + ";", conn)


def describe_variables(conn: sqlite3.Connection,
                       parameters: list[str] | None = None,
                       quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Summarise available parameters: sample counts, depth range, value range,
    units, instruments and platforms.

    Lookup tables (``UNITS``, ``INSTRUMENT``, ``PLATFORM``) resolve IDs to
    human-readable names when present; otherwise IDs are shown as strings.
    Parameters with multiple distinct values show all names separated by ``/``.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        parameters (list[str]): Parameter names (without ``P_`` prefix).
            ``None`` describes all available parameters.
        quality_flags (list[list[str, str]]): Optional QC filters applied
            when counting samples and computing ranges.
    Returns:
        pandas.DataFrame: One row per parameter with columns
            ``[parameter, n_samples, n_stations, n_profiles, n_instruments,
            depth_min_m, depth_max_m, val_min, val_max,
            units, instruments, platforms]``.
    """
    # Get a list of all parameters
    all_params = list_parameters(conn)

    # Iterate over given parameters
    if parameters is not None:
        # Warn if parameters are unknown
        unknown = set(parameters) - set(all_params)
        if unknown:
            raise ValueError(f"Unknown parameters: {sorted(unknown)}")
        params = parameters
    else:
        params = all_params

    # Warn if no parameter left
    if not params:
        logging.warning("describe_variables: no parameter tables found")
        return pd.DataFrame()

    # Compute aggregate statistics per parameter
    where = build_where_clause(quality_flags)
    qc_where_aliased = build_where_clause(quality_flags, "p")
    has_platform_id = _station_has_column(conn, "PLATFORM_ID")

    rows = []
    for param in params:
        # Validate identifier
        validate_identifier(param)

        # Define table name
        table = f"P_{param}"

        # Connect to db and fetch parameter statistics
        cur = conn.cursor()
        cur.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT ID), "
            f"COUNT(DISTINCT INSTRUMENT_ID), "
            f"MIN(LEV_M), MAX(LEV_M), MIN(VAL), MAX(VAL) "
            f"FROM {table} {where};"
        )
        n, n_sta, n_inst, dmin, dmax, vmin, vmax = cur.fetchone()

        # Count distinct (ID, PROFILE_NUMBER) pairs - PROFILE_NUMBER is only unique within a station
        cur.execute(
            f"SELECT COUNT(*) FROM "
            f"(SELECT DISTINCT ID, PROFILE_NUMBER FROM {table} {where});"
        )
        n_prof = cur.fetchone()[0]

        # Get IDs of units and instruments
        cur.execute(f"SELECT DISTINCT UNITS_ID FROM {table} {where};")
        unit_ids = [row[0] for row in cur.fetchall() if row[0] is not None]
        cur.execute(f"SELECT DISTINCT INSTRUMENT_ID FROM {table} {where};")
        inst_ids = [row[0] for row in cur.fetchall() if row[0] is not None]

        # Get platform IDs via station join
        platform_ids: list[int] = []
        if has_platform_id:
            try:
                cur.execute(
                    f"SELECT DISTINCT s.PLATFORM_ID "
                    f"FROM {table} p JOIN station s ON p.ID = s.ID "
                    f"{qc_where_aliased};"
                )
                platform_ids = [row[0] for row in cur.fetchall() if row[0] is not None]
            except Exception:
                pass

        rows.append({
            "parameter": param,
            "n_samples": n,
            "n_stations": n_sta,
            "n_profiles": n_prof,
            "n_instruments": n_inst,
            "depth_min_m": dmin,
            "depth_max_m": dmax,
            "val_min": vmin,
            "val_max": vmax,
            "units_ids": unit_ids,
            "instrument_ids": inst_ids,
            "platform_ids": platform_ids,
        })
    result = pd.DataFrame(rows)

    # Resolve unit IDs to names
    id_to_unit = _load_lookup_map(conn, "UNITS", "ID", "NAME_SHORT")
    result["units"] = result["units_ids"].apply(
        lambda ids: " / ".join(
            id_to_unit.get(i, str(i)) for i in sorted(ids)
        ) if ids else None
    )
    # Warn about mixed units
    for _, row in result.iterrows():
        if len(row["units_ids"]) > 1:
            logging.warning(
                "%s contains %d different units (%s) - values may not be comparable",
                row["parameter"], len(row["units_ids"]), row["units"],
            )
    result = result.drop(columns="units_ids")

    # Resolve instrument IDs to names
    id_to_instrument = _load_lookup_map(conn, "INSTRUMENT")
    result["instruments"] = result["instrument_ids"].apply(
        lambda ids: " / ".join(
            id_to_instrument.get(i, str(i)) for i in sorted(ids)
        ) if ids else None
    )
    result = result.drop(columns="instrument_ids")

    # Resolve platform IDs to names
    id_to_platform = _load_lookup_map(conn, "PLATFORM")
    result["platforms"] = result["platform_ids"].apply(
        lambda ids: " / ".join(
            id_to_platform.get(i, str(i)) for i in sorted(ids)
        ) if ids else None
    )
    result = result.drop(columns="platform_ids")

    return result


def _load_lookup_map(conn, table: str, id_col: str = "ID", name_col: str = "NAME") -> dict:
    """Return a dict mapping id_col -> name_col, or {} when the table is absent."""
    from .database.information import does_table_exist

    try:
        if not does_table_exist(conn, table):
            return {}
        cur = conn.cursor()
        cur.execute(f"SELECT {id_col}, {name_col} FROM {table};")
        return {row[0]: row[1] for row in cur.fetchall()}
    except Exception as exc:
        logging.debug("_load_lookup_map: could not resolve %s (%s)", table, exc)
        return {}


def _station_has_column(conn, column: str) -> bool:
    """Check whether the station table has a given column."""
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(station);")
        return any(row[1].upper() == column.upper() for row in cur.fetchall())
    except Exception:
        return False


def subset_region(df: pd.DataFrame,
                  lat_min: float | None = None, lat_max: float | None = None,
                  lon_min: float | None = None, lon_max: float | None = None,
                  depth_min: float | None = None, depth_max: float | None = None,
                  date_min: str | None = None, date_max: str | None = None) -> pd.DataFrame:
    """Filter a DataFrame to a geographic, depth and/or time region.

    Logs a warning when a requested bound cannot be applied because the
    corresponding column is absent. Recognised columns: LATITUDE, LONGITUDE,
    LEV_M, DATEANDTIME.

    Args:
        df (pandas.DataFrame): Extended parameter or station DataFrame.
        lat_min (float): Minimum latitude in degrees N.
        lat_max (float): Maximum latitude in degrees N.
        lon_min (float): Minimum longitude in degrees E.
        lon_max (float): Maximum longitude in degrees E.
        depth_min (float): Minimum depth in metres.
        depth_max (float): Maximum depth in metres.
        date_min (str or datetime-like): Start date (inclusive).
        date_max (str or datetime-like): End date (inclusive).
    Returns:
        pandas.DataFrame: Filtered copy.
    """

    # Apply a single bound, skip with warning if column absent
    def _apply(mask, col, bound_name, bound_val, op):
        if bound_val is None:
            return mask
        if col not in df.columns:
            logging.warning(
                "subset_region: %r requested but column %r not found - filter skipped",
                bound_name, col,
            )
            return mask
        return mask & op(df[col], bound_val)

    import operator as op

    # Spatial and depth bounds
    mask = pd.Series(True, index=df.index)
    mask = _apply(mask, "LATITUDE", "lat_min", lat_min, op.ge)
    mask = _apply(mask, "LATITUDE", "lat_max", lat_max, op.le)
    mask = _apply(mask, "LONGITUDE", "lon_min", lon_min, op.ge)
    mask = _apply(mask, "LONGITUDE", "lon_max", lon_max, op.le)
    mask = _apply(mask, "LEV_M", "depth_min", depth_min, op.ge)
    mask = _apply(mask, "LEV_M", "depth_max", depth_max, op.le)

    # Date range filter
    if date_min is not None or date_max is not None:
        if "DATEANDTIME" not in df.columns:
            logging.warning(
                "subset_region: date filter requested but column 'DATEANDTIME' not found - filter skipped"
            )
        else:
            dates = pd.to_datetime(df["DATEANDTIME"])
            if date_min is not None:
                mask &= dates >= pd.Timestamp(date_min)
            if date_max is not None:
                mask &= dates <= pd.Timestamp(date_max)

    return df[mask].copy()


def load_comfort(
        db_path_or_conn: str | Path | sqlite3.Connection,
        parameters: list[str] | None = None,
        quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None,
        lat_min: float | None = None, lat_max: float | None = None,
        lon_min: float | None = None, lon_max: float | None = None,
        depth_min: float | None = None, depth_max: float | None = None,
        date_min: str | None = None, date_max: str | None = None,
        target_depths: np.ndarray | list[float] | None = None,
        as_xarray: bool = True,
        normalise_columns: bool = False,
        convert_units: bool = False,
        limit: int | None = None,
) -> xr.Dataset | dict[str, pd.DataFrame]:
    """Load COMFORT data into a filtered xarray Dataset or dict of DataFrames.

    All filtering (QC flags, spatial bounds and date range) is pushed to SQL
    so only the requested rows are loaded into memory. Tries E_* views first
    and falls back to ``P_* JOIN station``.

    For xarray output each parameter is interpolated onto ``target_depths``
    (default: WOD standard levels) with dimensions ``(profile, depth)``.

    Args:
        db_path_or_conn (str, Path or sqlite3.Connection): Path to the COMFORT
            SQLite database or an existing open connection. When a path is
            given, the connection is opened and closed automatically. When a
            connection is passed, it is left open.
        parameters (list[str]): Parameter names (without prefix).
            ``None`` loads all available parameters.
        quality_flags: QC filter preset, e.g. :data:`comfort.qc.QC_GOOD`.
            ``None`` means no filtering.
        lat_min (float): Minimum latitude [degrees N].
        lat_max (float): Maximum latitude [degrees N].
        lon_min (float): Minimum longitude [degrees E].
        lon_max (float): Maximum longitude [degrees E].
        depth_min (float): Minimum depth [m].
        depth_max (float): Maximum depth [m].
        date_min (str or datetime-like): Start date (inclusive).
        date_max (str or datetime-like): End date (inclusive).
        target_depths (array-like): Depth levels [m] for interpolation.
            Defaults to :data:`DEPTH_INTERVALS`.
        as_xarray (bool): ``True`` returns ``xr.Dataset``, ``False`` returns
            ``dict[param_name, pandas.DataFrame]``.
        normalise_columns (bool): When ``True`` and ``as_xarray=False``,
            rename all DataFrame columns to lowercase. Default ``False``.
        convert_units (bool): When ``True``, convert each parameter's values
            to its default unit using :class:`comfort.units.UnitsConverter`.
            Requires ``DATABASE_TABLES`` and ``UNITS`` tables in the database;
            silently skipped when they are absent.
        limit (int): Maximum number of rows to load per parameter.
            ``None`` loads all matching rows. Useful as a safeguard for
            large tables like TEMPERATURE or SALINITY.
    Returns:
        xarray.Dataset or dict[str, pandas.DataFrame].
        For the dict path each DataFrame has the measurement column named
        after the parameter (e.g. ``"NITRATE"``), not ``"VAL"``.
    """
    # Collect geo/temporal filter arguments
    geo_kwargs = dict(
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
        depth_min=depth_min, depth_max=depth_max,
        date_min=date_min, date_max=date_max,
    )

    # Open connection if path given; leave caller-owned connections open
    own_conn = not isinstance(db_path_or_conn, sqlite3.Connection)
    conn = sqlite3.connect(db_path_or_conn) if own_conn else db_path_or_conn
    try:
        # Default to all available parameters
        if parameters is None:
            parameters = list_parameters(conn)

        # Resolve lookup tables for xarray coordinates
        _instrument_map = _load_lookup_map(conn, "INSTRUMENT")
        _platform_map = _load_lookup_map(conn, "PLATFORM")
        _units_map = _load_lookup_map(conn, "UNITS", "ID", "NAME_SHORT")

        # Station → platform_id mapping (when column exists)
        _station_platform: dict[int, int] = {}
        if _station_has_column(conn, "PLATFORM_ID"):
            try:
                cur = conn.cursor()
                cur.execute("SELECT ID, PLATFORM_ID FROM station;")
                _station_platform = {
                    row[0]: row[1] for row in cur.fetchall() if row[1] is not None
                }
            except Exception:
                pass

        # Build unit converter when requested
        _converter = None
        if convert_units:
            try:
                from .units import UnitsConverter
                _converter = UnitsConverter.from_connection(conn)
            except ValueError:
                logging.info("load_comfort: UNITS/DATABASE_TABLES not found, skipping unit conversion")

        # Track per-parameter unit IDs for xarray attrs and warnings
        _param_unit_ids: dict[str, set] = {}

        # Load each parameter with SQL-level filtering
        dfs = {}
        for param in parameters:
            try:
                df = _read_with_geo(conn, param, quality_flags, limit=limit, **geo_kwargs)
            except Exception as exc:
                logging.warning("load_comfort: could not read %s (%s)", param, exc)
                continue
            if not df.empty:
                # Track distinct units
                if "UNITS_ID" in df.columns:
                    uid_set = set(df["UNITS_ID"].dropna().unique())
                    _param_unit_ids[param] = uid_set
                    if len(uid_set) > 1 and _converter is None:
                        names = ", ".join(
                            _units_map.get(int(u), str(int(u)))
                            for u in sorted(uid_set)
                        )
                        logging.warning(
                            "%s contains %d different units (%s) - "
                            "values may not be comparable; consider convert_units=True",
                            param, len(uid_set), names,
                        )

                # Convert to default unit before renaming VAL
                if _converter is not None and "VAL" in df.columns:
                    df = _converter.convert_dataframe(df, f"P_{param}")

                # Rename generic VAL column to parameter name
                if "VAL" in df.columns:
                    df = df.rename(columns={"VAL": param})
                if normalise_columns:
                    df.columns = df.columns.str.lower()
                dfs[param] = df
    finally:
        if own_conn:
            conn.close()

    if not dfs:
        logging.warning("load_comfort: no data found for the given filters")
        if as_xarray:
            import xarray as xr
            return xr.Dataset()
        return {}

    # Return raw DataFrames if xarray not requested
    if not as_xarray:
        return dfs

    # Build xarray Dataset with dims (profile, depth)
    import xarray as xr

    # Target depth levels for interpolation
    depths = np.asarray(
        target_depths if target_depths is not None else DEPTH_INTERVALS,
        dtype=float,
    )

    # Collect unique (station_id, profile_number) pairs as canonical profile identities
    all_profiles = sorted(
        set().union(*[
            set(zip(df["ID"], df["PROFILE_NUMBER"]))
            if "ID" in df.columns and "PROFILE_NUMBER" in df.columns
            else {(None, pn) for pn in df["PROFILE_NUMBER"].unique()}
            for df in dfs.values()
        ])
    )
    n_profiles = len(all_profiles)
    n_depths = len(depths)

    # Extract station metadata from first row of each (ID, PROFILE_NUMBER) pair
    meta: dict = {}
    for df in dfs.values():
        group_keys = ["ID", "PROFILE_NUMBER"] if "ID" in df.columns else ["PROFILE_NUMBER"]
        for pid, group in df.groupby(group_keys):
            key = pid if isinstance(pid, tuple) else (None, pid)
            if key not in meta:
                row = group.iloc[0]
                sid = int(row["ID"]) if "ID" in row.index and row["ID"] is not None else -1
                meta[key] = {
                    "latitude": row.get("LATITUDE", np.nan),
                    "longitude": row.get("LONGITUDE", np.nan),
                    "time": row.get("DATEANDTIME", None),
                    "station_id": sid,
                    "profile_number": int(row["PROFILE_NUMBER"]) if "PROFILE_NUMBER" in row.index else -1,
                    "instrument_id": int(row["INSTRUMENT_ID"]) if "INSTRUMENT_ID" in row.index else -1,
                    "platform_id": _station_platform.get(sid, -1),
                }

    # Interpolate each parameter onto target depth levels
    data_vars = {}
    for param, df in dfs.items():
        interp = interpolate_depth_levels(df, depths, param_col=param, depth_col="LEV_M")
        if interp.empty:
            arr = np.full((n_profiles, n_depths), np.nan)
        else:
            # Build composite key column for pivot; auto-detected profile key matches all_profiles tuples
            if "ID" in interp.columns and "PROFILE_NUMBER" in interp.columns:
                interp = interp.assign(
                    _profile_key=list(zip(interp["ID"], interp["PROFILE_NUMBER"]))
                )
            else:
                interp = interp.assign(
                    _profile_key=[(None, pn) for pn in interp["PROFILE_NUMBER"]]
                )
            pivot = interp.pivot_table(
                index="_profile_key", columns="LEV_M",
                values=param, aggfunc="first",
            )
            pivot = pivot.reindex(index=all_profiles, columns=depths, fill_value=np.nan)
            arr = pivot.to_numpy(dtype=float)
        data_vars[param] = (["profile", "depth"], arr)

    # Build coordinate arrays from profile metadata
    lats = np.array([meta.get(p, {}).get("latitude", np.nan) for p in all_profiles])
    lons = np.array([meta.get(p, {}).get("longitude", np.nan) for p in all_profiles])
    station_ids = np.array([meta.get(p, {}).get("station_id", -1) for p in all_profiles])
    profile_numbers = np.array([meta.get(p, {}).get("profile_number", -1) for p in all_profiles])
    instrument_ids = np.array([meta.get(p, {}).get("instrument_id", -1) for p in all_profiles])
    platform_ids = np.array([meta.get(p, {}).get("platform_id", -1) for p in all_profiles])
    raw_times = [meta.get(p, {}).get("time") for p in all_profiles]
    try:
        times = pd.to_datetime(raw_times)
    except Exception:
        times = np.array(raw_times, dtype=object)

    # Resolve instrument IDs to names; fall back to stringified IDs
    instruments = np.array([
        _instrument_map.get(int(iid), str(int(iid))) for iid in instrument_ids
    ])

    # Resolve platform IDs to names; fall back to stringified IDs
    platforms = np.array([
        _platform_map.get(int(pid), str(int(pid))) for pid in platform_ids
    ])

    # Assemble xarray Dataset
    ds = xr.Dataset(
        data_vars,
        coords={
            "depth": depths,
            "profile_number": ("profile", profile_numbers),
            "station_id": ("profile", station_ids),
            "instrument": ("profile", instruments),
            "platform": ("profile", platforms),
            "latitude": ("profile", lats),
            "longitude": ("profile", lons),
            "time": ("profile", times),
        },
    )

    # Attach resolved unit names as variable attributes
    for param in data_vars:
        uid_set = _param_unit_ids.get(param, set())
        if len(uid_set) == 1:
            uid = next(iter(uid_set))
            ds[param].attrs["units"] = _units_map.get(int(uid), str(int(uid)))
        elif len(uid_set) > 1:
            ds[param].attrs["units"] = " / ".join(
                _units_map.get(int(u), str(int(u))) for u in sorted(uid_set)
            )

    # Climate and Forecast (CF) compliant metadata
    ds["depth"].attrs.update({"units": "m", "positive": "down"})
    ds["latitude"].attrs["units"] = "degrees_north"
    ds["longitude"].attrs["units"] = "degrees_east"
    return ds


def _read_with_geo(conn, param_name, quality_flags,
                   lat_min=None, lat_max=None, lon_min=None, lon_max=None,
                   depth_min=None, depth_max=None, date_min=None, date_max=None,
                   limit=None):
    """Read a parameter table with lat/lon/time, pushing all filters to SQL.

    Tries the E_* extended view first; falls back to an explicit P_* JOIN station.
    """
    from .database.information import does_table_exist

    # Validate identifier
    validate_identifier(param_name)

    params = []

    # Try E_* extended view first; fall back to P_* JOIN station
    view = f"E_{param_name.upper()}"
    if does_table_exist(conn, view, table_type="view"):
        # E_* view already contains lat/lon/time columns
        conditions = [f"{col}{cond}" for col, cond in (quality_flags or [])]
        _append_geo_conditions(
            conditions, params, "", "", "", "",
            lat_min, lat_max, lon_min, lon_max, depth_min, depth_max, date_min, date_max,
        )
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"SELECT * FROM {view} {where}"
    else:
        # Join P_* with station to get geo columns
        conditions = [f"p.{col}{cond}" for col, cond in (quality_flags or [])]
        _append_geo_conditions(
            conditions, params, "s.", "s.", "p.", "s.",
            lat_min, lat_max, lon_min, lon_max, depth_min, depth_max, date_min, date_max,
        )
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        query = (
            f"SELECT p.*, s.LATITUDE, s.LONGITUDE, s.DATEANDTIME "
            f"FROM P_{param_name.upper()} p JOIN station s ON p.ID = s.ID {where}"
        )

    # Add optional limit to query
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    # Execute query with parametrised geo/date values
    return pd.read_sql_query(query, conn, params=params)


def _append_geo_conditions(conditions, params, lat_pfx, lon_pfx, depth_pfx, date_pfx,
                           lat_min, lat_max, lon_min, lon_max,
                           depth_min, depth_max, date_min, date_max):
    """Append spatial/temporal SQL conditions and ? parameter values in-place."""
    if lat_min is not None:
        conditions.append(f"{lat_pfx}LATITUDE >= ?")
        params.append(float(lat_min))
    if lat_max is not None:
        conditions.append(f"{lat_pfx}LATITUDE <= ?")
        params.append(float(lat_max))
    if lon_min is not None:
        conditions.append(f"{lon_pfx}LONGITUDE >= ?")
        params.append(float(lon_min))
    if lon_max is not None:
        conditions.append(f"{lon_pfx}LONGITUDE <= ?")
        params.append(float(lon_max))
    if depth_min is not None:
        conditions.append(f"{depth_pfx}LEV_M >= ?")
        params.append(float(depth_min))
    if depth_max is not None:
        conditions.append(f"{depth_pfx}LEV_M <= ?")
        params.append(float(depth_max))
    if date_min is not None:
        conditions.append(f"{date_pfx}DATEANDTIME >= ?")
        params.append(pd.Timestamp(date_min).isoformat(sep=" ", timespec="seconds"))
    if date_max is not None:
        conditions.append(f"{date_pfx}DATEANDTIME <= ?")
        params.append(pd.Timestamp(date_max).isoformat(sep=" ", timespec="seconds"))
