"""Quality-flag presets, SQL clause builders and DataFrame QC filters."""
from __future__ import annotations

import pandas as pd
import logging
import operator as op
from collections import namedtuple

# Map of comparison operator strings to their Python equivalents
_OP_MAP = {
    ">=": op.ge,
    "<=": op.le,
    ">": op.gt,
    "<": op.lt,
    "==": op.eq,
    "!=": op.ne,
}
QCFilter = namedtuple("QCFilter", ["column", "condition"])

# Accept all records regardless of quality flags.
QC_ALL = []

# Accept records that passed basic quality control (PQF1 checked, PQF2 acceptable).
QC_GOOD = [QCFilter("PQF1", ">0"), QCFilter("PQF2", ">2")]


def build_where_clause(quality_flags: list[QCFilter] | list[tuple[str, str]] | None,
                       table_alias: str | None = None,
                       keyword: str = "WHERE") -> str:
    """Return a SQL clause for the given quality flags.

    Args:
        quality_flags: Quality flag criteria, e.g. ``QC_GOOD``.
        table_alias: Optional table alias to prefix column names.
        keyword: SQL keyword to lead with (``"WHERE"`` or ``"AND"``).
    Returns:
        Clause string, or ``""`` when *quality_flags* is empty.
    """
    if not quality_flags:
        return ""
    prefix = f"{table_alias}." if table_alias else ""
    return keyword + " " + " AND ".join(f"{prefix}{col}{cond}" for col, cond in quality_flags)


def apply_qc_flags(df: pd.DataFrame,
                   quality_flags: list[QCFilter] | list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Filter a DataFrame by quality flags.

    Same filtering as the SQL-level ``quality_flags`` on
    :func:`comfort.io.read_parameter`, but on an already-loaded DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame with QC flag columns (e.g. PQF1, PQF2, SQF).
        quality_flags (list[QCFilter] or list[tuple[str, str]]): Quality flag
            filters, e.g. :data:`QC_GOOD`. ``None`` or ``[]`` returns df unchanged.
    Returns:
        pandas.DataFrame: Filtered copy.
    """
    # Return plain df if no qc flags were provided
    if not quality_flags:
        return df.copy()

    # Start with all rows selected, then narrow down
    mask = pd.Series(True, index=df.index)
    for col, cond in quality_flags:

        # Check if column exists
        if col not in df.columns:
            logging.warning("apply_qc_flags: column %r not in DataFrame, skipped", col)
            continue

        # Parse the operator string and apply the respective comparison
        for sym, fn in _OP_MAP.items():
            if cond.startswith(sym):
                mask &= fn(df[col], float(cond[len(sym):]))
                break

    return df[mask].copy()
