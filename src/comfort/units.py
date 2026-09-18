"""Unit conversion for measured values in the COMFORT database.

Adapted from Jenniges (2025), doi:10.5281/zenodo.15827777
"""
from __future__ import annotations

import logging
import sqlite3
import gsw
import numpy as np
import pandas as pd

from .physics import convert_salinity, convert_temperature
from .util.sqlite_utils import validate_identifier


def _get_column(df: pd.DataFrame, name: str) -> str | None:
    """Return the column name matching name case-insensitively (or None)."""
    for col in df.columns:
        if col.lower() == name.lower():
            return col
    return None


class UnitsConverter:
    """Convert parameter units in the SQLite database to their default units."""

    def __init__(self, connection: sqlite3.Connection, default_units: pd.DataFrame,
                 units: pd.DataFrame) -> None:
        """
        Args:
            connection (sqlite3.Connection): Database connection.
            default_units (pandas.DataFrame): Must have columns NAME_TABLE and UNITS_ID_DEFAULT.
            units (pandas.DataFrame): Must have column ID (and NAME_SHORT for unit names).
        """
        self.connection = connection
        self.df_default_units = default_units
        self.df_units = units
        self.formulas = ConversionFormulas()
        logging.info("Initialised UnitsConverter")

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection) -> UnitsConverter:
        """Alternative constructor:
        Create a UnitsConverter by reading DATABASE_TABLES and UNITS automatically.

        Args:
            conn (sqlite3.Connection): COMFORT database connection.
        Returns:
            UnitsConverter
        Raises:
            ValueError: If required tables are not found.
        """
        from .database.information import get_table_as_df, does_table_exist

        # Check table existence
        if not does_table_exist(conn, "DATABASE_TABLES"):
            raise ValueError("DATABASE_TABLES table not found in database")
        if not does_table_exist(conn, "UNITS"):
            raise ValueError("UNITS table not found in database")

        # Fetch tables
        default_units = get_table_as_df(conn, "DATABASE_TABLES")
        units = get_table_as_df(conn, "UNITS")
        return cls(conn, default_units, units)

    def convert_dataframe(self, df: pd.DataFrame, param_table_name: str,
                          use_density: bool = False, on_missing: str = "drop") -> pd.DataFrame:
        """Convert a DataFrame's values to the default unit.

        Rows that cannot be converted (no UNITS_ID, no registered conversion
        formula, or missing temperature/salinity inputs) are handled according
        to *on_missing*. Counts are always logged.

        Args:
            df (pandas.DataFrame): Must contain ``VAL`` and ``UNITS_ID``.
            param_table_name (str): Table name (e.g. ``'P_TEMPERATURE'``).
            use_density (bool): Use computed density instead of the
                constant 1.025 kg/L.
            on_missing (str): Policy for unconvertible rows: ``'drop'``
                (default) removes them, ``'keep'`` retains them with their
                original VAL and UNITS_ID, ``'raise'`` raises ValueError.
        """
        if on_missing not in ("drop", "keep", "raise"):
            raise ValueError(f"on_missing must be 'drop', 'keep' or 'raise', got {on_missing!r}")
        if "UNITS_ID" not in df.columns or df.empty:
            return df

        # Check if default unit exists
        target = self.default_unit_id(param_table_name)
        if target is None:
            logging.warning("convert_dataframe: no default unit for %s", param_table_name)
            return df

        # Get used unit IDs
        unit_ids = df["UNITS_ID"].dropna().unique()
        n_no_units_id = int(df["UNITS_ID"].isna().sum())

        # Check if conversion is required
        if n_no_units_id == 0 and len(unit_ids) == 1 and int(unit_ids[0]) == target:
            return df

        vc = "VAL"
        parts = []
        n_no_formula = 0
        n_missing_input = 0

        # Rows without a unit ID cannot be converted
        if n_no_units_id:
            if on_missing == "raise":
                raise ValueError(f"{param_table_name}: {n_no_units_id} rows have no UNITS_ID")
            if on_missing == "keep":
                parts.append(df[df["UNITS_ID"].isna()])

        # Iterate over unit IDs that need conversion
        for uid in unit_ids:
            chunk = df[df["UNITS_ID"] == uid]

            # Already in the target unit
            if int(uid) == target:
                parts.append(chunk)
                continue

            # Look up conversion formula
            formula = self.formulas.conversions.get((int(uid), target))
            if formula is None:
                logging.warning(f"Conversion from unit {int(uid)} to {target} not implemented "
                                f"for {param_table_name}")
                if on_missing == "raise":
                    raise ValueError(f"{param_table_name}: no conversion from unit "
                                     f"{int(uid)} to {target}")
                if on_missing == "keep":
                    parts.append(chunk)
                n_no_formula += len(chunk)
                continue

            converted = formula(chunk, param_table_name, use_density)

            # NaN from a non-NaN input marks a failed conversion (missing T/S)
            failed = converted[vc].isna() & chunk[vc].notna()
            if failed.any():
                if on_missing == "raise":
                    raise ValueError(f"{param_table_name}: {int(failed.sum())} rows could not be "
                                     f"converted from unit {int(uid)} "
                                     f"(missing temperature/salinity)")
                if on_missing == "keep":
                    converted.loc[failed] = chunk.loc[failed]
                else:
                    converted = converted[~failed]
                n_missing_input += int(failed.sum())
            parts.append(converted)

        # Report unconvertible rows
        action = "kept unconverted" if on_missing == "keep" else "dropped"
        if n_no_units_id:
            logging.warning("%s: %d rows without UNITS_ID %s", param_table_name, n_no_units_id, action)
        if n_no_formula:
            logging.warning("%s: %d rows with no conversion formula to unit %d %s",
                            param_table_name, n_no_formula, target, action)
        if n_missing_input:
            logging.warning("%s: %d rows missing temperature/salinity inputs %s",
                            param_table_name, n_missing_input, action)

        if not parts:
            return df.iloc[0:0]

        # Restore the original row order
        return pd.concat(parts).sort_index()

    def default_unit_id(self, param_table_name: str) -> int | None:
        """Return the ID of a parameter's default unit.

        Args:
            param_table_name (str): Table name (e.g. ``'P_OXYGEN'``).
        Returns:
            int or None: Unit ID, or ``None`` when no default unit is
                registered for *param_table_name*.
        """
        mask = self.df_default_units["NAME_TABLE"] == param_table_name
        if not mask.any():
            return None
        return int(self.df_default_units.loc[mask, "UNITS_ID_DEFAULT"].iloc[0])

    def default_unit_name(self, param_table_name: str) -> str | None:
        """Return the human-readable name of a parameter's default unit.

        Args:
            param_table_name (str): Table name (e.g. ``'P_OXYGEN'``).
        Returns:
            str or None: Unit name (e.g. ``'µmol/kg'``), or ``None`` when
                no default unit is registered for *param_table_name*.
        """
        # Get target unit ID
        target = self.default_unit_id(param_table_name)
        if target is None:
            return None

        # Get the short name of the unit ID
        row = self.df_units.loc[self.df_units["ID"] == target, "NAME_SHORT"]
        return row.iloc[0] if not row.empty else None

    def convert_units(self, tables: list[str], use_density: bool = False,
                      override_old_tables: bool = False,
                      on_missing: str = "drop") -> dict[str, str]:
        """Convert units for the given parameter tables to their default unit.

        Args:
            tables (list[str]): Table names to convert.
            use_density (bool): Use in-situ density.
            override_old_tables (bool): Replace the original table.
            on_missing (str): Policy for unconvertible rows
                (``'drop'``, ``'keep'`` or ``'raise'``). Default ``'drop'``.
        Returns:
            dict[str, str]: Maps each input table name to the table holding
                the converted data (the input name itself when nothing changed).
        """
        # Iterate over tables
        new_tables = {}
        for table in tables:
            validate_identifier(table)
            logging.info("UnitsConverter: converting %s", table)

            # Get data as df
            ex = self.connection.cursor().execute(f"SELECT * FROM {table};")
            df = pd.DataFrame(ex.fetchall(), columns=[x[0] for x in ex.description])
            old_len = len(df)

            # Convert units
            df_new = self.convert_dataframe(df, table, use_density, on_missing)

            # Write converted data only when something changed
            changed = ("UNITS_ID" in df_new.columns
                       and (len(df_new) != old_len or not df_new["UNITS_ID"].equals(df["UNITS_ID"])))
            if changed:
                new_table_name = table if override_old_tables else f"converted_{table}"
                if_exists = "replace" if override_old_tables else "fail"
                df_new.to_sql(new_table_name, self.connection, if_exists=if_exists, index=False)
                logging.info("  rows removed during conversion: %d", old_len - len(df_new))
                new_tables[table] = new_table_name
            else:
                new_tables[table] = table
        return new_tables


class ConversionFormulas:
    """
    Registry of callables to convert between COMFORT units, keyed by
    (source_unit_id, target_unit_id). Based on COMFORT dataset v3 report, Appendix C.

    With ``use_density=True`` the report's density conventions apply:
    ``lab_dens`` (T = 22 °C) for most volumetric conversions, ``real_dens``
    (in-situ T) for chlorophyll and oxygen mL/L (both TEOS-10 at atmospheric
    pressure). Otherwise, the constant 1.025 kg/L is used.
    % saturation -> µmol/kg (oxygen) is a comfort-db extension beyond the
    report based on Benson & Krause 1984.
    """

    def __init__(self):
        # Molar masses for mass-to-molar conversions (COMFORT v3, Appendix C)
        self.molar_masses = pd.DataFrame(
            {"NAME_TABLE": ["P_BARIUM", "P_NITRATE", "P_NITRATENITRITE", "P_NITRITE",
                            "P_PHOSPHATE", "P_SILICATE"],
             "ELEMENT": ["BARIUM", "NITROGEN", "NITROGEN", "NITROGEN", "PHOSPHORUS", "SILICON"],
             "MOLAR_MASS": [137.327, 14.00672, 14.00672, 14.00672, 30.973762, 28.085530]}
        )

        # Known conversions: (source_unit_id, target_unit_id) -> formula
        self.conversions = {
            (5, 3): self.milliEquivalentPerLiter_micromolPerKilogram,
            (4, 14): self.microgramPerLiter_microgramPerKilogram,
            (7, 3): self.millimolPerLiter_micromolPerKilogram,
            (12, 19): self.nanomolPerKilogram_femtomolPerKilogram,
            (21, 3): self.milliliterPerLiter_micromolPerKilogram,
            (14, 3): self.microgramPerKilogram_micromolPerKilogram,
            (4, 3): self.microgramPerLiter_micromolPerKilogram,
            (26, 3): self.microgramAtomPerKilogram_micromolPerKilogram,
            (10, 3): self.percent_micromolPerKilogram,
        }

    def get_molar_mass(self, param_table_name: str) -> float:
        """Return the molar mass for param_table_name."""
        mask = self.molar_masses["NAME_TABLE"] == param_table_name
        if not mask.any():
            raise ValueError(f"Molar mass not defined for {param_table_name}")
        return float(self.molar_masses.loc[mask, "MOLAR_MASS"].iloc[0])

    def milliEquivalentPerLiter_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                                    use_density: bool = False) -> pd.DataFrame:
        """Convert meq/L to umol/kg (App. C: lab_dens)."""
        logging.debug("milliEquivalentPerLiter -> micromolPerKilogram")
        temp = df.copy()
        lab_dens = self.lab_density(df, use_density)
        return temp.assign(VAL=temp["VAL"] * 1000 / lab_dens, UNITS_ID=3)

    def microgramPerLiter_microgramPerKilogram(self, df: pd.DataFrame, param_name: str,
                                               use_density: bool = False) -> pd.DataFrame:
        """Convert ug/L to ug/kg (App. C: real_dens for chlorophyll, lab_dens otherwise)."""
        logging.debug("microgramPerLiter -> microgramPerKilogram")
        temp = df.copy()
        if param_name.upper().endswith("CHLOROPHYLL"):
            dens = self.real_density(df, use_density)
        else:
            dens = self.lab_density(df, use_density)
        return temp.assign(VAL=temp["VAL"] / dens, UNITS_ID=14)

    def millimolPerLiter_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                             use_density: bool = False) -> pd.DataFrame:
        """Convert mmol/L to umol/kg (App. C: lab_dens)."""
        logging.debug("millimolPerLiter -> micromolPerKilogram")
        temp = df.copy()
        lab_dens = self.lab_density(df, use_density)
        return temp.assign(VAL=temp["VAL"] * 1000 / lab_dens, UNITS_ID=3)

    def nanomolPerKilogram_femtomolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                               use_density: bool = False) -> pd.DataFrame:
        """Convert nmol/kg to fmol/kg."""
        logging.debug("nanomolPerKilogram -> femtomolPerKilogram")
        return df.copy().assign(VAL=df["VAL"] * 1_000_000, UNITS_ID=19)

    def milliliterPerLiter_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                               use_density: bool = False) -> pd.DataFrame:
        """Convert mL/L to umol/kg (oxygen-specific, factor 44.661; App. C: real_dens)."""
        logging.debug("milliliterPerLiter -> micromolPerKilogram (oxygen-specific, factor 44.661)")
        temp = df.copy()
        real_dens = self.real_density(df, use_density)
        return temp.assign(VAL=temp["VAL"] * 44.661 / real_dens, UNITS_ID=3)

    def microgramPerKilogram_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                                 use_density: bool = False) -> pd.DataFrame:
        """Convert ug/kg to umol/kg using molar mass."""
        logging.debug("microgramPerKilogram -> micromolPerKilogram")
        temp = df.copy()
        molar_mass = self.get_molar_mass(param_name)
        return temp.assign(VAL=temp["VAL"] / molar_mass, UNITS_ID=3)

    def microgramPerLiter_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                              use_density: bool = False) -> pd.DataFrame:
        """Convert ug/L to umol/kg using molar mass (App. C: lab_dens)."""
        logging.debug("microgramPerLiter -> micromolPerKilogram")
        temp = df.copy()
        lab_dens = self.lab_density(df, use_density)
        molar_mass = self.get_molar_mass(param_name)
        return temp.assign(VAL=temp["VAL"] / (molar_mass * lab_dens), UNITS_ID=3)

    def microgramAtomPerKilogram_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                                     use_density: bool = False) -> pd.DataFrame:
        """Convert ugAtom/kg to umol/kg (1:1 equivalence)."""
        logging.debug("microgramAtomPerKilogram -> micromolPerKilogram")
        return df.copy().assign(UNITS_ID=3)

    def percent_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                    use_density: bool = False) -> pd.DataFrame:
        """Convert % saturation to umol/kg using TEOS-10 oxygen solubility.

        Requires co-located temperature and salinity columns (any casing).
        Potential temperature is computed from co-located pressure/latitude/longitude
        when available (see :meth:`_potential_temperature_for_o2sol`), otherwise falls
        back to in-situ temperature as an approximation.
        """
        logging.debug("percent -> micromolPerKilogram (gsw.O2sol_SP_pt)")
        temp = df.copy()
        sal_col = _get_column(df, "SALINITY")
        temp_col = _get_column(df, "TEMPERATURE")

        # Check if salinity and temperature columns are given
        if sal_col is None or temp_col is None:
            logging.warning("percent -> micromolPerKilogram: temperature/salinity columns "
                            "missing - cannot compute oxygen saturation")
            return temp.assign(VAL=np.nan, UNITS_ID=3)

        # Unit conversion
        pt = self._potential_temperature_for_o2sol(df, sal_col, temp_col)
        o2_eq = gsw.O2sol_SP_pt(df[sal_col].to_numpy(dtype=float), pt)
        return temp.assign(VAL=temp["VAL"] * np.asarray(o2_eq) / 100, UNITS_ID=3)

    def _potential_temperature_for_o2sol(self, df: pd.DataFrame, sal_col: str,
                                         temp_col: str) -> np.ndarray:
        """Helper function to compute potential temperature (pt0) for oxygen solubility.
        Falls back to in-situ temperature when pressure/latitude/longitude are unavailable.
        """
        pressure_col = _get_column(df, "LEV_DBAR") or _get_column(df, "LEV_M")
        lat_col = _get_column(df, "LATITUDE")
        lon_col = _get_column(df, "LONGITUDE")
        if pressure_col is None or lat_col is None or lon_col is None:
            logging.warning("percent -> micromolPerKilogram: pressure/latitude/longitude "
                            "missing, using in-situ temperature as an approximation to "
                            "potential temperature")
            return df[temp_col].to_numpy(dtype=float)

        sa = convert_salinity(df[sal_col], df[pressure_col], df[lon_col], df[lat_col])
        pt = convert_temperature(df[temp_col], sa.to_numpy(), df[pressure_col], to="pt0")
        return pt.to_numpy()

    def lab_density(self, df: pd.DataFrame, use_density: bool) -> float | pd.Series:
        """Density at sample salinity, T = 22 °C, atmospheric pressure [kg/L].

        ``lab_dens`` of COMFORT report Appendix C. Falls back to the fixed
        seawater approximation 1.025 kg/L when disabled or inputs are missing.
        """
        return self._density(df, use_density, lab_temperature=22.0)

    def real_density(self, df: pd.DataFrame, use_density: bool) -> float | pd.Series:
        """Density at sample salinity, in-situ temperature, atmospheric pressure [kg/L].

        ``real_dens`` of COMFORT report Appendix C. Falls back to the fixed
        seawater approximation 1.025 kg/L when disabled or inputs are missing.
        """
        return self._density(df, use_density, lab_temperature=None)

    def _density(self, df: pd.DataFrame, use_density: bool,
                 lab_temperature: float | None) -> float | pd.Series:
        """TEOS-10 density [kg/L] at sea pressure 0; column names resolved case-insensitively."""
        if not use_density:
            return 1.025

        # Get column names
        sal_col = _get_column(df, "SALINITY")
        lat_col = _get_column(df, "LATITUDE")
        lon_col = _get_column(df, "LONGITUDE")
        needed = [("salinity", sal_col), ("latitude", lat_col), ("longitude", lon_col)]
        temp_col = None
        if lab_temperature is None:
            temp_col = _get_column(df, "TEMPERATURE")
            needed.append(("temperature", temp_col))

        # Check if all needed columns are given
        missing = [name for name, col in needed if col is None]
        if missing:
            logging.warning("density: columns missing (%s) - "
                            "using constant density 1.025 kg/L", ", ".join(missing))
            return 1.025

        # Compute and return density
        temperature = lab_temperature if lab_temperature is not None else df[temp_col]
        salinity_absolute = gsw.SA_from_SP(df[sal_col], 0.0, df[lon_col], df[lat_col])
        rho = gsw.density.rho_t_exact(salinity_absolute, temperature, 0.0)
        return pd.Series(np.asarray(rho) / 1000, index=df.index).fillna(1.025)


# --- Standalone formula functions --------------------------------------------------------------- #


def oxygen_saturation(salinity, temperature) -> np.ndarray | None:
    """Oxygen saturation from Benson & Krause (1984), eq. 31.

    Valid for 0 < T < 40 degC, 0 < S < 40.

    Args:
        salinity (double): Salinity [psu]
        temperature (double): Temperature [°C]

    Returns:
        oxygen_sat (double): Oxygen saturation
    """
    # Warning if temperature or salinity are missing
    if salinity is None or temperature is None:
        logging.warning("    units.oxygen_saturation: Cannot compute oxygen saturation since temperature and/or "
                        "salinity value not given.")
        return None

    salinity = np.asarray(salinity, dtype=float)
    temperature = np.asarray(temperature, dtype=float)

    # Warn outside the validity range of the fit (NaN inputs compare False)
    if ((temperature < 0) | (temperature > 40)).any():
        logging.warning("    units.oxygen_saturation: Temperature is out of valid range for this function "
                        "(0 < T < 40°C).")
    if ((salinity < 0) | (salinity > 40)).any():
        logging.warning("    units.oxygen_saturation: Salinity is out of valid range for this function "
                        "(0 < S < 40).")

    # Compute oxygen saturation
    temperature_kelvin = temperature + 273.15
    return np.exp(-135.29996 + 1.572288 * 10 ** 5 / temperature_kelvin
                  - 6.637149 * 10 ** 7 / temperature_kelvin ** 2
                  + 1.243678 * 10 ** 10 / temperature_kelvin ** 3
                  - 8.621061 * 10 ** 11 / temperature_kelvin ** 4
                  - salinity * (0.020573 - 12.142 / temperature_kelvin + 2363.1 / temperature_kelvin ** 2))
