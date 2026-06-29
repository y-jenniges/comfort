"""Unit conversion for measured values in the COMFORT database.

Adapted from Jenniges (2025), doi:10.5281/zenodo.15827777
"""
from __future__ import annotations

import logging
import sqlite3
import gsw
import numpy as np
import pandas as pd


class UnitsConverter:
    """Convert parameter units in the SQLite database to their default units."""

    def __init__(self, connection: sqlite3.Connection, default_units: pd.DataFrame,
                 units: pd.DataFrame, value_column: str = "VAL") -> None:
        """
        Args:
            connection (sqlite3.Connection): Database connection.
            default_units (pandas.DataFrame): Must have columns NAME_TABLE and UNITS_ID_DEFAULT.
            units (pandas.DataFrame): Must have column ID.
            value_column (str): Value column name. Default is ``'VAL'``.
        """
        self.connection = connection
        self.value_column = value_column
        self.df_default_units = default_units
        self.df_units = units.sort_values("ID")
        self.conversion_matrix = ConversionFormulas(df_units=self.df_units).conversion_matrix
        logging.info("Initialised UnitsConverter")

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection, value_column: str = "VAL") -> UnitsConverter:
        """Alternative constructor:
        Create a UnitsConverter by reading DATABASE_TABLES and UNITS automatically.

        Args:
            conn (sqlite3.Connection): COMFORT database connection.
            value_column (str): Value column name. Default is ``'VAL'``.
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
        return cls(conn, default_units, units, value_column)

    def convert_dataframe(self, df: pd.DataFrame, param_table_name: str,
                          use_density: bool = False) -> pd.DataFrame:
        """Convert a DataFrame's values to the default unit.

        Args:
            df (pandas.DataFrame): Must contain ``VAL`` and ``UNITS_ID``.
            param_table_name (str): Table name (e.g. ``'P_TEMPERATURE'``).
            use_density (bool): Use in-situ density for conversion.
        """
        if "UNITS_ID" not in df.columns or df.empty:
            return df

        # Check if default unit exists
        mask = self.df_default_units["NAME_TABLE"] == param_table_name
        if not mask.any():
            logging.warning("convert_dataframe: no default unit for %s", param_table_name)
            return df

        # Get target unit ID
        target = int(self.df_default_units.loc[mask, "UNITS_ID_DEFAULT"].iloc[0])

        # Get used unit IDs
        unit_ids = df["UNITS_ID"].dropna().unique()

        # Check if conversion is required
        if len(unit_ids) == 1 and int(unit_ids[0]) == target:
            return df

        # Iterate over unit IDs that need conversion
        parts = []
        for uid in unit_ids:
            # Filter for units ID and convert
            chunk = df[df["UNITS_ID"] == uid]
            parts.append(self.conversion_matrix[int(uid), target](chunk, param_table_name, use_density))
        if not parts:
            return df

        return pd.concat(parts).dropna(subset=[self.value_column])

    def convert_units(self, tables: list[str], use_density: bool = False,
                      override_old_tables: bool = False) -> dict[str, str]:
        """Convert units for the given parameter tables to their default unit.

        Args:
            tables (list[str]): Table names to convert.
            use_density (bool): Use in-situ density.
            override_old_tables (bool): Replace the original table.
        """
        # Iterate over tables
        new_tables = {}
        for table in tables:
            logging.info("UnitsConverter: converting %s", table)

            # Get data as df
            ex = self.connection.cursor().execute(f"SELECT * FROM {table};")
            df = pd.DataFrame(ex.fetchall(), columns=[x[0] for x in ex.description])
            old_len = len(df)

            # Convert units
            df_new = self.convert_dataframe(df, table, use_density)

            # Create a table in the DB for the converted data
            new_table_name = table if override_old_tables else f"converted_{table}"
            if_exists = "replace" if override_old_tables else "fail"

            if len(df_new) != old_len or not df_new["UNITS_ID"].equals(df["UNITS_ID"]):
                df_new.to_sql(new_table_name, self.connection, if_exists=if_exists, index=False)
                logging.info("  rows removed during conversion: %d", old_len - len(df_new))

            new_tables[table] = new_table_name
        return new_tables


class ConversionFormulas:
    """
    Conversion matrix [#units x #units] of callables to convert between COMFORT units.
    Based on COMFORT dataset v3 report, Appendix C.
    """

    def __init__(self, df_units):
        """
        Args:
            df_units (pandas.DataFrame): Unit table; must have column ID.
        """
        # Molar masses for mass-to-molar conversions (COMFORT v3, Appendix C)
        self.molar_masses = pd.DataFrame(
            {"NAME_TABLE": ["P_BARIUM", "P_NITRATE", "P_NITRATENITRITE", "P_NITRITE",
                            "P_PHOSPHATE", "P_SILICATE"],
             "ELEMENT": ["BARIUM", "NITROGEN", "NITROGEN", "NITROGEN", "PHOSPHORUS", "SILICON"],
             "MOLAR_MASS": [137.327, 14.00672, 14.00672, 14.00672, 30.973762, 28.085530]}
        )

        # Build conversion matrix: [source_unit_id, target_unit_id] -> callable
        n = df_units.shape[0] + 1
        self.conversion_matrix = np.full((n, n), self.no_conversion)
        np.fill_diagonal(self.conversion_matrix, self.identical_units)

        # Register known conversions
        self.conversion_matrix[5, 3] = self.milliEquivalentPerLiter_micromolPerKilogram
        self.conversion_matrix[4, 14] = self.microgramPerLiter_microgramPerKilogram
        self.conversion_matrix[7, 3] = self.millimolPerLiter_micromolPerKilogram
        self.conversion_matrix[12, 19] = self.nanomolPerKilogram_femtomolPerKilogram
        self.conversion_matrix[21, 3] = self.milliliterPerLiter_micromolPerKilogram
        self.conversion_matrix[14, 3] = self.microgramPerKilogram_micromolPerKilogram
        self.conversion_matrix[4, 3] = self.microgramPerLiter_micromolPerKilogram
        self.conversion_matrix[26, 3] = self.microgramAtomPerKilogram_micromolPerKilogram
        self.conversion_matrix[10, 3] = self.percent_micromolPerKilogram

    def get_molar_mass(self, param_table_name: str) -> float:
        """Return the molar mass for param_table_name."""
        mask = self.molar_masses["NAME_TABLE"] == param_table_name
        if not mask.any():
            raise ValueError(f"Molar mass not defined for {param_table_name}")
        return float(self.molar_masses.loc[mask, "MOLAR_MASS"].iloc[0])

    def identical_units(self, df: pd.DataFrame, param_name: str,
                        use_density: bool = False) -> pd.DataFrame:
        """Return df unchanged (source and target units are identical)."""
        logging.debug("No conversion necessary - units are identical")
        return df

    def no_conversion(self, df: pd.DataFrame, param_name: str,
                      use_density: bool = False) -> pd.DataFrame:
        """Return df unchanged (no conversion formula registered)."""
        logging.warning(f"Conversion not implemented for {param_name}")
        return df

    def milliEquivalentPerLiter_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                                    use_density: bool = False) -> pd.DataFrame:
        """Convert meq/L to umol/kg."""
        logging.debug("milliEquivalentPerLiter -> micromolPerKilogram")
        temp = df.copy()
        lab_dens = self.compute_lab_density(df, param_name, use_density)
        return temp.assign(VAL=temp["VAL"] * 1000 / lab_dens, UNITS_ID=3)

    def microgramPerLiter_microgramPerKilogram(self, df: pd.DataFrame, param_name: str,
                                               use_density: bool = False) -> pd.DataFrame:
        """Convert ug/L to ug/kg."""
        logging.debug("microgramPerLiter -> microgramPerKilogram")
        temp = df.copy()
        lab_dens = self.compute_lab_density(df, param_name, use_density)
        return temp.assign(VAL=temp["VAL"] / lab_dens, UNITS_ID=14)

    def millimolPerLiter_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                             use_density: bool = False) -> pd.DataFrame:
        """Convert mmol/L to umol/kg."""
        logging.debug("millimolPerLiter -> micromolPerKilogram")
        temp = df.copy()
        lab_dens = self.compute_lab_density(df, param_name, use_density)
        return temp.assign(VAL=temp["VAL"] * 1000 / lab_dens, UNITS_ID=3)

    def nanomolPerKilogram_femtomolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                               use_density: bool = False) -> pd.DataFrame:
        """Convert nmol/kg to fmol/kg."""
        logging.debug("nanomolPerKilogram -> femtomolPerKilogram")
        return df.copy().assign(VAL=df["VAL"] * 1_000_000, UNITS_ID=19)

    def milliliterPerLiter_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                               use_density: bool = False) -> pd.DataFrame:
        """Convert mL/L to umol/kg (oxygen-specific, factor 44.661)."""
        logging.debug("milliliterPerLiter -> micromolPerKilogram (oxygen-specific, factor 44.661)")
        temp = df.copy()
        lab_dens = self.compute_lab_density(df, param_name, use_density)
        return temp.assign(VAL=temp["VAL"] * 44.661 / lab_dens, UNITS_ID=3)

    def microgramPerKilogram_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                                 use_density: bool = False) -> pd.DataFrame:
        """Convert ug/kg to umol/kg using molar mass."""
        logging.debug("microgramPerKilogram -> micromolPerKilogram")
        temp = df.copy()
        molar_mass = self.get_molar_mass(param_name)
        return temp.assign(VAL=temp["VAL"] / molar_mass, UNITS_ID=3)

    def microgramPerLiter_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                              use_density: bool = False) -> pd.DataFrame:
        """Convert ug/L to umol/kg using molar mass and density."""
        logging.debug("microgramPerLiter -> micromolPerKilogram")
        temp = df.copy()
        lab_dens = self.compute_lab_density(df, param_name, use_density)
        molar_mass = self.get_molar_mass(param_name)
        return temp.assign(VAL=temp["VAL"] / (molar_mass * lab_dens), UNITS_ID=3)

    def microgramAtomPerKilogram_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                                     use_density: bool = False) -> pd.DataFrame:
        """Convert ugAtom/kg to umol/kg (1:1 equivalence)."""
        logging.debug("microgramAtomPerKilogram -> micromolPerKilogram")
        return df.copy().assign(UNITS_ID=3)

    def percent_micromolPerKilogram(self, df: pd.DataFrame, param_name: str,
                                    use_density: bool = False) -> pd.DataFrame:
        """Convert percent saturation to umol/kg using oxygen saturation formula."""
        logging.debug("percent -> micromolPerKilogram (oxygen saturation formula)")
        temp = df.copy()
        o2_sat = oxygen_saturation(df["salinity"], df["temperature"])
        return temp.assign(VAL=temp["VAL"] * o2_sat / 100 , UNITS_ID=3)

    def compute_lab_density(self, df: pd.DataFrame, param_name: str,
                            use_density: bool) -> float | pd.Series:
        """Return in-situ density [kg/l] per row or the fixed seawater approximation 1.025 kg/l."""
        if use_density:
            temp = df.copy()
            temp["salinity_absolute"] = gsw.SA_from_SP(
                temp["salinity"], temp["LEV_DBAR"] - 10.1325, temp["LONGITUDE"], temp["LATITUDE"]
            )
            density = gsw.density.rho_t_exact(
                temp["salinity_absolute"], temp["temperature"], temp["LEV_DBAR"] - 10.1325
            ) / 1000
            return density.fillna(1.025)
        return 1.025


# Standalone formula functions


def oxygen_saturation(salinity, temperature) -> np.ndarray | None:
    """Oxygen saturation from Benson & Krause (1984), eq. 31.

    Valid for 0 < T < 40 degC, 0 < S < 40.

    Args:
        salinity (double): Salinity [psu]
        temperature (double): Temperature [°C]

    Returns:
        oxygen_sat (double): Oxygen saturation
    """
    oxygen_sat = None
    if salinity is not None and temperature is not None:
        salinity = np.array(salinity)
        temperature = np.array(temperature)

        if isinstance(temperature, (int, float)):
            if temperature < 0 or temperature > 40:
                logging.warning("    units.oxygen_saturation: Temperature is out of valid range for this function "
                                "(0 < T < 40°C).")
        elif isinstance(temperature, (list, np.ndarray, pd.Series)):
            if (np.array(temperature) < 0).any() or (np.array(temperature) > 40).any():
                logging.warning("     units.oxygen_saturation: Temperature is out of valid range for this function "
                                "(0 < T < 40°C).")
        if isinstance(salinity, (int, float)):
            if salinity < 0 or salinity > 40:
                logging.warning("    units.oxygen_saturation: Salinity is out of valid range for this function "
                                "(0 < S < 40).")
            elif isinstance(salinity, (list, np.ndarray, pd.Series)):
                if (np.array(salinity) < 0).any() or (np.array(salinity) > 40).any():
                    logging.warning("    units.oxygen_saturation: Salinity is out of valid range for this function "
                                    "(0 < S < 40).")

        temperature_kelvin = temperature + 273.15
        oxygen_sat = np.exp(-135.29996 + 1.572288 * 10 ** 5 / temperature_kelvin
                            - 6.637149 * 10 ** 7 / temperature_kelvin ** 2
                            + 1.243678 * 10 ** 10 / temperature_kelvin ** 3
                            - 8.621061 * 10 ** 11 / temperature_kelvin ** 4
                            - salinity * (0.020573 - 12.142 / temperature_kelvin + 2363.1 / temperature_kelvin ** 2))
    else:
        logging.warning("    units.oxygen_saturation: Cannot compute oxygen saturation since temperature and/or "
                        "salinity value not given.")

    return oxygen_sat
