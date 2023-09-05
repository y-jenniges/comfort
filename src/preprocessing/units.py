""" Module to deal with the units of the measured values. """
import logging
import numpy as np
import pandas as pd
import gsw


class UnitsConverter:
    """ Converts units of parameters in the sqlite database to default units. """

    def __init__(self, connection, default_units, units, value_column="VAL"):
        """ Init the converter.

        Args:
            default_units (pandas.DataFrame): Table containing default units for every parameter. Required columns are
                                              NAME_TABLE, UNITS_ID_DEFAULT. For COMFORT, pass the table DATABASE_TABLES.
            units (pandas.DataFrame):   Table containing all available units. Required column is ID. For COMFORT, pass
                                        the table UNITS.
        """
        self.connection = connection
        self.value_column = value_column
        self.df_default_units = default_units
        self.df_units = units.sort_values("ID")

        self.conversion_matrix = ConversionFormulas(df_units=self.df_units).conversion_matrix
        print("Initialized units converter")

    def convert_units(self, tables, use_density=False, override_old_tables=False):
        new_tables = {}
        cur = self.connection.cursor()

        for table in tables:
            print(f"UnitsConverter: Converting {table}")
            # get target unit id
            target_unit_id = int(
                self.df_default_units[self.df_default_units["NAME_TABLE"] == table]["UNITS_ID_DEFAULT"])

            # get all distinct unit ids
            query = f"select distinct units_id from {table};"
            res = cur.execute(query).fetchall()
            unit_ids = [x[0] for x in res]

            # define the new table name
            new_table_name = f"converted_{table}"

            # if there is only one unit and it is already the default one, skip conversion
            if len(unit_ids) != 1 or unit_ids[0] != target_unit_id:
                print("    UnitsConverter: One or more units need to be converted. ")
                # load table
                q = f"select * from {table};"
                cur = self.connection.cursor()
                ex = cur.execute(q)
                df = pd.DataFrame(ex.fetchall(), columns=[x[0] for x in cur.description])
                old_len = len(df)

                # convert units
                df_new = pd.DataFrame(columns=df.columns)
                for unit in unit_ids:
                    df_converted = self.conversion_matrix[unit, target_unit_id](df[df["UNITS_ID"] == unit],
                                                                                table, use_density)
                    df_new = pd.concat([df_new, df_converted])

                # drop rows that have VAL==NULL (i.e. value could not be converted)
                df_new = df_new.dropna(subset=["VAL"])

                # store the new table
                if_exists = "fail"
                if override_old_tables:
                    if_exists = "replace"
                    new_table_name = table

                df_new.to_sql(new_table_name, self.connection, if_exists=if_exists, index=False, chunksize=None)
                new_len = len(df_new)
                print(f"    UnitsConverter: 0-Check: len(old_table) - len(new_table) = {old_len - new_len}")

            # save table name
            new_tables[table] = new_table_name

        return new_tables


class ConversionFormulas:
    """ Defines a matrix [#units]x[#units] containing formulas to convert a unit to another. Based on the report on
    the COMFORT dataset v3 appendix C."""

    def __init__(self, df_units):
        """ Init the conversion formulas.

            Args:
                df_units (pandas.DataFrame): Table containing all possible units.
                as SQL statement. Default is 'dataframe' (computations are executed on a DataFrame).
        """
        self.molar_masses = pd.DataFrame(
            {"NAME_TABLE": ["P_BARIUM", "P_NITRATE", "P_NITRATENITRITE", "P_NITRITE", "P_PHOSPHATE", "P_SILICATE"],
             "ELEMENT": ["BARIUM", "NITROGEN", "NITROGEN", "NITROGEN", "PHOSPHORUS", "SILICON"],
             "MOLAR_MASS": [137.327, 14.00672, 14.00672, 14.00672, 30.973762, 28.085530]})

        self.conversion_matrix = np.full((df_units.shape[0] + 1, df_units.shape[0] + 1), self.no_conversion)
        np.fill_diagonal(self.conversion_matrix, self.identical_units)

        self.conversion_matrix[5, 3] = self.milliEquivalentPerLiter_micromolPerKilogram  # from unit ID 5 to unit ID 3
        self.conversion_matrix[4, 14] = self.microgramPerLiter_microgramPerKilogram
        self.conversion_matrix[7, 3] = self.millimolPerLiter_micromolPerKilogram
        self.conversion_matrix[12, 19] = self.nanomolPerKilogram_femtomolPerKilogram
        self.conversion_matrix[21, 3] = self.milliliterPerLiter_micromolPerKilogram
        self.conversion_matrix[14, 3] = self.microgramPerKilogram_micromolPerKilogram
        self.conversion_matrix[4, 3] = self.microgramPerLiter_micromolPerKilogram
        self.conversion_matrix[26, 3] = self.microgramAtomPerKilogram_micromolPerKilogram
        self.conversion_matrix[10, 3] = self.percent_micromolPerKilogram

    def get_molar_mass(self, param_table_name):
        """ Returns molar mass of a given parameter if listed in the molar mass table. """
        if param_table_name in list(self.molar_masses["NAME_TABLE"]):
            return float(self.molar_masses[self.molar_masses["NAME_TABLE"] == param_table_name]["MOLAR_MASS"])
        else:
            print(f"Error: Molar mass is not defined for {param_table_name}.")
            return

    # conversion functions
    def identical_units(self, df, param_name, use_density=False):
        print("    No conversion necessary since units are identical.")
        return df

    def no_conversion(self, df, param_name, use_density=False):
        print(f"    Warning: Conversion is not possible or not implemented yet!")
        return df

    def milliEquivalentPerLiter_micromolPerKilogram(self, df, param_name, use_density=False):
        print("    milliEquivalentPerLiter -> micromolPerKilogram")
        temp = df.copy()
        lab_dens = self.compute_lab_density(df, param_name, use_density)
        temp = temp.assign(VAL=temp["VAL"] * 1000 / lab_dens, UNITS_ID=3)
        return temp

    def microgramPerLiter_microgramPerKilogram(self, df, param_name, use_density=False):
        print("    microgramPerLiter -> microgramPerKilogram")
        temp = df.copy()
        lab_dens = self.compute_lab_density(df, param_name, use_density)
        temp = temp.assign(VAL=temp["VAL"] / lab_dens, UNITS_ID=14)
        return temp

    def millimolPerLiter_micromolPerKilogram(self, df, param_name, use_density=False):
        print("    millimolPerLiter -> micromolPerKilogram")
        temp = df.copy()
        lab_dens = self.compute_lab_density(df, param_name, use_density)
        temp = temp.assign(VAL=temp["VAL"] * 1000 / lab_dens, UNITS_ID=3)
        return temp

    def nanomolPerKilogram_femtomolPerKilogram(self, df, param_name, use_density=False):
        print("    nanomolPerKilogram -> femtomolPerKilogram")
        temp = df.copy()
        temp = temp.assign(VAL=temp["VAL"] * 1000000, UNITS_ID=19)
        return temp

    def milliliterPerLiter_micromolPerKilogram(self, df, param_name, use_density=False):
        print("    milliliterPerLiter -> micromolPerKilogram")
        temp = df.copy()
        lab_dens = self.compute_lab_density(df, param_name, use_density)
        x = 44.661
        print("    Use this conversion with caution, it is only valid for oxygen.")
        temp = temp.assign(VAL=temp["VAL"] * x / lab_dens, UNITS_ID=3)
        return temp

    def microgramPerKilogram_micromolPerKilogram(self, df, param_name, use_density=False):
        # molar mass in g/mol
        print("    microgramPerKilogram -> micromolPerKilogram")
        temp = df.copy()
        molar_mass = self.get_molar_mass(param_name)
        temp = temp.assign(VAL=temp["VAL"] / molar_mass, UNITS_ID=3)
        return temp

    def microgramPerLiter_micromolPerKilogram(self, df, param_name, use_density=False):
        print("    microgramPerLiter -> micromolPerKilogram")
        temp = df.copy()
        lab_dens = self.compute_lab_density(df, param_name, use_density)
        molar_mass = self.get_molar_mass(param_name)
        temp = temp.assign(VAL=temp["VAL"] / (molar_mass * lab_dens), UNITS_ID=3)
        return temp

    def microgramAtomPerKilogram_micromolPerKilogram(self, df, param_name, use_density=False):
        print("    microgramAtomPerKilogram -> micromolPerKilogram")
        temp = df.copy()
        temp = temp.assign(UNITS_ID=3)
        return temp

    def percent_micromolPerKilogram(self, df, param_name, use_density=False):
        print("    Percent -> micromolPerKilogram: Treat with caution. Formula for oxygen saturation is only valid for "
              "0<T<40^C and 0<S<40.")
        temp = df.copy()
        oxygen_concentration = temp["VAL"] * oxygen_saturation(df["salinity"], df["temperature"]) * 100
        temp = temp.assign(VAL=oxygen_concentration, UNITS_ID=3)
        return temp

    def compute_lab_density(self, df, param_name, use_density):
        """ Computes density of a parameter or returns a constant factor to convert liters to kilograms in [kg/l]. """
        temp = df.copy()
        if use_density:
            print("    Computing density...")
            # compute absolute salinity from practical salinity
            temp["salinity_absolute"] = gsw.SA_from_SP(temp["salinity"], temp["LEV_DBAR"] - 10.1325,
                                                       temp["LONGITUDE"], temp["LATITUDE"])

            # compute in-situ density (if it could not be computed due to missing temp/sal, use fixed factor)
            # function returns [kg/m^3], we divide by 1000 to get [kg/l]
            temp["density"] = gsw.density.rho_t_exact(temp["salinity_absolute"], temp["temperature"],
                                                      temp["LEV_DBAR"] - 10.1325)/1000
            temp["density"].fillna(1.025, inplace=True)

            return temp["density"]
        else:
            return 1.025


# --- Additional functions --- #
def oxygen_saturation(salinity, temperature):
    """
    Oxygen saturation formula from Benson & Krause (Limnology & Oceanography, 29, 620-632, 1984), equation 31. Usable
    for 0 < T < 40°C and 0 < S < 40.
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
                logging.warning("units.oxygen_saturation: Temperature is out of valid range for this function "
                                "(0 < T < 40°C).")
        elif isinstance(temperature, (list, np.ndarray, pd.Series)):
            if (np.array(temperature) < 0).any() or (np.array(temperature) > 40).any():
                logging.warning("units.oxygen_saturation: Temperature is out of valid range for this function "
                                "(0 < T < 40°C).")
        if isinstance(salinity, (int, float)):
            if salinity < 0 or salinity > 40:
                logging.warning("units.oxygen_saturation: Salinity is out of valid range for this function "
                                "(0 < S < 40).")
            elif isinstance(salinity, (list, np.ndarray, pd.Series)):
                if (np.array(salinity) < 0).any() or (np.array(salinity) > 40).any():
                    logging.warning("units.oxygen_saturation: Salinity is out of valid range for this function "
                                    "(0 < S < 40).")

        temperature_kelvin = temperature + 273.15
        oxygen_sat = np.exp(-135.29996 + 1.572288 * 10 ** 5 / temperature_kelvin
                            - 6.637149 * 10 ** 7 / temperature_kelvin ** 2
                            + 1.243678 * 10 ** 10 / temperature_kelvin ** 3
                            - 8.621061 * 10 ** 11 / temperature_kelvin ** 4
                            - salinity * (0.020573 - 12.142 / temperature_kelvin + 2363.1 / temperature_kelvin ** 2))

    return oxygen_sat


def atg(salinity, temperature, pressure):
    """
    Adiabatic temperature gradient [°C/dbar].
    See also https://www.rdocumentation.org/packages/marelac/versions/2.1.10/topics/sw_adtgrad.

    Args:
        salinity (double or numpy array): Salinity [psu]
        temperature (double or numpy array): Temperature [°C]
        pressure(double or numpy array): Pressure [bar]

    Returns:
        atg (double or numpy array): Adiabatic temperature gradient
    """
    # constants
    a0 = 0.000035803
    a1 = 0.0000085258
    a2 = -0.00000006836
    a3 = 0.00000000066228
    b0 = 0.0000018932
    b1 = -0.000000042393
    c0 = 0.000000018741
    c1 = -0.00000000067795
    c2 = 0.000000000008733
    c3 = -5.4481E-14
    d0 = -0.00000000011351
    d1 = 2.7759E-12
    e0 = -4.6206E-13
    e1 = 1.8676E-14
    e2 = -2.1678E-16

    # computations
    p = pressure * 10
    atg = a0 + a1 * temperature + a2 * temperature ** 2 + a3 * temperature ** 3 + (b0 + b1 * temperature) * (
                salinity - 35)
    atg = atg + (c0 + c1 * temperature + c2 * temperature ** 2 + c3 * temperature ** 3 + (d0 + d1 * temperature) * (
                salinity - 35)) * p
    atg = atg + (e0 + e1 * temperature + e2 * temperature ** 2) * p ** 2

    return atg


def compute_theta(salinity, temperature, pressure, reference_pressure=0):
    """
    Potential temperature referred to a given pressure.
    See also https://www.rdocumentation.org/packages/marelac/versions/2.1.10/topics/sw_tpot
    Args:
        salinity (double or pandas.Series): Salinity [psu]
        temperature (double or pandas.Series): In situ temperature [°C], ITS-68
        pressure (double or pandas Series): In situ pressure [bar]
        reference_pressure (double or pandas Series): Reference pressure [bar]. Default is 0.

    Returns:
        Theta (double or pandas Series): Potential temperature [°C]
    """
    # for the computations, convert to numpy arrays
    # (pandas can mess up with the indexes in multiplications and produce NaN values)
    pressure = pressure.to_numpy()
    salinity = salinity.to_numpy()
    temperature = temperature.to_numpy()

    # computations
    dp = (reference_pressure - pressure) * 10
    theta = dp * atg(salinity, temperature, pressure)
    tt1 = temperature + 0.5 * theta

    q1 = theta
    theta = dp * atg(salinity, tt1, (pressure + 0.05 * dp))
    tt2 = tt1 + (1 - 1 / np.sqrt(2)) * (theta - q1)

    q2 = (2 - np.sqrt(2)) * theta + (3 / np.sqrt(2) - 2) * q1
    theta = dp * atg(salinity, tt2, (pressure + 0.05 * dp))
    tt1 = tt2 + (1 + 1 / np.sqrt(2)) * (theta - q2)

    q1 = (2 + np.sqrt(2)) * theta - (2 + 3 / np.sqrt(2)) * q2
    theta = dp * atg(salinity, tt1, (pressure + dp / 10))
    theta = tt1 + 1 / 6 * (theta - 2 * q1)

    return pd.Series(theta)
