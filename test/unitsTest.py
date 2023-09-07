import unittest
import pandas as pd
import numpy as np
from src.preprocessing.units import UnitsConverter, ConversionFormulas, oxygen_saturation, atg, compute_theta
from src.database.information import get_table_as_df, does_table_exist
from src.database.communication import create_connection

units_test_table = "test_files.xlsx"
db_path = "C:/Users/yvjennig/PycharmProjects/data/comfort.sqlite"
db_path_preprocessed = "C:/Users/yvjennig/PycharmProjects/data/6_comfort_potT.sqlite"


class UnitsConverterTest(unittest.TestCase):
    def test_convert_units(self):
        # init converter
        default_units_id = 19
        connection = create_connection(db_path)
        default_units = pd.DataFrame({"NAME_TABLE": ["dummy"], "UNITS_ID_DEFAULT": [default_units_id]})
        units = pd.DataFrame({"ID": list(range(0, 27))})
        value_column = "VAL"
        uc = UnitsConverter(connection=connection, default_units=default_units, units=units, value_column=value_column)

        # create a dummy table
        q = "create table dummy (VAL real, UNITS_ID int, LATITUDE real, LONGITUDE real, LEV_DBAR real, " \
            "salinity real, temperature real);"
        connection.execute(q)
        q = "insert into dummy (VAL, UNITS_ID, LATITUDE, LONGITUDE, LEV_DBAR, salinity, temperature) values" \
            "(10, 12, 50, 50, 0, 31, 4), " \
            "(20, 19, -10, 30, 10, 35, 10)"
        connection.execute(q)
        connection.commit()

        # convert units in dummy table (create new table, do not replace old one)
        new_tables_f = uc.convert_units(["dummy"], use_density=False, override_old_tables=False)
        self.assertTrue(does_table_exist(connection, new_tables_f["dummy"]))  # check if new table was created
        q = f"select distinct UNITS_ID from {new_tables_f['dummy']};"
        ex = connection.execute(q)
        res = ex.fetchall()[0]
        self.assertTrue(len(res) == 1)  # check if there is only one unit left
        self.assertTrue(res[0]== default_units_id)  # check if the leftover unit is the default one

        # remove converted table
        q = "drop table converted_dummy;"
        connection.execute(q)

        # convert units in dummy table (replace old table)
        new_tables_t = uc.convert_units(["dummy"], use_density=False, override_old_tables=True)
        self.assertTrue("dummy" == new_tables_t["dummy"])  # check if new table == old table
        self.assertFalse(does_table_exist(connection, "converted_dummy"))  # check if no new table was created
        q = f"select distinct UNITS_ID from dummy;"
        ex = connection.execute(q)
        res = ex.fetchall()[0]
        self.assertTrue(len(res) == 1)  # check if there is only one unit left
        self.assertTrue(res[0] == default_units_id)  # check if the leftover unit is the default one

        # remove dummy table
        q = "drop table dummy;"
        connection.execute(q)
        connection.close()


class ConversionFormulasTest(unittest.TestCase):
    def testCalculation(self):
        connection = create_connection(db_path_preprocessed)
        formulas = ConversionFormulas(get_table_as_df(connection, "UNITS").sort_values("ID"))

        param_name = "test"
        use_density = False
        df = pd.DataFrame()
        self.assertTrue(df.equals(formulas.identical_units(df, param_name, use_density)))
        # self.assertEqual(formulas.identical_units(df, param_name, use_density), df)

    def _test_func(self, conversion_func, df_in, param_name, precision):
        print(f"unitsTest.ConversionFormulasTest.test_{conversion_func.__name__}: {param_name}")
        # compute conversions with and without density
        df_out_f = conversion_func(df_in, param_name, False)
        df_out_t = conversion_func(df_in, param_name, True)

        # evaluate results
        df_out_f["diff"] = abs(df_out_f["VAL"] - df_in["VAL_CONVERTED_NO_DENSITY"])
        df_out_t["diff"] = abs(df_out_t["VAL"] - df_in["VAL_CONVERTED_DENSITY"])

        is_successful = (df_out_f["diff"] <= precision).all() and (df_out_t["diff"] <= precision).all()
        if not is_successful:
            print(f"    Test failed!")
        print(f"\n    Differences for no_density: \n{df_out_f['diff']} "
              f"\n    Difference for density: \n{df_out_t['diff']}")
        self.assertTrue(is_successful)

    def test_identical_units(self):
        connection = create_connection(db_path_preprocessed)
        formulas = ConversionFormulas(get_table_as_df(connection, "UNITS").sort_values("ID"))
        df_in = pd.DataFrame({"VAL": [None, 1, 2, 3]})
        df_out = formulas.identical_units(df_in, "NITRATE")
        connection.close()
        self.assertTrue(df_in.equals(df_out))

    def test_no_conversion(self):
        connection = create_connection(db_path_preprocessed)
        formulas = ConversionFormulas(get_table_as_df(connection, "UNITS").sort_values("ID"))
        df_in = pd.DataFrame({"VAL": [None, 1, 2, 3]})
        df_out = formulas.no_conversion(df_in, "NITRATE")
        connection.close()
        self.assertTrue(df_in.equals(df_out))

    def test_milliEquivalentPerLiter_micromolPerKilogram(self, precision=128):
        connection = create_connection(db_path_preprocessed)
        formulas = ConversionFormulas(get_table_as_df(connection, "UNITS").sort_values("ID"))

        # test cases downloaded from OceanShell application as described in the report of the COMFORT dataset v3
        param_name = "ALKALINITY"
        df_in = pd.read_excel(units_test_table)
        df_in = df_in[df_in["param_name"] == param_name]
        # df_in["LEV_DBAR"] = [10.1325, 10.1325, 10.1325]
        # df_in["temperature"] = [22, 22, 22]
        self._test_func(conversion_func=formulas.milliEquivalentPerLiter_micromolPerKilogram,
                        df_in=df_in, param_name=param_name, precision=precision)

    def test_microgramPerLiter_microgramPerKilogram(self, precision=0.5):
        connection = create_connection(db_path_preprocessed)
        formulas = ConversionFormulas(get_table_as_df(connection, "UNITS").sort_values("ID"))

        # test cases downloaded from OceanShell application as described in the report of the COMFORT dataset v3
        param_name = "CHLOROPHYLL"
        df_in = pd.read_excel(units_test_table)
        df_in = df_in[df_in["param_name"] == param_name]
        # df_in["LEV_DBAR"] = [10.1325, 10.1325, 10.1325]
        self._test_func(conversion_func=formulas.microgramPerLiter_microgramPerKilogram,
                        df_in=df_in, param_name=param_name, precision=precision)

        param_name = "AMMONIUM"
        df_in = pd.read_excel(units_test_table)
        df_in = df_in[df_in["param_name"] == param_name]
        # df_in["LEV_DBAR"] = [10.1325, 10.1325, 10.1325]
        # df_in["temperature"] = [22, 22, 22]
        self._test_func(conversion_func=formulas.microgramPerLiter_microgramPerKilogram,
                        df_in=df_in, param_name=param_name, precision=precision)

        param_name = "DIN"
        df_in = pd.read_excel(units_test_table)
        df_in = df_in[df_in["param_name"] == param_name]
        # df_in["LEV_DBAR"] = [10.1325, 10.1325, 10.1325]
        # df_in["temperature"] = [22, 22, 22]
        self._test_func(conversion_func=formulas.microgramPerLiter_microgramPerKilogram,
                        df_in=df_in, param_name=param_name, precision=precision)

    def test_millimolPerLiter_micromolPerKilogram(self, precision=0.5):
        connection = create_connection(db_path_preprocessed)
        formulas = ConversionFormulas(get_table_as_df(connection, "UNITS").sort_values("ID"))

        # test cases downloaded from OceanShell application as described in the report of the COMFORT dataset v3
        param_name = "DIC"
        df_in = pd.read_excel(units_test_table)
        df_in = df_in[df_in["param_name"] == param_name]
        # df_in["LEV_DBAR"] = [10.1325, 10.1325, 10.1325]
        # df_in["temperature"] = [22, 22, 22]

        self._test_func(conversion_func=formulas.millimolPerLiter_micromolPerKilogram,
                        df_in=df_in, param_name=param_name, precision=precision)

    def test_nanomolPerKilogram_femtomolPerKilogram(self, precision=0.005):
        connection = create_connection(db_path_preprocessed)
        formulas = ConversionFormulas(get_table_as_df(connection, "UNITS").sort_values("ID"))

        param_name = "SF6"
        df_in = pd.read_excel(units_test_table)
        df_in = df_in[(df_in["param_name"] == param_name) & (df_in["UNITS_ID_in"] == 12)]

        self._test_func(conversion_func=formulas.nanomolPerKilogram_femtomolPerKilogram,
                        df_in=df_in, param_name=param_name, precision=precision)

    def test_milliliterPerLiter_micromolPerKilogram(self, precision=0.05):
        connection = create_connection(db_path_preprocessed)
        formulas = ConversionFormulas(get_table_as_df(connection, "UNITS").sort_values("ID"))

        param_name = "OXYGEN"
        df_in = pd.read_excel(units_test_table)
        df_in = df_in[(df_in["param_name"] == param_name) & (df_in["UNITS_ID_in"] == 21)]
        # df_in["LEV_DBAR"] = [10.1325, 10.1325, 10.1325]

        self._test_func(conversion_func=formulas.milliliterPerLiter_micromolPerKilogram,
                        df_in=df_in, param_name=param_name, precision=precision)

    def test_microgramPerKilogram_micromolPerKilogram(self):
        pass

    def test_microgramPerLiter_micromolPerKilogram(self):
        pass

    def test_microgramAtomPerKilogram_micromolPerKilogram(self):
        pass

    def test_percent_micromolPerKilogram(self, precision=0.5):
        connection = create_connection(db_path_preprocessed)
        formulas = ConversionFormulas(get_table_as_df(connection, "UNITS").sort_values("ID"))

        param_name = "OXYGEN"
        df_in = pd.read_excel(units_test_table)
        df_in = df_in[(df_in["param_name"] == param_name) & (df_in["UNITS_ID_in"] == 10)]
        # df_in["LEV_DBAR"] = [10.1325, 10.1325, 10.1325]

        self._test_func(conversion_func=formulas.percent_micromolPerKilogram,
                        df_in=df_in, param_name=param_name, precision=precision)

    def test_compute_lab_density(self):
        pass


class FreeFunctionsTest(unittest.TestCase):
    def test_oxygen_saturation(self, precision=0.05):
        # test cases
        df_ox_sat = pd.DataFrame({"salinity": [None, None, 0, 0, 30, 35, 40, 40],
                                  "temperature": [None, 0, None, 0, 10, 20, 30, 40],
                                  "oxygen_saturation": [None, None, None, 457, 284.64, 225.54, 184.88, 159.53]})

        # compute oxygen saturation using the implemented formula
        df_ox_sat["computed"] = oxygen_saturation(df_ox_sat["salinity"], df_ox_sat["temperature"])
        df_ox_sat["is_equal"] = abs(df_ox_sat["oxygen_saturation"] - df_ox_sat["computed"]) <= precision

        # make nan==nan -> True (in pandas it is usually False)
        mask = pd.isnull(df_ox_sat["oxygen_saturation"]) & pd.isnull(df_ox_sat["computed"])
        df_ox_sat.loc[mask, "is_equal"] = True

        # evaluate results
        is_successful = df_ox_sat["is_equal"].all()
        if not is_successful:
            print("unitsTest.test_oxygen_saturation: Test failed!")
        self.assertTrue(is_successful)

    def test_atg(self, precision=0.0000005):
        # test cases (from R library Marelac)
        df_atg = pd.DataFrame({"salinity": [None, None, 0, 0, 25, 25, 25, 25, 40, 40],
                               "temperature": [None, 0, None, 0, 10, 10, 10, 30, 30, 0],
                               "pressure": [None, 0, 0, None, 0, 100, 1000, 0, 0, 100],
                               "atg": [None, None, None, None, 0.0001002, 0.0001135, 0.0002069, 0.0002417,
                                       0.0002510, 0.000063]})

        # compute ATG with the implemented formula
        df_atg["computed"] = atg(np.array(df_atg["salinity"]),
                                 np.array(df_atg["temperature"]),
                                 np.array(df_atg["pressure"]))
        df_atg["is_equal"] = abs(df_atg["atg"] - df_atg["computed"]) <= precision

        # make nan==nan -> True (in pandas it is usually False)
        mask = pd.isnull(df_atg["atg"]) & pd.isnull(df_atg["computed"])
        df_atg.loc[mask, "is_equal"] = True

        # evaluate results
        is_successful = df_atg["is_equal"].all()
        if not is_successful:
            print("unitsTest.test_atg: Test failed!")
        self.assertTrue(is_successful)

    def test_compute_theta(self, precision=0.0005):
        # test cases (from R library Marelac)
        df_pot_t = pd.DataFrame({"salinity": [None, 25, 25, 25, 25, 40],
                                 "temperature": [None, 40, 40, 10, 0, 40],
                                 "pressure": [None, 0, 100, 1000, 100, 1000],
                                 "reference_pressure": [None, 0, 0, 0, 0, 0],
                                 "pot_t": [None, 40, 36.6921, 8.4684, -0.0265, 36.89073]})

        # compute potential temperature with the implemented formula
        df_pot_t["computed"] = compute_theta(df_pot_t["salinity"],
                                             df_pot_t["temperature"],
                                             df_pot_t["pressure"],
                                             df_pot_t["reference_pressure"])
        df_pot_t["is_equal"] = abs(df_pot_t["pot_t"] - df_pot_t["computed"]) <= precision

        # make nan==nan -> True (in pandas it is usually False)
        mask = pd.isnull(df_pot_t["pot_t"]) & pd.isnull(df_pot_t["computed"])
        df_pot_t.loc[mask, "is_equal"] = True

        # evaluate results
        is_successful = df_pot_t["is_equal"].all()
        if not is_successful:
            print("unitsTest.test_compute_theta: Test failed!")
        self.assertTrue(is_successful)


if __name__ == "__main__":
    unittest.main()
