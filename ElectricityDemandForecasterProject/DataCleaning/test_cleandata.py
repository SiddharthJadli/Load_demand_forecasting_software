from unittest import mock
from cleaning_data import DataCleaning 
import pandas as pd
import unittest

class TestDataCleaning(unittest.TestCase):

    def setUp(self):
        self.dc = DataCleaning()

    def to_set(self,lst):
        output = set()
        for item in lst:
            output.add(item)
        return output
    
    def test_readfile(self):
        file_csv = self.dc.read_file("CleaningData_TestFiles/actuals_1.csv")
        file_xlsx = self.dc.read_file("CleaningData_TestFiles/actuals_1.xlsx")
        file_txt = self.dc.read_file("CleaningData_TestFiles/testing_txt.txt")

        self.assertEqual(type(file_xlsx),pd.DataFrame)
        self.assertEqual(type(file_csv),pd.DataFrame)
        self.assertEqual(file_txt,'Insert a file type of .xlsx or .csv')

    def test_rename_columns(self):

        forecast_cols = {"datetime", "temperature_c", "cloud_cover", "wind_direction", "wind_speed_kmh", "pressure_kpa",
                        "date", "month", "type_of_day", "hour"}
        actual_cols = {"datetime", "temperature_c", "cloud_cover", "wind_direction", "wind_speed_kmh","pressure_kpa", "load_kw",
                    "humidity","date", "month", "type_of_day", "hour"}

        #same column names as the original actuals training dataset
        file_1 = self.dc.read_file("CleaningData_TestFiles/actuals_1.csv")
        file_1 = self.dc.rename_columns(file_1)
        ac_colset_1 = self.to_set(file_1.columns.to_list())

        #different column names for actuals (includes humidity and load)
        file_2 = self.dc.read_file("CleaningData_TestFiles/actuals_3.csv")
        file_2 = self.dc.rename_columns(file_2)
        ac_colset_2 = self.to_set(file_2.columns.to_list())

        #same column names as the original forecasts training dataset
        file_3 = self.dc.read_file("CleaningData_TestFiles/forecast_2.csv")
        file_3 = self.dc.rename_columns(file_3)
        fc_colset_1 = self.to_set(file_3.columns.to_list())

        #different column names for actuals (includes humidity and load)
        file_4 = self.dc.read_file("CleaningData_TestFiles/forecast_3.csv")
        file_4 = self.dc.rename_columns(file_4)
        fc_colset_2 = self.to_set(file_4.columns.to_list())

        self.assertEqual(ac_colset_1, actual_cols)
        self.assertEqual(ac_colset_2, actual_cols)
        self.assertEqual(fc_colset_1,forecast_cols)
        self.assertEqual(fc_colset_2,forecast_cols)


    def test_combine_cols(self):
        cloudperc = self.dc.read_file("CleaningData_TestFiles/cloudperc.xlsx")
        pressure = self.dc.read_file("CleaningData_TestFiles/Pressure_kpa.xlsx")
        temp = self.dc.read_file("CleaningData_TestFiles/Temperature_Celcius.xlsx")
        winddir = self.dc.read_file("CleaningData_TestFiles/Winddirection_degree.xlsx")
        windsp = self.dc.read_file("CleaningData_TestFiles/Windspeed_kmh.xlsx")

        file = self.dc.combine_columns(pressure, temp, winddir, windsp, cloudperc)
        

        #correct number of columns
        self.assertEqual(file.shape[1],10)

        #consists of all the columns
        self.assertEqual(file.columns.to_list(), ['pressure_kpa','datetime','temperature_c','wind_direction','wind_speed_kmh','cloud_cover',
                                                    "date", "month", "type_of_day", "hour"])


    def test_merge_files(self):
        actuals_1 = self.dc.read_file("CleaningData_TestFiles/actuals_1.csv")
        actuals_2 = self.dc.read_file("CleaningData_TestFiles/actuals_2.csv")

        cloudperc = self.dc.read_file("CleaningData_TestFiles/cloudperc.xlsx")
        pressure = self.dc.read_file("CleaningData_TestFiles/Pressure_kpa.xlsx")
        temp = self.dc.read_file("CleaningData_TestFiles/Temperature_Celcius.xlsx")
        winddir = self.dc.read_file("CleaningData_TestFiles/Winddirection_degree.xlsx")
        windsp = self.dc.read_file("CleaningData_TestFiles/Windspeed_kmh.xlsx")
        forecasts_1 = self.dc.combine_columns(pressure, temp, winddir, windsp, cloudperc)

        forecasts_2 = self.dc.read_file("CleaningData_TestFiles/forecast_2.csv")

        actuals_1 = self.dc.rename_columns(actuals_1)
        actuals_2 = self.dc.rename_columns(actuals_2)
        forecasts_2 = self.dc.rename_columns(forecasts_2)

        actuals = self.dc.merge_files(actuals_1, actuals_2)
        forecasts = self.dc.merge_files(forecasts_1, forecasts_2)

        #same number of rows
        self.assertEqual(actuals.shape[0],forecasts.shape[0])
        
        #check for no duplicate dates!!
        self.assertEqual(actuals['datetime'].duplicated(), [])
        self.assertEqual(forecasts['datetime'].duplicated(), [])


    def test_separate_dt(self):
        datetime_df = self.dc.read_file("CleaningData_TestFiles/one_month.csv")
        datetime_df = self.dc.separate_datetime(datetime_df)

        full_datetime_df = self.dc.read_file("CleaningData_TestFiles/full_one_month.csv")


        self.assertTrue(False not in (datetime_df==full_datetime_df))
        
        
    @mock.patch('builtins.input', side_effect=['1','1','1','1','1'])
    def test_missing_values(self, mock_input):

        some_cells = self.dc.read_file("CleaningData_TestFiles/missing_cells.csv")
        some_cells = self.dc.rename_columns(some_cells)
        some_cells = self.dc.check_missing_values(some_cells)

        some_cells_nulls = (some_cells.isna().sum()==0).to_list()
        
        #has a few non-consecutive rows missing
        non_consec_rows = self.dc.read_file("CleaningData_TestFiles/missing_some_rows.csv")
        non_consec_rows = self.dc.rename_columns(non_consec_rows)
        non_consec_rows = self.dc.check_missing_values(non_consec_rows)

        non_consec_rows_nulls = (non_consec_rows.isna().sum()==0).to_list()

        #has consecutive rows missing (input=1)
        consec_rows = self.dc.read_file("CleaningData_TestFiles/missing_consec_rows.csv")
        consec_rows = self.dc.rename_columns(consec_rows)
        consec_rows = self.dc.check_missing_values(consec_rows)

        consec_rows_nulls = (consec_rows.isna().sum()==0).to_list()

        #a mix of consecutive and non-consecutive rows missing (input=2)
        multiple_rows = self.dc.read_file("CleaningData_TestFiles/missing_rows.csv")
        multiple_rows = self.dc.rename_columns(multiple_rows)
        multiple_rows = self.dc.check_missing_values(multiple_rows)

        multiple_rows_nulls = (multiple_rows.isna().sum()==0).to_list()

        #missing consec rows, non-consec rows, cells (input=2)
        missing_all = self.dc.read_file("CleaningData_TestFiles/missing_everything.csv")
        missing_all = self.dc.rename_columns(missing_all)
        missing_all = self.dc.check_missing_values(missing_all)

        missing_all_nulls = (missing_all.isna().sum()==0).to_list()


        self.assertTrue(False not in some_cells_nulls)
        self.assertEqual(some_cells.shape[0],744)

        self.assertTrue(False not in non_consec_rows_nulls)
        self.assertEqual(non_consec_rows.shape[0],744)

        self.assertTrue(False not in consec_rows_nulls)
        self.assertEqual(consec_rows.shape[0] == 720)

        self.assertTrue(False not in multiple_rows_nulls)
        self.assertEqual(multiple_rows.shape[0],720)

        self.assertTrue(False not in missing_all_nulls)
        self.assertEqual(missing_all.shape[0],731)



if __name__ == '__main__':
    unittest.main()
