import pandas as pd
import re
import json
import statistics
from datetime import timedelta

from Database.db_connector import DbConnector

class ForecastsHandler:
    def __init__(self):
        """
        Constructor for the ForecastsHandler class.
        Initializes an instance of DbConnector to interact with the database.
        Defines constants for column and data handling.

        Attributes:
        - dbClass (DbConnector): An instance of the DbConnector class.
        - NUM_OF_COLUMNS (int): Number of required columns in the DataFrame.
        - NUM_OF_ROWS (int): Expected number of rows in the DataFrame.
        - ONE_FILE_LENGTH (int): Expected length of a single uploaded file.
        - GAP_LIMIT (int): Maximum allowed gap between data points.
        """
        self.dbClass = DbConnector()
        self.NUM_OF_COLUMNS = 6
        self.NUM_OF_ROWS = 48
        self.ONE_FILE_LENGTH = 24
        self.GAP_LIMIT = 5

    def handle_predict_demand(self, uploaded_files):
        """
        Handles demand prediction files and processes them.

        Parameters:
        - uploaded_files (list): List of uploaded forecast files.

        Returns:
        - df (DataFrame): Processed DataFrame with demand predictions.
        - response (str): Response message indicating the outcome.
        """
        predictions = False
        valid = True
        response = "File(s) Received.\n"

        dfs = []
        for file in uploaded_files:
            # Check if the file type is CSV or Excel (XLSX or XLS)
            file_name = file.lower()
            if not file_name.endswith(('.csv', '.xlsx', '.xls')):
                return 'Unsupported file type. Please upload a CSV or Excel file.'

            # Process the uploaded file and clean the DataFrame
            df = self.handle_demand_prediction_file(file)
            df, valid, new_response = self.clean_df(df)
            response += new_response
            if valid:
                dfs.append(df)
            else:
                break

        if valid:
            df = self.combine_dfs(dfs)

            if len(df) > self.NUM_OF_ROWS:
                response = f"The data submitted is too large. It should contain up to {self.NUM_OF_ROWS} hours only."
                valid = False
            elif len(df) <= self.ONE_FILE_LENGTH:
                response = f"Please provide 2 days' worth of data (48 hours)."
                valid = False
            else:
                df, valid, new_response = self.clean_full_df(df)
                response += new_response

            if valid:
                response += "Making predictions\n"
                predictions = True

        print(response)
        if predictions:
            return df

    def check_files_consec(self, dfs):
        """
        Check if uploaded files represent consecutive days' worth of data.

        Parameters:
        - dfs (list): List of DataFrames representing forecast data.

        Returns:
        - consecutive (bool): True if files are consecutive, False otherwise.
        - response (str): Response message.
        """
        for i in range(1, len(dfs)):
            # Calculate the time difference
            time_diff_1 = dfs[i - 1]['datetime'].iloc[-1] - dfs[i]['datetime'].iloc[0]
            time_diff_2 = dfs[i]['datetime'].iloc[0] - dfs[i - 1]['datetime'].iloc[-1]

            # Check if the condition is met
            if time_diff_1 != pd.Timedelta(days=1) and time_diff_2 != pd.Timedelta(hours=1):
                return False, "Please provide 2 consecutive days' worth of data."
        return True, ""

    def clean_df(self, df):
        """
        Clean and preprocess a DataFrame representing forecast data.

        Parameters:
        - df (DataFrame): Input DataFrame with forecast data.

        Returns:
        - df (DataFrame): Cleaned DataFrame.
        - valid (bool): True if the DataFrame is valid, False otherwise.
        - response (str): Response message.
        """
        valid = True

        if len(df.columns) < self.NUM_OF_COLUMNS:
            return df, False, "Please provide all required columns."

        # Check if the file has a header
        has_header, response = self.check_df_has_header(df)
        if not has_header:
            return df, has_header, response

        # Rename columns to expected requirements
        df, valid, response = self.check_column_names(df)

        if not valid:
            return df, valid, response

        df = self.get_required_cols(df)

        # Change data types of columns
        df, valid, response = self.change_column_datatypes(df)
        if not valid:
            return df, valid, response

        df = self.remove_nulls(df)

        return df, valid, response

    def clean_full_df(self, df):
        """
        Clean and preprocess the entire DataFrame.

        Parameters:
        - df (DataFrame): Input DataFrame with forecast data.

        Returns:
        - df (DataFrame): Cleaned DataFrame.
        - valid (bool): True if the DataFrame is valid, False otherwise.
        - response (str): Response message.
        """
        valid = True
        response = ""

        # Check for other missing values throughout the DataFrame
        df, response = self.check_missing_values(df, response)

        init_len = len(df)
        df = self.check_duplicates(df)
        if len(df) < init_len:
            response += "Duplicates were detected in the given file.\n"

        # Check for valid date range
        valid = self.check_valid_dates(df)

        # Handle date gap if not valid
        if not valid:
            df, response, valid = self.handle_date_gap(df, response)

        # Handle outliers
        df, response = self.check_col_values(df, response)

        # Change data types of columns
        df, valid, _ = self.change_column_datatypes(df)

        # Add time-based features
        df = self.add_features(df)

        # Sort the DataFrame
        df = self.sort_df(df)

        return df, valid, response

    def sort_df(self, df):
        """
        Sort the DataFrame by the 'datetime' column in ascending order.

        Parameters:
        - df (DataFrame): Input DataFrame to sort.

        Returns:
        - df (DataFrame): Sorted DataFrame.
        """
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime', ascending=True)
        return df

    def remove_nulls(self, df):
        """
        Remove rows with all null (NaN) values from the DataFrame.

        Parameters:
        - df (DataFrame): Input DataFrame to remove null rows.

        Returns:
        - df (DataFrame): DataFrame with null rows removed.
        """
        df = df.dropna(how='all')
        return df

    def check_duplicates(self, df):
        """
        Check and remove duplicated rows based on the 'datetime' column.

        Parameters:
        - df (DataFrame): Input DataFrame to check for duplicates.

        Returns:
        - df (DataFrame): DataFrame with duplicate rows removed.
        """
        df = df.drop_duplicates(subset='datetime')
        df = df.reset_index(drop=True)
        return df

    def check_column_names(self, df):
        """
        Check and rename columns in the DataFrame based on patterns in the 'colForecastNames.json' file.

        Parameters:
        - df (DataFrame): Input DataFrame with forecast data.

        Returns:
        - df (DataFrame): DataFrame with renamed columns.
        - valid (bool): True if columns are renamed correctly, False otherwise.
        - response (str): Response message.
        """
        column_name_patterns = self.read_json_file('./ReferenceData/colForecastNames.json')

        changes = 0
        valid = True
        response = ""

        for column in df.columns:
            for pattern, replacement in column_name_patterns.items():
                if re.search(pattern, column, re.IGNORECASE):
                    df.rename(columns={column: replacement}, inplace=True)
                    changes += 1

        if changes != 6:
            valid = False
            response = "Please provide the correct columns. "

        return df, valid, response

    def get_required_cols(self, df):
        """
        Select and keep only the required columns in the DataFrame.

        Parameters:
        - df (DataFrame): Input DataFrame with forecast data.

        Returns:
        - df (DataFrame): DataFrame with only the required columns.
        """
        colnames = self.read_txt_file('./ReferenceData/weather_vars.txt') + ['datetime']
        df = df[colnames]
        return df

    def check_df_has_header(self, df):
        """
        Check if the DataFrame has a header row with the correct column names.

        Parameters:
        - df (DataFrame): Input DataFrame to check for a header.

        Returns:
        - has_header (bool): True if a header row with the correct column names is present, False otherwise.
        - response (str): Response message.
        """
        response = ""
        column_name_patterns = self.read_json_file('./ReferenceData/colForecastNames.json')
        column_name_patterns = list(column_name_patterns.keys())

        has_header = any(re.search(
            pattern, df.columns[0], re.IGNORECASE) for pattern in column_name_patterns)
        if not has_header:
            response = "Please provide a file with the correct column names"

        return has_header, response

    def change_column_datatypes(self, df):
        """
        Change data types of columns in the DataFrame based on patterns in the 'colForecastTypes.json' file.

        Parameters:
        - df (DataFrame): Input DataFrame to change column data types.

        Returns:
        - df (DataFrame): DataFrame with modified column data types.
        - valid (bool): True if data types are successfully modified, False otherwise.
        - response (str): Response message.
        """
        data_types = self.read_json_file('./ReferenceData/colForecastTypes.json')

        response = ""
        valid = True
        for column, dtype in data_types.items():
            try:
                df[column] = df[column].astype(dtype)
            except (ValueError, TypeError, KeyError):
                response = f"Error converting '{column}' to {dtype}"
                valid = False

        return df, valid, response

    def check_valid_dates(self, df):
        """
        Check if the dates in the DataFrame fall within a valid range.

        Parameters:
        - df (DataFrame): Input DataFrame to check date validity.

        Returns:
        - valid (bool): True if dates are within a valid range, False otherwise.
        """
        last_date = self.dbClass.get_last_date_predictions()
        desired_time_gap = timedelta(hours=1)
        first_datetime_in_df = df['datetime'].min()
        valid = True
        if first_datetime_in_df - last_date > desired_time_gap:
            valid = False

        return valid

    def handle_date_gap(self, df, response):
        """
        Handle date gap in the DataFrame by filling missing dates.

        Parameters:
        - df (DataFrame): Input DataFrame with date gaps.

        Returns:
        - df (DataFrame): DataFrame with filled date gaps.
        - response (str): Response message.
        - valid (bool): True if gaps are handled successfully, False otherwise.
        """
        valid = True
        column_names = self.read_txt_file('./ReferenceData/weather_vars.txt')
        last_date = self.dbClass.get_last_date_actuals()
        df = self.sort_df(df)
        first_datetime_in_df = df['datetime'].min()
        desired_time_gap = timedelta(hours=1)
        missing_dates = []
        missing_dates_values = []

        if first_datetime_in_df - last_date > timedelta(weeks=self.GAP_LIMIT):
            response += f"Gap between received data and last prediction is too large. Please provide actual values or request to predict from {last_date + desired_time_gap}"
            valid = False
            return df, response, valid

        current_datetime = last_date + desired_time_gap
        while current_datetime < first_datetime_in_df:
            missing_dates.append(current_datetime)
            current_datetime += desired_time_gap

        time_difference = (first_datetime_in_df - last_date).days
        go_back = timedelta(days=time_difference + 1)

        for missing_date in missing_dates:
            row = [missing_date]
            previous_go_back = missing_date - go_back

            data = self.dbClass.get_data_from_actuals(previous_go_back, missing_date, previous_go_back.hour)

            for col in column_names:
                median = data[col].median()
                row.append(median)

            missing_dates_values.append(row)

        gap_df = pd.DataFrame(missing_dates_values, columns=['datetime'] + column_names)

        gap_df = self.sort_df(gap_df)

        response += f"\nWARNING: You have not submitted actual values for previous days before {first_datetime_in_df}.\nSolution implemented is to make predictions for the missing days and use them as the actual values.\nAs a result, the prediction accuracy may be negatively impacted.\n"

        df = self.combine_dfs([gap_df, df])
        df = self.sort_df(df)

        return df, response, valid

    def check_col_values(self, df, response):
        """
        Check and handle column values for outliers in the DataFrame.

        Parameters:
        - df (DataFrame): Input DataFrame with columns to check.

        Returns:
        - df (DataFrame): DataFrame with outlier values handled.
        """
        var = df.copy()

        colNames = self.read_txt_file('./ReferenceData/weather_vars.txt')

        lastdate_db = self.dbClass.get_last_date_actuals()
        lastdate_df = df.datetime.max()

        actuals = self.dbClass.get_data_from_actuals()

        for column in colNames:
            if column in df.columns:
                if lastdate_db > lastdate_df:
                    start_date = lastdate_df - timedelta(days=3)
                    end_date = lastdate_df
                else:
                    start_date = lastdate_db - timedelta(days=3)
                    end_date = lastdate_db

                nearest_records = self.dbClass.get_data_from_actuals(start_date, end_date)

                median_col = statistics.median(list(nearest_records[column]))

                df.loc[(df[column] < actuals[column].min()), column] = median_col
                df.loc[(df[column] > actuals[column].max()), column] = median_col

        if not df.equals(var):
            response += "A few outliers were detected which were dealt with.\n"

        return df, response

    def combine_dfs(self, dfs):
        """
        Combine multiple DataFrames into a single DataFrame.

        Parameters:
        - dfs (list): List of DataFrames to combine.

        Returns:
        - df (DataFrame): Combined DataFrame.
        """
        df = pd.concat(dfs)
        df.reset_index(drop=True, inplace=True)
        return df
            

    def check_missing_values(self, df, response):
        """
        Check and handle missing values in the DataFrame.

        Parameters:
        - df (DataFrame): Input DataFrame with missing values.
        - response (str): Initial response message.

        Returns:
        - df (DataFrame): DataFrame with missing values handled.
        - response (str): Response message.
        - valid (bool): True if missing values are handled successfully, False otherwise.
        """
        column_names = self.read_txt_file('./ReferenceData/weather_vars.txt')

        df = df.dropna(how='all')

        var = df.copy()

        start_datetime = df['datetime'].min().replace(hour=8, minute=0, second=0)
        end_datetime = start_datetime + timedelta(hours=47)

        df = df.reset_index(drop=True)

        for i in range(len(df)):
            if pd.isna(df.at[i, 'datetime']):
                df.at[i, 'datetime'] = start_datetime + timedelta(hours=i)

        #gets the missing dates
        date_range = pd.date_range(start=start_datetime, end=end_datetime, freq='H')
        missing_dates = date_range[~date_range.isin(df['datetime'])]
        
        if not missing_dates.empty:
            response = f"File submitted contains some missing rows, which have been appropriately dealt with."
            missing_dates_fix = [] #stores the missing dates rows 

            for date in missing_dates:
                row = [date] + [None] * len(column_names)
                missing_dates_fix.append(row)
            
            missing_dates_fix = pd.DataFrame(missing_dates_fix, columns=['datetime'] + column_names)
            df = pd.concat([df, missing_dates_fix], ignore_index=True)

        #interpolate
        for col in column_names:
            df[col].interpolate(method='linear', limit_direction='both', inplace=True)

            # Interpolate in forward order across the column:
            df[col].interpolate(method ='linear', limit_direction ='forward', inplace=True)

        if not df.equals(var):
            response += "Missing values were detected in the file which were dealt with."

        df = self.sort_df(df)  # Sort the df just to be safe
        df = df.reset_index(drop=True)

        return df, response

    def add_features(self, df):
        """
        Add time-based features to the DataFrame.

        Parameters:
        - df (DataFrame): Input DataFrame to add time-based features.

        Returns:
        - df (DataFrame): DataFrame with added time-based features.
        """
        df['date'] = df['datetime'].apply(lambda x: x.date())
        df['month'] = df['datetime'].apply(lambda x: x.month)
        df['type_of_day'] = df['datetime'].apply(lambda x: x.isoweekday())
        df['hour'] = df['datetime'].apply(lambda x: x.hour)
        return df

    def handle_demand_prediction_file(self, f):
        """
        Handle demand prediction file (CSV or Excel) and return it as a DataFrame.

        Parameters:
        - f (str): File path for the demand prediction file.

        Returns:
        - df (DataFrame): DataFrame containing the demand prediction data.
        """
        df = pd.DataFrame()
        if f.endswith('.csv'):
            df = pd.read_csv(f)
        elif f.endswith('.xlsx'):
            df = pd.read_excel(f)
        return df

    def get_final_df(self, df):
        """
        Get the final DataFrame with 'datetime' as the index.

        Parameters:
        - df (DataFrame): Input DataFrame.

        Returns:
        - df (DataFrame): DataFrame with 'datetime' as the index.
        """
        df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
        df = df.set_index('datetime')
        df = df.asfreq('H')
        return df

    def read_txt_file(self, filename):
        """
        Read a text file and return its contents as a list.

        Parameters:
        - filename (str): Path to the text file.

        Returns:
        - wordlist (list): List of lines from the text file.
        """
        wordlist = []
        with open(filename, 'r') as file:
            for line in file:
                wordlist.append(line.strip())
        return wordlist

    def read_json_file(self, filename):
        """
        Read a JSON file and return its contents as a dictionary.

        Parameters:
        - filename (str): Path to the JSON file.

        Returns:
        - file (dict): JSON data as a dictionary.
        """
        wordlist = []
        with open(filename, "r") as json_file:
            file = json.load(json_file)
        return file
