import pandas as pd
import os, json, re, statistics
from datetime import timedelta

from Database.db_connector import DbConnector

class ActualsHandler:
    """
    The `ActualsHandler` class handles the upload and processing of actuals data, ensuring it meets specific criteria.
    It validates, cleans, and prepares the data for storage in the database.
    """

    def __init__(self):
        """
        Initialize the `ActualsHandler` class, including a connection to the database.
        """
        self.dbClass = DbConnector()
        self.NUM_OF_COLUMNS = 7
        self.NUM_OF_ROWS = 48

    def handle_upload_actuals(self, uploaded_files):
        """
        Handle the upload of actuals data.

        Parameters:
        - uploaded_files (list): List of uploaded files.

        Returns:
        - df (DataFrame or str): Cleaned and validated DataFrame if data is valid, otherwise an error message.

        This method handles the uploaded files, checking their validity, cleaning, and processing them.
        If the data is valid, it returns the cleaned DataFrame.
        """
        valid = True
        response = "File(s) Received.\n"


        dfs = []
        for file in uploaded_files:
            # Check if the file type is CSV or Excel (XLSX or XLS)
            file_name = file.lower()
            if not file_name.endswith(('.csv', '.xlsx', '.xls')):
                return 'Unsupported file type. Please upload a CSV or Excel file.'

            df = self.handle_actuals_file(file)
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
            else:
                df, valid, new_response = self.clean_full_df(df)
                response += new_response

                if valid:
                    response += "\nSaved into the database."
                    print(response)
                    # self.save_actuals(df)
        
        if not valid:
            print(response)


    def save_actuals(self, df):
        """
        Saves the data into the database's 'actuals' table by calling respective method from db_connector.py
        """
        self.dbClass.update_actuals_load(df)

    def sort_df(self, df):
        """
        Sort the DataFrame by datetime in ascending order.

        Parameters:
        - df (DataFrame): Input DataFrame to be sorted.

        Returns:
        - df (DataFrame): DataFrame sorted by datetime.
        """
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime', ascending=True)
        return df

    def handle_actuals_file(self, f):
        """
        Handle the actuals file, loading its data into a DataFrame.

        Parameters:
        - f (str): File path of the actuals file.

        Returns:
        - df (DataFrame): DataFrame containing the file data.
        """
        df = pd.DataFrame()

        if f.endswith('.csv'):
            df = pd.read_csv(f)

        elif f.endswith('.xlsx'):
            df = pd.read_excel(f)

        return df

    def check_files_consec(self, dfs):
        """
        Check if the datetime values in multiple DataFrames are consecutive with an hour interval.

        Parameters:
        - dfs (list): List of DataFrames to check.

        Returns:
        - bool: True if datetime values are consecutive, False otherwise.
        """
        for i in range(1, len(dfs)):
            # Calculate the time difference
            time_diff_1 = dfs[i - 1]['datetime'].iloc[-1] - dfs[i]['datetime'].iloc[0]
            time_diff_2 = dfs[i]['datetime'].iloc[-1] - dfs[i - 1]['datetime'].iloc[0]

            # Check if the condition is met
            if time_diff_1 != pd.Timedelta(hours=1) and time_diff_2 != pd.Timedelta(hours=1):
                return False
        return True

    def clean_df(self, df):
        """
        Clean and validate the DataFrame.

        Parameters:
        - df (DataFrame): Input DataFrame to be cleaned.

        Returns:
        - df (DataFrame): Cleaned DataFrame.
        - valid (bool): True if DataFrame is valid, False otherwise.
        - response (str): Message describing the cleaning process.

        This method performs cleaning and validation on the input DataFrame.
        """
        valid = True
        response = ""

        # Check if the DataFrame has enough columns
        if len(df.columns) < self.NUM_OF_COLUMNS:
            return df, False, "Please provide all required columns."

        # Check if the file has a header row
        has_header, new_response = self.check_df_has_header(df)
        if not has_header:
            return df, has_header, new_response

        # Rename columns to expected requirements
        df, valid, new_response = self.check_column_names(df)
        if not valid:
            return df, valid, new_response

        # Keep only the required columns
        df = self.get_required_cols(df)

        # Remove null values
        init_len = len(df)
        df = self.remove_nulls(df)
        if len(df)<init_len:
            if not "File contains missing values." in response: #So that it does not print twice
                response += "File contains missing values. "

        # Change data types of columns
        df, valid, new_response = self.change_column_datatypes(df)
        if not valid:
            return df, valid, new_response

        # Sort the DataFrame by datetime
        df = self.sort_df(df)

        return df, valid, response

    def clean_full_df(self, df):
        """
        Clean and process the entire DataFrame.

        Parameters:
        - df (DataFrame): Input DataFrame to be cleaned.

        Returns:
        - df (DataFrame): Cleaned DataFrame.
        - valid (bool): True if DataFrame is valid, False otherwise.
        - response (str): Message describing the cleaning process.

        This method performs comprehensive cleaning and processing on the entire DataFrame.
        """
        valid = True
        response = ""

        # Sort the DataFrame by datetime
        df = self.sort_df(df)

        # Check for and remove duplicated rows
        init_len = len(df)
        df = self.check_duplicates(df)
        if len (df) != init_len:
            response += "Duplicates were detected in the given file.\n"

        # Check for valid datetime values
        # valid, response = self.check_valid_dates(df, response)
        # if not valid:
        #     return df, valid, response

        # Check column values for outliers or incorrect values
        df, new_response = self.check_col_values(df, response)
        response += new_response

        # Add additional features to the DataFrame
        df = self.add_features(df)

        # Sort the DataFrame by datetime again
        df = self.sort_df(df)

        return df, valid, response

    def combine_dfs(self, dfs):
        """
        Combine multiple DataFrames into one.

        Parameters:
        - dfs (list): List of DataFrames to be combined.

        Returns:
        - df (DataFrame): Combined DataFrame.
        """
        df = pd.concat(dfs)
        df.reset_index(drop=True, inplace=True)
        return df

    def remove_nulls(self, df):
        """
        Remove rows with all null values.

        Parameters:
        - df (DataFrame): Input DataFrame with potential null values.

        Returns:
        - df (DataFrame): DataFrame with null rows removed.
        """
        df = df.dropna();
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


    def check_col_values(self, df, response):
        """
        Check column values for outliers and correct them if needed.

        Parameters:
        - df (DataFrame): Input DataFrame with column values to check.

        Returns:
        - df (DataFrame): DataFrame with corrected column values.
        """
        var = df.copy()

        #the number of days we want to go back to retrieve data, compute the median, and replace wrong values with is 3.
        colNames = self.read_txt_file('./ReferenceData/weather_vars.txt')
        colNames+= ['load']
        lastdate_db = self.dbClass.get_last_date_actuals()
        lastdate_df = df.datetime.max()

        actuals = self.dbClass.get_data_from_actuals()

        for column in colNames:
            if column in df.columns:
                if lastdate_db > lastdate_df: #get data from database then
                    start_date =lastdate_df - timedelta(days=3)
                    end_date = lastdate_df
                else: #use data from the dataframe
                    start_date = lastdate_db - timedelta(days=3)
                    end_date = lastdate_db

                nearest_records = self.dbClass.get_data_from_actuals(start_date, end_date)

                median_col = statistics.median(list(nearest_records[column]))

                # Replace values below min_value and above max_value with the median value
                df.loc[(df[column] < actuals[column].min()), column] = median_col
                df.loc[(df[column] > actuals[column].max()), column] = median_col

        if not df.equals(var):
            response += "A few outliers were detected which were dealt with. "

        return df, response

    def check_column_names(self, df):
        """
        Check and rename column names based on predefined patterns.

        Parameters:
        - df (DataFrame): Input DataFrame with column names to check and potentially rename.

        Returns:
        - df (DataFrame): DataFrame with renamed columns.
        - valid (bool): True if DataFrame is valid, False otherwise.
        - response (str): Response message.
        """
        column_name_patterns = self.read_json_file('./ReferenceData/colActualNames.json')
        changes = 0
        valid = True
        response = ""

        for column in df.columns:
            for pattern, replacement in column_name_patterns.items():
                if re.search(pattern, column, re.IGNORECASE):
                    df.rename(columns={column: replacement}, inplace=True)
                    changes += 1

        if changes != self.NUM_OF_COLUMNS:
            valid = False
            response = "Please provide all the correct columns. "

        return df, valid, response

    def change_column_datatypes(self, df):
        """
        Change column data types based on predefined types.

        Parameters:
        - df (DataFrame): Input DataFrame with columns to change data types.

        Returns:
        - df (DataFrame): DataFrame with updated data types.
        - valid (bool): True if DataFrame is valid, False otherwise.
        - response (str): Response message.
        """
        data_types = self.read_json_file('./ReferenceData/colActualTypes.json')
        response = ""
        valid = True

        for column, dtype in data_types.items():
            try:
                df[column] = df[column].astype(dtype)
            except (ValueError, TypeError, KeyError):
                response = f"Error converting '{column}' to {dtype}"
                valid = False

        return df, valid, response

    def check_valid_dates(self, df, response):
        """
        Check if datetime values are valid and within an expected range.

        Parameters:
        - df (DataFrame): Input DataFrame with datetime values to check.
        - response (str): Response message.

        Returns:
        - valid (bool): True if datetime values are valid, False otherwise.
        - response (str): Response message.
        """
        last_date = self.dbClass.get_last_date_actuals()
        try:
            last_predicted_date = self.dbClass.get_last_date_predictions()
        # If no prediction data exists
        except len(self.dbClass.get_data_from_predictions) == 0:
            last_predicted_date = df.datetime.min()
        desired_time_gap = timedelta(hours=1)
        valid = True
        first_datetime_in_df = df['datetime'].max()

        if first_datetime_in_df > last_predicted_date:
            valid = False
            response += f"Please request for a prediction from {last_predicted_date} first, then upload the actual values."

        return valid, response

    def get_required_cols(self, df):
        """
        Keep only required columns in the DataFrame.

        Parameters:
        - df (DataFrame): Input DataFrame with columns to be filtered.

        Returns:
        - df (DataFrame): DataFrame with only required columns.
        """
        colnames = self.read_txt_file('./ReferenceData/uiactualcolnames.txt')
        df = df[colnames]
        return df

    def check_df_has_header(self, df):
        """
        Check if the DataFrame has a header row.

        Parameters:
        - df (DataFrame): Input DataFrame to check for a header row.

        Returns:
        - has_header (bool): True if a header row is present, False otherwise.
        - response (str): Response message.
        """
        response = ""
        column_name_patterns = self.read_json_file('./ReferenceData/colActualNames.json')
        column_name_patterns = list(column_name_patterns.keys())

        has_header = any(re.search(
            pattern, df.columns[0], re.IGNORECASE) for pattern in column_name_patterns)
        if not has_header:
            response = "Please provide a file with the correct column names.\n"

        return has_header, response

    def add_features(self, df):
        """
        Add additional features to the DataFrame.

        Parameters:
        - df (DataFrame): Input DataFrame to add features to.

        Returns:
        - df (DataFrame): DataFrame with added features.
        """
        df['date'] = df['datetime'].apply(lambda x: x.date())
        df['month'] = df['datetime'].apply(lambda x: x.month)
        df['type_of_day'] = df['datetime'].apply(lambda x: x.isoweekday())
        df['hour'] = df['datetime'].apply(lambda x: x.hour)
        df['year'] = df['datetime'].apply(lambda x: x.year)
        return df

    def read_txt_file(self, filename):
        """
        Read and return the content of a text file as a list of words.

        Parameters:
        - filename (str): Path to the text file.

        Returns:
        - wordlist (list): List of words read from the text file.
        """
        wordlist = []
        filename = os.path.join(filename)
        with open(filename, 'r') as file:
            for line in file:
                wordlist.append(line.strip())
        return wordlist

    def read_json_file(self, filename):
        """
        Read and return the content of a JSON file as a dictionary.

        Parameters:
        - filename (str): Path to the JSON file.

        Returns:
        - file (dict): Dictionary read from the JSON file.
        """
        wordlist = []
        filename = os.path.join(filename)
        with open(filename, "r") as json_file:
            file = json.load(json_file)
        return file