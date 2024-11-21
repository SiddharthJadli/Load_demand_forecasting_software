import pandas as pd
from datetime import timedelta

""" 
This DataCleaning class should be used to clean and merge files that will be used for training 
the forecasting model. 
"""

class DataCleaning:

    def read_file(self, file_path:str):
        """ 
        Internal method used to read files from a path into a pandas dataframe. 
        """

        pd.set_option('display.float_format', lambda x: '%.5f' % x)

        if file_path.find(".xlsx") >= 0:
            return pd.read_excel(file_path, parse_dates=True)
            
            
        elif file_path.find(".csv")>=0: 
            return pd.read_csv(file_path, parse_dates=True)

        else:
            return 'Insert a file type of .xlsx or .csv'
    
    def rename_columns(self, df:pd.DataFrame, single_col=False):
        """
        Internal method to rename columns to ensure consistent column naming conventions throughout
        the entire file and database. This method assumes that the only columns provided are: 
        "Time", "Temperature", "Cloud Cover", "Wind Direction", "Wind Speed", "Pressure", "Load" and 
        "Humidity"

        This method will also convert the time column to its suitable data type, as well as sort all
        the rows within the file.
        """
        col_list = df.columns.to_list()

        #renaming the columns 
        for i in range(len(col_list)):
            column_lower = col_list[i].lower()

            if (column_lower.find("time") >= 0) or (column_lower.find("date") >= 0):
                df.rename(columns={col_list[i]: "datetime"}, inplace=True)

                #convert time to type datetime
                df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
                
            elif column_lower.find("temp") >= 0:
                df.rename(columns={col_list[i]: "temperature_c"}, inplace=True)
                
            elif column_lower.find("cloud") >= 0:
                df.rename(columns={col_list[i]: "cloud_cover"}, inplace=True)
                
            elif (column_lower.find("wind") >= 0) and (column_lower.find("dir")>=0):
                df.rename(columns={col_list[i]: "wind_direction"}, inplace=True)
            
            elif column_lower.find("wind") >= 0 and (column_lower.find("speed")) >= 0:
                df.rename(columns={col_list[i]: "wind_speed_kmh"}, inplace=True)
            
            elif column_lower.find("pressure") >= 0:
                df.rename(columns={col_list[i]: "pressure_kpa"}, inplace=True)

            elif column_lower.find("load") >= 0:
                df.rename(columns={col_list[i]: "load_kw"}, inplace=True)
            
            elif column_lower.find("humid") >= 0:
                df.rename(columns={col_list[i]: "humidity"}, inplace=True)

            
        #drop any columns that aren't a part of the names expected:
        wanted_cols = ["datetime", "temperature_c", "cloud_cover", "wind_direction", "wind_speed_kmh", 
                        "pressure_kpa", "load_kw", "humidity"]
        
        df_columns = df.columns.to_list()

        for col in df_columns: 
            if col not in wanted_cols:
                df.drop(columns=[col], inplace=True)
                
        #sort the rows by time
        df.sort_values(by='datetime', inplace=True)


        if not single_col: 
            self.separate_datetime(df)

        return df
    
    
    # for different weather variables in different files
    def combine_columns(self, pressure, temp, winddir, windsp, cloudperc, humidity=None, load=None):
        """ 
        This method handles when the variables are stored in their individual files. It will 
        merge all the files and return one single merged dataframe that holds all the variables
        provided. 
        """ 
       
        pressure = self.rename_columns(pressure, True)
        temp = self.rename_columns(temp, True)
        winddir = self.rename_columns(winddir, True)
        windsp = self.rename_columns(windsp, True)
        cloudperc = self.rename_columns(cloudperc, True)


        output_df = pd.merge(pressure, temp, on='datetime', how='inner') \
                    .merge(winddir, on='datetime', how='inner')\
                    .merge(windsp, on='datetime', how='inner')\
                    .merge(cloudperc, on='datetime', how='inner')

        if humidity != None:
            self.rename_columns(humidity)
            output_df = pd.merge(output_df, humidity, on='datetime', how='inner')

        if load != None:
            self.rename_columns(load)
            output_df = pd.merge(output_df, load, on='datetime', how='inner')

        #adds the date variables
        output_df = self.separate_datetime(output_df)

        return output_df


    #for merging pre and post, checks for repeated dates
    
    def merge_files(self, df_1, df_2):
        """
        Internal method used to concatenate 2 files of same data columns but different time.
        This method will also handle any overlapping dates by grouping the duplicated rows 
        and replacing their values with an average. This will ensure that there will be no
        duplicated values for any of the dates. 
        """
        #first check that the latest date in first file is the hour before the first date in second file
        #if first date<latest date then handle by averaging and merging

        merged = pd.concat([df_1, df_2])
        merged.sort_values(by='datetime', inplace=True)

        # dup_dates = merged['datetime'].duplicated()

        # dup_rows = merged.loc[(dup_dates)]

        col_list = merged.columns.to_list()

        for col_name in col_list:
            if col_name != "datetime":
                merged[col_name] = merged.groupby('datetime')[col_name].transform('mean')

        merged.drop_duplicates(inplace=True)

        return merged


    
    def separate_datetime(self, df):

        """
        Internal method that adds additional columns to the dataframe by creating additional 
        columns to hold the date, month, day and hour respectively.  
        """
        
        #checks that the datetime column is type datetime
        df['datetime'] = pd.to_datetime(df['datetime'])

        df['date'] = df.datetime.apply(lambda x: x.date())
        df['date'] = pd.to_datetime(df['date'])

        df['month'] = df.datetime.apply(lambda x: x.month)
        df['month'] = pd.to_numeric(df['month'])

        df['type_of_day'] = df.datetime.apply(lambda x: x.weekday() + 1)
        df['type_of_day'] = pd.to_numeric(df['type_of_day'])

        df['hour'] = df.datetime.apply(lambda x: x.hour)
        df['hour'] = pd.to_numeric(df['hour'])

        return df


    
    def check_missing_values(self, df):
        """ 
        Internal method that checks for any empty rows. If there are more than 5 consecutive missing rows, the 
        method will allow users to choose if they want to ignore the missing rows or if they want to re-invoke 
        the method with a new dataframe.     
        
        The method will assume that there are missing cells even if there are no missing rows and will handle that
        appropriately by invoking the handle_missing_cells() method.
        """
        start = df.datetime.min()
        end = df.datetime.max()

        #if start isn't at 00:00:00
        if start.hour != 0:
            start = pd.Timestamp(year=start.year,
                                month=start.month,
                                day=start.day,
                                hour=0)

        #if end isn't at 23:00:00
        if end.hour != 23:
            end = pd.Timestamp(year=end.year, 
                                month=end.month, 
                                day=end.day, 
                                hour=23)

        missing_hours = pd.date_range(start = start, end= end, freq='H').difference(df.datetime).to_list()    
        

        #iterate through the list to see if it there are any consecutive rows
        too_many_missing = False
        row_count = 1
        consec_rows = [False] * len(missing_hours)

        for i in range(1, len(missing_hours)):
            if missing_hours[i] - timedelta(hours=1) != missing_hours[i-1]:
                row_count = 1
            
            else:   #curr time is consec of the time before
                row_count += 1
                
                if row_count == 5:
                    too_many_missing = True
                    
                    for c in range(4,-1,-1):
                        consec_rows[i-c] = True

                if row_count > 5:
                    consec_rows[i] = True
        
        missing_hours = [missing_hours[m] for m in range(len(missing_hours)) if not consec_rows[m]]
        

        if too_many_missing:
            user_input = input(print("The data you have provided has too many missing values. Would you like to \n",
                                    "replace the data with a new one or ignore these missing values?:\n",
                                    "1: Ignore \n 2: Provide new data"))
            
            while user_input not in ["1","2"]:
                user_input = input(print("Please provide a valid choice\n",
                                    "1: Ignore \n 2: Provide new data"))

            if user_input == "1":
                return self.handle_missing_rows(df, missing_hours)
                

            else: 
                print("Re-invoke the method with new data")
                return df

            # return user_input
        #assume data has some missing values, just not consecutive
        else: 
            #some missing rows
            if len(missing_hours) > 0:
                print("missing hours?")
                return self.handle_missing_rows(df, missing_hours)
            
            #no missing rows, but check for missing cells and handle those 
            else: 
                return self.handle_missing_cells(df)



    def handle_missing_rows(self, df, missing_hours):
        """ 
        Internal method that handles any missing rows. This method will be invoked in the check_missing_rows
        method. The method will replace any missing rows by interpolating from either the previous value
        (if it isn't the first row of the dataframe) and the next value (if it is the first row of the dataframe)
        """
        
        replacement_rows = []
        col_names = df.columns.to_list()

        #ensures that datetime is the first element in the columns list
        col_names.remove('datetime')
        col_names.insert(0,'datetime')


        for t in missing_hours:
            row = [t] + [None] * (len(df.columns)-1)
            replacement_rows.append(row)

        #create a new dataframe for the replacements
        replacement_df = pd.DataFrame(data=replacement_rows, columns=col_names)
        
        #get the date, month, date, hour
        replacement_df = self.separate_datetime(replacement_df)

        complete = pd.concat([df,replacement_df])
        complete.sort_values('datetime', inplace=True)

        for col in col_names:
            complete[col].interpolate(method='bfill', inplace=True)
            complete[col].interpolate(method='ffill', inplace=True)

        
        return complete


    def handle_missing_cells(self, df):
        """ 
        Internal method that will replace values in any missing cells. This method is invoked in the
        check_missing_rows() method and will be called if there are rows found missing. 
        """
        col_names = df.columns.to_list()

        for col in col_names:
            df[col].interpolate(method='bfill', inplace=True)
            df[col].interpolate(method='ffill', inplace=True)

        return df
        