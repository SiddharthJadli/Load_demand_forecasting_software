from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from ..models import Actuals, Pracpredictions
from ..forms import PredictDemandForm
from django.urls import reverse
import pandas as pd
from datetime import datetime, timedelta
from django.core import serializers
from django.db.models import Count, Avg, Sum
from django.db.models.functions import TruncDay, TruncWeek, TruncMonth
from django.utils import timezone
import plotly.express as px
from plotly.subplots import make_subplots
from ..ElectricityForecasterModel.electricity_demand_forecaster import ElectricityForecaster
import os, re, json, statistics, joblib, tempfile


"""
This file processes the 'upload actual values' tab. 
It cleans the data,checks for errors, saves the data inton the database, and asks the model to check for retraining.
"""


#some global variables required for the methods
current_path = os.path.dirname(os.path.abspath(__file__))
desired_timezone = 'Australia/Sydney'
NUM_OF_COLUMNS = 6
NUM_OF_ROWS = 48
ONE_FILE_LENGTH = 24
GAP_LIMIT = 5 # 5 weeks cap


# method called in views.py
def handle_predict_demand(request):
    context = {}
    valid = True
    response = "File Received.\n"
    form_record = PredictDemandForm(request.POST, request.FILES) #get the form record
    if form_record.is_valid():
        # response, valid = determine_if_file_valid(request.FILES['file'])
        uploaded_files = request.FILES.getlist('files')
        dfs = []
        for file in uploaded_files:
            # Check if the file type is CSV or Excel (XLSX or XLS)
            valid, new_response = form_file_type_correct(file)
            if not valid:
                context['predict_message'] = new_response
                return form_record, context
             
           
            # order here is important

            #convert the file into a dataframe based on the file type
            df = handle_demand_prediction_file(file)

            #handle the individual file, checking for errors and making some changes like column names
            df, valid, new_response = clean_df(df)
            response += new_response
            if valid: #if df has no errors, append and continue
                dfs.append(df)
            else:
                break

        if valid:
            if valid:
                df = combine_dfs(dfs) #combine the files as a single dataframe

                #if the combined dataframe has more than 48 rows, there must be an error
                if len(df) > NUM_OF_ROWS:
                    response = f"The data submitted is too large. It should contain up to {NUM_OF_ROWS} hours only."
                    valid = False
                
                #on the other hand, if the number of rows is too small, there must be an error
                elif len(df) <= ONE_FILE_LENGTH:
                    response = f"Please provide 2 days worth of data. (48 hours)"
                    valid = False
                else:

                    #clean the entire dataframe, looking for errors, missing values, etc
                    df, valid, new_response = clean_full_df(df)
                    response += new_response

                    if valid: 

                        #everythign is good, proceed to make predictions
                        preds = make_prediction(df)

                       
                        #get data from the database as we need so we can plot it on the graph in the UI page
                        predictions = get_preds_from_db(df['datetime'].min(), df['datetime'].max())
                        actuals_past = get_actuals_from_db(df['datetime'].min(), df['datetime'].max())
                        pred_plot = get_predict_plot(actuals_past,  predictions)

                        context['pred_plot'] = pred_plot

    context['predict_message'] = response
    context['uploaded'] = True

    return form_record, context


#checks if the file type is correct
def form_file_type_correct(file):
    file_name = file.name.lower()
    if not file_name.endswith(('.csv', '.xlsx', '.xls')):
        return False, 'Unsupported file type. Please upload a CSV or Excel file.'
    return True, ""


#get the data from the actuals table based on a date range
def get_actuals_from_db(start_datetime, end_datetime):

    #if 48, then normal prediction:
    if end_datetime - start_datetime <= timedelta(hours=48):
        start = start_datetime - timedelta(hours=47)
        end = start_datetime + timedelta(hours=1)
    #else there was a date gap
    else:
        end = end_datetime - timedelta(hours=47) 
        start = end  - timedelta(hours=48) 
    # records = list(Actuals.objects.order_by(
    #         '-datetime').values('datetime', 'load'))[:48][::-1]
    records = list(Actuals.objects.filter(datetime__range=(start, end)).values('datetime', 'load'))
    return records

#get the data from the predictions table based on a date range
def get_preds_from_db(start_datetime, end_datetime):
    #if 48, then normal prediction:
    if end_datetime - start_datetime <= timedelta(hours=48):
        start = start_datetime - timedelta(hours=1)
        end = end_datetime
    #else there was a date gap
    else:
        end = end_datetime
        start = end - timedelta(hours=47)
    records = list(Pracpredictions.objects.filter(datetime__range=(start, end)).values('datetime', 'load'))
    return records


# ------------- PREDICTION GOES HERE --------------------
def make_prediction(df):

    #to be deleted
    csv_file_path = os.path.join(current_path, 'Compiled.csv')
    data = pd.read_csv(csv_file_path)
    data = data[['datetime','load', 'pressure_f', 'cloud_cov_f', 'temp_f',
       'wind_dir_f', 'wind_sp_f', 'date', 'month', 'hour', 'type_of_day', ' year']]
    data = data.rename(columns={'pressure_f':"pressure", 'cloud_cov_f': "cloud_cover", 'temp_f':"temperature",
        'wind_dir_f': "wind_direction", 'wind_sp_f': "wind_speed"})
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['date'] = pd.to_datetime(data['date'])
    
    # model = ElectricityForecaster()
    # data = model.transform_data_valid(data)
    # train, valid = model.split_data(data)
    # model.fit(train, True)
    # mae = model.predict_validation(valid)[2]
    # model.save_model()

    # preds, actuals, mae, mape, rmse, errors = model.predict_multiple(valid, True)

    model = joblib.load(os.path.join(current_path, '../ElectricityForecasterModel/ElectricityDemandForecaster.joblib'))
    predictions = model.predict(df)

    return predictions


"""
Cleans a single dataframe (file). It does so by checking for errors as well, such as wrong column names.
If there are no errors, then the method changes column names according to the developer's expectations, changes column datatypes, 
retrieves only the required columns (as there may be some additional and unnecessary columns), sorts the dataframe, and then removes
any null rows.
"""
def clean_df(df):
    valid = True
    response = "File Received Successfully! "
    # first columns are enough:
    if len(df.columns) < NUM_OF_COLUMNS:
        return df, False, "Please provide all required columns."

    # check if file has header first:
    has_header, response = check_df_has_header(df)
    if not has_header:
        return df, has_header, response

    # rename columns to expected requirements:
    df, valid, response = check_column_names(df)
    
    if not valid:
        return df, valid, response
     
    df = get_required_cols(df)


    # change data types:
    df, valid, response = change_column_datatypes(df)
    if not valid:
        return df, valid, response
    
    df = remove_nulls(df)

    return df, valid, response


"""
Cleans the entire data submitted. 
This includes checking for duplicate values, checking that the data's dates are valid (date gaps, etc), 
checking for missing values, outliers, adding additional columns before making predictions, sorting, etc.
"""
def clean_full_df(df):
    valid = True
    response = ""

    # check for other missing values throughout the dataframe
    df, response, valid = check_missing_values(df)

    df = check_duplicates(df)

    
    # if not, check valid date range (should be straight after the last prediction)
    valid = check_valid_dates(df)

    #In that case, there is a gap that should be handled
    if not valid:
        # instead, if not valid, then handle gap
        df, response, valid = handle_date_gap(df)

    # handle outliers:
    df = check_col_values(df)


    # change data types:
    df, valid, _ = change_column_datatypes(df)

    # add time features:
    df = add_features(df)

    # sort the df:
    df = sort_df(df)


    return df, valid, response


"""
sorts dataframe based onthe datetime column
"""
def sort_df(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by='datetime', ascending=True)
    return df


"""
remove any null rows
"""
def remove_nulls(df):
    df = df.dropna(how='all')
    return df

    
"""
checks and removes duplicates
"""
def check_duplicates(df):
    df = df.drop_duplicates(subset='datetime')
    df = df.reset_index(drop=True)
    return df 



"""
checks if the column names provided are correct. 
The program expects different variants of a column's name and stores it into a JSON file, which contains the column
pattern and the name to convert to.  
It uses regular expressions to see if the name meets the exepcted pattern, if so, it changes the name accordingly. 
Otherwise, if the column cannot be converted, that means the name is wrong and there is a user error. 
"""
def check_column_names(df):
    column_name_patterns = read_json_file('colNames.json')

    changes = 0
    valid = True
    response = ""

    for column in df.columns:
        # Iterate through the defined patterns
        for pattern, replacement in column_name_patterns.items():
            # Use re.search to check if the pattern matches the column name
            if re.search(pattern, column, re.IGNORECASE):
                df.rename(columns={column: replacement}, inplace=True)
                changes += 1
    
    if changes != NUM_OF_COLUMNS :
        valid = False
        response = "Please provide the correct columns. "

    return df, valid, response


"""
get only the required columns we are concerend with. 
For example, if retrieving the data from the database, then there could be an additional 'id' column that we do not want.
"""
def get_required_cols(df):

    colnames = read_txt_file('weather_vars.txt') + ['datetime']
    df = df[colnames]
    return df


"""
check the dataframe/file has column names. If there is not, then return an error message.
"""
def check_df_has_header(df):
    response = ""

    #use the json file which contains the column names and patterns.
    column_name_patterns = read_json_file('colNames.json')   
    column_name_patterns = list(column_name_patterns.keys())

    #if the first column of the dataframe matches any of the column names, that means there is a header.
    #otherwise, there is no header and/or columns are wrong
    has_header = any(re.search(
        pattern, df.columns[0], re.IGNORECASE) for pattern in column_name_patterns)
    if not has_header:
        response = "Please provide a file with the correct column names"

    return has_header, response


"""
changes the column's data types accordingly, by using a json file which contains the column name and its respective data type.
If there is a problem converting, then there is an error.
"""
def change_column_datatypes(df):

    data_types = read_json_file('forecastColTypes.json')

    # Iterate through the columns and their desired data types
    response = ""
    valid = True
    for column, dtype in data_types.items():
        try:
            # Attempt to convert the column to the desired data type
            df[column] = df[column].astype(dtype)

        except (ValueError, TypeError, KeyError):
            # Handle any potential errors (e.g., if conversion is not possible)
            response = f"Error converting '{column}' to {dtype}"
            valid = False
            
    return df, valid, response



"""
checks if the data submitted results in a date gap.
"""
def check_valid_dates(df):
    last_date = Pracpredictions.objects.latest('datetime').datetime
    desired_time_gap = timedelta(hours=1)  # Adjust the time gap as needed
    first_datetime_in_df = df['datetime'].min()
    valid = True
    if first_datetime_in_df - last_date > desired_time_gap:
        valid = False

    return valid


"""
converts data into a dataframe
"""
def get_df(data):
    data =  pd.DataFrame(data)
    data = sort_df(data)
    return data


"""
This method handles date gaps.
What this means is if a user requests for a prediction after a few days for example of the last prediction, then we have a date gap. 
To fix this, the missing dates (dates between last prediction and the current data) are retrieved, and the weather variables are
computed by taking in the median of the last few records and using them as the values for a given hour. 
The missing dates will be used to predict on and then saved as actual values. 
However, if the date gap was too large (in our case, 5 weeks), then the program does not accept that, as the gap is too large and would
heavily impact the accuracy of the model. 
Output: dataframe containing all the data from the last prediction to the end of the current file to be predicted on. 
"""
def handle_date_gap(df):
    response = ""
    valid = True
    column_names = read_txt_file('weather_vars.txt')
    last_date = Actuals.objects.latest('datetime').datetime
    #first sort just to make sure:
    df = sort_df(df)
    first_datetime_in_df = df['datetime'].min()
    desired_time_gap = timedelta(hours=1)
    missing_dates = []
    missing_dates_values = []

    if first_datetime_in_df - last_date > timedelta(weeks=GAP_LIMIT):
        response = f"Gap between received data and last prediction is too large. Please provide actual values or request to predict from {last_date + desired_time_gap}"
        valid = False
        return df, response, valid


    current_datetime = last_date + desired_time_gap
    #while we still have a gap, save the missing dates into the 'missing_dates' list
    while current_datetime < first_datetime_in_df:
        missing_dates.append(current_datetime)
        current_datetime += desired_time_gap

    # find how many days we want to go back by looking at the time difference:
    time_difference = (first_datetime_in_df - last_date).days
    # add another day to ensure all days get values from the database
    go_back = timedelta(days = time_difference + 1)

    # replace missing data by last record in the database
    for missing_date in missing_dates:
        row = [missing_date]
        previous_go_back = missing_date - go_back
        #get the data from the time difference to the current day we are filling in for, at the same hour
        data = get_df(Actuals.objects.filter(
            datetime__range=(previous_go_back, missing_date),
            hour=previous_go_back.hour).values()).drop(columns=['load']) #we dont want the load column
         
        for col in column_names:
            #replace with the median
            median = data[col].median()
            row.append(median)
        #this contains all the values, including load, which will be excluded by the model anyways
        missing_dates_values.append(row)

    gap_df = pd.DataFrame(missing_dates_values, columns=['datetime'] + column_names)

    gap_df = sort_df(gap_df)

    response = f"WARNING! you have not submitted actual values for previous days before {first_datetime_in_df}.\nSolution implemented is to make predictions for the missing days and use them as the actual values.\nAs a result, the prediction accuracy may be negatively impacted."

    df = combine_dfs([gap_df, df]) #combine the df containing the gap with the current dataframe
    df = sort_df(df) #sort again to be safe

    return df, response, valid


"""
Method that checks for outliers and/or wrong values.
In a JSON file, the appropriate value ranges for each weather variable and compares the provided data against.
"""
def check_col_values(df):
    #checks for outliers, wrong values, etc
    colvalues = read_json_file('colActualvalues.json')

    #the number of days we want to go back to retrieve data, compute the median, and replace wrong values with is 3.
    lastdate_db = Actuals.objects.latest('datetime').datetime
    lastdate_df = df.datetime.max()
    
    for column, (min_value, max_value) in colvalues.items():
        if column in df.columns:

            if lastdate_db > lastdate_df: #get data from database then
                start_date =lastdate_df - timedelta(days=3)
                end_date = lastdate_df
            else: #use data from the dataframe
                start_date = lastdate_db - timedelta(days=3)
                end_date = lastdate_db

            #get the data from the last 3 days
            nearest_records = Actuals.objects.filter(
                datetime__gte=start_date,
                datetime__lte=end_date,
            ).order_by('datetime').values_list(column, flat=True)

            #compute the median
            median_col = statistics.median(list(nearest_records))

            #replace outliers/wrong values with the median of each column.
            df.loc[(df[column] < min_value), column] =  median_col
            df.loc[(df[column] > max_value), column] =  median_col

    return df 


"""
combines a list of dataframes into a single dataframe
"""
def combine_dfs(dfs):
    df = pd.concat(dfs, axis=0)
    df.reset_index(drop=True, inplace=True)
    return df


"""
Method that checks for missing values. 
It checks the following:
1. if there are any NaT datetimes, then it replaces accordingly. 
2. checks for missing dates (there should be 48 hours). For missing dates/row, it retrieves the missing date, 
and computes the weather variables by computing interpolation. 
"""
def check_missing_values(df):
    valid = True
    response = ""

    column_names = read_txt_file('weather_vars.txt') 

    #First get rid of all NaN values:
    # First get rid of all NaN values:
    df = df.dropna(how='all')

    # Then check for NaN datetimes:
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

    df = sort_df(df) #sort the df just be safe
    df = df.reset_index(drop=True)


    #interpolate
    for col in column_names:
        df[col].interpolate(method='linear', limit_direction='both', inplace=True)

        # Interpolate in forward order across the column:
        df[col].interpolate(method ='linear', limit_direction ='forward', inplace=True)

    return df, response, valid


"""
Decomposes the datetime parameter into multiple parts, like date, month, hour and type of day
"""
def add_features(df):
    df['date'] = df['datetime'].apply(lambda x: x.date())
    df['month'] = df['datetime'].apply(lambda x: x.month)
    df['type_of_day'] = df['datetime'].apply(lambda x: x.isoweekday())
    df['hour'] = df['datetime'].apply(lambda x: x.hour)

    return df


"""
converts the file into a dataframe according to the file type.
"""
def handle_demand_prediction_file(f):

    df = pd.DataFrame()

    if f.name.endswith('.csv'):
        df = pd.read_csv(f)

    elif f.name.endswith('.xlsx'):
        df = pd.read_excel(f)

    return df


"""
Method responsible for plotting the data on the graph
input: actuals and predictions values as lists of dictionaries (since they had just been retrieved from the database)
output: html version of the plot
"""
def get_predict_plot(actuals,  predictions):

    actuals_df = get_df(actuals)
    preds_df = get_df(predictions)
    
 
    #combin ehte 
    combined = combine_dfs([actuals_df, preds_df])
    combined['datetime'] = pd.to_datetime(combined['datetime'], dayfirst=True)
    # combined.reset_index(drop=True, inplace=True)
    combined = combined.sort_values(by='datetime')

    #sets the colour of the line: first 48 hours are blue and second is red
    combined['colour'] = ['Past 48 Hours' if x <=
                          48 else 'Predictions' for x in range(len(combined))]

    fig = px.line(combined, x="datetime", y="load", color="colour",
                  title="Past 48 Hours of actual data vs Predictions")

    # Customize the legend labels
    fig.update_traces(line=dict(width=2.5))  # Adjust line width
    fig.update_layout(legend_title_text='Data Category')
    fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    plot = fig.to_html(full_html=False, include_plotlyjs=False)

    return plot


"""
Reads a text file by iterating through it. 
input: file name
output: list of values from each line of the text file
"""
def read_txt_file(filename):
    wordlist = []
    filename = "ReferenceData/" + filename
    filename = os.path.join(current_path, filename)
    with open(filename, 'r') as file:
        for line in file:
            # Remove leading and trailing whitespace and append the word to the list
            wordlist.append(line.strip())
    return wordlist


"""
reads a json file. 
input: filename of the required file 
output: the json file in the form of a dictionary
"""
def read_json_file(filename):
    filename = "ReferenceData/" + filename
    filename = os.path.join(current_path, filename)
    with open(filename, "r") as json_file:
        file = json.load(json_file)
    return file