from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from ..models import Actuals, Pracpredictions
from ..forms import UploadActualsForm
import pandas as pd
import os, json, re, statistics, joblib
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px


"""
This file processes the 'upload actual values' tab. 
It cleans the data,checks for errors, saves the data inton the database, and asks the model to check for retraining.
"""

#some global variables required for the methods
NUM_OF_COLUMNS = 7
NUM_OF_ROWS = 48
current_path = os.path.dirname(os.path.abspath(__file__))


# method called in views.py
def handle_upload_actuals(request):
    context = {}
    valid = True
    response = "File Received.\n"
    record = UploadActualsForm(request.POST, request.FILES) #get the form record
    if record.is_valid():
        # response, valid = determine_if_file_valid(request.FILES['file'])
        uploaded_files = request.FILES.getlist('files')
        dfs = [] #list that stores all the files submitted as dataframes, after cleaning 
        for file in uploaded_files:
            # Check if the file type is CSV or Excel (XLSX or XLS)

            valid, new_response = form_file_type_correct(file)
            if not valid:
                context['actual_message'] = new_response
                return record, context
            
            
            #convert the file into a dataframe based on the file type
            df = handle_actuals_file(file)

            #handle the individual file, checking for errors and making some changes like column names
            df, valid, new_response = clean_df(df)
            response += new_response

            #if df has no errors, append and continue
            if valid:
                dfs.append(df)
            else:
                break

        if valid:
            #combine the files as a single dataframe
            df = combine_dfs(dfs)

            #if the combined dataframe has more than 48 rows, there must be an error
            if len(df) > NUM_OF_ROWS:
                response = f"The data submitted is too large. It should contain up to {NUM_OF_ROWS} hours only."
                valid = False
            else:

                #clean the entire dataframe, looking for errors, missing values, etc
                df, valid, new_response = clean_full_df(df)
                response += new_response

                if valid:

                    #everything is good, save the data to the database
                    save_actuals(df)

                    #get the required dates and data from the database tables so they can be displayed on the plot
                    #as well as to get the error measures to be displayed
                    from_date = df.datetime.min()
                    to_date = df.datetime.max()
                    predictions_df = get_pred_data_from_db(from_date, to_date)
                    actuals = get_actual_data_from_db(from_date, to_date)
                    merged = get_merged_df(actuals, predictions_df)
                    mae, mape, rmse = calculate_metrics(merged)
                    context['mae'] = mae
                    context['mape'] = mape
                    context['rmse'] = rmse
   

                    # ge tthe plot to be displayed
                    context['actual_plot'] = get_actual_plot(actuals, predictions_df)

        context['actual_message'] = response
        context['valid'] = valid

    return record, context


#sort the dataframe based on datetime
def sort_df(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by='datetime', ascending=True)
    return df

"""
convert data into a dataframe.
input data is mostly from the database where it is in the form of a list of dictionaries.
"""
def get_df(data):
    data =  pd.DataFrame(data)
    data = sort_df(data)
    return data


#checks if the file type is correct
def form_file_type_correct(file):
    file_name = file.name.lower()
    if not file_name.endswith(('.csv', '.xlsx', '.xls')):
        return False, 'Unsupported file type. Please upload a CSV or Excel file.'
    return True, ""


#get the data from the predictions table based on a date range
def get_pred_data_from_db(from_date, to_date):
    try:
        data = list(Pracpredictions.objects.filter(
                datetime__range=(from_date, to_date)).values())
        df = get_df(data)
        return df
    except Pracpredictions.DoesNotExist:
        return None
    

#get the data from the actuals table based on a date range
def get_actual_data_from_db(from_date, to_date):
    data = list(Actuals.objects.filter(
            datetime__range=(from_date, to_date)).values())
    df = get_df(data)
    return df


"""
converts the file into a dataframe according to the file type.
"""
def handle_actuals_file(f):

    df = pd.DataFrame()

    if f.name.endswith('.csv'):
        df = pd.read_csv(f)

    elif f.name.endswith('.xlsx'):
        df = pd.read_excel(f)

    return df



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
    
    df = sort_df(df)
    
    df = remove_nulls(df)

    return df, valid, response


"""
Cleans the entire data submitted. 
This includes checking for duplicate values, checking that the data's dates are valid (not submitting data for days not predicted yet), 
checking for missing values, outliers, adding additional columns before saving to the db, sorting, etc.
"""
def clean_full_df(df):
    valid = True
    response = ""

    df = sort_df(df)

    df = check_duplicates(df)

    valid, response = check_valid_dates(df)
    if not valid:
        return df, valid, response
        
    df, valid, response = check_missing_values(df)


    df = check_col_values(df)

    df = add_features(df)

    df = sort_df(df)

    return df, valid, response


"""
combines a list of dataframes into a single dataframe
"""
def combine_dfs(dfs):
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)
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
Method that checks for missing values. 
For actuals files, we only want to check for missing dates. 
Howeover, if there are missing rows and/or cells for weather variables, then it does not handle that.
The reasons for that are:
1. Our program only cares about weather forecasts, so it does not store the actual weather variables.
2. If there are missing load values, then we do not want to perform functions such as interpolation or taking in the 
    median of the last few days. This is because by default, predictions are also saved into the actuals table, so there should
    already be a load value for those dates (since actuals after submitted after predicting), and it is more preferable to store
    the predicted load as the actual load instead of imputing the missing values.
"""
def check_missing_values(df):
    valid = True
    response = ""

    column_names = read_txt_file('uicolnames.txt')

    #First get rid of all NaN values:
    df = df.dropna(how='all')

    
    # Then check for NaN datetimes:
    start_datetime = df['datetime'].min().replace(hour=8, minute=0, second=0)
    end_datetime = start_datetime + timedelta(hours=47)

    for i in range(len(df)):
        if pd.isna(df.at[i, 'datetime']):
            df.at[i, 'datetime'] = start_datetime + timedelta(hours=i)

    return df, valid, response


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
    
    if changes != NUM_OF_COLUMNS:
        valid = False
        response = "Please provide all the correct columns. "

    return df, valid, response


"""
changes the column's data types accordingly, by using a json file which contains the column name and its respective data type.
If there is a problem converting, then there is an error.
"""
def change_column_datatypes(df):

    data_types = read_json_file('colTypes.json')

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


def check_valid_dates(df):
    last_date = Actuals.objects.latest('datetime').datetime
    try:
        last_predicted_date = Pracpredictions.objects.latest('datetime').datetime
    except Pracpredictions.DoesNotExist:
        last_predicted_date = df.datetime.min()
    desired_time_gap = timedelta(hours=1)  
    valid = True
    response = ""
    first_datetime_in_df = df['datetime'].max()
    print(f"time difference: {first_datetime_in_df -  last_date }")
    if first_datetime_in_df > last_predicted_date:
        valid = False
        response = f"Please request for a prediction from {first_datetime_in_df} first, then upload the actual values."

    return valid, response



"""
get only the required columns we are concerend with. 
For example, if retrieving the data from the database, then there could be an additional 'id' column that we do not want.
"""
def get_required_cols(df):
    colnames = read_txt_file('uicolnames.txt')
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
Decomposes the datetime parameter into multiple parts, like date, month, hour and type of day
"""
def add_features(df):
    df['date'] = df['datetime'].apply(lambda x: x.date())
    df['month'] = df['datetime'].apply(lambda x: x.month)
    df['type_of_day'] = df['datetime'].apply(lambda x: x.isoweekday())
    df['hour'] = df['datetime'].apply(lambda x: x.hour)
    df['year'] = df['datetime'].apply(lambda x: x.year)

    return df


"""
Saves the data into the database's 'actuals' table. 
It iterates through the dataframe and:
1. checks if the datetime value already exists in the database or not. It only updates the load value if there is. 
    This is because by default, predictions are also saved into the actuals table, so there should
    already be a load value for those dates (since actuals after submitted after predicting).
"""
def save_actuals(df):
    # assuming we have the cleaned data with all the variables:
    for index, row in df.iterrows():
        # Retrieve or create a record based on the datetime column
        record, created = Actuals.objects.get_or_create(datetime=row['datetime'])
        
        if created:
            print("created new record")
        else: 
            print("Updated existing record")
            # Update only the load as we want to keep the forecast values
            if row['load'] != None:
                record.load = row['load']
            print(record.load)
            # Save the record to the database
            record.save()

    # get model to check for retraining:
    check_model_for_retrain(df.datetime.min())



def check_model_for_retrain(current_date):
    # get model to check for retraining:
    model = joblib.load(os.path.join(current_path, '../ElectricityForecasterModel/ElectricityDemandForecaster.joblib'))
    model.check_for_retrain(current_date)



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



def calculate_metrics(df, actual_col = 'actual', predicted_col = 'prediction'):
    # Calculate Mean Absolute Error (MAE)
    mae = np.abs(df[actual_col] - df[predicted_col]).mean()

    # Calculate Mean Absolute Percentage Error (MAPE)
    ape = np.abs((df[actual_col] - df[predicted_col]) / df[actual_col])
    mape = ape.mean() * 100  # Convert to percentage

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(((df[actual_col] - df[predicted_col]) ** 2).mean())

    return mae, mape, rmse



def get_merged_df(actuals, preds):
    actuals['load'] = pd.to_numeric(actuals['load'])
    preds['load'] = pd.to_numeric(preds['load'])
    actuals['datetime'] = pd.to_datetime(actuals['datetime'])
    preds['datetime'] = pd.to_datetime(preds['datetime'])

    df = pd.merge(actuals[['datetime', 'load']], preds[['datetime', 'load']], on='datetime', how='inner')
    df = df.rename(columns={'load_x': 'actual', 'load_y': 'prediction'})
    df = df.sort_values(by='datetime', ascending=True)

    return df 


def get_actual_plot(actuals,  predictions):
 
    combined = actuals
    combined['predicted'] = predictions['load']


    fig = px.line(combined, x="datetime", y=['load', 'predicted'], 
                  title="Actual vs Predicted Load")

    # Customize the legend labels
    fig.update_traces(line=dict(width=2.5))  # Adjust line width
    fig.update_layout(legend_title_text='Data Category')
    fig.update_layout(xaxis_title='Datetime',
    yaxis_title='Load', 
    width=1000,  # Adjust width as needed
    height=500, 
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    plot = fig.to_html(full_html=False, include_plotlyjs=False)

    return plot