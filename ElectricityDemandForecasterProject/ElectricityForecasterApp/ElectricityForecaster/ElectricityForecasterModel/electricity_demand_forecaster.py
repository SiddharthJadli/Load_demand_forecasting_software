import psycopg2
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pingouin import partial_corr
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import pacf
import math, joblib, os, json
import xgboost as xgb
from datetime import datetime, timedelta
from ..models import Actuals,  Pracpredictions
from django.conf import settings
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind


class ElectricityForecaster:
    
    NUM_OF_STEPS = 48
    NUM_OF_MODELS = 48
    NUM_OF_LAGS = 168

    LAST_RETRAIN = datetime.strptime("2021-01-15 07:00:00", '%Y-%m-%d %H:%M:%S')
    

    def __init__(self):
        self.path =  os.path.dirname(os.path.abspath(__file__))
        self.exog = self.read_file("exog.txt")
        self.models = [xgb.XGBRegressor()  for _ in range(self.NUM_OF_MODELS)]

      
        
    def fit_and_predict(self, data):
        data = self.transform_data_valid(data)
        train, valid = self.split_data(data)
        self.fit(train)
        preds, actuals, mae, mape, rmse = self.predict_validation(valid)
        return preds, actuals, mae, mape, rmse
    
    def predict_validation(self, data, transformed=True):
        if not transformed:
            data = self.transform_data_valid(data)
        aes, apes, ses = [], [], []  
        preds = []
        actuals = []
        for i in range(self.NUM_OF_MODELS):

            x_columns = self.x_cols(i)
            y_columns = self.y_cols(i)

            valid_x = data[x_columns]
            valid_y = data[y_columns].values

            current_model = self.models[i]
            prediction = current_model.predict(valid_x)

            aes += list(self.absolute_errors(valid_y, prediction))
            apes += list(self.absolute_perc_errors(valid_y, prediction))
            ses += list((self.squared_errors(valid_y, prediction)))
            
            preds += prediction.tolist()
            actuals += list(valid_y)
            
        ##################  HERE  #######################
        #storing the latest 48 validation abs errors into the class for comparison
        
        # self.prev_errors['aes'] = aes[len(aes)-48: len(aes)]
        # self.prev_errors['apes'] = apes[len(apes)-48: len(apes)]
        # self.prev_errors['ses'] = ses[len(ses)-48: len(ses)]
            
        mae = self.get_mae(aes)
        mape = self.get_mape(apes)
        rmse = self.get_rmse(ses)
        
        print(f"mae: {mae}")
        print(f"mape: {mape}")
        print(f"rmse: {rmse}")

        return preds,actuals,  mae, mape, rmse
    

    def predict_multiple(self, data, transformed = False):
        aes, apes, ses = [], [], []  
        preds = []
        actuals = []
        errors = []
        if not transformed:
            data = self.transform_data(data)
        smaller_dfs = [data[i:i + 1] for i in range(0, len(data))]
        for i in range(len(smaller_dfs)):
             
            row = smaller_dfs[i]
            
            print(row)
            
            prediction, actual, ae, ape, se = self.predict_48_hours_validation(row, False)
            aes += ae
            apes += ape
            ses += se
            preds += prediction
            actuals += actual
            
            start_datetime = row.index.to_pydatetime()[0]
            end_datetime = start_datetime + timedelta(hours= self.NUM_OF_STEPS -1 )
            new_row = [start_datetime, end_datetime, self.get_mae(ae), self.get_mape(ape), self.get_rmse(aes)]
            errors.append(new_row)
            print(start_datetime)
            print(new_row)
            print(ae)
            print(ape)
            print(se)
            print("\n\n\n")

            
        mae = self.get_mae(aes)
        mape = self.get_mape(apes)
        rmse = self.get_rmse(ses)
        
        errors = pd.DataFrame(errors, columns=['start_datetime', 'end_datetime', 'mae', 'mape', 'rmse'])
        
        print(f"mae: {mae}")
        print(f"mape: {mape}")
        print(f"rmse: {rmse}")
        
        return preds,actuals,  mae, mape, rmse, errors
    

    def predict(self, data, transformed = False):
        preds = []

        smaller_dfs = self.get_smaller_dfs(data)
        for i in range(len(smaller_dfs)):
             
            row = smaller_dfs[i]

            print("before transformation: ")
            print(row)

            if not transformed:
                row = self.transform_data(row)
            
            print("After: ")
            print(row)
            
            print("predicting 48 hours")
            prediction= self.predict_48_hours(row)

            preds += prediction

            print(preds)
            print("\n\n\n")

        return preds
    
    
    def predict_48_hours(self,row):
        preds = []
        for i in range(self.NUM_OF_MODELS):

            x_columns = self.x_cols(i)

            valid_x = row[x_columns]

            current_model = self.models[i]
            prediction = current_model.predict(valid_x)

            preds += prediction.tolist()

        self.save_predictions_to_db(preds, row)
                
        return preds
    

    def predict_48_hours_validation(self,row, once=True):
        aes, apes, ses = [], [], []  
        preds = []
        actuals = []
        start_datetime = row.index.to_pydatetime()[0]
        for i in range(self.NUM_OF_MODELS):

            x_columns = self.x_cols(i)
            y_columns = self.y_cols(i)

            valid_x = row[x_columns]
            valid_y = row[y_columns].values

            current_model = self.models[i]
            prediction = current_model.predict(valid_x)
            preds += prediction.tolist()
            actuals += list(valid_y)
            
            aes += list(self.absolute_errors(valid_y, prediction))
            apes += list(self.absolute_perc_errors(valid_y, prediction))
            ses += list((self.squared_errors(valid_y, prediction)))
            
        mae = self.get_mae(aes)
        mape = self.get_mape(apes)
        rmse = self.get_rmse(ses)

        self.save_valid_data(row, actuals, preds)
        
        if not once:
            return preds, actuals, aes, apes, ses
        
        return preds, actuals, mae, mape, rmse
    
        
    def fit(self,data, transformed=True):
        if not transformed:
            data = self.transform_data_valid(data)
#         models_mp = joblib.Parallel(n_jobs=8)(
#     joblib.delayed(self.fit_on_model)(data, model_index) for model_index in range(self.NUM_OF_MODELS)
# )
        for i in range(self.NUM_OF_MODELS):
            self.fit_on_model(data, i)

        print("fitted")

            
    def fit_on_model(self, data, i):
        x_columns = self.x_cols(i)
        y_columns = self.y_cols(i)

        train_x = data[x_columns]
        train_y = data[y_columns]

        current_model = self.models[i]
        current_model.fit(train_x, train_y)
        
            
    def save_predictions_to_db(self, predictions, row):
        timestep = timedelta(hours=1)

        current_datetime = row.index.to_pydatetime()[0]

        print("data to save:")
        print(row)

        for i in range(len(predictions)):

            #save to predictions first:
            load = predictions[i]  # Assuming you have a list of load values
            # Try to get a record with the current datetime
            record, created = Pracpredictions.objects.get_or_create(datetime=current_datetime, defaults={'load': load})

            # If the record was created (datetime didn't exist), set the load value
            if created:
                print("created new prediction")
            else:
                print("updated existing prediction")

            record.load = load
            record.save()

            # save to actuals:
            # row/predictions should be 48 
            # row: datetime, lags, leads
            forecasts = self.get_lead_forecast_names(i)
            date_decomp = self.get_date_decomp(current_datetime)
            actuals = [current_datetime] + [load] + row[forecasts].values.tolist()[0] + date_decomp

            print("actual values to save:")
            print(actuals)

            
            datetime, load, pressure, temperature, cloud_cover, wind_direction, wind_speed, date, month, hour, type_of_day = actuals

            record, created = Actuals.objects.get_or_create(datetime=datetime,  defaults={'load': load})
            # if created:
            # else:
            #     # If the record already exists, update the values

            record.load = load
            record.pressure = pressure
            record.temperature = temperature
            record.cloud_cover = cloud_cover
            record.wind_direction = wind_direction
            record.wind_speed = wind_speed
            record.date = date
            record.month = month
            record.hour = hour
            record.type_of_day = type_of_day
            record.save()

            # Increment the datetime by an hour for the next iteration
            current_datetime += timestep


    def save_valid_data(self, row, actuals, preds):
        current_datetime = row.index.to_pydatetime()[0]
        timestep = timedelta(hours=1)

        for i in range(len(preds)):
            print(f"i: {i}")
            prediction = preds[i]

            print(f"prediction: {prediction}")
            record, created = Pracpredictions.objects.get_or_create(datetime=current_datetime, defaults={'load': prediction})

            if created:
                print("created new prediction")
            else:
                print("updated existing prediction")

            record.load = prediction
            record.save()

            actual = actuals[i]
            forecasts = self.get_lead_forecast_names(i)
            date_decomp = self.get_date_decomp(current_datetime)
            actual_row = [current_datetime] + [actual] + row[forecasts].values.tolist()[0] + date_decomp
            
            datetime, actual, pressure, temperature, cloud_cover, wind_direction, wind_speed, date, month, hour, type_of_day = actual_row

            record, created = Actuals.objects.get_or_create(datetime=datetime)
            if created:
                print("created new record")
            else:
                # If the record already exists, update the values
                print("updated existing record")

            record.load = actual
            record.pressure = pressure
            record.temperature = temperature
            record.cloud_cover = cloud_cover
            record.wind_direction = wind_direction
            record.wind_speed = wind_speed
            record.date = date
            record.month = month
            record.hour = hour
            record.type_of_day = type_of_day
            record.save()

            # Increment the datetime by an hour for the next iteration
            current_datetime += timestep




   
    def add_lags(self, df):

        #because the user may request to re-predict on dates that are already predicted on
        last_date_db = Actuals.objects.latest('datetime').datetime
        last_date_df = df['datetime'].min()

        print(f"last date in df: {last_date_df}")
        # if last_date_db > last_date_df:
        #     current_date = last_date_df
        # else:
        #     current_date = last_date_db
        current_date = last_date_df
        end_date = current_date - timedelta(hours=1)
        start_date = current_date - timedelta(days=7)

        print(f"start date: {start_date}")
        print(f"end date: {end_date}")

        rows = Actuals.objects.filter(datetime__range=(start_date, end_date))
        db_data = self.get_df(list(rows.values())).drop(columns=['id'])

        full_df = pd.concat([db_data, df]).sort_values(by='datetime')

        full_df = full_df.reset_index(drop=True)
        full_df['datetime'] = pd.to_datetime(full_df['datetime'])
        full_df = self.sort_df(full_df)

        print("lags: ")
        print(full_df)

        return full_df

    def get_date_decomp(self, datetime):
        date = datetime.date()
        month = datetime.month
        hour = datetime.hour
        type_of_day = datetime.isoweekday()

        return [date, month, hour, type_of_day]


    def transform_data(self, data):

        #add the lags from the db
        data = self.add_lags(data)

        data = self.change_column_datatypes(data)

        #cyclical encoding first
        data = self.encode(data, 'hour', 24)
        data = self.encode(data, 'month', 12)
        data = self.encode(data, 'type_of_day', 31)

        data = self.remove_extras(data)

        data = self.get_lag_lead(data)

        #get 8 am hour rows
        data = self.condense_data(data)

        # set datetime as index:
        data = self.set_datetime_index(data)
        
        return data
    

    def transform_data_valid(self, data):

        data = self.change_column_datatypes(data)

        #cyclical encoding first
        data = self.encode(data, 'hour', 24)
        data = self.encode(data, 'month', 12)
        data = self.encode(data, 'type_of_day', 31)
    
        data = self.remove_extras(data)      

        #get the lags and leads
        data = self.get_lag_lead_for_valid(data)
        
        #get 8 am hour rows
        data = self.condense_data_valid(data)
        
        return data
    
    
    def absolute_errors(self, actuals, predicted):
        ae = np.abs(actuals-predicted)
        return ae

    def squared_errors(self, actuals, predicted):
        se = np.square(actuals-predicted )
        return se

    def absolute_perc_errors(self, actuals, predicted):
        ape = ((np.abs(actuals-predicted))/ actuals) * 100
        return ape
    
    def get_mae(self, aes):
        return np.mean(aes)

    def get_rmse(self, ses):
        return np.sqrt(np.mean(ses))

    def get_mape(self, apes):
        return np.mean(apes)

    #get specific lead forecast column names from whole df for each model
    def get_lead_forecast_names(self, i):
        cols = self.exog
        lead_cols = []
        for c in cols: 
            lead_cols.append(f'{c}_lead_{i}')
        return lead_cols


    def x_cols(self, i):
        # Specify the text file
        word_list = self.read_file('train_cols.txt')

        word_list += self.get_lead_forecast_names(i)

        return word_list


    def y_cols(self, i):
        return f'load_lead_{i}'


    def split_data(self, data, test_size=0.1):
        return np.split(data, [int((1 - test_size) * data.shape[0]) + 1])  
    

    def condense_data_valid(self, data):
        desired_start_time = '08:00:00'
        desired_end_time = '08:00:00'
        start_datetime = pd.to_datetime(desired_start_time)
        end_datetime = pd.to_datetime(desired_end_time)
        data = data.between_time(start_datetime.time(), end_datetime.time())
        return data
        

    def condense_data(self, data):
        desired_start_time = '08:00:00'
        desired_datetime = pd.to_datetime(desired_start_time).time()
        filtered_df = data[data['datetime'].apply(lambda x: pd.to_datetime(x).time() == desired_datetime)]
        return filtered_df
    

    def remove_extras(self, data):
        to_remove = ['date','month', 'hour', 'type_of_day', 'date']
        data = data.drop(to_remove, axis=1)
        data = data.rename(columns = {'type_of_day_sin': 'day_sin', 
                            'type_of_day_cos': 'day_cos'})
        return data
    

    def get_lag_lead_for_valid(self, data, num_lags=168, forward_pred=48): 
        cols = self.exog + ['load']
        #for load
        for i in range(1, self.NUM_OF_LAGS + 1):
            data[f'load_lag_{i}'] = data['load'].shift(i) 

        #for lagged weather variables
        for c in self.exog:
            for i in range(1, self.NUM_OF_LAGS+1):
                data[f'{c}_lag_{i}'] = data[c].shift(i)

        for i in range(0, self. NUM_OF_STEPS):    
            data[f'load_lead_{i}'] = data['load'].shift(-i)

        #weather
        for c in self.exog:
            for i in range(0,self. NUM_OF_STEPS):
                data[f'{c}_lead_{i}'] = data[c].shift(-i)

        print("showing null rows:")
        print(data[data.isnull().any(axis=1)])


        data.dropna(inplace = True)    #drop nulls
        data = data.set_index('datetime')   #set index as datetime
        data = data.asfreq('H')

        data = data.drop(cols, axis=1)

        return data
    

    def get_lag_lead(self, data, num_lags=168, forward_pred=48): 
        cols = self.exog
        to_remove = cols + ['load']
        #for lagged load
        for i in range(1, self.NUM_OF_LAGS + 1):
            data[f'load_lag_{i}'] = data['load'].shift(i) 

        #for lagged weather variables
        for c in self.exog:
            for i in range(1, self.NUM_OF_LAGS+1):
                data[f'{c}_lag_{i}'] = data[c].shift(i)


        #weather
        for c in self.exog:
            for i in range(0,self.NUM_OF_STEPS):
                data[f'{c}_lead_{i}'] = data[c].shift(-i)

        data = data.drop(to_remove, axis=1)

        print("showing null rows:")
        print(data[data.isnull().any(axis=1)])
     
        data.dropna(inplace = True)    #drop nulls
        

        return data
    
    def set_datetime_index(self, data):
        data = data.set_index('datetime')   #set index as datetime
        data = data.asfreq('H')
        return data


    def encode(self, data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
        return data
    

    def get_smaller_dfs(self, df):
        sub_dataframes = []
        chunk_size = self.NUM_OF_STEPS
        step_size = chunk_size // 2

        # Iterate through the original DataFrame and split it into chunks
        for i in range(0, len(df) - chunk_size + 1, step_size):
            chunk = df.iloc[i:i + chunk_size]
            
            # Check if the chunk size is equal to the specified size (48 rows)
            if len(chunk) == chunk_size:
                sub_dataframes.append(chunk)
        return sub_dataframes


    def change_column_datatypes(self, df):

        data_types = self.read_json_file('colTypes.json')

        for column, dtype in data_types.items():
            try:
                # Attempt to convert the column to the desired data type
                df[column] = df[column].astype(dtype)

            except (ValueError, TypeError, KeyError):
                # Handle any potential errors (e.g., if conversion is not possible)
                response = f"Error converting '{column}' to {dtype}"
                valid = False
            
        return df

    
    def read_file(self, filename):
        # Initialize an empty list to store the words
        word_list = []
        filename = "ReferenceData/" + filename
        # Open the file and read its contents line by line
        filepath = os.path.join(self.path, filename)
        with open(filepath, 'r') as file:
            for line in file:
                # Remove leading and trailing whitespace and append the word to the list
                word_list.append(line.strip())
        return word_list
    


    def read_json_file(self, filename):
        filename = "ReferenceData/" + filename
        filename = os.path.join(self.path, filename)
        with open(filename, "r") as json_file:
            file = json.load(json_file)
        return file
    
    def save_model(self):
        filename = "ElectricityDemandForecaster.joblib"
        filepath = os.path.join(self.path, filename)
        joblib.dump(self, filepath)

    
    def sort_df(self, df):
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime', ascending=True)
        return df
    


    def get_df(self, data):
        data =  pd.DataFrame(data)
        data = self.sort_df(data)
        return data
    

    def retrain_required(self, Current_Date, drift_based=True):
                
        required = False
        
        if drift_based:
            #data drift code
            prev_prediction_end = self.LAST_RETRAIN  #last day of training
            prev_prediction_start = self.LAST_RETRAIN - timedelta(hours=47)   #48 hours before

            prev_preds_query = list(Pracpredictions.objects.filter(datetime__range=(prev_prediction_start, prev_prediction_end)).values('datetime', 'load'))
            prev_actuals_query = list(Actuals.objects.filter(datetime__range=(prev_prediction_start, prev_prediction_end)).values('datetime', 'load'))


            start_datetime = Current_Date 
            end_datetime = Current_Date + timedelta(hours=47)  

            curr_preds_query = list(Pracpredictions.objects.filter(datetime__range=(start_datetime, end_datetime)).values('datetime', 'load'))
            curr_actuals_query = list(Actuals.objects.filter(datetime__range=(start_datetime, end_datetime)).values('datetime', 'load'))

            prev_preds = []
            prev_actuals = []
            for val in prev_preds_query:
                prev_preds.append(float(val['load']))

            for val in prev_actuals_query:
                prev_actuals.append(float(val['load']))


            curr_preds = []
            curr_actuals = []
            for val in curr_preds_query:
                curr_preds.append(float(val['load']))

            for val in curr_actuals_query:
                curr_actuals.append(float(val['load']))


            print("actuals:")
            print(curr_actuals)

            print("predictions:")
            print(curr_preds)

            # preds = list(Pracpredictions.objects.filter(datetime__range=(start_datetime, end_datetime)).values_list('load', flat=True))
            # actuals = list(Actuals.objects.filter(datetime__range=(start_datetime, end_datetime)).values_list('load', flat=True))

            previous_errors = {
                'aes': np.abs(np.subtract(prev_actuals,prev_preds)),
                'apes': ((np.abs(np.subtract(prev_actuals,prev_preds)))/prev_actuals) * 100,
                'ses': np.square(np.subtract(prev_actuals,prev_preds))
            }

            current_errors = {
                'aes': np.abs(np.subtract(curr_actuals,curr_preds)),
                'apes': ((np.abs(np.subtract(curr_actuals,curr_preds)))/curr_actuals) * 100,
                'ses': np.square(np.subtract(curr_actuals,curr_preds))
            }

            #perform t-tests
            p_val_aes = ttest_ind(previous_errors['aes'], current_errors['aes'])[1]
            p_val_apes = ttest_ind(previous_errors['apes'], current_errors['apes'])[1]
            p_val_ses = ttest_ind(previous_errors['ses'], current_errors['ses'])[1]

            
            error_drifts = [p_val_aes < 0.05, p_val_apes < 0.05, p_val_ses < 0.05]
            
            #check that new errors are better than old errors
            prev_mae = self.get_mae(previous_errors['aes'])
            prev_mape = self.get_mape(previous_errors['apes'])
            prev_rmse = self.get_rmse(previous_errors['ses'])

            curr_mae = self.get_mae(current_errors['aes'])
            curr_mape = self.get_mape(current_errors['apes'])
            curr_rmse = self.get_rmse(current_errors['ses'])

            print("LAST RETRAIN DATE: ",self.LAST_RETRAIN)
            print(f"previous mae: {prev_mae}")
            print(f"current mae: {curr_mae}")


            #if any of the errors are lower than the previous error
            if curr_mae > prev_mae or curr_mape > prev_mape or curr_rmse > prev_rmse:
                if error_drifts.count(True) >= error_drifts.count(False):
                    
                    return True      

        print(f"current date: {Current_Date}")
        print(f"last retrained date: {self.LAST_RETRAIN}")
        delta = Current_Date - self.LAST_RETRAIN
        
        if delta.days >= 14:
            print("its been 2 weeks bro")
            return True
                
        return False
            
    
    def retrain(self, data, current_date):
        
        print('### RETRAINING MODEL... ###')
        
        new_forecaster = ElectricityForecaster()
         
        data = self.transform_data_valid(data)


        split = len(data) - 4
        train, valid = data[split-90:split], data[split:]
 
        current_mae = self.predict_validation(valid)[2]

        print(f"current mae: {current_mae}")
        
        new_forecaster.fit(train)
        new_mae = new_forecaster.predict_validation(valid)[2]

        print(f"new mae: {new_mae}")
        
        
        if new_mae < current_mae:
            print("\nRetraining Successful \nOld MAE = {} \nNew MAE = {}".format(current_mae,new_mae))
            self.models = new_forecaster.models 
            self.LAST_RETRAIN = current_date


    def check_for_retrain(self, current_date):

        print("checking for retraining")

        print(f"current date: {current_date}")

        if self.retrain_required(current_date):

            #get the last date inserted:
            last_date = Actuals.objects.latest('datetime').datetime
            #data 
            date_threshold = last_date - timedelta(days=184)
            data = Actuals.objects.filter(datetime__gte=date_threshold).order_by('datetime').values()

            data = self.get_df(data)

            self.retrain(data, current_date)

            self.save_model()
        
        # print(f"new retrained date: {self.LAST_RETRAIN}")


