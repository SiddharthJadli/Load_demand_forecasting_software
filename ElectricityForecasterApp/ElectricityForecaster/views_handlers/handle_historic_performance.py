from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from ..models import Actuals, Pracpredictions
from ..forms import PredictDemandForm
from ..forms import UploadActualsForm
import pandas as pd
import datetime
from django.core import serializers
from django.db.models import Count, Avg, Sum
from django.db.models.functions import TruncDay, TruncWeek, TruncMonth
from django.utils import timezone
import plotly.graph_objs as go
import plotly.express as px
import statsmodels.api as sm
from plotly.offline import plot
import plotly.colors as pc
from plotly.subplots import make_subplots
import os
import plotly.graph_objects as go
import numpy as np


"""
This file processes the 'Historic performance' page. 
It retrieves the required data from the database, computes the error measures, and plots the data using plotly graphs.
"""

DEPLOYMENT_DATE = '2021-01-15 08:00:00'

# methods called from views.py
def handle_historic_performance(request):
    context = {}

    #entire predictions data
    actuals_df, preds_df = get_data()
    if actuals_df is not None:
        df = get_merged_df(actuals_df, preds_df)
        plot = get_plot(df)
        context['plot'] = plot
        mae, mape, rmse = calculate_metrics(df)
        context['mae'] = mae
        context['mape'] = mape
        context['rmse'] = rmse

    #predictions data since deployment
    if Actuals.objects.filter(datetime=DEPLOYMENT_DATE).exists():
        actuals_df, preds_df = get_data(DEPLOYMENT_DATE)
        if actuals_df is not None:
            df = get_merged_df(actuals_df, preds_df)
            plot = get_plot(df)
            context['zoomedplot'] = plot
            mae, mape, rmse = calculate_metrics(df)
            context['zoomedmae'] = mae
            context['zoomedmape'] = mape
            context['zoomedrmse'] = rmse

    #predictions made since deployment
   
    return context


"""
Retrieves the data from the 'actuals' and 'pracpredictions' table
"""
def get_data(datetime=None):
    try:

        if Pracpredictions.objects.count() > 0:
            predictions = list(Pracpredictions.objects.values('datetime', 'load'))
            preds_df = get_df(predictions)

            if datetime: 
                date_from = datetime

            else:
                date_from = preds_df['datetime'].min()
            
            actuals = list(Actuals.objects.filter(
            datetime__gte=date_from).values('datetime', 'load'))

            actuals_df = get_df(actuals)

            return actuals_df, preds_df
    except Pracpredictions.DoesNotExist: #catches an error
        pass
    
    return None, None


"""
Generates the line plot
"""
def get_plot(df):

    fig = go.Figure()

    # Add traces for actuals and predicted data
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['actual'], mode='lines', name='Actuals'))
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['prediction'], mode='lines', name='Predicted', line=dict(width=2)))  # Adjust line width as needed

    fig.update_layout(legend=dict(x=0, y=1), height=700
                      )  # Position the legend as needed


    return fig.to_html(full_html=False, include_plotlyjs=False)

"""
Merges two dataframes together
"""
def get_merged_df(actuals, preds):
    actuals['load'] = pd.to_numeric(actuals['load'])
    preds['load'] = pd.to_numeric(preds['load'])
    actuals['datetime'] = pd.to_datetime(actuals['datetime'])
    preds['datetime'] = pd.to_datetime(preds['datetime'])

    #merge the dataframe based on the datetime, and get only the shared values
    df = pd.merge(actuals[['datetime', 'load']], preds[['datetime', 'load']], on='datetime', how='inner')
    df = df.rename(columns={'load_x': 'actual', 'load_y': 'prediction'})
    df = sort_df(df)

    return df 


"""
Computes the error measures (MAE, MAPE, and RMSE) from the dataframe using the actual and prediction columns
"""
def calculate_metrics(df, actual_col = 'actual', predicted_col = 'prediction'):
    # Calculate Mean Absolute Error (MAE)
    mae = np.abs(df[actual_col] - df[predicted_col]).mean()

    # Calculate Mean Absolute Percentage Error (MAPE)
    ape = np.abs((df[actual_col] - df[predicted_col]) / df[actual_col])
    mape = ape.mean() * 100  # Convert to percentage

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(((df[actual_col] - df[predicted_col]) ** 2).mean())

    return mae, mape, rmse



def get_df(data):
    data =  pd.DataFrame(data)
    data = sort_df(data)
    return data


def sort_df(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by='datetime', ascending=True)
    return df
