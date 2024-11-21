from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from ..models import Actuals
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


"""
This file processes the 'Insights' page. 
It generates all the plots displayed on the page, which are all created using the plotly library and some uses Apexcharts using 
Javascript inside the html template. 
"""


def handle_insights(request):

    # get the data
    context = {}

    #get the columns from the database separately
    full_load = list(Actuals.objects.order_by('datetime').values_list('load', flat=True))
    full_dates = list(Actuals.objects.order_by('datetime').values_list('datetime', flat=True))
    # pressure, cloud_cover, humidity, temperature, wind_speed, wind_direction
    temperature = list(Actuals.objects.order_by('datetime').values_list('temperature', flat=True))
    cloud_cover = list(Actuals.objects.order_by('datetime').values_list('cloud_cover', flat=True))
    wind_speed = list(Actuals.objects.order_by('datetime').values_list('wind_speed', flat=True))
    wind_direction = list(Actuals.objects.order_by('datetime').values_list('wind_direction', flat=True))
    pressure = list(Actuals.objects.order_by('datetime').values_list('pressure', flat=True))

    #pass the plots into the context
    context['monthly_data'] = monthly_data()
    context['daily_data'] = daily_data()
    context['weekly_data'] = weekly_data()
    context['complete_data'] = get_full_data_plot(full_dates, full_load)
    context['corr_matrix'] = get_corr_matrix(load=full_load, temperature=temperature,
                                             cloud_cover=cloud_cover, wind_speed=wind_speed,
                                             wind_direction=wind_direction, pressure=pressure)
    context['corr_scatter'] = get_corr_scatter(load=full_load, temperature=temperature,
                                               cloud_cover=cloud_cover,  wind_speed=wind_speed,
                                               wind_direction=wind_direction, pressure=pressure)
    context['pacf_plots'] = get_acf_pacf(full_load)
    context['summary'] = get_data_summary(load=full_load, temperature=temperature,
                                          cloud_cover=cloud_cover,  wind_speed=wind_speed,
                                          wind_direction=wind_direction,  pressure=pressure)
    context['bar_chart'] = get_bar_chart_corrs(load=full_load, temperature=temperature,
                                          cloud_cover=cloud_cover,  wind_speed=wind_speed,
                                          wind_direction=wind_direction,  pressure=pressure)

    return context


#takes in an unknown set of data and converts it into a single database.
def get_data_df(**kwargs):
    data = {}
    for name, value in kwargs.items():
        data[name] = value

    df = pd.DataFrame(data)

    return df

"""
generates a bar plot of the correlations between load and each of the weather variables
"""
def get_bar_chart_corrs(**kwargs):
    df = get_data_df(**kwargs)

    correlations = df.corr()['load'].drop('load')  # Exclude 'load' correlation with itself

    # Create a bar chart
    fig = px.bar(
        x=correlations.index,
        y=correlations.values,
        labels={'x': 'Weather Variable', 'y': 'Correlation with Load'},
        title='Correlation Between Load and Weather Variables',
    )

    fig.update_layout(xaxis_tickangle=-0)
    fig.update_layout(height=600) 

    return fig.to_html(full_html=False, include_plotlyjs=False)




"""
generates a correlation matrix between all variables
"""
def get_corr_matrix(**kwargs):

    df = get_data_df(**kwargs)

    correlation_matrix = df.corr()

    heatmap_data = [go.Heatmap(
        z=correlation_matrix.values.tolist(),
        x=correlation_matrix.columns.tolist(),
        y=correlation_matrix.columns.tolist(),
        colorscale='YlOrRd',
    )]

    layout = {
        'title': 'Correlation Matrix Heatmap',
        'xaxis': {'title': 'Variables'},
        'yaxis': {'title': 'Variables'},
    }

    fig = go.Figure(data=heatmap_data, layout=layout)

    # Convert the figure to HTML
    corr_matrix = plot(fig, output_type='div', include_plotlyjs=False)

    return corr_matrix


"""
generates a correlation scatter plot between load and all weather variables
"""
def get_corr_scatter(**kwargs):

    df = get_data_df(**kwargs)

    # Create a scatter plot for the initial variable (e.g., temperature)
    fig1 = make_subplots(rows=1, cols=1)
    scatterPlots = []
    scatter = go.Scatter(
        # Change 'temperature' to the desired weather variable
        x=df['temperature'],
        y=df['load'],
        mode='markers',
        name='Correlation',
    )
    scatterPlots.append(scatter)
    fig1.add_trace(scatter)
    # Create buttons for switching between weather variables
    buttons = []
    scatter_plots = []
    weather_variables = ['temperature', 'wind_speed', 'cloud_cover', 'wind_direction', 'pressure']
    for variable in weather_variables:
        scatter = go.Scatter(
            x=df[variable],
            y=df['load'],
            mode='markers',
            name='Correlation',
        )
        scatterPlots.append(scatter)
        button = dict(
            label=variable.capitalize(),
            method='update',
            args=[{'x': [df[variable]], 'y': [df['load']]}],
        )
        # fig1.add_trace(scatter)
        scatter_plots.append(scatter)
        buttons.append(button)
    # Add buttons to the layout
    updatemenu = list([dict(
        type='buttons',
        showactive=True,
        buttons=buttons)])
    fig1.update_layout(
        updatemenus=updatemenu,
        title='Scatter plots of weather variables',
        xaxis_title='Weather variable',
        yaxis_title='Load',
    )
    # # Convert the figure to HTML
    corr_scatter = fig1.to_html(full_html=False, include_plotlyjs=False)

    return corr_scatter


"""
generetes a statistical summary table of the data
"""
def get_data_summary(**kwargs):
    df = get_data_df(**kwargs)

    #convert the columns' datatypes to numeric so all statistics show
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])

    summary = df.describe()

    return summary


"""
retrieves data from the database and calculates the average for each day type.
"""
def weekly_data():
    # Weekly aggregation
    weekly_data = list(Actuals.objects.filter(type_of_day__isnull=False).values(
        'type_of_day').annotate(avg_load=Avg('load')).order_by('type_of_day'))

    return weekly_data



"""
retrieves data from the database and calculates the average for each hour.
"""
def daily_data():
    # daily aggregation
    daily_data = list(Actuals.objects.filter(hour__isnull=False).values(
        'hour').annotate(avg_load=Avg('load')).order_by('hour'))
    return daily_data



"""
retrieves data from the database and calculates the average for each month.
"""
def monthly_data():
    # Monthly aggregation
    monthly_data = list(Actuals.objects.filter(month__isnull=False).values(
        'month').annotate(avg_load=Avg('load')).order_by('month'))
    return monthly_data



"""
Generates the autocorrelation and partial-autocorrelation plots
"""
def get_acf_pacf(data):
    acf = sm.tsa.acf(data, nlags=168)
    pacf = sm.tsa.pacf(data, nlags=168)
    fig1 = make_subplots(rows=1, cols=2, subplot_titles=(
        "Autocorrelation Function (ACF)", "Partial Autocorrelation Function (PACF)"))
    # Add ACF trace
    fig1.add_trace(go.Scatter(x=list(range(len(acf))), y=acf,
                   mode='markers+lines', name='ACF'), row=1, col=1)
    # Add PACF trace
    fig1.add_trace(go.Scatter(x=list(range(len(pacf))), y=pacf,
                   mode='markers+lines', name='PACF'), row=1, col=2)
    # Update layout
    fig1.update_layout(title='ACF and PACF Plots', xaxis_title='Lag',
                       yaxis_title='Correlation', showlegend=True)
    # Convert the figure to HTML
    plots = fig1.to_html(full_html=False, include_plotlyjs=False)

    return plots

"""
Generates a line plot of the entire data
"""
def get_full_data_plot(dates, load):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=load,
                             mode='lines', name='Seasonal Component')).update_layout(
        title="Full data",
        xaxis_title="Date",
        yaxis_title="Load",

        xaxis=dict(showline=True, showgrid=False),
        yaxis=dict(showline=True, showgrid=False)
    )

    return fig.to_html()
