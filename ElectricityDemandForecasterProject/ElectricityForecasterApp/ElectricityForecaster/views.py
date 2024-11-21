from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .forms import PredictDemandForm
from .forms import UploadActualsForm
import pandas as pd
import datetime
from django.utils import timezone
import os
from .views_handlers.predict_demand_handler import handle_predict_demand
from .views_handlers.upload_actuals_handler import handle_upload_actuals
from .views_handlers.handle_insights import handle_insights
from .views_handlers.handle_historic_performance import handle_historic_performance
from .views_handlers.handle_about_us import handle_aboutus
from .models import Actuals, Pracpredictions
import tempfile

desired_timezone = 'Australia/Sydney'

"""
this is the file where all the views' methods go. 
Each method is responsible for a specific page on the interface.
"""


def about_us(request):
    return handle_aboutus(request)


"""
Handles the forecaster page, including bith tabs
"""
def forecaster(request):
    predict_form = PredictDemandForm()
    actuals_form = UploadActualsForm()
    context = {}
    context['active_tab'] = '#predictdemand'
    context['last_predicted_date'] = Pracpredictions.objects.latest('datetime').datetime
    context['last_actual_date'] = Actuals.objects.latest('datetime').datetime

    #if the method is post then process the form according to the tab in which the form was submitted.
    if request.method == 'POST':
        tab = request.POST.get('tab')

        if tab == 'predictdemand':
            predict_form, more_context = forecaster_predict(request)
            context.update(more_context)

        elif tab == 'uploadactuals':
            context['active_tab'] = '#uploadactuals'
            actuals_form, more_context = forecaster_upload_actuals(request)
            context.update(more_context)

        context['uploaded'] = True
    else:
        context['predict_message'] = ""
        context['actual_message'] = ""
        context['uploaded'] = False

    #pass the forms back
    context['predict_demand_form'] = predict_form
    context['upload_actuals_form'] = actuals_form

    return render(request, 'forecaster.html', context)

def forecaster_predict(request):
    return handle_predict_demand(request)


def forecaster_upload_actuals(request):
    return handle_upload_actuals(request)


def historic_performance(request):
    context = handle_historic_performance(request)
    return render(request, 'historic.html', context)


def insights(request):

    context = handle_insights(request)

    return render(request, 'insights.html', context)

