from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from ..models import Actuals, Pracpredictions
from ..forms import PredictDemandForm, UploadActualsForm
import pandas as pd
from datetime import datetime, timedelta
import json, re, statistics, os, joblib
from django.db.models import Avg
from datetime import datetime, timedelta
import numpy as np



"""
This file processes the 'About Us' page. 
There is no code required as the page only displays a write-up put in the HTMl template.
"""

def handle_aboutus(request):
    return render(request, 'aboutus.html')