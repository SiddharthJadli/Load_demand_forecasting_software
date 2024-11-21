from django.urls import path
from ElectricityForecaster import views

urlpatterns = [
    path('aboutus/', views.about_us, name='aboutus'),
    path('', views.forecaster, name='forecaster'),
    path('forecaster/', views.forecaster, name='forecaster'),
    path('historic/', views.historic_performance, name='historic'),
    path('insights/', views.insights, name='insights')
]