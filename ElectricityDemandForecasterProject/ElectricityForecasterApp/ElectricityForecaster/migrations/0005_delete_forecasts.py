# Generated by Django 4.2.5 on 2023-10-06 03:45

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        (
            "ElectricityForecaster",
            "0004_errors_pracpredictions_delete_practiceactuals_and_more",
        ),
    ]

    operations = [
        migrations.DeleteModel(
            name="Forecasts",
        ),
    ]