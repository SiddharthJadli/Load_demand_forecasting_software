# Generated by Django 4.2.5 on 2023-09-23 09:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ElectricityForecaster", "0002_practicepredictions"),
    ]

    operations = [
        migrations.CreateModel(
            name="PracticeActuals",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("datetime", models.DateTimeField()),
                (
                    "load",
                    models.DecimalField(
                        blank=True, decimal_places=5, max_digits=15, null=True
                    ),
                ),
                (
                    "pressure",
                    models.DecimalField(
                        blank=True, decimal_places=5, max_digits=15, null=True
                    ),
                ),
                (
                    "cloud_cover",
                    models.DecimalField(
                        blank=True, decimal_places=5, max_digits=15, null=True
                    ),
                ),
                (
                    "humidity",
                    models.DecimalField(
                        blank=True, decimal_places=5, max_digits=15, null=True
                    ),
                ),
                (
                    "temperature",
                    models.DecimalField(
                        blank=True, decimal_places=5, max_digits=15, null=True
                    ),
                ),
                (
                    "wind_speed",
                    models.DecimalField(
                        blank=True, decimal_places=5, max_digits=15, null=True
                    ),
                ),
                (
                    "wind_direction",
                    models.DecimalField(
                        blank=True, decimal_places=5, max_digits=15, null=True
                    ),
                ),
                ("date", models.DateField(blank=True, null=True)),
                ("month", models.IntegerField(blank=True, null=True)),
                ("hour", models.IntegerField(blank=True, null=True)),
                ("type_of_day", models.IntegerField(blank=True, null=True)),
                ("covid", models.IntegerField(blank=True, null=True)),
                ("holiday", models.IntegerField(blank=True, null=True)),
            ],
            options={
                "db_table": "practiceactuals",
                "managed": False,
            },
        ),
    ]
