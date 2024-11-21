#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os, sys
import pandas as pd


def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                          "electricity_forecaster.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    # from ElectricityForecaster.electricity_demand_forecaster import ElectricityForecaster
    current_path = os.path.dirname(os.path.abspath(__file__))
    # csv_file_path = os.path.join(current_path, 'Compiled.csv')
    # model = ElectricityForecaster()
    # data = pd.read_csv(csv_file_path)
    # data = data[['datetime','load', 'pressure_f', 'cloud_cov_f', 'temp_f',
    #    'wind_dir_f', 'wind_sp_f', 'date', 'month', 'hour', 'type_of_day', ' year']]
    # data = data.rename(columns={'pressure_f':"pressure", 'cloud_cov_f': "cloud_cover", 'temp_f':"temperature",
    #     'wind_dir_f': "wind_direction", 'wind_sp_f': "wind_speed"})
    # data['datetime'] = pd.to_datetime(data['datetime'])
    # data['date'] = pd.to_datetime(data['date'])
    # train, valid = model.split_data(data)

    # model.fit(data, False)
    # model.save_model()

    main()
