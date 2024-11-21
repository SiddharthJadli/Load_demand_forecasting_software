# make sure to install if not installed already
# !pip install psycopg2-binary

import psycopg2
import pandas as pd
from pprint import pprint
from psycopg2 import sql
from datetime import timedelta, datetime

"""
This DBConnector class is a class that can be used to connect to the database
and use to execute queries on the given database's tables. 

Its main purposes are to automate the process of connecting and querying the database, without 
the need for manually connecting and using raw SQL, which is beneficial for developers who may
not be too familiar with the SQL language. 

Moreover, the class aims to separate the functionality of database operations into a single class,
fulfilling the 'Single Responsibility' principle; where other functions and files may create 
instance(s) of this class to query the database instead of doing everything in one function and/or class.
"""
class DbConnector:

    def __init__(self):
        conn = None
        cursor = None

    """
    Internal method used to connect to the database. 
    For good coding practice, the database should be closed after being finished with it. 
    Thus, the class will internally connect to the database and then close once the query
    operation is completed.
    """
    def _initiate_connection(self):

        try:
            # First, connect to the Database using the aws hosting
            self.conn = psycopg2.connect(
            host="fit3164db.cxkhqsoitzhb.ap-southeast-2.rds.amazonaws.com",
            port=5432,
            user="postgres",
            password="fit3164d13edas",
            database = "testdb"
            )

            #set up an automated commit for all changes made to the DB
            self.conn.autocommit = True

            #intiate cursor to use for executing queries
            self.cursor = self.conn.cursor()

        except psycopg2.Error as e:
            print(f'Error: {e}')



    """
    Internal method used to close the connection with the database when the database is not 
    being used.
    """
    def _close_connection(self):
        self.cursor.close()
        self.conn.close()


    """
    Deletes data from the 'actuals' table.
    If no datetime value is being passed, the default is to remove all content. 
    Otherwise, delete all data from that date
    """
    def delete_from_actuals(self, start_datetime=None, end_datetime = None):
        self._initiate_connection()

        try:
            if start_datetime and end_datetime:
                #now we know the datetime exists, delete the data
                query = """
                    DELETE
                    FROM actuals
                    WHERE datetime >= %s and datetime <= %s
                """
                self.cursor.execute(query, (start_datetime, end_datetime))
            elif start_datetime and not end_datetime:
                #otherwise, delete the entire data:
                query = """
                        DELETE
                        FROM actuals 
                        WHERE datetime >= %s
                    """
                self.cursor.execute(query, (start_datetime,))
            elif not start_datetime and end_datetime:
                #otherwise, delete the entire data:
                query = """
                        DELETE
                        FROM actuals 
                        WHERE datetime <= %s
                    """
                self.cursor.execute(query, (end_datetime,))
            else:
                #otherwise, delete the entire data:
                query = """
                        DELETE
                        FROM actuals
                    """
                self.cursor.execute(query)


        except psycopg2.Error as e:
            return 

        #close the connection
        self._close_connection()

        return

    """
    Deletes data from the 'pracpredictions' table.
    If no datetime value is being passed, the default is to remove all content. 
    Otherwise, delete all data from that date
    """
    def delete_from_predictions(self, start_datetime=None, end_datetime=None):
        self._initiate_connection()
        
        try:
            if start_datetime and end_datetime:
                #now we know the datetime exists, delete the data
                query = """
                    DELETE
                    FROM pracpredictions
                    WHERE datetime >= %s and  datetime <= %s
                """
                self.cursor.execute(query, (start_datetime, end_datetime))
            elif start_datetime and not end_datetime:
                #otherwise, delete the entire data:
                query = """
                        DELETE
                        FROM pracpredictions 
                        WHERE datetime >= %s
                    """
                self.cursor.execute(query, (start_datetime,))
            elif not start_datetime and end_datetime:
                #otherwise, delete the entire data:
                query = """
                        DELETE
                        FROM pracpredictions 
                        WHERE datetime <= %s
                    """
                self.cursor.execute(query, (end_datetime,))
            else:
                #otherwise, delete the entire data:
                query = """
                        DELETE
                        FROM pracpredictions
                    """
                self.cursor.execute(query)


        except psycopg2.Error as e:
            return f"{e}"

        #close the connection
        self._close_connection()

        return


    """
    This method accepts data as a dataframe and inserts all the data rows into the 'actuals'
    table. 
    Assumptions:
    1. Data is already cleaned (since this is the responsibility of another class)
    2. All required colunms are provided.
    3. It checks if a datetime already exists, then it replaces the values.
    """
    def insert_into_actuals(self, df):

        df = df.reset_index() #reset the index just to be safe

        self._initiate_connection()

        for index, row in df.iterrows():
            try:
                datetime = row['datetime']
                load = row['load']
                pressure = row['pressure']
                cloud_cover = row['cloud_cover']
                temperature = row['temperature']
                wind_speed = row['wind_speed']
                wind_direction = row['wind_direction']
                date = row['date']
                month = row['month']
                hour = row['hour']
                type_of_day = row['type_of_day']

                query = sql.SQL("""
                INSERT INTO actuals (datetime, load, pressure, temperature, cloud_cover, wind_direction, wind_speed, date, month, hour, type_of_day)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (datetime) DO UPDATE
                SET
                    load = excluded.load,
                    pressure = excluded.pressure,
                    temperature = excluded.temperature,
                    cloud_cover = excluded.cloud_cover,
                    wind_direction = excluded.wind_direction,
                    wind_speed = excluded.wind_speed,
                    date = excluded.date,
                    month = excluded.month,
                    hour = excluded.hour,
                    type_of_day = excluded.type_of_day
                """)

                self.cursor.execute(
                query,
                (datetime, load, pressure, temperature, cloud_cover, wind_direction, wind_speed, date, month, hour, type_of_day)
                )
            except psycopg2.Error as e:
                return f'Error: {e}'

        self._close_connection()

    """
    This method accepts data as a dataframe and inserts all the data rows into the 'pracpredictions'
    table. 
    Assumptions:
    1. Data is already cleaned (since this is the responsibility of another class)
    2. All required colunms are provided.
    3. It checks if a datetime already exists, then it replaces the values.
    """
    def insert_into_predictions(self, df):

        df = df.reset_index() #reset the index just to be safe

        self._initiate_connection()

        for index, row in df.iterrows():
            try:
                datetime = row['datetime']
                load = row['load']

                query = sql.SQL("""
                INSERT INTO pracpredictions (datetime, load)
                VALUES (%s, %s)
                ON CONFLICT (datetime) DO UPDATE
                SET
                    load = excluded.load,
                """)

                self.cursor.execute(
                query,
                (datetime, load)
                )

            except psycopg2.Error as e:
                return f'Error: {e}'

        self._close_connection()


    """
    This method allows for saving both predictions and actual values at once. 
    This would be convenient if the developer had just made predictions which typically are
    saved into a list, while also saving the predictions into the 'actuals' table at the same
    time.
    Assumptions:
    1. Data is already cleaned (since this is the responsibility of another class)
    2. All required colunms are provided.
    3. It checks if a datetime already exists, then it replaces the values.
    
    To use this method, provide it with:
    1. Predicts list
    2. Actuals 'row' (day we are predicting for)
    3. forecast lead column names (as a list)
    4. Date decomposition values as a list 
    Both of the last 2 requirements should have methods available in the modelling class, as
    this is where the below method will be called from.
    
    This method can be used when predictions are either:
    1. first created for those dates
    2. updated for those dates
    """
    def save_predictions(self, predictions, actuals, forecast_columns, datetime = None):

        self._initiate_connection()

        timestep = timedelta(hours=1)

        if not datetime:
            #assumes the datetime is set as index
            current_datetime = actuals.index.to_pydatetime()[0]
        else:
            current_datetime = datetime

        for i in range(len(predictions)):

            #save to predictions first:
            load = predictions[i]  # Assuming you have a list of load values

            query = sql.SQL("""
                INSERT INTO pracpredictions (datetime, load)
                VALUES (%s, %s)
                ON CONFLICT (datetime) DO UPDATE
                SET load = excluded.load
            """)

            self.cursor.execute(query, (current_datetime, load))
            self.conn.commit()

            # save to actuals:
            # row/predictions should be 48
            # row: datetime, lags, leads

            #decompose the date:
            date = current_datetime.date()
            month = current_datetime.month
            hour = current_datetime.hour
            type_of_day = current_datetime.isoweekday()
            date_decomp_columns =  [date, month, hour, type_of_day]

            #forecast names:
            for i in range(len(forecast_columns)):
                forecast_columns[i] = forecast_columns[i][:-1] + str(i)

            actual = [current_datetime] + [load] + actuals[forecast_columns].values.tolist()[0] + date_decomp_columns

            datetime, load, pressure, temperature, cloud_cover, wind_direction, wind_speed, date, month, hour, type_of_day = actual

            query = sql.SQL("""
                INSERT INTO actuals (datetime, load, pressure, temperature, cloud_cover, wind_direction, wind_speed, date, month, hour, type_of_day)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (datetime) DO UPDATE
                SET
                    load = excluded.load,
                    pressure = excluded.pressure,
                    temperature = excluded.temperature,
                    cloud_cover = excluded.cloud_cover,
                    wind_direction = excluded.wind_direction,
                    wind_speed = excluded.wind_speed,
                    date = excluded.date,
                    month = excluded.month,
                    hour = excluded.hour,
                    type_of_day = excluded.type_of_day
            """)

            self.cursor.execute(
                query,
                (datetime, load, pressure, temperature, cloud_cover, wind_direction, wind_speed, date, month, hour, type_of_day)
            )

            # Increment the datetime by an hour for the next iteration
            current_datetime += timestep
        self._close_connection()


    def save_predictions_for_valid(self, predictions, row, actuals, forecast_columns, datetime=None):
        
        self._initiate_connection()

        if not datetime:
            #assumes the datetime is set as index
            current_datetime = actuals.index.to_pydatetime()[0]
        else:
            current_datetime = datetime
        
        timestep = timedelta(hours=1)

        for i in range(len(predictions)):

            #save to predictions first:
            load = predictions[i]  # Assuming you have a list of load values
            
            query = sql.SQL("""
                INSERT INTO pracpredictions (datetime, load)
                VALUES (%s, %s)
                ON CONFLICT (datetime) DO UPDATE
                SET load = excluded.load
            """)

            self.cursor.execute(query, (current_datetime, load))
            self.conn.commit()

            # save to actuals:
            # row/predictions should be 48 
            # row: datetime, lags, leads
            
            #save the actual load
            #decompose the date:
            date = current_datetime.date()
            month = current_datetime.month
            hour = current_datetime.hour
            type_of_day = current_datetime.isoweekday()
            date_decomp_columns =  [date, month, hour, type_of_day]

            #forecast names:
            for i in range(len(forecast_columns)):
                forecast_columns[i] = forecast_columns[i][:-1] + str(i)

            load = actuals[i]

            actual = [current_datetime] + [load] + row[forecast_columns].values.tolist()[0] + date_decomp_columns

            datetime, load, pressure, temperature, cloud_cover, wind_direction, wind_speed, date, month, hour, type_of_day = actual

            query = sql.SQL("""
                INSERT INTO actuals (datetime, load, pressure, temperature, cloud_cover, wind_direction, wind_speed, date, month, hour, type_of_day)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (datetime) DO UPDATE
                SET
                    load = excluded.load,
                    pressure = excluded.pressure,
                    temperature = excluded.temperature,
                    cloud_cover = excluded.cloud_cover,
                    wind_direction = excluded.wind_direction,
                    wind_speed = excluded.wind_speed,
                    date = excluded.date,
                    month = excluded.month,
                    hour = excluded.hour,
                    type_of_day = excluded.type_of_day
            """)

            self.cursor.execute(
                query,
                (datetime, load, pressure, temperature, cloud_cover, wind_direction, wind_speed, date, month, hour, type_of_day)
            )

            # Increment the datetime by an hour for the next iteration
            current_datetime += timestep
        self._close_connection()


    """
    Drops the 'actuals' table from the database
    """
    def drop_actuals_table(self):
        self._initiate_connection()

        try:

            query = sql.SQL("""
            DROP TABLE actuals
            """)

            self.cursor.execute(query)

        except psycopg2.Error as e:
            return f'Error: {e}'

        self._close_connection()

    """
    Drops the 'pracpredictions' table from the database
    """
    def drop_predictions_table(self):
        self._initiate_connection()

        try:

            query = sql.SQL("""
            DROP TABLE pracpredictions
            """)

            self.cursor.execute(query)

        except psycopg2.Error as e:
            return f'Error: {e}'

        self._close_connection()


    """
    Input:
    -start_datetime and end_datetime as string. 
    -hour as an integer.
    -list of columns to retrieve, default is all.
    
    retrieves records from the database from the 'actuals' table:
    - Date range if end_datetime and start_datetime are provided.
    - From date if start_datetime is provided but no end_datetime. 
    - Up to date if end_datetime is provided but no end_datetime.
    - Entire data if no start or end datetime are provided.
    - If hour is used, then retrieve records between those dates at that specific hour
    """
    def get_data_from_actuals(self, start_datetime=None, end_datetime=None, hour=None, cols=["*"]):
        self._initiate_connection()

        cols = ', '.join(cols)

        try:
            if hour:
                if not start_datetime and not start_datetime:
                    query = sql.SQL("""
                    SELECT {} FROM ACTUALS
                    WHERE hour = %s
                    ORDER BY DATETIME 
                    """).format(sql.SQL(cols))
                    self.cursor.execute(query, (hour,))

                elif not start_datetime and end_datetime:
                    query = sql.SQL("""
                    SELECT {} FROM ACTUALS
                    WHERE datetime <= %s AND hour = %s
                    ORDER BY DATETIME 
                    """).format(sql.SQL(cols))
                    self.cursor.execute(query, (end_datetime, hour))

                elif start_datetime and not end_datetime:
                    query = sql.SQL("""
                    SELECT {} FROM ACTUALS
                    WHERE datetime >= %s AND hour = %s
                    ORDER BY DATETIME 
                    """).format(sql.SQL(cols))
                    self.cursor.execute(query, (start_datetime, hour))

                else: #that means we need a date range here
                    query = sql.SQL("""
                    SELECT {} FROM ACTUALS
                    WHERE datetime >= %s AND datetime <= %s AND hour = %s
                    ORDER BY DATETIME 
                    """).format(sql.SQL(cols))
                    self.cursor.execute(query, (start_datetime, end_datetime, hour))
            else:
                if not start_datetime and not start_datetime:
                    query = sql.SQL("""
                    SELECT {} FROM ACTUALS
                    ORDER BY DATETIME 
                    """).format(sql.SQL(cols))
                    self.cursor.execute(query)

                elif not start_datetime and end_datetime:
                    query = sql.SQL("""
                    SELECT {} FROM ACTUALS
                    WHERE datetime <= %s
                    ORDER BY DATETIME 
                    """).format(sql.SQL(cols))
                    self.cursor.execute(query, (end_datetime,))

                elif start_datetime and not end_datetime:
                    query = sql.SQL("""
                    SELECT {} FROM ACTUALS
                    WHERE datetime >= %s
                    ORDER BY DATETIME 
                    """).format(sql.SQL(cols))
                    self.cursor.execute(query, (start_datetime,))

                else: #that means we need a date range here
                    query = sql.SQL("""
                    SELECT {} FROM ACTUALS
                    WHERE datetime >= %s AND datetime <= %s
                    ORDER BY DATETIME 
                    """).format(sql.SQL(cols))
                    self.cursor.execute(query, (start_datetime, end_datetime))



            result = self.cursor.fetchall()
            if result:
                column_names = [desc[0] for desc in self.cursor.description]
                df = pd.DataFrame(result, columns= column_names)
            else:
                df = None

        except psycopg2.Error as e:
            return None

        self._close_connection()

        return df

    """
    Input:
    -start_datetime and end_datetime as string. 
    -list of columns to retrieve, default is all.
    
    retrieves records from the database from the 'pracpredictions' table:
    - Date range if end_datetime and start_datetime are provided.
    - From date if start_datetime is provided but no end_datetime. 
    - Up to date if end_datetime is provided but no end_datetime.
    - Entire data if no start or end datetime are provided.
    """
    def get_data_from_predictions(self, start_datetime=None, end_datetime=None, cols=["*"]):
        self._initiate_connection()

        cols = ', '.join(cols)

        try:
            if not start_datetime and not start_datetime:
                query = sql.SQL("""
                SELECT {} FROM PRACPREDICTIONS
                ORDER BY DATETIME 
                """).format(sql.SQL(cols))
                self.cursor.execute(query)

            elif not start_datetime and end_datetime:
                query = sql.SQL("""
                SELECT * FROM PRACPREDICTIONS
                WHERE datetime <= %s AND
                ORDER BY DATETIME 
                """).format(sql.SQL(cols))
                self.cursor.execute(query, (end_datetime))

            elif start_datetime and not end_datetime:
                query = sql.SQL("""
                SELECT {} FROM PRACPREDICTIONS
                WHERE datetime >= %s
                ORDER BY DATETIME 
                """).format(sql.SQL(cols))
                self.cursor.execute(query, (start_datetime))

            else: #that means we need a date range here
                query = sql.SQL("""
                SELECT {} FROM PRACPREDICTIONS
                WHERE datetime >= %s AND datetime <= %s
                ORDER BY DATETIME 
                """).format(sql.SQL(cols))
                self.cursor.execute(query, (start_datetime, end_datetime))

            result = self.cursor.fetchall()
            if result:
                column_names = [desc[0] for desc in self.cursor.description]
                df = pd.DataFrame(result, columns= column_names)
            else:
                df = None

        except psycopg2.Error as e:
            return f'Error: {e}'

        self._close_connection()

        return df

    """
    retrieves the last datetime saved into the 'actuals' table
    """
    def get_last_date_actuals(self):

        self._initiate_connection()

        try:

            query = sql.SQL("""
            SELECT datetime
            FROM actuals
            ORDER BY datetime DESC
            LIMIT 1
            """)

            self.cursor.execute(query)

            result = self.cursor.fetchone()[0]


        except psycopg2.Error as e:
            return f'Error: {e}'

        self._close_connection()

        return result


    """
    retrieves the last datetime saved into the 'actuals' table
    """
    def get_last_date_predictions(self):

        self._initiate_connection()

        try:

            query = sql.SQL("""
            SELECT datetime
            FROM pracpredictions
            ORDER BY datetime DESC
            LIMIT 1
            """)

            self.cursor.execute(query)

            result = self.cursor.fetchone()[0]


        except psycopg2.Error as e:
            return f'Error: {e}'

        self._close_connection()

        return result

    """
    Updates the values of 'load' in the 'actuals' table. 
    
    This method would be used when the user provides the actuals file, and we would need to update the load
    with the actual load values (since previously we had the predicted load saved).
    """
    def update_actuals_load(self, df):
        df = df.reset_index() #reset the index just to be safe

        self._initiate_connection()

        for index, row in df.iterrows():
            try:
                datetime = row['datetime']
                load = row['load']

                query = sql.SQL("""
                UPDATE actuals
                SET load = %s
                WHERE datetime = %s
                """)

                self.cursor.execute(
                query,
                (load, datetime)
                )

            except psycopg2.Error as e:
                return f'Error: {e}'

        self._close_connection()