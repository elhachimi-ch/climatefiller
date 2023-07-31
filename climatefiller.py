from data_science_toolkit.gis import GIS
from data_science_toolkit.dataframe import DataFrame
import datetime
import os
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from quantilesdetector import PercentileDetection
from lib import Lib
import ee
import geemap
import re



class ClimateFiller():
    """The ClimateFiller class
    """
    
    def __init__(self, data_link=None, data_type='csv', datetime_column_name='datetime', date_time_format='%Y-%m-%d %H:%M:%S', backend=None):
        """
        Initializes an instance of the class with the specified parameters.

        Args:
            self (object): The instance of the class.
            data_link (str or None): A string representing the link or path to the data source. Defaults to None.
            data_type (str): The type of the data source. Defaults to 'csv'.
            datetime_column_name (str): The name of the column that contains datetime information. Defaults to 'datetime'.
            date_time_format (str): The format of the datetime values in the data source. Defaults to '%Y-%m-%d %H:%M:%S'.

        Returns:
            None

        Notes:
            - The initialization of the class instance allows for handling and processing of the data.
            - The data_link parameter specifies the location of the data source, which can be a link or a local path.
            - The data_type parameter indicates the format or type of the data source, with 'csv' as the default value.
            - The datetime_column_name parameter identifies the column in the data source that contains datetime information.
            - The date_time_format parameter defines the format of the datetime values in the data source.
            - If data_link is not provided, the instance will be initialized without any data source.
        """
        self.datetime_column_name = datetime_column_name
        self.backend = backend
        if backend == 'gee':
            ee.Initialize()
        if data_link is None:
            self.data = DataFrame()
        else:
            self.data = DataFrame(data_link=data_link, data_type=data_type)
            self.data.column_to_date(datetime_column_name, date_time_format)
            self.datetime_column_name = datetime_column_name
        
    def show(self, number_of_row=None):
        """
        Displays a specified number of rows from the data source.

        Args:
            self (object): The instance of the class.
            number_of_row (int or None, optional): The number of rows to display. Defaults to None.

        Returns:
            None

        Notes:
            - The show method is used to visualize a specified number of rows from the data source.
            - If the number_of_row parameter is not provided, all available rows will be displayed.
            - The displayed rows provide a preview or snapshot of the data in the data source.
        """
        
        if number_of_row is None:
            return self.data.get_dataframe()
        elif number_of_row < 0:
            return self.data.get_dataframe().tail(abs(number_of_row)) 
        else:
            return self.data.get_dataframe().head(number_of_row)
    
    def recursive_fill(self, column_to_fill_name='ta', 
                              variable='ta', 
                              latitude=31.66749781,
                              longitude=-7.593311291):
        
        """
        Recursively fills missing values in the specified column using a specified variable and coordinates.

        Args:
            column_to_fill_name (str): The name of the column to fill. Defaults to 'ta'.
            variable (str): The variable to use for filling missing values. Defaults to 'ta'.
            latitude (float): The latitude coordinate to use for filling missing values. Defaults to 31.66749781.
            longitude (float): The longitude coordinate to use for filling missing values. Defaults to -7.593311291.

        Returns:
            None
        """
        if self.missing_data_checking(column_to_fill_name) == 0:
            print("No missing data found.")
        elif self.missing_data_checking(column_to_fill_name) > 1000:
            import numpy as np
            data_chuncks = np.array_split(self.data.get_dataframe(), 2)
            return DataFrame(ClimateFiller(data_chuncks[0], data_type='df').fill(column_to_fill_name,
                                                                                 variable,
                                                                                 latitude,
                                                                                 longitude), data_type='df').append_dataframe(ClimateFiller(data_chuncks[1], data_type='df').fill(
                                                                                 column_to_fill_name,
                                                                                 variable,
                                                                                 latitude,
                                                                                 longitude))
    
    def fill(self, column_to_fill_name='ta', 
                              longitude=-7.593311291,
                              latitude=31.66749781,
                              product="era5_Land",
                              machine_learning_enabled=False,
                              backend=None
                              ):
        """
        Fills missing values in the specified column using data retrieval and optionally machine learning techniques.

        Args:
            self (object): The instance of the class.
            column_to_fill_name (str): The name of the column to fill. Defaults to 'ta'.
            longitude (float): The longitude coordinate to use for data retrieval. Defaults to -7.593311291.
            latitude (float): The latitude coordinate to use for data retrieval. Defaults to 31.66749781.
            product (str): The data product to retrieve for filling missing values. Defaults to "era5_Land".
            machine_learning_enabled (bool): Whether to use machine learning techniques for filling missing values. Defaults to False.
            backend (str or None): The backend to use for data retrieval. Defaults to None.

        Returns:
            None

        Notes:
            - Missing values in the specified column will be replaced with appropriate data retrieved from the specified coordinates.
            - The data product specified will be used to retrieve relevant data for filling missing values.
            - The option to enable machine learning techniques allows for more sophisticated filling strategies.
            - If the backend is not specified, the method will use the default backend associated with the class.
            - The effectiveness of the filling process may depend on the data availability and the chosen backend.
        """
        if self.missing_data_checking(column_to_fill_name, verbose=False) == 0:
            print('No missing data found in ' + column_to_fill_name)
            return
        
        if column_to_fill_name == 'ta':
            era5_land_variables = ['2m_temperature']
        elif column_to_fill_name == 'rh':
            era5_land_variables = ['2m_temperature', '2m_dewpoint_temperature']
        elif column_to_fill_name == 'rs':
            era5_land_variables = ['surface_solar_radiation_downwards']
        elif column_to_fill_name == 'ws':
            era5_land_variables = ['10m_u_component_of_wind', '10m_v_component_of_wind']
            
            
        from data_science_toolkit.gis import GIS
        import cdsapi
        c = cdsapi.Client()

        if self.datetime_column_name is not None:
            self.data.reindex_dataframe(self.datetime_column_name)

        indexes = []
        for p in self.data.get_missing_data_indexes_in_column(column_to_fill_name):
            if isinstance(p, str) is True:
                indexes.append(datetime.datetime.strptime(p, '%Y-%m-%d %H:%M:%S'))
            else:
                indexes.append(p)
            
        years = set()
        for p in indexes:
            years.add(p.year)     
        missing_data_dates = {}    
        years = list(years)
        print("Found missing data for {} in year(s): {}".format(column_to_fill_name, years))  
        for y in years:
            missing_data_dict = {}
            missing_data_dict['month'] = set()   
            missing_data_dict['day'] = set() 
            
            for p in indexes:
                if p.year == y:
                    missing_data_dict['month'].add(p.strftime('%m'))
                    missing_data_dict['day'].add(p.strftime('%d'))
            missing_data_dict['month'] = list(missing_data_dict['month'])
            missing_data_dict['day'] = list(missing_data_dict['day'])
            missing_data_dates[y] = missing_data_dict
            for month in missing_data_dict['month']:
                if len(era5_land_variables) == 1:
                    if os.path.exists("data\era5_r3_" + column_to_fill_name + '_' + str(y) + '_' + month + '.grib') is False:
                        c.retrieve(
                        'reanalysis-era5-land',
                        {
                            'format': 'grib',
                            'variable': era5_land_variables,
                            'year': str(y),
                            'month':  month,
                            'day': missing_data_dict['day'],
                            'time': [
                                '00:00', '01:00', '02:00',
                                '03:00', '04:00', '05:00',
                                '06:00', '07:00', '08:00',
                                '09:00', '10:00', '11:00',
                                '12:00', '13:00', '14:00',
                                '15:00', '16:00', '17:00',
                                '18:00', '19:00', '20:00',
                                '21:00', '22:00', '23:00',
                            ],
                            'area': [
                                latitude, longitude, latitude,
                                longitude
                            ],
                        },
                        'data/era5_r3_' + column_to_fill_name + '_' + str(y) + '_' + month + '.grib')
                else:
                    for p in era5_land_variables:
                        if os.path.exists("data\era5_r3_" + column_to_fill_name + '_' + str(p) + '_' + str(y) + '_' + month + '.grib') is False:
                            c.retrieve(
                            'reanalysis-era5-land',
                            {
                                'format': 'grib',
                                'variable': p,
                                'year': str(y),
                                'month':  month,
                                'day': missing_data_dict['day'],
                                'time': [
                                    '00:00', '01:00', '02:00',
                                    '03:00', '04:00', '05:00',
                                    '06:00', '07:00', '08:00',
                                    '09:00', '10:00', '11:00',
                                    '12:00', '13:00', '14:00',
                                    '15:00', '16:00', '17:00',
                                    '18:00', '19:00', '20:00',
                                    '21:00', '22:00', '23:00',
                                ],
                                'area': [
                                    latitude, longitude, latitude,
                                    longitude
                                ],
                            },
                            'data/era5_r3_' + column_to_fill_name + '_' + str(p) + '_' + str(y) + '_' + month + '.grib')
                    
        
        gis = GIS()
        data = DataFrame()
        
        if column_to_fill_name == 'ta':
            for year in missing_data_dates:
                for month in missing_data_dates[year]['month']:
                    data.append_dataframe(gis.get_era5_land_grib_as_dataframe("data/era5_r3_" + column_to_fill_name + '_' + str(year) + '_' + month + ".grib", "ta"),)
                    
            data.reset_index()
            data.reindex_dataframe("valid_time")
            data.missing_data('t2m')
            data.transform_column('t2m', lambda o: o - 273.15)
            nan_indices = self.data.get_nan_indexes_of_column(column_to_fill_name)
            for p in nan_indices:
                self.data.set_row('ta', p, data.get_row(p)['t2m'])
            print('Imputation of missing data for ta from ERA5-Land was done!')
            
        elif column_to_fill_name == 'rh':
            data_t2m = DataFrame()
            data_d2m = DataFrame()
            for year in missing_data_dates:
                for month in missing_data_dates[year]['month']:
                    data_t2m.append_dataframe(gis.get_era5_land_grib_as_dataframe("data\era5_r3_" + column_to_fill_name + '_' + '2m_temperature' + '_' + str(year) + '_' + month + ".grib", "ta"),)
            for year in missing_data_dates:
                for month in missing_data_dates[year]['month']:
                    data_d2m.append_dataframe(gis.get_era5_land_grib_as_dataframe("data\era5_r3_" + column_to_fill_name + '_' + '2m_dewpoint_temperature' + '_' + str(year) + '_' + month + ".grib", "ta"),)
            
            data_d2m.reset_index()
            data_d2m.reindex_dataframe("valid_time")
            data_d2m.keep_columns(['d2m'])
            data_t2m.reset_index()
            data_t2m.reindex_dataframe("valid_time")
            data_t2m.keep_columns(['t2m'])
            data_t2m.join(data_d2m.get_dataframe())
            data = data_t2m
            data.missing_data('t2m')
            data.transform_column('t2m', lambda o: o - 273.15)
            data.transform_column('d2m', lambda o: o - 273.15)
            
            """RH: =100*(EXP((17.625*TD)/(243.04+TD))/EXP((17.625*T)/(243.04+T)))
            TD: =243.04*(LN(RH/100)+((17.625*T)/(243.04+T)))/(17.625-LN(RH/100)-((17.625*T)/(243.04+T)))
            T: =243.04*(((17.625*TD)/(243.04+TD))-LN(RH/100))/(17.625+LN(RH/100)-((17.625*TD)/(243.04+TD)))"""
            
            data.add_transformed_columns('era5_hr', '100*exp(-((243.12*17.62*t2m)-(d2m*17.62*t2m)-d2m*17.62*(243.12+t2m))/((243.12+t2m)*(243.12+d2m)))')
            data.missing_data('era5_hr')
            nan_indices = self.data.get_missing_data_indexes_in_column(column_to_fill_name)
            for p in nan_indices:
                self.data.set_row('rh', p, data.get_row(p)['era5_hr'])
             
            print('Imputation of missing data for rh from ERA5-Land was done!')
            
        elif column_to_fill_name == 'ws':
            data_u10 = DataFrame()
            data_v10 = DataFrame()
            for year in missing_data_dates:
                for month in missing_data_dates[year]['month']:
                    data_u10.append_dataframe(gis.get_era5_land_grib_as_dataframe("data\era5_r3_" + column_to_fill_name + '_' + '10m_u_component_of_wind' + '_' + str(year) + '_' + month + ".grib", "ta"),)
            for year in missing_data_dates:
                for month in missing_data_dates[year]['month']:
                    data_v10.append_dataframe(gis.get_era5_land_grib_as_dataframe("data\era5_r3_" + column_to_fill_name + '_' + '10m_v_component_of_wind' + '_' + str(year) + '_' + month + ".grib", "ta"),)
            
            data_u10.reset_index()
            data_u10.reindex_dataframe("valid_time")
            data_u10.keep_columns(['u10'])
            data_v10.reset_index()
            data_v10.reindex_dataframe("valid_time")
            data_v10.keep_columns(['v10'])
            data_v10.join(data_u10.get_dataframe())
            data = data_v10
            data.add_column_based_on_function('era5_ws', Lib.get_2m_wind_speed)
            nan_indices = self.data.get_missing_data_indexes_in_column(column_to_fill_name)
            data.missing_data('u10')
            data.missing_data('era5_ws')
            for p in nan_indices:
                self.data.set_row('ws', p, data.get_row(p)['era5_ws'])
            
            print('Imputation of missing data for wind speed from ERA5-Land was done!')
            
        elif column_to_fill_name == 'rs':
            for year in missing_data_dates:
                for month in missing_data_dates[year]['month']:
                    data.append_dataframe(gis.get_era5_land_grib_as_dataframe("data\era5_r3_" + column_to_fill_name + '_' + str(year) + '_' + month + ".grib", "ta"),)
                    
            data.reset_index()
            data.reindex_dataframe("valid_time")
            data.missing_data('ssrd')
            l = []
            for p in data.get_index():
                if p.hour == 1:
                    new_value = data.get_row(p)['ssrd']/3600
                else:
                    try:
                        previous_hour = data.get_row(p-timedelta(hours=1))['ssrd']
                    except KeyError: # if age is not convertable to int
                        previous_hour = data.get_row(p)['ssrd']
                        
                    new_value = (data.get_row(p)['ssrd'] - previous_hour)/3600
                l.append(new_value)
            data.add_column('rs', l)
            data.keep_columns(['rs'])
            data.rename_columns({'rs': 'ssrd'})
            print(data.show())
            
            data.transform_column('ssrd', lambda o : o if abs(o) < 1500 else 0 )    
            nan_indices = self.data.get_nan_indexes_of_column(column_to_fill_name)
            for p in nan_indices:
                self.data.set_row('rs', p, data.get_row(p)['ssrd'])
            
            print('Imputation of missing data for rs from ERA5-Land was done!')
        
        self.data.index_to_column()
    

    def missing_data_checking(self, column_name=None, verbose=True):
        """Function Name: missing_data_checking

            Description:
            This function checks for missing data in a column of a dataframe.

            Parameters:

            self: the instance of the class that the function is a part of.
            column_name: (optional) the name of the column to check for missing data. If not provided, the function will check for missing data in all columns of the dataframe. Default value is None.
            verbose: (optional) a boolean value that determines whether the function should print a summary of the missing data. Default value is True.
            Returns:
            A dictionary that contains the number and percentage of missing values for each column checked.

            Note:
            This function assumes that the dataframe has already been loaded into the class instance.
        """
        miss = 0
        if column_name is not None:
            if any(pd.isna(self.data.get_dataframe()[column_name])) is True:
                miss = self.data.get_dataframe()[column_name].isnull().sum()
                missing_data_percent = round((miss/self.data.get_shape()[0])*100, 2)
                if verbose is True:
                    print("{} has {} missing value(s) which represents {}% of dataset size".format(column_name, miss, missing_data_percent))
            else:
                if verbose is True:
                    print("No missed data in column " + column_name)
        else:
            miss = []
            for c in self.data.get_dataframe().columns:
                miss_by_column = self.data.get_dataframe()[c].isnull().sum()
                if miss_by_column>0:
                    missing_data_percent = round((miss_by_column/self.data.get_shape()[0])*100, 2)
                    if verbose is True:
                        print("{} has {} missing value(s) which represents {}% of dataset size".format(c, miss_by_column, missing_data_percent))
                else:
                    if verbose is True:
                        print("{} has NO missing value!".format(c))
                miss.append(miss_by_column)
        if verbose is False:
            return miss
    
    def eliminate_outliers(self, climate_varibale_column_name='ta', method='lof', n_neighbors=48, contamination=0.005, n_estimators=100):
        """
        Eliminates outliers in the specified climate variable column using the specified outlier detection method.

        Args:
            self (object): The instance of the class.
            climate_variable_column_name (str): The name of the climate variable column to eliminate outliers from.
                Defaults to 'ta'.

            method (str): The outlier detection method to use. Currently supported methods include:
                - 'lof': Local Outlier Factor algorithm, which measures the local deviation of a data point
                with respect to its neighbors. Defaults to 'lof'.

            n_neighbors (int): The number of neighbors to consider for outlier detection.
                This parameter is only applicable to certain outlier detection methods. Defaults to 48.

            contamination (float): The expected proportion of outliers in the data.
                This parameter is only applicable to certain outlier detection methods. Defaults to 0.005.

            n_estimators (int): The number of base estimators to use for ensemble-based outlier detection methods.
                This parameter is only applicable to certain outlier detection methods. Defaults to 100.

        Returns:
            None

        Notes:
            - Outliers are data points that significantly deviate from the majority of the data.
            - The specified climate variable column will be processed to identify and eliminate outliers.
            - The chosen outlier detection method will be applied to identify and mark outliers in the data.
            - The method aims to improve the quality and reliability of the climate variable data by removing outliers.
            - The effectiveness and performance of the outlier elimination process may vary depending on the method and parameters used.
        """
        if method == 'lof':
            outliers_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            self.data.get_dataframe()['inlier'] = outliers_model.fit_predict(self.data.get_columns([climate_varibale_column_name]))
            print('Number of detected outliers: {}'.format(self.data.count_occurence_of_each_row('inlier').iloc[0]))
            self.data.dataframe.loc[self.data.get_dataframe()['inlier'] == -1, climate_varibale_column_name] = None
            self.data.drop_column('inlier')
        
        elif method == 'isolation_forest':
            outliers_model = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
            self.data.get_dataframe()['inlier'] = outliers_model.fit_predict(self.data.get_columns([climate_varibale_column_name]))
            print('Number of detected outliers: {}'.format(self.data.count_occurence_of_each_row('inlier').iloc[0]))
            self.data.dataframe.loc[self.data.get_dataframe()['inlier'] == -1, climate_varibale_column_name] = 2000
            self.data.drop_column('inlier')
        
        elif method == 'quantiles':
            outliers_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            self.data.get_dataframe()['inlier'] = outliers_model.fit_predict(self.data.get_columns([climate_varibale_column_name]))
            print('Number of detected outliers: {}'.format(self.data.count_occurence_of_each_row('inlier').iloc[0]))
            self.data.dataframe.loc[self.data.get_dataframe()['inlier'] == -1, climate_varibale_column_name] = None
            self.data.drop_column('inlier')
    
    def evaluate_products(self):
        pass
    
    def plot_column(self, column):
        """Function Name: plot_column

            Description:
            This function creates a time-series plot of a column in a dataframe.

            Parameters:

            self: the instance of the class that the function is a part of.
            column: the name of the column to plot.
            Returns:
            None. The function generates a time-series plot of the specified column.

            Note:
            This function assumes that the dataframe has already been loaded into the class instance. This function requires the matplotlib and seaborn libraries to be installed.
        """
        self.data.get_column(column).plot()
        plt.show()
    
    def et0_estimation(self, 
                       air_temperture_column_name='ta',
                       global_solar_radiation_column_name='rs',
                       air_relative_humidity_column_name='rh',
                       wind_speed_column_name='ws',
                       latitude=31.65410805,
                       longitude=-7.603140831,
                       method='pm',
                       verbose=False
                       ):
        """
        Estimates reference evapotranspiration (ET0) using the specified meteorological data and method.

        Args:
            self (object): The instance of the class.
            air_temperature_column_name (str): The name of the column that contains air temperature data. Defaults to 'ta'.
            global_solar_radiation_column_name (str): The name of the column that contains global solar radiation data. Defaults to 'rs'.
            air_relative_humidity_column_name (str): The name of the column that contains air relative humidity data. Defaults to 'rh'.
            wind_speed_column_name (str): The name of the column that contains wind speed data. Defaults to 'ws'.
            date_time_column_name (str): The name of the column that contains date and time information. Defaults to 'date_time'.
            latitude (float): The latitude coordinate of the location for ET0 estimation. Defaults to 31.65410805.
            longitude (float): The longitude coordinate of the location for ET0 estimation. Defaults to -7.603140831.
            method (str): The method to use for ET0 estimation. Currently supported methods include:
                - 'pm': Penman-Monteith method, which is based on the FAO56 Penman-Monteith equation. Defaults to 'pm'.
            in_place (bool): Whether to replace the original ET0 column in the dataset or create a new column. Defaults to True.

        Returns:
            None

        Notes:
            - ET0 estimation is a measure of the potential evapotranspiration from a reference crop.
            - The method utilizes meteorological data such as air temperature, global solar radiation, air relative humidity, and wind speed.
            - The specified columns in the dataset will be used for ET0 estimation.
            - The latitude and longitude coordinates define the location for ET0 estimation.
            - The chosen method will be applied to calculate ET0 values.
            - If in_place is True, the original ET0 column will be replaced; otherwise, a new column will be created.
        """
        
        et0_data = DataFrame()
        et0_data.add_column('ta_mean', self.data.resample_timeseries(in_place=False)[air_temperture_column_name])
        et0_data.add_column('ta_max', self.data.resample_timeseries(in_place=False, agg='max')[air_temperture_column_name])
        et0_data.add_column('ta_min', self.data.resample_timeseries(in_place=False, agg='min')[air_temperture_column_name], )
        et0_data.add_column('rh_max', self.data.resample_timeseries(in_place=False, agg='max')[air_relative_humidity_column_name])
        et0_data.add_column('rh_min', self.data.resample_timeseries(in_place=False, agg='min')[air_relative_humidity_column_name])
        et0_data.add_column('rh_mean', self.data.resample_timeseries(in_place=False)[air_relative_humidity_column_name])
        et0_data.add_column('u2_mean', self.data.resample_timeseries(in_place=False)[wind_speed_column_name])
        et0_data.add_column('rg_mean', self.data.resample_timeseries(in_place=False)[global_solar_radiation_column_name])
        et0_data.index_to_column()
        et0_data.add_doy_column('datetime')
        et0_data.add_one_value_column('elevation', Lib.get_elevation_and_latitude(latitude, longitude))
        et0_data.add_one_value_column('lat', latitude)
        
        if method == 'pm':
            et0_data.add_column_based_on_function('et0_pm', Lib.et0_penman_monteith)
        elif method == 'hargreaves':
            et0_data.add_column_based_on_function('et0_hargreaves', Lib.et0_hargreaves)
            
        self.data.set_dataframe(et0_data.get_dataframe())
        
        if verbose is True:
            print(et0_data.get_dataframe())
            
        return self.data.get_dataframe()
    
    def apply_quality_control_criteria(self, variable_column_name, decision_func=lambda x:x>0):
        """
        Applies quality control criteria to the specified variable column based on a decision function.

        Args:
            self (object): The instance of the class.
            variable_column_name (str): The name of the column containing the variable to apply quality control to.
            decision_func (function, optional): The decision function used to determine if a value passes quality control.
                Defaults to lambda x: x > 0, which checks if the value is greater than zero.

        Returns:
            None

        Notes:
            - The apply_quality_control_criteria method is used to perform quality control on a variable column.
            - The specified variable_column_name is the column in the dataset that will undergo quality control.
            - The decision_func parameter allows customization of the quality control criteria by providing a decision function.
            - The decision function should take a value as input and return True if it passes quality control, False otherwise.
            - Values in the variable column that do not meet the quality control criteria will be marked or processed accordingly.
            - The quality control process helps identify and handle data points that may be inaccurate, erroneous, or outliers.
        """
        
        self.data.add_column('decision', self.data.get_column(variable_column_name).apply(decision_func))
        self.data.get_dataframe().loc[ self.data.get_dataframe()['decision'] == False, variable_column_name] = None
        self.data.drop_column('decision')
        
    def apply_constraint(self, column_name, constraint):
        """Function Name: apply_constraint

            Description:
            This function applies a constraint to a column of a dataframe.

            Parameters:

            self: the instance of the class that the function is a part of.
            column_name: the name of the column to apply the constraint to.
            constraint: a string that represents the constraint to apply. The string should be in the form of a valid Python expression. The constraint will be applied to the column using the eval() function.
            Returns:
            A dataframe with the specified constraint applied to the specified column.

            Note:
            This function assumes that the dataframe has already been loaded into the class instance. The constraint parameter should be a valid Python expression that can be evaluated using the eval() function. This function requires the pandas library to be installed.
        """
        self.data.filter_dataframe(column_name, constraint)
    
    def missing_data(self, drop_row_if_nan_in_column=None, filling_dict_colmn_val=None, method='ffill',
                     column_to_fill='ta', date_column_name=None):
        """Function Name: missing_data

            Description:
            This function fills or drops missing data in a dataframe.

            Parameters:

            self: the instance of the class that the function is a part of.
            drop_row_if_nan_in_column: (optional) the name of a column in the dataframe. If provided, the function will drop rows where this column contains NaN values. Default value is None.
            filling_dict_colmn_val: (optional) a dictionary containing column names and values to be used to fill missing data in those columns. The keys of the dictionary should be the names of the columns to be filled, and the values should be the values to use for filling. Default value is None.
            method: (optional) a string that determines the method used for filling missing data. Possible values are 'ffill' for forward filling, 'bfill' for backward filling, or 'interpolate' for linear interpolation. Default value is 'ffill'.
            column_to_fill: (optional) the name of the column to be filled. This parameter is only used when method is set to 'ffill' or 'bfill'. Default value is 'ta'.
            date_column_name: (optional) the name of the column in the dataframe that contains the date-time information. This parameter is only used when method is set to 'interpolate'. Default value is None.
            Returns:
            A dataframe with missing values filled or dropped according to the specified parameters.

            Note:
            This function assumes that the dataframe has already been loaded into the class instance. If filling_dict_colmn_val is used, the keys of the dictionary should correspond to columns in the dataframe. If method is set to 'interpolate', the date_column_name parameter must be provided. This function requires the pandas library to be installed.
        """
        if filling_dict_colmn_val is None and drop_row_if_nan_in_column is None:
            if method == 'ffill':
                self.data.get_dataframe().fillna(method='pad', inplace=True)
            elif method == 'bfill':
                self.data.get_dataframe().fillna(method='backfill', inplace=True)
       
        if filling_dict_colmn_val is not None:
            self.data.get_dataframe().fillna(filling_dict_colmn_val, inplace=True)
            
        if drop_row_if_nan_in_column is not None:
            if drop_row_if_nan_in_column == 'all':
                for p in self.data.get_columns_names():
                    self.data.set_dataframe(self.data.get_dataframe()[self.data.get_dataframe()[p].notna()])
            else:
                # a = a[~(np.isnan(a).all(axis=1))] # removes rows containing all nan
                self.data.set_dataframe(self.data.get_dataframe()[self.data.get_dataframe()[drop_row_if_nan_in_column].notna()])
                #self.__dataframe = self.__dataframe[~(np.isnan(self.__dataframe).any(axis=1))] # removes rows containing at least one nan

    def download(self, 
    variable, 
    lon=-7.593311291,
    lat=31.66749781,
    start_date='2021-01-01',
    end_date='2021-02-01',
    product='era5-Land',
    sequential_downloading=False
    ):
        """
        Downloads meteorological data for the specified variable and spatiotemporal range.

        Args:
            self (object): The instance of the class.
            variable (str): The variable to download meteorological data for.
            start_date (str): The start date of the spatiotemporal range in 'YYYY-MM-DD' format. Defaults to '2021-01-01'.
            end_date (str): The end date of the spatiotemporal range in 'YYYY-MM-DD' format. Defaults to '2021-02-01'.
            latitude (float): The latitude coordinate for the data download. Defaults to 31.66749781.
            longitude (float): The longitude coordinate for the data download. Defaults to -7.593311291.
            product (str): The product or dataset to download data from. Defaults to 'era5-Land'.
            backend (str or None): The backend to use for data retrieval. Defaults to None.

        Returns:
            None

        Notes:
            - The download method is used to retrieve meteorological data for a specific variable.
            - The variable parameter specifies the variable of interest, such as temperature, precipitation, etc.
            - The start_date and end_date parameters define the spatiotemporal range to download data for.
            - The latitude and longitude coordinates specify the location for data retrieval.
            - The product parameter identifies the specific dataset or product to download data from.
            - If the backend is not specified, the method will use the default backend associated with the class.
            - The downloaded data can be used for further analysis, processing, or visualization.
            - The availability of data and the chosen backend may affect the success of the download process.
        """
        
        self.check_directory_existance('data')
        self.check_directory_existance('data/cache')
        
        if product == 'era5-Land':
            # Convert the start date and end date to datetime objects
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            
            if self.backend == 'gee':
                data = DataFrame()
                
                if variable == 'ta':
                    era5_land_variables = ['temperature_2m']
                    
                    output_file = 'data/' + '_'.join([variable, str(lon), str(lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                    if os.path.exists(output_file):
                        print(f"Time series already downloaded on: {output_file}")
                    else:
                        self.download_era5_land_data_by_years(era5_land_variables, lon, lat, start_date, end_date + timedelta(1))
                        
                        for year in range(start_date.year, end_date.year + 1):
                            cache_path = 'data/cache/' + '_'.join([str(s) for s in era5_land_variables] + [str(lon), str(lat), str(year)]) + '.csv'
                            temp_data = DataFrame(cache_path)
                            data.append_dataframe(temp_data.get_dataframe())
                            
                        if not 'first' in temp_data.get_columns_names():
                            print(f'No data found in GEE about {variable} for ({lon, lat})')
                            #cf = ClimateFiller()
                            #cf.download(variable, start_datetime, start_datetime + next_year, longitude, latitude)
                        else:
                            
                            data.rename_columns({'first': 't2m'})
                            data.transform_column('t2m', lambda o: o - 273.15)
                            data.column_to_date('datetime')
                            data.reindex_dataframe('datetime')
                            end_date += timedelta(1)
                            data.select_datetime_range(start_date.isoformat(), end_date.isoformat())
                            data.export(output_file, index=True)
                            
                            if sequential_downloading is True:
                                if self.data.is_empty():
                                    self.data.set_dataframe(data.get_dataframe())
                                else:
                                    self.data.join(data.get_dataframe())
                    
                elif variable == 'rh':
                    era5_land_variables = ['temperature_2m', 'dewpoint_temperature_2m']
                    
                    output_file = 'data/' + '_'.join([variable, str(lon), str(lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                    if os.path.exists(output_file):
                        print(f"Time series already downloaded on: {output_file}")
                    else:
                        self.download_era5_land_data_by_years(era5_land_variables, lon, lat, start_date, end_date + timedelta(1))
                        
                        for year in range(start_date.year, end_date.year + 1):
                            cache_path = 'data/cache/' + '_'.join([str(s) for s in era5_land_variables] + [str(lon), str(lat), str(year)]) + '.csv'
                            temp_data = DataFrame(cache_path)
                            data.append_dataframe(temp_data.get_dataframe())
                            
                        if not 'temperature_2m' in temp_data.get_columns_names() and 'dewpoint_temperature_2m' in temp_data.get_columns_names():
                            print(f'No data found in GEE about {variable} for ({lon, lat})')
                            #cf = ClimateFiller()
                            #cf.download(variable, start_datetime, start_datetime + next_year, longitude, latitude)
                        else:
                            output_file = 'data/' + '_'.join([variable, str(lon), str(lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                            data.rename_columns({'temperature_2m': 't2m', 'dewpoint_temperature_2m': 'd2m'})
                            data.transform_column('t2m', lambda o: o - 273.15)
                            data.transform_column('d2m', lambda o: o - 273.15)
                            data.add_transformed_columns('era5_hr', '100*exp(-((243.12*17.62*t2m)-(d2m*17.62*t2m)-d2m*17.62*(243.12+t2m))/((243.12+t2m)*(243.12+d2m)))')
                            data.drop_columns(['t2m', 'd2m'])
                            data.column_to_date('datetime')
                            data.reindex_dataframe('datetime')
                            end_date += timedelta(1)
                            data.select_datetime_range(start_date.isoformat(), end_date.isoformat())
                            data.export(output_file, index=True)
                            
                            if sequential_downloading is True:
                                if self.data.is_empty():
                                    self.data.set_dataframe(data.get_dataframe())
                                else:
                                    self.data.join(data.get_dataframe())

                elif variable == 'rs':
                    era5_land_variables = ['surface_solar_radiation_downwards']
                    
                    output_file = 'data/' + '_'.join([variable, str(lon), str(lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                    if os.path.exists(output_file):
                        print(f"Time series already downloaded on: {output_file}")
                    else:
                        self.download_era5_land_data_by_years(era5_land_variables, lon, lat, start_date, end_date + timedelta(1))
                        
                        for year in range(start_date.year, end_date.year + 1):
                            cache_path = 'data/cache/' + '_'.join([str(s) for s in era5_land_variables] + [str(lon), str(lat), str(year)]) + '.csv'
                            temp_data = DataFrame(cache_path)
                            data.append_dataframe(temp_data.get_dataframe())
                            
                        if not 'first' in temp_data.get_columns_names():
                            print(f'No data found in GEE about {variable} for ({lon, lat})')
                            #cf = ClimateFiller()
                            #cf.download(variable, start_datetime, start_datetime + next_year, longitude, latitude)
                        else:
                            output_file = 'data/' + '_'.join([variable, str(lon), str(lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                            data.rename_columns({'first': 'ssrd'})
                            data.column_to_date('datetime')
                            data.reindex_dataframe('datetime')
                            
                            
                            l = []
                            for p in data.get_index():
                                if p.hour == 1:
                                    new_value = data.get_row(p)['ssrd']/3600
                                else:
                                    try:
                                        previous_hour = data.get_row(p-timedelta(hours=1))['ssrd']
                                    except KeyError:
                                        previous_hour = data.get_row(p)['ssrd']
                                        
                                    new_value = (data.get_row(p)['ssrd'] - previous_hour)/3600
                                l.append(new_value)
                            data.add_column('rs', l)
                            data.keep_columns(['rs'])
                            data.rename_columns({'rs': 'ssrd'})
                            end_date += timedelta(1)
                            data.select_datetime_range(start_date.isoformat(), end_date.isoformat())
                            data.export(output_file, index=True)
                            
                            if sequential_downloading is True:
                                if self.data.is_empty():
                                    self.data.set_dataframe(data.get_dataframe())
                                else:
                                    self.data.join(data.get_dataframe())
                        
                elif variable == 'ws':
                    era5_land_variables = ['u_component_of_wind_10m', 'v_component_of_wind_10m']

                    output_file = 'data/' + '_'.join([variable, str(lon), str(lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                    if os.path.exists(output_file):
                        print(f"Time series already downloaded on: {output_file}")
                    else:
                        self.download_era5_land_data_by_years(era5_land_variables, lon, lat, start_date, end_date + timedelta(1))
                        
                        for year in range(start_date.year, end_date.year + 1):
                            cache_path = 'data/cache/' + '_'.join([str(s) for s in era5_land_variables] + [str(lon), str(lat), str(year)]) + '.csv'
                            temp_data = DataFrame(cache_path)
                            data.append_dataframe(temp_data.get_dataframe())
                            
                        if not 'u_component_of_wind_10m' in temp_data.get_columns_names() and 'v_component_of_wind_10m' in temp_data.get_columns_names():
                            print(f'No data found in GEE about {variable} for ({lon, lat})')
                            #cf = ClimateFiller()
                            #cf.download(variable, start_datetime, start_datetime + next_year, longitude, latitude)
                        else:
                            output_file = 'data/' + '_'.join([variable, str(lon), str(lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                            data.rename_columns({'u_component_of_wind_10m': 'u10', 'v_component_of_wind_10m': 'v10'})
                            data.add_column_based_on_function('era5_ws', Lib.get_2m_wind_speed)
                            data.drop_columns(['u10', 'v10'])
                            data.column_to_date('datetime')
                            data.reindex_dataframe('datetime')
                            end_date += timedelta(1)
                            data.select_datetime_range(start_date.isoformat(), end_date.isoformat())
                            data.export(output_file, index=True)
                            
                            if sequential_downloading is True:
                                if self.data.is_empty():
                                    self.data.set_dataframe(data.get_dataframe())
                                else:
                                    self.data.join(data.get_dataframe())

            else:
                
                if variable == 'ta':
                    era5_land_variables = ['2m_temperature']
                elif variable == 'rh':
                    era5_land_variables = ['2m_temperature', '2m_dewpoint_temperature']
                elif variable == 'rs':
                    era5_land_variables = ['surface_solar_radiation_downwards']
                elif variable == 'ws':
                    era5_land_variables = ['10m_u_component_of_wind', '10m_v_component_of_wind']
                
                    
                from data_science_toolkit.gis import GIS
                import cdsapi
                c = cdsapi.Client()
                

                if len(self.data.get_dataframe()) == 0:
                    # create the target time series
                    target_time_series = DataFrame.generate_datetime_range(start_date, end_date)
                    self.data.set_dataframe_index(target_time_series)
                    self.data.rename_index('datetime')
                    self.data.index_to_column()
                
                self.data.add_one_value_column(variable, None)
                self.fill(variable, lon, lat)
                self.export()
            
        elif product == 'mera':
            pass
        else:
            pass
    
    def export(self, path_link='data/climate_ts.csv', data_type='csv'):
        """
        Exports the processed data to a specified file or location.

        Args:
            self (object): The instance of the class.
            path_link (str): The path or link to export the processed data. Defaults to 'data/climate_ts.csv'.
            data_type (str): The type of the exported data. Defaults to 'csv'.

        Returns:
            None

        Notes:
            - The export method is used to save the processed data to a file or location.
            - The path_link parameter specifies the destination path or link for the exported data.
            - The data_type parameter indicates the format or type of the exported data, with 'csv' as the default value.
            - The processed data will be saved according to the specified file format and location.
            - The exported data can be used for further analysis, sharing, or storage.
        """
        self.data.export(path_link, data_type)
  
  
    def download_era5_land_data_by_years(self, variables, lon, lat, start_date, end_date):
        point = ee.Geometry.Point(lon, lat)
        era5_land = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterBounds(point)
        
        if isinstance(start_date, str) and isinstance(end_date, str):
            # Convert the start date and end date to datetime objects
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        for year in range(start_date.year, end_date.year + 1):
            cache_path = 'data/cache/' + '_'.join([str(s) for s in variables] + [str(lon), str(lat), str(year)]) + '.csv'
            
            if os.path.exists(cache_path):
                print(f"Time series already downloaded on: {cache_path}")
            else:
                # Filter the ERA5 land dataset by the year's date range
                era5_land_filtered = era5_land \
                    .filterDate(str(year) + '-01-01', str(year + 1) + '-01-01') \
                    .select(variables)

                # Download or perform further processing for the data for each year
                # Convert the image collection to a feature collection
                feature_collection = era5_land_filtered.map(lambda image: image.reduceRegions(reducer=ee.Reducer.first(), collection=ee.FeatureCollection(point)))

                # Flatten the feature collection
                flattened_collection = feature_collection.flatten()

                task = ee.batch.Export.table.toDrive( 
                    collection=flattened_collection,
                    description='ERA5_Land_Data',
                    fileFormat='CSV',
                    folder = 'era5_land_data'
                )
                task.start()
                
                # Export the TS to Loccal from Google Drive
                geemap.ee_export_vector(flattened_collection , filename=cache_path)
                temp_data = DataFrame(cache_path)
                temp_data.rename_columns({'system:index': 'datetime'})
                temp_data.column_to_date('datetime', extraction_func=self.extract_datetime)
                temp_data.export(cache_path)
                
                """
                if not 't2m' in temp_data.get_columns_names():
                                    print(f'No data found in GEE about {variable} for ({longitude, latitude})')
                                    cf = ClimateFiller()
                                    cf.download(variable, start_datetime, start_datetime + next_year, longitude, latitude)"""
        
        #output_path = 'data/' + '_'.join([str(s) for s in variables] + [str(lon), str(lat), str(actual_year)]) + '.csv'
        
    
    @staticmethod
    def extract_datetime(row):
        """
        Extracts the datetime value from a row of data.

        Args:
            row (object): A row of data from which the datetime value will be extracted.

        Returns:
            datetime: The extracted datetime value.

        Notes:
            - The extract_datetime static method is used to extract the datetime value from a row of data.
            - The row parameter represents a single row of data, which can be an object or a dictionary-like structure.
            - The method extracts and returns the datetime value from the specified row.
            - The datetime value is typically used for time-based operations, analysis, or visualization.
        """
        

        if isinstance(row, datetime.datetime):
            return row
        else:
            # Example string
            date_string = row

            # Define the regular expression pattern
            pattern = r'(\d{4})(\d{2})(\d{2})T(\d{2})_(\d{1})'

            # Match and extract the date components using the pattern
            match = re.match(pattern, date_string)

            # Extract the date components from the match
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            hour = int(match.group(4))
            minute = int(match.group(5))
            return datetime.datetime(year, month, day, hour, minute)
        
    def check_directory_existance(self, directory_path='data'):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory created: {directory_path}")
        


    """def learn_error(self,):
        data_x =
        data_y =

        model = model.best_model()

        model.train()
        
        for p in missing_data_row:
            data.set_row('', model.predict(data.get_row))"""