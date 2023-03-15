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
from lib import *

class ClimateFiller():
    """The ClimateFiller class
    """
    
    def __init__(self, data_link='data.csv', data_type='csv', datetime_column_name='date_time', date_time_format='%Y-%m-%d %H:%M:%S', machine_learning_enabled=False):
        self.data = DataFrame(data_link=data_link, data_type=data_type)
        self.data.column_to_date(datetime_column_name, date_time_format)
        self.datetime_column_name = datetime_column_name
        
    def show(self, number_of_row=None):
        if number_of_row is None:
            return self.data.get_dataframe()
        elif number_of_row < 0:
            return self.data.get_dataframe().tail(abs(number_of_row)) 
        else:
            return self.data.get_dataframe().head(number_of_row)
    
    def download_era5_land_dataframe(self):
        pass
    
    def recursive_fill(self, column_to_fill_name='ta', 
                              variable='Ta', 
                              latitude=31.66749781,
                              longitude=-7.593311291):
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
                              variable='air_temperature', 
                              datetime_column_name='date_time',
                              latitude=31.66749781,
                              longitude=-7.593311291
                              ):
        """Function Name: fill

            Description:
            This function fills missing values in a column of a dataframe with values from a weather API using latitude, longitude, and date-time information.

            Parameters:

            self: the instance of the class that the function is a part of.
            column_to_fill_name: (optional) the name of the column in the dataframe that will be filled with the weather data. Default value is 'ta'.
            variable: (optional) the name of the weather variable to be queried from the API. Default value is 'air_temperature'.
            datetime_column_name: (optional) the name of the column in the dataframe that contains the date-time information. Default value is 'date_time'.
            latitude: (optional) the latitude of the location for which weather data is being queried. Default value is 31.66749781.
            longitude: (optional) the longitude of the location for which weather data is being queried. Default value is -7.593311291.
            Returns:
            None. The function modifies the dataframe in place, filling in missing values with data from the weather API.

            Note:
            This function assumes that the dataframe already contains a column with the date-time information in the format YYYY-MM-DD HH:MM:SS. The function also requires an API key for the weather API, which must be set as an attribute of the class instance before calling the function. The API key can be obtained by registering for an account with the weather API provider.
                    """
        if self.missing_data_checking(column_to_fill_name, verbose=False) == 0:
            print('No missing data found in ' + column_to_fill_name)
            return
        
        if variable == 'Ta':
            era5_land_variables = ['2m_temperature']
        elif variable == 'Hr':
            era5_land_variables = ['2m_temperature', '2m_dewpoint_temperature']
        elif variable == 'Rg':
            era5_land_variables = ['surface_solar_radiation_downwards']
            
        from data_science_toolkit.gis import GIS
        import cdsapi
        c = cdsapi.Client()

        if datetime_column_name is not None:
            self.data.reindex_dataframe(datetime_column_name)

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
                    if os.path.exists("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + str(y) + '_' + month + '.grib') is False:
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
                        'era5_r3_' + column_to_fill_name + '_' + str(y) + '_' + month + '.grib')
                else:
                    for p in era5_land_variables:
                        if os.path.exists("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + str(p) + '_' + str(y) + '_' + month + '.grib') is False:
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
                            'era5_r3_' + column_to_fill_name + '_' + str(p) + '_' + str(y) + '_' + month + '.grib')
                    
        
        gis = GIS()
        data = DataFrame()
        
        if column_to_fill_name == 'Ta':
            for year in missing_data_dates:
                for month in missing_data_dates[year]['month']:
                    data.append_dataframe(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + str(year) + '_' + month + ".grib", "ta"),)
                    
            data.reset_index()
            data.reindex_dataframe("valid_time")
            data.missing_data('t2m')
            data.transform_column('t2m', 't2m', lambda o: o - 273.15)
            nan_indices = self.data.get_nan_indexes_of_column(column_to_fill_name)
            for p in nan_indices:
                self.data.set_row('Ta', p, data.get_row(p)['t2m'])
            print('Imputation of missing data for Ta from ERA5-Land was done!')
            
        elif column_to_fill_name == 'Hr':
            data_t2m = DataFrame()
            data_d2m = DataFrame()
            for year in missing_data_dates:
                for month in missing_data_dates[year]['month']:
                    data_t2m.append_dataframe(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + '2m_temperature' + '_' + str(year) + '_' + month + ".grib", "ta"),)
            for year in missing_data_dates:
                for month in missing_data_dates[year]['month']:
                    data_d2m.append_dataframe(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + '2m_dewpoint_temperature' + '_' + str(year) + '_' + month + ".grib", "ta"),)
            
            data_d2m.reset_index()
            data_d2m.reindex_dataframe("valid_time")
            data_d2m.keep_columns(['d2m'])
            data_t2m.reset_index()
            data_t2m.reindex_dataframe("valid_time")
            data_t2m.keep_columns(['t2m'])
            data_t2m.join(data_d2m.get_dataframe())
            data = data_t2m
            data.missing_data('t2m')
            data.transform_column('t2m', 't2m', lambda o: o - 273.15)
            data.transform_column('d2m', 'd2m', lambda o: o - 273.15)
            data.add_transformed_columns('era5_hr', '100*exp(-((243.12*17.62*t2m)-(d2m*17.62*t2m)-d2m*17.62*(243.12+t2m))/((243.12+t2m)*(243.12+d2m)))')
            data.missing_data('era5_hr')
            nan_indices = self.data.get_missing_data_indexes_in_column(column_to_fill_name)
            for p in nan_indices:
                self.data.set_row('Hr', p, data.get_row(p)['era5_hr'])
            
            print('Imputation of missing data for Hr from ERA5-Land was done!')
        elif column_to_fill_name == 'Rg':
            for year in missing_data_dates:
                for month in missing_data_dates[year]['month']:
                    data.append_dataframe(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + str(year) + '_' + month + ".grib", "ta"),)
                    
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
            data.add_column(l, 'rg')
            data.keep_columns(['rg'])
            data.rename_columns({'rg': 'ssrd'})
            print(data.show())
            
            data.transform_column('ssrd', 'ssrd', lambda o : o if abs(o) < 1500 else 0 )    
            data.export('rg.csv', index=True)
            nan_indices = self.data.get_nan_indexes_of_column(column_to_fill_name)
            for p in nan_indices:
                self.data.set_row('Rg', p, data.get_row(p)['ssrd'])
            
            print('Imputation of missing data for Rg from ERA5-Land was done!')
    

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
    
    def anomaly_detection(self, climate_varibale_column='Air temperature', method='knn', ):
        pass
    
    def eliminate_outliers(self, climate_varibale_column_name='ta', method='lof', n_neighbors=48, contamination=0.005):
        """Function Name: eliminate_outliers

            Description:
            This function eliminates outliers from a column of a dataframe using the Local Outlier Factor (LOF) algorithm, the Isolation Forest algorithm or quantiles.

            Parameters:

            self: the instance of the class that the function is a part of.
            climate_varibale_column_name: (optional) the name of the column in the dataframe that contains the climate variable. Default value is 'ta'.
            method: (optional) the name of the algorithm used to identify outliers. Possible values are 'lof' for Local Outlier Factor, and 'isolation_forest' for Isolation Forest. Default value is 'lof'.
            n_neighbors: (optional) the number of neighbors to consider when calculating the LOF score. This parameter is only used when method is set to 'lof'. Default value is 48.
            contamination: (optional) the proportion of outliers in the dataset. This parameter is only used when method is set to 'isolation_forest'. Default value is 0.005.
            Returns:
            A dataframe that contains the original data with the identified outliers removed.

            Note:
            This function assumes that the dataframe has already been loaded into the class instance. If the LOF algorithm is used, this function requires the scikit-learn library to be installed. If the Isolation Forest algorithm is used, this function requires the PyOD library to be installed.
        """
        if method == 'lof':
            outliers_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            self.data.get_dataframe()['inlier'] = outliers_model.fit_predict(self.data.get_columns([climate_varibale_column_name]))
            print('Number of detected outliers: {}'.format(self.data.count_occurence_of_each_row('inlier').iloc[0]))
            self.data.get_dataframe().loc[self.data.get_dataframe()['inlier'] == -1, climate_varibale_column_name] = None
            self.data.drop_column('inlier')
            #self.data.set_dataframe(self.data.get_dataframe().loc[self.data.get_dataframe().inlier == 1,
        elif method == 'isolation_forest':
            outliers_model = IsolationForest(contamination=contamination, random_state=42)
            self.data.get_dataframe()['inlier'] = outliers_model.fit_predict(self.data.get_columns([climate_varibale_column_name]))
            print('Number of detected outliers: {}'.format(self.data.count_occurence_of_each_row('inlier').iloc[0]))
            self.data.get_dataframe().loc[self.data.get_dataframe()['inlier'] == -1, climate_varibale_column_name] = 2000
            self.data.drop_column('inlier')
            #self.data.set_dataframe(self.data.get_dataframe().loc[self.data.get_dataframe().inlier == 1,
        elif method == 'quantiles':
            outliers_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            self.data.get_dataframe()['inlier'] = outliers_model.fit_predict(self.data.get_columns([climate_varibale_column_name]))
            print('Number of detected outliers: {}'.format(self.data.count_occurence_of_each_row('inlier').iloc[0]))
            self.data.get_dataframe().loc[self.data.get_dataframe()['inlier'] == -1, climate_varibale_column_name] = None
            self.data.drop_column('inlier')
            #self.data.set_dataframe(self.data.get_dataframe().loc[self.data.get_dataframe().inlier == 1,
                                                       # self.data.get_dataframe().columns.tolist()]) 
    
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
                       date_time_column_name='date_time',
                       latitude=31.65410805,
                       longitude=-7.603140831,
                       method='pm',
                       in_place=True
                       ):
        
        et0_data = DataFrame()
        et0_data.add_column('ta_mean', self.resample_timeseries(in_place=False)[air_temperture_column_name])
        et0_data.add_column('ta_max', self.resample_timeseries(in_place=False, agg='max')[air_temperture_column_name])
        et0_data.add_column('ta_min', self.resample_timeseries(in_place=False, agg='min')[air_temperture_column_name], )
        et0_data.add_column('rh_max', self.resample_timeseries(in_place=False, agg='max')[air_relative_humidity_column_name])
        et0_data.add_column('rh_min', self.resample_timeseries(in_place=False, agg='min')[air_relative_humidity_column_name])
        et0_data.add_column('rh_mean', self.resample_timeseries(in_place=False)[air_relative_humidity_column_name])
        et0_data.add_column('u2_mean', self.resample_timeseries(in_place=False)[wind_speed_column_name])
        et0_data.add_column('rg_mean', self.resample_timeseries(in_place=False)[global_solar_radiation_column_name])
        et0_data.index_to_column()
        et0_data.add_doy_column('date_time')
        et0_data.add_one_value_column('elevation', get_elevation_and_latitude(latitude, longitude))
        et0_data.add_one_value_column('lat', latitude)
        
        if method == 'pm':
            et0_data.add_column_based_on_function('et0_pm', et0_penman_monteith)
        elif method == 'hargreaves':
            et0_data.add_column_based_on_function('et0_hargreaves', et0_hargreaves)
            
        if in_place == True:
            self.__dataframe = et0_data.get_dataframe()
            
        return et0_data.get_dataframe()
    
    def apply_quality_control_criteria(self, variable_column_name, decision_func=lambda x:x>0):
        self.data.add_column(self.data.get_column(variable_column_name).apply(decision_func), 'decision')
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
                     column_to_fill='Ta', date_column_name=None):
        """Function Name: missing_data

            Description:
            This function fills or drops missing data in a dataframe.

            Parameters:

            self: the instance of the class that the function is a part of.
            drop_row_if_nan_in_column: (optional) the name of a column in the dataframe. If provided, the function will drop rows where this column contains NaN values. Default value is None.
            filling_dict_colmn_val: (optional) a dictionary containing column names and values to be used to fill missing data in those columns. The keys of the dictionary should be the names of the columns to be filled, and the values should be the values to use for filling. Default value is None.
            method: (optional) a string that determines the method used for filling missing data. Possible values are 'ffill' for forward filling, 'bfill' for backward filling, or 'interpolate' for linear interpolation. Default value is 'ffill'.
            column_to_fill: (optional) the name of the column to be filled. This parameter is only used when method is set to 'ffill' or 'bfill'. Default value is 'Ta'.
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
    start_datetime,
    end_datetime,
    latitude=31.66749781,
    longitude=-7.593311291,
    product='era5-Land'):
        if product == 'era5-Land':
            if variable == 'Ta':
                era5_land_variables = ['2m_temperature']
            elif variable == 'Hr':
                era5_land_variables = ['2m_temperature', '2m_dewpoint_temperature']
            elif variable == 'Rg':
                era5_land_variables = ['surface_solar_radiation_downwards']
                
            from data_science_toolkit.gis import GIS
            import cdsapi
            c = cdsapi.Client()

            # create the target time series
            target_time_series = datetime.datetime()

            for y in target_time_series.years:
                for month in target_time_series.months:
                    if len(era5_land_variables) == 1:
                        if os.path.exists("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + str(y) + '_' + month + '.grib') is False:
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
                            'era5_r3_' + column_to_fill_name + '_' + str(y) + '_' + month + '.grib')
                    else:
                        for p in era5_land_variables:
                            if os.path.exists("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + str(p) + '_' + str(y) + '_' + month + '.grib') is False:
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
                                'era5_r3_' + column_to_fill_name + '_' + str(p) + '_' + str(y) + '_' + month + '.grib')
                        
            
            gis = GIS()
            data = DataFrame()
            
            if column_to_fill_name == 'Ta':
                for year in target_time_series.years:
                    for month in target_time_series.months:
                        data.append_dataframe(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + str(year) + '_' + month + ".grib", "ta"),)
                        
                data.reset_index()
                data.reindex_dataframe("valid_time")
                data.missing_data('t2m')
                data.transform_column('t2m', 't2m', lambda o: o - 273.15)
                
            elif column_to_fill_name == 'Hr':
                data_t2m = DataFrame()
                data_d2m = DataFrame()
                for year in missing_data_dates:
                    for month in missing_data_dates[year]['month']:
                        data_t2m.append_dataframe(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + '2m_temperature' + '_' + str(year) + '_' + month + ".grib", "ta"),)
                for year in missing_data_dates:
                    for month in missing_data_dates[year]['month']:
                        data_d2m.append_dataframe(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + '2m_dewpoint_temperature' + '_' + str(year) + '_' + month + ".grib", "ta"),)
                
                data_d2m.reset_index()
                data_d2m.reindex_dataframe("valid_time")
                data_d2m.keep_columns(['d2m'])
                data_t2m.reset_index()
                data_t2m.reindex_dataframe("valid_time")
                data_t2m.keep_columns(['t2m'])
                data_t2m.join(data_d2m.get_dataframe())
                data = data_t2m
                data.missing_data('t2m')
                data.transform_column('t2m', 't2m', lambda o: o - 273.15)
                data.transform_column('d2m', 'd2m', lambda o: o - 273.15)
                data.add_transformed_columns('era5_hr', '100*exp(-((243.12*17.62*t2m)-(d2m*17.62*t2m)-d2m*17.62*(243.12+t2m))/((243.12+t2m)*(243.12+d2m)))')
                data.missing_data('era5_hr')
                nan_indices = self.data.get_missing_data_indexes_in_column(column_to_fill_name)
                for p in nan_indices:
                    self.data.set_row('Hr', p, data.get_row(p)['era5_hr'])
                
                print('Imputation of missing data for Hr from ERA5-Land was done!')
            elif column_to_fill_name == 'Rg':
                for year in missing_data_dates:
                    for month in missing_data_dates[year]['month']:
                        data.append_dataframe(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + str(year) + '_' + month + ".grib", "ta"),)
                        
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
                data.add_column(l, 'rg')
                data.keep_columns(['rg'])
                data.rename_columns({'rg': 'ssrd'})
                print(data.show())
                
                data.transform_column('ssrd', 'ssrd', lambda o : o if abs(o) < 1500 else 0 )    
                data.export('rg.csv', index=True)
                nan_indices = self.data.get_nan_indexes_of_column(column_to_fill_name)
                for p in nan_indices:
                    self.data.set_row('Rg', p, data.get_row(p)['ssrd'])
                
                print('Imputation of missing data for Rg from ERA5-Land was done!')
        elif product == 'mera':
            pass
        else:
            pass
    
    def reference_evapotranspiration(self, climate_variables_path='data/in_situ_data.xls', data_type='xls', method='pm'):
        pass


    """def learn_error(self,):
        data_x =
        data_y =

        model = model.best_model()

        model.train()
        
        for p in missing_data_row:
            data.set_row('', model.predict(data.get_row))"""