from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 
from catboost import CatBoostRegressor
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from data_science_toolkit.dataframe import DataFrame
import datetime
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from lib import Lib
import warnings
import cdsapi
import xarray as xr
import glob 
warnings.filterwarnings('ignore')


class ClimateFiller():
    """The ClimateFiller class
    """
    
    def __init__(self, data_link=None, data_type='csv', datetime_column_name='date_time', date_time_format='%Y-%m-%d %H:%M:%S'):
        self.datetime_column_name = datetime_column_name
        self.data_reanalysis = DataFrame()
        if data_link is None:
            self.data = DataFrame()
        else:
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
                              variable='ta', 
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
                              latitude=31.66749781,
                              longitude=-7.593311291,
                              product="era5-Land",
                              machine_learning_enabled=False,
                              backend=None,
                              ):    
        '''
        Function Name: fill

            Description:
            This function fills missing values in a column of a dataframe with values from a weather API using latitude, longitude, and date-time information.

            Parameters:

            self: the instance of the class that the function is a part of.
            column_to_fill_name: (optional) the name of the column in the dataframe that will be filled with the weather data. Default value is 'ta'.
            variable: (optional) the name of the weather variable to be queried from the API. Default value is 'air_temperature'.
            datetime_column_name: (optional) the name of the column in the dataframe that contains the date-time information. Default value is 'date_time'.
            latitude: (optional) the latitude of the location for which weather data is being queried. Default value is 31.66749781.
            longitude: (optional) the longitude of the location for which weather data is being queried. Default value is -7.593311291.
            product: (optional) the name of the weather product to be queried from the API. Default value is 'era5-Land'.
            machine_learning_enabled: (optional) a boolean value indicating whether machine learning should be used to fill missing values. Default value is False.
            backend: (optional) the name of the backend to be used for querying the weather API. Default value is None.
            Returns:
            None. The function modifies the dataframe in place, filling in missing values with data from the weather API.

            Note:
            This function assumes that the dataframe already contains a column with the date-time information in the format YYYY-MM-DD HH:MM:SS. The function also requires an API key for the weather API, which must be set as an attribute of the class instance before calling the function. The API key can be obtained by registering for an account with the weather API provider.
        '''        
        if product == 'era5-Land':

            if self.missing_data_checking(column_to_fill_name, verbose=False) == 0:
                print('No missing data found in ' + column_to_fill_name)
                return
            
            if backend == 'gee':

                era5_land_variables = {
                    'ta': ([['temperature_2m']], ['t2m']),
                    'rh': ([['temperature_2m', 'dewpoint_temperature_2m']], ['t2m', 'd2m']),
                    'rs': (['surface_solar_radiation_downwards'], ['ssrd']),
                    'ws': ([['u_component_of_wind_10m', 'v_component_of_wind_10m']], ['u10', 'v10'])
                }

                start = self.data.get_dataframe().datetime[0]
                end = self.data.get_dataframe().datetime[len(self.data.get_dataframe().datetime)-1]

                start = datetime.datetime.strftime(start, '%Y-%m-%d')
                end = datetime.datetime.strftime(end, '%Y-%m-%d')

                self.download(column_to_fill_name , start, end, latitude, longitude , product , backend)

                if self.datetime_column_name is not None:
                    self.data.reindex_dataframe(self.datetime_column_name)

                nan_indices = self.data.get_missing_data_indexes_in_column(column_to_fill_name)
                if len(nan_indices) == 0 :
                    print("No missing data found.")
                    return

                target_time_series = pd.DataFrame(nan_indices, columns=['datetime'])

                df = self.data_reanalysis.get_dataframe()
                print(df)
                df.sort_values(by=['datetime'], inplace=True)

                merged_df = pd.merge(target_time_series, df, how='left')

                variable, columns_var = era5_land_variables.get(column_to_fill_name, (None, None))
                
                if variable is None or columns_var is None:
                    print(f'Invalid column_to_fill_name: {column_to_fill_name}')
                    return

                if machine_learning_enabled:
                    self.function_cross_validation(column_to_fill_name, latitude, longitude, product)
                    for _, p in merged_df.iterrows():
                        if p['datetime'] in nan_indices:
                            row_data = p[columns_var].values.reshape(1, -1)
                            prediction = self.best_model.predict(row_data)
                            self.data.set_row(column_to_fill_name, p['datetime'], prediction[0])
                else:
                    for _, p in merged_df.iterrows():
                        if p['datetime'] in nan_indices:
                            corresponding_row = merged_df.loc[merged_df['datetime'] == p['datetime'], columns_var]
                            if not corresponding_row.empty:
                                self.data.set_row(column_to_fill_name, p['datetime'], corresponding_row.values[0])
                
            elif backend is None:

                era5_land_variables = {
                    'ta': (['2m_temperature'], ['t2m']),
                    'rh': (['2m_temperature', '2m_dewpoint_temperature'], ['t2m', 'd2m']),
                    'rs': (['surface_solar_radiation_downwards'], ['ssrd']),
                    'ws': (['10m_u_component_of_wind', '10m_v_component_of_wind'], ['u10', 'v10'])
                }

                variable, columns_var = era5_land_variables.get(column_to_fill_name, (None, None))
                if variable is None or columns_var is None:
                    print(f'Invalid column_to_fill_name: {column_to_fill_name}')
                    return

                c = cdsapi.Client()

                if self.datetime_column_name is not None:
                    self.data.reindex_dataframe(self.datetime_column_name)

                nan_indices = self.data.get_missing_data_indexes_in_column(column_to_fill_name)

                target_time_series = pd.DataFrame(nan_indices, columns=['datetime'])
                years = target_time_series['datetime'].dt.year.unique()

                for year in years:
                    months = target_time_series[target_time_series['datetime'].dt.year == year]['datetime'].dt.month.unique()
                    for month in months:
                        days = target_time_series[(target_time_series['datetime'].dt.year == year) &
                                                (target_time_series['datetime'].dt.month == month)]['datetime'].dt.day.unique()

                        file_path = f'./data/era5-Land_{column_to_fill_name}-{str(longitude)}-{str(latitude)}-{year}-{month}.nc'
                        if os.path.exists(file_path):
                            continue

                        c.retrieve(
                            'reanalysis-era5-land',
                            {
                                'variable': variable,
                                'year': str(year),
                                'month': str(month).zfill(2),
                                'day': [str(day).zfill(2) for day in days],
                                'time': [
                                    f'{hour:02d}:00' for hour in range(24)
                                ],
                                'area': [
                                    latitude, longitude, latitude, longitude
                                ],
                                'format': 'netcdf'
                            },
                            file_path
                        )

                files = glob.glob('./data/era5-Land_{}-{}-{}*.nc'.format(column_to_fill_name, str(longitude), str(latitude)))
                dfs = []

                for file in files:
                    with xr.open_dataset(file) as ds:
                        data = ds.to_dataframe().reset_index()
                        selected_columns = ['time'] + columns_var
                        df = data[selected_columns]
                        dfs.append(df)

                df = pd.concat(dfs)
                df.sort_values(by=['time'], inplace=True)

                merged_df = pd.merge(target_time_series, df, left_on='datetime', right_on='time', how='left')

                if len(nan_indices) > 0:
                    if machine_learning_enabled:
                        self.function_cross_validation(column_to_fill_name, latitude, longitude, product)
                        for _, p in merged_df.iterrows():
                            if p['time'] in nan_indices:
                                row_data = p[columns_var].values.reshape(1, -1)
                                prediction = self.best_model.predict(row_data)
                                self.data.set_row(column_to_fill_name, p['time'], prediction[0])
                    else:
                        for _, p in merged_df.iterrows():
                            if p['time'] in nan_indices:
                                corresponding_row = merged_df.loc[merged_df['time'] == p['time'], columns_var]
                                if not corresponding_row.empty:
                                    self.data.set_row(column_to_fill_name, p['time'], corresponding_row.values[0])

            self.data.index_to_column()
            print('Imputation of missing data for ' + column_to_fill_name + ' from ERA5_Land was done!')

            
        elif product == 'merra2':
            if self.missing_data_checking(column_to_fill_name, verbose=False) == 0:
                print('No missing data found in ' + column_to_fill_name)
                return

            merra2_variables = {
                'ta': 'T2M',
                'rh': 'RH2M',
                'ws': 'WS2M',
                'rs': 'ALLSKY_SFC_SW_DWN',
                'pr': 'PRECTOTCORR',
                'wd': 'WD2M'
            }

            if column_to_fill_name not in merra2_variables:
                print(f'Invalid column_to_fill_name: {column_to_fill_name}')
                return

            if self.datetime_column_name is not None:
                self.data.reindex_dataframe(self.datetime_column_name)

            start = self.data.get_dataframe().index[0]
            end = self.data.get_dataframe().index[-1]
            start = datetime.datetime.strftime(start, '%Y%m%d')
            end = datetime.datetime.strftime(end, '%Y%m%d')

            api_url = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
            format = 'json'
            community = 'ag'
            timezone = 'utc'

            params = {
                'start': start,
                'end': end,
                'latitude': latitude,
                'longitude': longitude,
                'community': community,
                'parameters': merra2_variables[column_to_fill_name],
                'format': format,
                'user': 'ysouidi1',
                'header': 'true',
                'time-standard': timezone
            }

            response = requests.get(api_url, params=params)

            if response.status_code != 200:
                print('Failed to retrieve data:', response.status_code)
                return None

            data_merra = response.json()
            result = data_merra['properties']['parameter'][merra2_variables[column_to_fill_name]]
            df = pd.DataFrame(result.items(), columns=['datetime', column_to_fill_name])
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H')

            nan_indices = self.data.get_missing_data_indexes_in_column(column_to_fill_name)

            if len(nan_indices) == 0:
                return

            if machine_learning_enabled:
                self.function_cross_validation(column_to_fill_name, latitude, longitude, product)
                for p in nan_indices:
                    if not df.loc[df['datetime'] == p, column_to_fill_name].empty:
                        prediction = self.best_model.predict(df.loc[df['datetime'] == p, column_to_fill_name].values[0].reshape(1, -1))
                        self.data.set_row(column_to_fill_name, p, prediction)
            else:
                for p in nan_indices:
                    corresponding_row = df[df['datetime'] == p][column_to_fill_name]
                    if not corresponding_row.empty:
                        self.data.set_row(column_to_fill_name, p, df.loc[df['datetime'] == p, column_to_fill_name].values[0])

            self.data.index_to_column()
            print('Imputation of missing data for ' + column_to_fill_name + ' from MERRA2 was done!')

        else:
            pass


    def function_cross_validation(self, column_to_fill_name, latitude, longitude, product):
        list_models = [
            LinearRegression(),
            DecisionTreeRegressor(),
            RandomForestRegressor(),
            XGBRegressor(),
            CatBoostRegressor(verbose=False),
            Ridge(),
            Lasso(),
            ElasticNet()
        ]

        start_datetime = self.data.get_dataframe().index[0]
        end_datetime = self.data.get_dataframe().index[-1]

        self.download(column_to_fill_name, start_datetime, end_datetime, latitude, longitude, product)

        X = self.data_reanalysis.get_dataframe()
        X_index_column = 'datetime' if product == 'merra2' else 'time'
        X = X.set_index(X_index_column)

        y = self.data.get_dataframe()[column_to_fill_name]
        nan_indices = self.data.get_missing_data_indexes_in_column(column_to_fill_name)
        y = y.drop(nan_indices)

        joined_data = pd.merge(X, y, left_index=True, right_index=True)
        X = joined_data.drop([column_to_fill_name], axis=1)
        y = joined_data[column_to_fill_name]

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        rmse_scores = {}
        r2_scores = {}

        for model in list_models:
            model_rmse_scores = []
            model_r2_scores = []

            for train_index, test_index in kfold.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                model_rmse_scores.append(rmse)
                model_r2_scores.append(r2)

            avg_rmse = np.mean(model_rmse_scores)
            avg_r2 = np.mean(model_r2_scores)

            rmse_scores[model.__class__.__name__] = avg_rmse
            r2_scores[model.__class__.__name__] = avg_r2

        best_model_name = min(rmse_scores, key=rmse_scores.get)
        best_model = next((model for model in list_models if model.__class__.__name__ == best_model_name), None)
        self.best_model = best_model

    def missing_data_checking(self, column_name=None, verbose=True):
        if column_name is not None:
            column_data = self.data.get_dataframe()[column_name]
            missing_count = column_data.isnull().sum()
            if verbose:
                if missing_count > 0:
                    missing_percent = round((missing_count / self.data.get_shape()[0]) * 100, 2)
                    print("{} has {} missing value(s) which represents {}% of the dataset size".format(column_name, missing_count, missing_percent))
                else:
                    print("No missing data in column " + column_name)
        else:
            missing_counts = self.data.get_dataframe().isnull().sum()
            for c, missing_count in missing_counts.items():
                if verbose:
                    if missing_count > 0:
                        missing_percent = round((missing_count / self.data.get_shape()[0]) * 100, 2)
                        print("{} has {} missing value(s) which represents {}% of the dataset size".format(c, missing_count, missing_percent))
                    else:
                        print("{} has NO missing value!".format(c))
            if not verbose:
                return missing_counts.tolist()

    def anomaly_detection(self, climate_varibale_column='Air temperature', method='knn', ):
        pass
    
    def eliminate_outliers(self, climate_varibale_column_name='ta', method='lof', n_neighbors=48, contamination=0.005, n_estimators=100):
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
                       date_time_column_name='date_time',
                       latitude=31.65410805,
                       longitude=-7.603140831,
                       method='pm',
                       in_place=True
                       ):
        
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
        et0_data.add_doy_column('date_time')
        et0_data.add_one_value_column('elevation',(latitude, longitude))
        et0_data.add_one_value_column('lat', latitude)
        
        if method == 'pm':
            et0_data.add_column_based_on_function('et0_pm', Lib.et0_penman_monteith)
        elif method == 'hargreaves':
            et0_data.add_column_based_on_function('et0_hargreaves', Lib.et0_hargreaves)
            
        if in_place == True:
            self.data = et0_data
            
        return et0_data.get_dataframe()
    
    def apply_quality_control_criteria(self, variable_column_name, decision_func=lambda x:x>0):
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

    def download(self, variable, start_datetime, end_datetime, latitude=31.66749781, longitude=-7.593311291, product='era5-Land',
                 backend=None):
        if product == "era5-Land":
            if backend is None:
                era5_land_variables = {
                    'ta': {'variables': ['2m_temperature'], 'columns': ['t2m']},
                    'rh': {'variables': ['2m_temperature', '2m_dewpoint_temperature'], 'columns': ['t2m', 'd2m']},
                    'rs': {'variables': ['surface_solar_radiation_downwards'], 'columns': ['ssrd']},
                    'ws': {'variables': ['10m_u_component_of_wind', '10m_v_component_of_wind'], 'columns': ['u10', 'v10']}
                }

                if variable not in era5_land_variables:
                    print("Variable '{}' is not supported for 'era5-Land'".format(variable))
                    return

                era5_land_data = era5_land_variables[variable]

                era5_land_vars = era5_land_data['variables']
                columns = era5_land_data['columns']

                target_time_series = pd.date_range(start=start_datetime, end=end_datetime, freq='D')

                for date in target_time_series:
                    year = date.year
                    month = date.month
                    day = date.day

                    file_path = './data/era5-Land_{}-{}-{}-{}-{}.nc'.format(variable,str(longitude) , str(latitude),year, month)
                    if len(glob.glob(file_path)) > 0:
                        continue

                    c = cdsapi.Client()
                    c.retrieve(
                        'reanalysis-era5-land',
                        {
                            'variable': era5_land_vars,
                            'year': str(year),
                            'month': str(month).zfill(2),
                            'day': str(day).zfill(2),
                            'time': [
                                '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                                '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                                '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                                '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
                            ],
                            'area': [
                                latitude, longitude, latitude, longitude
                            ],
                            'format': 'netcdf'
                        },
                        file_path
                    )

                files = glob.glob('./data/era5-Land_{}-{}-{}-*.nc'.format(variable , str(longitude) , str(latitude)))
                dfs = []

                for file in files:
                    ds = xr.open_dataset(file)
                    data = ds.to_dataframe()
                    data = data.reset_index()

                    selected_columns = ['time'] + columns
                    df = data[selected_columns]

                    dfs.append(df)

                df = pd.concat(dfs)
                df.sort_values(by=['time'], inplace=True)
                df = df[(df['time'] >= start_datetime) & (df['time'] <= end_datetime)]

                if variable == 'ta':
                    df['t2m'] = df['t2m'] - 273.15
                    self.data_reanalysis.set_dataframe(df)
                    print('Downloaded data from ERA5-Land for temperature is done')

                elif variable == 'rh':
                    df['t2m'] = df['t2m'] - 273.15
                    df['d2m'] = df['d2m'] - 273.15
                    df['era5_rh'] = 100 * np.exp(-((243.12 * 17.62 * df['t2m']) - (df['d2m'] * 17.62 * df['t2m']) - df['d2m'] * 17.62 * (243.12 + df['t2m'])) / ((243.12 + df['t2m']) * (243.12 + df['d2m'])))
                    df = df.drop(['t2m', 'd2m'], axis=1)
                    self.data_reanalysis.set_dataframe(df)
                    print('Downloaded data from ERA5-Land for relative humidity is done')

                elif variable == 'ws':
                    df['era5_ws'] = df.apply(self.get_2m_wind_speed, axis=1)
                    df = df.drop(['u10', 'v10'], axis=1)
                    self.data_reanalysis.set_dataframe(df)
                    print('Downloaded data from ERA5-Land for wind speed is done')

                elif variable == 'rs':
                    df['ssrd'] = df['ssrd'] / 3600
                    df['ssrd'] = df['ssrd'].apply(lambda o: o if abs(o) < 1500 else 0)
                    self.data_reanalysis.set_dataframe(df)
                    print('Downloaded data from ERA5-Land for solar radiation is done')

            elif backend == 'gee' :
                import ee# Initialize Earth Engine
                service_account = 'climatefiller@climatefiller.iam.gserviceaccount.com'
                credentials = ee.ServiceAccountCredentials(service_account,"climatefiller_credentials.json")
                ee.Initialize(credentials)
                ee.Initialize()
                point = ee.Geometry.Point(longitude, latitude)

                era5_land_variables = {
                    'ta': {'variables': ['temperature_2m'], 'columns': ['t2m']},
                    'rh': {'variables': ['temperature_2m', 'dewpoint_temperature_2m'], 'columns': ['t2m', 'd2m']},
                    'rs': {'variables': ['surface_solar_radiation_downwards'], 'columns': ['ssrd']}, 
                    'ws': {'variables': ['u_component_of_wind_10m', 'v_component_of_wind_10m'], 'columns': ['u10', 'v10']}
                } 

                variables_info = era5_land_variables[variable]
                bands = variables_info['variables']
                columns = variables_info['columns']

                def get_values(image):
                    values = ee.List(bands).map(
                        lambda band: image.reduceRegion(
                            reducer=ee.Reducer.first(),
                            geometry=point,
                            scale=1000  # Adjust the scale as needed
                        ).get(band)
                    )

                    return image.set('values', values)

                dfs = []
                start_datetime = datetime.datetime.strptime(start_datetime, '%Y-%m-%d')
                end_datetime = datetime.datetime.strptime(end_datetime, '%Y-%m-%d')
                start_year = start_datetime.year
                end_year = end_datetime.year

                for year in range(start_year, end_year + 1):
                    start_date = f"{year}-01-01"
                    end_date = f"{year+1}-01-01"

                    image_collection = (
                        ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
                        .filterBounds(point)
                        .filterDate(start_date, end_date)
                    )


                    # Map the get_values function over the image collection
                    collection_with_values = image_collection.map(get_values)

                    # Reduce regions and retrieve the values and datetimes
                    values_dict = collection_with_values.aggregate_array('values').getInfo()
                    datetimes = pd.to_datetime(collection_with_values.aggregate_array('system:time_start').getInfo(), unit='ms')

                    # Create a dictionary to store the values for each band
                    values = {band: [] for band in bands}

                    # Extract the values for each band from the dictionary
                    for i, band in enumerate(bands):
                        values[band] = [d[i] for d in values_dict]

                    # Create a pandas DataFrame
                    df = pd.DataFrame(values)
                    
                    df.columns = variables_info['columns']

                    df['datetime'] = datetimes

                    dfs.append(df)

                df = pd.concat(dfs)
                
                # Filter the DataFrame based on the start and end datetime
                df = df[(df['datetime'] >= start_datetime) &(df['datetime'] <= end_datetime)]

                if variable == 'ta':
                            df['t2m'] = df['t2m'] - 273.15
                            print('Downloaded data from ERA5-Land for temperature is done')

                elif variable == 'rh':
                    df['t2m'] = df['t2m'] - 273.15
                    df['d2m'] = df['d2m'] - 273.15
                    df['era5_rh'] = 100 * np.exp(-((243.12 * 17.62 * df['t2m']) - (df['d2m'] * 17.62 * df['t2m']) - df['d2m'] * 17.62 * (243.12 + df['t2m'])) / ((243.12 + df['t2m']) * (243.12 + df['d2m'])))
                    df = df.drop(['t2m', 'd2m'], axis=1)
                    print('Downloaded data from ERA5-Land for relative humidity is done')

                elif variable == 'ws':
                    df['era5_ws'] = df.apply(Lib.get_2m_wind_speed, axis=1)
                    df = df.drop(['u10', 'v10'], axis=1)
                    print('Downloaded data from ERA5-Land for wind speed is done')

                elif variable == 'rs':
                    df['ssrd'] = df['ssrd'] / 3600
                    df['ssrd'] = df['ssrd'].apply(lambda o: o if abs(o) < 1500 else 0)
                    print('Downloaded data from ERA5-Land for solar radiation is done')
                
                self.data_reanalysis.set_dataframe(df)

        elif product == 'merra2':
            merra2_variables = {
                'ta': 'T2M',
                'rh': 'RH2M',
                'rs': 'ALLSKY_SFC_SW_DWN',
                'ws': 'WS2M',
                'pr': 'PRECTOTCORR',
                'wd': 'WD2M'
            }

            if variable not in merra2_variables:
                print("Variable '{}' is not supported for 'merra2'".format(variable))
                return

            merra2_variable = merra2_variables[variable]

            api_url = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
            start = start_datetime.strftime('%Y%m%d')
            end = end_datetime.strftime('%Y%m%d')
            format = 'json'
            community = 'ag'
            timezone = 'utc'
            params = {
                'start': start,
                'end': end,
                'latitude': latitude,
                'longitude': longitude,
                'community': community,
                'parameters': merra2_variable,
                'format': format,
                'user': 'ysouidi1',
                'header': 'true',
                'time-standard': timezone
            }

            response = requests.get(api_url, params=params)

            if response.status_code != 200:
                print('Failed to retrieve data:', response.status_code)
                return None

            data_merra = response.json()
            result = data_merra['properties']['parameter'][merra2_variable]
            df = pd.DataFrame(result.items(), columns=['datetime', merra2_variable])
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H')
            print('Downloaded data from Merra2 for {} is done'.format(variable))
            self.data_reanalysis.set_dataframe(df)

        else:
            print("Invalid product '{}'".format(product))
    
    def export(self, path_link='data/climate_ts.csv', data_type='csv'):
        self.data.export(path_link, data_type)