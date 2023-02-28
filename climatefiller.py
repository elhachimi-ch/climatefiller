from gis import GIS
from dataframe import DataFrame
import datetime
import os
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from quantilesdetector import PercentileDetection

class ClimateFiller():
    
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
    
    def download_era5(self):
        pass
    
    """def rec_fill(self, column_to_fill_name='ta', 
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
                                                                                 longitude))"""
    
    
    
    def fill(self, column_to_fill_name='ta', 
                              variable='air_temperature', 
                              datetime_column_name='date_time',
                              latitude=31.66749781,
                              longitude=-7.593311291
                              ):
        if self.missing_data_checking(column_to_fill_name, verbose=False) == 0:
            print('No missing data found in ' + column_to_fill_name)
            return
        
        if variable == 'Ta':
            era5_land_variables = ['2m_temperature']
        elif variable == 'Hr':
            era5_land_variables = ['2m_temperature', '2m_dewpoint_temperature']
        elif variable == 'Rg':
            era5_land_variables = ['surface_solar_radiation_downwards']
            
        from gis import GIS
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
        self.data.get_column(column).plot()
        plt.show()
    
    def apply_quality_control_criteria(self, variable_column_name, decision_func=lambda x:x>0):
        self.data.add_column(self.data.get_column(variable_column_name).apply(decision_func), 'decision')
        self.data.get_dataframe().loc[ self.data.get_dataframe()['decision'] == False, variable_column_name] = None
        self.data.drop_column('decision')
        
    def apply_constraint(self, column_name, constraint):
            self.data.filter_dataframe(column_name, constraint)
    
    def missing_data(self, drop_row_if_nan_in_column=None, filling_dict_colmn_val=None, method='ffill',
                     column_to_fill='Ta', date_column_name=None):
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
          