from gis import GIS
from dataframe import DataFrame
import datetime
import os
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor


class ClimateFiller():
    
    def __init__(self, data_link='data.csv', data_type='csv'):
        self.data = DataFrame(data_link=data_link, data_type=data_type)
        
    def show(self, number_of_row=None):
        if number_of_row is None:
            return self.data.get_dataframe()
        elif number_of_row < 0:
            return self.data.get_dataframe().tail(abs(number_of_row)) 
        else:
            return self.data.get_dataframe().head(number_of_row)
    
    def download_era5(self):
        pass
    
    def fill(self, column_to_fill_name='ta', 
                              variable='air_temperature', 
                              datetime_column_name='date_time',
                              latitude=31.66749781,
                              longitude=-7.593311291
                              ):
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

            if os.path.exists("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + str(y) + ".grib") is False:
                c.retrieve(
                    'reanalysis-era5-land',
                    {
                        'format': 'grib',
                        'variable': era5_land_variables,
                        'year': str(y),
                        'month':  missing_data_dict['month'],
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
                    'era5_r3_' + column_to_fill_name + '_' + str(y) +'.grib')
        
        gis = GIS()

        data = DataFrame(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + str(years[0]) + ".grib", "ta"),
                        data_type="df")
        data.reset_index()
        data.resample_timeseries(skip_rows=2)
        data.reindex_dataframe("valid_time")
        if column_to_fill_name == 'Ta':
            data.keep_columns(['t2m'])
            for y in years[1:]:
                data_temp = DataFrame(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + str(y) + ".grib", "ta"),
                                    data_type='df')
                data_temp.reset_index()
                data_temp.resample_timeseries(skip_rows=2)
                data_temp.reindex_dataframe("valid_time")
                data_temp.keep_columns(['t2m'])
                data.append_dataframe(data_temp.get_dataframe())
            
            data.transform_column('t2m', 't2m', lambda o: o - 273.15)
            nan_indices = self.data.get_nan_indexes_of_column(column_to_fill_name)
            for p in nan_indices:
                self.data.set_row('Ta', p, data.get_row(p)['t2m'])
            print('Imputation of missing data for Ta from ERA5-Land was done!')
            
        elif column_to_fill_name == 'Hr':
            data.keep_columns(['t2m', 'd2m'])
            for y in years[1:]:
                data_temp = DataFrame(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + str(y) + ".grib", "ta"),
                                    data_type='df')
                data_temp.reset_index()
                data_temp.resample_timeseries(skip_rows=2)
                data_temp.reindex_dataframe("valid_time")
                data_temp.keep_columns(['t2m', 'd2m'])
                data.append_dataframe(data_temp.get_dataframe())
            data.transform_column('t2m', 't2m', lambda o: o - 273.15)
            data.transform_column('d2m', 'd2m', lambda o: o - 273.15)
            data.add_transformed_columns('era5_hr', '100*exp(-((243.12*17.62*t2m)-(d2m*17.62*t2m)-d2m*17.62*(243.12+t2m))/((243.12+t2m)*(243.12+d2m)))')
            nan_indices = self.data.get_missing_data_indexes_in_column(column_to_fill_name)
            for p in nan_indices:
                self.data.set_row('Hr', p, data.get_row(p)['era5_hr'])
            
            print('Imputation of missing data for Hr from ERA5-Land was done!')
        elif column_to_fill_name == 'Rg':
            data.keep_columns(['ssrd'])
            for y in years[1:]:
                data_temp = DataFrame(gis.get_era5_land_grib_as_dataframe("E:\projects\pythonsnippets\era5_r3_" + column_to_fill_name + '_' + str(y) + ".grib", "ta"),
                                    data_type='df')
                data_temp.reset_index()
                data_temp.resample_timeseries(skip_rows=2)
                data_temp.reindex_dataframe("valid_time")
                data_temp.keep_columns(['ssrd'])
                data.append_dataframe(data_temp.get_dataframe())
            
            
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
            data.transform_column('ssrd', 'ssrd', lambda o : o if abs(o) < 1500 else 0 )    
            data.export('rg.csv', index=True)
            nan_indices = self.data.get_nan_indexes_of_column(column_to_fill_name)
            for p in nan_indices:
                self.data.set_row('Rg', p, data.get_row(p)['ssrd'])
            
            print('Imputation of missing data for Rg from ERA5-Land was done!')

    def missing_data_checking(self, column=None):
        if column is not None:
            if any(pd.isna(self.data.get_dataframe()[column])) is True:
                print("Missed data found in column " + column)
            else:
                print("No missed data in column " + column)
        else:
            for c in self.data.get_dataframe().columns:
                miss = self.data.get_dataframe()[c].isnull().sum()
                if miss>0:
                    missing_data_percent = round((miss/self.data.get_shape()[0])*100, 2)
                    print("{} has {} missing value(s) which represents {}% of dataset size".format(c,miss, missing_data_percent))
                else:
                    print("{} has NO missing value!".format(c))
    
    def anomaly_detection(self, climate_varibale_column='Air temperature', method='knn', ):
        pass
    
    def eliminate_outliers(self, climate_varibale_column='ta',method='lof', n_neighbors=20, contamination=.05):
        if method == 'lof':
            outliers = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            self.data.get_dataframe()['inlier'] = outliers.fit_predict(self.data.get_columns([climate_varibale_column]))
            self.data.set_dataframe(self.get_dataframe().loc[self.get_dataframe().inlier == 1,
                                                        self.get_dataframe().columns.tolist()]) 
    
    def evaluate_products(self):
        pass
    
    def plot_column(self, column):
        self.data.get_column(column).plot()
        plt.show()
    
    def apply_quality_control_criteria(self, variable_column_name, decision_func=lambda x:x>0):
        self.data.add_column(self.data.get_column(variable_column_name).apply(decision_func), 'decision')
        self.data.get_dataframe().loc[ self.data.get_dataframe()['decision'] == False, variable_column_name] = None
        self.data.drop_column('decision')