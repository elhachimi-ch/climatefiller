import time
from data_science_toolkit.dataframe import DataFrame
from climatefiller import ClimateFiller
from lib import Lib


def main():
    ti = time.time()
    
    # Read the time series 
    #data.reindex_dataframe('datetime')
    
    climate_filler = ClimateFiller(r"C:\Users\elhac\OneDrive\Desktop\kitchen\data\stations_from_youness.xlsx",
                                   'xls',
                                   sheet_name='R3',
                                   datetime_column_name='DateBis',
                                   usecols=[0, 1, 2, 3, 4],
                                   backend='gee')
    			
    climate_filler.data.rename_columns({'DateBis':'datetime', 'R3_Tair_(°C)':'ta', 'R3_hr_(%)':'rh', 'WS':'ws', 'R3_Rg_(W/m2)':'rs', 'R3_vv_(m/s)':'u'})
    climate_filler.data.resample_timeseries(frequency='H')
    climate_filler.fill('ta', product='era5_land', )
    climate_filler.fill('rs', product='era5_land',)
    climate_filler.fill('rh', product='era5_land',)
    climate_filler.fill('u', product='era5_land',)
    print(climate_filler.data.show())
    print(climate_filler.data.missing_data_statistics())
    
    
    
    #climate_filler.et0_estimation('R3_Tair', 'R3_Rg', 'R3_Hr', 'R3_Vv')
    et0_pm_daily_california_data = DataFrame('E:\projects\pythonsnippets\data\california\cimis_data.csv')
    et0_pm_daily_california_data.column_to_date('datetime')
    et0_pm_daily_california_data.reindex_dataframe('datetime')
    et0_pm_daily_california_data.resample_timeseries(agg='sum')
    et0_pm_daily_california_data.keep_columns('PM ETo (mm)')
    et0_pm_daily_california_data.show()
    
    
    climate_filler = ClimateFiller(r"E:\projects\pythonsnippets\data\california\cimis_data.csv", 
                                   backend='gee',
                                   datetime_column_name='datetime',
                                   lat=38.535694, lon=-121.776360, standard_meridian=-120, elevation=18.288
                                   )
    climate_filler.et0_estimation(
        'Air Temp (C)',
        'Sol Rad (W/sq.m)',
        'Rel Hum (%)',
        'Wind Speed (m/s)',
        method='pm', 
        freq='h',)
    
    climate_filler.data.reindex_dataframe('datetime')
    climate_filler.data.resample_timeseries(agg='sum', between_time_tuple=('09:00:00', '18:00:00'))
    
    cimis_dataset = DataFrame('E:\projects\pythonsnippets\data\california\cimis_data.csv')
    cimis_dataset.column_to_date('datetime')
    cimis_dataset.reindex_dataframe('datetime')
    cimis_dataset.drop_column('ETo (mm)')
    cimis_dataset.drop_column('PM ETo (mm)')
    cimis_dataset.rename_columns({'Sol Rad (W/sq.m)': 'rs', 'Rel Hum (%)': 'rh', 'Air Temp (C)': 'ta', 'Wind Speed (m/s)': 'u'})
    cimis_dataset.resample_timeseries()
    
    climate_filler.data.keep_columns(['et0_pm'])
    cimis_dataset.join(climate_filler.data.dataframe)
    cimis_dataset.join(et0_pm_daily_california_data.get_dataframe())
    
    climate_filler = ClimateFiller(r"E:\projects\pythonsnippets\data\california\cimis_data.csv", 
                                   backend='gee',
                                   datetime_column_name='datetime',
                                   lat=38.535694, lon=-121.776360, standard_meridian=-120, elevation=18.288
                                   )
    climate_filler.et0_estimation(
        'Air Temp (C)',
        'Sol Rad (W/sq.m)',
        'Rel Hum (%)',
        'Wind Speed (m/s)',
        method='pt', 
        freq='d',) 
    
    climate_filler.data.reindex_dataframe('datetime')
    cimis_dataset.join(climate_filler.data.dataframe)
    
    print(f"PM: {cimis_dataset.similarity_measure('PM ETo (mm)', 'et0_pm', 'ts')}")
    print(f"PT: {cimis_dataset.similarity_measure('PM ETo (mm)', 'et0_pt', 'ts')}")
    print(f"PT vs PM: {cimis_dataset.similarity_measure('et0_pm', 'et0_pt', 'ts')}")
    
    
    cimis_dataset.export('data/cimis_et0_pm_pt_daily.csv', index=True)
    
    """climate_filler.et0_estimation(freq='h', method='pm')
    climate_filler.data.reindex_dataframe('datetime')
    climate_filler.data.resample_timeseries(agg='sum')
    climate_filler.data.export('data/et0.csv', index=True)"""
    
    """climate_filler.data.resample_timeseries(skip_rows=2)
    climate_filler.data.drop_columns(['wd'])
    climate_filler.fill('ta')
    climate_filler.fill('rs')
    climate_filler.fill('rh')
    climate_filler.fill('ws')
    climate_filler.fill('p')
    climate_filler.export('data/r3_era5land_imputed.csv', index=True)"""
    
    """climate_filler.et0_estimation(method='pm', freq='h')
    hourly_et0 = DataFrame(climate_filler.data.dataframe, data_type='df')
    hourly_et0.export('data/et0_hourly.csv', index=True)
    hourly_et0.column_to_date('datetime')
    hourly_et0.reindex_dataframe('datetime')
    hourly_et0.resample_timeseries(agg='sum')

    climate_filler.data.reindex_dataframe('datetime')
    climate_filler.et0_estimation(method='pm')    
    daily_et0 = DataFrame(climate_filler.data.dataframe, data_type='df')
    daily_et0.reindex_dataframe('datetime')
    print(hourly_et0.show())
    print(daily_et0.show())
    daily_et0.add_column('et0_from_hourly', hourly_et0.get_column('et0_pm'))
    
    daily_et0.export('data/et0pm.csv', index=True)"""

    #climate_filler.data.resample_timeseries(frequency='H')
    #climate_filler.data.rename_columns({'R3_Tair':'ta'})
    #climate_filler.fill('ta', product='era5_land', machine_learning_enabled=True)
    
    
    # -8.87,32.56,
    #climate_filler.download('ta', longitude=-8.87, latitude=32.56, start_date='2020-01-01', end_date='2022-07-01', backend='gee')
    #climate_filler.download('ta', backend='gee', start_date='2000-05-01', end_date='2002-05-01') 
    #climate_filler.download('rs', backend='gee', start_date='2001-01-01', end_date='2001-01-05') 
    #climate_filler.download('ws', backend='gee', start_date='2000-05-01', end_date='2001-07-01') 
    #climate_filler.download('rh', backend='gee', start_date='2000-05-01', end_date='2001-07-01') 

    # Rename target colmn 
    #data.rename_columns({'air_temperature':'Ta'})

    # Initilize the ClimateFiller object
    #climate_filler = ClimateFiller(data.get_dataframe(), data_type='df', datetime_column_name='datetime')

    # Replace missing values with 0
    #climate_filler.missing_data(filling_dict_colmn_val={'Ta': 0})

    # Detect and eliminate outliers
    #climate_filler.eliminate_outliers('Ta', )

   
    print(time.time() - ti)


if __name__ == '__main__':
    main()


