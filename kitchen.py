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


