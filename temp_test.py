import time
from data_science_toolkit.dataframe import DataFrame
from climatefiller import ClimateFiller


def main():
    ti = time.time()
    
    # Read the time series 
    data = DataFrame(r"E:\projects\data\r3_weather_data\R3_meteo_N0_Unification.xlsx", data_type='xls', sheet_name='R3_2013-2020')
    
    data.keep_columns(['R3_Tair', 'DateBis'])

    # Rename target colmn 
    data.rename_columns({'R3_Tair':'ta'})

    # Initilize the ClimateFiller object
    climate_filler = ClimateFiller(data.get_dataframe(), data_type='df', datetime_column_name='DateBis')

    climate_filler.plot_column('ta')

   
    print(time.time() - ti)


if __name__ == '__main__':
    main()


