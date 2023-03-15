import time
from data_science_toolkit.dataframe import DataFrame
from climatefiller import ClimateFiller


def main():
    ti = time.time()
    
    # Read the time series 
    data = DataFrame(r"E:\projects\data\r3_weather_data\r3_weather_data.xlsx", data_type='xls', sheet_name='2013_2020')

    # Initilize the ClimateFiller object
    climate_filler = ClimateFiller(data.get_dataframe(), data_type='df', datetime_column_name='date_time')

    # Replace missing values with 0
    climate_filler.et0_estimation('R3_Tair', 'R3_Rg', 'R3_Hr', 'R3_Vv')

    # Detect and eliminate outliers
    climate_filler.plot_column('et0_pm')

   
    print(time.time() - ti)


if __name__ == '__main__':
    main()


