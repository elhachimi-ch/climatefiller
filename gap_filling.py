import time
from data_science_toolkit.dataframe import DataFrame
from climatefiller import ClimateFiller


def main():
    ti = time.time()
    
    # Initilize the ClimateFiller object
    climate_filler = ClimateFiller(
        r"C:\Users\elhac\OneDrive\Desktop\kitchen\projects\pythonsnippets\data\california\cimis_data_hourly.csv",
        lon=-7.60655,
        lat=31.65936,
        )
    
    # Davis lat 38.535694 lon -121.776360 
    # R3  lat 31.65936 lon -7.60655

    climate_filler.download(
        ['temperature_2m', 'dewpoint_temperature_2m'],
        start_date='2012-01-01', 
        end_date='2020-12-31'
        )

   
    print(time.time() - ti)


if __name__ == '__main__':
    main()


