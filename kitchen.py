import time
from data_science_toolkit.dataframe import DataFrame
from climatefiller import ClimateFiller
from lib import Lib


def main():
    ti = time.time()
    
    # Read the time series 
    #data.reindex_dataframe('datetime')
    cf = ClimateFiller()
    cf.download_era5_land_data_by_months('ta', -7.593311291, 31.66749781, '2020-01-01', '2021-12-31')
    
    

   
    print(time.time() - ti)


if __name__ == '__main__':
    main()


