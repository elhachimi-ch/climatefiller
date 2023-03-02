import time
from data_science_toolkit.dataframe import DataFrame
from climatefiller import ClimateFiller


def main():
    ti = time.time()
    
    # Read the time series 
    data = DataFrame("E:\projects\data\r3\stations_from_youness.xlsx", data_type='xls', sheet_name='oukaimeden')

    # Rename target colmn 
    data.rename_columns({'Ouka_1_Tair_(°C)':'Ta'})
    
    

    # Initilize the ClimateFiller object
    climate_filler = ClimateFiller(data.get_dataframe(), data_type='df', datetime_column_name='datetime')

    # Replace missing values with 0
    climate_filler.missing_data(filling_dict_colmn_val={'Ta': 0})

    # Detect and eliminate outliers
    climate_filler.eliminate_outliers('Ta', )

   
    print(time.time() - ti)


if __name__ == '__main__':
    main()


