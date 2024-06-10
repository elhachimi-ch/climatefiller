import time
from data_science_toolkit.dataframe import DataFrame
from climatefiller import ClimateFiller
from lib import Lib


def main():
    ti = time.time()
    
    # Read the time series 
    #data.reindex_dataframe('datetime')
    cf = ClimateFiller(r"C:\Users\elhac\OneDrive\Desktop\kitchen\data\stations_from_youness.xlsx", 'xls', 
                       sheet_name='oukaimeden', 
                       usecols=[0, 1, 3, 2, 4],
                       datetime_column_name='Date')
    
    # columns name in the excel file
    # DateBis	R3_hr_(%)	R3_Rg_(W/m2)	R3_Tair_(°C)	R3_vv_(m/s)	R3_P30(mm)
    # Date	Ouka_1_Hr_(%)	Ouka_1_Rg_(W/m2)	Ouka_1_Tair_(°C)	Ouka_1_Vv_(m/s)

    cf.data.resample_timeseries(skip_rows=2)
    #cf.data.rename_columns({'R3_hr_(%)': 'rh', 'R3_Rg_(W/m2)': 'rs', 'R3_Tair_(°C)': 'ta', 'R3_vv_(m/s)': 'ws', 'R3_P30(mm)': 'p'})
    cf.data.rename_columns({'Ouka_1_Hr_(%)': 'rh', 'Ouka_1_Rg_(W/m2)': 'rs', 'Ouka_1_Tair_(°C)': 'ta', 'Ouka_1_Vv_(m/s)': 'ws'})
    
    print(cf.data.show())
    
    # Chichaoua	31.42767	-8.65185
    # Sidi Rahal	31.65936	-7.60655
    # Armed	31.13003	-7.91898
    # Oukaimden	31.18205	-7.86553

    
    cf.fill('ta', -7.86553, 31.18205)
    cf.fill('rs', -7.86553, 31.18205)
    cf.fill('rh', -7.86553, 31.18205)
    cf.fill('ws', -7.86553, 31.18205)
    print(cf.data.missing_data_statistics())
    cf.data.export("data/oukaimden_full.csv", index=True)

   
    print(time.time() - ti)


if __name__ == '__main__':
    main()


