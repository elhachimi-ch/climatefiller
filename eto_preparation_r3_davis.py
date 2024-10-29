import time
from data_science_toolkit.dataframe import DataFrame
from climatefiller import ClimateFiller


def main():
    ti = time.time()
    
    # Initilize the ClimateFiller object
    climate_filler = ClimateFiller(
        r"C:\Users\elhac\OneDrive\Desktop\kitchen\projects\pythonsnippets\data\california\cimis_data_hourly.csv",
        lon=-121.776360,
        lat=38.535694,
        )
    # for Davis
    climate_filler.data.drop_column('et0_pm')
    climate_filler.et0_estimation()
    data_eto_pm = DataFrame(climate_filler.et0_output_data.dataframe, 'df')
    climate_filler.et0_estimation(method='hs',)
    eto_hs = climate_filler.et0_output_data.dataframe['et0_hs']
    
    climate_filler.extraterrestrial_radiation_daily()
    ra = climate_filler.data.resample_timeseries()['ra']
    
    data_eto_pm.add_column('eto_hs', eto_hs)
    print(ra)
    data_eto_pm.show()
    data_eto_pm.add_column('ra', ra)
    data_eto_pm.export('data/medgu/davis_eto_pm_hs_ra.csv')
    
    
    
    # HS calibration for R3: a=11.3, b=0.39, C=0.0035
    # R3 "C:\Users\elhac\OneDrive - Université Mohammed VI Polytechnique\crsa\data\aws_al_haouz\r3_full.csv"
    # Davis "C:\Users\elhac\OneDrive\Desktop\kitchen\projects\pythonsnippets\data\california\cimis_data_hourly.csv"
    # Davis lat 38.535694 lon -121.776360 
    # R3  lat 31.65936 lon -7.60655

  
   
    print(time.time() - ti)


if __name__ == '__main__':
    main()


