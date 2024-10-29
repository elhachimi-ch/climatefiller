import time
from data_science_toolkit.dataframe import DataFrame
from climatefiller import ClimateFiller


def main():
    ti = time.time()
    
    # Initilize the ClimateFiller object
    climate_filler = ClimateFiller(
        r"C:\Users\elhac\OneDrive - Université Mohammed VI Polytechnique\crsa\data\aws_al_haouz\r3_full.csv",
        lon=-7.60655,
        lat=31.65936,
        )
    
    climate_filler.et0_estimation()
    climate_filler.et0_output_data.show()
    
    # HS calibration for R3: a=11.3, b=0.39, C=0.0035
    # R3 "C:\Users\elhac\OneDrive - Université Mohammed VI Polytechnique\crsa\data\aws_al_haouz\r3_full.csv"
    # Davis "C:\Users\elhac\OneDrive\Desktop\kitchen\projects\pythonsnippets\data\california\cimis_data_hourly.csv"
    # Davis lat 38.535694 lon -121.776360 
    # R3  lat 31.65936 lon -7.60655

  
   
    print(time.time() - ti)


if __name__ == '__main__':
    main()


