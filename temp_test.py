import time
from data_science_toolkit.dataframe import DataFrame
from climatefiller import ClimateFiller


def main():
    ti = time.time()
    

    # Initilize the ClimateFiller object
    climate_filler = ClimateFiller()

    climate_filler.download('rs', '2020-01-01', '2020-02-01')

   
    print(time.time() - ti)


if __name__ == '__main__':
    main()


