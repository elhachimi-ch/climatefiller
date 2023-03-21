import time
from data_science_toolkit.dataframe import DataFrame
from climatefiller import ClimateFiller


def main():
    ti = time.time()
    

    # Initilize the ClimateFiller object
    climate_filler = ClimateFiller()

    climate_filler.download('rs', '1951-01-01', '1999-12-31')
    climate_filler.plot_column('rs')

   
    print(time.time() - ti)


if __name__ == '__main__':
    main()


