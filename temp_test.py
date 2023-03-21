import time
from data_science_toolkit.dataframe import DataFrame
from climatefiller import ClimateFiller


def main():
    ti = time.time()
    

    # Initilize the ClimateFiller object
    climate_filler = ClimateFiller()

    climate_filler.download('ws', '2020-01-01', '2020-02-01')
    climate_filler.plot_column('ws')

   
    print(time.time() - ti)


if __name__ == '__main__':
    main()


