from climatefiller import ClimateFiller
from data_science_toolkit.dataframe import DataFrame

# Read the time series 
data = DataFrame(r"data/sample_time_series.csv")

# Rename target colmn 
data.rename_columns({'air_temperature':'Ta'})

# Initilize the ClimateFiller object
climate_filler = ClimateFiller(data.get_dataframe(), data_type='df')

# Detect and eliminate outliers
climate_filler.eliminate_outliers('Ta')