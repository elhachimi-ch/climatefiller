# ClimateFiller framework

[![readthedocs](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://data-science-toolkit.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Reproducible Capsule](https://img.shields.io/static/v1?label=&message=code+ocean&color=blue)](https://codeocean.com/capsule/1309232/tree)

ClimateFiller is a python framework that implements various data-driven methods to make manipulating in-situ climate time series easier. It offers various services such as (1) automating gap-filling (2) using machine learning and ERA5-Land for bias correction (3) using Isolation Forest, Local Outlier Factor, and quantiles to detect and eliminate outliers. It was tested on several Automatic Weather Stations (AWS) installed in Morocco.

DISCLAIMER:
Please note that this project is currently in the BETA stage and will remain experimental for the foreseeable future. As a result, there is a high probability of Classes, methods names, and other functionalities undergoing modifications.

## Simple Demo

```python
import time
from data_science_toolkit.dataframe import DataFrame
from climatefiller import ClimateFiller
    
# Read the time series 
data = DataFrame("data/los_angeles_sair_temperature.csv")

# Rename target colmn 
data.rename_columns({'air_temperature':'Ta'})

# Initilize the ClimateFiller object
climate_filler = ClimateFiller(data.get_dataframe(), data_type='df', datetime_column_name='datetime')

# Replace missing values with 0
climate_filler.missing_data(filling_dict_colmn_val={'Ta': 0})

# Detect and eliminate outliers
climate_filler.eliminate_outliers('Ta')
```


## Documentation

More information can be found on the [ClimateFiller framework documentation site.](https://data-science-toolkit.readthedocs.io)

### Contributing

Contrubution and suggestions are welcome via GitHub Pull Requests.

### Maintainership

We're actively enhacing the repo with new algorithms.

### How to cite

