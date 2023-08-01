# ClimateFiller framework

[![readthedocs](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://climatefiller.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Reproducible Capsule](https://img.shields.io/static/v1?label=&message=code+ocean&color=blue)](https://codeocean.com/capsule/1309232/tree)
 
ClimateFiller is a python framework that implements various data-driven methods to make manipulating in-situ climate time series easier. It offers various services such as (1) automating gap-filling (2) using machine learning and ERA5-Land for bias correction (3) using Isolation Forest, Local Outlier Factor, and quantiles to detect and eliminate outliers. It was tested on several Automatic Weather Stations (AWS) installed in Morocco.

DISCLAIMER:
Please note that this project is currently in the BETA stage and will remain experimental for the foreseeable future. As a result, there is a high probability of Classes, methods names, and other functionalities undergoing modifications.


## How to Install

# Climate Data Store API Setup

1- Obtain a Climate Data Store (CDS) API key: To access the Climate Data Store API, you'll need to register and obtain an API key from their website: https://cds.climate.copernicus.eu/user/register

2- Configure the API key: Once you have the API key, create a file named .cdsapirc in the project's root directory and add the API key to it:

```javascript
{
  "url": "https://cds.climate.copernicus.eu/api/v2",
  "key": "YOUR_CDS_API_KEY"
}
```

Replace YOUR_CDS_API_KEY with your actual CDS API key.


# Google Earth Engine Setup


1- Sign up for Google Earth Engine: To use Google Earth Engine, sign up for an Earth Engine account at https://earthengine.google.com/signup/. Note that Google Earth Engine access is currently limited and may require approval.

2- Install the Earth Engine Python API: Install the Earth Engine Python API using pip:

```bash
pip install earthengine-api
```

3- Authenticate the Earth Engine API: After installing the Earth Engine Python API, authenticate it by running:

```bash
earthengine authenticate
```

4- Follow the instructions to authorize the Earth Engine API with your Google account.


## Project Installation

Clone the repository: Clone this GitHub repository to your local machine using the following command:

```bash
git clone https://github.com/your-username/your-repo.git
```

Replace your-username and your-repo with your GitHub username and repository name.

Install project dependencies: Change to the project directory and install the required Python packages using pip:

```bash
cd your-repo
pip install -r requirements.txt
```

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

More information can be found on the [ClimateFiller framework documentation site.](https://climatefiller.readthedocs.io/)

### Contributing

Contrubution and suggestions are welcome via GitHub Pull Requests.

### Maintainership

We're actively enhacing the repo with new algorithms.

### How to cite
