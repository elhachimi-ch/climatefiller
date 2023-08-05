# ClimateFiller framework

[![readthedocs](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://climatefiller.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Reproducible Capsule](https://img.shields.io/static/v1?label=&message=code+ocean&color=blue)](https://codeocean.com/capsule/1309232/tree)
 
ClimateFiller is a python framework that implements various data-driven methods to make manipulating in-situ climate time series easier. It offers various services such as (1) automating gap-filling (2) using machine learning and ERA5-Land for bias correction (3) using Isolation Forest, Local Outlier Factor, and quantiles to detect and eliminate outliers. It was tested on several Automatic Weather Stations (AWS) installed in Morocco.

DISCLAIMER:
Please note that this project is currently in the BETA stage and will remain experimental for the foreseeable future. As a result, there is a high probability of Classes, methods names, and other functionalities undergoing modifications.


# How to Install

## 1- Climate Data Store API Setup

<ol>

<li> 
Obtain a Climate Data Store (CDS) API key: <br> To access the Climate Data Store API, you'll need to register and obtain an API key from their website: https://cds.climate.copernicus.eu/user/register
<br> Go to your profile and copy your key from the <b>API key</b> section

</li>

<li>
Configure the API key: <br> Once you have the API key, create a file named <b>.cdsapirc</b> in the project's root directory and add the API key to it:
</li>


```javascript
{
    "url": "https://cds.climate.copernicus.eu/api/v2",
    "key": "YOUR_CDS_API_KEY"
}
```

<li>
Replace YOUR_CDS_API_KEY with your actual CDS API key.
</li>

</ol>


## 2- Google Earth Engine Setup

<ol>

<li>

Sign up for Google Earth Engine: <br> To use Google Earth Engine, sign up for an Earth Engine account at https://earthengine.google.com/signup/
</li>

<li>
Download and install Google Cloud SDK at: <br> <a href='https://cloud.google.com/sdk/docs/install'> https://cloud.google.com/sdk/docs/install </a>
</li>

<li>
Ceate a new project: <br>

<a href='https://earthengine.google.com/signup'> https://earthengine.google.com/signup </a>

</li>


<li>

Authenticate the Earth Engine API by running in terminal:
</li>

```bash
earthengine authenticate
```
Follow the instructions to authorize the Earth Engine API with your Google account. This method, enables you access the API without having to request authorization every time.






</ol>



## 3- Project Installation

<ol>

<li>
Clone this GitHub repository to your local machine using the following command:
</li>

```bash
git clone https://github.com/elhachimi-ch/climatefiller.git
```

<li>
Install project dependencies using conda (Preferred way):
</li>

```bash
conda env create -f environment.yml
```

```bash
conda activate climate_filler
```

```bash
python kitchen.py
```



Or you can install project dependencies using pip (not recommended):


```bash
pip install -r requirements.txt
```

</ol>



## 4- Simple Demo

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

# Documentation

More information can be found on the [ClimateFiller framework documentation site.](https://climatefiller.readthedocs.io/)
# Contributing

Contrubution and suggestions are welcome via GitHub Pull Requests.

# Maintainership

We're actively enhacing the repo with new algorithms.

# How to cite

The research paper of the project is under review