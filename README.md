# Data Science Toolkit

[![readthedocs](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://data-science-toolkit.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Reproducible Capsule](https://img.shields.io/static/v1?label=&message=code+ocean&color=blue)](https://codeocean.com/capsule/1309232/tree)

Data Science Toolkit (DST) is a Python library that helps implement data science related project with ease.


## Simple Demo

```python
from data_science_toolkit.dataframe import DataFrame
from data_science_toolkit.model import Model

data = DataFrame()
data.load_dataset('iris')
y = data.get_column('target')
data.drop_column('target')

# decision tree model
model = Model(data_x=data.get_dataframe(), data_y=y, model_type='dt', training_percent=0.8)

# train the model
model.train()

# get all classification evaluation metrics
model.report()

#get the cross validation
model.cross_validation(5)
```


## Documentation

More information can be found on the [DST documentation site.](https://data-science-toolkit.readthedocs.io)

### Contributing

Contrubution and suggestions are welcome via GitHub Pull Requests.

### Maintainership

We're actively enhacing the repo with new algorithms.

### How to cite

