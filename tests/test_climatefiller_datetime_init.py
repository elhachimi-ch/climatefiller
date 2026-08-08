import os

import pandas as pd

from climatefiller import ClimateFiller


def test_init_with_datetime_column_for_dataframe_and_parquet(tmp_path):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    df = pd.DataFrame({
        'date': ['2020-01-01 00:00:00', '2020-01-01 01:00:00'],
        'value': [1.0, 2.0],
    })

    cf = ClimateFiller(df, datetime_column_name='date', backend='local')
    assert cf.datetime_column_name == 'datetime'
    assert cf.data.get_dataframe().index.name == 'datetime'

    parquet_path = tmp_path / 'sample.parquet'
    df.to_parquet(parquet_path)

    cf_parquet = ClimateFiller(str(parquet_path), datetime_column_name='date', backend='local')
    assert cf_parquet.datetime_column_name == 'datetime'
    assert cf_parquet.data.get_dataframe().index.name == 'datetime'
