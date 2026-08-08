import os
import sys
import types

import numpy as np
import pandas as pd


class _StubDataFrame:
    def __init__(self, data_path=None, data_type=None, **kwargs):
        if isinstance(data_path, pd.DataFrame):
            self.dataframe = data_path.copy()
        elif isinstance(data_path, (str, os.PathLike)):
            path_str = str(data_path)
            lower_path = path_str.lower()
            if lower_path.endswith('.parquet'):
                self.dataframe = pd.read_parquet(path_str)
            elif lower_path.endswith('.csv'):
                self.dataframe = pd.read_csv(path_str)
            else:
                self.dataframe = pd.DataFrame()
        else:
            self.dataframe = pd.DataFrame()
        self.data_type = data_type or 'df'

    def set_row(self, column_name, row_index, value):
        return None

    def get_missing_data_indexes_in_column(self, column_name):
        return self.dataframe.index[self.dataframe[column_name].isna()].tolist()

    def rename_columns(self, mapping):
        self.dataframe.rename(columns=mapping, inplace=True)

    def column_to_date(self, column_name, datetime_format='%Y-%m-%d %H:%M:%S'):
        self.dataframe[column_name] = pd.to_datetime(self.dataframe[column_name], format=datetime_format)
        self.dataframe.set_index(column_name, inplace=True)

    def reindex_dataframe(self, column_name):
        self.dataframe = self.dataframe.sort_index().reindex(pd.DatetimeIndex(self.dataframe.index))

    def get_columns_names(self):
        return list(self.dataframe.columns)

    def get_dataframe(self):
        return self.dataframe

    def set_dataframe(self, dataframe, data_type='df'):
        self.dataframe = dataframe
        self.data_type = data_type

    def export(self, path_link, data_type=None, *args, **kwargs):
        self.last_export_path = path_link
        self.last_export_data_type = data_type
        self.last_export_kwargs = kwargs


class _StubModel:
    def __init__(self, *args, **kwargs):
        pass


sys.modules.setdefault('data_science_toolkit', types.ModuleType('data_science_toolkit'))
sys.modules.setdefault('data_science_toolkit.dataframe', types.ModuleType('data_science_toolkit.dataframe'))
sys.modules.setdefault('data_science_toolkit.model', types.ModuleType('data_science_toolkit.model'))
sys.modules['data_science_toolkit.dataframe'].DataFrame = _StubDataFrame
sys.modules['data_science_toolkit.model'].Model = _StubModel

sys.modules.setdefault('ee', types.ModuleType('ee'))
sys.modules['ee'].Initialize = lambda *args, **kwargs: None

sys.modules.setdefault('geemap', types.ModuleType('geemap'))

sys.modules.setdefault('xgboost', types.ModuleType('xgboost'))
sys.modules['xgboost'].XGBRegressor = object

sys.modules.setdefault('catboost', types.ModuleType('catboost'))
sys.modules['catboost'].CatBoostRegressor = object

sys.modules.setdefault('geocoder', types.ModuleType('geocoder'))
try:
    import geopandas as gpd
except ModuleNotFoundError:  # pragma: no cover - test environment fallback
    gpd = None

sys.modules.setdefault('geopandas', types.ModuleType('geopandas'))
if gpd is not None:
    sys.modules['geopandas'].GeoDataFrame = gpd.GeoDataFrame
    sys.modules['geopandas'].points_from_xy = staticmethod(lambda x, y, crs=None: None)
else:
    class _FallbackGeoDataFrame(pd.DataFrame):
        _metadata = ['crs']

        def __init__(self, *args, **kwargs):
            geometry = kwargs.pop('geometry', None)
            crs = kwargs.pop('crs', None)
            super().__init__(*args, **kwargs)
            self.geometry = geometry
            self.crs = crs

        @property
        def _constructor(self):
            return _FallbackGeoDataFrame

    sys.modules['geopandas'].GeoDataFrame = _FallbackGeoDataFrame
    sys.modules['geopandas'].points_from_xy = staticmethod(lambda x, y, crs=None: None)

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


def test_instance_exposes_dataframe_methods_directly():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    df = pd.DataFrame({
        'date': ['2020-01-01 00:00:00', '2020-01-01 01:00:00'],
        'value': [1.0, 2.0],
    })

    cf = ClimateFiller(df, datetime_column_name='date', backend='local')

    head = cf.head(1)
    assert len(head) == 1
    assert cf.shape[0] == 2
    assert cf.columns.tolist() == ['value']


def test_constructor_parses_timezone_suffixed_datetime_strings(tmp_path):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    csv_path = tmp_path / 'sample.csv'
    pd.DataFrame({'date': ['2020-01-01 00:00:00+00:00', '2020-01-01 01:00:00+00:00'], 'value': [1.0, 2.0]}).to_csv(csv_path, index=False)

    cf = ClimateFiller(str(csv_path), datetime_column_name='date', backend='local')

    assert cf.shape[0] == 2
    assert str(cf.index[0]) == '2020-01-01 00:00:00+00:00'


def test_daily_column_names_resolve_to_expected_climate_variable_and_aggregation():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]}), datetime_column_name='date', backend='local')

    temp_spec = cf._resolve_imputation_variable_context('t2m_max')
    assert temp_spec['canonical'] == 'ta'
    assert temp_spec['aggregation'] == 'max'

    humidity_spec = cf._resolve_imputation_variable_context('rh_mean')
    assert humidity_spec['canonical'] == 'rh'
    assert humidity_spec['aggregation'] == 'mean'


def test_align_source_series_to_daily_target_frequency():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    daily_index = pd.date_range('2020-01-01 00:00:00', periods=3, freq='D')
    source_series = pd.Series(
        np.arange(72, dtype=float),
        index=pd.date_range('2020-01-01 00:00:00', periods=72, freq='H'),
    )

    cf = ClimateFiller(pd.DataFrame({'date': daily_index, 'rs': [np.nan, np.nan, np.nan]}), datetime_column_name='date', backend='local')

    aligned = cf._align_source_series_to_target_frequency(source_series, 'rs', daily_index)

    assert aligned.index.equals(daily_index)
    daytime_values = source_series.between_time('09:00', '18:00')
    expected = daytime_values.groupby(daytime_values.index.floor('D')).mean().iloc[0]
    assert np.isclose(aligned.iloc[0], expected)


def test_fill_from_source_series_updates_target_column_without_set_row():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    daily_index = pd.date_range('2020-01-01 00:00:00', periods=2, freq='D')
    df = pd.DataFrame({'date': daily_index, 'rs': [np.nan, np.nan]})
    cf = ClimateFiller(df, datetime_column_name='date', backend='local')

    source_series = pd.Series([10.0, 20.0], index=daily_index)
    cf._fill_from_source_series('rs', source_series, 'era5_land', machine_learning_enabled=False)

    assert cf.data.get_dataframe()['rs'].notna().all()
    assert cf.data.get_dataframe().loc[daily_index[0], 'rs'] == 10.0


def test_build_geodataframe_uses_source_crs_when_available():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    source_df = pd.DataFrame({'lon': [0.0], 'lat': [1.0]})
    source_df.crs = 'EPSG:32631'

    cf = ClimateFiller(pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]}), datetime_column_name='date', backend='local')

    gdf = cf._build_geodataframe_from_dataframe(source_df, crs=None)

    assert gdf.crs == 'EPSG:32631'


def test_export_infers_format_from_path_extension():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]}), datetime_column_name='date', backend='local')

    cf.export('data/output.parquet', index=True)

    assert cf.data.last_export_data_type == 'parquet'
    assert cf.data.last_export_kwargs.get('index') is True


def test_export_uses_source_crs_when_crs_is_none_for_geospatial_output():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    source_df = pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'lon': [0.0], 'lat': [1.0], 'value': [1.0]})
    source_df.crs = 'EPSG:32631'

    cf = ClimateFiller(source_df, datetime_column_name='date', backend='local')

    gdf = cf.export('data/output.parquet', crs=None)

    assert gdf.crs == 'EPSG:32631'


def test_impute_batch_processes_files_and_writes_outputs(tmp_path, monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    output_dir.mkdir()

    source_df = pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]})
    source_df.to_parquet(input_dir / 'sample.parquet')

    def fake_impute(self, column_to_fill_name='ta', product='era5_land', machine_learning_enabled=False, train_ratio=1, model_name='xgboost', export_dataset=False, **kwargs):
        self.data.get_dataframe()['filled'] = 1.0
        return self

    monkeypatch.setattr(ClimateFiller, 'impute', fake_impute)

    cf = ClimateFiller(pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]}), datetime_column_name='date', backend='local')
    output_paths = cf.impute_batch(str(input_dir), str(output_dir), column_to_fill_name='rs', prefix='sample')

    assert len(output_paths) == 1
    written = pd.read_parquet(output_dir / 'sample.parquet')
    assert 'filled' in written.columns
