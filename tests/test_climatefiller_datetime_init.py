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


def test_explicit_frequency_is_used_for_frequency_inference():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]}),
        datetime_column_name='date',
        backend='local',
        frequency='d',
    )

    inferred = cf._infer_frequency_label_from_index(pd.DatetimeIndex([pd.Timestamp('2020-01-01 00:00:00')]))

    assert inferred == 'daily'


def test_fill_from_source_series_matches_timezone_normalized_indexes():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    df = pd.DataFrame({
        'date': ['2020-01-01 00:00:00', '2020-01-01 01:00:00'],
        'rh_max': [np.nan, np.nan],
    })
    cf = ClimateFiller(df, datetime_column_name='date', backend='local')

    source_series = pd.Series(
        [10.0, 20.0],
        index=pd.DatetimeIndex([
            pd.Timestamp('2020-01-01 00:00:00', tz='UTC'),
            pd.Timestamp('2020-01-01 01:00:00', tz='UTC'),
        ]),
    )

    cf._fill_from_source_series('rh_max', source_series, 'era5_land', machine_learning_enabled=False)

    assert cf.data.get_dataframe()['rh_max'].notna().all()


def test_align_source_series_to_target_frequency_aggregates_to_configured_resolution():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00', '2020-01-01 01:00:00'], 'rh_max': [np.nan, np.nan]}),
        datetime_column_name='date',
        backend='local',
        frequency='h',
    )

    source_series = pd.Series(
        [10.0, 30.0],
        index=pd.date_range('2020-01-01 00:00:00', periods=2, freq='H'),
    )

    aligned = cf._align_source_series_to_target_frequency(source_series, 'rh_max', target_index=source_series.index)

    assert aligned.shape[0] == 2
    assert aligned.iloc[0] == 10.0
    assert aligned.iloc[1] == 30.0


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


def test_prepare_datetime_column_infers_non_literal_source_column_names():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]}), datetime_column_name='date', backend='local')
    source_frame = pd.DataFrame({'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'], 'value': [1.0, 2.0]})

    prepared = cf._prepare_datetime_column(source_frame)

    assert prepared.index.name == 'datetime'
    assert prepared.shape[0] == 2


def test_constructor_resolves_lon_and_lat_from_dataframe_columns_once():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    df = pd.DataFrame({
        'date': ['2020-01-01 00:00:00'],
        'lon': [12.34],
        'lat': [56.78],
        'value': [1.0],
    })

    cf = ClimateFiller(df, datetime_column_name='date', backend='local', lon='lon', lat='lat')

    assert cf.lon == 12.34
    assert cf.lat == 56.78


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


def test_normalize_datetime_index_handles_mixed_timezone_and_naive_values():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    values = [
        pd.Timestamp('2020-01-01 00:00:00+00:00'),
        '2020-01-01 01:00:00',
    ]

    normalized = ClimateFiller._normalize_datetime_index(values, preserve_timezone=False)

    assert len(normalized) == 2
    assert normalized[0] == pd.Timestamp('2020-01-01 00:00:00')
    assert normalized[1] == pd.Timestamp('2020-01-01 01:00:00')


def test_fill_from_source_series_deduplicates_datetime_index_before_assignment():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    duplicate_index = pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:00:00', '2020-01-02 00:00:00'])
    df = pd.DataFrame({'date': duplicate_index, 'ta': [np.nan, np.nan, np.nan]})
    cf = ClimateFiller(df, datetime_column_name='date', backend='local')

    source_series = pd.Series([10.0, 20.0], index=pd.DatetimeIndex(['2020-01-01 00:00:00', '2020-01-02 00:00:00']))
    cf._fill_from_source_series('ta', source_series, 'era5_land', machine_learning_enabled=False)

    filled_df = cf.data.get_dataframe()
    assert filled_df.index.is_unique
    assert filled_df.shape[0] == 2
    assert filled_df.loc[pd.Timestamp('2020-01-01 00:00:00'), 'ta'] == 10.0


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


def test_export_includes_index_by_default_for_table_outputs():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    class _RecordingDataFrame:
        def __init__(self):
            self.export_calls = []
            self.last_export_path = None
            self.last_export_data_type = None
            self.last_export_kwargs = None

        def get_dataframe(self):
            return pd.DataFrame({'value': [1.0]})

        def export(self, path_link, data_type=None, **kwargs):
            self.export_calls.append({'path': path_link, 'data_type': data_type, 'kwargs': kwargs})
            return self

    cf = ClimateFiller(pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]}), datetime_column_name='date', backend='local')
    cf.data = _RecordingDataFrame()

    cf.export('data/output.csv')

    assert cf.data.export_calls[0]['kwargs']['index'] is True


def test_impute_single_column_normalizes_mixed_timezone_indexes(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({
            'date': ['2020-01-01 00:00:00+00:00', '2020-01-01 01:00:00+00:00'],
            'rh_max': [np.nan, np.nan],
        }),
        datetime_column_name='date',
        backend='gee',
    )

    monkeypatch.setattr(cf.data, 'get_missing_data_indexes_in_column', lambda column: [
        pd.Timestamp('2020-01-01 00:00:00+00:00'),
        pd.Timestamp('2020-01-01 01:00:00+00:00'),
    ])
    monkeypatch.setattr(cf, '_build_source_cache_path', lambda *args, **kwargs: 'dummy.csv')
    monkeypatch.setattr('climatefiller.os.path.exists', lambda path: True)

    captured = {}

    def fake_load_source_series_cache(path):
        return pd.Series(
            [10.0, 20.0],
            index=pd.DatetimeIndex([
                pd.Timestamp('2020-01-01 00:00:00'),
                pd.Timestamp('2020-01-01 01:00:00'),
            ]),
        )

    def fake_fill_from_source_series(column, source_series, product, machine_learning_enabled=False):
        captured['column'] = column

    monkeypatch.setattr(cf, '_load_source_series_cache', fake_load_source_series_cache)
    monkeypatch.setattr(cf, '_fill_from_source_series', fake_fill_from_source_series)

    cf._impute_single_column('rh_max', product='era5_land')

    assert captured['column'] == 'rh_max'


def test_impute_accepts_multiple_columns_and_processes_each(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'ta': [np.nan], 'rs': [np.nan]}), datetime_column_name='date', backend='local')
    seen = []

    def fake_impute_single(self, column_to_fill_name, **kwargs):
        seen.append(column_to_fill_name)
        self.data.get_dataframe()[column_to_fill_name] = 1.0
        return self

    monkeypatch.setattr(ClimateFiller, '_impute_single_column', fake_impute_single)

    result = cf.impute(['ta', 'rs'])

    assert seen == ['ta', 'rs']
    assert result is cf
    assert cf.data.get_dataframe().loc[pd.Timestamp('2020-01-01 00:00:00'), 'ta'] == 1.0
    assert cf.data.get_dataframe().loc[pd.Timestamp('2020-01-01 00:00:00'), 'rs'] == 1.0


def test_missing_data_checking_accepts_multiple_columns_and_returns_counts():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({
            'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'],
            'ta': [np.nan, 1.0],
            'rs': [2.0, np.nan],
        }),
        datetime_column_name='date',
        backend='local',
    )

    result = cf.missing_data_checking(['ta', 'rs'], verbose=False)

    assert result == {'ta': 1, 'rs': 1}


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
    output_paths = cf.impute_batch(str(input_dir), str(output_dir), column_to_fill_list='rs', prefix='sample')

    assert len(output_paths) == 1
    written = pd.read_parquet(output_dir / 'sample.parquet')
    assert 'filled' in written.columns
