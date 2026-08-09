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
        if column_name in self.dataframe.columns:
            self.dataframe[column_name] = pd.to_datetime(self.dataframe[column_name])
            self.dataframe = self.dataframe.set_index(column_name).sort_index()
        else:
            self.dataframe = self.dataframe.sort_index().reindex(pd.DatetimeIndex(self.dataframe.index))

    def index_to_column(self, column_name='datetime'):
        self.dataframe = self.dataframe.copy()
        self.dataframe[column_name] = self.dataframe.index
        return self

    def add_doy_column(self, datetime_column_name='datetime'):
        if datetime_column_name in self.dataframe.columns:
            dt = pd.to_datetime(self.dataframe[datetime_column_name])
        else:
            dt = pd.to_datetime(self.dataframe.index)
        self.dataframe['doy'] = dt.dt.dayofyear
        return self

    def add_one_value_column(self, column_name, value):
        self.dataframe[column_name] = value
        return self

    def add_column(self, column_name, values):
        self.dataframe[column_name] = values
        return self

    def add_column_based_on_function(self, column_name, func):
        self.dataframe[column_name] = self.dataframe.apply(func, axis=1)
        return self

    def transform_column(self, column_name, func):
        self.dataframe[column_name] = self.dataframe[column_name].apply(func)
        return self

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
    assert 'date' in written.columns


def test_impute_batch_preserves_original_geoparquet_crs(tmp_path, monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')
    if gpd is None:
        return

    from shapely.geometry import Point

    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    output_dir.mkdir()

    source_gdf = gpd.GeoDataFrame(
        {
            'date': ['2020-01-01 00:00:00'],
            'lon': [500000.0],
            'lat': [3500000.0],
            'rs': [1.0],
        },
        geometry=[Point(500000.0, 3500000.0)],
        crs='EPSG:32631',
    )
    source_gdf.to_parquet(input_dir / 'sample.parquet')

    def fake_impute(self, column_to_fill_list='ta', product='era5_land', machine_learning_enabled=False, train_ratio=1, model_name='xgboost', export_dataset=False, **kwargs):
        self.data.get_dataframe()['filled'] = 1.0
        return self

    monkeypatch.setattr(ClimateFiller, 'impute', fake_impute)

    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'lon': [0.0], 'lat': [1.0], 'rs': [1.0]}),
        datetime_column_name='date',
        backend='local',
        lon='lon',
        lat='lat',
    )
    output_paths = cf.impute_batch(str(input_dir), str(output_dir), column_to_fill_list='rs', prefix='sample')

    assert len(output_paths) == 1
    written = gpd.read_parquet(output_dir / 'sample.parquet')
    assert written.crs is not None
    assert written.crs.to_string() == 'EPSG:32631'
    assert 'filled' in written.columns
    assert 'date' in written.columns


def test_eto_estimation_daily_pm_uses_preaggregated_columns(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    daily_df = pd.DataFrame(
        {
            'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'],
            't2m_max': [30.0, 31.0],
            't2m_min': [18.0, 19.0],
            'rh_max': [90.0, 88.0],
            'rh_min': [40.0, 42.0],
            'ws_mean': [2.0, 2.5],
            'rs': [220.0, 230.0],
        }
    )

    cf = ClimateFiller(
        daily_df,
        datetime_column_name='date',
        backend='local',
        lat=31.65,
        lon=-7.6,
        elevation=500,
    )

    monkeypatch.setattr(
        'climatefiller.Lib.eto_penman_monteith_daily',
        lambda row: 4.2,
    )

    result = cf.eto_estimation_daily(
        ta_max_column_name='t2m_max',
        ta_min_column_name='t2m_min',
        rh_max_column_name='rh_max',
        rh_min_column_name='rh_min',
        ws_mean_column_name='ws_mean',
        rs_mean_column_name='rs',
        methods_list=['pm'],
    )

    assert 'eto_pm' in result.columns
    assert list(result['eto_pm']) == [4.2, 4.2]
    assert 'ta_max' in result.columns
    assert 'rs_mean' in result.columns
    assert 'elevation' in result.columns


def test_eto_estimation_daily_multiple_methods_add_columns(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    daily_df = pd.DataFrame(
        {
            'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'],
            't2m_max': [30.0, 31.0],
            't2m_min': [18.0, 19.0],
            'rh_max': [80.0, 75.0],
            'rh_min': [40.0, 35.0],
            'ws_mean': [2.0, 2.5],
            'rs': [18.0, 20.0],
        }
    )
    cf = ClimateFiller(
        daily_df,
        datetime_column_name='date',
        backend='local',
        lat=31.65,
        lon=-7.6,
        elevation=500,
    )

    monkeypatch.setattr(
        'climatefiller.Lib.eto_penman_monteith_daily',
        lambda row: 4.2,
    )
    monkeypatch.setattr(
        'climatefiller.Lib.eto_hargreaves_samani',
        lambda row, c=0.0023, a=17.8, b=0.5: 3.1,
    )

    result = cf.eto_estimation_daily(
        ta_max_column_name='t2m_max',
        ta_min_column_name='t2m_min',
        rh_max_column_name='rh_max',
        rh_min_column_name='rh_min',
        ws_mean_column_name='ws_mean',
        rs_mean_column_name='rs',
        methods_list=['pm', 'hs'],
    )

    assert 'eto_pm' in result.columns
    assert 'eto_hs' in result.columns
    assert list(result['eto_pm']) == [4.2, 4.2]
    assert list(result['eto_hs']) == [3.1, 3.1]
    assert cf.eto_output_data.get_dataframe() is not None
    assert 'eto_pm' in cf.eto_output_data.get_dataframe().columns
    assert 'eto_hs' in cf.eto_output_data.get_dataframe().columns


def test_eto_estimation_daily_requires_method_specific_columns():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    daily_df = pd.DataFrame(
        {
            'date': ['2020-01-01 00:00:00'],
            'ta_max': [30.0],
            'ta_min': [18.0],
        }
    )
    cf = ClimateFiller(daily_df, datetime_column_name='date', backend='local', elevation=500)

    try:
        cf.eto_estimation_daily(methods_list=['pm'])
        raised = False
    except ValueError as exc:
        raised = True
        assert 'rs_mean' in str(exc) or 'Missing required daily column' in str(exc)

    assert raised


def test_elevation_number_is_kept_as_value():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    df = pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0], 'alt': [123.0]})
    cf = ClimateFiller(df, datetime_column_name='date', backend='local', elevation=450.5)

    assert cf.elevation == 450.5


def test_elevation_string_is_kept_as_column_name():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    df = pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0], 'alt': [812.0]})
    cf = ClimateFiller(df, datetime_column_name='date', backend='local', elevation='alt')

    assert cf.elevation == 'alt'
    assert cf._get_numeric_elevation() == 812.0


def test_elevation_string_missing_column_raises():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    df = pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]})
    try:
        ClimateFiller(df, datetime_column_name='date', backend='local', elevation='alt')
        raised = False
    except ValueError as exc:
        raised = True
        assert 'alt' in str(exc)

    assert raised


def test_add_elevation_column_uses_number_column_or_api(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    df = pd.DataFrame(
        {
            'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'],
            'alt': [100.0, 100.0],
            'value': [1.0, 2.0],
        }
    )

    # number
    cf_num = ClimateFiller(df, datetime_column_name='date', backend='local', elevation=250.0)
    target_num = cf_num.eto_output_data
    target_num.set_dataframe(cf_num.data.get_dataframe().copy())
    cf_num._add_elevation_column(target_num)
    assert list(target_num.get_dataframe()['elevation']) == [250.0, 250.0]

    # column name
    cf_col = ClimateFiller(df, datetime_column_name='date', backend='local', elevation='alt')
    target_col = cf_col.eto_output_data
    target_col.set_dataframe(cf_col.data.get_dataframe().copy())
    cf_col._add_elevation_column(target_col)
    assert list(target_col.get_dataframe()['elevation']) == [100.0, 100.0]

    # None -> API fallback
    cf_none = ClimateFiller(df, datetime_column_name='date', backend='local', elevation=None)
    monkeypatch.setattr('climatefiller.Lib.get_elevation', lambda lat, lon: 777.0)
    target_none = cf_none.eto_output_data
    target_none.set_dataframe(cf_none.data.get_dataframe().copy())
    cf_none._add_elevation_column(target_none)
    assert list(target_none.get_dataframe()['elevation']) == [777.0, 777.0]


def test_era5_cache_validation_detects_missing_value_column(tmp_path):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    valid_path = tmp_path / 'valid.csv'
    pd.DataFrame({'datetime': ['2016-01-01 00:00:00'], 'first': [1.0]}).to_csv(valid_path, index=False)

    invalid_path = tmp_path / 'invalid.csv'
    pd.DataFrame({'datetime': ['2016-01-01 00:00:00']}).to_csv(invalid_path, index=False)

    cf = ClimateFiller(pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]}), datetime_column_name='date', backend='local')

    assert cf._era5_cache_is_valid(str(valid_path)) is True
    assert cf._era5_cache_is_valid(str(invalid_path)) is False
    assert cf._invalidate_era5_cache_if_invalid(str(invalid_path)) is True
    assert not invalid_path.exists()


def test_ensure_era5_value_column_renames_band_or_first():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]}), datetime_column_name='date', backend='local')

    data = _StubDataFrame(pd.DataFrame({
        'datetime': pd.to_datetime(['2016-01-01 00:00:00']),
        'surface_solar_radiation_downwards': [10.0],
    }))
    mapped = ClimateFiller._ensure_era5_value_column(
        data,
        'ssrd',
        ['first', 'surface_solar_radiation_downwards', 'ssrd'],
    )
    assert mapped == 'ssrd'
    assert 'ssrd' in data.get_columns_names()


def test_eto_estimation_daily_batch_writes_outputs_with_datetime(tmp_path, monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    output_dir.mkdir()

    source_df = pd.DataFrame(
        {
            'date': ['2020-01-01 00:00:00'],
            't2m_max': [30.0],
            't2m_min': [18.0],
            'rh_max': [90.0],
            'rh_min': [40.0],
            'ws_mean': [2.0],
            'rs': [220.0],
            'lon': [0.0],
            'lat': [1.0],
            'alt': [100.0],
        }
    )
    source_df.to_parquet(input_dir / 'sample.parquet')

    def fake_eto_daily(self, **kwargs):
        out = self.data.get_dataframe().copy()
        out['eto_pm'] = 4.2
        out['lon'] = 0.0
        out['lat'] = 1.0
        self.eto_output_data.set_dataframe(out)
        return out

    monkeypatch.setattr(ClimateFiller, 'eto_estimation_daily', fake_eto_daily)

    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'lon': [0.0], 'lat': [1.0], 'alt': [100.0], 'value': [1.0]}),
        datetime_column_name='date',
        backend='local',
        lon='lon',
        lat='lat',
        elevation='alt',
        frequency='d',
    )
    output_paths = cf.eto_estimation_daily_batch(
        str(input_dir),
        str(output_dir),
        ta_max_column_name='t2m_max',
        ta_min_column_name='t2m_min',
        rh_max_column_name='rh_max',
        rh_min_column_name='rh_min',
        ws_mean_column_name='ws_mean',
        rs_mean_column_name='rs',
        methods_list=['pm'],
        prefix='sample',
    )

    assert len(output_paths) == 1
    written = pd.read_parquet(output_dir / 'sample.parquet')
    assert 'eto_pm' in written.columns
    assert 'date' in written.columns


def test_eto_estimation_batch_writes_outputs_with_datetime(tmp_path, monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    output_dir.mkdir()

    source_df = pd.DataFrame(
        {
            'date': ['2020-01-01 00:00:00', '2020-01-01 01:00:00'],
            'ta': [20.0, 21.0],
            'rh': [60.0, 55.0],
            'ws': [2.0, 2.5],
            'rs': [100.0, 120.0],
            'lon': [0.0, 0.0],
            'lat': [1.0, 1.0],
            'alt': [100.0, 100.0],
        }
    )
    source_df.to_parquet(input_dir / 'sample.parquet')

    def fake_eto(self, **kwargs):
        out = self.data.get_dataframe().copy()
        out['eto_pm'] = 3.5
        out['lon'] = 0.0
        out['lat'] = 1.0
        self.eto_output_data.set_dataframe(out)
        return out

    monkeypatch.setattr(ClimateFiller, 'eto_estimation', fake_eto)

    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'lon': [0.0], 'lat': [1.0], 'alt': [100.0], 'value': [1.0]}),
        datetime_column_name='date',
        backend='local',
        lon='lon',
        lat='lat',
        elevation='alt',
        frequency='h',
    )
    output_paths = cf.eto_estimation_batch(
        str(input_dir),
        str(output_dir),
        ta_column_name='ta',
        rh_column_name='rh',
        ws_column_name='ws',
        rs_column_name='rs',
        method='pm',
        freq='d',
        prefix='sample',
    )

    assert len(output_paths) == 1
    written = pd.read_parquet(output_dir / 'sample.parquet')
    assert 'eto_pm' in written.columns
    assert 'date' in written.columns
