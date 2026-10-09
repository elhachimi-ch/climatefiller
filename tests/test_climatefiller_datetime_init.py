import os
import sys
import types

import numpy as np
import pandas as pd
import pytest


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

    def add_hod_column(self, datetime_column_name='datetime'):
        self.dataframe['hod'] = pd.to_datetime(self.dataframe[datetime_column_name]).dt.hour
        return self

    def add_one_value_column(self, column_name, value):
        self.dataframe[column_name] = value
        return self

    def add_column(self, column_name, values):
        self.dataframe[column_name] = values
        return self

    def resample_timeseries(self, in_place=False, agg='mean', freq='D'):
        if agg == 'max':
            out = self.dataframe.resample(freq).max()
        elif agg == 'min':
            out = self.dataframe.resample(freq).min()
        elif agg == 'median':
            out = self.dataframe.resample(freq).median()
        else:
            out = self.dataframe.resample(freq).mean()
        if in_place:
            self.dataframe = out
            return self.dataframe
        return out

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
import climatefiller as climatefiller_module


def test_gee_project_pool_parses_and_deduplicates_environment_values(monkeypatch):
    monkeypatch.setenv('EE_PROJECT_POOL', 'project-a, project-b, project-a,')
    monkeypatch.setenv('GEE_PROJECT', 'legacy-project')

    assert ClimateFiller._get_gee_project_pool() == ['project-a', 'project-b']


def test_gee_initialization_tries_configured_projects_in_order(monkeypatch, tmp_path):
    monkeypatch.setenv('EE_PROJECT_POOL', 'unavailable-project, working-project')
    monkeypatch.setattr(climatefiller_module.random, 'shuffle', lambda projects: None)
    initialized_projects = []

    def initialize(project=None):
        initialized_projects.append(project)
        if project == 'unavailable-project':
            raise RuntimeError('Project unavailable')

    monkeypatch.setattr(climatefiller_module.ee, 'Initialize', initialize)
    cf = ClimateFiller(backend='gee', artifact_folder=str(tmp_path / 'artifacts'))

    assert initialized_projects == ['unavailable-project', 'working-project']
    assert cf._gee_project_index == 1


def test_gee_request_retries_with_next_project_after_quota_failure():
    class _DummyClimateFiller:
        backend = 'gee'
        _gee_projects = ['project-a', 'project-b']
        _gee_project_index = 0

        _is_gee_project_failure = staticmethod(ClimateFiller._is_gee_project_failure)

        def _initialize_gee_project(self, project_index):
            self._gee_project_index = project_index

        @climatefiller_module._with_gee_project_failover
        def fetch(self):
            if self._gee_projects[self._gee_project_index] == 'project-a':
                raise RuntimeError('Earth Engine quota exceeded')
            return self._gee_projects[self._gee_project_index]

    assert _DummyClimateFiller().fetch() == 'project-b'


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


def test_add_extraterrestrial_radiation_daily_column_uses_latitude_and_day_of_year(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00']}),
        datetime_column_name='date',
        backend='local',
        lat=31.5,
    )
    inputs = []

    def extraterrestrial_radiation(lat, doy):
        inputs.append((lat, doy))
        return lat + doy / 10

    monkeypatch.setattr(climatefiller_module.Lib, 'extraterrestrial_radiation_daily', extraterrestrial_radiation)
    cf.add_extraterrestrial_radiation_daily_column(nbr_decimal_places=1)

    dataframe = cf.data.get_dataframe()
    assert inputs == [(31.5, 1), (31.5, 2)]
    assert dataframe['ra'].tolist() == [31.6, 31.7]
    assert dataframe.index.equals(pd.DatetimeIndex(['2020-01-01', '2020-01-02'], name='datetime'))


def test_add_extraterrestrial_radiation_daily_column_accepts_latitude_and_doy_columns(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({
            'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'],
            'latitude': [10.0, 20.0],
            'day_number': [100, 200],
        }),
        datetime_column_name='date',
        backend='local',
    )
    inputs = []
    monkeypatch.setattr(
        climatefiller_module.Lib,
        'extraterrestrial_radiation_daily',
        lambda lat, doy: inputs.append((lat, doy)) or lat + doy,
    )

    cf.add_extraterrestrial_radiation_daily_column(
        lat_column_name='latitude',
        doy_column_name='day_number',
    )

    assert inputs == [(10.0, 100), (20.0, 200)]
    assert cf.data.get_dataframe()['ra'].tolist() == [110.0, 220.0]


def test_add_extraterrestrial_radiation_daily_column_derives_doy_from_named_datetime(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({
            'date': ['2020-01-01 00:00:00', '2020-12-31 00:00:00'],
            'observation_time': ['2020-02-03', '2020-03-04'],
        }),
        datetime_column_name='date',
        backend='local',
        lat=31.5,
    )
    inputs = []
    monkeypatch.setattr(
        climatefiller_module.Lib,
        'extraterrestrial_radiation_daily',
        lambda lat, doy: inputs.append((lat, doy)) or doy,
    )

    cf.add_extraterrestrial_radiation_daily_column(
        datetime_column_name='observation_time',
    )

    assert inputs == [(31.5, 34), (31.5, 64)]
    assert cf.data.get_dataframe()['ra'].tolist() == [34, 64]


def test_add_extraterrestrial_radiation_daily_column_rejects_both_doy_sources():
    os.environ.setdefault('GEE_PROJECT', 'dummy')
    cf = ClimateFiller(
        pd.DataFrame({
            'date': ['2020-01-01 00:00:00'],
            'doy': [1],
        }),
        datetime_column_name='date',
        backend='local',
    )

    with pytest.raises(ValueError, match='either doy_column_name or datetime_column_name'):
        cf.add_extraterrestrial_radiation_daily_column(
            doy_column_name='doy',
            datetime_column_name='date',
        )


def test_add_extraterrestrial_radiation_daily_column_uses_named_lat_column(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'], 'latitude': [10.0, 20.0]}),
        datetime_column_name='date',
        backend='local',
        lat='latitude',
    )
    inputs = []
    monkeypatch.setattr(
        climatefiller_module.Lib,
        'extraterrestrial_radiation_daily',
        lambda lat, doy: inputs.append((lat, doy)) or lat,
    )

    cf.add_extraterrestrial_radiation_daily_column()

    assert inputs == [(10.0, 1), (20.0, 2)]


def test_add_extraterrestrial_radiation_daily_column_requires_latitude():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]}),
        datetime_column_name='date',
        backend='local',
    )

    with pytest.raises(ValueError, match='Extraterrestrial radiation needs lat'):
        cf.add_extraterrestrial_radiation_daily_column()


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


def test_constructor_lat_lon_default_to_none_without_columns():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]}),
        datetime_column_name='date',
        backend='local',
    )

    assert cf.lat is None
    assert cf.lon is None
    assert 'lat' not in cf.data.get_columns_names()
    assert 'lon' not in cf.data.get_columns_names()


def test_constructor_adds_class_level_lat_lon_as_columns():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'], 'value': [1.0, 2.0]}),
        datetime_column_name='date',
        backend='local',
        lat=30.2708,
        lon=66.9398,
    )

    dataframe = cf.data.get_dataframe()
    assert dataframe['lat'].tolist() == [30.2708, 30.2708]
    assert dataframe['lon'].tolist() == [66.9398, 66.9398]
    assert (cf.lat, cf.lon) == (30.2708, 66.9398)


def test_constructor_reads_lat_lon_only_from_named_columns(caplog):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    df = pd.DataFrame({
        'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'],
        'lat': [30.2708, 30.2708],
        'lon': [66.9398, 66.9398],
    })

    named = ClimateFiller(df, datetime_column_name='date', backend='local', lat='lat', lon='lon')
    assert (named.lat, named.lon) == (30.2708, 66.9398)
    assert named.data.get_dataframe()['lat'].tolist() == [30.2708, 30.2708]

    unnamed = ClimateFiller(df, datetime_column_name='date', backend='local')
    assert (unnamed.lat, unnamed.lon) == (None, None)

    with caplog.at_level('WARNING'):
        numbers = ClimateFiller(df, datetime_column_name='date', backend='local', lat=29.325, lon=71.819)
    assert (numbers.lat, numbers.lon) == (29.325, 71.819)
    assert numbers.data.get_dataframe()['lat'].tolist() == [29.325, 29.325]
    assert "lat=29.325 replaces the data's 'lat' column" in caplog.text
    assert "pass lon='lon' to use the column instead" in caplog.text


def test_constructor_rejects_missing_named_lat_column():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    with pytest.raises(ValueError, match="Latitude column 'latitude' was not found"):
        ClimateFiller(
            pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]}),
            datetime_column_name='date',
            backend='local',
            lat='latitude',
        )


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


def test_convert_era5_rs_to_mj_m2_day_sums_joules():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    # Accumulated SSRD in J/m2 with a typical ERA5-Land daily reset around 01:00.
    index = pd.date_range('2020-01-01 00:00:00', periods=26, freq='h')
    accumulated = [
        0, 0, 0, 1000, 5000, 15000, 40000, 90000,
        160000, 250000, 360000, 490000, 640000, 810000, 1_000_000,
        1_210_000, 1_440_000, 1_440_000, 1_440_000, 1_440_000, 1_440_000,
        1_440_000, 1_440_000, 1_440_000, 1_440_000, 0,
    ]
    ssrd = pd.Series(accumulated, index=index)
    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'rs': [np.nan]}),
        datetime_column_name='date',
        backend='local',
        frequency='d',
    )

    default_wm2 = cf._convert_era5_rs_series_for_target_unit(ssrd, target_unit=None, target_frequency_label='hourly')
    assert default_wm2.index.equals(ssrd.index)
    assert np.isclose(default_wm2.loc[pd.Timestamp('2020-01-01 04:00:00')], (5000 - 1000) / 3600.0)

    mj_daily = cf._convert_era5_rs_series_for_target_unit(
        ssrd,
        target_unit='mj/m2/day',
        target_frequency_label='daily',
    )
    # Net positive energy over 2020-01-01 from the accumulated profile.
    expected_mj = 1_440_000 / 1_000_000.0
    assert np.isclose(mj_daily.loc[pd.Timestamp('2020-01-01')], expected_mj, rtol=1e-6)


def test_align_rs_with_mj_unit_sums_hourly_wm2_to_daily_mj():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    daily_index = pd.date_range('2020-01-01 00:00:00', periods=1, freq='D')
    # 100 W/m2 for 24 hours -> 100 * 3600 * 24 / 1e6 = 8.64 MJ/m2/day
    source_series = pd.Series(
        [100.0] * 24,
        index=pd.date_range('2020-01-01 00:00:00', periods=24, freq='h'),
    )
    cf = ClimateFiller(
        pd.DataFrame({'date': daily_index, 'rs': [np.nan]}),
        datetime_column_name='date',
        backend='local',
        frequency='d',
    )
    cf._impute_unit_dict = {'rs': 'mj/m2/day'}

    aligned = cf._align_source_series_to_target_frequency(
        source_series,
        'rs',
        daily_index,
        target_unit='mj/m2/day',
    )
    assert np.isclose(aligned.iloc[0], 8.64)


def test_impute_accepts_unit_dict(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    df = pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'rs': [np.nan], 'ta': [np.nan]})
    cf = ClimateFiller(df, datetime_column_name='date', backend='local', frequency='d')

    seen = {}

    def fake_impute_single(self, column_to_fill_name='ta', **kwargs):
        seen[column_to_fill_name] = kwargs.get('unit_dict')

    monkeypatch.setattr(ClimateFiller, '_impute_single_column', fake_impute_single)

    cf.impute(column_to_fill_list=['rs', 'ta'], unit_dict={'rs': 'mj/m2/day'})

    assert seen['rs'] == {'rs': 'mj/m2/day'}
    assert seen['ta'] == {'rs': 'mj/m2/day'}
    assert cf._impute_unit_dict == {'rs': 'mj/m2/day'}


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


def test_export_creates_missing_parent_directory_for_table_output(tmp_path):
    class _RecordingDataFrame:
        def __init__(self):
            self.parent_existed_when_exported = False

        def export(self, path_link, data_type=None, **kwargs):
            self.parent_existed_when_exported = os.path.isdir(os.path.dirname(path_link))

    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]}),
        datetime_column_name='date',
        backend='local',
    )
    cf.data = _RecordingDataFrame()
    output_path = tmp_path / 'new' / 'nested' / 'output.csv'

    cf.export(output_path)

    assert cf.data.parent_existed_when_exported


def test_download_solar_radiation_mcd18_modis_appends_daily_column(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01 00:00:00', periods=48, freq='h').astype(str),
        'lon': 71.8,
        'lat': 29.3,
        'value': 1.0,
    })
    cf = ClimateFiller(df, datetime_column_name='date', lon='lon', lat='lat', backend='local')
    calls = []

    def fake_fetch(self, lon, lat, start_date, end_date, scale=1000, collection_name='x', show_progress=True):
        calls.append((lon, lat, start_date, end_date))
        return pd.Series([8.64], index=pd.to_datetime(['2020-01-01']))

    monkeypatch.setattr(ClimateFiller, '_mcd18_fetch_daily_dsr', fake_fetch)

    cf.download_solar_radiation_mcd18_modis()
    result = cf.data.get_dataframe()

    assert calls[0][:2] == (71.8, 29.3)
    assert calls[0][2].strftime('%Y-%m-%d') == '2020-01-01'
    assert calls[0][3].strftime('%Y-%m-%d') == '2020-01-02'
    assert (result['rs_mcd18_modis'].iloc[:24] == 8.64).all()
    assert result['rs_mcd18_modis'].iloc[24:].isna().all()

    cf.download_solar_radiation_mcd18_modis(column_name='rs_w', unit='W/m2')
    assert abs(cf.data.get_dataframe()['rs_w'].iloc[0] - 100.0) < 1e-9


def test_compare_columns_writes_reports_and_figures(tmp_path):
    import matplotlib
    matplotlib.use('Agg')

    rng = np.random.default_rng(0)
    dates = pd.date_range('2019-01-01', '2021-12-31', freq='D')
    seasonal = 15 + 8 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365)
    reference = seasonal + rng.normal(0, 1.5, len(dates))
    estimate = np.asarray(1.05 * reference + 0.5 + rng.normal(0, 1.0, len(dates)))
    estimate[10:20] = np.nan
    df = pd.DataFrame({'date': dates.astype(str), 'rs': reference, 'rs_mcd18_modis': estimate})
    cf = ClimateFiller(df, datetime_column_name='date', backend='local')

    out = tmp_path / 'nested' / 'report'
    result = cf.compare_columns('rs', 'rs_mcd18_modis', output_folder=out, figure_formats=('png',), dpi=60)

    for name in (
        'summary_metrics.csv', 'metrics_by_year.csv', 'metrics_by_month_of_year.csv',
        'metrics_by_season.csv', 'paired_native.csv', 'paired_weekly.csv', 'paired_monthly.csv',
        'paired_yearly.csv', 'report.md', 'report.txt', 'fig_scatter_by_scale.png',
        'fig_timeseries_by_scale.png', 'fig_seasonal_cycle.png', 'fig_bland_altman.png',
        'fig_error_distribution.png', 'fig_metrics_by_year.png',
    ):
        assert (out / name).exists(), name

    native = result['summary'].set_index('scale').loc['native']
    assert native['n'] == len(dates) - 10
    assert native['r'] > 0.95
    assert native['bias'] > 0
    assert len(result['by_year']) == 3
    assert list(result['by_season']['season']) == ['DJF', 'MAM', 'JJA', 'SON']
    assert 'Suggested text for the methods section' in result['report_markdown']


def test_agreement_metrics_known_values():
    from lib import Lib

    perfect = Lib.agreement_metrics([1, 2, 3, 4], [1, 2, 3, 4])
    assert perfect['rmse'] == 0 and perfect['bias'] == 0
    assert abs(perfect['nse'] - 1) < 1e-12 and abs(perfect['kge'] - 1) < 1e-12
    shifted = Lib.agreement_metrics([1, 2, 3, 4], [2, 3, 4, np.nan])
    assert shifted['n'] == 3 and shifted['bias'] == 1.0 and abs(shifted['r'] - 1) < 1e-12


def test_impute_single_column_normalizes_mixed_timezone_indexes(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({
            'date': ['2020-01-01 00:00:00+00:00', '2020-01-01 01:00:00+00:00'],
            'rh_max': [np.nan, np.nan],
        }),
        datetime_column_name='date',
        backend='gee',
        lat=31.65,
        lon=-7.6,
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
        lambda row, units_dict=None: 4.2,
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
        lambda row, units_dict=None: 4.2,
    )
    monkeypatch.setattr(
        'climatefiller.Lib.eto_hargreaves_samani',
        lambda row, c=0.0023, a=17.8, b=0.5, units_dict=None: 3.1,
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


def test_convert_rs_to_mj_m2_day_respects_units_dict():
    from lib import Lib

    # Default (no unit): already MJ/m2/day
    assert Lib.convert_rs_to_mj_m2_day(18.0) == 18.0
    assert Lib.convert_rs_to_mj_m2_day(18.0, units_dict=None) == 18.0
    assert Lib.convert_rs_to_mj_m2_day(18.0, units_dict={}) == 18.0
    assert Lib.convert_rs_to_mj_m2_day(100.0, units_dict={}, legacy_factor=0.0864) == 100.0 * 0.0864

    # Already MJ/m2/day: no conversion
    assert Lib.convert_rs_to_mj_m2_day(18.0, units_dict={'rs': 'MJ/m2/day'}) == 18.0
    assert Lib.convert_rs_to_mj_m2_day(18.0, units_dict={'rs_mean': 'MJ/m2/day'}) == 18.0

    # Explicit W/m2: convert
    assert Lib.convert_rs_to_mj_m2_day(100.0, units_dict={'rs': 'W/m2'}) == 100.0 * 0.0864


def test_wind_speed_z_source_to_z_target_converts_scalar_and_column_values():
    from lib import Lib

    factor = np.log(2.0 / 0.03) / np.log(10.0 / 0.03)
    assert np.isclose(Lib.wind_speed_z_source_to_z_target(5.0), 5.0 * factor)

    speeds = pd.Series([5.0, 10.0], name='wind_speed')
    converted = Lib.wind_speed_z_source_to_z_target(speeds)

    assert converted.name == 'wind_speed'
    np.testing.assert_allclose(converted.to_numpy(), [5.0 * factor, 10.0 * factor])


def test_climatefiller_converts_named_wind_column_in_place():
    factor = np.log(2.0 / 0.03) / np.log(10.0 / 0.03)
    cf = ClimateFiller(
        pd.DataFrame({
            'date': ['2020-01-01 00:00:00', '2020-01-01 01:00:00'],
            'wind_speed_10m': [5.0, np.nan],
        }),
        datetime_column_name='date',
        backend='local',
    )

    result = cf.wind_speed_z_source_to_z_target('wind_speed_10m')

    assert result is cf
    converted = cf.data.get_dataframe()['wind_speed_10m']
    assert np.isclose(converted.iloc[0], 5.0 * factor)
    assert pd.isna(converted.iloc[1])


def test_climatefiller_wind_conversion_rejects_unknown_column():
    cf = ClimateFiller(
        pd.DataFrame({
            'date': ['2020-01-01 00:00:00'],
            'wind_speed_10m': [5.0],
        }),
        datetime_column_name='date',
        backend='local',
    )

    with np.testing.assert_raises_regex(ValueError, "Wind-speed column 'missing'"):
        cf.wind_speed_z_source_to_z_target('missing')


def test_wind_speed_z_source_to_z_target_validates_profile_heights():
    from lib import Lib

    with np.testing.assert_raises_regex(ValueError, 'greater than'):
        Lib.wind_speed_z_source_to_z_target(5.0, z_source=0.03)


def test_logarithmic_wind_profile_uses_height_conversion():
    from lib import Lib

    source_speed = np.hypot(3.0, 4.0)
    expected = Lib.wind_speed_z_source_to_z_target(source_speed)

    assert np.isclose(Lib.logarithmic_wind_profile(3.0, 4.0), expected)


def test_eto_estimation_daily_units_dict_skips_rs_conversion_when_mj():
    os.environ.setdefault('GEE_PROJECT', 'dummy')
    from lib import Lib

    base_row = {
        'ta_max': 30.0,
        'ta_min': 18.0,
        'rh_max': 80.0,
        'rh_min': 40.0,
        'ws_mean': 2.0,
        'lat': 31.65,
        'elevation': 500.0,
        'doy': 1,
    }

    # Same physical radiation: 18 MJ/m2/day == 18/0.0864 W/m2
    rs_mj = 18.0
    rs_wm2 = rs_mj / 0.0864

    eto_legacy = Lib.eto_penman_monteith_daily(
        {**base_row, 'rs_mean': rs_wm2}, units_dict={'rs': 'W/m2'}
    )
    eto_mj = Lib.eto_penman_monteith_daily(
        {**base_row, 'rs_mean': rs_mj},
        units_dict={'rs': 'MJ/m2/day'},
    )
    eto_default = Lib.eto_penman_monteith_daily({**base_row, 'rs_mean': rs_mj})

    assert abs(eto_legacy - eto_mj) < 1e-9
    # Without units_dict, rs is assumed to be MJ/m2/day
    assert abs(eto_default - eto_mj) < 1e-9

    daily_df = pd.DataFrame(
        {
            'date': ['2020-01-01 00:00:00'],
            'ta_max': [30.0],
            'ta_min': [18.0],
            'rh_max': [80.0],
            'rh_min': [40.0],
            'ws_mean': [2.0],
            'rs_mean': [rs_mj],
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
    result = cf.eto_estimation_daily(
        methods_list=['pm', 'ab', 'mk'],
        units_dict={'rs': 'MJ/m2/day'},
    )
    assert 'eto_pm' in result.columns
    assert 'eto_ab' in result.columns
    assert 'eto_mk' in result.columns
    assert abs(result['eto_pm'].iloc[0] - round(eto_mj, 2)) < 1e-9


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
    cf_none = ClimateFiller(df, datetime_column_name='date', backend='local', elevation=None, lat=31.65, lon=-7.6)
    monkeypatch.setattr('climatefiller.Lib.get_elevation', lambda lat, lon: 777.0)
    target_none = cf_none.eto_output_data
    target_none.set_dataframe(cf_none.data.get_dataframe().copy())
    cf_none._add_elevation_column(target_none)
    assert list(target_none.get_dataframe()['elevation']) == [777.0, 777.0]


def test_constructor_adds_class_level_elevation_as_column():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'], 'value': [1.0, 2.0]}),
        datetime_column_name='date',
        backend='local',
        elevation=1590,
    )

    assert cf.elevation == 1590.0
    assert cf.data.get_dataframe()['elevation'].tolist() == [1590.0, 1590.0]


def test_constructor_reads_elevation_only_from_named_column(monkeypatch, caplog):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    monkeypatch.setattr('climatefiller.Lib.get_elevation', lambda lat, lon: 777.0)
    df = pd.DataFrame({
        'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'],
        'elevation': [1590.0, 1600.0],
    })

    def elevation_rows(cf):
        target = cf.eto_output_data
        target.set_dataframe(cf.data.get_dataframe()[[]].copy())
        cf._add_elevation_column(target)
        return target.get_dataframe()['elevation'].tolist()

    named = ClimateFiller(df, datetime_column_name='date', backend='local', elevation='elevation')
    assert named.elevation == 'elevation'
    assert elevation_rows(named) == [1590.0, 1600.0]

    unnamed = ClimateFiller(df, datetime_column_name='date', backend='local', lat=31.65, lon=-7.6)
    assert unnamed.elevation is None
    assert elevation_rows(unnamed) == [777.0, 777.0]

    with caplog.at_level('WARNING'):
        number = ClimateFiller(df, datetime_column_name='date', backend='local', elevation=120)
    assert number.data.get_dataframe()['elevation'].tolist() == [120.0, 120.0]
    assert elevation_rows(number) == [120.0, 120.0]
    assert "elevation=120 replaces the data's 'elevation' column (first value 1590.0)" in caplog.text


def test_constructor_rejects_non_numeric_named_elevation_column():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    with pytest.raises(ValueError, match="Elevation column 'alt' does not contain numeric values"):
        ClimateFiller(
            pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'alt': ['n/a']}),
            datetime_column_name='date',
            backend='local',
            elevation='alt',
        )


def test_eto_estimation_daily_reads_named_elevation_column_row_by_row(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame(
            {
                'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'],
                'ta_max': [30.0, 30.0],
                'ta_min': [18.0, 18.0],
                'rh_max': [80.0, 80.0],
                'rh_min': [40.0, 40.0],
                'ws_mean': [2.0, 2.0],
                'rs_mean': [18.0, 18.0],
                'alt': [100.0, 200.0],
                'elevation': [999.0, 999.0],
            }
        ),
        datetime_column_name='date',
        backend='local',
        lat=31.65,
        elevation='alt',
    )
    elevations = []
    monkeypatch.setattr(
        'climatefiller.Lib.eto_penman_monteith_daily',
        lambda row, units_dict=None: elevations.append(row['elevation']) or 4.2,
    )

    cf.eto_estimation_daily(methods_list=['pm'])

    assert elevations == [100.0, 200.0]


def test_eto_estimation_resamples_elevation_column_to_daily(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    hourly = _hourly_station_dataframe(lat=[30.25] * 48, lon=[66.75] * 48, alt=[1590.0] * 48)
    hourly['date'] = hourly['date'] + pd.Timedelta(minutes=30)  # no row on the daily bin labels
    cf = ClimateFiller(
        hourly,
        datetime_column_name='date',
        backend='local',
        lat='lat',
        lon='lon',
        elevation='alt',
        frequency='h',
    )
    elevations = []
    monkeypatch.setattr(
        'climatefiller.Lib.eto_penman_monteith_daily',
        lambda row, units_dict=None: elevations.append(row['elevation']) or 4.2,
    )

    result = cf.eto_estimation(methods_list=['pm'], freq='d')

    assert elevations == [1590.0, 1590.0]
    assert result['elevation'].tolist() == [1590.0, 1590.0]


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


def test_eto_estimation_multiple_methods_add_columns(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    hourly_df = pd.DataFrame(
        {
            'date': pd.date_range('2020-01-01', periods=24, freq='h'),
            'ta': [20.0] * 24,
            'rh': [60.0] * 24,
            'ws': [2.0] * 24,
            'rs': [10.0] * 24,
        }
    )
    cf = ClimateFiller(
        hourly_df,
        datetime_column_name='date',
        backend='local',
        lat=31.65,
        lon=-7.6,
        elevation=500,
        frequency='h',
    )

    monkeypatch.setattr(
        'climatefiller.Lib.eto_penman_monteith_daily',
        lambda row, units_dict=None: 4.2,
    )
    monkeypatch.setattr(
        'climatefiller.Lib.eto_hargreaves_samani',
        lambda row, c=0.0023, a=17.8, b=0.5, units_dict=None: 3.1,
    )

    result = cf.eto_estimation(
        ta_column_name='ta',
        rh_column_name='rh',
        ws_column_name='ws',
        rs_column_name='rs',
        methods_list=['pm', 'hs'],
        freq='d',
    )

    assert 'eto_pm' in result.columns
    assert 'eto_hs' in result.columns
    assert list(result['eto_pm']) == [4.2]
    assert list(result['eto_hs']) == [3.1]


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
        methods_list=['pm'],
        freq='d',
        prefix='sample',
    )

    assert len(output_paths) == 1
    written = pd.read_parquet(output_dir / 'sample.parquet')
    assert 'eto_pm' in written.columns
    assert 'date' in written.columns


def _hourly_station_dataframe(periods=48, **extra_columns):
    hours = [h % 24 for h in range(periods)]
    data = {
        'date': pd.date_range('2020-06-01', periods=periods, freq='h'),
        'ta': [15.0 + 0.5 * h for h in hours],
        'rh': [40.0 + h for h in hours],
        'ws': [2.0] * periods,
        'rs': [500.0 if 6 <= h <= 18 else 0.0 for h in hours],
    }
    data.update(extra_columns)
    return pd.DataFrame(data)


def test_eto_estimation_uses_lat_lon_columns_from_data(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        _hourly_station_dataframe(lat=[30.25] * 48, lon=[66.75] * 48),
        datetime_column_name='date',
        backend='local',
        lat='lat',
        lon='lon',
        elevation=1590,
        frequency='h',
    )
    rows = []
    monkeypatch.setattr(
        'climatefiller.Lib.eto_penman_monteith_daily',
        lambda row, units_dict=None: rows.append((row['lat'], row['lon'])) or 4.2,
    )

    result = cf.eto_estimation(methods_list=['pm'], freq='d')

    assert rows == [(30.25, 66.75)] * 2
    assert result['lat'].tolist() == [30.25] * 2
    assert result['lon'].tolist() == [66.75] * 2


def test_eto_estimation_uses_instance_lat_lon_without_columns(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        _hourly_station_dataframe(),
        datetime_column_name='date',
        backend='local',
        lat=29.325,
        lon=71.819,
        elevation=120,
        frequency='h',
    )
    # e.g. a resample() that did not keep the lat/lon columns added at init
    cf.data.set_dataframe(cf.data.get_dataframe().drop(columns=['lat', 'lon']))
    rows = []
    monkeypatch.setattr(
        'climatefiller.Lib.eto_penman_monteith_daily',
        lambda row, units_dict=None: rows.append((row['lat'], row['lon'])) or 4.2,
    )

    cf.eto_estimation(methods_list=['pm'], freq='d')

    assert rows == [(29.325, 71.819)] * 2


def test_eto_estimation_units_dict_accepts_source_column_names():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    reference = _hourly_station_dataframe()
    converted = reference.rename(columns={'ta': 't2m', 'rs': 'ssrd'})
    converted['t2m'] = converted['t2m'] + 273.15
    converted['ssrd'] = converted['ssrd'] * 0.0036

    estimates = []
    for dataframe, kwargs in (
        (reference, {'units_dict': {'rs': 'W/m2'}}),
        (
            converted,
            {
                'ta_column_name': 't2m',
                'rs_column_name': 'ssrd',
                'units_dict': {'t2m': 'K', 'ssrd': 'MJ/m2/h'},
            },
        ),
    ):
        cf = ClimateFiller(
            dataframe,
            datetime_column_name='date',
            backend='local',
            lat=31.65,
            lon=-7.6,
            elevation=500,
            frequency='h',
        )
        estimates.append(
            cf.eto_estimation(methods_list=['pm', 'hs'], freq='d', nbr_decimal_places=6, **kwargs)
        )

    legacy, with_units = estimates
    assert (legacy['eto_pm'] > 0).all()
    assert np.allclose(legacy['eto_pm'], with_units['eto_pm'], atol=1e-5)
    assert np.allclose(legacy['eto_hs'], with_units['eto_hs'], atol=1e-5)


def test_eto_estimation_rejects_non_dict_units_dict():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        _hourly_station_dataframe(),
        datetime_column_name='date',
        backend='local',
        elevation=500,
        frequency='h',
    )

    with pytest.raises(TypeError, match='units_dict must be a dict'):
        cf.eto_estimation(units_dict=['MJ/m2/day'])


def test_eto_estimation_hourly_keeps_lat_lon_columns_and_forwards_units_dict(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        _hourly_station_dataframe(periods=3, lat=[30.25] * 3, lon=[66.75] * 3),
        datetime_column_name='date',
        backend='local',
        lat='lat',
        lon='lon',
        elevation=1590,
        frequency='h',
    )
    calls = []

    def fake_penman_monteith_hourly(row, ta, rs, rh, ws, tz_offset, reference_crop, units_dict=None):
        calls.append((row['lat'], row['lon'], units_dict))
        return 0.5

    monkeypatch.setattr('climatefiller.Lib.eto_penman_monteith_hourly', fake_penman_monteith_hourly)

    result = cf.eto_estimation(methods_list=['pm'], freq='h', units_dict={'rs': 'MJ/m2/h'})

    assert [call[:2] for call in calls] == [(30.25, 66.75)] * 3
    assert all(call[2]['rs'] == 'MJ/m2/h' for call in calls)
    assert result['lat'].tolist() == [30.25] * 3
    assert result['lon'].tolist() == [66.75] * 3


def test_eto_estimation_daily_uses_lat_lon_columns_from_data(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame(
            {
                'date': ['2015-09-17 00:00:00', '2015-09-18 00:00:00'],
                'ta_max': [33.0, 33.0],
                'ta_min': [21.7, 16.2],
                'lat': [30.25, 30.25],
                'lon': [66.75, 66.75],
            }
        ),
        datetime_column_name='date',
        backend='local',
        lat='lat',
        lon='lon',
    )
    rows = []
    monkeypatch.setattr(
        'climatefiller.Lib.eto_hargreaves_samani',
        lambda row, c=0.0023, a=17.8, b=0.5, units_dict=None: rows.append((row['lat'], row['lon'])) or 5.0,
    )

    cf.eto_estimation_daily(methods_list=['hs'])

    assert rows == [(30.25, 66.75)] * 2


def test_eto_estimation_batch_forwards_units_dict(tmp_path, monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    _hourly_station_dataframe(periods=2).to_csv(input_dir / 'station.csv', index=False)
    received = {}

    def fake_eto(self, **kwargs):
        received.update(kwargs)
        out = self.data.get_dataframe().copy()
        out['eto_pm'] = 3.5
        self.eto_output_data.set_dataframe(out)
        return out

    monkeypatch.setattr(ClimateFiller, 'eto_estimation', fake_eto)

    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'value': [1.0]}),
        datetime_column_name='date',
        backend='local',
    )
    output_paths = cf.eto_estimation_batch(
        str(input_dir),
        str(output_dir),
        units_dict={'rs': 'MJ/m2/day'},
    )

    assert len(output_paths) == 1
    assert received['units_dict'] == {'rs': 'MJ/m2/day'}


def test_eto_estimation_requires_location_only_for_methods_that_use_it():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    unknown = ClimateFiller(
        _hourly_station_dataframe(),
        datetime_column_name='date',
        backend='local',
        elevation=500,
        frequency='h',
    )
    with pytest.raises(ValueError, match=r"ETo \(pm\) needs lat: pass lat= to ClimateFiller"):
        unknown.eto_estimation(methods_list=['pm'], freq='d')
    without_location = unknown.eto_estimation(methods_list=['ab'], freq='d')
    assert 'eto_ab' in without_location.columns
    assert 'lat' not in without_location.columns

    # 'lat'/'lon' columns are only used when named with lat='lat', lon='lon'.
    unnamed_columns = ClimateFiller(
        _hourly_station_dataframe(lat=[30.25] * 48, lon=[66.75] * 48),
        datetime_column_name='date',
        backend='local',
        elevation=500,
        frequency='h',
    )
    with pytest.raises(ValueError, match=r"ETo \(pm\) needs lat"):
        unnamed_columns.eto_estimation(methods_list=['pm'], freq='d')

    lat_only = ClimateFiller(
        _hourly_station_dataframe(),
        datetime_column_name='date',
        backend='local',
        lat=30.25,
        elevation=500,
        frequency='h',
    )
    daily = lat_only.eto_estimation(methods_list=['pm'], freq='d')
    assert daily['lat'].tolist() == [30.25] * 2
    assert 'lon' not in daily.columns
    with pytest.raises(ValueError, match=r"ETo \(pm\) needs lon"):
        lat_only.eto_estimation(methods_list=['pm'], freq='h')


def test_impute_requires_location_only_when_values_are_missing():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    dates = ['2020-01-01 00:00:00', '2020-01-01 01:00:00']
    complete = ClimateFiller(
        pd.DataFrame({'date': dates, 'ta': [20.0, 21.0]}),
        datetime_column_name='date',
        backend='local',
    )
    complete._impute_single_column('ta')

    gappy = ClimateFiller(
        pd.DataFrame({'date': dates, 'ta': [20.0, np.nan]}),
        datetime_column_name='date',
        backend='local',
    )
    with pytest.raises(ValueError, match=r"Imputing 'ta' from era5_land needs lat and lon"):
        gappy._impute_single_column('ta')


def test_export_without_coordinates_writes_plain_parquet(tmp_path):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'], 'value': [1.0, 2.0]}),
        datetime_column_name='date',
        backend='local',
    )
    output_path = tmp_path / 'out.parquet'

    assert cf.export(str(output_path)) is None

    written = pd.read_parquet(output_path)
    assert written['value'].tolist() == [1.0, 2.0]
    assert 'geometry' not in written.columns
    assert isinstance(written.index, pd.DatetimeIndex)
    with pytest.raises(ValueError, match='Longitude/latitude columns were not found'):
        cf.export(str(tmp_path / 'out.geojson'))


def test_eto_estimation_daily_batch_forwards_lat_lon_column_names(tmp_path, monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    for station, latitude in (('a', 10.0), ('b', 20.0)):
        pd.DataFrame(
            {
                'date': ['2020-01-01', '2020-01-02'],
                'ta_max': [30.0, 31.0],
                'ta_min': [18.0, 19.0],
                'lat': [latitude, latitude],
                'lon': [5.0, 5.0],
            }
        ).to_csv(input_dir / f'{station}.csv', index=False)
    latitudes = []
    monkeypatch.setattr(
        'climatefiller.Lib.eto_hargreaves_samani',
        lambda row, c=0.0023, a=17.8, b=0.5, units_dict=None: latitudes.append(row['lat']) or 5.0,
    )

    # The template's own data resolves lat to 0.0; the files must still use their columns.
    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'lat': [0.0], 'lon': [0.0]}),
        datetime_column_name='date',
        backend='local',
        lat='lat',
        lon='lon',
    )
    cf.eto_estimation_daily_batch(str(input_dir), str(output_dir), methods_list=['hs'], datetime_format='%Y-%m-%d')

    assert latitudes == [10.0, 10.0, 20.0, 20.0]


def test_build_geodataframe_uses_named_coordinate_columns(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({
            'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'],
            'LAT_DEG': [10.0, 20.0],
            'LON_DEG': [1.0, 2.0],
            'x': [500.0, 600.0],
        }),
        datetime_column_name='date',
        backend='local',
        lat='LAT_DEG',
        lon='LON_DEG',
    )

    captured = {}
    monkeypatch.setattr(
        climatefiller_module.gpd,
        'points_from_xy',
        lambda x, y, crs=None: captured.update(x=list(x), y=list(y)),
    )

    cf._build_geodataframe_from_dataframe(cf.data.get_dataframe().copy())

    assert captured == {'x': [1.0, 2.0], 'y': [10.0, 20.0]}


def test_constructor_validates_doy_column_name():
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    df = pd.DataFrame({'date': ['2020-01-01 00:00:00'], 'day_number': [0]})
    with pytest.raises(ValueError, match="Day-of-year column 'doy' was not found"):
        ClimateFiller(df, datetime_column_name='date', backend='local', doy_column_name='doy')
    with pytest.raises(ValueError, match="'day_number' must hold values from 1 to 366"):
        ClimateFiller(df, datetime_column_name='date', backend='local', doy_column_name='day_number')
    with pytest.raises(TypeError, match='doy_column_name must be a column name string'):
        ClimateFiller(df, datetime_column_name='date', backend='local', doy_column_name=1)


def test_eto_estimation_daily_uses_doy_column_name(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    df = pd.DataFrame({
        'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'],
        'ta_max': [30.0, 30.0],
        'ta_min': [18.0, 18.0],
        'day_number': [100, 101],
    })
    doys = []
    monkeypatch.setattr(
        'climatefiller.Lib.eto_hargreaves_samani',
        lambda row, c=0.0023, a=17.8, b=0.5, units_dict=None: doys.append(row['doy']) or 5.0,
    )

    derived = ClimateFiller(df, datetime_column_name='date', backend='local', lat=31.65)
    derived.eto_estimation_daily(methods_list=['hs'])
    named = ClimateFiller(df, datetime_column_name='date', backend='local', lat=31.65, doy_column_name='day_number')
    named.eto_estimation_daily(methods_list=['hs'])

    assert doys == [1, 2, 100, 101]


def test_eto_estimation_uses_doy_column_name_for_daily_and_hourly(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        _hourly_station_dataframe(day_number=[150] * 24 + [151] * 24),
        datetime_column_name='date',
        backend='local',
        lat=30.25,
        lon=66.75,
        elevation=1590,
        frequency='h',
        doy_column_name='day_number',
    )
    daily_doys = []
    hourly_doys = []
    monkeypatch.setattr(
        'climatefiller.Lib.eto_penman_monteith_daily',
        lambda row, units_dict=None: daily_doys.append(row['doy']) or 4.2,
    )
    monkeypatch.setattr(
        'climatefiller.Lib.eto_penman_monteith_hourly',
        lambda row, *args, units_dict=None: hourly_doys.append(row['doy']) or 0.2,
    )

    cf.eto_estimation(methods_list=['pm'], freq='d')
    cf.eto_estimation(methods_list=['pm'], freq='h')

    assert daily_doys == [150, 151]
    assert hourly_doys == [150] * 24 + [151] * 24


def test_add_extraterrestrial_radiation_daily_column_uses_class_level_doy_column(monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    cf = ClimateFiller(
        pd.DataFrame({'date': ['2020-01-01 00:00:00', '2020-01-02 00:00:00'], 'day_number': [100, 200]}),
        datetime_column_name='date',
        backend='local',
        lat=31.5,
        doy_column_name='day_number',
    )
    inputs = []
    monkeypatch.setattr(
        climatefiller_module.Lib,
        'extraterrestrial_radiation_daily',
        lambda lat, doy: inputs.append((lat, doy)) or doy,
    )

    cf.add_extraterrestrial_radiation_daily_column()
    cf.add_extraterrestrial_radiation_daily_column(column_name='ra_calendar', datetime_column_name='datetime')

    assert inputs == [(31.5, 100), (31.5, 200), (31.5, 1), (31.5, 2)]


def test_eto_estimation_daily_batch_forwards_doy_column_name(tmp_path, monkeypatch):
    os.environ.setdefault('GEE_PROJECT', 'dummy')

    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    input_dir.mkdir()
    pd.DataFrame(
        {
            'date': ['2020-01-01', '2020-01-02'],
            'ta_max': [30.0, 31.0],
            'ta_min': [18.0, 19.0],
            'day_number': [100, 101],
        }
    ).to_csv(input_dir / 'station.csv', index=False)
    doys = []
    monkeypatch.setattr(
        'climatefiller.Lib.eto_hargreaves_samani',
        lambda row, c=0.0023, a=17.8, b=0.5, units_dict=None: doys.append(row['doy']) or 5.0,
    )

    cf = ClimateFiller(backend='local', lat=31.65, doy_column_name='day_number')
    cf.eto_estimation_daily_batch(str(input_dir), str(output_dir), methods_list=['hs'], datetime_format='%Y-%m-%d')

    assert doys == [100, 101]
