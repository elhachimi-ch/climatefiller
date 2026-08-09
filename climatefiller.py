from data_science_toolkit.model import Model
from data_science_toolkit.dataframe import DataFrame
import datetime
import json
import logging
import os
import time
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from lib import Lib
import ee
import geemap
import re
import requests
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, train_test_split
import geopandas as gpd
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')





class ClimateFiller():
    """The ClimateFiller class
    """
    
    def __init__(self, data_path=None, datetime_column_name='datetime', 
                 datetime_format='%Y-%m-%d %H:%M:%S', backend='gee', 
                 lat=31.65410805, lon=-7.603140831, tz_offset=-7, elevation=None,
                 artifact_folder='climatefiller_artifact', frequency='h', **kwargs):
        """
        Initializes an instance of the class with the specified parameters.

        Args:
            self (object): The instance of the class.
            data_path (str, os.PathLike, pandas.DataFrame, or None): Path to the data source or an in-memory dataframe. Defaults to None.
            datetime_column_name (str): The name of the column that contains datetime information. Defaults to 'datetime'.
            date_time_format (str): The format of the datetime values in the data source. Defaults to '%Y-%m-%d %H:%M:%S'.
            tz_offset (int): The time zone offset in hours comparint to GMT. Defaults to -7.
        Returns:
            None

        Notes:
            - The initialization of the class instance allows for handling and processing of the data.
            - The data_path parameter specifies the location of the data source, which can be a link or a local path.
            - The input data type is inferred from data_path extension (e.g., .csv, .xls, .xlsx, .json, .parquet).
            - The datetime_column_name parameter identifies the column in the data source that contains datetime information.
            - The date_time_format parameter defines the format of the datetime values in the data source.
            - If data_path is not provided, the instance will be initialized without any data source.
        """
        if data_path is None and 'data_link' in kwargs:
            # Backward compatibility for old constructor calls.
            data_path = kwargs.pop('data_link')

        self.datetime_column_name = datetime_column_name
        self.lat = lat
        self.lon = lon
        self.tz_offset = tz_offset
        self.elevation = elevation
        self.backend = backend
        self.artifact_folder = artifact_folder
        self.frequency = frequency
        self._source_crs = None
        self.check_directory_existance(self.artifact_folder)
        self._ml_impute_config = {
            'train_ratio': 1,
            'model_name': 'xgboost',
            'model_kwargs': {},
            'export_dataset': False,
        }
        self.et0_output_data = DataFrame()
        self.data = DataFrame()
        if backend == 'gee':
            gee_project = self._get_gee_project_name()
            ee.Initialize(project=gee_project)
        if data_path is None:
            self.data = DataFrame()
            
        else:
            if isinstance(data_path, pd.DataFrame):
                self._source_crs = getattr(data_path, 'crs', None)
                if self._source_crs is None and hasattr(data_path, 'attrs'):
                    self._source_crs = data_path.attrs.get('crs')
            inferred_data_type = self._infer_data_type(data_path)
            self.data = self._create_data_wrapper(data_path, inferred_data_type, **kwargs)
            self._materialize_dataset_for_column_ops()
            if datetime_column_name not in self.data.get_columns_names():
                raise ValueError(
                    f"please enter a valide datetime column name. '{datetime_column_name}' was not found. "
                    f"Available columns: {self.data.get_columns_names()}"
                )
            self.data.rename_columns({datetime_column_name:'datetime'})
            datetime_column_name = 'datetime'
            self._resolve_lon_lat_from_data()
            self._normalize_datetime_column(datetime_column_name, datetime_format)
            self.datetime_column_name = datetime_column_name
        
        self.data_reanalysis = DataFrame()

    def _get_underlying_dataframe(self):
        if hasattr(self.data, 'get_dataframe'):
            return self.data.get_dataframe()
        return self.data

    def _resolve_lon_lat_from_data(self):
        dataframe = self._get_underlying_dataframe()
        if dataframe is None:
            return

        if isinstance(self.lon, str):
            lon_value = dataframe[self.lon] if self.lon in dataframe.columns else None
            if lon_value is not None and not lon_value.empty:
                if pd.api.types.is_numeric_dtype(lon_value):
                    self.lon = float(lon_value.iloc[0])
                else:
                    resolved = pd.to_numeric(lon_value, errors='coerce')
                    if not resolved.dropna().empty:
                        self.lon = float(resolved.dropna().iloc[0])

        if isinstance(self.lat, str):
            lat_value = dataframe[self.lat] if self.lat in dataframe.columns else None
            if lat_value is not None and not lat_value.empty:
                if pd.api.types.is_numeric_dtype(lat_value):
                    self.lat = float(lat_value.iloc[0])
                else:
                    resolved = pd.to_numeric(lat_value, errors='coerce')
                    if not resolved.dropna().empty:
                        self.lat = float(resolved.dropna().iloc[0])

    def _normalize_datetime_column(self, column_name, datetime_format='%Y-%m-%d %H:%M:%S'):
        dataframe = self._get_underlying_dataframe().copy()
        if column_name not in dataframe.columns:
            raise KeyError(column_name)

        raw_values = dataframe[column_name]
        try:
            values = pd.to_datetime(raw_values, format=datetime_format)
        except (TypeError, ValueError):
            try:
                values = pd.to_datetime(raw_values, format='ISO8601')
            except (TypeError, ValueError):
                try:
                    values = pd.to_datetime(raw_values, utc=True, errors='coerce')
                except (TypeError, ValueError):
                    values = pd.to_datetime(raw_values, errors='coerce')

        dataframe[column_name] = values
        dataframe.set_index(column_name, inplace=True)
        dataframe.index = self._normalize_datetime_index(dataframe.index, preserve_timezone=True)
        dataframe = dataframe.sort_index()
        if dataframe.index.has_duplicates:
            dataframe = dataframe[~dataframe.index.duplicated(keep='last')]

        if hasattr(self.data, 'set_dataframe'):
            self.data.set_dataframe(dataframe)
        elif hasattr(self.data, 'dataframe'):
            self.data.dataframe = dataframe
        else:
            self.data = DataFrame()

        return dataframe

    def _prepare_datetime_column(self, dataframe, datetime_column_name=None):
        if datetime_column_name is None:
            datetime_column_name = self.datetime_column_name

        if dataframe is None:
            return None

        if not isinstance(dataframe, pd.DataFrame):
            dataframe = pd.DataFrame(dataframe)

        source_name = None
        for candidate in [datetime_column_name, 'datetime', 'date']:
            if candidate in dataframe.columns:
                source_name = candidate
                break

        if source_name is None:
            return dataframe

        prepared = dataframe.copy()
        prepared = prepared.rename(columns={source_name: 'datetime'})
        prepared['datetime'] = pd.to_datetime(prepared['datetime'], errors='coerce')
        prepared = prepared.set_index('datetime')
        prepared.index = self._normalize_datetime_index(prepared.index, preserve_timezone=True)
        prepared = prepared.sort_index()
        if prepared.index.has_duplicates:
            prepared = prepared[~prepared.index.duplicated(keep='last')]
        return prepared

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)

        dataframe = self._get_underlying_dataframe()
        if hasattr(dataframe, name):
            return getattr(dataframe, name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key):
        return self._get_underlying_dataframe()[key]

    def __setitem__(self, key, value):
        dataframe = self._get_underlying_dataframe()
        dataframe[key] = value
        if hasattr(self.data, 'set_dataframe'):
            self.data.set_dataframe(dataframe)

    def __len__(self):
        return len(self._get_underlying_dataframe())

    def _create_data_wrapper(self, data_path, inferred_data_type, **kwargs):
        """Create the third-party dataframe wrapper in a way that works across versions."""
        if isinstance(data_path, pd.DataFrame):
            for candidate_data_type in ('dataframe', 'df'):
                try:
                    return DataFrame(data_path=data_path, data_type=candidate_data_type, **kwargs)
                except Exception:
                    continue

            raise TypeError("Could not initialize dataframe wrapper from a pandas DataFrame.")

        return DataFrame(data_path=data_path, data_type=inferred_data_type, **kwargs)

    def _materialize_dataset_for_column_ops(self):
        """Load parquet-backed datasets into an in-memory dataframe for column operations.

        The third-party DataFrame wrapper exposes parquet datasets through pyarrow,
        but its rename and column assignment helpers do not update the dataset
        schema. Materializing the dataset once keeps later datetime normalization
        and index operations consistent.
        """
        if getattr(self.data, 'data_type', None) != 'parquet':
            return

        if not hasattr(self.data, 'dataset') or self.data.dataset is None:
            return

        try:
            pandas_df = self.data.dataset.to_table().to_pandas()
        except Exception:
            return

        try:
            self.data.set_dataframe(pandas_df, data_type='df')
        except Exception:
            if hasattr(self.data, 'dataframe'):
                self.data.dataframe = pandas_df
            else:
                self.data.__dict__['dataframe'] = pandas_df
        self.data.data_type = 'df'

    @staticmethod
    def _infer_data_type(data_path):
        if isinstance(data_path, pd.DataFrame):
            return 'df'

        if isinstance(data_path, (str, os.PathLike)):
            extension = os.path.splitext(str(data_path))[1].lower()
            extension_to_type = {
                '.csv': 'csv',
                '.xls': 'xls',
                '.xlsx': 'xlsx',
                '.json': 'json',
                '.parquet': 'parquet',
                '.geoparquet': 'parquet',
                '.pq': 'parquet',
                '.pqt': 'parquet'
            }
            inferred = extension_to_type.get(extension)
            if inferred is not None:
                return inferred

            raise ValueError(f"Unsupported data format '{extension}'.")

        raise ValueError(
            "Unsupported data_path type. Use a path string, os.PathLike, or pandas.DataFrame."
        )

    @staticmethod
    def _get_gee_project_name():
        env_candidates = [
            os.getenv('GEE_PROJECT'),
            os.getenv('EE_PROJECT'),
            os.getenv('GOOGLE_EARTH_ENGINE_PROJECT')
        ]
        for value in env_candidates:
            if value:
                return value

        env_files = [
            os.path.join(os.getcwd(), '.env'),
            os.path.join(os.path.dirname(__file__), '.env')
        ]
        keys = ('GEE_PROJECT', 'EE_PROJECT', 'GOOGLE_EARTH_ENGINE_PROJECT')
        for env_file in env_files:
            if not os.path.exists(env_file):
                continue

            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    raw = line.strip()
                    if not raw or raw.startswith('#') or '=' not in raw:
                        continue
                    key, value = raw.split('=', 1)
                    key = key.strip()
                    if key in keys:
                        parsed = value.strip().strip('"').strip("'")
                        if parsed:
                            return parsed

        raise ValueError(
            "GEE project is not configured. Please define GEE_PROJECT in .env."
        )

    @staticmethod
    def _as_timestamp(value):
        if isinstance(value, str):
            value = pd.to_datetime(value)
        if isinstance(value, pd.Timestamp):
            if getattr(value, 'tzinfo', None) is not None:
                return value.tz_convert(None)
            return value
        if isinstance(value, pd.DatetimeIndex):
            return value
        return value

    @staticmethod
    def _normalize_datetime_index(index, preserve_timezone=True):
        if index is None:
            return None

        if isinstance(index, pd.DatetimeIndex):
            if getattr(index, 'tz', None) is None:
                return index
            if not preserve_timezone:
                try:
                    return index.tz_convert(None)
                except Exception:
                    return index.tz_localize(None)
            return index

        try:
            parsed = pd.to_datetime(index, errors='coerce')
        except Exception:
            parsed = pd.Index(index)

        if isinstance(parsed, pd.DatetimeIndex):
            if getattr(parsed, 'tz', None) is None:
                return parsed
            if not preserve_timezone:
                try:
                    return parsed.tz_convert(None)
                except Exception:
                    return parsed.tz_localize(None)
            return parsed

        values = []
        found_timezone = None
        for value in list(parsed):
            try:
                parsed_value = pd.to_datetime(value, errors='coerce')
            except Exception:
                parsed_value = value

            if isinstance(parsed_value, pd.Timestamp):
                if getattr(parsed_value, 'tzinfo', None) is not None:
                    if found_timezone is None:
                        found_timezone = parsed_value.tzinfo
                    if preserve_timezone:
                        values.append(parsed_value)
                    else:
                        try:
                            values.append(parsed_value.tz_convert(None))
                        except Exception:
                            values.append(parsed_value.tz_localize(None))
                else:
                    if preserve_timezone and found_timezone is not None:
                        try:
                            values.append(parsed_value.tz_localize(found_timezone))
                        except Exception:
                            values.append(parsed_value)
                    else:
                        values.append(parsed_value)
            else:
                values.append(parsed_value)

        try:
            return pd.DatetimeIndex(values)
        except Exception:
            return pd.Index(values)

    @staticmethod
    def _normalize_datetime_value(value):
        if value is None:
            return None
        try:
            parsed = pd.to_datetime(value, utc=True, errors='coerce')
        except Exception:
            try:
                parsed = pd.to_datetime(value, errors='coerce')
            except Exception:
                parsed = pd.Timestamp(value)
        if isinstance(parsed, pd.Timestamp):
            if getattr(parsed, 'tzinfo', None) is not None:
                try:
                    return parsed.tz_convert(None)
                except Exception:
                    return parsed.tz_localize(None)
            return parsed
        if isinstance(parsed, pd.DatetimeIndex):
            return ClimateFiller._normalize_datetime_index(parsed, preserve_timezone=False)
        return parsed

    @staticmethod
    def _build_error_features(index, source_values):
        dt_index = pd.to_datetime(index)
        src = pd.Series(source_values, index=dt_index).astype(float)
        return pd.DataFrame(
            {
                'source_value': src.values,
            },
            index=dt_index,
        )

    @staticmethod
    def _resolve_model_type(model_name):
        mapping = {
            'xgboost': 'xb',
            'xb': 'xb',
            'random_forest': 'rf',
            'rf': 'rf',
            'decision_tree': 'dt',
            'dt': 'dt',
            'linear_regression': 'lr',
            'lr': 'lr',
            'knn': 'knn',
            'gradient_boosting': 'gb',
            'gb': 'gb',
            'svm': 'svm',
        }
        model_type = mapping.get(str(model_name).lower())
        if model_type is None:
            raise ValueError(
                f"Unsupported model_name '{model_name}'. "
                "Supported values are: xgboost, random_forest, decision_tree, linear_regression, knn, gradient_boosting, svm"
            )
        return model_type

    @staticmethod
    def _format_coord_for_cache(value):
        formatted = f"{float(value):.5f}"
        return formatted.replace('-', 'm').replace('.', 'p')

    @staticmethod
    def _sanitize_cache_token(value):
        return re.sub(r'[^A-Za-z0-9]+', '-', str(value)).strip('-').lower()

    @staticmethod
    def _normalize_frequency_label(freq):
        if freq is None:
            return 'unknown'

        freq_token = str(freq).strip().lower()
        if freq_token in {'h', '1h', 'hour', 'hourly'}:
            return 'hourly'
        if freq_token in {'d', '1d', 'day', 'daily'}:
            return 'daily'
        if freq_token in {'m', 'min', 'minute', 'minutely'}:
            return 'minutely'

        return ClimateFiller._sanitize_cache_token(freq_token)

    def _infer_frequency_label_from_index(self, dt_index):
        if dt_index is None or len(dt_index) < 2:
            return self._normalize_frequency_label(self.frequency)

        try:
            inferred = pd.infer_freq(pd.DatetimeIndex(dt_index))
            if inferred is not None:
                return self._normalize_frequency_label(inferred)
        except Exception:
            pass

        try:
            sorted_index = pd.DatetimeIndex(dt_index).sort_values()
            deltas = sorted_index.to_series().diff().dropna()
            if len(deltas) == 0:
                return 'unknown'
            median_delta = deltas.median()
            if pd.isna(median_delta):
                return 'unknown'

            seconds = median_delta.total_seconds()
            if seconds <= 5400:
                return 'hourly'
            if seconds <= 129600:
                return 'daily'
            return self._normalize_frequency_label(median_delta)
        except Exception:
            return self._normalize_frequency_label(self.frequency)

    def _align_source_series_to_target_frequency(self, source_series, column_to_fill_name, target_index=None):
        if source_series is None:
            return source_series

        source_series = pd.Series(source_series).copy()
        if source_series.empty:
            return source_series

        try:
            source_series.index = self._normalize_datetime_index(source_series.index, preserve_timezone=False)
        except Exception:
            return source_series

        source_series = source_series[~source_series.index.duplicated(keep='first')].sort_index()

        if target_index is None:
            target_index = self.data.get_dataframe().index

        try:
            target_index = self._normalize_datetime_index(target_index, preserve_timezone=False).sort_values()
        except Exception:
            return source_series

        if len(target_index) == 0:
            return source_series

        target_frequency_label = self._infer_frequency_label_from_index(target_index)
        source_frequency_label = self._infer_frequency_label_from_index(source_series.index)

        configured_target_label = self._normalize_frequency_label(self.frequency)
        if target_frequency_label in {'unknown'} and configured_target_label in {'hourly', 'daily', 'minutely'}:
            target_frequency_label = configured_target_label

        if target_frequency_label == 'daily' and source_frequency_label in {'hourly', 'unknown'}:
            variable_context = self._resolve_imputation_variable_context(column_to_fill_name)
            canonical = variable_context['canonical']
            requested_aggregation = variable_context['aggregation']
            radiation_like = canonical in {'rs'}
            precipitation_like = canonical in {'p'}
            if radiation_like:
                daytime_series = source_series.between_time('09:00', '18:00')
                resampled = daytime_series.groupby(daytime_series.index.floor('D')).mean()
            elif precipitation_like:
                resampled = source_series.resample('D').agg('sum')
            else:
                aggregation = requested_aggregation if requested_aggregation in {'max', 'min', 'mean'} else 'mean'
                resampled = source_series.resample('D').agg(aggregation)
            if len(resampled.index) == 0:
                return source_series
            return resampled.reindex(target_index)

        if target_frequency_label == 'hourly' and source_frequency_label == 'daily':
            variable_context = self._resolve_imputation_variable_context(column_to_fill_name)
            canonical = variable_context['canonical']
            requested_aggregation = variable_context['aggregation']
            radiation_like = canonical in {'rs'}
            precipitation_like = canonical in {'p'}
            if radiation_like:
                daytime_series = source_series.between_time('09:00', '18:00')
                resampled = daytime_series.groupby(daytime_series.index.floor('D')).mean()
            elif precipitation_like:
                resampled = source_series.resample('H').agg('sum')
            else:
                aggregation = requested_aggregation if requested_aggregation in {'max', 'min', 'mean'} else 'mean'
                resampled = source_series.resample('H').agg(aggregation)
            if len(resampled.index) == 0:
                return source_series
            return resampled.reindex(target_index)

        if target_frequency_label == 'hourly' and source_frequency_label in {'hourly', 'unknown'}:
            variable_context = self._resolve_imputation_variable_context(column_to_fill_name)
            canonical = variable_context['canonical']
            requested_aggregation = variable_context['aggregation']
            radiation_like = canonical in {'rs'}
            precipitation_like = canonical in {'p'}
            if radiation_like:
                daytime_series = source_series.between_time('09:00', '18:00')
                resampled = daytime_series.resample('h').mean()
            elif precipitation_like:
                resampled = source_series.resample('h').agg('sum')
            else:
                aggregation = requested_aggregation if requested_aggregation in {'max', 'min', 'mean'} else 'mean'
                resampled = source_series.resample('h').agg(aggregation)
            if len(resampled.index) == 0:
                return source_series
            return resampled.reindex(target_index)

        if not source_series.index.equals(target_index):
            try:
                return source_series.reindex(target_index)
            except Exception:
                return source_series

        return source_series

    def _resolve_imputation_variable_context(self, column_name):
        normalized = str(column_name).strip().lower()
        if normalized in {'ta', 't2m', 'temperature_2m'}:
            return {'canonical': 'ta', 'aggregation': 'mean', 'source_var': 'ta'}

        lookup = {
            't2m_max': {'canonical': 'ta', 'aggregation': 'max', 'source_var': 'ta'},
            't2m_min': {'canonical': 'ta', 'aggregation': 'min', 'source_var': 'ta'},
            't2m_mean': {'canonical': 'ta', 'aggregation': 'mean', 'source_var': 'ta'},
            'ta_max': {'canonical': 'ta', 'aggregation': 'max', 'source_var': 'ta'},
            'ta_min': {'canonical': 'ta', 'aggregation': 'min', 'source_var': 'ta'},
            'ta_mean': {'canonical': 'ta', 'aggregation': 'mean', 'source_var': 'ta'},
            'temperature_2m_max': {'canonical': 'ta', 'aggregation': 'max', 'source_var': 'ta'},
            'temperature_2m_min': {'canonical': 'ta', 'aggregation': 'min', 'source_var': 'ta'},
            'temperature_2m_mean': {'canonical': 'ta', 'aggregation': 'mean', 'source_var': 'ta'},
            'rh_max': {'canonical': 'rh', 'aggregation': 'max', 'source_var': 'rh'},
            'rh_min': {'canonical': 'rh', 'aggregation': 'min', 'source_var': 'rh'},
            'rh_mean': {'canonical': 'rh', 'aggregation': 'mean', 'source_var': 'rh'},
            'relative_humidity_max': {'canonical': 'rh', 'aggregation': 'max', 'source_var': 'rh'},
            'relative_humidity_min': {'canonical': 'rh', 'aggregation': 'min', 'source_var': 'rh'},
            'relative_humidity_mean': {'canonical': 'rh', 'aggregation': 'mean', 'source_var': 'rh'},
            'ws_max': {'canonical': 'ws', 'aggregation': 'max', 'source_var': 'ws'},
            'ws_min': {'canonical': 'ws', 'aggregation': 'min', 'source_var': 'ws'},
            'ws_mean': {'canonical': 'ws', 'aggregation': 'mean', 'source_var': 'ws'},
            'wind_speed_max': {'canonical': 'ws', 'aggregation': 'max', 'source_var': 'ws'},
            'wind_speed_min': {'canonical': 'ws', 'aggregation': 'min', 'source_var': 'ws'},
            'wind_speed_mean': {'canonical': 'ws', 'aggregation': 'mean', 'source_var': 'ws'},
            'rs_max': {'canonical': 'rs', 'aggregation': 'max', 'source_var': 'rs'},
            'rs_min': {'canonical': 'rs', 'aggregation': 'min', 'source_var': 'rs'},
            'rs_mean': {'canonical': 'rs', 'aggregation': 'mean', 'source_var': 'rs'},
            'ssrd_max': {'canonical': 'rs', 'aggregation': 'max', 'source_var': 'rs'},
            'ssrd_min': {'canonical': 'rs', 'aggregation': 'min', 'source_var': 'rs'},
            'ssrd_mean': {'canonical': 'rs', 'aggregation': 'mean', 'source_var': 'rs'},
            'p_max': {'canonical': 'p', 'aggregation': 'max', 'source_var': 'p'},
            'p_min': {'canonical': 'p', 'aggregation': 'min', 'source_var': 'p'},
            'p_mean': {'canonical': 'p', 'aggregation': 'mean', 'source_var': 'p'},
            'tp_max': {'canonical': 'p', 'aggregation': 'max', 'source_var': 'p'},
            'tp_min': {'canonical': 'p', 'aggregation': 'min', 'source_var': 'p'},
            'tp_mean': {'canonical': 'p', 'aggregation': 'mean', 'source_var': 'p'},
            'precipitation_max': {'canonical': 'p', 'aggregation': 'max', 'source_var': 'p'},
            'precipitation_min': {'canonical': 'p', 'aggregation': 'min', 'source_var': 'p'},
            'precipitation_mean': {'canonical': 'p', 'aggregation': 'mean', 'source_var': 'p'},
            'total_precipitation_max': {'canonical': 'p', 'aggregation': 'max', 'source_var': 'p'},
            'total_precipitation_min': {'canonical': 'p', 'aggregation': 'min', 'source_var': 'p'},
            'total_precipitation_mean': {'canonical': 'p', 'aggregation': 'mean', 'source_var': 'p'},
        }

        if normalized in lookup:
            return lookup[normalized]

        base_name = normalized
        for suffix in ('_max', '_min', '_mean'):
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break

        if base_name.startswith('ta_') or base_name.startswith('t2m_'):
            return {'canonical': 'ta', 'aggregation': 'mean', 'source_var': 'ta'}
        if base_name.startswith('rh_') or base_name.startswith('relative_humidity_'):
            return {'canonical': 'rh', 'aggregation': 'mean', 'source_var': 'rh'}
        if base_name.startswith('ws_') or base_name.startswith('wind_speed_'):
            return {'canonical': 'ws', 'aggregation': 'mean', 'source_var': 'ws'}
        if base_name.startswith('rs_') or base_name.startswith('ssrd_'):
            return {'canonical': 'rs', 'aggregation': 'mean', 'source_var': 'rs'}
        if base_name.startswith('p_') or base_name.startswith('tp_') or base_name.startswith('precipitation_') or base_name.startswith('total_precipitation_'):
            return {'canonical': 'p', 'aggregation': 'mean', 'source_var': 'p'}

        canonical_column_to_fill_name = {
            't2m': 'ta',
            'temperature_2m': 'ta',
            'tp': 'p',
            'precipitation': 'p',
        }.get(normalized, normalized)
        return {'canonical': canonical_column_to_fill_name, 'aggregation': 'mean', 'source_var': canonical_column_to_fill_name}

    def _build_era5_year_cache_path(self, variables, lon, lat, year, frequency=None):
        lon_key = self._format_coord_for_cache(lon)
        lat_key = self._format_coord_for_cache(lat)
        freq_label = self._normalize_frequency_label(frequency)
        vars_key = '_'.join([str(s) for s in variables])
        filename = f"era5_land_freq_{freq_label}_{vars_key}_{lon_key}_{lat_key}_{year}.csv"
        return os.path.join('data', 'cache', filename)

    def _resolve_era5_year_cache_path(self, variables, lon, lat, year, frequency=None):
        freq_path = self._build_era5_year_cache_path(variables, lon, lat, year, frequency)
        if os.path.exists(freq_path):
            return freq_path

        legacy_path = os.path.join(
            'data',
            'cache',
            'era5_land_' + '_'.join([str(s) for s in variables] + [str(lon), str(lat), str(year)]) + '.csv',
        )
        return legacy_path

    def _build_source_cache_path(self, product, variable, lon, lat, start_datetime, end_datetime, frequency=None):
        start_ts = pd.to_datetime(start_datetime).strftime('%Y%m%d%H%M%S')
        end_ts = pd.to_datetime(end_datetime).strftime('%Y%m%d%H%M%S')
        lon_key = self._format_coord_for_cache(lon)
        lat_key = self._format_coord_for_cache(lat)
        freq_label = self._normalize_frequency_label(frequency)
        filename = (
            f"impute_source_backend_{self.backend}_product_{product}_var_{variable}_"
            f"freq_{freq_label}_lon_{lon_key}_lat_{lat_key}_start_{start_ts}_end_{end_ts}.csv"
        )
        return os.path.join('data', 'cache', filename)

    @staticmethod
    def _load_source_series_cache(cache_path):
        cached = pd.read_csv(cache_path)
        if 'datetime' in cached.columns:
            cached['datetime'] = pd.to_datetime(cached['datetime'])
            cached.set_index('datetime', inplace=True)
        else:
            first_col = cached.columns[0]
            cached[first_col] = pd.to_datetime(cached[first_col])
            cached.set_index(first_col, inplace=True)

        if 'source_value' in cached.columns:
            source_series = cached['source_value']
        else:
            source_series = cached.iloc[:, 0]

        source_series = source_series[~source_series.index.duplicated(keep='first')].sort_index()
        return source_series

    @staticmethod
    def _save_source_series_cache(source_series, cache_path):
        source_series = source_series[~source_series.index.duplicated(keep='first')].sort_index()
        cache_df = source_series.rename('source_value').to_frame()
        cache_df.index.name = 'datetime'
        cache_df.to_csv(cache_path)
        LOGGER.info("Source cache saved: %s (%d rows)", cache_path, len(cache_df))
        print(f"Saved source cache to {cache_path}")

    @staticmethod
    def _to_json_safe(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: ClimateFiller._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ClimateFiller._to_json_safe(v) for v in value]
        return value

    def _save_error_model_artifact(self, model, feature_columns, column_to_fill_name, product, model_name, performance):
        model_artifact_name = f"{product}_{column_to_fill_name}_{model_name}_error_model.data"
        model_artifact_path = os.path.join(self.artifact_folder, model_artifact_name)
        model.save_model(model_artifact_path)

        metadata_name = f"{product}_{column_to_fill_name}_{model_name}_error_model_meta.json"
        metadata_path = os.path.join(self.artifact_folder, metadata_name)
        metadata = {
            'feature_columns': list(feature_columns),
            'column_to_fill_name': column_to_fill_name,
            'product': product,
            'model_name': model_name,
            'trained_at': datetime.datetime.utcnow().isoformat(),
            'performance': self._to_json_safe(performance),
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        LOGGER.info("Model artifact saved: %s", model_artifact_path)
        LOGGER.info("Model metadata saved: %s", metadata_path)
        print(f"Saved ML artifact to {model_artifact_path}")
        print(f"Saved ML metadata to {metadata_path}")

    def _save_training_dataset_artifact(self, train_dataset, column_to_fill_name, product, model_name):
        dataset_name = f"{product}_{column_to_fill_name}_{model_name}_training_dataset.csv"
        dataset_path = os.path.join(self.artifact_folder, dataset_name)
        export_df = train_dataset.copy()
        export_df.index.name = 'datetime'
        export_df.to_csv(dataset_path)
        LOGGER.info("Training dataset saved: %s (%d rows)", dataset_path, len(export_df))
        print(f"Saved training dataset to {dataset_path}")

    @staticmethod
    def _model_regression_report(model, y_true, y_pred):
        y_true_series = pd.Series(np.asarray(y_true).reshape(-1), name='y_test')
        y_pred_array = np.asarray(y_pred).reshape(-1)
        try:
            return model.regression_report(y_true_series, y_pred_array)
        except Exception as e:
            LOGGER.warning("Model.regression_report failed (%s). Falling back to direct metric computation.", e)
            # Keep imputation running if the toolkit report fails due internal plotting/dataframe constraints.
            return {
                'R2': float(r2_score(y_true_series, y_pred_array)),
                'R': float(np.corrcoef(y_true_series.to_numpy(), y_pred_array)[0][1]),
                'MSE': float(mean_squared_error(y_true_series, y_pred_array)),
                'RMSE': float(np.sqrt(mean_squared_error(y_true_series, y_pred_array))),
                'MAE': float(mean_absolute_error(y_true_series, y_pred_array)),
                'MEDAE': float(median_absolute_error(y_true_series, y_pred_array)),
            }

    def _train_error_model(self, column_to_fill_name, source_series, product, train_ratio=1, model_name='xgboost', export_dataset=False, **model_kwargs):
        LOGGER.info(
            "Training error model: variable=%s product=%s model=%s train_ratio=%s export_dataset=%s",
            column_to_fill_name,
            product,
            model_name,
            train_ratio,
            export_dataset,
        )
        source_series = source_series[~source_series.index.duplicated(keep='first')].sort_index()
        insitu_series = self.data.get_dataframe()[column_to_fill_name]
        train_df = pd.concat(
            [
                insitu_series.rename('insitu_value'),
                source_series.rename('source_value')
            ],
            axis=1
        ).dropna()

        if train_df.shape[0] < 24:
            print(
                f"Insufficient overlap ({train_df.shape[0]} samples) to train ML error model for {column_to_fill_name}. "
                "Using raw downloaded data for gap filling."
            )
            return None, None, None

        X = self._build_error_features(train_df.index, train_df['source_value'])
        y = train_df['insitu_value']
        target_column_name = 'insitu_value'
        LOGGER.info("Training feature columns: %s", list(X.columns))
        LOGGER.info("Training target column: %s", target_column_name)
        print(f"Features used for training: {list(X.columns)}")
        print(f"Target used for training: {target_column_name}")
        model_type = self._resolve_model_type(model_name)
        train_ratio = float(train_ratio)
        if train_ratio <= 0 or train_ratio > 1:
            raise ValueError("train_ratio must be in (0, 1].")

        start = time.perf_counter()
        if train_ratio < 1:
            x_train, x_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=train_ratio,
                test_size=1 - train_ratio,
                random_state=42,
            )
        else:
            x_train, y_train = X, y
            x_test, y_test = None, None

        if export_dataset:
            train_dataset = x_train.copy()
            train_dataset[target_column_name] = y_train
            self._save_training_dataset_artifact(
                train_dataset,
                column_to_fill_name,
                product,
                model_name,
            )

        try:
            model = Model(
                data_x=x_train,
                data_y=y_train,
                model_type=model_type,
                task='r',
                training_percent=1,
                **model_kwargs,
            )
        except TypeError as e:
            raise ValueError(f"Invalid kwargs for Model: {e}") from e

        model.train()
        training_time_sec = time.perf_counter() - start

        performance = {
            'training_time_sec': training_time_sec,
            'train_ratio': train_ratio,
            'regression_report': None,
            'n_samples': int(len(X)),
            'n_train': int(len(x_train)),
            'n_test': int(len(x_test)) if x_test is not None else 0,
        }

        x_eval, y_eval = (x_test, y_test) if x_test is not None and len(x_test) > 0 else (x_train, y_train)
        y_eval_pred = model.predict(x_eval)
        y_eval_pred = np.asarray(y_eval_pred).reshape(-1)
        y_eval = np.asarray(y_eval).reshape(-1)
        report = self._model_regression_report(model, y_eval, y_eval_pred)
        performance['regression_report'] = report

        print(f"Regression report ({'test' if x_test is not None and len(x_test) > 0 else 'train'}): {report}")
        LOGGER.info(
            "Regression report (%s): %s",
            'test' if x_test is not None and len(x_test) > 0 else 'train',
            report,
        )

        LOGGER.info(
            "Model metrics: training_time_sec=%.4f n_train=%d n_test=%d",
            performance['training_time_sec'],
            performance['n_train'],
            performance['n_test'],
        )

        self._save_error_model_artifact(
            model,
            X.columns,
            column_to_fill_name,
            product,
            model_name,
            performance,
        )
        return model, list(X.columns), performance

    def _fill_from_source_series(self, column_to_fill_name, source_series, product, machine_learning_enabled=False):
        source_series = self._align_source_series_to_target_frequency(
            source_series,
            column_to_fill_name,
            self.data.get_dataframe().index,
        )
        source_series = source_series[~source_series.index.duplicated(keep='first')].sort_index()
        missing_indexes = self.data.get_missing_data_indexes_in_column(column_to_fill_name)

        if len(missing_indexes) == 0:
            return

        model = None
        feature_columns = None
        if machine_learning_enabled:
            model, feature_columns, performance = self._train_error_model(
                column_to_fill_name,
                source_series,
                product,
                train_ratio=self._ml_impute_config.get('train_ratio', 1),
                model_name=self._ml_impute_config.get('model_name', 'xgboost'),
                export_dataset=self._ml_impute_config.get('export_dataset', False),
                **self._ml_impute_config.get('model_kwargs', {}),
            )
            if performance is not None:
                print(
                    f"ML performance for {column_to_fill_name}: "
                    f"training_time_sec={performance['training_time_sec']:.3f}, "
                    f"regression_report={performance['regression_report']}"
                )

        missing_iter = tqdm(missing_indexes, desc=f"Impute {column_to_fill_name}", unit="row")
        df = self._get_underlying_dataframe()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df.index = self._normalize_datetime_index(df.index)
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep='last')].sort_index()
        if column_to_fill_name not in df.columns:
            df[column_to_fill_name] = np.nan

        for p in missing_iter:
            p_ts = self._as_timestamp(p)
            normalized_source_index = self._normalize_datetime_index(source_series.index, preserve_timezone=False)
            if p_ts not in normalized_source_index:
                continue

            source_value = source_series.loc[p_ts]
            if isinstance(source_value, pd.Series):
                source_value = source_value.iloc[0]

            filled_value = float(source_value)
            if model is not None:
                features = self._build_error_features([p_ts], [source_value])
                features = features[feature_columns]
                predicted_insitu_value = float(model.predict(features)[0])
                filled_value = predicted_insitu_value

            try:
                self.data.set_row(column_to_fill_name, p, filled_value)
            except Exception:
                pass

            assignment_done = False
            for candidate in [p_ts, p]:
                if candidate is None:
                    continue
                try:
                    if isinstance(candidate, str):
                        continue
                    if candidate in df.index:
                        df.loc[df.index == candidate, column_to_fill_name] = filled_value
                        assignment_done = True
                        break
                except Exception:
                    continue

            if not assignment_done:
                try:
                    df[column_to_fill_name] = df[column_to_fill_name].astype(float)
                    normalized_index = self._normalize_datetime_index(df.index, preserve_timezone=False)
                    normalized_candidate = self._normalize_datetime_index(pd.Index([p_ts]), preserve_timezone=False)[0]
                    mask = normalized_index == normalized_candidate
                    if mask.any():
                        df.loc[mask, column_to_fill_name] = filled_value
                        assignment_done = True
                except Exception:
                    pass

            if not assignment_done:
                try:
                    df.at[p_ts, column_to_fill_name] = filled_value
                    assignment_done = True
                except Exception:
                    pass

            if not assignment_done:
                try:
                    df.at[p, column_to_fill_name] = filled_value
                except Exception:
                    pass

            if hasattr(self.data, 'set_dataframe'):
                self.data.set_dataframe(df)
            elif hasattr(self.data, 'dataframe'):
                self.data.dataframe = df
            else:
                self.data = DataFrame(df)

    def _ensure_impute_target_column(self, column_name):
        df = self.data.get_dataframe().copy()
        if column_name not in df.columns:
            df[column_name] = np.nan
            self.data.set_dataframe(df)
            LOGGER.info("Created missing target column for imputation: %s", column_name)
        return column_name

    @staticmethod
    def _extract_remote_series(dataframe, preferred_columns=None):
        preferred_columns = preferred_columns or []
        for col in preferred_columns:
            if col in dataframe.columns:
                return dataframe[col]

        ignored = {'datetime', 'valid_time'}
        for col in dataframe.columns:
            if str(col).lower() in ignored:
                continue
            return dataframe[col]

        raise ValueError(
            f"Unable to find a remote value column. Available columns: {list(dataframe.columns)}"
        )
        
    def show(self, number_of_row=None):
        """
        Displays a specified number of rows from the data source.

        Args:
            self (object): The instance of the class.
            number_of_row (int or None, optional): The number of rows to display. Defaults to None.

        Returns:
            None

        Notes:
            - The show method is used to visualize a specified number of rows from the data source.
            - If the number_of_row parameter is not provided, all available rows will be displayed.
            - The displayed rows provide a preview or snapshot of the data in the data source.
        """
        
        if number_of_row is None:
            return print(self.data.get_dataframe())
        elif number_of_row < 0:
            return print(self.data.get_dataframe().tail(abs(number_of_row)))
        else:
            return print(self.data.get_dataframe().head(number_of_row))

    def resample(self, column_agg_map, frequency='D', keep_columns=None):
        """
        Resample in-situ data with per-column aggregation specs.

        Args:
            column_agg_map (dict): Mapping of column name to aggregation spec.
                Example: {'ta': ['max', 'min', 'mean'], 'p': 'sum'}
            frequency (str): Pandas frequency string, e.g., 'H', 'D', 'M'.
            keep_columns (list/tuple/set or None): Optional output columns filter.
                Supports exact names (e.g., 'ta_max') and base names (e.g., 'ta').

        Returns:
            pandas.DataFrame: Resampled dataframe.
        """
        df = self.data.get_dataframe().copy()
        resampled_df = self._resample_dataframe(
            df,
            column_agg_map=column_agg_map,
            frequency=frequency,
            keep_columns=keep_columns,
            datetime_candidates=[self.datetime_column_name, 'datetime', 'date'],
        )

        self.data.set_dataframe(resampled_df)
        LOGGER.info(
            "In-situ data resampled: frequency=%s columns=%s",
            frequency,
            list(resampled_df.columns),
        )
        return self.data.get_dataframe()

    @staticmethod
    def _load_dataframe_from_path(file_path):
        data_type = ClimateFiller._infer_data_type(file_path)
        if data_type == 'csv':
            return pd.read_csv(file_path)
        if data_type == 'xls' or data_type == 'xlsx':
            return pd.read_excel(file_path)
        if data_type == 'json':
            return pd.read_json(file_path)
        if data_type == 'parquet':
            return pd.read_parquet(file_path)
        raise ValueError(f"Unsupported file type for batch resample: {file_path}")

    @staticmethod
    def _save_dataframe_to_path(df, output_path):
        data_type = ClimateFiller._infer_data_type(output_path)
        if data_type == 'csv':
            df.to_csv(output_path)
            return
        if data_type == 'xls' or data_type == 'xlsx':
            df.to_excel(output_path)
            return
        if data_type == 'json':
            df.to_json(output_path, orient='records', date_format='iso')
            return
        if data_type == 'parquet':
            df.to_parquet(output_path)
            return
        raise ValueError(f"Unsupported output file type for batch resample: {output_path}")

    @staticmethod
    def _resample_dataframe(df, column_agg_map, frequency='D', keep_columns=None, datetime_candidates=None):
        if not isinstance(column_agg_map, dict) or len(column_agg_map) == 0:
            raise ValueError("column_agg_map must be a non-empty dict.")

        if datetime_candidates is None:
            datetime_candidates = ['datetime']

        data = df.copy()
        if not isinstance(data.index, pd.DatetimeIndex):
            datetime_col = None
            for candidate in datetime_candidates:
                if candidate in data.columns:
                    datetime_col = candidate
                    break

            if datetime_col is None:
                raise ValueError(
                    "Data must have a DatetimeIndex or a valid datetime column to resample."
                )

            data[datetime_col] = pd.to_datetime(data[datetime_col])
            data = data.set_index(datetime_col)

        normalized = {}
        missing_cols = []
        for column_name, agg_spec in column_agg_map.items():
            if column_name not in data.columns:
                missing_cols.append(column_name)
                continue

            if isinstance(agg_spec, (list, tuple, set)):
                normalized[column_name] = list(agg_spec)
            else:
                normalized[column_name] = [agg_spec]

        if missing_cols:
            raise ValueError(f"Columns not found for resampling: {missing_cols}")

        resampled_df = data.resample(frequency).agg(normalized)

        if isinstance(resampled_df.columns, pd.MultiIndex):
            flat_columns = []
            for column_name, agg_name in resampled_df.columns:
                if callable(agg_name):
                    agg_label = getattr(agg_name, '__name__', 'agg')
                else:
                    agg_label = str(agg_name)
                flat_columns.append(f"{column_name}_{agg_label}")
            resampled_df.columns = flat_columns

        if keep_columns is not None:
            if not isinstance(keep_columns, (list, tuple, set)):
                raise ValueError("keep_columns must be a list/tuple/set of column names.")

            keep_set = set(keep_columns)
            selected_columns = [
                column_name
                for column_name in resampled_df.columns
                if column_name in keep_set or column_name.split('_')[0] in keep_set
            ]

            for keep_col in keep_set:
                if keep_col in data.columns and keep_col not in selected_columns:
                    selected_columns.append(keep_col)

            if len(selected_columns) == 0:
                raise ValueError(
                    f"No columns matched keep_columns={list(keep_columns)}. "
                    f"Available columns: {list(resampled_df.columns)}"
                )

            static_cols = [col for col in selected_columns if col in data.columns]
            if len(static_cols) > 0:
                static_resampled = data[static_cols].resample(frequency).first()
                resampled_df = resampled_df[[c for c in selected_columns if c in resampled_df.columns]]
                for static_col in static_cols:
                    resampled_df[static_col] = static_resampled[static_col]

            final_columns = [c for c in selected_columns if c in resampled_df.columns]
            resampled_df = resampled_df[final_columns]

        return resampled_df

    def resample_batch(self, input_folder, output_folder, column_agg_map, frequency='D', keep_columns=None, prefix=None):
        """
        Batch resample in-situ files from input_folder and save to output_folder.

        Args:
            input_folder (str): Folder containing in-situ files.
            output_folder (str): Destination folder for resampled files.
            column_agg_map (dict): Same as resample().
            frequency (str): Same as resample().
            keep_columns (list/tuple/set or None): Same as resample().
            prefix (str or None): If provided, process only files that start with prefix.

        Returns:
            list: Output file paths generated.
        """
        if not os.path.isdir(input_folder):
            raise ValueError(f"input_folder does not exist: {input_folder}")

        self.check_directory_existance(output_folder)

        supported_exts = {'.csv', '.xls', '.xlsx', '.json', '.parquet', '.geoparquet', '.pq', '.pqt'}
        files = []
        for name in os.listdir(input_folder):
            path = os.path.join(input_folder, name)
            if not os.path.isfile(path):
                continue
            if prefix is not None and not name.startswith(prefix):
                continue
            if os.path.splitext(name)[1].lower() not in supported_exts:
                continue
            files.append(name)

        files = sorted(files)

        if len(files) == 0:
            LOGGER.warning("No input files found for resample_batch in %s with prefix=%s", input_folder, prefix)
            return []

        LOGGER.info(
            "Starting batch resample: %d file(s), frequency=%s, prefix=%s",
            len(files),
            frequency,
            prefix,
        )

        output_paths = []
        file_iter = tqdm(files, total=len(files), desc="Batch resample", unit="file")
        for filename in file_iter:
            file_iter.set_postfix_str(filename)
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            df = self._load_dataframe_from_path(input_path)
            resampled_df = self._resample_dataframe(
                df,
                column_agg_map=column_agg_map,
                frequency=frequency,
                keep_columns=keep_columns,
                datetime_candidates=[self.datetime_column_name, 'datetime', 'date'],
            )
            self._save_dataframe_to_path(resampled_df, output_path)
            output_paths.append(output_path)
            LOGGER.info("Resampled file saved: %s", output_path)

        LOGGER.info("Batch resample completed: %d output file(s)", len(output_paths))

        return output_paths

    def impute_batch(self, input_folder, output_folder, column_to_fill_list='ta', product='era5_land', machine_learning_enabled=False, train_ratio=1, model_name='xgboost', export_dataset=False, prefix=None, datetime_format='%Y-%m-%d %H:%M:%S', **kwargs):
        """
        Batch impute files from input_folder and save the imputed results to output_folder.

        Args:
            input_folder (str): Folder containing input files to impute.
            output_folder (str): Destination folder for imputed files.
            column_to_fill_name (str): Column to impute.
            product (str): Source product name passed to impute().
            machine_learning_enabled (bool): Whether to use the ML error model.
            train_ratio (float): Train ratio for the ML error model.
            model_name (str): Model name for the ML error model.
            export_dataset (bool): Whether to export the training dataset artifact.
            prefix (str or None): If provided, process only files that start with prefix.
            datetime_format (str): Datetime parsing format used when initializing per-file instances.

        Returns:
            list: Output file paths generated.
        """
        if not os.path.isdir(input_folder):
            raise ValueError(f"input_folder does not exist: {input_folder}")

        self.check_directory_existance(output_folder)

        supported_exts = {'.csv', '.xls', '.xlsx', '.json', '.parquet', '.geoparquet', '.pq', '.pqt'}
        files = []
        for name in os.listdir(input_folder):
            path = os.path.join(input_folder, name)
            if not os.path.isfile(path):
                continue
            if prefix is not None and not name.startswith(prefix):
                continue
            if os.path.splitext(name)[1].lower() not in supported_exts:
                continue
            files.append(name)

        files = sorted(files)

        if len(files) == 0:
            LOGGER.warning("No input files found for impute_batch in %s with prefix=%s", input_folder, prefix)
            return []

        LOGGER.info(
            "Starting batch impute: %d file(s), variable=%s, prefix=%s",
            len(files),
            column_to_fill_list,
            prefix,
        )

        output_paths = []
        file_iter = tqdm(files, total=len(files), desc="Batch impute", unit="file")
        for filename in file_iter:
            file_iter.set_postfix_str(filename)
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            input_df = self._load_dataframe_from_path(input_path)
            datetime_column_candidates = [self.datetime_column_name, 'datetime', 'date', 'time', 'timestamp']
            detected_datetime_column = None
            for candidate in datetime_column_candidates:
                if candidate in input_df.columns:
                    detected_datetime_column = candidate
                    break
            if detected_datetime_column is None:
                detected_datetime_column = next((col for col in input_df.columns if pd.api.types.is_datetime64_any_dtype(input_df[col])), None)
            if detected_datetime_column is None:
                raise ValueError(
                    f"Could not infer a datetime column for batch imputation from file: {input_path}"
                )

            imputer = self.__class__(
                data_path=input_df,
                datetime_column_name=detected_datetime_column,
                datetime_format=datetime_format,
                backend=self.backend,
                lat=self.lat,
                lon=self.lon,
                tz_offset=self.tz_offset,
                elevation=self.elevation,
                artifact_folder=self.artifact_folder,
            )
            imputer.impute(
                column_to_fill_list=column_to_fill_list,
                product=product,
                machine_learning_enabled=machine_learning_enabled,
                train_ratio=train_ratio,
                model_name=model_name,
                export_dataset=export_dataset,
                **kwargs,
            )
            self._save_dataframe_to_path(imputer.data.get_dataframe().copy(), output_path)
            output_paths.append(output_path)
            LOGGER.info("Imputed file saved: %s", output_path)

        LOGGER.info("Batch impute completed: %d output file(s)", len(output_paths))
        return output_paths

    def to_geo_dataframe(self, output_path, lon_column=None, lat_column=None, crs=None):
        """
        Export current in-situ data as a GeoDataFrame file.

        File type is inferred from output_path extension.

        Args:
            output_path (str): Destination path, e.g. .geoparquet, .parquet, .geojson, .gpkg, .shp.
            lon_column (str or None): Longitude column name. If None, inferred.
            lat_column (str or None): Latitude column name. If None, inferred.
            crs (str or None): Coordinate reference system. Defaults to the source CRS when available, otherwise EPSG:4326.

        Returns:
            geopandas.GeoDataFrame: Exported GeoDataFrame.
        """
        df = self.data.get_dataframe().copy()
        gdf = self._build_geodataframe_from_dataframe(
            df,
            lon_column=lon_column,
            lat_column=lat_column,
            crs=crs,
        )

        output_dir = os.path.dirname(output_path)
        if output_dir:
            self.check_directory_existance(output_dir)

        extension = os.path.splitext(output_path)[1].lower()
        if extension in ('.parquet', '.geoparquet', '.pq', '.pqt'):
            gdf.to_parquet(output_path, index=False)
        elif extension in ('.geojson', '.json'):
            gdf.to_file(output_path, driver='GeoJSON')
        elif extension == '.gpkg':
            gdf.to_file(output_path, driver='GPKG')
        elif extension == '.shp':
            gdf.to_file(output_path, driver='ESRI Shapefile')
        else:
            raise ValueError(
                f"Unsupported geospatial output format '{extension}'. "
                "Supported formats: .geoparquet, .parquet, .geojson, .json, .gpkg, .shp"
            )

        LOGGER.info("GeoDataFrame exported: %s (%d rows)", output_path, len(gdf))
        print(f"GeoDataFrame exported to {output_path}")
        return gdf

    def _build_geodataframe_from_dataframe(self, df, lon_column=None, lat_column=None, crs=None):
        columns = list(df.columns)
        lower_to_original = {str(col).lower(): col for col in columns}

        if crs is None:
            source_crs = getattr(df, 'crs', None)
            if source_crs is None and hasattr(df, 'attrs'):
                source_crs = df.attrs.get('crs')
            if source_crs is None and hasattr(df, 'geometry'):
                source_crs = getattr(df.geometry, 'crs', None)
            if source_crs is None:
                source_crs = getattr(self, '_source_crs', None)
            crs = source_crs or 'EPSG:4326'

        # If class-level lon/lat are provided as column names, prioritize them.
        if lon_column is None and isinstance(self.lon, str):
            lon_key = self.lon.lower()
            if lon_key in lower_to_original:
                lon_column = lower_to_original[lon_key]

        if lat_column is None and isinstance(self.lat, str):
            lat_key = self.lat.lower()
            if lat_key in lower_to_original:
                lat_column = lower_to_original[lat_key]

        if lon_column is None:
            for candidate in ('lon', 'longitude', 'x'):
                if candidate in lower_to_original:
                    lon_column = lower_to_original[candidate]
                    break

        if lat_column is None:
            for candidate in ('lat', 'latitude', 'y'):
                if candidate in lower_to_original:
                    lat_column = lower_to_original[candidate]
                    break

        if lon_column is None or lat_column is None:
            if not isinstance(self.lon, (int, float, np.number)) or not isinstance(self.lat, (int, float, np.number)):
                raise ValueError(
                    "Longitude/latitude columns were not found and class-level lon/lat are not numeric fallback coordinates."
                )
            lon_column = lon_column or 'lon'
            lat_column = lat_column or 'lat'
            df[lon_column] = float(self.lon)
            df[lat_column] = float(self.lat)

        df[lon_column] = pd.to_numeric(df[lon_column], errors='coerce')
        df[lat_column] = pd.to_numeric(df[lat_column], errors='coerce')
        df = df.dropna(subset=[lon_column, lat_column]).copy()

        geometry = gpd.points_from_xy(df[lon_column], df[lat_column], crs=crs)
        if geometry is None:
            try:
                from shapely.geometry import Point
            except Exception:
                geometry = None
            else:
                geometry = [Point(float(x), float(y)) for x, y in zip(df[lon_column], df[lat_column])]

        if geometry is None:
            return gpd.GeoDataFrame(df, crs=crs)

        return gpd.GeoDataFrame(
            df,
            geometry=geometry,
            crs=crs,
        )

    def to_geo_dataframe_batch(self, input_folder, output_folder, prefix=None, lon_column=None, lat_column=None, crs='EPSG:4326', file_type='parquet'):
        """
        Convert all supported files in input_folder to GeoDataFrames and export them.

        Args:
            input_folder (str): Folder containing input tabular files.
            output_folder (str): Destination folder for geospatial outputs.
            prefix (str or None): If provided, process only files that start with prefix.
            lon_column (str or None): Optional explicit longitude column name.
            lat_column (str or None): Optional explicit latitude column name.
            crs (str): Coordinate reference system. Defaults to EPSG:4326.
            file_type (str): Export format for all output files.
                Supported: geoparquet, parquet, geojson, json, gpkg, shp. Defaults to parquet.

        Returns:
            list: Output file paths generated.
        """
        if not os.path.isdir(input_folder):
            raise ValueError(f"input_folder does not exist: {input_folder}")

        self.check_directory_existance(output_folder)

        supported_exts = {'.csv', '.xls', '.xlsx', '.json', '.parquet', '.geoparquet', '.pq', '.pqt', '.geojson', '.gpkg', '.shp'}
        file_type_to_ext = {
            'geoparquet': '.geoparquet',
            '.geoparquet': '.geoparquet',
            'parquet': '.parquet',
            '.parquet': '.parquet',
            'pq': '.parquet',
            '.pq': '.parquet',
            'pqt': '.parquet',
            '.pqt': '.parquet',
            'geojson': '.geojson',
            '.geojson': '.geojson',
            'json': '.json',
            '.json': '.json',
            'gpkg': '.gpkg',
            '.gpkg': '.gpkg',
            'shp': '.shp',
            '.shp': '.shp',
        }
        output_ext = file_type_to_ext.get(str(file_type).lower())
        if output_ext is None:
            raise ValueError(
                f"Unsupported file_type '{file_type}'. "
                "Supported: geoparquet, parquet, geojson, json, gpkg, shp"
            )

        files = []
        for name in os.listdir(input_folder):
            path = os.path.join(input_folder, name)
            if not os.path.isfile(path):
                continue
            if prefix is not None and not name.startswith(prefix):
                continue
            if os.path.splitext(name)[1].lower() not in supported_exts:
                continue
            files.append(name)

        files = sorted(files)
        if len(files) == 0:
            LOGGER.warning("No input files found for to_geo_dataframe_batch in %s with prefix=%s", input_folder, prefix)
            return []

        LOGGER.info(
            "Starting GeoDataFrame batch export: %d file(s), prefix=%s, file_type=%s",
            len(files),
            prefix,
            output_ext,
        )

        output_paths = []
        file_iter = tqdm(files, total=len(files), desc="GeoDataFrame batch", unit="file")
        for filename in file_iter:
            file_iter.set_postfix_str(filename)
            input_path = os.path.join(input_folder, filename)
            output_name = os.path.splitext(filename)[0] + output_ext
            output_path = os.path.join(output_folder, output_name)

            df = self._load_dataframe_from_path(input_path)
            gdf = self._build_geodataframe_from_dataframe(
                df,
                lon_column=lon_column,
                lat_column=lat_column,
                crs=crs,
            )

            if output_ext in ('.parquet', '.geoparquet', '.pq', '.pqt'):
                gdf.to_parquet(output_path, index=False)
            elif output_ext in ('.geojson', '.json'):
                gdf.to_file(output_path, driver='GeoJSON')
            elif output_ext == '.gpkg':
                gdf.to_file(output_path, driver='GPKG')
            elif output_ext == '.shp':
                gdf.to_file(output_path, driver='ESRI Shapefile')
            else:
                raise ValueError(
                    f"Unsupported geospatial output format '{output_ext}' for file '{filename}'."
                )

            output_paths.append(output_path)
            LOGGER.info("GeoDataFrame file saved: %s", output_path)

        LOGGER.info("GeoDataFrame batch export completed: %d output file(s)", len(output_paths))
        return output_paths

    def recursive_fill(self, column_to_fill_name='ta', 
                              variable='ta', 
                              longitude=-7.593311291,
                              latitude=31.66749781):
        
        """
        Recursively fills missing values in the specified column using a specified variable and coordinates.

        Args:
            column_to_fill_name (str): The name of the column to fill. Defaults to 'ta'.
            variable (str): The variable to use for filling missing values. Defaults to 'ta'.
            latitude (float): The latitude coordinate to use for filling missing values. Defaults to 31.66749781.
            longitude (float): The longitude coordinate to use for filling missing values. Defaults to -7.593311291.

        Returns:
            None
        """
        if self.missing_data_checking(column_to_fill_name) == 0:
            print("No missing data found.")
        elif self.missing_data_checking(column_to_fill_name) > 1000:
            import numpy as np
            data_chuncks = np.array_split(self.data.get_dataframe(), 2)
            return DataFrame(ClimateFiller(data_path=data_chuncks[0]).fill(column_to_fill_name,
                                                                                 variable,
                                                                                 latitude,
                                                                                 longitude), data_type='df').append_dataframe(ClimateFiller(data_path=data_chuncks[1]).fill(
                                                                                 column_to_fill_name,
                                                                                 variable,
                                                                                 latitude,
                                                                                 longitude))
    
    def _impute_single_column(self, column_to_fill_name='ta', 
                              product="era5_land",
                              machine_learning_enabled=False,
                              train_ratio=1,
                              model_name='xgboost',
                              export_dataset=False,
                              **kwargs
                              ):
        """
        Fills missing values in the specified column using data retrieval and optionally machine learning techniques.

        Args:
            self (object): The instance of the class.
            column_to_fill_name (str): The name of the column to fill. Defaults to 'ta'.
            longitude (float): The longitude coordinate to use for data retrieval. Defaults to -7.593311291.
            latitude (float): The latitude coordinate to use for data retrieval. Defaults to 31.66749781.
            product (str): The data product to retrieve for filling missing values. Defaults to "era5_Land".
            machine_learning_enabled (bool): Whether to use machine learning techniques for filling missing values. Defaults to False.
            export_dataset (bool): Whether to export the training dataset used for ML error modeling to artifact_folder. Defaults to False.
            backend (str or None): The backend to use for data retrieval. Defaults to None.

        Returns:
            None

        Notes:
            - Missing values in the specified column will be replaced with appropriate data retrieved from the specified coordinates.
            - The data product specified will be used to retrieve relevant data for filling missing values.
            - The option to enable machine learning techniques allows for more sophisticated filling strategies.
            - If the backend is not specified, the method will use the default backend associated with the class.
            - The effectiveness of the filling process may depend on the data availability and the chosen backend.
        """
        self._ml_impute_config = {
            'train_ratio': train_ratio,
            'model_name': model_name,
            'model_kwargs': kwargs,
            'export_dataset': export_dataset,
        }

        # Use coordinates defined at class initialization.
        # Keep backward compatibility for legacy calls passing lon/lat through kwargs.
        lon = kwargs.pop('lon', self.lon)
        lat = kwargs.pop('lat', self.lat)
        self._ml_impute_config['model_kwargs'] = kwargs

        requested_column_to_fill_name = column_to_fill_name
        variable_context = self._resolve_imputation_variable_context(requested_column_to_fill_name)
        canonical_column_to_fill_name = variable_context['canonical']
        target_column_to_fill_name = self._ensure_impute_target_column(requested_column_to_fill_name)
        target_frequency = self._infer_frequency_label_from_index(self.data.get_dataframe().index)

        missing_count = self.missing_data_checking(target_column_to_fill_name, verbose=False)
        dataframe = self._get_underlying_dataframe()
        total_rows = dataframe.shape[0]
        missing_percent = round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0
        print(
            "Missing data statistic for {}: {} missing value(s) out of {} rows ({}%).".format(
                target_column_to_fill_name,
                missing_count,
                total_rows,
                missing_percent,
            )
        )

        if self.backend == 'gee':
            if missing_count == 0:
                print('No missing data found in ' + target_column_to_fill_name)
                return
            
            if product=='era5_land':
                era5_land_variable_map = {
                    'ta': ['temperature_2m'],
                    'rh': ['temperature_2m', 'dewpoint_temperature_2m'],
                    'rs': ['surface_solar_radiation_downwards'],
                    'ws': ['u_component_of_wind_10m', 'v_component_of_wind_10m'],
                    'p': ['total_precipitation'],
                }
                era5_land_variables = era5_land_variable_map.get(canonical_column_to_fill_name)
                raw_remote_passthrough = era5_land_variables is None
                if raw_remote_passthrough:
                    era5_land_variables = [canonical_column_to_fill_name]
                    LOGGER.info(
                        "Using raw remote pass-through for variable '%s' in ERA5-Land GEE branch.",
                        canonical_column_to_fill_name,
                    )
                    
                indexes = []
                indexes_source = self.data.get_dataframe().index if machine_learning_enabled else self.data.get_missing_data_indexes_in_column(target_column_to_fill_name)
                for p in tqdm(indexes_source, desc="Collect timestamps", unit="ts"):
                    normalized_value = self._normalize_datetime_value(p)
                    if normalized_value is not None:
                        indexes.append(normalized_value)

                years = set()
                for p in indexes:
                    years.add(p.year)
                missing_data_dates = {}
                years = list(years)
                years.sort()
                range_start = min(indexes)
                range_end = max(indexes)
                source_cache_path = self._build_source_cache_path(
                    product,
                    canonical_column_to_fill_name,
                    lon,
                    lat,
                    range_start,
                    range_end,
                    frequency=target_frequency,
                )
                legacy_source_cache_path = self._build_source_cache_path(
                    product,
                    canonical_column_to_fill_name,
                    lon,
                    lat,
                    range_start,
                    range_end,
                    frequency=None,
                )
                source_cache_candidate = source_cache_path if os.path.exists(source_cache_path) else legacy_source_cache_path
                if os.path.exists(source_cache_candidate):
                    source_cache_path = source_cache_candidate
                    print(f"Reusing cached source data from: {source_cache_path}")
                    LOGGER.info("Source cache hit: %s", source_cache_path)
                    source_series = self._load_source_series_cache(source_cache_path)
                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        source_series,
                        product,
                        machine_learning_enabled,
                    )
                    print('Imputation of missing data for ' + column_to_fill_name + ' from ERA5-Land was done.')
                    return
                print("Found missing data for {} in year(s): {}".format(column_to_fill_name, years))  
                self.download_era5_land_data_by_years(era5_land_variables, datetime.datetime(min(years), 1, 1), datetime.datetime(max(years), 12, 31))
                            
                from data_science_toolkit.gis import GIS
                gis = GIS()
                data = DataFrame()
                
                if canonical_column_to_fill_name == 'ta':
                    
                    for year in tqdm(years, desc="Load ERA5 yearly cache", unit="year"):
                        cache_path = self._resolve_era5_year_cache_path(era5_land_variables, lon, lat, year, frequency=target_frequency)
                        data_year = DataFrame(cache_path)
                        data.append_dataframe(data_year.dataframe)
                        
                    data.rename_columns({'first': 't2m'})
                    data.set_dataframe(self._prepare_datetime_column(data.get_dataframe()))
                    data.missing_data('t2m')
                    data.transform_column('t2m', lambda o: o - 273.15)
                    data.drop_duplicated_indexes()

                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        data.get_dataframe()['t2m'],
                        product,
                        machine_learning_enabled,
                    )
                    self._save_source_series_cache(data.get_dataframe()['t2m'], source_cache_path)
                            
                    print('Imputation of missing data for ta from ERA5-Land was done!')
                    
                elif canonical_column_to_fill_name == 'rh':
                    
                    for year in tqdm(years, desc="Load ERA5 yearly cache", unit="year"):
                        cache_path = self._resolve_era5_year_cache_path(era5_land_variables, lon, lat, year, frequency=target_frequency)
                        data_year = DataFrame(cache_path)
                        data.append_dataframe(data_year.dataframe)
                    data.set_dataframe(self._prepare_datetime_column(data.get_dataframe()))
                    data.rename_columns({'temperature_2m': 't2m', 'dewpoint_temperature_2m': 'd2m'})
                    data.missing_data('t2m')
                    data.missing_data('d2m')
                    data.transform_column('t2m', lambda o: o - 273.15)
                    data.transform_column('d2m', lambda o: o - 273.15)
                    data.add_column_based_on_function('era5_hr', lambda row: Lib.relative_humidity_magnus(row['t2m'], row['d2m']))
                    data.missing_data('era5_hr')
                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        data.get_dataframe()['era5_hr'],
                        product,
                        machine_learning_enabled,
                    )
                    self._save_source_series_cache(data.get_dataframe()['era5_hr'], source_cache_path)
                    
                    print('Imputation of missing data for rh from ERA5-Land was done!')
                    
                elif canonical_column_to_fill_name == 'ws':
                    
                    for year in tqdm(years, desc="Load ERA5 yearly cache", unit="year"):
                        cache_path = self._resolve_era5_year_cache_path(era5_land_variables, lon, lat, year, frequency=target_frequency)
                        data_year = DataFrame(cache_path)
                        data.append_dataframe(data_year.dataframe)
                    data.set_dataframe(self._prepare_datetime_column(data.get_dataframe()))
                    data.rename_columns({'u_component_of_wind_10m': 'u10', 'v_component_of_wind_10m': 'v10'})
                  
                    data.add_column_based_on_function('era5_ws', lambda row: Lib.logarithmic_wind_profile(row['u10'], row['v10']))
                    data.missing_data('u10')
                    data.missing_data('era5_ws')
                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        data.get_dataframe()['era5_ws'],
                        product,
                        machine_learning_enabled,
                    )
                    self._save_source_series_cache(data.get_dataframe()['era5_ws'], source_cache_path)
                    
                    print('Imputation of missing data for wind speed from ERA5-Land was done!')
                    
                elif canonical_column_to_fill_name == 'rs':
                    
                    for year in tqdm(years, desc="Load ERA5 yearly cache", unit="year"):
                        cache_path = self._resolve_era5_year_cache_path(era5_land_variables, lon, lat, year, frequency=target_frequency)
                        data_year = DataFrame(cache_path)
                        data.append_dataframe(data_year.dataframe)
                        
                    data.rename_columns({'first': 'ssrd'})
                    data.set_dataframe(self._prepare_datetime_column(data.get_dataframe()))
                    data.missing_data('ssrd')
                    l = []
                    for p in data.get_index():
                        if p.hour == 1:
                            new_value = data.get_row(p)['ssrd']/3600
                        else:
                            try:
                                previous_hour = data.get_row(p-timedelta(hours=1))['ssrd']
                            except KeyError: # if age is not convertable to int
                                previous_hour = data.get_row(p)['ssrd']
                                
                            new_value = (data.get_row(p)['ssrd'] - previous_hour)/3600
                        l.append(new_value)
                    data.add_column('rs', l)
                    data.keep_columns(['rs'])
                    data.rename_columns({'rs': 'ssrd'})
                    
                    data.transform_column('ssrd', lambda o : o if abs(o) < 1500 else 0 )    
                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        data.get_dataframe()['ssrd'],
                        product,
                        machine_learning_enabled,
                    )
                    self._save_source_series_cache(data.get_dataframe()['ssrd'], source_cache_path)
                
                    print('Imputation of missing data for ' + column_to_fill_name + ' from ERA5-Land was done.')
                
                elif canonical_column_to_fill_name == 'p':
                    for year in tqdm(years, desc="Load ERA5 yearly cache", unit="year"):
                        cache_path = self._resolve_era5_year_cache_path(era5_land_variables, lon, lat, year, frequency=target_frequency)
                        data_year = DataFrame(cache_path)
                        data.append_dataframe(data_year.dataframe)
                        
                    data.rename_columns({'first': 'tp'})
                    data.set_dataframe(self._prepare_datetime_column(data.get_dataframe()))
                    data.missing_data('tp')
                    nan_indices = self.data.get_nan_indexes_of_column(target_column_to_fill_name)
                    data.drop_duplicated_indexes()
                    
                    l = [] 
                    for p in data.get_index():
                        if p.hour == 1:
                            new_value = data.get_row(p)['tp'] * 1000
                        else:
                            try:
                                previous_hour = data.get_row(p-timedelta(hours=1))['tp']
                            except KeyError:
                                previous_hour = data.get_row(p)['tp']
                                
                            new_value = (data.get_row(p)['tp'] - previous_hour)*1000
                        l.append(new_value)
            
                    data.add_column('p', l)
                    data.keep_columns(['p'])
                    data.rename_columns({'p': 'tp'})
                    
                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        data.get_dataframe()['tp'],
                        product,
                        machine_learning_enabled,
                    )
                    self._save_source_series_cache(data.get_dataframe()['tp'], source_cache_path)
                
                    print('Imputation of missing data for ' + column_to_fill_name + ' from ERA5-Land was done.')

                else:
                    for year in tqdm(years, desc="Load ERA5 yearly cache", unit="year"):
                        cache_path = self._resolve_era5_year_cache_path(era5_land_variables, lon, lat, year, frequency=target_frequency)
                        data_year = DataFrame(cache_path)
                        data.append_dataframe(data_year.dataframe)

                    data.set_dataframe(self._prepare_datetime_column(data.get_dataframe()))

                    remote_series = self._extract_remote_series(
                        data.get_dataframe(),
                        preferred_columns=[canonical_column_to_fill_name, 'first'],
                    )
                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        remote_series,
                        product,
                        machine_learning_enabled,
                    )
                    self._save_source_series_cache(remote_series, source_cache_path)

                    print(
                        'Imputation of missing data for '
                        + column_to_fill_name
                        + ' from ERA5-Land raw remote variable was done.'
                    )
                
            elif product == 'merra2':
                merra2_variables = {
                    'ta': 'T2M',
                    'rh': 'RH2M',
                    'ws': 'WS2M',
                    'rs': 'ALLSKY_SFC_SW_DWN',
                    'p': 'PRECTOTCORR',
                    'pr': 'PRECTOTCORR',
                    'wd': 'WD2M'
                }
                merra2_parameter = merra2_variables.get(canonical_column_to_fill_name, canonical_column_to_fill_name)

                start = self.data.get_dataframe().index[0]
                end = self.data.get_dataframe().index[-1]
                source_cache_path = self._build_source_cache_path(
                    product,
                    canonical_column_to_fill_name,
                    lon,
                    lat,
                    start,
                    end,
                    frequency=target_frequency,
                )
                legacy_source_cache_path = self._build_source_cache_path(
                    product,
                    canonical_column_to_fill_name,
                    lon,
                    lat,
                    start,
                    end,
                    frequency=None,
                )
                source_cache_candidate = source_cache_path if os.path.exists(source_cache_path) else legacy_source_cache_path
                if os.path.exists(source_cache_candidate):
                    source_cache_path = source_cache_candidate
                    print(f"Reusing cached source data from: {source_cache_path}")
                    LOGGER.info("Source cache hit: %s", source_cache_path)
                    source_series = self._load_source_series_cache(source_cache_path)
                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        source_series,
                        product,
                        machine_learning_enabled,
                    )
                    self.data.index_to_column()
                    print('Imputation of missing data for ' + column_to_fill_name + ' from MERRA2 was done.')
                    return
                start = datetime.datetime.strftime(start, '%Y%m%d')
                end = datetime.datetime.strftime(end, '%Y%m%d')

                api_url = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
                format = 'json'
                community = 'ag'
                timezone = 'utc'

                params = {
                    'start': start,
                    'end': end,
                    'latitude': lat,
                    'longitude': lon,
                    'community': community,
                    'parameters': merra2_parameter,
                    'format': format,
                    'user': 'ysouidi1',
                    'header': 'true',
                    'time-standard': timezone
                }

                response = requests.get(api_url, params=params)

                if response.status_code != 200:
                    print('Failed to retrieve data:', response.status_code)
                    return None

                data_merra = response.json()
                result = data_merra['properties']['parameter'].get(merra2_parameter)
                if result is None:
                    available = list(data_merra.get('properties', {}).get('parameter', {}).keys())
                    raise ValueError(
                        f"Remote variable '{canonical_column_to_fill_name}' not available in MERRA2 response. Available: {available}"
                    )

                df = pd.DataFrame(result.items(), columns=['datetime', target_column_to_fill_name])
                df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H')

                if len(self.data.get_missing_data_indexes_in_column(target_column_to_fill_name)) == 0:
                    return

                source_series = df.set_index('datetime')[target_column_to_fill_name]
                self._fill_from_source_series(
                    target_column_to_fill_name,
                    source_series,
                    product,
                    machine_learning_enabled,
                )
                self._save_source_series_cache(source_series, source_cache_path)

                self.data.index_to_column()
                print('Imputation of missing data for ' + column_to_fill_name + ' from MERRA2 was done.')
            
            # other data source
            else:
                pass
            
        
            pass
        else:
            if missing_count == 0:
                print('No missing data found in ' + target_column_to_fill_name)
                return
            
            if product=='era5_land':
                era5_land_variable_map = {
                    'ta': ['2m_temperature'],
                    'rh': ['2m_temperature', '2m_dewpoint_temperature'],
                    'rs': ['surface_solar_radiation_downwards'],
                    'ws': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
                    'p': ['total_precipitation'],
                }
                era5_land_variables = era5_land_variable_map.get(canonical_column_to_fill_name)
                raw_remote_passthrough = era5_land_variables is None
                if raw_remote_passthrough:
                    era5_land_variables = [canonical_column_to_fill_name]
                    LOGGER.info(
                        "Using raw remote pass-through for variable '%s' in ERA5-Land CDS branch.",
                        canonical_column_to_fill_name,
                    )
                    
                    
                from data_science_toolkit.gis import GIS
                import cdsapi
                c = cdsapi.Client()

                """if self.datetime_column_name is not None:
                    self.data.reindex_dataframe(self.datetime_column_name)"""

                indexes = []
                indexes_source = self.data.get_dataframe().index if machine_learning_enabled else self.data.get_missing_data_indexes_in_column(target_column_to_fill_name)
                for p in tqdm(indexes_source, desc="Collect timestamps", unit="ts"):
                    if isinstance(p, str) is True:
                        indexes.append(datetime.datetime.strptime(p, '%Y-%m-%d %H:%M:%S'))
                    else:
                        indexes.append(p)
                    
                years = set()
                for p in indexes:
                    years.add(p.year)     
                missing_data_dates = {}    
                years = list(years)
                range_start = min(indexes)
                range_end = max(indexes)
                source_cache_path = self._build_source_cache_path(
                    product,
                    canonical_column_to_fill_name,
                    lon,
                    lat,
                    range_start,
                    range_end,
                    frequency=target_frequency,
                )
                legacy_source_cache_path = self._build_source_cache_path(
                    product,
                    canonical_column_to_fill_name,
                    lon,
                    lat,
                    range_start,
                    range_end,
                    frequency=None,
                )
                source_cache_candidate = source_cache_path if os.path.exists(source_cache_path) else legacy_source_cache_path
                if os.path.exists(source_cache_candidate):
                    source_cache_path = source_cache_candidate
                    print(f"Reusing cached source data from: {source_cache_path}")
                    LOGGER.info("Source cache hit: %s", source_cache_path)
                    source_series = self._load_source_series_cache(source_cache_path)
                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        source_series,
                        product,
                        machine_learning_enabled,
                    )
                    print('Imputation of missing data for ' + column_to_fill_name + ' from ERA5-Land was done.')
                    return
                print("Found missing data for {} in year(s): {}".format(column_to_fill_name, years))  
                for y in years:
                    missing_data_dict = {}
                    missing_data_dict['month'] = set()   
                    missing_data_dict['day'] = set() 
                    
                    for p in indexes:
                        if p.year == y:
                            missing_data_dict['month'].add(p.strftime('%m'))
                            missing_data_dict['day'].add(p.strftime('%d'))
                    missing_data_dict['month'] = list(missing_data_dict['month'])
                    missing_data_dict['day'] = list(missing_data_dict['day'])
                    missing_data_dates[y] = missing_data_dict
                    for month in missing_data_dict['month']:
                        for p in era5_land_variables:
                            data_month_path = 'data\era5land_' + p + '_' + str(lon) + '_' + str(lat) + '_' + str(y) + '_' + month + '.grib'
                            if os.path.exists(data_month_path) is False:
                                c.retrieve(
                                'reanalysis-era5-land',
                                {
                                    'format': 'grib',
                                    'variable': p,
                                    'year': str(y),
                                    'month':  month,
                                    'day': missing_data_dict['day'],
                                    'time': [
                                        '00:00', '01:00', '02:00',
                                        '03:00', '04:00', '05:00',
                                        '06:00', '07:00', '08:00',
                                        '09:00', '10:00', '11:00',
                                        '12:00', '13:00', '14:00',
                                        '15:00', '16:00', '17:00',
                                        '18:00', '19:00', '20:00',
                                        '21:00', '22:00', '23:00',
                                    ],
                                    'area': [lat, lon, lat, lon],
                                },
                                data_month_path)
                            else:
                                print(f'Data of {y}-{month} found in {data_month_path}')
                            
                
                gis = GIS()
                data = DataFrame()
                
                if canonical_column_to_fill_name == 'ta':
                    for year in missing_data_dates:
                        for month in missing_data_dates[year]['month']:
                            data_month_path = 'data\era5land_' + column_to_fill_name + '_' + str(lon) + '_' + str(lat) + '_' + str(year) + '_' + month + '.grib'
                            data.append_dataframe(gis.get_era5_land_grib_as_dataframe(data_month_path, "ta"),)
                    
                    data.reset_index()
                    data.column_to_date('valid_time')
                    data.reindex_dataframe("valid_time")
                    data.set_dataframe(data.get_dataframe().sort_index())
                    data.missing_data('t2m')
                    data.transform_column('t2m', lambda o: o - 273.15)
                    data.drop_duplicated_indexes()
                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        data.get_dataframe()['t2m'],
                        product,
                        machine_learning_enabled,
                    )
                    self._save_source_series_cache(data.get_dataframe()['t2m'], source_cache_path)
                            
                    print('Imputation of missing data for ta from ERA5-Land was done!')
                    
                elif canonical_column_to_fill_name == 'rh':
                    data_t2m = DataFrame()
                    data_d2m = DataFrame()
                    
                    for year in missing_data_dates:
                        for month in missing_data_dates[year]['month']:
                            month_data_t2m = 'data\era5land_' + '2m_temperature' + '_' + str(lon) + '_' + str(lat) + '_' + str(year) + '_' + month + '.grib'
                            data_t2m.append_dataframe(gis.get_era5_land_grib_as_dataframe(month_data_t2m, "ta"),)
                    
                    for year in missing_data_dates:
                        for month in missing_data_dates[year]['month']:
                            month_data_d2m = 'data\era5land_' + '2m_dewpoint_temperature' + '_' + str(lon) + '_' + str(lat) + '_' + str(year) + '_' + month + '.grib'
                            data_d2m.append_dataframe(gis.get_era5_land_grib_as_dataframe(month_data_d2m, "ta"),)
                    
                    data_d2m.reset_index()
                    data_d2m.reindex_dataframe("valid_time")
                    data_d2m.keep_columns(['d2m'])
                    data_t2m.reset_index()
                    data_t2m.reindex_dataframe("valid_time")
                    data_t2m.keep_columns(['t2m'])
                    data_t2m.join(data_d2m.get_dataframe())
                    data = data_t2m
                    data.missing_data('t2m')
                    data.transform_column('t2m', lambda o: o - 273.15)
                    data.transform_column('d2m', lambda o: o - 273.15)
                    data.add_column_based_on_function('era5_hr', lambda row: Lib.get_relative_humidity(row['t2m', 'd2m']))
                    #data.add_transformed_columns('era5_hr', '100*exp(-((243.12*17.62*t2m)-(d2m*17.62*t2m)-d2m*17.62*(243.12+t2m))/((243.12+t2m)*(243.12+d2m)))')
                    data.missing_data('era5_hr')
                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        data.get_dataframe()['era5_hr'],
                        product,
                        machine_learning_enabled,
                    )
                    self._save_source_series_cache(data.get_dataframe()['era5_hr'], source_cache_path)
                    
                    print('Imputation of missing data for rh from ERA5-Land was done!')
                    
                elif canonical_column_to_fill_name == 'ws':
                    data_u10 = DataFrame()
                    data_v10 = DataFrame()
                    for year in missing_data_dates:
                        for month in missing_data_dates[year]['month']:
                            month_data_u10 = 'data\era5land_' + '10m_u_component_of_wind' + '_' + str(lon) + '_' + str(lat) + '_' + str(year) + '_' + month + '.grib'
                            data_u10.append_dataframe(gis.get_era5_land_grib_as_dataframe(month_data_u10, "ta"),)
                    for year in missing_data_dates:
                        for month in missing_data_dates[year]['month']:
                            month_data_v10 = 'data\era5land_' + '10m_v_component_of_wind' + '_' + str(lon) + '_' + str(lat) + '_' + str(year) + '_' + month + '.grib'
                            data_v10.append_dataframe(gis.get_era5_land_grib_as_dataframe(month_data_v10, "ta"),)
                    
                    data_u10.reset_index()
                    data_u10.reindex_dataframe("valid_time")
                    data_u10.keep_columns(['u10'])
                    data_v10.reset_index()
                    data_v10.reindex_dataframe("valid_time")
                    data_v10.keep_columns(['v10'])
                    data_v10.join(data_u10.get_dataframe())
                    data = data_v10
                    data.add_column_based_on_function('era5_ws', Lib.get_2m_wind_speed)
                    data.missing_data('u10')
                    data.missing_data('era5_ws')
                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        data.get_dataframe()['era5_ws'],
                        product,
                        machine_learning_enabled,
                    )
                    self._save_source_series_cache(data.get_dataframe()['era5_ws'], source_cache_path)
                    
                    print('Imputation of missing data for wind speed from ERA5-Land was done!')
                    
                elif canonical_column_to_fill_name == 'rs':
                    for year in missing_data_dates:
                        for month in missing_data_dates[year]['month']:
                            data_month_path = 'data\era5land_' + 'surface_solar_radiation_downwards' + '_' + str(lon) + '_' + str(lat) + '_' + str(year) + '_' + month + '.grib'
                            data.append_dataframe(gis.get_era5_land_grib_as_dataframe(data_month_path, "ta"),)
                            
                    data.reset_index()
                    data.reindex_dataframe("valid_time")
                    data.missing_data('ssrd')
                    l = []
                    for p in data.get_index():
                        if p.hour == 1:
                            new_value = data.get_row(p)['ssrd']/3600
                        else:
                            try:
                                previous_hour = data.get_row(p-timedelta(hours=1))['ssrd']
                            except KeyError: # if age is not convertable to int
                                previous_hour = data.get_row(p)['ssrd']
                                
                            new_value = (data.get_row(p)['ssrd'] - previous_hour)/3600
                        l.append(new_value)
                    data.add_column('rs', l)
                    data.keep_columns(['rs'])
                    data.rename_columns({'rs': 'ssrd'})
                    
                    data.transform_column('ssrd', lambda o : o if abs(o) < 1500 else 0 )    
                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        data.get_dataframe()['ssrd'],
                        product,
                        machine_learning_enabled,
                    )
                    self._save_source_series_cache(data.get_dataframe()['ssrd'], source_cache_path)
                
                    print('Imputation of missing data for ' + column_to_fill_name + ' from ERA5-Land was done.')
                
                elif canonical_column_to_fill_name == 'p':
                    for year in missing_data_dates:
                        for month in missing_data_dates[year]['month']:
                            data_month_path = 'data\era5land_' + 'total_precipitation' + '_' + str(lon) + '_' + str(lat) + '_' + str(year) + '_' + month + '.grib'
                            data.append_dataframe(gis.get_era5_land_grib_as_dataframe(data_month_path, "ta"),)
                    
                    
                    data.reset_index()
                    data.column_to_date('valid_time')
                    data.reindex_dataframe("valid_time")
                    data.set_dataframe(data.get_dataframe().sort_index())
                    data.missing_data('tp')
                    nan_indices = self.data.get_nan_indexes_of_column(target_column_to_fill_name)
                    data.drop_duplicated_indexes()
                    
                    l = []
                    for p in data.get_index():
                        if p.hour == 1:
                            new_value = data.get_row(p)['tp'] * 1000
                        else:
                            try:
                                previous_hour = data.get_row(p-timedelta(hours=1))['tp']
                            except KeyError:
                                previous_hour = data.get_row(p)['tp']
                                
                            new_value = (data.get_row(p)['tp'] - previous_hour)*1000
                        l.append(new_value)
            
                    data.add_column('p', l)
                    data.keep_columns(['p'])
                    data.rename_columns({'p': 'tp'})
                    
                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        data.get_dataframe()['tp'],
                        product,
                        machine_learning_enabled,
                    )
                    self._save_source_series_cache(data.get_dataframe()['tp'], source_cache_path)
                
                    print('Imputation of missing data for ' + column_to_fill_name + ' from ERA5-Land was done.')

                else:
                    for year in years:
                        for month in missing_data_dates[year]['month']:
                            data_month_path = 'data\era5land_' + canonical_column_to_fill_name + '_' + str(lon) + '_' + str(lat) + '_' + str(year) + '_' + month + '.grib'
                            data.append_dataframe(gis.get_era5_land_grib_as_dataframe(data_month_path, "ta"),)

                    data.reset_index()
                    data.column_to_date('valid_time')
                    data.reindex_dataframe("valid_time")
                    data.set_dataframe(data.get_dataframe().sort_index())
                    data.drop_duplicated_indexes()

                    remote_series = self._extract_remote_series(
                        data.get_dataframe(),
                        preferred_columns=[canonical_column_to_fill_name],
                    )
                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        remote_series,
                        product,
                        machine_learning_enabled,
                    )
                    self._save_source_series_cache(remote_series, source_cache_path)

                    print(
                        'Imputation of missing data for '
                        + column_to_fill_name
                        + ' from ERA5-Land raw remote variable was done.'
                    )
                
            elif product == 'merra2':
                merra2_variables = {
                    'ta': 'T2M',
                    'rh': 'RH2M',
                    'ws': 'WS2M',
                    'rs': 'ALLSKY_SFC_SW_DWN',
                    'p': 'PRECTOTCORR',
                    'pr': 'PRECTOTCORR',
                    'wd': 'WD2M'
                }
                merra2_parameter = merra2_variables.get(canonical_column_to_fill_name, canonical_column_to_fill_name)

                start = self.data.get_dataframe().index[0]
                end = self.data.get_dataframe().index[-1]
                source_cache_path = self._build_source_cache_path(
                    product,
                    canonical_column_to_fill_name,
                    lon,
                    lat,
                    start,
                    end,
                    frequency=target_frequency,
                )
                legacy_source_cache_path = self._build_source_cache_path(
                    product,
                    canonical_column_to_fill_name,
                    lon,
                    lat,
                    start,
                    end,
                    frequency=None,
                )
                source_cache_candidate = source_cache_path if os.path.exists(source_cache_path) else legacy_source_cache_path
                if os.path.exists(source_cache_candidate):
                    source_cache_path = source_cache_candidate
                    print(f"Reusing cached source data from: {source_cache_path}")
                    LOGGER.info("Source cache hit: %s", source_cache_path)
                    source_series = self._load_source_series_cache(source_cache_path)
                    self._fill_from_source_series(
                        target_column_to_fill_name,
                        source_series,
                        product,
                        machine_learning_enabled,
                    )
                    self.data.index_to_column()
                    print('Imputation of missing data for ' + column_to_fill_name + ' from MERRA2 was done.')
                    return
                start = datetime.datetime.strftime(start, '%Y%m%d')
                end = datetime.datetime.strftime(end, '%Y%m%d')

                api_url = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
                format = 'json'
                community = 'ag'
                timezone = 'utc'

                params = {
                    'start': start,
                    'end': end,
                    'latitude': lat,
                    'longitude': lon,
                    'community': community,
                    'parameters': merra2_parameter,
                    'format': format,
                    'user': 'ysouidi1',
                    'header': 'true',
                    'time-standard': timezone
                }

                response = requests.get(api_url, params=params)

                if response.status_code != 200:
                    print('Failed to retrieve data:', response.status_code)
                    return None

                data_merra = response.json()
                result = data_merra['properties']['parameter'].get(merra2_parameter)
                if result is None:
                    available = list(data_merra.get('properties', {}).get('parameter', {}).keys())
                    raise ValueError(
                        f"Remote variable '{canonical_column_to_fill_name}' not available in MERRA2 response. Available: {available}"
                    )

                df = pd.DataFrame(result.items(), columns=['datetime', target_column_to_fill_name])
                df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H')

                if len(self.data.get_missing_data_indexes_in_column(target_column_to_fill_name)) == 0:
                    return

                source_series = df.set_index('datetime')[target_column_to_fill_name]
                self._fill_from_source_series(
                    target_column_to_fill_name,
                    source_series,
                    product,
                    machine_learning_enabled,
                )
                self._save_source_series_cache(source_series, source_cache_path)

                self.data.index_to_column()
                print('Imputation of missing data for ' + column_to_fill_name + ' from MERRA2 was done.')
            
            # other data source
            else:
                pass
            
        

    def impute(self, column_to_fill_list='ta',
                              product="era5_land",
                              machine_learning_enabled=False,
                              train_ratio=1,
                              model_name='xgboost',
                              export_dataset=False,
                              **kwargs
                              ):
        """
        Fill missing values for one or more columns in a single call.

        Args:
            column_to_fill_list (str or list[str]): Column name or list of column names to impute.

        Returns:
            ClimateFiller: The current instance for chaining.
        """
        if isinstance(column_to_fill_list, str):
            column_names = [column_to_fill_list]
        else:
            column_names = list(column_to_fill_list)

        for column_name in column_names:
            self._impute_single_column(
                column_name,
                product=product,
                machine_learning_enabled=machine_learning_enabled,
                train_ratio=train_ratio,
                model_name=model_name,
                export_dataset=export_dataset,
                **kwargs,
            )

        return self

    def best_ml_model(self, column_to_fill_name, lon, lat, product, metric='rmse'):
        list_models = [
            LinearRegression(),
            DecisionTreeRegressor(),
            RandomForestRegressor(),
            XGBRegressor(),
            CatBoostRegressor(verbose=False),
            Ridge(),
            Lasso(),
            ElasticNet()
        ]
        print(f'Finding the best machine learning model...')
        start_datetime = self.data.get_dataframe().index[0]
        end_datetime = self.data.get_dataframe().index[-1]
        self.download(column_to_fill_name, lon, lat, start_datetime, end_datetime, product)
        
        output_file = 'data/era5_land_' + '_'.join([str(column_to_fill_name), str(lon), str(lat), str(start_datetime.strftime('%Y-%m-%d')), str(end_datetime.strftime('%Y-%m-%d'))]) + '.csv'
        data_temp = DataFrame(output_file)
        self.data_reanalysis.set_dataframe(data_temp.get_dataframe())
        
        self.data_reanalysis.column_to_date('datetime')
        self.data_reanalysis.reindex_dataframe('datetime')

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        self.data.keep_columns([column_to_fill_name])
        self.data_reanalysis.join(self.data.get_dataframe())
        self.data_reanalysis.missing_data(column_to_fill_name)
        
        self.data_reanalysis.reset_index()
        
        y = self.data_reanalysis.get_column(column_to_fill_name)
        X = self.data_reanalysis.drop_column(column_to_fill_name) 
        
        rmse_scores = {}
        r2_scores = {}
        
        if metric=='rmse':
            
            for model in list_models:
                model_rmse_scores = []
                for train_index, test_index in kfold.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)

                    model_rmse_scores.append(rmse)
                
                avg_rmse = np.mean(model_rmse_scores)
                rmse_scores[model.__class__.__name__] = avg_rmse
        
        elif metric=='r2':
            
            for model in list_models:
                model_r2_scores = []
                for train_index, test_index in kfold.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)

                    model_r2_scores.append(r2)
                    
                avg_r2 = np.mean(model_r2_scores)
                r2_scores[model.__class__.__name__] = avg_r2

        best_model_name = min(rmse_scores, key=rmse_scores.get)
        best_model = next((model for model in list_models if model.__class__.__name__ == best_model_name), None)
        self.best_model = best_model
        print(f'The Best perfoming model is: {best_model_name} with {metric.upper()}={rmse_scores[best_model_name]}')
    
    def missing_data_checking(self, column_names_list=None, verbose=True):
        """Check missing values for one column, many columns, or the full dataframe.

        Args:
            column_names_list (str | list[str] | None): Column name, list of column names, or None for all columns.
            verbose (bool): Whether to print a human-readable summary.

        Returns:
            int | dict[str, int]: Missing-value count for a single column, or a mapping of
            column names to missing-value counts for multiple columns or the full dataframe.
        """
        dataframe = self.data.get_dataframe()
        total_rows = dataframe.shape[0]

        if column_names_list is None:
            columns_to_check = list(dataframe.columns)
            missing_counts = {}
            for column in columns_to_check:
                miss_by_column = int(dataframe[column].isnull().sum())
                missing_counts[column] = miss_by_column
                if verbose:
                    if miss_by_column > 0:
                        missing_data_percent = round((miss_by_column / total_rows) * 100, 2) if total_rows else 0
                        print("{} has {} missing value(s) which represents {}% of dataset size".format(column, miss_by_column, missing_data_percent))
                    else:
                        print("{} has NO missing value!".format(column))
            if verbose:
                print('Detail of missing values in the dataset: {}'.format(missing_counts))
            return missing_counts

        if isinstance(column_names_list, (list, tuple, set)):
            column_names = list(column_names_list)
            missing_counts = {}
            for column in column_names:
                miss_by_column = int(dataframe[column].isnull().sum())
                missing_counts[column] = miss_by_column
                if verbose:
                    if miss_by_column > 0:
                        missing_data_percent = round((miss_by_column / total_rows) * 100, 2) if total_rows else 0
                        print("{} has {} missing value(s) which represents {}% of dataset size".format(column, miss_by_column, missing_data_percent))
                    else:
                        print("No missed data in column " + column)
            if verbose:
                print('Detail of missing values in the dataset: {}'.format(missing_counts))
            return missing_counts

        miss = int(dataframe[column_names_list].isnull().sum())
        if miss > 0:
            missing_data_percent = round((miss / total_rows) * 100, 2) if total_rows else 0
            if verbose:
                print("{} has {} missing value(s) which represents {}% of dataset size".format(column_names_list, miss, missing_data_percent))
        elif verbose:
            print("No missed data in column " + column_names_list)

        if verbose:
            print('Detail of missing values in the dataset: {}'.format(miss))
        return miss
    
    def eliminate_outliers(self, climate_varibale_column_name='ta', method='lof', n_neighbors=48, contamination=0.005, n_estimators=100):
        """
        Eliminates outliers in the specified climate variable column using the specified outlier detection method.

        Args:
            self (object): The instance of the class.
            climate_variable_column_name (str): The name of the climate variable column to eliminate outliers from.
                Defaults to 'ta'.

            method (str): The outlier detection method to use. Currently supported methods include:
                - 'lof': Local Outlier Factor algorithm, which measures the local deviation of a data point
                with respect to its neighbors. Defaults to 'lof'.

            n_neighbors (int): The number of neighbors to consider for outlier detection.
                This parameter is only applicable to certain outlier detection methods. Defaults to 48.

            contamination (float): The expected proportion of outliers in the data.
                This parameter is only applicable to certain outlier detection methods. Defaults to 0.005.

            n_estimators (int): The number of base estimators to use for ensemble-based outlier detection methods.
                This parameter is only applicable to certain outlier detection methods. Defaults to 100.

        Returns:
            None

        Notes:
            - Outliers are data points that significantly deviate from the majority of the data.
            - The specified climate variable column will be processed to identify and eliminate outliers.
            - The chosen outlier detection method will be applied to identify and mark outliers in the data.
            - The method aims to improve the quality and reliability of the climate variable data by removing outliers.
            - The effectiveness and performance of the outlier elimination process may vary depending on the method and parameters used.
        """
        if method == 'lof':
            outliers_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            self.data.get_dataframe()['inlier'] = outliers_model.fit_predict(self.data.get_columns([climate_varibale_column_name]))
            print('Number of detected outliers: {}'.format(self.data.count_occurence_of_each_row('inlier').iloc[0]))
            self.data.dataframe.loc[self.data.get_dataframe()['inlier'] == -1, climate_varibale_column_name] = None
            self.data.drop_column('inlier')
        
        elif method == 'isolation_forest':
            outliers_model = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
            self.data.get_dataframe()['inlier'] = outliers_model.fit_predict(self.data.get_columns([climate_varibale_column_name]))
            print('Number of detected outliers: {}'.format(self.data.count_occurence_of_each_row('inlier').iloc[0]))
            self.data.dataframe.loc[self.data.get_dataframe()['inlier'] == -1, climate_varibale_column_name] = 2000
            self.data.drop_column('inlier')
        
        elif method == 'quantiles':
            outliers_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            self.data.get_dataframe()['inlier'] = outliers_model.fit_predict(self.data.get_columns([climate_varibale_column_name]))
            print('Number of detected outliers: {}'.format(self.data.count_occurence_of_each_row('inlier').iloc[0]))
            self.data.dataframe.loc[self.data.get_dataframe()['inlier'] == -1, climate_varibale_column_name] = None
            self.data.drop_column('inlier')
    
    def evaluate_products(self):
        pass
    
    def plot_column(self, column):
        """Function Name: plot_column

            Description:
            This function creates a time-series plot of a column in a dataframe.

            Parameters:

            self: the instance of the class that the function is a part of.
            column: the name of the column to plot.
            Returns:
            None. The function generates a time-series plot of the specified column.

            Note:
            This function assumes that the dataframe has already been loaded into the class instance. This function requires the matplotlib and seaborn libraries to be installed.
        """
        self.data.get_column(column).plot()
        plt.show()
        
    def extraterrestrial_radiation_daily(self, column_name='ra', nbr_decimal_places=None):
        self.data.index_to_column()
        self.data.add_doy_column(datetime_column_name=self.datetime_column_name)
        self.data.add_one_value_column('lat', self.lat)
        self.data.add_column_based_on_function(
            column_name, 
            lambda row: Lib.extraterrestrial_radiation_daily(
                    row['lat'],
                    row['doy'])
        )
        if nbr_decimal_places is not None:
            self.data.transform_column(column_name, lambda o: round(o, nbr_decimal_places))
        self.data.reindex_dataframe(self.datetime_column_name)
    
    def et0_estimation(self, 
                       ta_column_name='ta',
                       rs_column_name='rs',
                       rh_column_name='rh',
                       ws_column_name='ws',
                       method='pm',
                       freq='d',
                       reference_crop='grass',
                       nbr_decimal_places=2,
                       c_hs=0.0023,
                       a_hs=17.8,
                       b_hs=0.5,
                       k1_ab=0.53,
                       alpha_pt=1.26,
                       ):
        """
        Estimates reference evapotranspiration (ET0) using the specified meteorological data and method.

        Args:
            self (object): The instance of the class.
            air_temperature_column_name (str): The name of the column that contains air temperature data. Defaults to 'ta'.
            global_solar_radiation_column_name (str): The name of the column that contains global solar radiation data. Defaults to 'rs'.
            air_relative_humidity_column_name (str): The name of the column that contains air relative humidity data. Defaults to 'rh'.
            wind_speed_column_name (str): The name of the column that contains wind speed data. Defaults to 'ws'.
            date_time_column_name (str): The name of the column that contains date and time information. Defaults to 'date_time'.
            latitude (float): The latitude coordinate of the location for ET0 estimation. Defaults to 31.65410805.
            longitude (float): The longitude coordinate of the location for ET0 estimation. Defaults to -7.603140831.
            method (str): The method to use for ET0 estimation. Currently supported methods include:
                - 'pm': Penman-Monteith method, which is based on the FAO56 Penman-Monteith equation. Defaults to 'pm'.
            in_place (bool): Whether to replace the original ET0 column in the dataset or create a new column. Defaults to True.

        Returns:
            None

        Notes:
            - ET0 estimation is a measure of the potential evapotranspiration from a reference crop.
            - The method utilizes meteorological data such as air temperature, global solar radiation, air relative humidity, and wind speed.
            - The specified columns in the dataset will be used for ET0 estimation.
            - The latitude and longitude coordinates define the location for ET0 estimation.
            - The chosen method will be applied to calculate ET0 values.
            - If in_place is True, the original ET0 column will be replaced; otherwise, a new column will be created.
        """
        
        
        data_temp = DataFrame() 
        
        if freq == 'd':
            data_temp.add_column('ta_mean', self.data.resample_timeseries(in_place=False)[ta_column_name])
            data_temp.add_column('ta_max', self.data.resample_timeseries(in_place=False, agg='max')[ta_column_name])
            data_temp.add_column('ta_min', self.data.resample_timeseries(in_place=False, agg='min')[ta_column_name], )
            
            data_temp.index_to_column()
            data_temp.add_doy_column(datetime_column_name=self.datetime_column_name)
            data_temp.add_one_value_column('lat', self.lat)
            data_temp.add_one_value_column('lon', self.lon)
            data_temp.reindex_dataframe(self.datetime_column_name)
            
            if method == 'pm':
                data_temp.add_column('rh_max', self.data.resample_timeseries(in_place=False, agg='max')[rh_column_name])
                data_temp.add_column('rh_min', self.data.resample_timeseries(in_place=False, agg='min')[rh_column_name])
                data_temp.add_column('rh_mean', self.data.resample_timeseries(in_place=False)[rh_column_name])
                data_temp.add_column('ws_mean', self.data.resample_timeseries(in_place=False)[ws_column_name])
                data_temp.add_column('rs_mean', self.data.resample_timeseries(in_place=False)[rs_column_name])
                
                if self.elevation is None:
                    data_temp.add_one_value_column('elevation', Lib.get_elevation(self.lat, self.lon))
                else:
                    data_temp.add_one_value_column('elevation', self.elevation)
                
                data_temp.add_column_based_on_function('et0_pm', lambda row: Lib.et0_penman_monteith_daily(row))
                data_temp.transform_column('et0_pm', lambda o: o if o > 0 else 0)
                data_temp.transform_column('et0_pm', lambda o: round(o, nbr_decimal_places))
            
            elif method == 'hs':
                data_temp.add_column_based_on_function('et0_hs', lambda row: Lib.et0_hargreaves_samani(
                    row,
                    c=c_hs,
                    a=a_hs,
                    b=b_hs))
                data_temp.transform_column('et0_hs', lambda o: o if o > 0 else 0)
                data_temp.transform_column('et0_hs', lambda o: round(o, nbr_decimal_places))

            elif method == 'pt':
                data_temp.add_column('rh_max', self.data.resample_timeseries(in_place=False, agg='max')[rh_column_name])
                data_temp.add_column('rh_min', self.data.resample_timeseries(in_place=False, agg='min')[rh_column_name])
                data_temp.add_column('rh_mean', self.data.resample_timeseries(in_place=False)[rh_column_name])
                data_temp.add_column('rs_mean', self.data.resample_timeseries(in_place=False)[rs_column_name])
                
                if self.elevation is None:
                    data_temp.add_one_value_column('elevation', Lib.get_elevation(self.lat, self.lon))
                else:
                    data_temp.add_one_value_column('elevation', self.elevation)
                
                data_temp.add_one_value_column('lat', self.lat)
                
                data_temp.add_column_based_on_function('et0_pt', lambda row: Lib.et0_priestley_taylor_daily(row, alpha_pt))
                data_temp.transform_column('et0_pt', lambda o: o if o > 0 else 0)
                data_temp.transform_column('et0_pt', lambda o: round(o, nbr_decimal_places))
                
                
            elif method == 'sd':
                data_temp.add_column('rh_mean', self.data.resample_timeseries(in_place=False)[rh_column_name])
                data_temp.add_column_based_on_function('et0_sd', Lib.et0_schendel)
                
            elif method == 'ab':
                data_temp.add_column('rs_mean', self.data.resample_timeseries(in_place=False)[rs_column_name])
                data_temp.add_column_based_on_function('et0_ab', lambda row: Lib.et0_abtew(row, k1=k1_ab))
                data_temp.transform_column('et0_ab', lambda o: round(o, nbr_decimal_places))
                
            elif method == 'tu':
                data_temp.add_column('rh_mean', self.data.resample_timeseries(in_place=False)[rh_column_name])
                data_temp.add_column('rs_mean', self.data.resample_timeseries(in_place=False)[rs_column_name])
                data_temp.add_column_based_on_function('et0_tu', Lib.et0_turc)
           
            elif method == 'mk':
                if self.elevation is None:
                    data_temp.add_one_value_column('elevation', Lib.get_elevation(self.lat, self.lon))
                else:
                    data_temp.add_one_value_column('elevation', self.elevation)
                data_temp.add_column('rs_mean', self.data.resample_timeseries(in_place=False)[rs_column_name])
                data_temp.add_column_based_on_function('et0_mk', Lib.et0_makkink)
                data_temp.transform_column('et0_mk', lambda o: o if o > 0 else 0)
                data_temp.transform_column('et0_mk', lambda o: round(o, nbr_decimal_places))
            
            self.et0_output_data.set_dataframe(data_temp.get_dataframe())
                
        elif freq == 'h':
            self.et0_output_data.dataframe = self.data.dataframe.copy()
            self.et0_output_data.transform_column(rs_column_name, lambda o: o if o > 0 else 0)
            self.et0_output_data.index_to_column()
            self.et0_output_data.add_doy_column(datetime_column_name=self.datetime_column_name)
            self.et0_output_data.add_hod_column(datetime_column_name=self.datetime_column_name)
            self.et0_output_data.transform_column('hod', lambda o: o + 1)
            self.et0_output_data.reindex_dataframe(self.datetime_column_name)
            
            if self.elevation is None:
                self.et0_output_data.add_one_value_column('elevation', Lib.get_elevation(self.lat, self.lon))
            else:
                self.et0_output_data.add_one_value_column('elevation', self.elevation)
                
            self.et0_output_data.add_one_value_column('lat', self.lat)
            self.et0_output_data.add_one_value_column('lon', self.lon)
            
            
            if method == 'pm':
                self.et0_output_data.add_column_based_on_function('et0_pm', lambda row: Lib.et0_penman_monteith_hourly(
                    row, 
                    ta_column_name,
                    rs_column_name,
                    rh_column_name,
                    ws_column_name,
                    self.tz_offset,
                    reference_crop
                    ))
                self.et0_output_data.transform_column('et0_pm', lambda o: o if o > 0 else 0)
                self.et0_output_data.transform_column('et0_pm', lambda o: round(o, 2))
                
            
            elif method == 'pt':
                self.et0_output_data.add_column_based_on_function('et0_pt', lambda row: Lib.et0_priestley_taylor(
                    row, 
                    ta_column_name,
                    rs_column_name,
                    rh_column_name,
                    reference_crop))
                self.et0_output_data.transform_column('et0_pt', lambda o: o if o > 0 else 0)
                
            elif method == 'ab':
                self.et0_output_data.add_column_based_on_function('et0_ab', lambda row: Lib.et0_priestley_taylor_hourly(row, ta_column_name, rs_column_name))
            elif method == 'tu':
                self.et0_output_data.add_column_based_on_function('et0_tu', lambda row: Lib.et0_priestley_taylor_hourly(row, ta_column_name, rs_column_name))
            elif method == 'sd':
                self.et0_output_data.add_column_based_on_function('et0_sd', lambda row: Lib.et0_priestley_taylor_hourly(row, ta_column_name, rs_column_name))

            
        return self.et0_output_data.get_dataframe()
    
    def apply_quality_control_criteria(self, variable_column_name, decision_func=lambda x:x>0):
        """
        Applies quality control criteria to the specified variable column based on a decision function.

        Args:
            self (object): The instance of the class.
            variable_column_name (str): The name of the column containing the variable to apply quality control to.
            decision_func (function, optional): The decision function used to determine if a value passes quality control.
                Defaults to lambda x: x > 0, which checks if the value is greater than zero.

        Returns:
            None

        Notes:
            - The apply_quality_control_criteria method is used to perform quality control on a variable column.
            - The specified variable_column_name is the column in the dataset that will undergo quality control.
            - The decision_func parameter allows customization of the quality control criteria by providing a decision function.
            - The decision function should take a value as input and return True if it passes quality control, False otherwise.
            - Values in the variable column that do not meet the quality control criteria will be marked or processed accordingly.
            - The quality control process helps identify and handle data points that may be inaccurate, erroneous, or outliers.
        """
        
        self.data.add_column('decision', self.data.get_column(variable_column_name).apply(decision_func))
        self.data.get_dataframe().loc[ self.data.get_dataframe()['decision'] == False, variable_column_name] = None
        self.data.drop_column('decision')
        
    def apply_constraint(self, column_name, constraint):
        """Function Name: apply_constraint

            Description:
            This function applies a constraint to a column of a dataframe.

            Parameters:

            self: the instance of the class that the function is a part of.
            column_name: the name of the column to apply the constraint to.
            constraint: a string that represents the constraint to apply. The string should be in the form of a valid Python expression. The constraint will be applied to the column using the eval() function.
            Returns:
            A dataframe with the specified constraint applied to the specified column.

            Note:
            This function assumes that the dataframe has already been loaded into the class instance. The constraint parameter should be a valid Python expression that can be evaluated using the eval() function. This function requires the pandas library to be installed.
        """
        self.data.filter_dataframe(column_name, constraint)
    
    def missing_data(self, drop_row_if_nan_in_column=None, filling_dict_colmn_val=None, method='ffill',
                     column_to_fill='ta', date_column_name=None):
        """Function Name: missing_data

            Description:
            This function fills or drops missing data in a dataframe.

            Parameters:

            self: the instance of the class that the function is a part of.
            drop_row_if_nan_in_column: (optional) the name of a column in the dataframe. If provided, the function will drop rows where this column contains NaN values. Default value is None.
            filling_dict_colmn_val: (optional) a dictionary containing column names and values to be used to fill missing data in those columns. The keys of the dictionary should be the names of the columns to be filled, and the values should be the values to use for filling. Default value is None.
            method: (optional) a string that determines the method used for filling missing data. Possible values are 'ffill' for forward filling, 'bfill' for backward filling, or 'interpolate' for linear interpolation. Default value is 'ffill'.
            column_to_fill: (optional) the name of the column to be filled. This parameter is only used when method is set to 'ffill' or 'bfill'. Default value is 'ta'.
            date_column_name: (optional) the name of the column in the dataframe that contains the date-time information. This parameter is only used when method is set to 'interpolate'. Default value is None.
            Returns:
            A dataframe with missing values filled or dropped according to the specified parameters.

            Note:
            This function assumes that the dataframe has already been loaded into the class instance. If filling_dict_colmn_val is used, the keys of the dictionary should correspond to columns in the dataframe. If method is set to 'interpolate', the date_column_name parameter must be provided. This function requires the pandas library to be installed.
        """
        if filling_dict_colmn_val is None and drop_row_if_nan_in_column is None:
            if method == 'ffill':
                self.data.get_dataframe().fillna(method='pad', inplace=True)
            elif method == 'bfill':
                self.data.get_dataframe().fillna(method='backfill', inplace=True)
       
        if filling_dict_colmn_val is not None:
            self.data.get_dataframe().fillna(filling_dict_colmn_val, inplace=True)
            
        if drop_row_if_nan_in_column is not None:
            if drop_row_if_nan_in_column == 'all':
                for p in self.data.get_columns_names():
                    self.data.set_dataframe(self.data.get_dataframe()[self.data.get_dataframe()[p].notna()])
            else:
                # a = a[~(np.isnan(a).all(axis=1))] # removes rows containing all nan
                self.data.set_dataframe(self.data.get_dataframe()[self.data.get_dataframe()[drop_row_if_nan_in_column].notna()])
                #self.__dataframe = self.__dataframe[~(np.isnan(self.__dataframe).any(axis=1))] # removes rows containing at least one nan

    def download(self, 
    variable, 
    start_date='2021-01-01',
    end_date='2021-02-01',
    product='era5_land',
    sequential_downloading=False
    ):
        """
        Downloads meteorological data for the specified variable and spatiotemporal range.

        Args:
            self (object): The instance of the class.
            variable (str): The variable to download meteorological data for.
            start_date (str): The start date of the spatiotemporal range in 'YYYY-MM-DD' format. Defaults to '2021-01-01'.
            end_date (str): The end date of the spatiotemporal range in 'YYYY-MM-DD' format. Defaults to '2021-02-01'.
            latitude (float): The latitude coordinate for the data download. Defaults to 31.66749781.
            longitude (float): The longitude coordinate for the data download. Defaults to -7.593311291.
            product (str): The product or dataset to download data from. Defaults to 'era5-Land'.
            backend (str or None): The backend to use for data retrieval. Defaults to None.

        Returns:
            None

        Notes:
            - The download method is used to retrieve meteorological data for a specific variable.
            - The variable parameter specifies the variable of interest, such as temperature, precipitation, etc.
            - The start_date and end_date parameters define the spatiotemporal range to download data for.
            - The latitude and longitude coordinates specify the location for data retrieval.
            - The product parameter identifies the specific dataset or product to download data from.
            - If the backend is not specified, the method will use the default backend associated with the class.
            - The downloaded data can be used for further analysis, processing, or visualization.
            - The availability of data and the chosen backend may affect the success of the download process.
        """
        
        self.check_directory_existance('data')
        self.check_directory_existance('data/cache')
        
        if product == 'era5_land':
            # Convert the start date and end date to datetime objects
            if not isinstance(start_date, datetime.datetime) and not isinstance(end_date, datetime.datetime):
                start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
                end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            
            if self.backend == 'gee':
                data = DataFrame()
                
                if variable == 'ta':
                    era5_land_variables = ['temperature_2m']
                    
                    output_file = 'data/era5_land_' + '_'.join([variable, str(lon), str(lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                    if os.path.exists(output_file):
                        print(f"Time series already downloaded on: {output_file}")
                    else:
                        self.download_era5_land_data_by_years(era5_land_variables, lon, lat, start_date, end_date + timedelta(1))
                        
                        for year in range(start_date.year, end_date.year + 1):
                            cache_path = 'data/cache/era5_land_' + '_'.join([str(s) for s in era5_land_variables] + [str(lon), str(lat), str(year)]) + '.csv'
                            temp_data = DataFrame(cache_path)
                            data.append_dataframe(temp_data.get_dataframe())
                            
                        if not 'first' in temp_data.get_columns_names():
                            print(f'No data found in GEE about {variable} for ({lon, lat})')
                            #cf = ClimateFiller()
                            #cf.download(variable, start_datetime, start_datetime + next_year, longitude, latitude)
                        else:
                            
                            data.rename_columns({'first': 't2m'})
                            data.transform_column('t2m', lambda o: o - 273.15)
                            data.set_dataframe(self._prepare_datetime_column(data.get_dataframe()))
                            end_date += timedelta(1)
                            data.select_datetime_range(start_date, end_date)
                            data.index_to_column()
                            #data.rename_columns({'t2m': 'ta'})
                            data.export(output_file)
                            
                            if sequential_downloading is True:
                                if self.data.is_empty():
                                    self.data.set_dataframe(data.get_dataframe())
                                else:
                                    self.data.join(data.get_dataframe())
                                    
                elif variable == 'p':
                    era5_land_variables = ['total_precipitation']
                    
                    output_file = 'data/era5_land_' + '_'.join([variable, str(lon), str(lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                    if os.path.exists(output_file):
                        print(f"Time series already downloaded on: {output_file}")
                    else:
                        self.download_era5_land_data_by_years(era5_land_variables, lon, lat, start_date, end_date + timedelta(365))
                        
                        for year in range(start_date.year, end_date.year + 1):
                            cache_path = 'data/cache/era5_land_' + '_'.join([str(s) for s in era5_land_variables] + [str(lon), str(lat), str(year)]) + '.csv'
                            temp_data = DataFrame(cache_path)
                            data.append_dataframe(temp_data.get_dataframe())
                            
                        if not 'first' in temp_data.get_columns_names():
                            print(f'No data found in GEE about {variable} for ({lon, lat})')
                            #cf = ClimateFiller()
                            #cf.download(variable, start_datetime, start_datetime + next_year, longitude, latitude)
                        else:
                            
                            data.rename_columns({'first': 'p5'})
                            data.set_dataframe(self._prepare_datetime_column(data.get_dataframe()))
                            
                            l = []
                            for p in data.get_index():
                                if p.hour == 1:
                                    new_value = data.get_row(p)['p5'] * 1000
                                else:
                                    try:
                                        previous_hour = data.get_row(p-timedelta(hours=1))['p5']
                                    except KeyError:
                                        previous_hour = data.get_row(p)['p5']
                                        
                                    new_value = (data.get_row(p)['p5'] - previous_hour)*1000
                                l.append(new_value)
                            
                            
                            data.add_column('p', l)
                            data.keep_columns(['p'])
                            end_date += timedelta(1)
                            data.select_datetime_range(start_date, end_date)
                            data.index_to_column()
                            data.export(output_file)
                            
                            if sequential_downloading is True:
                                if self.data.is_empty():
                                    self.data.set_dataframe(data.get_dataframe())
                                else:
                                    self.data.join(data.get_dataframe())
                    
                elif variable == 'rh':
                    era5_land_variables = ['temperature_2m', 'dewpoint_temperature_2m']
                    
                    output_file = 'data/era5_land_' + '_'.join([variable, str(self.lon), str(self.lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                    if os.path.exists(output_file):
                        print(f"Time series already downloaded on: {output_file}")
                    else:
                        self.download_era5_land_data_by_years(era5_land_variables, self.lon, self.lat, start_date, end_date + timedelta(1))
                        
                        for year in range(start_date.year, end_date.year + 1):
                            cache_path = 'data/cache/era5_land_' + '_'.join([str(s) for s in era5_land_variables] + [str(lon), str(lat), str(year)]) + '.csv'
                            temp_data = DataFrame(cache_path)
                            data.append_dataframe(temp_data.get_dataframe())
                            
                        if not 'temperature_2m' in temp_data.get_columns_names() and 'dewpoint_temperature_2m' in temp_data.get_columns_names():
                            print(f'No data found in GEE about {variable} for ({self.lon, self.lat})')
                            #cf = ClimateFiller()
                            #cf.download(variable, start_datetime, start_datetime + next_year, longitude, latitude)
                        else:
                            output_file = 'data/' + '_'.join([variable, str(self.lon), str(self.lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                            data.rename_columns({'temperature_2m': 't2m', 'dewpoint_temperature_2m': 'd2m'})
                            data.transform_column('t2m', lambda o: o - 273.15)
                            data.transform_column('d2m', lambda o: o - 273.15)
                            data.add_transformed_columns('era5_hr', '100*exp(-((243.12*17.62*t2m)-(d2m*17.62*t2m)-d2m*17.62*(243.12+t2m))/((243.12+t2m)*(243.12+d2m)))')
                            data.drop_columns(['t2m', 'd2m'])
                            data.set_dataframe(self._prepare_datetime_column(data.get_dataframe()))
                            end_date += timedelta(1)
                            data.select_datetime_range(start_date.isoformat(), end_date.isoformat())
                            data.export(output_file, index=True)
                            
                            if sequential_downloading is True:
                                if self.data.is_empty():
                                    self.data.set_dataframe(data.get_dataframe())
                                else:
                                    self.data.join(data.get_dataframe())

                elif variable == 'rs':
                    era5_land_variables = ['surface_solar_radiation_downwards']
                    
                    output_file = 'data/' + '_'.join([variable, str(lon), str(lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                    if os.path.exists(output_file):
                        print(f"Time series already downloaded on: {output_file}")
                    else:
                        self.download_era5_land_data_by_years(era5_land_variables, lon, lat, start_date, end_date + timedelta(1))
                        
                        for year in range(start_date.year, end_date.year + 1):
                            cache_path = 'data/cache/' + '_'.join([str(s) for s in era5_land_variables] + [str(lon), str(lat), str(year)]) + '.csv'
                            temp_data = DataFrame(cache_path)
                            data.append_dataframe(temp_data.get_dataframe())
                            
                        if not 'first' in temp_data.get_columns_names():
                            print(f'No data found in GEE about {variable} for ({lon, lat})')
                            #cf = ClimateFiller()
                            #cf.download(variable, start_datetime, start_datetime + next_year, longitude, latitude)
                        else:
                            output_file = 'data/' + '_'.join([variable, str(lon), str(lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                            data.rename_columns({'first': 'ssrd'})
                            data.set_dataframe(self._prepare_datetime_column(data.get_dataframe()))
                            
                            
                            l = []
                            for p in data.get_index():
                                if p.hour == 1:
                                    new_value = data.get_row(p)['ssrd']/3600
                                else:
                                    try:
                                        previous_hour = data.get_row(p-timedelta(hours=1))['ssrd']
                                    except KeyError:
                                        previous_hour = data.get_row(p)['ssrd']
                                        
                                    new_value = (data.get_row(p)['ssrd'] - previous_hour)/3600
                                l.append(new_value)
                            data.add_column('rs', l)
                            data.keep_columns(['rs'])
                            data.rename_columns({'rs': 'ssrd'})
                            end_date += timedelta(1)
                            data.select_datetime_range(start_date.isoformat(), end_date.isoformat())
                            data.index_to_column()
                            data.rename_columns({'ssrd': 'rs'})
                            data.export(output_file)
                            
                            
                            if sequential_downloading is True:
                                if self.data.is_empty():
                                    self.data.set_dataframe(data.get_dataframe())
                                else:
                                    self.data.join(data.get_dataframe())
                        
                elif variable == 'ws':
                    era5_land_variables = ['u_component_of_wind_10m', 'v_component_of_wind_10m']

                    output_file = 'data/' + '_'.join([variable, str(lon), str(lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                    if os.path.exists(output_file):
                        print(f"Time series already downloaded on: {output_file}")
                    else:
                        self.download_era5_land_data_by_years(era5_land_variables, lon, lat, start_date, end_date + timedelta(1))
                        
                        for year in range(start_date.year, end_date.year + 1):
                            cache_path = 'data/cache/' + '_'.join([str(s) for s in era5_land_variables] + [str(lon), str(lat), str(year)]) + '.csv'
                            temp_data = DataFrame(cache_path)
                            data.append_dataframe(temp_data.get_dataframe())
                            
                        if not 'u_component_of_wind_10m' in temp_data.get_columns_names() and 'v_component_of_wind_10m' in temp_data.get_columns_names():
                            print(f'No data found in GEE about {variable} for ({lon, lat})')
                            #cf = ClimateFiller()
                            #cf.download(variable, start_datetime, start_datetime + next_year, longitude, latitude)
                        else:
                            output_file = 'data/' + '_'.join([variable, str(lon), str(lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                            data.rename_columns({'u_component_of_wind_10m': 'u10', 'v_component_of_wind_10m': 'v10'})
                            data.add_column_based_on_function('era5_ws', Lib.get_2m_wind_speed)
                            data.drop_columns(['u10', 'v10'])
                            data.set_dataframe(self._prepare_datetime_column(data.get_dataframe()))
                            end_date += timedelta(1)
                            data.select_datetime_range(start_date.isoformat(), end_date.isoformat())
                            data.export(output_file, index=True)
                            
                            if sequential_downloading is True:
                                if self.data.is_empty():
                                    self.data.set_dataframe(data.get_dataframe())
                                else:
                                    self.data.join(data.get_dataframe())

                else:
                    
                    # era5_land_variables = ['temperature_2m', 'dewpoint_temperature_2m']
                    era5_land_variables = variable
                    
                    output_file = 'data/era5_land_' + '_'.join(variable + [str(self.lon), str(self.lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                    if os.path.exists(output_file):
                        print(f"Time series already downloaded on: {output_file}")
                    else:
                        self.download_era5_land_data_by_years(era5_land_variables, start_date, end_date + timedelta(1))
                        
                        for year in range(start_date.year, end_date.year + 1):
                            cache_path = 'data/cache/era5_land_' + '_'.join([str(s) for s in era5_land_variables] + [str(self.lon), str(self.lat), str(year)]) + '.csv'
                            temp_data = DataFrame(cache_path)
                            data.append_dataframe(temp_data.get_dataframe())
                            
                        
                        output_file = 'data/era5_land_' + '_'.join(variable + [str(self.lon), str(self.lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                        data.set_dataframe(self._prepare_datetime_column(data.get_dataframe()))
                        end_date += timedelta(1)
                        data.select_datetime_range(start_date.isoformat(), end_date.isoformat())
                        data.export(output_file, index=True)
                
            else:
                output_file = 'data/ta_' + '_'.join([str(lon), str(lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
                if os.path.exists(output_file):
                    print(f"Time series already downloaded on: {output_file}")
                else:
                    if variable == 'ta':
                        era5_land_variables = ['2m_temperature']
                    elif variable == 'rh':
                        era5_land_variables = ['2m_temperature', '2m_dewpoint_temperature']
                    elif variable == 'rs':
                        era5_land_variables = ['surface_solar_radiation_downwards']
                    elif variable == 'ws':
                        era5_land_variables = ['10m_u_component_of_wind', '10m_v_component_of_wind']
                    elif variable == 'p':
                        era5_land_variables = ['total_precipitation']
                    
                        
                    from data_science_toolkit.gis import GIS
                    import cdsapi
                    c = cdsapi.Client()
                    

                    if len(self.data.get_dataframe()) == 0:
                        # create the target time series
                        target_time_series = DataFrame.generate_datetime_range(start_date, end_date)
                        self.data.set_dataframe_index(target_time_series)
                        self.data.rename_index('datetime')
                        self.data.index_to_column()
                    
                    self.data.add_one_value_column(variable, None)
                    
                    self.fill(variable, lon, lat)
                    self.data.export(output_file, index=True)
            
        elif product == 'merra2':
            
            output_file = 'data/merra2_ta_' + '_'.join([str(lon), str(lat), str(start_date.strftime('%Y-%m-%d')), str(end_date.strftime('%Y-%m-%d'))]) + '.csv'
            if os.path.exists(output_file):
                print(f"Time series already downloaded on: {output_file}")
                data_temp = DataFrame(output_file)
                self.data_reanalysis.set_dataframe(data_temp.get_dataframe())
            else:
                merra2_variables = {
                    'ta': 'T2M',
                    'rh': 'RH2M',
                    'rs': 'ALLSKY_SFC_SW_DWN',
                    'ws': 'WS2M',
                    'pr': 'PRECTOTCORR',
                    'wd': 'WD2M'
                }

                if variable not in merra2_variables:
                    print("Variable '{}' is not supported for 'merra2'".format(variable))
                    return

                merra2_variable = merra2_variables[variable]

                api_url = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
                start = start_date.strftime('%Y%m%d')
                end = end_date.strftime('%Y%m%d')
                format = 'json'
                community = 'ag'
                timezone = 'utc'
                params = {
                    'start': start,
                    'end': end,
                    'latitude': lat,
                    'longitude': lon,
                    'community': community,
                    'parameters': merra2_variable,
                    'format': format,
                    'user': 'ysouidi1',
                    'header': 'true',
                    'time-standard': timezone
                }
                
                print(f'Downloading data for {variable} from MERRA2 API...')

                response = requests.get(api_url, params=params)

                if response.status_code != 200:
                    print('Failed to retrieve data:', response.status_code)
                    return None

                data_merra = response.json()
                result = data_merra['properties']['parameter'][merra2_variable]
                df = pd.DataFrame(result.items(), columns=['datetime', merra2_variable])
                df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H')
                self.data_reanalysis.set_dataframe(df)
                self.data_reanalysis.export(output_file, index=True)
                print(f'Downloading of {variable} from Merra2 completed.')
        
        # If other data source
        else:
            pass
    
    def export(self, path_link='data/climate_ts.csv', data_type=None, crs=None, **kwargs):
        """
        Exports the processed data to a specified file or location.

        Args:
            self (object): The instance of the class.
            path_link (str): The path or link to export the processed data. Defaults to 'data/climate_ts.csv'.
            data_type (str or None): Optional explicit export type. If omitted, the type is inferred from the file extension.
            crs (str or None): Optional coordinate reference system. If None, geospatial exports preserve the source CRS when available.

        Returns:
            None or geopandas.GeoDataFrame: For geospatial exports, a GeoDataFrame is returned.

        Notes:
            - The export method is used to save the processed data to a file or location.
            - The path_link parameter specifies the destination path or link for the exported data.
            - If data_type is not provided, the format is inferred from the destination file extension.
            - Geospatial exports preserve the source CRS by default when crs is None.
            - The processed data will be saved according to the specified file format and location.
            - The exported data can be used for further analysis, sharing, or storage.
        """
        if data_type is None:
            path_lower = str(path_link).lower()
            if path_lower.endswith('.parquet') or path_lower.endswith('.geoparquet') or path_lower.endswith('.pq') or path_lower.endswith('.pqt'):
                data_type = 'parquet'
            elif path_lower.endswith('.csv'):
                data_type = 'csv'
            elif path_lower.endswith('.json') or path_lower.endswith('.geojson'):
                data_type = 'json'
            elif path_lower.endswith('.xlsx') or path_lower.endswith('.xls'):
                data_type = 'xls'
            else:
                data_type = 'csv'

        path_lower = str(path_link).lower()
        is_geospatial_output = data_type in {'parquet', 'geoparquet', 'geojson', 'json', 'gpkg', 'shp'} or path_lower.endswith(('.parquet', '.geoparquet', '.pq', '.pqt', '.geojson', '.gpkg', '.shp'))

        export_kwargs = dict(kwargs)
        if 'index' not in export_kwargs and not is_geospatial_output:
            export_kwargs['index'] = True
        try:
            setattr(self.data, 'last_export_path', path_link)
            setattr(self.data, 'last_export_data_type', data_type)
            setattr(self.data, 'last_export_kwargs', export_kwargs)
        except Exception:
            pass

        if is_geospatial_output:
            output_dir = os.path.dirname(path_link)
            if output_dir:
                self.check_directory_existance(output_dir)

            df = self.data.get_dataframe().copy()
            lon_column = kwargs.pop('lon_column', None)
            lat_column = kwargs.pop('lat_column', None)
            gdf = self._build_geodataframe_from_dataframe(
                df,
                lon_column=lon_column,
                lat_column=lat_column,
                crs=crs,
            )

            extension = os.path.splitext(path_link)[1].lower()
            if extension in ('.parquet', '.geoparquet', '.pq', '.pqt'):
                gdf.to_parquet(path_link, index=kwargs.pop('index', False))
            elif extension in ('.geojson', '.json'):
                gdf.to_file(path_link, driver='GeoJSON')
            elif extension == '.gpkg':
                gdf.to_file(path_link, driver='GPKG')
            elif extension == '.shp':
                gdf.to_file(path_link, driver='ESRI Shapefile')
            else:
                raise ValueError(
                    f"Unsupported geospatial output format '{extension}'. "
                    "Supported formats: .geoparquet, .parquet, .geojson, .json, .gpkg, .shp"
                )

            print(f"Exported geospatial file: {os.path.abspath(path_link)}")
            return gdf

        result = self.data.export(path_link, data_type, **export_kwargs)
        print(f"Exported file: {os.path.abspath(path_link)}")
        return result
        
    def download_era5_land_data_by_years(self, variables, start_date, end_date):
        self.check_directory_existance('data')
        self.check_directory_existance('data/cache')
        point = ee.Geometry.Point(self.lon, self.lat)
        era5_land = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterBounds(point)
        
        if isinstance(start_date, str) and isinstance(end_date, str):
            # Convert the start date and end date to datetime objects
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        frequency = self._infer_frequency_label_from_index(self.data.get_dataframe().index)

        for year in range(start_date.year, end_date.year + 1):
            cache_path = self._build_era5_year_cache_path(variables, self.lon, self.lat, year, frequency=frequency)
            
            if os.path.exists(cache_path):
                print(f"Time series already downloaded on: {cache_path}")
            else:
                # Filter the ERA5 land dataset by the year's date range
                era5_land_filtered = era5_land \
                    .filterDate(str(year) + '-01-01', str(year + 1) + '-01-01') \
                    .select(variables)

                # Download or perform further processing for the data for each year
                # Convert the image collection to a feature collection
                feature_collection = era5_land_filtered.map(lambda image: image.reduceRegions(reducer=ee.Reducer.first(), collection=ee.FeatureCollection(point)))

                # Flatten the feature collection
                flattened_collection = feature_collection.flatten()

                task = ee.batch.Export.table.toDrive( 
                    collection=flattened_collection,
                    description='ERA5_Land_Data',
                    fileFormat='CSV',
                    folder = 'era5_land_data'
                )
                task.start()
                
                # Export the TS to Loccal from Google Drive
                geemap.ee_export_vector(flattened_collection , filename=cache_path)
                temp_data = DataFrame(cache_path)
                temp_data.rename_columns({'system:index': 'datetime'})
                temp_data.column_to_date('datetime', extraction_func=self.extract_datetime)
                temp_data.export(cache_path, index=True)
                
    def download_era5_land_data_by_months(self, variables, lon, lat, start_date, end_date):
        point = ee.Geometry.Point(lon, lat)
        era5_land = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterBounds(point)
        
        if isinstance(start_date, str) and isinstance(end_date, str):
            # Convert the start date and end date to datetime objects
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        for year in range(start_date.year, end_date.year):
            cache_path = 'data/cache/era5_land_' + '_'.join([str(s) for s in variables] + [str(lon), str(lat), str(year)]) + '.csv'
            
            if os.path.exists(cache_path):
                print(f"Time series already downloaded on: {cache_path}")
            else:
                # Filter the ERA5 land dataset by the year's date range
                era5_land_filtered = era5_land \
                    .filterDate(str(year) + '-01-01', str(year + 1) + '-01-01') \
                    .select(variables)

                # Download or perform further processing for the data for each year
                # Convert the image collection to a feature collection
                feature_collection = era5_land_filtered.map(lambda image: image.reduceRegions(reducer=ee.Reducer.first(), collection=ee.FeatureCollection(point)))

                # Flatten the feature collection
                flattened_collection = feature_collection.flatten()

                task = ee.batch.Export.table.toDrive( 
                    collection=flattened_collection,
                    description='ERA5_Land_Data',
                    fileFormat='CSV',
                    folder = 'era5_land_data'
                )
                task.start()
                
                # Export the TS to Loccal from Google Drive
                geemap.ee_export_vector(flattened_collection , filename=cache_path)
                temp_data = DataFrame(cache_path)
                temp_data.rename_columns({'system:index': 'datetime'})
                temp_data.column_to_date('datetime', extraction_func=self.extract_datetime)
                temp_data.export(cache_path, index=True)
    
    
    @staticmethod
    def extract_datetime(row):
        """
        Extracts the datetime value from a row of data.

        Args:
            row (object): A row of data from which the datetime value will be extracted.

        Returns:
            datetime: The extracted datetime value.

        Notes:
            - The extract_datetime static method is used to extract the datetime value from a row of data.
            - The row parameter represents a single row of data, which can be an object or a dictionary-like structure.
            - The method extracts and returns the datetime value from the specified row.
            - The datetime value is typically used for time-based operations, analysis, or visualization.
        """
        

        if isinstance(row, datetime.datetime):
            return row
        else:
            # Example string
            date_string = row

            # Define the regular expression pattern
            pattern = r'(\d{4})(\d{2})(\d{2})T(\d{2})_(\d{1})'

            # Match and extract the date components using the pattern
            match = re.match(pattern, date_string)

            # Extract the date components from the match
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            hour = int(match.group(4))
            minute = int(match.group(5))
            return datetime.datetime(year, month, day, hour, minute)
        
    def check_directory_existance(self, directory_path='data'):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory created: {directory_path}")
            
    def watt_to_megaj_per_hour(self, column_name='rs'):
        self.data.transform_column(column_name, lambda o: o * 0.0036)
        
      
    def climate_zones_classification(self,):
        """
        Classifies the climate zones based on temperature, precipitation and evapotranspration data.

        Args:
            self (object): The instance of the class.

        Returns:
            None

        Notes:
            - The climate_zones_classification method is used to classify the climate zones based on temperature and precipitation data.
            - The method iterates over the data and assigns a climate zone based on the temperature and precipitation values.
            - The classified climate zones are stored in a new column named 'climate_zone'.
            - The method assumes that the temperature and precipitation data are available in the data.
        """
        self.data.index_to_column()
        self.data.column_to_date(self.datetime_column_name)

        # DataFrame to store results
        results = []
        self.data.add_year_column()
        # Calculate indices for each year
        station_data = self.data.dataframe.copy()

        yearly_results = []

        for year in station_data['year'].unique():
            yearly_data = station_data[station_data['year'] == year]
            # reindex yearly_data using datetime column
            yearly_data.set_index('datetime', inplace=True)
            if yearly_data.shape[0] == 365 or yearly_data.shape[0] == 366:
                annual_precip = yearly_data['p'].sum()
                annual_temp = yearly_data['ta'].mean()
                annual_pet = yearly_data['et0_pm'].sum()
                
                tsi = Lib.temperature_seasonality_index(yearly_data['ta'])
                psi = Lib.precipitation_seasonality_index(yearly_data['p'])
                tmi = Lib.thornthwaite_moisture_index(annual_precip, annual_pet)
                ai = Lib.aridity_index(annual_precip, annual_pet)
                #print(yearly_data['ta'])
                #kg_classification = Lib.classify_koppen_geiger(yearly_data['ta'].tolist(), yearly_data['p'].tolist(), annual_precip, annual_temp)
                kg_classification = Lib.classify_koppen_geiger_daily(yearly_data['ta'], yearly_data['p'])
                yearly_results.append({
                    'Year': year,
                    'Temperature Seasonality Index (TSI)': tsi,
                    'Precipitation Seasonality Index (PSI)': psi,
                    'Thornthwaite Moisture Index (TMI)': tmi,
                    'Aridity Index (AI)': ai,
                    'Köppen-Geiger Classification': kg_classification,
                    'Annual Potential Evapotranspiration (mm)': annual_pet,
                })
            
            else:
                # drop the year from the dataframe if it is not complete
                station_data = station_data[station_data['year'] != year]
                

        # Calculate average indices over the years
        yearly_df = pd.DataFrame(yearly_results)
        avg_tsi = yearly_df['Temperature Seasonality Index (TSI)'].mean()
        avg_psi = yearly_df['Precipitation Seasonality Index (PSI)'].mean()
        avg_tmi = yearly_df['Thornthwaite Moisture Index (TMI)'].mean()
        avg_ai = yearly_df['Aridity Index (AI)'].mean()
        print(yearly_df)
        results.append({
            'Most Frequent Köppen-Geiger Classification': yearly_df['Köppen-Geiger Classification'].mode()[0],  # Most frequent classification
            'Average Thornthwaite Moisture Index (TMI)': np.round(avg_tmi , 2),
            'Average Aridity Index (AI)': np.round(avg_ai , 2),
            'Average Temperature Seasonality Index (TSI)': np.round(avg_tsi , 2),
            'Average Preicipitaton Seasonality Index (PSI)': np.round(avg_psi , 2),
            'Mean Annual Temperature (°C)': np.round(station_data.groupby('year')['ta'].mean().mean(), 2),
            'Total Annual Precipitation (mm)': np.round(station_data.groupby('year')['p'].sum().mean(), 2),
            'Annual Reference Evapotranspiration (mm)': np.round(station_data.groupby('year')['et0_pm'].sum().mean(), 2),
        })

        print(results)
        results_df = pd.DataFrame(yearly_df)
        
        return results_df, results
    
    
    @staticmethod
    def _load_noaa_country_names(timeout=60):
        """Return sorted NOAA country names from country-list.txt (uppercase)."""
        url = "https://www.ncei.noaa.gov/pub/data/noaa/country-list.txt"
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        names = []
        for line in resp.text.splitlines():
            raw = line.rstrip("\n")
            if not raw or raw.upper().startswith("FIPS"):
                continue
            # Format: "AA          ARUBA"
            parts = raw.split(None, 1)
            if len(parts) < 2:
                continue
            name = parts[1].strip().upper()
            if name:
                names.append(name)
        return sorted(set(names))

    @classmethod
    def _infer_noaa_countries_from_roi(cls, roi, bbox=None, roi_geometry=None, timeout=60, verbose=False):
        """
        Infer NOAA ``nearest_stations_noaa`` country name(s) from an ROI.

        Intersects the ROI with Natural Earth admin-0 countries and maps names
        onto NOAA's country-list.txt (exact uppercase match, then aliases).
        """
        from shapely.geometry import box as shapely_box

        if bbox is None:
            bbox = cls._roi_to_bbox_list(roi)
        if bbox is None:
            raise ValueError("Could not resolve bbox from roi for country inference.")
        min_lon, min_lat, max_lon, max_lat = bbox

        if roi_geometry is None:
            if isinstance(roi, str):
                ext = os.path.splitext(roi)[1].lower()
                if ext in (".parquet", ".geoparquet"):
                    roi_gdf = gpd.read_parquet(roi)
                else:
                    roi_gdf = gpd.read_file(roi)
                if roi_gdf.crs is None or roi_gdf.crs.to_epsg() != 4326:
                    roi_gdf = roi_gdf.to_crs(epsg=4326)
                roi_geometry = roi_gdf.unary_union
            else:
                roi_geometry = shapely_box(min_lon, min_lat, max_lon, max_lat)

        # Natural Earth 110m countries (small download, cached by GDAL/fiona where possible)
        ne_url = (
            "https://naciscdn.org/naturalearth/110m/cultural/"
            "ne_110m_admin_0_countries.zip"
        )
        try:
            world = gpd.read_file(ne_url)
        except Exception as exc:
            raise RuntimeError(
                "Could not load Natural Earth countries to infer country from ROI. "
                f"Details: {exc}"
            ) from exc

        if world.crs is None or world.crs.to_epsg() != 4326:
            world = world.to_crs(epsg=4326)

        name_col = None
        for cand in ("NAME", "ADMIN", "NAME_EN", "name", "admin"):
            if cand in world.columns:
                name_col = cand
                break
        if name_col is None:
            raise RuntimeError(
                "Natural Earth countries layer has no recognizable country-name column."
            )

        try:
            hit = world[world.intersects(roi_geometry)].copy()
        except Exception:
            # Fallback for invalid/self-intersecting ROI geometries
            hit = world[world.intersects(roi_geometry.buffer(0))].copy()

        if hit.empty:
            # Last resort: country containing ROI center
            center = shapely_box(min_lon, min_lat, max_lon, max_lat).centroid
            hit = world[world.contains(center)].copy()
        if hit.empty:
            raise RuntimeError(
                "Could not infer any country from ROI. "
                "Pass a clearer ROI or check that it intersects land."
            )

        noaa_names = cls._load_noaa_country_names(timeout=timeout)
        noaa_set = set(noaa_names)

        # Common Natural Earth → NOAA name differences
        aliases = {
            "UNITED STATES OF AMERICA": "UNITED STATES",
            "UNITED STATES OF AMERICA (THE)": "UNITED STATES",
            "RUSSIAN FEDERATION": "RUSSIA",
            "RUSSIA": "RUSSIA",
            "SOUTH KOREA": "KOREA SOUTH",
            "NORTH KOREA": "KOREA NORTH",
            "REPUBLIC OF KOREA": "KOREA SOUTH",
            "DEMOCRATIC PEOPLE'S REPUBLIC OF KOREA": "KOREA NORTH",
            "CZECHIA": "CZECH REPUBLIC",
            "ESWATINI": "SWAZILAND",
            "NORTH MACEDONIA": "MACEDONIA",
            "MYANMAR": "BURMA",
            "VIET NAM": "VIETNAM",
            "SYRIA": "SYRIA",
            "SYRIAN ARAB REPUBLIC": "SYRIA",
            "IRAN": "IRAN",
            "IRAN (ISLAMIC REPUBLIC OF)": "IRAN",
            "TANZANIA": "TANZANIA",
            "UNITED REPUBLIC OF TANZANIA": "TANZANIA",
            "BOLIVIA": "BOLIVIA",
            "BOLIVIA (PLURINATIONAL STATE OF)": "BOLIVIA",
            "VENEZUELA": "VENEZUELA",
            "VENEZUELA (BOLIVARIAN REPUBLIC OF)": "VENEZUELA",
            "MOLDOVA": "MOLDOVA",
            "REPUBLIC OF MOLDOVA": "MOLDOVA",
            "LAOS": "LAOS",
            "LAO PEOPLE'S DEMOCRATIC REPUBLIC": "LAOS",
            "BRUNEI": "BRUNEI",
            "BRUNEI DARUSSALAM": "BRUNEI",
            "DEMOCRATIC REPUBLIC OF THE CONGO": "CONGO DEMOCRATIC REPUBLIC",
            "DEMOCRATIC REPUBLIC OF CONGO": "CONGO DEMOCRATIC REPUBLIC",
            "CONGO": "CONGO",
            "REPUBLIC OF THE CONGO": "CONGO",
            "IVORY COAST": "COTE D'IVOIRE",
            "CÔTE D'IVOIRE": "COTE D'IVOIRE",
            "COTE D'IVOIRE": "COTE D'IVOIRE",
            "CAPE VERDE": "CAPE VERDE",
            "CABO VERDE": "CAPE VERDE",
            "GAMBIA": "GAMBIA THE",
            "THE GAMBIA": "GAMBIA THE",
            "BAHAMAS": "BAHAMAS THE",
            "THE BAHAMAS": "BAHAMAS THE",
            "UNITED KINGDOM": "UNITED KINGDOM",
            "UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND": "UNITED KINGDOM",
            "CENTRAL AFRICAN REPUBLIC": "CENTRAL AFRICAN REPUBLIC",
            "SOUTH SUDAN": "SOUTH SUDAN",
            "EQ. GUINEA": "EQUATORIAL GUINEA",
            "EQUATORIAL GUINEA": "EQUATORIAL GUINEA",
            "W. SAHARA": "WESTERN SAHARA",
            "WESTERN SAHARA": "WESTERN SAHARA",
            "S. SUDAN": "SOUTH SUDAN",
            "BOSNIA AND HERZ.": "BOSNIA AND HERZEGOVINA",
            "BOSNIA AND HERZEGOVINA": "BOSNIA AND HERZEGOVINA",
            "DOMINICAN REP.": "DOMINICAN REPUBLIC",
            "SOLOMON IS.": "SOLOMON ISLANDS",
            "N. CYPRUS": "CYPRUS",
            "E. TIMOR": "EAST TIMOR",
            "TIMOR-LESTE": "EAST TIMOR",
            "SÃO TOMÉ AND PRINCIPE": "SAO TOME AND PRINCIPE",
            "SAO TOME AND PRINCIPE": "SAO TOME AND PRINCIPE",
        }

        def _to_noaa(name):
            key = str(name).strip().upper()
            if not key:
                return None
            if key in noaa_set:
                return key
            mapped = aliases.get(key)
            if mapped and mapped in noaa_set:
                return mapped
            # Loose contains match (prefer shortest NOAA name that contains / is contained)
            candidates = [
                n for n in noaa_names
                if key in n or n in key
            ]
            if candidates:
                return sorted(candidates, key=len)[0]
            return None

        inferred = []
        unmatched = []
        for raw_name in hit[name_col].astype(str).tolist():
            mapped = _to_noaa(raw_name)
            if mapped:
                inferred.append(mapped)
            else:
                unmatched.append(raw_name)

        inferred = sorted(set(inferred))
        if not inferred:
            raise RuntimeError(
                "ROI intersects land but no Natural Earth country name could be "
                "mapped to NOAA country-list.txt. Unmatched: "
                f"{sorted(set(unmatched))}"
            )

        if verbose:
            print(
                f"[list_in_situ_stations] Inferred {len(inferred)} NOAA country(ies) "
                f"from ROI: {', '.join(inferred)}"
            )
            if unmatched:
                print(
                    "[list_in_situ_stations] Warning: unmatched Natural Earth names "
                    f"(skipped): {sorted(set(unmatched))}"
                )
        return inferred

    def list_in_situ_stations(
        self,
        roi,
        start_date,
        end_date,
        *,
        no_of_stations=None,
        timeout=120,
        export=False,
        output_file="data/in_situ/climate_station_report.txt",
        verbose=True,
    ):
        """
        List available NOAA ISD in-situ stations for an ROI and date range.

        Downloads NOAA ``isd-history.csv`` (same catalog used by R
        ``climate::nearest_stations_noaa``), filters stations by ROI
        (bbox and optional polygon) and by operation-period overlap with
        ``start_date``/``end_date``. No R / Rscript installation is required.

        Countries intersecting the ROI are inferred for metadata only; spatial
        filtering is done directly against the ROI.

        Parameters
        ----------
        roi : list | tuple | str
            Bounding box [min_lon, min_lat, max_lon, max_lat] or vector file path.
        start_date : str
            Start date (YYYY-MM-DD) used for availability overlap filtering.
        end_date : str
            End date (YYYY-MM-DD) used for availability overlap filtering.
        no_of_stations : int | None, optional
            If set, keep only the N stations nearest to the ROI center.
            If None (default), keep all stations inside the ROI.
        timeout : int | float, optional
            HTTP timeout in seconds for NOAA catalog downloads.
        export : bool, optional
            If True, export a text report to ``output_file`` and a GeoParquet
            stations file alongside it (``*_stations.parquet``, EPSG:4326).
        output_file : str, optional
            Text report path used when ``export=True``. The stations GeoParquet
            is written with the same base name and ``_stations.parquet`` suffix.
        verbose : bool, optional
            Print progress and summary logs.

        Returns
        -------
        dict
            {
              "source": "NOAA-ISD",
              "countries": list[str],
              "bbox": [min_lon, min_lat, max_lon, max_lat],
              "start_date": str,
              "end_date": str,
              "stations_count": int,
              "stations": list[dict],
              "report_file": str | None,
              "stations_parquet_file": str | None,
            }
        """
        from io import StringIO

        if roi is None:
            raise ValueError("roi is required and must be a bbox or vector file path.")
        if not start_date or not end_date:
            raise ValueError("start_date and end_date are required.")
        if no_of_stations is not None and int(no_of_stations) < 1:
            raise ValueError("no_of_stations must be >= 1 when provided.")

        bbox = self._roi_to_bbox_list(roi)
        if bbox is None:
            raise ValueError("Could not resolve bbox from roi.")
        min_lon, min_lat, max_lon, max_lat = bbox
        center_lon = (min_lon + max_lon) / 2.0
        center_lat = (min_lat + max_lat) / 2.0

        start_dt = pd.to_datetime(start_date, errors="coerce")
        end_dt = pd.to_datetime(end_date, errors="coerce")
        if pd.isna(start_dt) or pd.isna(end_dt):
            raise ValueError("start_date and end_date must be valid dates in YYYY-MM-DD format.")
        if start_dt > end_dt:
            raise ValueError("start_date must be <= end_date.")

        roi_geometry = None
        if isinstance(roi, str):
            try:
                ext = os.path.splitext(roi)[1].lower()
                if ext in (".parquet", ".geoparquet"):
                    roi_gdf = gpd.read_parquet(roi)
                else:
                    roi_gdf = gpd.read_file(roi)
                if roi_gdf.crs is None or roi_gdf.crs.to_epsg() != 4326:
                    roi_gdf = roi_gdf.to_crs(epsg=4326)
                roi_geometry = roi_gdf.unary_union
            except Exception as exc:
                raise RuntimeError(
                    f"Could not read roi vector geometry for polygon filtering: {exc}"
                ) from exc

        countries = []
        try:
            countries = self._infer_noaa_countries_from_roi(
                roi=roi,
                bbox=bbox,
                roi_geometry=roi_geometry,
                timeout=timeout,
                verbose=verbose,
            )
        except Exception as exc:
            if verbose:
                print(
                    f"[list_in_situ_stations] Warning: could not infer countries "
                    f"from ROI ({exc}). Continuing with spatial filter only."
                )

        if verbose:
            print("[list_in_situ_stations] Downloading NOAA ISD station catalog ...")

        isd_url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
        try:
            resp = requests.get(isd_url, timeout=timeout)
            resp.raise_for_status()
            stations_df = pd.read_csv(StringIO(resp.text))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download NOAA ISD history catalog from {isd_url}: {exc}"
            ) from exc

        if stations_df.empty:
            if verbose:
                print("[list_in_situ_stations] NOAA ISD catalog is empty.")
            return {
                "source": "NOAA-ISD",
                "countries": countries,
                "bbox": bbox,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "stations_count": 0,
                "stations": [],
                "report_file": None,
                "stations_parquet_file": None,
            }

        # Attach human-readable country names from NOAA country-list when possible.
        try:
            cl_url = "https://www.ncei.noaa.gov/pub/data/noaa/country-list.txt"
            cl_resp = requests.get(cl_url, timeout=timeout)
            cl_resp.raise_for_status()
            rows = []
            for line in cl_resp.text.splitlines():
                raw = line.rstrip("\n")
                if not raw or raw.upper().startswith("FIPS"):
                    continue
                parts = raw.split(None, 1)
                if len(parts) < 2:
                    continue
                rows.append({"CTRY": parts[0].strip(), "COUNTRY": parts[1].strip().upper()})
            country_df = pd.DataFrame(rows)
            if "CTRY" in stations_df.columns and not country_df.empty:
                stations_df = stations_df.merge(country_df, on="CTRY", how="left")
        except Exception as exc:
            if verbose:
                print(
                    f"[list_in_situ_stations] Warning: could not join NOAA country "
                    f"names ({exc})."
                )

        columns_lower = {c.lower(): c for c in stations_df.columns}
        lon_col = columns_lower.get("lon")
        lat_col = columns_lower.get("lat")
        begin_col = columns_lower.get("begin")
        end_col = columns_lower.get("end")

        if lon_col is None or lat_col is None:
            raise RuntimeError(
                "NOAA ISD catalog is missing coordinate columns (expected LON/LAT). "
                f"Columns: {list(stations_df.columns)}"
            )

        stations_df[lon_col] = pd.to_numeric(stations_df[lon_col], errors="coerce")
        stations_df[lat_col] = pd.to_numeric(stations_df[lat_col], errors="coerce")
        stations_df = stations_df.dropna(subset=[lon_col, lat_col]).copy()

        # Filter by bbox first.
        stations_df = stations_df[
            (stations_df[lon_col] >= min_lon)
            & (stations_df[lon_col] <= max_lon)
            & (stations_df[lat_col] >= min_lat)
            & (stations_df[lat_col] <= max_lat)
        ].copy()

        # Optional polygon clipping when roi is vector geometry.
        if roi_geometry is not None and not stations_df.empty:
            points_gdf = gpd.GeoDataFrame(
                stations_df,
                geometry=gpd.points_from_xy(
                    stations_df[lon_col], stations_df[lat_col], crs="EPSG:4326"
                ),
                crs="EPSG:4326",
            )
            inside_mask = points_gdf.intersects(roi_geometry)
            stations_df = points_gdf.loc[inside_mask].drop(columns=["geometry"]).copy()

        # Filter by station operation period overlap with requested range.
        if begin_col is not None and end_col is not None and not stations_df.empty:
            begin_dt = pd.to_datetime(
                stations_df[begin_col].astype(str), format="%Y%m%d", errors="coerce"
            )
            end_dt_col = pd.to_datetime(
                stations_df[end_col].astype(str), format="%Y%m%d", errors="coerce"
            )
            # Fallback for already-ISO strings
            bad_begin = begin_dt.isna() & stations_df[begin_col].notna()
            bad_end = end_dt_col.isna() & stations_df[end_col].notna()
            if bad_begin.any():
                begin_dt.loc[bad_begin] = pd.to_datetime(
                    stations_df.loc[bad_begin, begin_col], errors="coerce"
                )
            if bad_end.any():
                end_dt_col.loc[bad_end] = pd.to_datetime(
                    stations_df.loc[bad_end, end_col], errors="coerce"
                )

            overlap_mask = (
                begin_dt.notna()
                & end_dt_col.notna()
                & (begin_dt <= end_dt)
                & (end_dt_col >= start_dt)
            )
            stations_df = stations_df.loc[overlap_mask].copy()

        # Optional nearest-N selection relative to ROI center.
        if no_of_stations is not None and not stations_df.empty:
            # Approximate km distance (same scale factor used by climate R package).
            dist = (
                ((stations_df[lon_col] - center_lon) ** 2)
                + ((stations_df[lat_col] - center_lat) ** 2)
            ) ** 0.5 * 112.196672
            stations_df = stations_df.assign(distance_km=dist)
            stations_df = (
                stations_df.sort_values("distance_km", ascending=True)
                .head(int(no_of_stations))
                .copy()
            )

        stations_df = stations_df.reset_index(drop=True)
        stations_records = stations_df.to_dict(orient="records")

        report_file = None
        stations_parquet_file = None
        if export:
            folder = os.path.dirname(output_file)
            if folder:
                os.makedirs(folder, exist_ok=True)

            if hasattr(datetime, "datetime"):
                timestamp_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                timestamp_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

            name_col = columns_lower.get("station name") or columns_lower.get("name")
            usaf_col = columns_lower.get("usaf")
            wban_col = columns_lower.get("wban")
            ctry_col = columns_lower.get("country") or columns_lower.get("ctry")
            begin_name = begin_col
            end_name = end_col

            lines = [
                "NOAA ISD In-Situ Station Discovery Report",
                "=" * 42,
                f"Generated: {timestamp_utc}",
                "Source: NOAA ISD history (isd-history.csv)",
                f"BBox [min_lon,min_lat,max_lon,max_lat]: {bbox}",
                f"Date range: {start_date} -> {end_date}",
                f"Countries inferred from ROI: {', '.join(countries) if countries else 'none'}",
                f"Stations found: {len(stations_records)}",
                "",
                "Available Stations:",
            ]
            if stations_records:
                for row in stations_records:
                    usaf = row.get(usaf_col) if usaf_col else None
                    wban = row.get(wban_col) if wban_col else None
                    sname = row.get(name_col) if name_col else None
                    lat = row.get(lat_col)
                    lon = row.get(lon_col)
                    ctry = row.get(ctry_col) if ctry_col else None
                    begin_v = row.get(begin_name) if begin_name else None
                    end_v = row.get(end_name) if end_name else None
                    sid = f"{usaf}-{wban}" if usaf is not None or wban is not None else "unknown"
                    lines.append(
                        f"- {sid} | name={sname} | country={ctry} | "
                        f"lat={lat} | lon={lon} | begin={begin_v} | end={end_v}"
                    )
            else:
                lines.append("- None")

            with open(output_file, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            report_file = output_file

            parquet_root, _ = os.path.splitext(output_file)
            stations_parquet_file = f"{parquet_root}_stations.parquet"
            try:
                stations_gdf = gpd.GeoDataFrame(
                    stations_df.copy(),
                    geometry=gpd.points_from_xy(
                        stations_df[lon_col],
                        stations_df[lat_col],
                        crs="EPSG:4326",
                    ),
                    crs="EPSG:4326",
                )
                stations_gdf.to_parquet(stations_parquet_file, index=False)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to export in-situ stations parquet: {stations_parquet_file} ({exc})"
                ) from exc

        if verbose:
            print(f"[list_in_situ_stations] Stations found: {len(stations_records)}")
            if report_file:
                print(f"[list_in_situ_stations] Report exported to: {report_file}")
            if stations_parquet_file:
                print(
                    f"[list_in_situ_stations] Stations parquet exported to: "
                    f"{stations_parquet_file}"
                )

        return {
            "source": "NOAA-ISD",
            "countries": countries,
            "bbox": bbox,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "stations_count": len(stations_records),
            "stations": stations_records,
            "report_file": report_file,
            "stations_parquet_file": stations_parquet_file,
        }


    @staticmethod
    def _roi_to_bbox_list(roi):
        """Return [min_lon, min_lat, max_lon, max_lat] from a bbox list/tuple or vector file path.
        Returns None when roi is None (no spatial filter)."""
        if roi is None:
            return None
        if isinstance(roi, (list, tuple)) and len(roi) == 4 and all(isinstance(v, (int, float)) for v in roi):
            return list(roi)
        if isinstance(roi, str):
            import geopandas as gpd
            ext = os.path.splitext(roi)[1].lower()
            if ext in (".parquet", ".geoparquet"):
                gdf = gpd.read_parquet(roi)
            else:
                gdf = gpd.read_file(roi)
            if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            minx, miny, maxx, maxy = gdf.total_bounds
            return [float(minx), float(miny), float(maxx), float(maxy)]
        raise TypeError(f"roi must be a bbox list/tuple or a file path string, got {type(roi)}")

    @staticmethod
    def _load_noaa_country_names(timeout=60):
        """Return sorted NOAA country names from country-list.txt (uppercase)."""
        url = "https://www.ncei.noaa.gov/pub/data/noaa/country-list.txt"
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        names = []
        for line in resp.text.splitlines():
            raw = line.rstrip("\n")
            if not raw or raw.upper().startswith("FIPS"):
                continue
            # Format: "AA          ARUBA"
            parts = raw.split(None, 1)
            if len(parts) < 2:
                continue
            name = parts[1].strip().upper()
            if name:
                names.append(name)
        return sorted(set(names))

    @classmethod
    def _infer_noaa_countries_from_roi(cls, roi, bbox=None, roi_geometry=None, timeout=60, verbose=False):
        """
        Infer NOAA ``nearest_stations_noaa`` country name(s) from an ROI.

        Intersects the ROI with Natural Earth admin-0 countries and maps names
        onto NOAA's country-list.txt (exact uppercase match, then aliases).
        """
        from shapely.geometry import box as shapely_box

        if bbox is None:
            bbox = cls._roi_to_bbox_list(roi)
        if bbox is None:
            raise ValueError("Could not resolve bbox from roi for country inference.")
        min_lon, min_lat, max_lon, max_lat = bbox

        if roi_geometry is None:
            if isinstance(roi, str):
                ext = os.path.splitext(roi)[1].lower()
                if ext in (".parquet", ".geoparquet"):
                    roi_gdf = gpd.read_parquet(roi)
                else:
                    roi_gdf = gpd.read_file(roi)
                if roi_gdf.crs is None or roi_gdf.crs.to_epsg() != 4326:
                    roi_gdf = roi_gdf.to_crs(epsg=4326)
                roi_geometry = roi_gdf.unary_union
            else:
                roi_geometry = shapely_box(min_lon, min_lat, max_lon, max_lat)

        # Natural Earth 110m countries (small download, cached by GDAL/fiona where possible)
        ne_url = (
            "https://naciscdn.org/naturalearth/110m/cultural/"
            "ne_110m_admin_0_countries.zip"
        )
        try:
            world = gpd.read_file(ne_url)
        except Exception as exc:
            raise RuntimeError(
                "Could not load Natural Earth countries to infer country from ROI. "
                f"Details: {exc}"
            ) from exc

        if world.crs is None or world.crs.to_epsg() != 4326:
            world = world.to_crs(epsg=4326)

        name_col = None
        for cand in ("NAME", "ADMIN", "NAME_EN", "name", "admin"):
            if cand in world.columns:
                name_col = cand
                break
        if name_col is None:
            raise RuntimeError(
                "Natural Earth countries layer has no recognizable country-name column."
            )

        try:
            hit = world[world.intersects(roi_geometry)].copy()
        except Exception:
            # Fallback for invalid/self-intersecting ROI geometries
            hit = world[world.intersects(roi_geometry.buffer(0))].copy()

        if hit.empty:
            # Last resort: country containing ROI center
            center = shapely_box(min_lon, min_lat, max_lon, max_lat).centroid
            hit = world[world.contains(center)].copy()
        if hit.empty:
            raise RuntimeError(
                "Could not infer any country from ROI. "
                "Pass a clearer ROI or check that it intersects land."
            )

        noaa_names = cls._load_noaa_country_names(timeout=timeout)
        noaa_set = set(noaa_names)

        # Common Natural Earth → NOAA name differences
        aliases = {
            "UNITED STATES OF AMERICA": "UNITED STATES",
            "UNITED STATES OF AMERICA (THE)": "UNITED STATES",
            "RUSSIAN FEDERATION": "RUSSIA",
            "RUSSIA": "RUSSIA",
            "SOUTH KOREA": "KOREA SOUTH",
            "NORTH KOREA": "KOREA NORTH",
            "REPUBLIC OF KOREA": "KOREA SOUTH",
            "DEMOCRATIC PEOPLE'S REPUBLIC OF KOREA": "KOREA NORTH",
            "CZECHIA": "CZECH REPUBLIC",
            "ESWATINI": "SWAZILAND",
            "NORTH MACEDONIA": "MACEDONIA",
            "MYANMAR": "BURMA",
            "VIET NAM": "VIETNAM",
            "SYRIA": "SYRIA",
            "SYRIAN ARAB REPUBLIC": "SYRIA",
            "IRAN": "IRAN",
            "IRAN (ISLAMIC REPUBLIC OF)": "IRAN",
            "TANZANIA": "TANZANIA",
            "UNITED REPUBLIC OF TANZANIA": "TANZANIA",
            "BOLIVIA": "BOLIVIA",
            "BOLIVIA (PLURINATIONAL STATE OF)": "BOLIVIA",
            "VENEZUELA": "VENEZUELA",
            "VENEZUELA (BOLIVARIAN REPUBLIC OF)": "VENEZUELA",
            "MOLDOVA": "MOLDOVA",
            "REPUBLIC OF MOLDOVA": "MOLDOVA",
            "LAOS": "LAOS",
            "LAO PEOPLE'S DEMOCRATIC REPUBLIC": "LAOS",
            "BRUNEI": "BRUNEI",
            "BRUNEI DARUSSALAM": "BRUNEI",
            "DEMOCRATIC REPUBLIC OF THE CONGO": "CONGO DEMOCRATIC REPUBLIC",
            "DEMOCRATIC REPUBLIC OF CONGO": "CONGO DEMOCRATIC REPUBLIC",
            "CONGO": "CONGO",
            "REPUBLIC OF THE CONGO": "CONGO",
            "IVORY COAST": "COTE D'IVOIRE",
            "CÔTE D'IVOIRE": "COTE D'IVOIRE",
            "COTE D'IVOIRE": "COTE D'IVOIRE",
            "CAPE VERDE": "CAPE VERDE",
            "CABO VERDE": "CAPE VERDE",
            "GAMBIA": "GAMBIA THE",
            "THE GAMBIA": "GAMBIA THE",
            "BAHAMAS": "BAHAMAS THE",
            "THE BAHAMAS": "BAHAMAS THE",
            "UNITED KINGDOM": "UNITED KINGDOM",
            "UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND": "UNITED KINGDOM",
            "CENTRAL AFRICAN REPUBLIC": "CENTRAL AFRICAN REPUBLIC",
            "SOUTH SUDAN": "SOUTH SUDAN",
            "EQ. GUINEA": "EQUATORIAL GUINEA",
            "EQUATORIAL GUINEA": "EQUATORIAL GUINEA",
            "W. SAHARA": "WESTERN SAHARA",
            "WESTERN SAHARA": "WESTERN SAHARA",
            "S. SUDAN": "SOUTH SUDAN",
            "BOSNIA AND HERZ.": "BOSNIA AND HERZEGOVINA",
            "BOSNIA AND HERZEGOVINA": "BOSNIA AND HERZEGOVINA",
            "DOMINICAN REP.": "DOMINICAN REPUBLIC",
            "SOLOMON IS.": "SOLOMON ISLANDS",
            "N. CYPRUS": "CYPRUS",
            "E. TIMOR": "EAST TIMOR",
            "TIMOR-LESTE": "EAST TIMOR",
            "SÃO TOMÉ AND PRINCIPE": "SAO TOME AND PRINCIPE",
            "SAO TOME AND PRINCIPE": "SAO TOME AND PRINCIPE",
        }

        def _to_noaa(name):
            key = str(name).strip().upper()
            if not key:
                return None
            if key in noaa_set:
                return key
            mapped = aliases.get(key)
            if mapped and mapped in noaa_set:
                return mapped
            # Loose contains match (prefer shortest NOAA name that contains / is contained)
            candidates = [
                n for n in noaa_names
                if key in n or n in key
            ]
            if candidates:
                return sorted(candidates, key=len)[0]
            return None

        inferred = []
        unmatched = []
        for raw_name in hit[name_col].astype(str).tolist():
            mapped = _to_noaa(raw_name)
            if mapped:
                inferred.append(mapped)
            else:
                unmatched.append(raw_name)

        inferred = sorted(set(inferred))
        if not inferred:
            raise RuntimeError(
                "ROI intersects land but no Natural Earth country name could be "
                "mapped to NOAA country-list.txt. Unmatched: "
                f"{sorted(set(unmatched))}"
            )

        if verbose:
            print(
                f"[list_in_situ_stations] Inferred {len(inferred)} NOAA country(ies) "
                f"from ROI: {', '.join(inferred)}"
            )
            if unmatched:
                print(
                    "[list_in_situ_stations] Warning: unmatched Natural Earth names "
                    f"(skipped): {sorted(set(unmatched))}"
                )
        return inferred
    
    @staticmethod
    def _climate_station_id(usaf, wban):
        """Build climate-style NOAA ISH station id: ``USAF-WBAN`` (zero-padded)."""
        if usaf in (None, "") or wban in (None, ""):
            return None
        try:
            usaf_i = int(float(str(usaf).strip()))
            wban_i = int(float(str(wban).strip()))
        except Exception:
            return None
        return f"{usaf_i:06d}-{wban_i:05d}"

    @staticmethod
    def _climate_safe_token(value, default="unknown"):
        tok = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value if value not in (None, "") else default)).strip("_")
        return tok or default

    @staticmethod
    def _usaf_to_wmo(usaf):
        """Map ISD USAF id to WMO block used by Ogimet / Wyoming (best-effort)."""
        try:
            usaf_i = int(float(str(usaf).strip()))
        except Exception:
            return None
        s = f"{usaf_i:06d}"
        # Many WMO stations are stored as WWWWW0 in USAF.
        if s.endswith("0") and len(s) == 6:
            return int(s[:-1])
        return usaf_i

    @staticmethod
    def _native_climate_interval(source):
        """Return (interval_name, short_label) for each climate source's native cadence."""
        src = str(source).strip().lower()
        native = {
            "noaa_hourly": ("hourly", "h"),
            "ogimet": ("hourly", "h"),
            "noaa_co2": ("monthly", "m"),
            "sounding_wyoming": ("sounding", "sounding"),
            "imgw": ("hourly", "h"),
        }
        if src not in native:
            raise ValueError(f"Unknown source for native interval: {source!r}")
        return native[src]

    @staticmethod
    def _rh_august_roche_magnus(t_c, td_c):
        """
        Relative humidity [%] from air temperature and dewpoint (°C) using the
        August–Roche–Magnus approximation (Alduchov & Eskridge 1996 constants):

            e_s(T) = 6.1094 * exp(17.625 * T / (T + 243.04))
            RH    = 100 * e_s(Td) / e_s(T)
                  = 100 * exp(γ(Td) - γ(T))
            γ(T)  = 17.625 * T / (T + 243.04)
        """
        t = pd.to_numeric(t_c, errors="coerce")
        td = pd.to_numeric(td_c, errors="coerce")
        a = 17.625
        b = 243.04
        # Avoid division by zero at T = -b (physically unreachable for air temps)
        denom_t = b + t
        denom_td = b + td
        gamma_t = np.where(np.isfinite(denom_t) & (denom_t != 0), (a * t) / denom_t, np.nan)
        gamma_td = np.where(np.isfinite(denom_td) & (denom_td != 0), (a * td) / denom_td, np.nan)
        rh = 100.0 * np.exp(gamma_td - gamma_t)
        rh = np.asarray(rh, dtype="float64")
        rh[~np.isfinite(rh)] = np.nan
        rh = np.clip(rh, 0.0, 100.0)
        return rh

    @classmethod
    def _attach_estimated_rh(cls, df, *, column="rh"):
        """
        Add August–Roche–Magnus RH [%] from temperature / dewpoint columns.

        Looks for ``t2m``+``dpt2m`` first (NOAA / Ogimet synop), then common
        aliases. Returns ``df`` unchanged when those columns are missing.
        """
        if df is None or df.empty:
            return df

        lower = {str(c).strip().lower(): c for c in df.columns}

        def _pick(*names):
            for n in names:
                if n in lower:
                    return lower[n]
            return None

        t_col = _pick("t2m", "tc", "ta", "temp", "temperature", "t (c)", "t(c)")
        td_col = _pick("dpt2m", "tdc", "td", "dewpoint", "dew_point", "td (c)", "td(c)")
        if t_col is None or td_col is None:
            return df

        out = df.copy()
        out[column] = cls._rh_august_roche_magnus(out[t_col], out[td_col])
        return out

    @staticmethod
    def _expected_native_observation_rows(
        source,
        start_date,
        end_date,
        *,
        sounding_hours=(0, 12),
    ):
        """Expected observation rows for the date window at the source native cadence."""
        start_dt = pd.to_datetime(start_date).normalize()
        end_dt = pd.to_datetime(end_date).normalize()
        if pd.isna(start_dt) or pd.isna(end_dt) or end_dt < start_dt:
            return 0
        n_days = int((end_dt - start_dt).days) + 1
        src = str(source).strip().lower()
        if src in {"noaa_hourly", "ogimet", "imgw"}:
            return n_days * 24
        if src == "noaa_co2":
            return (
                (end_dt.year - start_dt.year) * 12
                + (end_dt.month - start_dt.month)
                + 1
            )
        if src == "sounding_wyoming":
            hours = tuple(sounding_hours) if sounding_hours is not None else (0, 12)
            return n_days * max(1, len(hours))
        return n_days

    @staticmethod
    def _count_csv_data_rows(file_path):
        """Count non-header data lines in a CSV file."""
        if not os.path.exists(file_path):
            return 0
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            n = sum(1 for _ in fh)
        return max(0, n - 1)

    def _apply_missing_data_tolerance(
        self,
        file_path,
        *,
        source,
        start_date,
        end_date,
        missing_data_tolerance_ratio,
        sounding_hours=(0, 12),
        verbose=False,
    ):
        """
        Keep file if missing_ratio <= tolerance; otherwise delete it.

        missing_ratio = 1 - min(1, n_data_rows / expected_native_rows)
        """
        if missing_data_tolerance_ratio is None:
            return True, 0.0, self._count_csv_data_rows(file_path)

        tol = float(missing_data_tolerance_ratio)
        if not (0.0 <= tol <= 1.0):
            raise ValueError("missing_data_tolerance_ratio must be in [0, 1] or None.")

        expected = self._expected_native_observation_rows(
            source, start_date, end_date, sounding_hours=sounding_hours
        )
        n_rows = self._count_csv_data_rows(file_path)
        if expected <= 0:
            return True, 0.0, n_rows

        completeness = min(1.0, float(n_rows) / float(expected))
        missing_ratio = 1.0 - completeness
        if missing_ratio <= tol:
            return True, missing_ratio, n_rows

        try:
            os.remove(file_path)
        except Exception as exc:
            if verbose:
                print(
                    f"[download_in_situ_data] Could not remove incomplete file "
                    f"{file_path}: {exc}"
                )
        if verbose:
            print(
                f"[download_in_situ_data] Removed {os.path.basename(file_path)}: "
                f"rows={n_rows}/{expected}, missing_ratio={missing_ratio:.3f} "
                f"> tolerance={tol:.3f}"
            )
        return False, missing_ratio, n_rows

    def _climate_output_path(
        self,
        output_folder,
        source,
        native_label,
        country,
        station_id,
        start_date,
        end_date,
        *,
        combined=False,
    ):
        country_tok = self._climate_safe_token(country, default="UNKNOWN")
        sid_tok = self._climate_safe_token(
            "combined" if combined else station_id, default="unknown"
        )
        src_tok = self._climate_safe_token(source, default="source")
        tr_tok = self._climate_safe_token(native_label, default="native")
        fname = (
            f"climate_{src_tok}_{tr_tok}_{country_tok}_{sid_tok}_"
            f"{start_date}_{end_date}.csv"
        )
        return os.path.join(output_folder, fname)

    def _download_meteo_noaa_hourly_station(
        self,
        station_id,
        years,
        *,
        fm12=True,
        timeout=120,
        verbose=False,
    ):
        """
        Mirror ``climate::meteo_noaa_hourly`` for one station across *years*.

        Downloads ``https://www.ncei.noaa.gov/pub/data/noaa/{year}/{station}-{year}.gz``,
        parses fixed-width ISD records, optionally keeps FM-12 only, and returns
        a DataFrame with climate-compatible columns.
        """
        import gzip
        import io

        widths = [
            4, 6, 5, 4, 2, 2, 2, 2, 1, 6,
            7, 5, 5, 5, 4, 3, 1, 1, 4, 1,
            5, 1, 1, 1, 6, 1, 1, 1, 5, 1,
            5, 1, 5, 1,
        ]
        col_year, col_month, col_day, col_hour = 3, 4, 5, 6
        col_lat, col_lon, col_alt = 9, 10, 12
        col_fm12 = 11
        col_wd, col_ws = 15, 18
        col_vis, col_t2m, col_dpt, col_slp = 24, 28, 30, 32

        base_url = "https://www.ncei.noaa.gov/pub/data/noaa/"
        frames = []

        for year in years:
            year = int(year)
            url = f"{base_url}{year}/{station_id}-{year}.gz"
            try:
                resp = requests.get(url, timeout=timeout)
                if resp.status_code >= 400:
                    if verbose:
                        print(
                            f"[download_in_situ_data] Skip {station_id} {year}: "
                            f"HTTP {resp.status_code}"
                        )
                    continue
                content = resp.content or b""
                if len(content) <= 100:
                    if verbose:
                        print(
                            f"[download_in_situ_data] Skip {station_id} {year}: "
                            f"empty/short payload ({len(content)} bytes)"
                        )
                    continue

                with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
                    text_body = gz.read().decode("utf-8", errors="replace")

                dat = pd.read_fwf(
                    io.StringIO(text_body),
                    widths=widths,
                    header=None,
                    dtype=str,
                )
            except Exception as exc:
                if verbose:
                    print(
                        f"[download_in_situ_data] Skip {station_id} {year}: {exc}"
                    )
                continue

            if dat is None or dat.empty:
                continue

            if fm12:
                fm_col = dat.iloc[:, col_fm12].astype(str).str.strip()
                dat = dat.loc[fm_col == "FM-12"].copy()
                if dat.empty:
                    continue

            out = pd.DataFrame({
                "year": pd.to_numeric(dat.iloc[:, col_year], errors="coerce"),
                "month": pd.to_numeric(dat.iloc[:, col_month], errors="coerce"),
                "day": pd.to_numeric(dat.iloc[:, col_day], errors="coerce"),
                "hour": pd.to_numeric(dat.iloc[:, col_hour], errors="coerce"),
                "lat": pd.to_numeric(dat.iloc[:, col_lat], errors="coerce"),
                "lon": pd.to_numeric(dat.iloc[:, col_lon], errors="coerce"),
                "alt": pd.to_numeric(dat.iloc[:, col_alt], errors="coerce"),
                "wd": pd.to_numeric(dat.iloc[:, col_wd], errors="coerce"),
                "ws": pd.to_numeric(dat.iloc[:, col_ws], errors="coerce"),
                "visibility": pd.to_numeric(dat.iloc[:, col_vis], errors="coerce"),
                "t2m": pd.to_numeric(dat.iloc[:, col_t2m], errors="coerce"),
                "dpt2m": pd.to_numeric(dat.iloc[:, col_dpt], errors="coerce"),
                "slp": pd.to_numeric(dat.iloc[:, col_slp], errors="coerce"),
            })

            out.loc[out["t2m"] == 9999, "t2m"] = pd.NA
            out.loc[out["dpt2m"] == 9999, "dpt2m"] = pd.NA
            out.loc[out["ws"] == 9999, "ws"] = pd.NA
            out.loc[out["wd"] == 999, "wd"] = pd.NA
            out.loc[out["slp"] == 99999, "slp"] = pd.NA
            out.loc[out["visibility"] == 999999, "visibility"] = pd.NA

            out["lon"] = out["lon"] / 1000.0
            out["lat"] = out["lat"] / 1000.0
            out["ws"] = out["ws"] / 10.0
            out["t2m"] = out["t2m"] / 10.0
            out["dpt2m"] = out["dpt2m"] / 10.0
            out["slp"] = out["slp"] / 10.0

            out["date"] = pd.to_datetime(
                {
                    "year": out["year"],
                    "month": out["month"],
                    "day": out["day"],
                    "hour": out["hour"],
                },
                errors="coerce",
                utc=True,
            )
            out = out.dropna(subset=["date"]).copy()
            if out.empty:
                continue
            frames.append(out)

        if not frames:
            return pd.DataFrame(
                columns=[
                    "date", "year", "month", "day", "hour",
                    "lon", "lat", "alt",
                    "t2m", "dpt2m", "ws", "wd", "slp", "visibility",
                ]
            )

        all_data = pd.concat(frames, ignore_index=True)
        all_data = all_data[
            [
                "date", "year", "month", "day", "hour",
                "lon", "lat", "alt",
                "t2m", "dpt2m", "ws", "wd", "slp", "visibility",
            ]
        ]
        return all_data.sort_values("date").reset_index(drop=True)

    def _download_meteo_noaa_co2(self, start_date, end_date, *, timeout=120, verbose=False):
        """Mirror ``climate::meteo_noaa_co2`` (Mauna Loa monthly CO2)."""
        from io import StringIO

        url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
        if verbose:
            print(f"[download_in_situ_data] Downloading NOAA CO2: {url}")
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        # Skip comment lines starting with '#'
        lines = [ln for ln in resp.text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        co2 = pd.read_csv(
            StringIO("\n".join(lines)),
            sep=r"\s+",
            header=None,
            engine="python",
            na_values=["-9.99", "-0.99", "-99.99"],
        )
        # climate colnames: yy, mm, yy_d, co2_avg, co2_interp, co2_seas, ndays, st_dev_days
        n = min(8, co2.shape[1])
        names = ["yy", "mm", "yy_d", "co2_avg", "co2_interp", "co2_seas", "ndays", "st_dev_days"][:n]
        co2 = co2.iloc[:, :n].copy()
        co2.columns = names
        co2["date"] = pd.to_datetime(
            dict(year=co2["yy"], month=co2["mm"], day=1),
            errors="coerce",
            utc=True,
        )
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True) + pd.offsets.MonthEnd(0)
        co2 = co2.dropna(subset=["date"])
        co2 = co2[(co2["date"] >= start_dt) & (co2["date"] <= end_dt)].copy()
        return co2.reset_index(drop=True)

    @staticmethod
    def _ogimet_http_get(url, *, params=None, timeout=60, retries=2, verbose=False):
        """
        GET from Ogimet with a short connect timeout and a few retries.

        Returns response text, or ``None`` on failure.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "close",
        }
        # (connect, read) — fail fast when the host is unreachable.
        connect_s = min(10.0, float(timeout) if timeout else 10.0)
        read_s = float(timeout) if timeout else 60.0
        last_exc = None
        for attempt in range(max(1, int(retries) + 1)):
            try:
                resp = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=(connect_s, read_s),
                )
                resp.raise_for_status()
                return resp.text
            except Exception as exc:
                last_exc = exc
                if verbose:
                    print(
                        f"[download_in_situ_data] Ogimet request failed "
                        f"(attempt {attempt + 1}/{retries + 1}): {exc}"
                    )
                if attempt < retries:
                    time.sleep(min(2.0 * (attempt + 1), 5.0))
        if verbose and last_exc is not None:
            print(f"[download_in_situ_data] Ogimet giving up: {last_exc}")
        return None

    def _ogimet_probe(self, timeout=10, verbose=True):
        """Return True if www.ogimet.com accepts a TCP/HTTPS connection."""
        try:
            resp = requests.head(
                "https://www.ogimet.com/",
                timeout=(min(5.0, float(timeout)), float(timeout)),
                allow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            ok = resp.status_code < 500
        except Exception as exc:
            if verbose:
                print(
                    f"[download_in_situ_data] Ogimet unreachable "
                    f"(www.ogimet.com): {exc}"
                )
            return False
        if verbose and not ok:
            print(
                f"[download_in_situ_data] Ogimet probe HTTP {resp.status_code}"
            )
        return bool(ok)

    @staticmethod
    def _decode_aaxx_fm12(parte, *, default_year=None, default_month=None):
        """
        Decode a WMO FM-12 SYNOP (``AAXX``) telegram into scalar weather fields.

        Returns a dict with keys compatible with the NOAA hourly mirror where
        possible: ``t2m``, ``dpt2m``, ``wd``, ``ws``, ``slp``, ``station_pressure``,
        ``visibility``, ``cloud_cover``, ``precip_mm``, plus ``synop_day``,
        ``synop_hour``, ``wind_unit``. Missing / unparsable groups become ``None``.

        Notes
        -----
        Covers the common Section 0–1 groups used by Ogimet ``getsynop``
        ``PARTE`` strings. Section 3/5 national groups are ignored.
        """
        out = {
            "t2m": None,
            "dpt2m": None,
            "wd": None,
            "ws": None,
            "slp": None,
            "station_pressure": None,
            "visibility": None,
            "cloud_cover": None,
            "precip_mm": None,
            "synop_day": None,
            "synop_hour": None,
            "wind_unit": None,
        }
        if parte is None or (isinstance(parte, float) and pd.isna(parte)):
            return out
        text = str(parte).strip().rstrip("=")
        if not text:
            return out

        # Tokenise; keep 5-char groups and AAXX / YYGGiw / IIiii
        tokens = [t for t in re.split(r"\s+", text) if t]
        if not tokens:
            return out

        # Drop leading AAXX if present
        if tokens[0].upper() == "AAXX":
            tokens = tokens[1:]
        if not tokens:
            return out

        def _vis_vv(vv_code):
            try:
                vv = int(vv_code)
            except Exception:
                return None
            if vv == 0:
                return 0.0
            if 1 <= vv <= 50:
                return float(vv) * 100.0          # metres
            if 56 <= vv <= 80:
                return (vv - 50) * 1000.0         # km → m
            if 81 <= vv <= 89:
                return (vv - 80) * 5000.0 + 30_000.0
            if vv == 90:
                return 50.0
            if vv == 91:
                return 200.0
            if vv == 92:
                return 500.0
            if vv == 93:
                return 1000.0
            if vv == 94:
                return 2000.0
            if vv == 95:
                return 4000.0
            if vv == 96:
                return 10000.0
            if vv == 97:
                return 20000.0
            if vv == 98:
                return 50000.0
            if vv == 99:
                return 50000.0
            return None

        def _temp_sTTT(group):
            # 1sTTT / 2sTTT
            if len(group) != 5 or not group[1:].isdigit():
                return None
            sign = group[1]
            tenths = int(group[2:])
            if sign not in "01":
                return None
            val = tenths / 10.0
            return -val if sign == "1" else val

        def _pressure_PPPP(pppp):
            if not pppp.isdigit() or len(pppp) != 4:
                return None
            code = int(pppp)
            # WMO: omit thousands digit when P >= 1000 hPa (→ 0xxx);
            # values near 900–999 use 9xxx.
            if code < 5000:
                return 1000.0 + code / 10.0
            return code / 10.0

        def _precip_RRR(rrr):
            if not rrr.isdigit() or len(rrr) != 3:
                return None
            code = int(rrr)
            if code == 0:
                return 0.0
            if 1 <= code <= 989:
                return float(code)               # mm
            if code == 990:
                return 0.0
            if 991 <= code <= 999:
                return (code - 990) / 10.0       # 0.1–0.9 mm
            return None

        iw = None
        # YYGGiw
        if len(tokens[0]) == 5 and tokens[0].isdigit():
            yyggiw = tokens[0]
            out["synop_day"] = int(yyggiw[0:2])
            out["synop_hour"] = int(yyggiw[2:4])
            iw = int(yyggiw[4])
            # WMO iw: 0/1 = m/s, 3/4 = knots
            out["wind_unit"] = "m/s" if iw in (0, 1) else "kt"
            tokens = tokens[1:]

        # IIiii station id — skip
        if tokens and len(tokens[0]) == 5 and tokens[0].isdigit():
            tokens = tokens[1:]

        # iRixhVV
        if tokens and len(tokens[0]) == 5 and tokens[0][0].isdigit():
            g = tokens[0]
            if g[3:].isdigit() or g[3:] == "//":
                if g[3:].isdigit():
                    out["visibility"] = _vis_vv(g[3:])
                tokens = tokens[1:]

        # Nddff
        if tokens and len(tokens[0]) == 5:
            g = tokens[0]
            if g[0] in "0123456789/" and (g[1:].isdigit() or "/" in g[1:]):
                if g[0].isdigit():
                    n = int(g[0])
                    out["cloud_cover"] = None if n == 9 else float(n)  # oktas; 9 = sky obscured
                dd = g[1:3]
                ff = g[3:5]
                if dd.isdigit():
                    ddi = int(dd)
                    if ddi == 0:
                        out["wd"] = 0.0  # calm / variable handled with ff
                    elif 1 <= ddi <= 36:
                        out["wd"] = float(ddi * 10)
                    # 99 = variable → leave None or set special
                if ff.isdigit():
                    ffi = int(ff)
                    if ffi != 99:  # 99 often means missing / see 00fff
                        ws = float(ffi)
                        if out["wind_unit"] == "kt":
                            ws = ws * 0.514444  # → m/s for consistency with NOAA path
                        out["ws"] = ws
                        if ffi == 0:
                            out["wd"] = 0.0
                tokens = tokens[1:]

        # Remaining groups until 333 / 555 / NIL
        for g in tokens:
            gu = g.upper()
            if gu in {"333", "555", "NIL"}:
                break
            if len(g) != 5:
                continue
            head = g[0]
            rest = g[1:]
            if head == "1" and rest[0] in "01" and rest[1:].isdigit():
                out["t2m"] = _temp_sTTT(g)
            elif head == "2" and len(rest) == 4 and rest.isdigit():
                # Classic dewpoint 2sTTT; some automatics use 29uuu for RH%
                if rest[0] == "9":
                    continue
                if rest[0] in "01":
                    out["dpt2m"] = _temp_sTTT(g)
            elif head == "3" and rest.isdigit():
                out["station_pressure"] = _pressure_PPPP(rest)
            elif head == "4" and rest.isdigit():
                out["slp"] = _pressure_PPPP(rest)
            elif head == "6" and len(rest) == 4 and rest[:3].isdigit():
                out["precip_mm"] = _precip_RRR(rest[:3])
            # 5appp, 7wwW1W2, 8NhCLCMCH, 9GGgg ignored for now

        # Silence unused year/month (reserved for future 24h precip day-roll)
        _ = (default_year, default_month)
        return out

    def _ogimet_attach_decoded_synop(self, df):
        """Add decoded FM-12 columns from Ogimet synop ``PARTE`` (or similar) column."""
        if df is None or df.empty:
            return df

        parte_col = None
        for cand in ("PARTE", "Parte", "parte", "SYNOP", "synop", "telegram"):
            if cand in df.columns:
                parte_col = cand
                break
        if parte_col is None:
            # Sometimes the last column holds the telegram without a stable name
            for c in df.columns:
                sample = df[c].astype(str).head(20).str.upper()
                if sample.str.contains("AAXX", regex=False).any():
                    parte_col = c
                    break
        if parte_col is None:
            return df

        year_col = None
        month_col = None
        for yc in ("year", "Year", "ANO", "Ano"):
            if yc in df.columns:
                year_col = yc
                break
        for mc in ("month", "Month", "MES", "Mes"):
            if mc in df.columns:
                month_col = mc
                break

        decoded_rows = []
        for idx, row in df.iterrows():
            y = int(row[year_col]) if year_col is not None and pd.notna(row[year_col]) else None
            m = int(row[month_col]) if month_col is not None and pd.notna(row[month_col]) else None
            decoded_rows.append(
                self._decode_aaxx_fm12(row[parte_col], default_year=y, default_month=m)
            )
        decoded_df = pd.DataFrame(decoded_rows, index=df.index)

        # Prefer existing date from YYGGiw when ANO/MES/DIA/HORA present but
        # fill gaps from decoded synop_day/hour if needed — leave date as-is.
        for col in decoded_df.columns:
            if col in ("synop_day", "synop_hour", "wind_unit"):
                # keep as metadata helpers
                df[col] = decoded_df[col]
            else:
                df[col] = decoded_df[col]
        return df

    @staticmethod
    def _ogimet_ensure_date_utc(df):
        """
        Ensure an Ogimet frame has a UTC-aware ``date`` column.

        Ogimet ``getsynop`` / ``gsynres`` observation times are UTC (WMO SYNOP).
        Component columns (year/month/day/hour or ANO/MES/DIA/HORA) and any
        existing ``date`` values are interpreted as UTC, never as local time.
        """
        if df is None or df.empty:
            return df

        out = df.copy()
        lower = {str(c).strip().lower(): c for c in out.columns}

        def _col(*names):
            for n in names:
                if n in lower:
                    return lower[n]
            return None

        if "date" in lower:
            date_col = lower["date"]
            out["date"] = pd.to_datetime(out[date_col], errors="coerce", utc=True)
            if date_col != "date":
                out = out.drop(columns=[date_col])
            return out

        y = _col("year", "ano")
        m = _col("month", "mes")
        d = _col("day", "dia")
        h = _col("hour", "hora")
        mi = _col("min", "minute", "minuto")
        if y is not None and m is not None and d is not None:
            out["date"] = pd.to_datetime(
                {
                    "year": pd.to_numeric(out[y], errors="coerce"),
                    "month": pd.to_numeric(out[m], errors="coerce"),
                    "day": pd.to_numeric(out[d], errors="coerce"),
                    "hour": pd.to_numeric(out[h], errors="coerce") if h else 0,
                    "minute": pd.to_numeric(out[mi], errors="coerce") if mi else 0,
                },
                errors="coerce",
                utc=True,
            )
            return out

        # HTML gsynres: often Date + HH:MM (UTC) as the first two columns
        flat_cols = []
        for c in out.columns:
            if isinstance(c, tuple):
                flat_cols.append(" ".join(str(x) for x in c if str(x) != "nan").strip())
            else:
                flat_cols.append(str(c).strip())
        # Prefer columns whose names look like date / time
        date_like = None
        time_like = None
        for i, name in enumerate(flat_cols):
            nl = name.lower()
            if date_like is None and ("date" in nl or nl in {"fecha"}):
                # MultiIndex header pollution: value may already be MM/DD/YYYY
                date_like = out.columns[i]
            if time_like is None and (
                nl in {"utc", "time", "hora"} or re.fullmatch(r"\d{1,2}:\d{2}", name)
            ):
                time_like = out.columns[i]
        if date_like is not None:
            date_s = out[date_like].astype(str).str.strip()
            if time_like is not None:
                time_s = out[time_like].astype(str).str.strip()
                stamp = date_s + " " + time_s
            else:
                stamp = date_s
            parsed = pd.to_datetime(stamp, errors="coerce", utc=True)
            if parsed.notna().any():
                out["date"] = parsed
        return out

    @staticmethod
    def _ogimet_parse_gsynres_html(html):
        """
        Parse Ogimet ``gsynres`` decoded-synop HTML into a clean DataFrame.

        Ogimet pages are malformed (unclosed ``<tr>``/``</thead>``), so
        ``pandas.read_html`` puts observation cells into MultiIndex headers and
        also scrapes layout/ad tables. This parser extracts only the
        ``Decoded synop data`` table and returns UTC ``date`` plus weather
        columns. Returns an empty DataFrame when the page has no observations
        (``NO DATA FOUND``), is rate-limited, or has no usable table.
        """
        if not html or not str(html).strip():
            return pd.DataFrame()

        text_upper = str(html).upper()
        if "NO DATA FOUND" in text_upper:
            return pd.DataFrame()
        if "LIMIT FOR OLD DATA QUERIES EXCEEDED" in text_upper:
            # Signal to caller via empty + sentinel attribute is awkward;
            # raise so the download loop can retry after a cooldown.
            raise RuntimeError(
                "Ogimet gsynres rate limit: max 1 old-data query per 20s per IP"
            )

        # Prefer the nucleo content (center panel); fall back to full page.
        nucleo = re.search(
            r"<!--\s*nucleo.*?-->(.*?)(?:<!--\s*fin del nucleo|</BODY>)",
            html,
            flags=re.I | re.S,
        )
        body = nucleo.group(1) if nucleo else html

        # Table whose caption mentions decoded synop / synop data
        table_match = None
        for m in re.finditer(r"<table\b[^>]*>.*?</table>", body, flags=re.I | re.S):
            block = m.group(0)
            cap = re.search(r"<caption\b[^>]*>(.*?)</caption>", block, flags=re.I | re.S)
            cap_txt = re.sub(r"<[^>]+>", " ", cap.group(1) if cap else "")
            cap_txt = " ".join(cap_txt.split()).lower()
            if "decoded synop" in cap_txt or "synop data" in cap_txt:
                table_match = block
                break
        if table_match is None:
            # Fallback: largest table in nucleo that has Date + T headers
            candidates = list(
                re.finditer(r"<table\b[^>]*>.*?</table>", body, flags=re.I | re.S)
            )
            best = None
            best_n = 0
            for m in candidates:
                block = m.group(0)
                ntd = len(re.findall(r"<td\b", block, flags=re.I))
                if ntd > best_n and re.search(r"<th\b[^>]*>\s*Date", block, flags=re.I):
                    best = block
                    best_n = ntd
            table_match = best
        if not table_match:
            return pd.DataFrame()

        def _cell_text(cell_html):
            # Drop images/scripts; keep alt text when present for WW icons
            cell = re.sub(
                r"<img\b[^>]*\balt=['\"]([^'\"]*)['\"][^>]*>",
                r" \1 ",
                cell_html,
                flags=re.I,
            )
            cell = re.sub(r"<br\s*/?>", " ", cell, flags=re.I)
            cell = re.sub(r"<[^>]+>", " ", cell)
            cell = (
                cell.replace("&nbsp;", " ")
                .replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
            )
            return " ".join(cell.split()).strip()

        # Header cells from thead / first header row
        thead = re.search(r"<thead\b[^>]*>(.*?)</thead>", table_match, flags=re.I | re.S)
        header_src = thead.group(1) if thead else table_match
        # If </thead> is missing (common on Ogimet), take th's before first <td>
        if thead is None:
            first_td = re.search(r"<td\b", table_match, flags=re.I)
            header_src = table_match[: first_td.start()] if first_td else table_match

        th_matches = list(
            re.finditer(r"<th\b([^>]*)>(.*?)</th>", header_src, flags=re.I | re.S)
        )
        headers = []
        for mth in th_matches:
            attrs, inner = mth.group(1), mth.group(2)
            name = _cell_text(inner)
            colspan_m = re.search(r"colspan\s*=\s*['\"]?(\d+)", attrs, flags=re.I)
            span = int(colspan_m.group(1)) if colspan_m else 1
            if span >= 2 and name.lower() in {"date", "fecha"}:
                headers.extend(["Date", "UTC"])
                for _ in range(span - 2):
                    headers.append(f"col_{len(headers)}")
            else:
                headers.append(name if name else f"col_{len(headers)}")
                for _ in range(max(span - 1, 0)):
                    headers.append(f"col_{len(headers)}")

        # Data rows: groups of <td>...</td> belonging to each observation.
        # Ogimet often omits </tr>, so split by <tr> and also fall back to
        # chunking consecutive td's by expected width.
        tr_parts = re.split(r"<tr\b[^>]*>", table_match, flags=re.I)[1:]
        raw_rows = []
        for part in tr_parts:
            tds = re.findall(r"<td\b[^>]*>(.*?)</td>", part, flags=re.I | re.S)
            if not tds:
                continue
            vals = [_cell_text(td) for td in tds]
            # Skip pure header leftovers
            if vals and vals[0].lower() in {"date", "fecha"}:
                continue
            raw_rows.append(vals)

        if not raw_rows:
            return pd.DataFrame()

        width = len(headers) if headers else max(len(r) for r in raw_rows)
        if not headers:
            headers = [f"col_{i}" for i in range(width)]
        # Pad / trim
        norm = []
        for r in raw_rows:
            if len(r) < 2:
                continue
            # Observation rows start with MM/DD/YYYY or YYYY-MM-DD
            if not re.match(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$", r[0]):
                continue
            if len(r) < width:
                r = r + [None] * (width - len(r))
            elif len(r) > width:
                r = r[:width]
            norm.append(r)
        if not norm:
            return pd.DataFrame()

        # Unique column names
        seen = {}
        uniq_headers = []
        for h in headers[:width]:
            base = h or "col"
            n = seen.get(base, 0)
            seen[base] = n + 1
            uniq_headers.append(base if n == 0 else f"{base}_{n}")

        df = pd.DataFrame(norm, columns=uniq_headers)

        # Build UTC date from Date + UTC/time columns
        date_col = next((c for c in df.columns if c.lower() == "date"), None)
        time_col = next(
            (c for c in df.columns if c.lower() in {"utc", "time", "hora", "hour"}),
            None,
        )
        if date_col is not None:
            if time_col is not None:
                stamp = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
            else:
                stamp = df[date_col].astype(str).str.strip()
            df["date"] = pd.to_datetime(stamp, errors="coerce", utc=True)

        # Friendly aliases aligned with synop decode / NOAA where possible
        rename = {}
        for c in df.columns:
            cl = c.lower().replace(" ", "")
            if cl in {"t(c)", "t"}:
                rename[c] = "t2m"
            elif cl in {"td(c)", "td"}:
                rename[c] = "dpt2m"
            elif cl in {"psea(hpa)", "pseahpa", "psea"}:
                rename[c] = "slp"
            elif cl in {"ffkmh", "ff"}:
                rename[c] = "ws_kmh"
            elif cl in {"viskm", "vis"}:
                rename[c] = "visibility_km"
            elif cl in {"hr%", "hr"}:
                rename[c] = "rh"
        if rename:
            df = df.rename(columns=rename)
        # Convert wind km/h → m/s for consistency with synop decode
        if "ws_kmh" in df.columns and "ws" not in df.columns:
            ws = pd.to_numeric(df["ws_kmh"], errors="coerce")
            df["ws"] = ws / 3.6
        return df

    def _download_meteo_ogimet_station(
        self,
        wmo_id,
        start_date,
        end_date,
        *,
        interval="daily",
        backend=None,
        country_name=None,
        timeout=120,
        verbose=False,
        decode_synop=True,
    ):
        """
        Pure-Python mirror of ``climate::meteo_ogimet``.

        Observation times are always treated as **UTC** (WMO SYNOP convention).
        The returned ``date`` column is timezone-aware UTC.

        - backend ``synop`` (default for hourly): Ogimet ``getsynop`` raw CSV
          (``PARTE`` AAXX telegrams). When ``decode_synop=True`` (default),
          FM-12 groups are decoded into ``t2m``, ``dpt2m``, ``wd``, ``ws``,
          ``slp``, etc.
        - backend ``html`` (default for daily): Ogimet ``gsynres`` HTML tables.
          Decoded HTML pages are truncated by Ogimet for long ranges, so this
          path requests ≤3-day chunks ending at 23:00 UTC and sleeps ~21s
          between calls (slow for multi-week ranges; prefer ``synop``).
        """
        from io import StringIO
        import time as _time

        wmo = int(wmo_id)
        effective = backend
        if effective is None:
            effective = "synop" if interval == "hourly" else "html"
        effective = str(effective).lower()
        if effective not in {"synop", "html"}:
            raise ValueError("ogimet backend must be 'synop' or 'html'.")

        # Ogimet begin/end query params are UTC (YYYYMMDDhhmm).
        begin = pd.to_datetime(start_date, utc=True).strftime("%Y%m%d0000")
        end = pd.to_datetime(end_date, utc=True).strftime("%Y%m%d2359")

        if effective == "synop":
            params = {"begin": begin, "end": end, "header": "yes"}
            if country_name:
                params["state"] = str(country_name)
            else:
                params["block"] = str(wmo)
            url = "https://www.ogimet.com/cgi-bin/getsynop"
            if verbose:
                print(
                    f"[download_in_situ_data] Ogimet synop (UTC): "
                    f"{url} params={params}"
                )
            text_body = self._ogimet_http_get(
                url, params=params, timeout=timeout, retries=2, verbose=verbose
            )
            if not text_body:
                return pd.DataFrame()
            text_body = text_body.strip()
            if not text_body or "Sorry" in text_body[:200]:
                return pd.DataFrame()
            df = pd.read_csv(StringIO(text_body))
            # Normalize column names
            cols = {c: c.strip() for c in df.columns}
            df = df.rename(columns=cols)
            if df.empty:
                return pd.DataFrame()
            if "station" not in df.columns:
                df.insert(0, "station", wmo)
            else:
                df["station"] = wmo
            df = self._ogimet_ensure_date_utc(df)
            if decode_synop:
                df = self._ogimet_attach_decoded_synop(df)
                df = self._ogimet_ensure_date_utc(df)
            return df

        # HTML backend via gsynres. Ogimet silently truncates large decoded
        # HTML pages (~2–3 days of 3-hourly rows), so we request small chunks
        # ending at 23:00 UTC and pause between calls (≈1 query / 20s).
        frames = []
        cur = pd.to_datetime(start_date, utc=True).floor("D")
        end_ts = pd.to_datetime(end_date, utc=True).floor("D")
        max_chunk_days = 3
        while cur <= end_ts:
            chunk_end = min(cur + pd.Timedelta(days=max_chunk_days - 1), end_ts)
            ndays = int((chunk_end - cur).days) + 1
            # Reference date/hora must be the chunk END in UTC (include that day).
            url = (
                "https://www.ogimet.com/cgi-bin/gsynres"
                f"?lang=en&ind={wmo}"
                f"&ano={chunk_end.year}&mes={int(chunk_end.month):02d}"
                f"&day={int(chunk_end.day):02d}&hora=23"
                f"&ndays={ndays}&decoded=yes"
            )
            if verbose:
                print(f"[download_in_situ_data] Ogimet html (UTC): {url}")

            tab = None
            for attempt in range(3):
                html = self._ogimet_http_get(
                    url, timeout=timeout, retries=2, verbose=verbose
                )
                if not html:
                    break
                try:
                    tab = self._ogimet_parse_gsynres_html(html)
                    break
                except RuntimeError as exc:
                    # Rate limit: wait and retry (Ogimet allows ~1 query / 20s)
                    if verbose:
                        print(
                            f"[download_in_situ_data] {exc}; "
                            f"sleeping 21s (attempt {attempt + 1}/3)"
                        )
                    _time.sleep(21)
                    tab = None
                except Exception as exc:
                    if verbose:
                        print(
                            f"[download_in_situ_data] Ogimet html parse failed: {exc}"
                        )
                    tab = None
                    break

            if tab is not None and not tab.empty:
                tab = tab.copy()
                tab.insert(0, "station_ID", wmo)
                tab = self._ogimet_ensure_date_utc(tab)
                frames.append(tab)
            cur = chunk_end + pd.Timedelta(days=1)
            # Be polite between chunks to avoid the 20s rate limit
            if cur <= end_ts:
                _time.sleep(21)

        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if "date" in out.columns:
            out = out.drop_duplicates(subset=["date"], keep="last")
            out = out.sort_values("date").reset_index(drop=True)
        return out

    def _download_sounding_wyoming_station(
        self,
        wmo_id,
        start_date,
        end_date,
        *,
        hours=(0, 12),
        timeout=120,
        verbose=False,
    ):
        """Mirror ``climate::sounding_wyoming`` over a date range (TEMP/LIST text)."""
        from io import StringIO

        wmo = int(wmo_id)
        frames = []
        days = pd.date_range(start_date, end_date, freq="D")
        for day in days:
            for hh in hours:
                url = (
                    "https://weather.uwyo.edu/wsgi/sounding?"
                    f"datetime={day.year:04d}-{day.month:02d}-{day.day:02d}"
                    f"%20{int(hh):02d}:00:00&id={wmo:05d}&src=UNKNOWN&type=TEXT:LIST"
                )
                try:
                    resp = requests.get(url, timeout=timeout)
                    if resp.status_code >= 400 or len(resp.text) < 800:
                        continue
                    # Extract PRE> ... data block if present; else parse fixed-width body.
                    text = resp.text
                    if "PRE>" not in text.upper() and "PRES" not in text.upper():
                        continue
                    # Keep raw text rows that look like sounding levels
                    rows = []
                    for ln in text.splitlines():
                        s = re.sub(r"<[^>]+>", " ", ln).strip()
                        if not s:
                            continue
                        parts = s.split()
                        if len(parts) >= 11:
                            try:
                                float(parts[0])
                                float(parts[1])
                            except Exception:
                                continue
                            rows.append(parts[:11])
                    if not rows:
                        continue
                    df = pd.DataFrame(
                        rows,
                        columns=[
                            "PRES", "HGHT", "TEMP", "DWPT", "RELH",
                            "MIXR", "DRCT", "SKNT", "THTA", "THTE", "THTV",
                        ],
                    )
                    for c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    df.insert(0, "sounding_time_utc", f"{day.date()} {int(hh):02d}:00")
                    df.insert(0, "wmo_id", wmo)
                    frames.append(df)
                    if verbose:
                        print(
                            f"[download_in_situ_data] Sounding {wmo} "
                            f"{day.date()} {hh:02d}Z: {len(df)} levels"
                        )
                except Exception as exc:
                    if verbose:
                        print(f"[download_in_situ_data] Sounding skip: {exc}")
                    continue
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _download_meteo_imgw(
        self,
        start_date,
        end_date,
        *,
        interval="daily",
        rank="synop",
        station_ids=None,
        timeout=120,
        verbose=False,
    ):
        """
        Pure-Python mirror of ``climate::meteo_imgw`` for hourly/daily/monthly.

        Downloads public IMGW-PIB zip archives from danepubliczne.imgw.pl and
        optionally filters to station ids/names.
        """
        import io
        import zipfile

        rank = str(rank).lower()
        if rank not in {"synop", "climate", "precip"}:
            raise ValueError("imgw_rank must be 'synop', 'climate', or 'precip'.")
        if interval == "10min":
            raise NotImplementedError(
                "imgw 10-min datastore (meteo_imgw_datastore) is not "
                "implemented in the pure-Python backend yet."
            )
        interval_map = {"hourly": "godzinowe", "daily": "dobowe", "monthly": "miesieczne"}
        if interval not in interval_map:
            raise ValueError("imgw interval must be hourly, daily, or monthly.")
        folder = interval_map[interval]
        years = range(pd.to_datetime(start_date).year, pd.to_datetime(end_date).year + 1)
        frames = []
        station_filter = None
        if station_ids:
            station_filter = {str(s).strip().upper() for s in station_ids if s not in (None, "")}

        for year in years:
            # Prefer yearly pack; fall back to monthly packs.
            candidates = [
                f"https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/"
                f"dane_meteorologiczne/{folder}/{rank}/{year}/{year}_{rank[0]}.zip",
            ]
            for month in range(1, 13):
                candidates.append(
                    f"https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/"
                    f"dane_meteorologiczne/{folder}/{rank}/{year}/"
                    f"{year}_{month:02d}_{rank[0]}.zip"
                )

            for url in candidates:
                try:
                    resp = requests.get(url, timeout=timeout)
                    if resp.status_code >= 400 or len(resp.content) < 100:
                        continue
                    if verbose:
                        print(f"[download_in_situ_data] IMGW zip: {url}")
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                        for name in zf.namelist():
                            if not name.lower().endswith((".csv", ".txt")):
                                continue
                            with zf.open(name) as fh:
                                raw = fh.read()
                            for enc in ("utf-8", "cp1250", "latin-1"):
                                try:
                                    text_body = raw.decode(enc)
                                    break
                                except Exception:
                                    text_body = None
                            if not text_body:
                                continue
                            try:
                                df = pd.read_csv(io.StringIO(text_body), header=None, dtype=str)
                            except Exception:
                                continue
                            if df.empty:
                                continue
                            if station_filter is not None:
                                # Station id/name often in first few columns.
                                mask = False
                                for col in df.columns[:3]:
                                    mask = mask | df[col].astype(str).str.upper().isin(station_filter)
                                df = df.loc[mask].copy()
                            if df.empty:
                                continue
                            df["source_file"] = os.path.basename(name)
                            df["imgw_year"] = year
                            frames.append(df)
                    # If yearly pack worked, skip monthly candidates for that year.
                    if url.endswith(f"{year}_{rank[0]}.zip"):
                        break
                except Exception as exc:
                    if verbose:
                        print(f"[download_in_situ_data] IMGW skip {url}: {exc}")
                    continue

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def download_in_situ_data(
        self,
        roi=None,
        start_date=None,
        end_date=None,
        *,
        source="noaa_hourly",
        separate_stations=True,
        output_folder="data/in_situ",
        fm12=True,
        ogimet_backend=None,
        country_name=None,
        imgw_rank="synop",
        sounding_hours=(0, 12),
        no_of_stations=None,
        missing_data_tolerance_ratio=None,
        n_jobs=None,
        timeout=120,
        verbose=True,
        estimate_RH=True,
    ):
        """
        Download in-situ meteorological data (climate-package style backends).

        Pure-Python mirrors of the global/Polish climate R functions:

        - ``noaa_hourly`` → ``meteo_noaa_hourly`` (native hourly)
        - ``ogimet`` → ``meteo_ogimet`` (native hourly SYNOP)
        - ``noaa_co2`` → ``meteo_noaa_co2`` (native monthly; ROI ignored)
        - ``sounding_wyoming`` → ``sounding_wyoming`` (native sounding times)
        - ``imgw`` → ``meteo_imgw`` (native hourly)

        Each source is downloaded at its native temporal cadence.

        Output filenames include source, native cadence, country, and station id:
        ``climate_{source}_{native}_{country}_{station_id}_{start}_{end}.csv``

        Parameters
        ----------
        roi : list | tuple | str | None
            ROI for station discovery. Required except for ``source='noaa_co2'``.
        start_date, end_date : str
            ISO dates (YYYY-MM-DD).
        source : str
            One of ``noaa_hourly``, ``ogimet``, ``noaa_co2``, ``sounding_wyoming``, ``imgw``.
        separate_stations : bool
            Write one CSV per station when True.
        output_folder : str
            Destination folder.
        fm12 : bool
            NOAA ISH FM-12 filter (``noaa_hourly`` only).
        ogimet_backend : str | None
            ``'synop'``, ``'html'``, or None (defaults to ``synop`` for native
            hourly). ``synop`` returns Ogimet ``getsynop`` CSV and, by default,
            decodes FM-12 ``PARTE`` AAXX groups into ``t2m``, ``dpt2m``, ``wd``,
            ``ws``, ``slp``, etc. ``html`` uses Ogimet ``gsynres`` decoded tables.
            For both backends, observation times and the ``date`` column are
            **UTC** (never local time).
        country_name : str | None
            Ogimet country-mode bulk download (synop backend).
        imgw_rank : str
            IMGW station rank: ``synop``, ``climate``, or ``precip``.
        sounding_hours : tuple[int, ...]
            UTC hours to request for Wyoming soundings (default 00 and 12).
        no_of_stations : int | None
            Cap stations from ROI discovery.
        missing_data_tolerance_ratio : float | None
            If set (0–1), keep a station file only when
            ``missing_ratio = 1 - rows/expected_native_rows`` is **≤** this
            tolerance. Files that fail are deleted. ``None`` disables the check.
        n_jobs : int | None
            Parallel workers for per-station downloads. ``None`` or ``<= 1``
            runs sequentially; ``> 1`` uses a thread pool.
        timeout : int | float
            HTTP timeout seconds.
        verbose : bool
            Progress logs.
        estimate_RH : bool
            If True (default), add an ``rh`` column [%] estimated from air
            temperature and dewpoint via the August–Roche–Magnus approximation
            when ``t2m``/``dpt2m`` (or aliases) are present. If False, leave
            station tables as downloaded (no RH estimation).
        """
        source = str(source).strip().lower()
        allowed = {
            "noaa_hourly", "ogimet", "noaa_co2", "sounding_wyoming", "imgw",
        }
        if source not in allowed:
            raise ValueError(f"source must be one of {sorted(allowed)}")
        if not start_date or not end_date:
            raise ValueError("start_date and end_date are required.")
        estimate_RH = bool(estimate_RH)
        if missing_data_tolerance_ratio is not None:
            tol = float(missing_data_tolerance_ratio)
            if not (0.0 <= tol <= 1.0):
                raise ValueError("missing_data_tolerance_ratio must be in [0, 1] or None.")
            missing_data_tolerance_ratio = tol
        if n_jobs is not None:
            n_jobs = int(n_jobs)
            if n_jobs < 1:
                raise ValueError("n_jobs must be >= 1 or None.")

        interval, native_label = self._native_climate_interval(source)

        if source == "ogimet":
            if not self._ogimet_probe(timeout=min(10, float(timeout) if timeout else 10), verbose=verbose):
                raise RuntimeError(
                    "www.ogimet.com is unreachable from this network "
                    "(TCP/HTTPS connect timeout). Use source='noaa_hourly' "
                    "instead, or retry with a VPN / different network."
                )

        start_dt = pd.to_datetime(start_date, errors="coerce", utc=True)
        end_dt = pd.to_datetime(end_date, errors="coerce", utc=True)
        if pd.isna(start_dt) or pd.isna(end_dt):
            raise ValueError("start_date and end_date must be valid YYYY-MM-DD dates.")
        if start_dt > end_dt:
            raise ValueError("start_date must be <= end_date.")
        end_dt_inclusive = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        os.makedirs(output_folder, exist_ok=True)

        bbox = None
        station_meta = []  # list of dicts: station_id, country, wmo_id, ...

        if source == "noaa_co2":
            station_meta = [{
                "station_id": "MLO",
                "country": "UNITED_STATES",
                "wmo_id": None,
            }]
        else:
            if roi is None:
                raise ValueError(f"roi is required for source='{source}'.")
            bbox = self._roi_to_bbox_list(roi)
            if bbox is None:
                raise ValueError("Could not resolve bbox from roi.")

            if verbose:
                print("[download_in_situ_data] Discovering stations via list_in_situ_stations ...")
            discovery = self.list_in_situ_stations(
                roi=roi,
                start_date=start_date,
                end_date=end_date,
                no_of_stations=no_of_stations,
                timeout=timeout,
                export=False,
                verbose=verbose,
            )
            for row in discovery.get("stations") or []:
                usaf = wban = country = None
                for k, v in row.items():
                    lk = str(k).lower()
                    if lk == "usaf" and usaf is None:
                        usaf = v
                    elif lk == "wban" and wban is None:
                        wban = v
                    elif lk in {"country", "countries"} and country is None:
                        country = v
                sid = self._climate_station_id(usaf, wban)
                if not sid:
                    continue
                station_meta.append({
                    "station_id": sid,
                    "country": (str(country).strip().upper() if country not in (None, "") else "UNKNOWN"),
                    "wmo_id": self._usaf_to_wmo(usaf),
                    "usaf": usaf,
                    "wban": wban,
                    "name": row.get("STATION NAME") or row.get("station_name") or row.get("NAME"),
                })

            # Deduplicate by station_id
            seen = set()
            uniq = []
            for s in station_meta:
                if s["station_id"] in seen:
                    continue
                seen.add(s["station_id"])
                uniq.append(s)
            station_meta = uniq

        if verbose:
            print(f"[download_in_situ_data] source={source}, native_interval={interval}, stations={len(station_meta)}")

        if not station_meta:
            return {
                "source": source,
                "native_interval": interval,
                "bbox": bbox,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "stations_count": 0,
                "station_ids": [],
                "separate_stations": bool(separate_stations),
                "output_files": [],
                "failed": [],
                "removed_incomplete": [],
                "missing_data_tolerance_ratio": missing_data_tolerance_ratio,
                "n_jobs": n_jobs,
            }

        years = list(range(int(start_dt.year), int(end_dt.year) + 1))
        output_files = []
        failed = []
        removed_incomplete = []
        combined_frames = []

        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None

        def _keep_or_drop(path_out, station_key):
            keep, missing_ratio, n_rows = self._apply_missing_data_tolerance(
                path_out,
                source=source,
                start_date=start_date,
                end_date=end_date,
                missing_data_tolerance_ratio=missing_data_tolerance_ratio,
                sounding_hours=sounding_hours,
                verbose=verbose,
            )
            if keep:
                output_files.append(path_out)
                return True
            removed_incomplete.append(
                {
                    "station_id": station_key,
                    "file": path_out,
                    "missing_ratio": missing_ratio,
                    "rows": n_rows,
                }
            )
            return False

        # ---- noaa_co2 (single series) ----
        if source == "noaa_co2":
            try:
                df = self._download_meteo_noaa_co2(
                    start_date, end_date, timeout=timeout, verbose=verbose
                )
            except Exception as exc:
                raise RuntimeError(f"noaa_co2 download failed: {exc}") from exc
            meta = station_meta[0]
            if df is None or df.empty:
                failed.append(meta["station_id"])
            else:
                df.insert(0, "country", meta["country"])
                df.insert(0, "station_id", meta["station_id"])
                if estimate_RH:
                    df = self._attach_estimated_rh(df)
                path_out = self._climate_output_path(
                    output_folder, source, native_label, meta["country"],
                    meta["station_id"], start_date, end_date,
                )
                df.to_csv(path_out, index=False)
                _keep_or_drop(path_out, meta["station_id"])
            return {
                "source": source,
                "native_interval": interval,
                "bbox": bbox,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "stations_count": 1,
                "station_ids": [meta["station_id"]],
                "separate_stations": True,
                "output_files": output_files,
                "failed": failed,
                "removed_incomplete": removed_incomplete,
                "missing_data_tolerance_ratio": missing_data_tolerance_ratio,
                "n_jobs": n_jobs,
            }

        # ---- imgw (archive bulk, optional station filter) ----
        if source == "imgw":
            filter_ids = []
            for s in station_meta:
                if s.get("name"):
                    filter_ids.append(str(s["name"]).upper())
                if s.get("wmo_id") is not None:
                    filter_ids.append(str(s["wmo_id"]))
                filter_ids.append(s["station_id"])
            try:
                df = self._download_meteo_imgw(
                    start_date,
                    end_date,
                    interval=interval,
                    rank=imgw_rank,
                    station_ids=filter_ids,
                    timeout=timeout,
                    verbose=verbose,
                )
            except Exception as exc:
                raise RuntimeError(f"imgw download failed: {exc}") from exc
            country = "POLAND"
            if df is None or df.empty:
                failed = [s["station_id"] for s in station_meta]
            else:
                if estimate_RH:
                    df = self._attach_estimated_rh(df)
                path_out = self._climate_output_path(
                    output_folder, source, native_label, country,
                    "combined", start_date, end_date, combined=True,
                )
                df.to_csv(path_out, index=False)
                _keep_or_drop(path_out, "combined")
            return {
                "source": source,
                "native_interval": interval,
                "bbox": bbox,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "stations_count": len(station_meta),
                "station_ids": [s["station_id"] for s in station_meta],
                "separate_stations": False,
                "output_files": output_files,
                "failed": failed,
                "removed_incomplete": removed_incomplete,
                "missing_data_tolerance_ratio": missing_data_tolerance_ratio,
                "n_jobs": n_jobs,
            }

        # ---- per-station sources ----
        # Ogimet country-mode skips per-station downloads (single bulk request).
        run_per_station = not (source == "ogimet" and country_name)
        n_workers = max(1, int(n_jobs or 1)) if run_per_station else 1
        # Quieter per-station HTTP logs when parallel to avoid interleaved spam.
        worker_verbose = bool(verbose) and n_workers <= 1

        def _fetch_station(meta):
            """Download one station; return a result dict (thread-safe)."""
            station_id = meta["station_id"]
            country = meta.get("country") or "UNKNOWN"
            wmo_id = meta.get("wmo_id")
            try:
                if source == "noaa_hourly":
                    df = self._download_meteo_noaa_hourly_station(
                        station_id=station_id,
                        years=years,
                        fm12=fm12,
                        timeout=timeout,
                        verbose=worker_verbose,
                    )
                    if df is not None and not df.empty:
                        mask = (df["date"] >= start_dt) & (df["date"] <= end_dt_inclusive)
                        df = df.loc[mask].copy()
                elif source == "ogimet":
                    if wmo_id is None:
                        raise RuntimeError("Could not derive WMO id for ogimet.")
                    df = self._download_meteo_ogimet_station(
                        wmo_id=wmo_id,
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval,
                        backend=ogimet_backend if ogimet_backend is not None else "synop",
                        country_name=None,
                        timeout=timeout,
                        verbose=worker_verbose,
                    )
                    # Ogimet observation times are UTC; clip to the UTC request window.
                    if df is not None and not df.empty and "date" in df.columns:
                        dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
                        mask = (dates >= start_dt) & (dates <= end_dt_inclusive)
                        df = df.loc[mask].copy()
                        df["date"] = dates.loc[mask]
                elif source == "sounding_wyoming":
                    if wmo_id is None:
                        raise RuntimeError("Could not derive WMO id for sounding_wyoming.")
                    df = self._download_sounding_wyoming_station(
                        wmo_id=wmo_id,
                        start_date=start_date,
                        end_date=end_date,
                        hours=tuple(sounding_hours),
                        timeout=timeout,
                        verbose=worker_verbose,
                    )
                else:
                    raise RuntimeError(f"Unhandled source: {source}")
            except Exception as exc:
                return {
                    "status": "failed",
                    "station_id": station_id,
                    "error": str(exc),
                    "df": None,
                    "path_out": None,
                    "sid_for_name": None,
                }

            if df is None or df.empty:
                return {
                    "status": "empty",
                    "station_id": station_id,
                    "error": None,
                    "df": None,
                    "path_out": None,
                    "sid_for_name": None,
                }

            if "station_id" not in df.columns:
                df.insert(0, "station_id", station_id)
            if "country" not in df.columns:
                df.insert(1, "country", country)

            if estimate_RH:
                df = self._attach_estimated_rh(df)

            sid_for_name = station_id if source != "ogimet" else (wmo_id or station_id)
            path_out = None
            if separate_stations:
                path_out = self._climate_output_path(
                    output_folder, source, native_label, country,
                    sid_for_name,
                    start_date, end_date,
                )
                df.to_csv(path_out, index=False)
                return {
                    "status": "ok",
                    "station_id": station_id,
                    "error": None,
                    "df": None,
                    "path_out": path_out,
                    "sid_for_name": sid_for_name,
                }
            return {
                "status": "ok",
                "station_id": station_id,
                "error": None,
                "df": df,
                "path_out": None,
                "sid_for_name": sid_for_name,
            }

        def _commit_station_result(result):
            station_id = result["station_id"]
            status = result["status"]
            if status == "failed":
                failed.append(station_id)
                if verbose:
                    print(f"[download_in_situ_data] Failed {station_id}: {result['error']}")
                return
            if status == "empty":
                failed.append(station_id)
                if verbose:
                    print(f"[download_in_situ_data] No data for {station_id}")
                return
            if separate_stations and result["path_out"]:
                _keep_or_drop(result["path_out"], result["sid_for_name"])
            elif result["df"] is not None:
                combined_frames.append(result["df"])

        if run_per_station:
            if verbose and n_workers > 1:
                print(
                    f"[download_in_situ_data] Parallel station downloads: "
                    f"n_jobs={n_workers}, stations={len(station_meta)}"
                )

            if n_workers <= 1:
                iterator = station_meta
                if tqdm is not None:
                    iterator = tqdm(station_meta, desc=f"climate {source}", unit="station")
                for meta in iterator:
                    if verbose and tqdm is None:
                        print(
                            f"[download_in_situ_data] Downloading {source}: "
                            f"{meta['station_id']} ({meta.get('country') or 'UNKNOWN'})"
                        )
                    _commit_station_result(_fetch_station(meta))
            else:
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = {
                        executor.submit(_fetch_station, meta): meta["station_id"]
                        for meta in station_meta
                    }
                    done_iter = concurrent.futures.as_completed(futures)
                    if tqdm is not None:
                        done_iter = tqdm(
                            done_iter,
                            total=len(futures),
                            desc=f"climate {source}",
                            unit="station",
                        )
                    for future in done_iter:
                        _commit_station_result(future.result())

        # Ogimet country-mode bulk (optional)
        if source == "ogimet" and country_name:
            try:
                df = self._download_meteo_ogimet_station(
                    wmo_id=0,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    backend=ogimet_backend if ogimet_backend is not None else "synop",
                    country_name=country_name,
                    timeout=timeout,
                    verbose=verbose,
                )
                if df is not None and not df.empty:
                    if "date" in df.columns:
                        dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
                        mask = (dates >= start_dt) & (dates <= end_dt_inclusive)
                        df = df.loc[mask].copy()
                        df["date"] = dates.loc[mask]
                    if estimate_RH:
                        df = self._attach_estimated_rh(df)
                    path_out = self._climate_output_path(
                        output_folder, source, native_label,
                        str(country_name).upper().replace(" ", "_"),
                        "COUNTRY",
                        start_date, end_date, combined=True,
                    )
                    df.to_csv(path_out, index=False)
                    _keep_or_drop(path_out, "COUNTRY")
            except Exception as exc:
                if verbose:
                    print(f"[download_in_situ_data] Ogimet country_name failed: {exc}")

        if not separate_stations and combined_frames:
            combined = pd.concat(combined_frames, ignore_index=True)
            if estimate_RH:
                combined = self._attach_estimated_rh(combined)
            path_out = self._climate_output_path(
                output_folder, source, native_label, "MULTI",
                "combined", start_date, end_date, combined=True,
            )
            combined.to_csv(path_out, index=False)
            _keep_or_drop(path_out, "combined")

        if verbose:
            print(
                f"[download_in_situ_data] Done: saved={len(output_files)}, "
                f"failed/empty={len(failed)}, "
                f"removed_incomplete={len(removed_incomplete)}"
            )

        return {
            "source": source,
            "native_interval": interval,
            "bbox": bbox,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "stations_count": len(station_meta),
            "station_ids": [s["station_id"] for s in station_meta],
            "separate_stations": bool(separate_stations),
            "output_files": output_files,
            "failed": failed,
            "removed_incomplete": removed_incomplete,
            "missing_data_tolerance_ratio": missing_data_tolerance_ratio,
            "n_jobs": n_jobs,
        }


    def download_in_situ_data_noaa(
        self,
        roi,
        start_date,
        end_date,
        variables_list=None,
        dataset="global-hourly",
        separate_stations=True,
        output_folder="data/noaa",
        include_attributes=False,
        units=None,
        timeout=120,
        verbose=True,
    ):
        """
        Discover NOAA stations inside an ROI and download in-situ weather data as CSV.

        Parameters
        ----------
        roi : list | tuple | str
            Bounding box [min_lon, min_lat, max_lon, max_lat] or vector file path.
            When a vector file is provided, stations are filtered to those whose
            point locations fall inside the vector geometry (not just its bbox).
        variables_list : list[str] | tuple[str] | None
            Optional and ignored for station discovery/download scope. The
            method downloads all available station data for the requested
            dataset/date range.
        start_date : str
            ISO date string (YYYY-MM-DD) for request start.
        end_date : str
            ISO date string (YYYY-MM-DD) for request end.
        dataset : str, optional
            NOAA dataset id exposed by data-service API. Default is "global-hourly".
        separate_stations : bool, optional
            When True (default), write one CSV per station. If False, write one
            combined CSV with all discovered stations.
        output_folder : str, optional
            Folder where output CSV file(s) are written.
        include_attributes : bool, optional
            Forwarded to NOAA ``includeAttributes`` parameter.
        units : str | None, optional
            NOAA units conversion: "metric" or "standard".
        timeout : int | float, optional
            HTTP timeout in seconds for NOAA requests.
        verbose : bool, optional
            Print progress information.

        Returns
        -------
        dict
            {
              "dataset": str,
              "bbox": [min_lon, min_lat, max_lon, max_lat],
              "stations_count": int,
              "station_ids": list[str],
              "separate_stations": bool,
              "output_files": list[str],
            }
        """
        if roi is None:
            raise ValueError("roi is required and must be a bbox or vector file path.")
        if variables_list is not None and not isinstance(variables_list, (list, tuple)):
            raise ValueError("variables_list must be None, list, or tuple.")
        if not start_date or not end_date:
            raise ValueError("start_date and end_date are required.")
        if not dataset:
            raise ValueError("dataset is required.")
        if units is not None and units not in {"metric", "standard"}:
            raise ValueError("units must be None, 'metric', or 'standard'.")

        roi_geometry = None
        if isinstance(roi, str):
            try:
                ext = os.path.splitext(roi)[1].lower()
                if ext in (".parquet", ".geoparquet"):
                    roi_gdf = gpd.read_parquet(roi)
                else:
                    roi_gdf = gpd.read_file(roi)
                if roi_gdf.crs is None or roi_gdf.crs.to_epsg() != 4326:
                    roi_gdf = roi_gdf.to_crs(epsg=4326)
                roi_geometry = roi_gdf.unary_union
            except Exception as exc:
                raise RuntimeError(f"Could not read roi vector geometry for polygon filtering: {exc}") from exc

        bbox = self._roi_to_bbox_list(roi)
        if bbox is None:
            raise ValueError("Could not resolve bbox from roi.")

        min_lon, min_lat, max_lon, max_lat = bbox
        # NOAA API expects bbox as N,W,S,E.
        noaa_bbox = f"{max_lat},{min_lon},{min_lat},{max_lon}"

        os.makedirs(output_folder, exist_ok=True)

        endpoint = "https://www.ncei.noaa.gov/access/services/data/v1"
        search_endpoint = "https://www.ncei.noaa.gov/access/services/search/v1/data"

        def _bool01(v):
            return "1" if bool(v) else "0"

        def _safe_station_token(station_id):
            sid = str(station_id).strip()
            return re.sub(r"[^A-Za-z0-9._-]+", "_", sid) or "unknown_station"

        station_id_candidates = (
            "station",
            "STATION",
            "stationId",
            "stationID",
            "station_id",
            "STATION_ID",
            "id",
            "ID",
        )
        station_name_candidates = (
            "name",
            "NAME",
            "stationName",
            "station_name",
            "STATION_NAME",
        )
        lat_candidates = ("latitude", "LATITUDE", "lat", "LAT")
        lon_candidates = ("longitude", "LONGITUDE", "lon", "LON")

        def _extract_first(row, keys):
            for k in keys:
                if k in row and row[k] not in (None, ""):
                    return row[k]
            return None

        def _normalize_station_id(value):
            if value in (None, ""):
                return None
            sid = str(value).strip()
            if not sid:
                return None
            if ":" in sid:
                sid = sid.split(":")[-1].strip()
            if sid.lower().endswith(".csv"):
                sid = sid[:-4].strip()
            sid = sid.strip("'\" ")
            if not sid:
                return None
            if re.search(r"[^A-Za-z0-9._:-]", sid):
                return None
            return sid

        base_params = {
            "dataset": dataset,
            "startDate": str(start_date),
            "endDate": str(end_date),
            "bbox": noaa_bbox,
            "includeAttributes": _bool01(include_attributes),
            "includeStationName": "1",
            "includeStationLocation": "1",
        }
        if units is not None:
            base_params["units"] = units

        discover_params = dict(base_params)
        discover_params["format"] = "json"

        def _extract_station_ids_recursive(obj, out_set):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    lk = str(k).lower()
                    if (
                        lk in {"station", "stationid", "station_id", "id"}
                        and not isinstance(v, (dict, list, tuple, set))
                        and v not in (None, "")
                    ):
                        out_set.add(str(v).strip())
                    elif lk == "stations" and v not in (None, ""):
                        if isinstance(v, str):
                            for sid in v.split(","):
                                sid = sid.strip()
                                if sid:
                                    out_set.add(sid)
                        elif isinstance(v, (list, tuple, set)):
                            for sid in v:
                                if sid not in (None, ""):
                                    out_set.add(str(sid).strip())
                        elif isinstance(v, dict):
                            buckets = v.get("buckets")
                            if isinstance(buckets, list):
                                for b in buckets:
                                    if isinstance(b, dict):
                                        sid = b.get("key") or b.get("station") or b.get("id")
                                        if sid not in (None, ""):
                                            out_set.add(str(sid).strip())
                    _extract_station_ids_recursive(v, out_set)
            elif isinstance(obj, (list, tuple, set)):
                for it in obj:
                    _extract_station_ids_recursive(it, out_set)

        def _rows_from_payload(payload):
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                for key in ("results", "data", "items", "stations"):
                    rows = payload.get(key)
                    if isinstance(rows, list):
                        return rows
                return [payload]
            return []

        if verbose:
            print("[download_in_situ_data_noaa] Discovering stations in ROI...")

        records = []
        station_ids_fallback = set()
        try:
            discover_resp = requests.get(endpoint, params=discover_params, timeout=timeout)
            discover_resp.raise_for_status()
            try:
                discover_payload = discover_resp.json()
            except Exception as exc:
                raise RuntimeError("NOAA discovery response is not valid JSON.") from exc
            records = _rows_from_payload(discover_payload)
            _extract_station_ids_recursive(discover_payload, station_ids_fallback)
        except requests.HTTPError as exc:
            body = ""
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            try:
                body = exc.response.text[:1000] if getattr(exc, "response", None) is not None else ""
            except Exception:
                pass

            # NOAA may reject bbox discovery on data/v1 and require explicit stations.
            if status_code == 400 and "station is required" in body.lower():
                search_params = {
                    "dataset": dataset,
                    "startDate": str(start_date),
                    "endDate": str(end_date),
                    "bbox": noaa_bbox,
                    "limit": 1000,
                    "offset": 0,
                }
                try:
                    search_resp = requests.get(search_endpoint, params=search_params, timeout=timeout)
                    search_resp.raise_for_status()
                    search_payload = search_resp.json()
                    records = _rows_from_payload(search_payload)
                    _extract_station_ids_recursive(search_payload, station_ids_fallback)
                except Exception as search_exc:
                    raise RuntimeError(
                        "NOAA discovery request failed: data/v1 requires stations and "
                        f"search fallback failed ({search_exc})."
                    ) from search_exc
            else:
                raise RuntimeError(
                    f"NOAA discovery request failed: HTTP {status_code if status_code is not None else 'unknown'} - {body}"
                ) from exc
        except Exception as exc:
            raise RuntimeError(f"NOAA discovery request failed: {exc}") from exc

        stations_meta = {}
        for row in records:
            if not isinstance(row, dict):
                continue
            station_id = _normalize_station_id(_extract_first(row, station_id_candidates))
            if station_id is None:
                continue
            if station_id not in stations_meta:
                stations_meta[station_id] = {
                    "station_id": station_id,
                    "station_name": _extract_first(row, station_name_candidates),
                    "latitude": _extract_first(row, lat_candidates),
                    "longitude": _extract_first(row, lon_candidates),
                }

        if station_ids_fallback:
            for sid in station_ids_fallback:
                nsid = _normalize_station_id(sid)
                if nsid is None or nsid in stations_meta:
                    continue
                stations_meta[nsid] = {
                    "station_id": nsid,
                    "station_name": None,
                    "latitude": None,
                    "longitude": None,
                }

        # Enrich station metadata (name/lat/lon) using NOAA data endpoint.
        # Search/discovery responses may not include station coordinates.
        station_ids_for_enrich = sorted(stations_meta.keys())
        if station_ids_for_enrich:
            preferred_fallback_vars = ["TMAX", "TMIN", "PRCP", "TAVG", "AWND"]
            enrich_data_types = []
            for v in preferred_fallback_vars:
                if v and v not in enrich_data_types:
                    enrich_data_types.append(v)
            if not enrich_data_types:
                enrich_data_types = ["TMAX"]

            batch_size = 80
            batch_starts = list(range(0, len(station_ids_for_enrich), batch_size))
            enrich_iter = tqdm(batch_starts, desc="NOAA metadata enrichment", unit="batch")
            for i in enrich_iter:
                batch = station_ids_for_enrich[i:i + batch_size]
                for dt_var in enrich_data_types:
                    enrich_params = {
                        "dataset": dataset,
                        "stations": ",".join(batch),
                        "startDate": str(start_date),
                        "endDate": str(end_date),
                        "dataTypes": dt_var,
                        "includeStationName": "1",
                        "includeStationLocation": "1",
                        "format": "json",
                    }
                    try:
                        enrich_resp = requests.get(endpoint, params=enrich_params, timeout=timeout)
                        enrich_resp.raise_for_status()
                        enrich_rows = enrich_resp.json()
                        if not isinstance(enrich_rows, list):
                            continue
                        for erow in enrich_rows:
                            if not isinstance(erow, dict):
                                continue
                            sid = _normalize_station_id(_extract_first(erow, station_id_candidates))
                            if sid is None or sid not in stations_meta:
                                continue
                            if stations_meta[sid].get("station_name") in (None, ""):
                                stations_meta[sid]["station_name"] = _extract_first(erow, station_name_candidates)
                            if stations_meta[sid].get("latitude") in (None, ""):
                                stations_meta[sid]["latitude"] = _extract_first(erow, lat_candidates)
                            if stations_meta[sid].get("longitude") in (None, ""):
                                stations_meta[sid]["longitude"] = _extract_first(erow, lon_candidates)
                    except Exception as exc:
                        if verbose:
                            print(
                                "[download_in_situ_data_noaa] Metadata enrichment batch failed "
                                f"for stations {i}..{min(i + batch_size - 1, len(station_ids_for_enrich) - 1)} "
                                f"using dataType={dt_var}: {exc}"
                            )

                if verbose:
                    enriched_coords = 0
                    for sid in batch:
                        lat = stations_meta.get(sid, {}).get("latitude")
                        lon = stations_meta.get(sid, {}).get("longitude")
                        if lat not in (None, "") and lon not in (None, ""):
                            enriched_coords += 1
                    print(
                        "[download_in_situ_data_noaa] Metadata enrichment batch "
                        f"{i}..{min(i + batch_size - 1, len(station_ids_for_enrich) - 1)}: "
                        f"{enriched_coords}/{len(batch)} station(s) with coordinates"
                    )
                enrich_iter.set_postfix_str(
                    f"coords {sum(1 for sid in batch if stations_meta.get(sid, {}).get('latitude') not in (None, '') and stations_meta.get(sid, {}).get('longitude') not in (None, ''))}/{len(batch)}"
                )

        if roi_geometry is not None and stations_meta:
            stations_df_filter = pd.DataFrame(stations_meta.values())
            stations_df_filter["latitude"] = pd.to_numeric(stations_df_filter.get("latitude"), errors="coerce")
            stations_df_filter["longitude"] = pd.to_numeric(stations_df_filter.get("longitude"), errors="coerce")
            stations_df_filter = stations_df_filter.dropna(subset=["latitude", "longitude"]).copy()

            if not stations_df_filter.empty:
                stations_points = gpd.GeoDataFrame(
                    stations_df_filter,
                    geometry=gpd.points_from_xy(
                        stations_df_filter["longitude"],
                        stations_df_filter["latitude"],
                        crs="EPSG:4326",
                    ),
                    crs="EPSG:4326",
                )
                inside_mask = stations_points.intersects(roi_geometry)
                keep_ids = set(stations_points.loc[inside_mask, "station_id"].astype(str).tolist())
                before_count = len(stations_meta)
                stations_meta = {k: v for k, v in stations_meta.items() if str(k) in keep_ids}
                if verbose:
                    print(
                        "[download_in_situ_data_noaa] Vector ROI filter applied: "
                        f"{len(stations_meta)} of {before_count} station(s) inside polygon geometry."
                    )
            else:
                if verbose:
                    print(
                        "[download_in_situ_data_noaa] Vector ROI filter skipped: "
                        "no station coordinates available after metadata enrichment."
                    )

        station_ids = sorted(stations_meta.keys())
        if verbose:
            print(f"[download_in_situ_data_noaa] Stations found: {len(station_ids)}")

        if not station_ids:
            return {
                "dataset": dataset,
                "bbox": bbox,
                "stations_count": 0,
                "station_ids": [],
                "separate_stations": bool(separate_stations),
                "output_files": [],
            }

        output_files = []

        if separate_stations:
            station_iter = tqdm(station_ids, desc="NOAA station downloads", unit="station")
            for station_id in station_iter:
                params = dict(base_params)
                params["stations"] = station_id
                params["format"] = "csv"
                if verbose:
                    print(f"[download_in_situ_data_noaa] Downloading station: {station_id}")
                station_iter.set_postfix_str(station_id)

                try:
                    resp = requests.get(endpoint, params=params, timeout=timeout)
                    resp.raise_for_status()
                except requests.HTTPError as exc:
                    raise RuntimeError(
                        f"NOAA station download failed for '{station_id}': HTTP {resp.status_code} - {resp.text[:1000]}"
                    ) from exc
                except Exception as exc:
                    raise RuntimeError(f"NOAA station download failed for '{station_id}': {exc}") from exc

                station_token = _safe_station_token(station_id)
                file_path = os.path.join(
                    output_folder,
                    f"noaa_{dataset}_{station_token}_{start_date}_{end_date}.csv",
                )
                with open(file_path, "wb") as fh:
                    fh.write(resp.content)
                output_files.append(file_path)
        else:
            params = dict(base_params)
            params["stations"] = ",".join(station_ids)
            params["format"] = "csv"
            if verbose:
                print(
                    "[download_in_situ_data_noaa] Downloading combined CSV for "
                    f"{len(station_ids)} station(s)..."
                )

            try:
                resp = requests.get(endpoint, params=params, timeout=timeout)
                resp.raise_for_status()
            except requests.HTTPError as exc:
                raise RuntimeError(
                    f"NOAA combined download failed: HTTP {resp.status_code} - {resp.text[:1000]}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(f"NOAA combined download failed: {exc}") from exc

            file_path = os.path.join(
                output_folder,
                f"noaa_{dataset}_combined_{start_date}_{end_date}.csv",
            )
            with open(file_path, "wb") as fh:
                fh.write(resp.content)
            output_files.append(file_path)

        if verbose:
            print(f"[download_in_situ_data_noaa] Saved {len(output_files)} file(s).")

        return {
            "dataset": dataset,
            "bbox": bbox,
            "stations_count": len(station_ids),
            "station_ids": station_ids,
            "separate_stations": bool(separate_stations),
            "output_files": output_files,
        }

    def list_stations_noaa(
        self,
        roi,
        dataset="global-hourly",
        start_date=None,
        end_date=None,
        limit=5000,
        timeout=120,
        verbose=True,
        export=False,
        output_file="data/noaa/noaa_station_variable_report.txt",
    ):
        """
        List NOAA stations and available variables for a given ROI.

        Parameters
        ----------
        roi : list | tuple | str
            Bounding box [min_lon, min_lat, max_lon, max_lat] or vector file path.
        dataset : str, optional
            NOAA dataset id to query. Default is "global-hourly".
            Common dataset ids include:
            - "global-hourly"
            - "daily-summaries"
            - "global-summary-of-the-day"
            - "global-summary-of-the-month"
            - "global-summary-of-the-year"
            Use a dataset id supported by NOAA Access Search Service for your
            target variables and period.
        start_date : str | None, optional
            Optional ISO date (YYYY-MM-DD) to constrain station/variable discovery.
        end_date : str | None, optional
            Optional ISO date (YYYY-MM-DD) to constrain station/variable discovery.
        limit : int, optional
            Search-service limit forwarded to NOAA. Higher values may improve
            fallback extraction from ``results``.
        timeout : int | float, optional
            HTTP timeout in seconds for NOAA requests.
        verbose : bool, optional
            Print progress information.
        export : bool, optional
            If True, export the station/variable report to a text file.
        output_file : str, optional
            Text report path used when ``export=True``. The stations parquet file
            is written alongside it using the same base name with
            ``_stations.parquet`` suffix as GeoParquet (EPSG:4326).

        Returns
        -------
        dict
            {
              "dataset": str,
              "bbox": [min_lon, min_lat, max_lon, max_lat],
              "start_date": str | None,
              "end_date": str | None,
              "stations_count": int,
              "variables_count": int,
              "stations": list[dict],
              "variables": list[dict],
              "total_count": int | None,
              "report_file": str | None,
                            "stations_parquet_file": str | None,
            }
        """
        if roi is None:
            raise ValueError("roi is required and must be a bbox or vector file path.")
        if not dataset:
            raise ValueError("dataset is required.")
        if start_date and not end_date:
            raise ValueError("end_date is required when start_date is provided.")
        if end_date and not start_date:
            raise ValueError("start_date is required when end_date is provided.")

        roi_geometry = None
        if isinstance(roi, str):
            try:
                ext = os.path.splitext(roi)[1].lower()
                if ext in (".parquet", ".geoparquet"):
                    roi_gdf = gpd.read_parquet(roi)
                else:
                    roi_gdf = gpd.read_file(roi)
                if roi_gdf.crs is None or roi_gdf.crs.to_epsg() != 4326:
                    roi_gdf = roi_gdf.to_crs(epsg=4326)
                roi_geometry = roi_gdf.unary_union
            except Exception as exc:
                raise RuntimeError(f"Could not read roi vector geometry for polygon filtering: {exc}") from exc

        bbox = self._roi_to_bbox_list(roi)
        if bbox is None:
            raise ValueError("Could not resolve bbox from roi.")

        min_lon, min_lat, max_lon, max_lat = bbox
        # NOAA API expects bbox as N,W,S,E.
        noaa_bbox = f"{max_lat},{min_lon},{min_lat},{max_lon}"

        endpoint = "https://www.ncei.noaa.gov/access/services/search/v1/data"

        if verbose:
            print("[list_stations_noaa] Querying NOAA search service...")

        params = {
            "dataset": dataset,
            "bbox": noaa_bbox,
            "limit": int(limit),
            "offset": 0,
        }
        if start_date and end_date:
            params["startDate"] = str(start_date)
            params["endDate"] = str(end_date)

        try:
            resp = requests.get(endpoint, params=params, timeout=timeout)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            body = ""
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            try:
                body = exc.response.text[:1000] if getattr(exc, "response", None) is not None else ""
            except Exception:
                pass
            raise RuntimeError(
                f"NOAA station/variable listing failed: HTTP {status_code if status_code is not None else 'unknown'} - {body}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"NOAA station/variable listing failed: {exc}") from exc

        try:
            payload = resp.json()
        except Exception as exc:
            raise RuntimeError("NOAA search response is not valid JSON.") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("NOAA search response did not return a JSON object.")

        station_name_candidates = (
            "name",
            "NAME",
            "stationName",
            "station_name",
            "STATION_NAME",
        )
        station_id_candidates = (
            "station",
            "STATION",
            "stationId",
            "stationID",
            "station_id",
            "STATION_ID",
            "id",
            "ID",
        )
        lat_candidates = ("latitude", "LATITUDE", "lat", "LAT")
        lon_candidates = ("longitude", "LONGITUDE", "lon", "LON")

        def _extract_first(row, keys):
            for k in keys:
                if k in row and row[k] not in (None, ""):
                    return row[k]
            return None

        def _extract_datatype_entries(value):
            """Return list of {'id': <datatype_id>, 'name': <datatype_name_or_none>} entries."""
            out = []
            if value in (None, ""):
                return out

            def _append_entry(dt_id, dt_name=None):
                if dt_id in (None, ""):
                    return
                dt_id = str(dt_id).strip()
                if not dt_id:
                    return
                if dt_name in (None, ""):
                    out.append({"id": dt_id, "name": None})
                else:
                    out.append({"id": dt_id, "name": str(dt_name).strip()})

            if isinstance(value, dict):
                _append_entry(
                    value.get("id") or value.get("key") or value.get("dataType"),
                    value.get("name") or value.get("label") or value.get("description"),
                )
                return out

            if isinstance(value, str):
                for tok in value.split(","):
                    tok = tok.strip()
                    if tok:
                        _append_entry(tok)
                return out

            if isinstance(value, (list, tuple, set)):
                for item in value:
                    if isinstance(item, dict):
                        _append_entry(
                            item.get("id") or item.get("key") or item.get("dataType"),
                            item.get("name") or item.get("label") or item.get("description"),
                        )
                    elif item not in (None, ""):
                        _append_entry(str(item).strip())
                return out

            _append_entry(str(value).strip())
            return out

        def _normalize_station_id(value):
            if value in (None, ""):
                return None
            sid = str(value).strip()
            if not sid:
                return None
            if ":" in sid:
                sid = sid.split(":")[-1].strip()
            if sid.lower().endswith(".csv"):
                sid = sid[:-4].strip()
            sid = sid.strip("'\" ")
            if not sid:
                return None
            # Keep practical NOAA ids; reject obvious non-id payload artifacts.
            if re.search(r"[^A-Za-z0-9._:-]", sid):
                return None
            return sid

        stations_map = {}
        variables_map = {}

        station_buckets = ((payload.get("stations") or {}).get("buckets") if isinstance(payload.get("stations"), dict) else None)
        if isinstance(station_buckets, list):
            for b in station_buckets:
                if not isinstance(b, dict):
                    continue
                sid = _normalize_station_id(b.get("key") or b.get("station") or b.get("id"))
                if sid is None:
                    continue
                stations_map[sid] = {
                    "station_id": sid,
                    "station_name": None,
                    "latitude": None,
                    "longitude": None,
                    "records_count": int(b.get("docCount", 0)) if str(b.get("docCount", "")).isdigit() else b.get("docCount"),
                }

        data_type_buckets = ((payload.get("dataTypes") or {}).get("buckets") if isinstance(payload.get("dataTypes"), dict) else None)
        if isinstance(data_type_buckets, list):
            for b in data_type_buckets:
                if not isinstance(b, dict):
                    continue
                dt_raw = b.get("key") or b.get("dataType") or b.get("id")
                dts = _extract_datatype_entries(dt_raw)
                if not dts:
                    continue
                for dt in dts:
                    dt_id = dt["id"]
                    variables_map[dt_id] = {
                        "variable": dt_id,
                        "variable_name": dt.get("name"),
                        "records_count": int(b.get("docCount", 0)) if str(b.get("docCount", "")).isdigit() else b.get("docCount"),
                    }

        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        for row in results:
            if not isinstance(row, dict):
                continue

            station_id = _normalize_station_id(_extract_first(row, station_id_candidates))
            if station_id is not None:
                if station_id not in stations_map:
                    stations_map[station_id] = {
                        "station_id": station_id,
                        "station_name": _extract_first(row, station_name_candidates),
                        "latitude": _extract_first(row, lat_candidates),
                        "longitude": _extract_first(row, lon_candidates),
                        "records_count": None,
                    }
                else:
                    if stations_map[station_id].get("station_name") in (None, ""):
                        stations_map[station_id]["station_name"] = _extract_first(row, station_name_candidates)
                    if stations_map[station_id].get("latitude") in (None, ""):
                        stations_map[station_id]["latitude"] = _extract_first(row, lat_candidates)
                    if stations_map[station_id].get("longitude") in (None, ""):
                        stations_map[station_id]["longitude"] = _extract_first(row, lon_candidates)

            for key in ("dataTypes", "dataType", "datatype", "datatypes"):
                for dt in _extract_datatype_entries(row.get(key)):
                    dt_id = dt["id"]
                    if dt_id not in variables_map:
                        variables_map[dt_id] = {
                            "variable": dt_id,
                            "variable_name": dt.get("name"),
                            "records_count": None,
                        }
                    elif variables_map[dt_id].get("variable_name") in (None, "") and dt.get("name") not in (None, ""):
                        variables_map[dt_id]["variable_name"] = dt.get("name")

        # Enrich station metadata (name/lat/lon) using NOAA data endpoint.
        # Search service often returns station ids without station coordinates.
        if stations_map and start_date and end_date:
            enrich_endpoint = "https://www.ncei.noaa.gov/access/services/data/v1"
            station_ids_for_enrich = sorted(stations_map.keys())
            candidate_vars = sorted([str(v).strip() for v in variables_map.keys() if str(v).strip()])
            preferred_vars = ["TMAX", "TMIN", "PRCP", "TAVG", "AWND"]
            data_type_for_enrich = next((v for v in preferred_vars if v in candidate_vars), None)
            if data_type_for_enrich is None:
                data_type_for_enrich = next(
                    (v for v in candidate_vars if re.fullmatch(r"[A-Za-z0-9_-]+", v)),
                    "TMAX",
                )

            batch_size = 80
            for i in range(0, len(station_ids_for_enrich), batch_size):
                batch = station_ids_for_enrich[i:i + batch_size]
                enrich_params = {
                    "dataset": dataset,
                    "stations": ",".join(batch),
                    "startDate": str(start_date),
                    "endDate": str(end_date),
                    "dataTypes": data_type_for_enrich,
                    "includeStationName": "1",
                    "includeStationLocation": "1",
                    "format": "json",
                }
                try:
                    enrich_resp = requests.get(enrich_endpoint, params=enrich_params, timeout=timeout)
                    enrich_resp.raise_for_status()
                    enrich_rows = enrich_resp.json()
                    if not isinstance(enrich_rows, list):
                        continue
                    for erow in enrich_rows:
                        if not isinstance(erow, dict):
                            continue
                        sid = _normalize_station_id(_extract_first(erow, station_id_candidates))
                        if sid is None or sid not in stations_map:
                            continue
                        existing_name = stations_map[sid].get("station_name")
                        enriched_name = _extract_first(erow, station_name_candidates)
                        if existing_name in (None, ""):
                            stations_map[sid]["station_name"] = enriched_name
                        elif isinstance(existing_name, str):
                            low = existing_name.lower()
                            if low.endswith(".csv") or ".tar.gz:" in low:
                                stations_map[sid]["station_name"] = enriched_name
                        if stations_map[sid].get("latitude") in (None, ""):
                            stations_map[sid]["latitude"] = _extract_first(erow, lat_candidates)
                        if stations_map[sid].get("longitude") in (None, ""):
                            stations_map[sid]["longitude"] = _extract_first(erow, lon_candidates)
                except Exception as exc:
                    if verbose:
                        print(
                            "[list_stations_noaa] Metadata enrichment batch failed "
                            f"for stations {i}..{min(i + batch_size - 1, len(station_ids_for_enrich) - 1)}: {exc}"
                        )

        if roi_geometry is not None and stations_map:
            stations_df_filter = pd.DataFrame(stations_map.values())
            stations_df_filter["latitude"] = pd.to_numeric(stations_df_filter.get("latitude"), errors="coerce")
            stations_df_filter["longitude"] = pd.to_numeric(stations_df_filter.get("longitude"), errors="coerce")
            stations_df_filter = stations_df_filter.dropna(subset=["latitude", "longitude"]).copy()

            if not stations_df_filter.empty:
                stations_points = gpd.GeoDataFrame(
                    stations_df_filter,
                    geometry=gpd.points_from_xy(
                        stations_df_filter["longitude"],
                        stations_df_filter["latitude"],
                        crs="EPSG:4326",
                    ),
                    crs="EPSG:4326",
                )
                inside_mask = stations_points.intersects(roi_geometry)
                keep_ids = set(stations_points.loc[inside_mask, "station_id"].astype(str).tolist())
            else:
                keep_ids = set()

            before_count = len(stations_map)
            stations_map = {k: v for k, v in stations_map.items() if str(k) in keep_ids}
            if verbose:
                print(
                    "[list_stations_noaa] Vector ROI filter applied: "
                    f"{len(stations_map)} of {before_count} station(s) inside polygon geometry."
                )

        stations = sorted(stations_map.values(), key=lambda x: str(x.get("station_id", "")))
        variables = sorted(
            [
                v
                for v in variables_map.values()
                if re.fullmatch(r"[A-Za-z0-9_-]+", str(v.get("variable", "")).strip())
            ],
            key=lambda x: str(x.get("variable", "")),
        )

        report_file = None
        stations_parquet_file = None
        if export:
            folder = os.path.dirname(output_file)
            if folder:
                os.makedirs(folder, exist_ok=True)

            if hasattr(datetime, "datetime"):
                timestamp_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                timestamp_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            lines = [
                "NOAA Station and Variable Discovery Report",
                "=" * 42,
                f"Generated: {timestamp_utc}",
                f"Dataset: {dataset}",
                f"BBox [min_lon,min_lat,max_lon,max_lat]: {bbox}",
                f"NOAA bbox (N,W,S,E): {noaa_bbox}",
                f"Date range: {start_date if start_date else 'not specified'} -> {end_date if end_date else 'not specified'}",
                f"Stations found: {len(stations)}",
                f"Variables found: {len(variables)}",
                "",
                "Available Variables:",
            ]
            if variables:
                for v in variables:
                    vname = v.get("variable_name")
                    if vname not in (None, ""):
                        lines.append(f"- {v['variable']} - {vname} (records_count={v.get('records_count')})")
                    else:
                        lines.append(f"- {v['variable']} (records_count={v.get('records_count')})")
            else:
                lines.append("- None")

            lines.append("")
            lines.append("Available Stations:")
            if stations:
                for s in stations:
                    sid = s.get("station_id")
                    sname = s.get("station_name")
                    lat = s.get("latitude")
                    lon = s.get("longitude")
                    rc = s.get("records_count")
                    lines.append(
                        f"- {sid} | name={sname} | lat={lat} | lon={lon} | records_count={rc}"
                    )
            else:
                lines.append("- None")

            with open(output_file, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            report_file = output_file

            parquet_root, _ = os.path.splitext(output_file)
            stations_parquet_file = f"{parquet_root}_stations.parquet"
            try:
                stations_df = pd.DataFrame(stations)
                if "latitude" in stations_df.columns:
                    stations_df["latitude"] = pd.to_numeric(stations_df["latitude"], errors="coerce")
                if "longitude" in stations_df.columns:
                    stations_df["longitude"] = pd.to_numeric(stations_df["longitude"], errors="coerce")

                # Export as GeoParquet so GIS software can load geometry directly.
                stations_gdf = gpd.GeoDataFrame(
                    stations_df,
                    geometry=gpd.points_from_xy(
                        stations_df["longitude"],
                        stations_df["latitude"],
                        crs="EPSG:4326",
                    ),
                    crs="EPSG:4326",
                )
                stations_gdf.to_parquet(stations_parquet_file, index=False)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to export NOAA stations parquet: {stations_parquet_file} ({exc})"
                ) from exc

        if verbose:
            print(f"[list_stations_noaa] Stations found: {len(stations)}")
            print(f"[list_stations_noaa] Variables found: {len(variables)}")
            if report_file:
                print(f"[list_stations_noaa] Report exported to: {report_file}")
            if stations_parquet_file:
                print(f"[list_stations_noaa] Stations parquet exported to: {stations_parquet_file}")

        return {
            "dataset": dataset,
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
            "stations_count": len(stations),
            "variables_count": len(variables),
            "stations": stations,
            "variables": variables,
            "total_count": payload.get("totalCount", payload.get("count")),
            "report_file": report_file,
            "stations_parquet_file": stations_parquet_file,
        }
           
    
    """def learn_error(self,):
        data_x =
        data_y =

        model = model.best_model()

        model.train()
        
        for p in missing_data_row:
            data.set_row('', model.predict(data.get_row))"""
        
                