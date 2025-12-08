"""
Data loader module for Alibaba Microservices Trace v2022.
Handles loading, preprocessing, and creating datasets for workload prediction.

Enhanced version:
- Loads all available data (MSMetrics + MSRTMCR)
- Supports multi-source feature integration
- Improved time-based train/val/test split
- Feature engineering (lag, rolling, time features)
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import warnings
warnings.filterwarnings('ignore')

from config import DATA_CONFIG, MODEL_CONFIG, EXPERIMENT_CONFIG


class MSMetricsLoader:
    """
    Loader for MSMetrics data (microservice resource utilization).
    Fields: timestamp, msname, msinstanceid, nodeid, cpu_utilization, memory_utilization
    """
    
    def __init__(self, config: 'DataConfig' = None):
        self.config = config or DATA_CONFIG
        self.data = None
        
    def load_single_file(self, file_idx: int) -> pd.DataFrame:
        """Load a single MSMetrics file by index."""
        file_path = os.path.join(
            self.config.msmetrics_path,
            self.config.msmetrics_pattern.format(file_idx)
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        return df
    
    def load_range(self, start_idx: int = 0, end_idx: int = None, 
                   num_workers: int = 4, verbose: bool = True) -> pd.DataFrame:
        """Load multiple MSMetrics files in a range with parallel loading."""
        if end_idx is None:
            end_idx = self.config.msmetrics_range[1]
        
        dfs = []
        file_indices = list(range(start_idx, end_idx))
        
        if num_workers > 1:
            # Parallel loading
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self.load_single_file, idx): idx 
                          for idx in file_indices}
                for i, future in enumerate(as_completed(futures)):
                    idx = futures[future]
                    try:
                        df = future.result()
                        dfs.append(df)
                        if verbose and (i + 1) % 5 == 0:
                            print(f"Loaded MSMetrics file {i + 1}/{len(file_indices)}")
                    except FileNotFoundError:
                        if verbose:
                            print(f"Warning: MSMetrics file {idx} not found, skipping...")
        else:
            # Sequential loading
            for idx in file_indices:
                try:
                    df = self.load_single_file(idx)
                    dfs.append(df)
                    if verbose and idx % 5 == 0:
                        print(f"Loaded MSMetrics file {idx}/{end_idx}")
                except FileNotFoundError:
                    if verbose:
                        print(f"Warning: MSMetrics file {idx} not found, skipping...")
        
        if not dfs:
            raise ValueError("No MSMetrics files loaded")
        
        self.data = pd.concat(dfs, ignore_index=True)
        if verbose:
            print(f"Total MSMetrics records: {len(self.data):,}")
        return self.data
    
    def get_service_data(self, msname: str) -> pd.DataFrame:
        """Get data for a specific microservice."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_range() first.")
        return self.data[self.data['msname'] == msname].copy()
    
    def get_unique_services(self) -> List[str]:
        """Get list of unique microservice names."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_range() first.")
        return self.data['msname'].unique().tolist()


class MSRTMCRLoader:
    """
    Loader for MSRTMCR data (MCR: Microservice Call Rate, RT: Response Time).
    Contains metrics for different communication paradigms (RPC, MQ, HTTP, DB, MC).
    """
    
    def __init__(self, config: 'DataConfig' = None):
        self.config = config or DATA_CONFIG
        self.data = None
        
    def load_single_file(self, file_idx: int) -> pd.DataFrame:
        """Load a single MSRTMCR file by index."""
        file_path = os.path.join(
            self.config.msrtmcr_path,
            self.config.msrtmcr_pattern.format(file_idx)
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        return df
    
    def load_range(self, start_idx: int = 0, end_idx: int = None,
                   num_workers: int = 4, verbose: bool = True) -> pd.DataFrame:
        """Load multiple MSRTMCR files in a range with parallel loading."""
        if end_idx is None:
            end_idx = self.config.msrtmcr_range[1]
        
        dfs = []
        file_indices = list(range(start_idx, end_idx))
        
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self.load_single_file, idx): idx 
                          for idx in file_indices}
                for i, future in enumerate(as_completed(futures)):
                    idx = futures[future]
                    try:
                        df = future.result()
                        dfs.append(df)
                        if verbose and (i + 1) % 40 == 0:
                            print(f"Loaded MSRTMCR file {i + 1}/{len(file_indices)}")
                    except FileNotFoundError:
                        pass
        else:
            for idx in file_indices:
                try:
                    df = self.load_single_file(idx)
                    dfs.append(df)
                    if verbose and idx % 40 == 0:
                        print(f"Loaded MSRTMCR file {idx}/{end_idx}")
                except FileNotFoundError:
                    continue
        
        if not dfs:
            raise ValueError("No MSRTMCR files loaded")
        
        self.data = pd.concat(dfs, ignore_index=True)
        if verbose:
            print(f"Total MSRTMCR records: {len(self.data):,}")
        return self.data


class NodeMetricsLoader:
    """
    Loader for NodeMetrics data (bare-metal node resource utilization).
    Fields: timestamp, nodeid, cpu_utilization, memory_utilization
    """
    
    def __init__(self, config: 'DataConfig' = None):
        self.config = config or DATA_CONFIG
        self.data = None
        
    def load(self) -> pd.DataFrame:
        """Load NodeMetrics data."""
        file_path = os.path.join(
            self.config.nodemetrics_path,
            self.config.nodemetrics_pattern.format(0)
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.data = pd.read_csv(file_path)
        print(f"Total NodeMetrics records: {len(self.data):,}")
        return self.data


class CallGraphLoader:
    """
    Loader for CallGraph data (service call dependencies).
    Fields: timestamp, traceid, service, rpc_id, rpctype, um, uminstanceid, 
            interface, dm, dminstanceid, rt
    """
    
    def __init__(self, config: 'DataConfig' = None):
        self.config = config or DATA_CONFIG
        self.data = None
        
    def load_single_file(self, file_idx: int) -> pd.DataFrame:
        """Load a single CallGraph file by index."""
        file_path = os.path.join(
            self.config.callgraph_path,
            self.config.callgraph_pattern.format(file_idx)
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        return df
    
    def load_range(self, start_idx: int = 0, end_idx: int = None,
                   num_workers: int = 4, verbose: bool = True) -> pd.DataFrame:
        """Load multiple CallGraph files in a range."""
        if end_idx is None:
            end_idx = self.config.callgraph_range[1]
        
        dfs = []
        file_indices = list(range(start_idx, end_idx))
        
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self.load_single_file, idx): idx 
                          for idx in file_indices}
                for i, future in enumerate(as_completed(futures)):
                    idx = futures[future]
                    try:
                        df = future.result()
                        dfs.append(df)
                        if verbose and (i + 1) % 40 == 0:
                            print(f"Loaded CallGraph file {i + 1}/{len(file_indices)}")
                    except FileNotFoundError:
                        pass
        else:
            for idx in file_indices:
                try:
                    df = self.load_single_file(idx)
                    dfs.append(df)
                    if verbose and idx % 40 == 0:
                        print(f"Loaded CallGraph file {idx}/{end_idx}")
                except FileNotFoundError:
                    continue
        
        if not dfs:
            raise ValueError("No CallGraph files loaded")
        
        self.data = pd.concat(dfs, ignore_index=True)
        if verbose:
            print(f"Total CallGraph records: {len(self.data):,}")
        return self.data


class FeatureEngineer:
    """
    Feature engineering for time series data.
    Creates lag features, rolling statistics, and time-based features.
    """
    
    def __init__(self, config: 'ExperimentConfig' = None):
        self.config = config or EXPERIMENT_CONFIG
    
    def add_lag_features(self, df: pd.DataFrame, target_col: str, 
                         lag_steps: List[int] = None) -> pd.DataFrame:
        """Add lag features for a target column."""
        if lag_steps is None:
            lag_steps = self.config.lag_steps
        
        df = df.copy()
        for lag in lag_steps:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, target_col: str,
                            windows: List[int] = None) -> pd.DataFrame:
        """Add rolling statistics features."""
        if windows is None:
            windows = self.config.rolling_windows
        
        df = df.copy()
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window).max()
        
        return df
    
    def add_time_features(self, df: pd.DataFrame, time_col: str = 'timestamp') -> pd.DataFrame:
        """Add time-based features from timestamp."""
        df = df.copy()
        
        # Convert timestamp (ms) to datetime
        if df[time_col].dtype != 'datetime64[ns]':
            df['datetime'] = pd.to_datetime(df[time_col], unit='ms')
        else:
            df['datetime'] = df[time_col]
        
        # Extract time components
        df['minute_of_day'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / 1440)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / 1440)
        
        return df
    
    def add_diff_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add difference features (first and second order)."""
        df = df.copy()
        df[f'{target_col}_diff_1'] = df[target_col].diff(1)
        df[f'{target_col}_diff_2'] = df[target_col].diff(2)
        return df


class WorkloadDataPreprocessor:
    """
    Preprocessor for creating workload prediction datasets.
    Aggregates and processes data for time series prediction.
    """
    
    def __init__(self, config: 'DataConfig' = None):
        self.config = config or DATA_CONFIG
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.feature_engineer = FeatureEngineer()
        
    def aggregate_by_service_time(
        self, 
        df: pd.DataFrame, 
        time_col: str = 'timestamp',
        service_col: str = 'msname',
        features: List[str] = None,
        agg_func: str = 'mean'
    ) -> pd.DataFrame:
        """
        Aggregate data by microservice and timestamp.
        For instance-level data, compute mean/sum per service per timestamp.
        """
        if features is None:
            features = ['cpu_utilization', 'memory_utilization']
        
        # Filter valid features that exist in dataframe
        valid_features = [f for f in features if f in df.columns]
        if not valid_features:
            raise ValueError(f"No valid features found. Available: {df.columns.tolist()}")
        
        # Group by service and timestamp, aggregate
        agg_dict = {f: agg_func for f in valid_features}
        aggregated = df.groupby([service_col, time_col]).agg(agg_dict).reset_index()
        
        return aggregated
    
    def create_service_timeseries(
        self,
        df: pd.DataFrame,
        msname: str,
        time_col: str = 'timestamp',
        service_col: str = 'msname',
        features: List[str] = None,
        add_features: bool = True
    ) -> pd.DataFrame:
        """
        Create a time series dataframe for a specific microservice.
        Fills missing timestamps and handles NaN values.
        """
        # Filter for specific service
        service_data = df[df[service_col] == msname].copy()
        
        if len(service_data) < self.config.min_data_points:
            return None
        
        # Sort by timestamp
        service_data = service_data.sort_values(time_col)
        
        # Select features
        if features is None:
            features = self.config.msmetrics_features
        valid_features = [f for f in features if f in service_data.columns]
        
        # Keep timestamp and features
        cols_to_keep = [time_col] + valid_features
        service_data = service_data[cols_to_keep].copy()
        
        # Add engineered features if requested
        if add_features and EXPERIMENT_CONFIG.use_lag_features:
            for feat in valid_features:
                service_data = self.feature_engineer.add_lag_features(service_data, feat)
                service_data = self.feature_engineer.add_rolling_features(service_data, feat)
                service_data = self.feature_engineer.add_diff_features(service_data, feat)
        
        if add_features and EXPERIMENT_CONFIG.use_time_features:
            service_data = self.feature_engineer.add_time_features(service_data, time_col)
        
        # Set timestamp as index
        service_data = service_data.set_index(time_col)
        
        # Remove non-numeric columns
        service_data = service_data.select_dtypes(include=[np.number])
        
        # Fill missing values
        if self.config.fillna_method == 'ffill':
            service_data = service_data.ffill().bfill()
        elif self.config.fillna_method == 'zero':
            service_data = service_data.fillna(0)
        else:
            service_data = service_data.fillna(service_data.mean())
        
        # Drop rows with any remaining NaN
        service_data = service_data.dropna()
        
        return service_data
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        features: List[str] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize features using MinMaxScaler.
        """
        df_norm = df.copy()
        
        if features is None:
            features = df.columns.tolist()
        
        for feature in features:
            if feature not in df.columns:
                continue
                
            if fit or feature not in self.scalers:
                self.scalers[feature] = MinMaxScaler()
                df_norm[feature] = self.scalers[feature].fit_transform(
                    df[[feature]].values
                )
            else:
                df_norm[feature] = self.scalers[feature].transform(
                    df[[feature]].values
                )
        
        return df_norm
    
    def inverse_normalize(
        self,
        values: np.ndarray,
        feature: str
    ) -> np.ndarray:
        """
        Inverse transform normalized values back to original scale.
        """
        if feature not in self.scalers:
            return values
        
        return self.scalers[feature].inverse_transform(
            values.reshape(-1, 1)
        ).flatten()


class WorkloadTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for workload time series prediction.
    Creates sliding window sequences for supervised learning.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        seq_length: int,
        pred_length: int = 1,
        target_idx: int = 0
    ):
        """
        Args:
            data: Numpy array of shape (num_timestamps, num_features)
            seq_length: Length of input sequence
            pred_length: Length of prediction horizon
            target_idx: Index of target feature in data
        """
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.target_idx = target_idx
        
        # Calculate number of valid samples
        self.num_samples = len(data) - seq_length - pred_length + 1
        
        if self.num_samples <= 0:
            raise ValueError(f"Not enough data points. Need at least "
                           f"{seq_length + pred_length} points, got {len(data)}")
        
    def __len__(self) -> int:
        return max(0, self.num_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Input sequence of shape (seq_length, num_features)
            y: Target values of shape (pred_length,)
        """
        # Input sequence
        x = self.data[idx:idx + self.seq_length]
        
        # Target: predict the target feature for next pred_length steps
        y = self.data[
            idx + self.seq_length:idx + self.seq_length + self.pred_length,
            self.target_idx
        ]
        
        return x, y


class MultiServiceDataset(Dataset):
    """
    Dataset for training on multiple microservices.
    Pools data from multiple services for better generalization.
    """
    
    def __init__(
        self,
        service_data_dict: Dict[str, np.ndarray],
        seq_length: int,
        pred_length: int = 1,
        target_idx: int = 0
    ):
        """
        Args:
            service_data_dict: Dictionary mapping service names to their data arrays
            seq_length: Length of input sequence
            pred_length: Length of prediction horizon
            target_idx: Index of target feature
        """
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.target_idx = target_idx
        
        # Collect all samples from all services
        self.samples = []
        for msname, data in service_data_dict.items():
            n_samples = len(data) - seq_length - pred_length + 1
            if n_samples > 0:
                for i in range(n_samples):
                    x = data[i:i + seq_length]
                    y = data[i + seq_length:i + seq_length + pred_length, target_idx]
                    self.samples.append((torch.FloatTensor(x), torch.FloatTensor(y)))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class DataModule:
    """
    High-level data module that combines loading, preprocessing, and dataset creation.
    Enhanced version with multi-source support.
    """
    
    def __init__(
        self,
        data_config: 'DataConfig' = None,
        model_config: 'ModelConfig' = None,
        exp_config: 'ExperimentConfig' = None
    ):
        self.data_config = data_config or DATA_CONFIG
        self.model_config = model_config or MODEL_CONFIG
        self.exp_config = exp_config or EXPERIMENT_CONFIG
        
        # Loaders
        self.msmetrics_loader = MSMetricsLoader(self.data_config)
        self.msrtmcr_loader = MSRTMCRLoader(self.data_config)
        self.node_loader = NodeMetricsLoader(self.data_config)
        self.callgraph_loader = CallGraphLoader(self.data_config)
        
        # Preprocessor
        self.preprocessor = WorkloadDataPreprocessor(self.data_config)
        
        # Data storage
        self.msmetrics_data = None
        self.msrtmcr_data = None
        self.merged_data = None
        self.service_timeseries: Dict[str, pd.DataFrame] = {}
        
    def load_data(
        self,
        num_msmetrics_files: int = None,
        num_msrtmcr_files: int = None,
        load_msrtmcr: bool = True,
        num_workers: int = 4,
        verbose: bool = True
    ):
        """Load data from files."""
        # Default to loading all files
        if num_msmetrics_files is None:
            num_msmetrics_files = self.data_config.msmetrics_range[1]
        if num_msrtmcr_files is None:
            num_msrtmcr_files = min(self.data_config.msrtmcr_range[1], 240)
        
        print("=" * 60)
        print("Loading MSMetrics data...")
        self.msmetrics_data = self.msmetrics_loader.load_range(
            0, num_msmetrics_files, num_workers=num_workers, verbose=verbose
        )
        
        print(f"\nMSMetrics Data Shape: {self.msmetrics_data.shape}")
        print(f"Unique microservices: {self.msmetrics_data['msname'].nunique():,}")
        print(f"Timestamp range: {self.msmetrics_data['timestamp'].min()} - "
              f"{self.msmetrics_data['timestamp'].max()}")
        
        if load_msrtmcr and self.exp_config.use_msrtmcr:
            print("\n" + "=" * 60)
            print("Loading MSRTMCR data...")
            try:
                self.msrtmcr_data = self.msrtmcr_loader.load_range(
                    0, num_msrtmcr_files, num_workers=num_workers, verbose=verbose
                )
                print(f"\nMSRTMCR Data Shape: {self.msrtmcr_data.shape}")
            except Exception as e:
                print(f"Warning: Could not load MSRTMCR data: {e}")
                self.msrtmcr_data = None
    
    def merge_data_sources(self) -> pd.DataFrame:
        """Merge MSMetrics and MSRTMCR data by msname and timestamp."""
        if self.msmetrics_data is None:
            raise ValueError("MSMetrics data not loaded")
        
        if self.msrtmcr_data is None:
            print("MSRTMCR data not available, using MSMetrics only")
            self.merged_data = self.msmetrics_data.copy()
            return self.merged_data
        
        print("\nMerging MSMetrics and MSRTMCR data...")
        
        # Aggregate MSRTMCR by msname and timestamp
        msrtmcr_features = [f for f in self.data_config.msrtmcr_features 
                          if f in self.msrtmcr_data.columns]
        
        if msrtmcr_features:
            msrtmcr_agg = self.msrtmcr_data.groupby(['msname', 'timestamp'])[msrtmcr_features].mean().reset_index()
            
            # Merge with MSMetrics
            self.merged_data = pd.merge(
                self.msmetrics_data,
                msrtmcr_agg,
                on=['msname', 'timestamp'],
                how='left'
            )
            
            # Fill NaN values in MSRTMCR features with 0
            for feat in msrtmcr_features:
                if feat in self.merged_data.columns:
                    self.merged_data[feat] = self.merged_data[feat].fillna(0)
            
            print(f"Merged Data Shape: {self.merged_data.shape}")
        else:
            self.merged_data = self.msmetrics_data.copy()
        
        return self.merged_data
    
    def prepare_service_data(
        self,
        msname: str,
        features: List[str] = None,
        normalize: bool = False,
        add_features: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Prepare time series data for a specific microservice.
        
        Note: normalize is now False by default. Normalization should be done 
        in create_dataloaders() AFTER splitting to avoid data leakage.
        Set normalize=True only if you need pre-normalized data and understand
        the implications.
        """
        # Use merged data if available, otherwise use msmetrics
        data_source = self.merged_data if self.merged_data is not None else self.msmetrics_data
        
        if data_source is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if features is None:
            features = self.data_config.msmetrics_features.copy()
            if self.merged_data is not None and self.exp_config.use_msrtmcr:
                # Add MSRTMCR features that exist in the data
                for feat in self.data_config.msrtmcr_features:
                    if feat in data_source.columns:
                        features.append(feat)
        
        # Aggregate instance-level data to service-level
        aggregated = self.preprocessor.aggregate_by_service_time(
            data_source,
            features=features
        )
        
        # Create time series for the service
        ts_data = self.preprocessor.create_service_timeseries(
            aggregated,
            msname,
            features=features,
            add_features=add_features
        )
        
        if ts_data is None:
            return None
        
        # Normalize if explicitly requested (not recommended - use create_dataloaders instead)
        if normalize:
            print("Warning: Pre-normalizing data. This may cause data leakage if used with create_dataloaders().")
            ts_data = self.preprocessor.normalize_features(ts_data)
        
        self.service_timeseries[msname] = ts_data
        return ts_data
    
    def get_top_services(self, n: int = 10, min_points: int = None) -> List[Tuple[str, int]]:
        """
        Get top N microservices by number of data points.
        """
        if self.msmetrics_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if min_points is None:
            min_points = self.data_config.min_data_points
        
        service_counts = self.msmetrics_data.groupby('msname').size()
        
        # Filter by minimum points
        service_counts = service_counts[service_counts >= min_points]
        
        top_services = service_counts.nlargest(n)
        
        return list(zip(top_services.index, top_services.values))
    
    def create_dataloaders(
        self,
        data: np.ndarray,
        target_idx: int = 0,
        shuffle_train: bool = False,
        normalize: bool = True
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        """
        Create train, validation, and test DataLoaders with TIME-BASED split.
        
        Important: For time series, we must preserve temporal order.
        The data is split chronologically (train -> val -> test) and by default
        shuffle_train=False to maintain temporal dependencies within the training set.
        
        IMPORTANT: Normalization is now done AFTER splitting to avoid data leakage.
        The scaler is fit only on training data and applied to val/test.
        
        Args:
            data: Numpy array of shape (num_timestamps, num_features)
            target_idx: Index of target feature in data
            shuffle_train: Whether to shuffle training data. Default False to preserve
                          temporal order for time series. Set True only if you understand
                          the implications for time series forecasting.
            normalize: Whether to normalize data (default True). Normalization is
                      applied after splitting to avoid data leakage.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader). 
            If train dataset creation fails, raises ValueError instead of returning None.
            val_loader and test_loader may be None if insufficient data.
        
        Raises:
            ValueError: If there is insufficient data to create the training dataset.
        """
        n = len(data)
        
        # Time-based split (chronological order preserved)
        train_end = int(n * self.model_config.train_ratio)
        val_end = int(n * (self.model_config.train_ratio + self.model_config.val_ratio))
        
        # Split data chronologically
        train_data = data[:train_end].copy()
        val_data = data[train_end:val_end].copy()
        test_data = data[val_end:].copy()
        
        # Apply normalization AFTER split (fit on train only) to avoid data leakage
        if normalize:
            from sklearn.preprocessing import MinMaxScaler
            self._data_scaler = MinMaxScaler()
            train_data = self._data_scaler.fit_transform(train_data)
            if len(val_data) > 0:
                val_data = self._data_scaler.transform(val_data)
            if len(test_data) > 0:
                test_data = self._data_scaler.transform(test_data)
        
        print(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Create training dataset (required)
        min_required = self.model_config.seq_length + self.model_config.pred_length
        if len(train_data) < min_required:
            raise ValueError(
                f"Insufficient training data: {len(train_data)} samples, "
                f"need at least {min_required} (seq_length={self.model_config.seq_length} + "
                f"pred_length={self.model_config.pred_length}). "
                f"Try loading more data files or reducing sequence length."
            )
        
        train_dataset = WorkloadTimeSeriesDataset(
            train_data,
            seq_length=self.model_config.seq_length,
            pred_length=self.model_config.pred_length,
            target_idx=target_idx
        )
        
        # Create validation dataset (optional)
        val_dataset = None
        if len(val_data) >= min_required:
            try:
                val_dataset = WorkloadTimeSeriesDataset(
                    val_data,
                    seq_length=self.model_config.seq_length,
                    pred_length=self.model_config.pred_length,
                    target_idx=target_idx
                )
            except ValueError as e:
                print(f"Warning: Could not create validation dataset: {e}")
        else:
            print(f"Warning: Insufficient validation data ({len(val_data)} samples)")
        
        # Create test dataset (optional)
        test_dataset = None
        if len(test_data) >= min_required:
            try:
                test_dataset = WorkloadTimeSeriesDataset(
                    test_data,
                    seq_length=self.model_config.seq_length,
                    pred_length=self.model_config.pred_length,
                    target_idx=target_idx
                )
            except ValueError as e:
                print(f"Warning: Could not create test dataset: {e}")
        else:
            print(f"Warning: Insufficient test data ({len(test_data)} samples)")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=shuffle_train,  # Default False for time series
            num_workers=0,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=0
        ) if val_dataset is not None else None
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=0
        ) if test_dataset is not None else None
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset) if val_dataset else 0}")
        print(f"Test samples: {len(test_dataset) if test_dataset else 0}")
        
        return train_loader, val_loader, test_loader
    
    def create_multi_service_dataloaders(
        self,
        service_names: List[str],
        features: List[str] = None,
        normalize: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create dataloaders using data from multiple services.
        Useful for training a universal model.
        """
        train_data_dict = {}
        val_data_dict = {}
        test_data_dict = {}
        
        for msname in service_names:
            ts_data = self.prepare_service_data(msname, features, normalize)
            if ts_data is not None and len(ts_data) >= self.model_config.seq_length + self.model_config.pred_length + 10:
                data = ts_data.values
                n = len(data)
                train_end = int(n * self.model_config.train_ratio)
                val_end = int(n * (self.model_config.train_ratio + self.model_config.val_ratio))
                
                train_data_dict[msname] = data[:train_end]
                val_data_dict[msname] = data[train_end:val_end]
                test_data_dict[msname] = data[val_end:]
        
        if not train_data_dict:
            raise ValueError("No services with enough data points")
        
        print(f"\nPrepared data for {len(train_data_dict)} services")
        
        # Create multi-service datasets
        train_dataset = MultiServiceDataset(
            train_data_dict,
            seq_length=self.model_config.seq_length,
            pred_length=self.model_config.pred_length,
            target_idx=0
        )
        
        val_dataset = MultiServiceDataset(
            val_data_dict,
            seq_length=self.model_config.seq_length,
            pred_length=self.model_config.pred_length,
            target_idx=0
        )
        
        test_dataset = MultiServiceDataset(
            test_data_dict,
            seq_length=self.model_config.seq_length,
            pred_length=self.model_config.pred_length,
            target_idx=0
        )
        
        # Create dataloaders
        # Note: shuffle=False for time series to preserve temporal patterns
        # Each service's data maintains its temporal order within the pooled dataset
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,  # Preserve temporal order for time series
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Multi-service - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def save_cache(self, filename: str = "data_cache.pkl"):
        """Save processed data to cache."""
        cache_path = os.path.join(self.exp_config.cache_path, filename)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        cache_data = {
            'msmetrics_data': self.msmetrics_data,
            'msrtmcr_data': self.msrtmcr_data,
            'merged_data': self.merged_data,
            'service_timeseries': self.service_timeseries,
            'scalers': self.preprocessor.scalers
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Data cache saved to {cache_path}")
    
    def load_cache(self, filename: str = "data_cache.pkl") -> bool:
        """Load processed data from cache."""
        cache_path = os.path.join(self.exp_config.cache_path, filename)
        
        if not os.path.exists(cache_path):
            return False
        
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.msmetrics_data = cache_data['msmetrics_data']
        self.msrtmcr_data = cache_data['msrtmcr_data']
        self.merged_data = cache_data['merged_data']
        self.service_timeseries = cache_data['service_timeseries']
        self.preprocessor.scalers = cache_data['scalers']
        
        print(f"Data cache loaded from {cache_path}")
        return True


def quick_load_sample(num_files: int = 3) -> pd.DataFrame:
    """
    Utility function to quickly load a sample of data for exploration.
    """
    loader = MSMetricsLoader()
    data = loader.load_range(0, num_files)
    return data


if __name__ == "__main__":
    # Test data loading
    print("Testing enhanced data loading...")
    
    dm = DataModule()
    
    # Load data (use fewer files for testing)
    dm.load_data(num_msmetrics_files=4, num_msrtmcr_files=20, load_msrtmcr=True)
    
    # Merge data sources
    dm.merge_data_sources()
    
    # Get top services
    top_services = dm.get_top_services(10)
    print("\nTop 10 microservices by data points:")
    for svc, count in top_services:
        print(f"  {svc}: {count:,} records")
    
    # Prepare data for top service
    if top_services:
        top_svc = top_services[0][0]
        print(f"\nPreparing data for {top_svc}...")
        ts_data = dm.prepare_service_data(top_svc, add_features=True)
        
        if ts_data is not None:
            print(f"Time series shape: {ts_data.shape}")
            print(f"Features: {ts_data.columns.tolist()[:10]}...")  # First 10 features
            
            # Create dataloaders
            print("\nCreating dataloaders...")
            train_loader, val_loader, test_loader = dm.create_dataloaders(
                ts_data.values,
                target_idx=0
            )
            
            if train_loader:
                # Test one batch
                for x, y in train_loader:
                    print(f"Input shape: {x.shape}")
                    print(f"Target shape: {y.shape}")
                    break
    
    print("\nData loading test completed!")
