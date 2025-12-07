"""
Data loader module for Alibaba Microservices Trace v2022.
Handles loading, preprocessing, and creating datasets for workload prediction.
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
    
    def load_range(self, start_idx: int = 0, end_idx: int = None) -> pd.DataFrame:
        """Load multiple MSMetrics files in a range."""
        if end_idx is None:
            end_idx = self.config.msmetrics_range[1]
        
        dfs = []
        for idx in range(start_idx, end_idx):
            try:
                df = self.load_single_file(idx)
                dfs.append(df)
                if idx % 5 == 0:
                    print(f"Loaded MSMetrics file {idx}/{end_idx}")
            except FileNotFoundError:
                print(f"Warning: MSMetrics file {idx} not found, skipping...")
                continue
        
        if not dfs:
            raise ValueError("No MSMetrics files loaded")
        
        self.data = pd.concat(dfs, ignore_index=True)
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
    
    def load_range(self, start_idx: int = 0, end_idx: int = None) -> pd.DataFrame:
        """Load multiple MSRTMCR files in a range."""
        if end_idx is None:
            end_idx = self.config.msrtmcr_range[1]
        
        dfs = []
        for idx in range(start_idx, end_idx):
            try:
                df = self.load_single_file(idx)
                dfs.append(df)
                if idx % 20 == 0:
                    print(f"Loaded MSRTMCR file {idx}/{end_idx}")
            except FileNotFoundError:
                print(f"Warning: MSRTMCR file {idx} not found, skipping...")
                continue
        
        if not dfs:
            raise ValueError("No MSRTMCR files loaded")
        
        self.data = pd.concat(dfs, ignore_index=True)
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
    
    def load_range(self, start_idx: int = 0, end_idx: int = None) -> pd.DataFrame:
        """Load multiple CallGraph files in a range."""
        if end_idx is None:
            end_idx = self.config.callgraph_range[1]
        
        dfs = []
        for idx in range(start_idx, end_idx):
            try:
                df = self.load_single_file(idx)
                dfs.append(df)
                if idx % 20 == 0:
                    print(f"Loaded CallGraph file {idx}/{end_idx}")
            except FileNotFoundError:
                print(f"Warning: CallGraph file {idx} not found, skipping...")
                continue
        
        if not dfs:
            raise ValueError("No CallGraph files loaded")
        
        self.data = pd.concat(dfs, ignore_index=True)
        print(f"Total CallGraph records: {len(self.data):,}")
        return self.data


class WorkloadDataPreprocessor:
    """
    Preprocessor for creating workload prediction datasets.
    Aggregates and processes data for time series prediction.
    """
    
    def __init__(self, config: 'DataConfig' = None):
        self.config = config or DATA_CONFIG
        self.scalers: Dict[str, MinMaxScaler] = {}
        
    def aggregate_by_service_time(
        self, 
        df: pd.DataFrame, 
        time_col: str = 'timestamp',
        service_col: str = 'msname',
        features: List[str] = None
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
        agg_dict = {f: 'mean' for f in valid_features}
        aggregated = df.groupby([service_col, time_col]).agg(agg_dict).reset_index()
        
        return aggregated
    
    def create_service_timeseries(
        self,
        df: pd.DataFrame,
        msname: str,
        time_col: str = 'timestamp',
        service_col: str = 'msname',
        features: List[str] = None
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
        
        # Set timestamp as index
        service_data = service_data.set_index(time_col)
        
        # Select features
        if features is None:
            features = ['cpu_utilization', 'memory_utilization']
        valid_features = [f for f in features if f in service_data.columns]
        service_data = service_data[valid_features]
        
        # Fill missing values
        if self.config.fillna_method == 'ffill':
            service_data = service_data.ffill().bfill()
        elif self.config.fillna_method == 'zero':
            service_data = service_data.fillna(0)
        else:
            service_data = service_data.fillna(service_data.mean())
        
        return service_data
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        features: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize features using MinMaxScaler.
        """
        df_norm = df.copy()
        
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
        
    def __len__(self) -> int:
        return self.num_samples
    
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


class DataModule:
    """
    High-level data module that combines loading, preprocessing, and dataset creation.
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
        
        # Preprocessor
        self.preprocessor = WorkloadDataPreprocessor(self.data_config)
        
        # Data storage
        self.msmetrics_data = None
        self.service_timeseries: Dict[str, pd.DataFrame] = {}
        
    def load_data(
        self,
        num_msmetrics_files: int = 5,
        num_msrtmcr_files: int = 20
    ):
        """Load data from files."""
        print("=" * 50)
        print("Loading MSMetrics data...")
        self.msmetrics_data = self.msmetrics_loader.load_range(
            0, num_msmetrics_files
        )
        
        print("\n" + "=" * 50)
        print(f"MSMetrics Data Shape: {self.msmetrics_data.shape}")
        print(f"Unique microservices: {self.msmetrics_data['msname'].nunique():,}")
        print(f"Timestamp range: {self.msmetrics_data['timestamp'].min()} - "
              f"{self.msmetrics_data['timestamp'].max()}")
        
    def prepare_service_data(
        self,
        msname: str,
        features: List[str] = None,
        normalize: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Prepare time series data for a specific microservice.
        """
        if self.msmetrics_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if features is None:
            features = self.exp_config.msmetrics_features
        
        # Aggregate instance-level data to service-level
        aggregated = self.preprocessor.aggregate_by_service_time(
            self.msmetrics_data,
            features=features
        )
        
        # Create time series for the service
        ts_data = self.preprocessor.create_service_timeseries(
            aggregated,
            msname,
            features=features
        )
        
        if ts_data is None:
            return None
        
        # Normalize if requested
        if normalize:
            ts_data = self.preprocessor.normalize_features(
                ts_data,
                features=features
            )
        
        self.service_timeseries[msname] = ts_data
        return ts_data
    
    def get_top_services(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get top N microservices by number of data points.
        """
        if self.msmetrics_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        service_counts = self.msmetrics_data.groupby('msname').size()
        top_services = service_counts.nlargest(n)
        
        return list(zip(top_services.index, top_services.values))
    
    def create_dataloaders(
        self,
        data: np.ndarray,
        target_idx: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test DataLoaders.
        """
        n = len(data)
        train_end = int(n * self.model_config.train_ratio)
        val_end = int(n * (self.model_config.train_ratio + self.model_config.val_ratio))
        
        # Split data
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        # Create datasets
        train_dataset = WorkloadTimeSeriesDataset(
            train_data,
            seq_length=self.model_config.seq_length,
            pred_length=self.model_config.pred_length,
            target_idx=target_idx
        )
        
        val_dataset = WorkloadTimeSeriesDataset(
            val_data,
            seq_length=self.model_config.seq_length,
            pred_length=self.model_config.pred_length,
            target_idx=target_idx
        )
        
        test_dataset = WorkloadTimeSeriesDataset(
            test_data,
            seq_length=self.model_config.seq_length,
            pred_length=self.model_config.pred_length,
            target_idx=target_idx
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=True,
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
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader


def quick_load_sample(num_files: int = 3) -> pd.DataFrame:
    """
    Utility function to quickly load a sample of data for exploration.
    """
    loader = MSMetricsLoader()
    data = loader.load_range(0, num_files)
    return data


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    
    dm = DataModule()
    dm.load_data(num_msmetrics_files=2)
    
    # Get top services
    top_services = dm.get_top_services(5)
    print("\nTop 5 microservices by data points:")
    for svc, count in top_services:
        print(f"  {svc}: {count:,} records")
    
    # Prepare data for top service
    top_svc = top_services[0][0]
    print(f"\nPreparing data for {top_svc}...")
    ts_data = dm.prepare_service_data(top_svc)
    
    if ts_data is not None:
        print(f"Time series shape: {ts_data.shape}")
        print(f"Features: {ts_data.columns.tolist()}")
        
        # Create dataloaders
        print("\nCreating dataloaders...")
        train_loader, val_loader, test_loader = dm.create_dataloaders(
            ts_data.values,
            target_idx=0
        )
        
        # Test one batch
        for x, y in train_loader:
            print(f"Input shape: {x.shape}")
            print(f"Target shape: {y.shape}")
            break
    
    print("\nData loading test completed!")


