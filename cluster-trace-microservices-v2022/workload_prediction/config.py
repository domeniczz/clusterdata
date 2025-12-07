"""
Configuration file for microservices workload prediction project.
Contains all configurable parameters for data processing, model training, and evaluation.
"""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    # Data paths
    data_root: str = "/root/autodl-tmp/clusterdata/data"
    msmetrics_dir: str = "MSMetrics"
    msrtmcr_dir: str = "MSRTMCR"
    nodemetrics_dir: str = "NodeMetrics"
    callgraph_dir: str = "CallGraph"
    
    # Data file patterns
    msmetrics_pattern: str = "MSMetricsUpdate_{}.csv"
    msrtmcr_pattern: str = "MCRRTUpdate_{}.csv"
    nodemetrics_pattern: str = "NodeMetricsUpdate_{}.csv"
    callgraph_pattern: str = "CallGraph_{}.csv"
    
    # Data range (file indices)
    msmetrics_range: tuple = (0, 24)    # 24 files (30 min each = 12 hours)
    msrtmcr_range: tuple = (0, 240)     # 240 files (3 min each = 12 hours)
    callgraph_range: tuple = (0, 240)   # 240 files (3 min each = 12 hours)
    
    # Time configuration
    timestamp_unit: str = "ms"  # milliseconds
    time_interval: int = 60000  # 60 seconds in ms (data recording interval)
    
    # Preprocessing
    min_data_points: int = 10   # Minimum data points required for a microservice
    normalize: bool = True      # Whether to normalize features
    fillna_method: str = "ffill"  # Method to fill missing values
    
    @property
    def msmetrics_path(self) -> str:
        return os.path.join(self.data_root, self.msmetrics_dir)
    
    @property
    def msrtmcr_path(self) -> str:
        return os.path.join(self.data_root, self.msrtmcr_dir)
    
    @property
    def nodemetrics_path(self) -> str:
        return os.path.join(self.data_root, self.nodemetrics_dir)
    
    @property
    def callgraph_path(self) -> str:
        return os.path.join(self.data_root, self.callgraph_dir)


@dataclass
class ModelConfig:
    """Configuration for prediction models."""
    # Sequence parameters
    seq_length: int = 12        # Input sequence length (12 time steps = 12 minutes)
    pred_length: int = 1        # Prediction horizon (1 time step = 1 minute)
    
    # LSTM model parameters
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    
    # Transformer model parameters
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 50
    early_stopping_patience: int = 10
    weight_decay: float = 1e-5
    
    # Data split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Device configuration
    device: str = "cuda"  # "cuda" or "cpu"


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    # Output directories
    output_dir: str = "/root/repository/clusterdata/cluster-trace-microservices-v2022/workload_prediction/outputs"
    model_save_dir: str = "models"
    figures_dir: str = "figures"
    results_dir: str = "results"
    
    # Experiment settings
    experiment_name: str = "workload_prediction_v1"
    
    # Features to use for prediction
    # MSMetrics features
    msmetrics_features: List[str] = None
    
    # MSRTMCR features (MCR and RT metrics)
    msrtmcr_features: List[str] = None
    
    # Target variable for prediction
    target_variable: str = "cpu_utilization"
    
    # Logging
    log_interval: int = 10  # Log every N batches
    save_model: bool = True
    
    def __post_init__(self):
        if self.msmetrics_features is None:
            self.msmetrics_features = ["cpu_utilization", "memory_utilization"]
        
        if self.msrtmcr_features is None:
            self.msrtmcr_features = [
                "providerrpc_mcr", "consumerrpc_mcr",
                "providerrpc_rt", "consumerrpc_rt",
                "http_mcr", "http_rt"
            ]
    
    @property
    def model_path(self) -> str:
        return os.path.join(self.output_dir, self.model_save_dir)
    
    @property
    def figures_path(self) -> str:
        return os.path.join(self.output_dir, self.figures_dir)
    
    @property
    def results_path(self) -> str:
        return os.path.join(self.output_dir, self.results_dir)


# Default configurations
DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
EXPERIMENT_CONFIG = ExperimentConfig()


def create_output_dirs():
    """Create output directories if they don't exist."""
    dirs = [
        EXPERIMENT_CONFIG.output_dir,
        EXPERIMENT_CONFIG.model_path,
        EXPERIMENT_CONFIG.figures_path,
        EXPERIMENT_CONFIG.results_path
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


