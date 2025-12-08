"""
Configuration file for microservices workload prediction project.
Contains all configurable parameters for data processing, model training, and evaluation.

Enhanced version: supports all data sources, baseline models, and advanced deep learning models.
"""

import os
from dataclasses import dataclass, field
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
    
    # Data range (file indices) - USE ALL AVAILABLE DATA
    msmetrics_range: tuple = (0, 24)    # 24 files (30 min each = 12 hours)
    msrtmcr_range: tuple = (0, 240)     # 240 files (3 min each = 12 hours)
    callgraph_range: tuple = (0, 240)   # 240 files (3 min each = 12 hours)
    
    # Time configuration
    timestamp_unit: str = "ms"  # milliseconds
    time_interval: int = 60000  # 60 seconds in ms (data recording interval)
    
    # Preprocessing
    min_data_points: int = 50   # Minimum data points required for a microservice
    normalize: bool = True      # Whether to normalize features
    fillna_method: str = "ffill"  # Method to fill missing values
    
    # Feature sets
    msmetrics_features: List[str] = field(default_factory=lambda: [
        "cpu_utilization", "memory_utilization"
    ])
    
    # MSRTMCR features - MCR (call rate) and RT (response time) metrics
    msrtmcr_features: List[str] = field(default_factory=lambda: [
        "providerrpc_mcr", "consumerrpc_mcr",
        "providerrpc_rt", "consumerrpc_rt",
        "http_mcr", "http_rt",
        "readdb_mcr", "readdb_rt",
        "writedb_mcr", "writedb_rt",
        "readmc_mcr", "readmc_rt",
        "writemc_mcr", "writemc_rt",
        "providermq_mcr", "providermq_rt",
        "consumermq_mcr", "consumermq_rt"
    ])
    
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
    seq_length: int = 12        # Input sequence length (12 time steps)
    pred_length: int = 1        # Prediction horizon (1 time step = 1 minute)
    
    # LSTM model parameters
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    
    # GRU model parameters (same structure as LSTM)
    gru_hidden_size: int = 64
    gru_num_layers: int = 2
    gru_dropout: float = 0.2
    
    # Transformer model parameters
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dropout: float = 0.1
    transformer_dim_feedforward: int = 256
    
    # TCN model parameters
    tcn_num_channels: int = 64
    tcn_kernel_size: int = 3
    tcn_num_levels: int = 4
    tcn_dropout: float = 0.2
    
    # Informer model parameters (for long-sequence prediction)
    informer_d_model: int = 64
    informer_n_heads: int = 4
    informer_e_layers: int = 2
    informer_d_ff: int = 256
    informer_dropout: float = 0.1
    informer_factor: int = 5
    
    # Linear models (DLinear/NLinear) parameters
    linear_individual: bool = False
    
    # XGBoost parameters
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    
    # LightGBM parameters
    lgb_n_estimators: int = 100
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.1
    lgb_num_leaves: int = 31
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 15
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Learning rate scheduler
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    
    # Data split ratios (time-based split)
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
    cache_dir: str = "cache"
    
    # Experiment settings
    experiment_name: str = "workload_prediction_v2"
    
    # Target variable for prediction
    target_variable: str = "cpu_utilization"
    
    # Feature groups to use
    use_msmetrics: bool = True      # CPU, Memory utilization
    use_msrtmcr: bool = True        # Call rate, Response time
    use_time_features: bool = True  # Hour, minute patterns
    use_lag_features: bool = True   # Lag features
    
    # Lag feature configuration
    lag_steps: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12])
    
    # Rolling statistics configuration
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12])
    
    # Models to compare
    baseline_models: List[str] = field(default_factory=lambda: [
        "naive",           # Last value prediction
        "moving_average",  # Simple moving average
        "exp_smoothing",   # Exponential smoothing
        "xgboost",         # XGBoost regressor
        "lightgbm",        # LightGBM regressor
        "random_forest"    # Random Forest regressor
    ])
    
    deep_learning_models: List[str] = field(default_factory=lambda: [
        # RNN-based models
        "lstm",            # Standard LSTM
        "gru",             # GRU (lighter than LSTM)
        "attention_lstm",  # LSTM with attention (FIXED version)
        
        # Transformer-based models (SOTA)
        "transformer",     # Transformer encoder
        "patchtst",        # PatchTST (ICLR 2023) - patches + transformer
        "informer",        # Informer (AAAI 2021) - long sequence forecasting
        "autoformer",      # Autoformer (NeurIPS 2021) - auto-correlation
        
        # CNN-based models
        "tcn",             # Temporal Convolutional Network
        "timesnet",        # TimesNet (ICLR 2023) - 2D convolution
        
        # Linear models (strong baselines)
        "nlinear",         # NLinear (AAAI 2023) - simple but effective
        "dlinear"          # DLinear (AAAI 2023) - decomposition + linear
    ])
    
    # Logging
    log_interval: int = 10  # Log every N batches
    save_model: bool = True
    verbose: bool = True
    
    @property
    def model_path(self) -> str:
        return os.path.join(self.output_dir, self.model_save_dir)
    
    @property
    def figures_path(self) -> str:
        return os.path.join(self.output_dir, self.figures_dir)
    
    @property
    def results_path(self) -> str:
        return os.path.join(self.output_dir, self.results_dir)
    
    @property
    def cache_path(self) -> str:
        return os.path.join(self.output_dir, self.cache_dir)


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
        EXPERIMENT_CONFIG.results_path,
        EXPERIMENT_CONFIG.cache_path
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_all_features() -> List[str]:
    """Get list of all features based on experiment config."""
    features = []
    
    if EXPERIMENT_CONFIG.use_msmetrics:
        features.extend(DATA_CONFIG.msmetrics_features)
    
    if EXPERIMENT_CONFIG.use_msrtmcr:
        features.extend(DATA_CONFIG.msrtmcr_features)
    
    return features


if __name__ == "__main__":
    # Print configuration summary
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print("\nData Configuration:")
    print(f"  Data root: {DATA_CONFIG.data_root}")
    print(f"  MSMetrics files: {DATA_CONFIG.msmetrics_range[0]}-{DATA_CONFIG.msmetrics_range[1]}")
    print(f"  MSRTMCR files: {DATA_CONFIG.msrtmcr_range[0]}-{DATA_CONFIG.msrtmcr_range[1]}")
    
    print("\nModel Configuration:")
    print(f"  Sequence length: {MODEL_CONFIG.seq_length}")
    print(f"  Prediction horizon: {MODEL_CONFIG.pred_length}")
    print(f"  Batch size: {MODEL_CONFIG.batch_size}")
    print(f"  Learning rate: {MODEL_CONFIG.learning_rate}")
    print(f"  Epochs: {MODEL_CONFIG.num_epochs}")
    
    print("\nExperiment Configuration:")
    print(f"  Target: {EXPERIMENT_CONFIG.target_variable}")
    print(f"  Baseline models: {EXPERIMENT_CONFIG.baseline_models}")
    print(f"  Deep learning models: {EXPERIMENT_CONFIG.deep_learning_models}")
    
    print("\nFeatures:")
    print(f"  All features: {get_all_features()}")
