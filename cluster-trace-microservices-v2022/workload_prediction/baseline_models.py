"""
Baseline models for workload prediction comparison.
Includes statistical models and machine learning baselines.

These baselines are essential for evaluating whether deep learning models
provide meaningful improvements over simpler approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_CONFIG, EXPERIMENT_CONFIG


class BasePredictor(ABC):
    """Abstract base class for all predictors."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BasePredictor':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray) -> np.ndarray:
        """Fit and predict in one step."""
        self.fit(X_train, y_train)
        return self.predict(X_test)


# ==================== Statistical Baselines ====================

class NaivePredictor(BasePredictor):
    """
    Naive predictor: predict the last observed value.
    This is the simplest baseline - if a model can't beat this, it's not useful.
    """
    
    def __init__(self):
        super().__init__("Naive (Last Value)")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaivePredictor':
        """No fitting needed for naive predictor."""
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the last value in each sequence."""
        # X shape: (n_samples, seq_length, n_features)
        # Return the last value of target feature (index 0)
        if len(X.shape) == 3:
            return X[:, -1, 0]  # Last timestep, first feature (target)
        else:
            return X[:, -1]


class MovingAveragePredictor(BasePredictor):
    """
    Moving Average predictor: predict the average of last n values.
    """
    
    def __init__(self, window: int = 5):
        super().__init__(f"Moving Average (window={window})")
        self.window = window
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MovingAveragePredictor':
        """No fitting needed."""
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the moving average of the last window values."""
        if len(X.shape) == 3:
            # Use last 'window' values of target feature
            window = min(self.window, X.shape[1])
            return np.mean(X[:, -window:, 0], axis=1)
        else:
            window = min(self.window, X.shape[1])
            return np.mean(X[:, -window:], axis=1)


class ExponentialSmoothingPredictor(BasePredictor):
    """
    Simple Exponential Smoothing predictor.
    Gives more weight to recent observations.
    """
    
    def __init__(self, alpha: float = 0.3):
        super().__init__(f"Exp Smoothing (alpha={alpha})")
        self.alpha = alpha
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ExponentialSmoothingPredictor':
        """No fitting needed."""
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing."""
        predictions = []
        
        for i in range(len(X)):
            if len(X.shape) == 3:
                series = X[i, :, 0]  # Target feature
            else:
                series = X[i, :]
            
            # Simple exponential smoothing
            smoothed = series[0]
            for t in range(1, len(series)):
                smoothed = self.alpha * series[t] + (1 - self.alpha) * smoothed
            
            predictions.append(smoothed)
        
        return np.array(predictions)


class SeasonalNaivePredictor(BasePredictor):
    """
    Seasonal Naive predictor: predict the value from the same time in the previous period.
    Useful when data has periodic patterns.
    """
    
    def __init__(self, season_length: int = 12):
        super().__init__(f"Seasonal Naive (period={season_length})")
        self.season_length = season_length
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SeasonalNaivePredictor':
        """No fitting needed."""
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using value from one season ago."""
        if len(X.shape) == 3:
            seq_len = X.shape[1]
            if seq_len >= self.season_length:
                return X[:, -self.season_length, 0]
            else:
                return X[:, 0, 0]  # Fall back to first value
        else:
            seq_len = X.shape[1]
            if seq_len >= self.season_length:
                return X[:, -self.season_length]
            else:
                return X[:, 0]


# ==================== Machine Learning Baselines ====================

class LinearRegressionPredictor(BasePredictor):
    """
    Linear Regression baseline using sklearn.
    Flattens the input sequence for prediction.
    """
    
    def __init__(self):
        super().__init__("Linear Regression")
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionPredictor':
        """Fit linear regression model."""
        from sklearn.linear_model import Ridge
        
        # Flatten X: (n_samples, seq_length, n_features) -> (n_samples, seq_length * n_features)
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.flatten()
        
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_flat, y_flat)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)


class RandomForestPredictor(BasePredictor):
    """
    Random Forest Regressor baseline.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestPredictor':
        """Fit random forest model."""
        from sklearn.ensemble import RandomForestRegressor
        
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.flatten()
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_flat, y_flat)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)


class XGBoostPredictor(BasePredictor):
    """
    XGBoost Regressor baseline.
    One of the most effective tree-based methods for tabular data.
    """
    
    def __init__(self, config: 'ModelConfig' = None):
        super().__init__("XGBoost")
        self.config = config or MODEL_CONFIG
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostPredictor':
        """Fit XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.flatten()
        
        self.model = xgb.XGBRegressor(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            random_state=self.config.random_seed,
            n_jobs=-1,
            verbosity=0
        )
        self.model.fit(X_flat, y_flat)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)


class LightGBMPredictor(BasePredictor):
    """
    LightGBM Regressor baseline.
    Fast and memory-efficient gradient boosting.
    """
    
    def __init__(self, config: 'ModelConfig' = None):
        super().__init__("LightGBM")
        self.config = config or MODEL_CONFIG
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LightGBMPredictor':
        """Fit LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.flatten()
        
        self.model = lgb.LGBMRegressor(
            n_estimators=self.config.lgb_n_estimators,
            max_depth=self.config.lgb_max_depth,
            learning_rate=self.config.lgb_learning_rate,
            num_leaves=self.config.lgb_num_leaves,
            random_state=self.config.random_seed,
            n_jobs=-1,
            verbose=-1
        )
        self.model.fit(X_flat, y_flat)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)


class SVRPredictor(BasePredictor):
    """
    Support Vector Regression baseline.
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0):
        super().__init__(f"SVR ({kernel})")
        self.kernel = kernel
        self.C = C
        self.model = None
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVRPredictor':
        """Fit SVR model."""
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.flatten()
        
        # SVR works better with scaled data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_flat)
        
        self.model = SVR(kernel=self.kernel, C=self.C)
        self.model.fit(X_scaled, y_flat)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled)


# ==================== Baseline Factory ====================

def create_baseline_model(model_type: str, **kwargs) -> BasePredictor:
    """
    Factory function to create baseline models.
    
    Args:
        model_type: One of 'naive', 'moving_average', 'exp_smoothing', 
                   'seasonal_naive', 'linear', 'random_forest', 'xgboost', 
                   'lightgbm', 'svr'
        **kwargs: Additional model-specific arguments
    
    Returns:
        BasePredictor instance
    """
    models = {
        'naive': NaivePredictor,
        'moving_average': MovingAveragePredictor,
        'exp_smoothing': ExponentialSmoothingPredictor,
        'seasonal_naive': SeasonalNaivePredictor,
        'linear': LinearRegressionPredictor,
        'random_forest': RandomForestPredictor,
        'xgboost': XGBoostPredictor,
        'lightgbm': LightGBMPredictor,
        'svr': SVRPredictor
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)


def get_all_baseline_models(include_ml: bool = True) -> List[BasePredictor]:
    """
    Get a list of all baseline models for comparison.
    
    Args:
        include_ml: Whether to include ML baselines (RF, XGB, LGBM)
    
    Returns:
        List of BasePredictor instances
    """
    models = [
        NaivePredictor(),
        MovingAveragePredictor(window=3),
        MovingAveragePredictor(window=6),
        MovingAveragePredictor(window=12),
        ExponentialSmoothingPredictor(alpha=0.2),
        ExponentialSmoothingPredictor(alpha=0.5),
        SeasonalNaivePredictor(season_length=12),
        LinearRegressionPredictor()
    ]
    
    if include_ml:
        models.extend([
            RandomForestPredictor(n_estimators=100, max_depth=10),
            XGBoostPredictor(),
            LightGBMPredictor()
        ])
    
    return models


# ==================== Evaluation Utilities ====================

def evaluate_baseline(
    model: BasePredictor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate a baseline model and return metrics.
    
    Returns:
        Dictionary with MSE, RMSE, MAE, MAPE, R2
    """
    # Fit and predict
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Flatten targets
    targets = y_test.flatten()
    predictions = predictions.flatten()
    
    # Compute metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    # MAPE (avoid division by zero)
    mask = np.abs(targets) > 1e-8
    if mask.sum() > 0:
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    else:
        mape = float('inf')
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'model': model.name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }


def compare_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    include_ml: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare all baseline models on the same data.
    
    Returns:
        DataFrame with comparison results
    """
    models = get_all_baseline_models(include_ml=include_ml)
    results = []
    
    for model in models:
        if verbose:
            print(f"Evaluating {model.name}...", end=" ")
        
        try:
            metrics = evaluate_baseline(model, X_train, y_train, X_test, y_test)
            results.append(metrics)
            if verbose:
                print(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            continue
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        df = df.sort_values('rmse')
    
    return df


if __name__ == "__main__":
    # Test baseline models with synthetic data
    print("Testing baseline models...")
    
    np.random.seed(42)
    
    # Create synthetic time series data
    n_samples = 500
    seq_length = 12
    n_features = 2
    
    # Generate data with trend and seasonality
    t = np.arange(n_samples + seq_length)
    trend = 0.001 * t
    seasonality = 0.1 * np.sin(2 * np.pi * t / 12)
    noise = 0.05 * np.random.randn(len(t))
    series = 0.5 + trend + seasonality + noise
    series = np.clip(series, 0, 1)
    
    # Create sequences
    X = []
    y = []
    for i in range(n_samples):
        X.append(np.column_stack([
            series[i:i+seq_length],
            np.random.rand(seq_length)  # Second feature
        ]))
        y.append(series[i+seq_length])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nData shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"            X_test={X_test.shape}, y_test={y_test.shape}")
    
    # Compare baselines
    print("\n" + "=" * 60)
    print("BASELINE MODEL COMPARISON")
    print("=" * 60)
    
    # Try without ML baselines first (faster)
    results = compare_baselines(X_train, y_train, X_test, y_test, include_ml=True)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results.to_string(index=False))
    
    print("\nBaseline models test completed!")

