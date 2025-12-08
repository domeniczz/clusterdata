"""
Training and evaluation module for workload prediction models.
Provides comprehensive training loops, metrics, and utilities.

Enhanced version:
- Time-based train/val/test split
- Walk-forward validation support
- Comprehensive metrics and visualization
- Support for both deep learning and baseline models
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_CONFIG, EXPERIMENT_CONFIG, create_output_dirs
from models import create_model, get_model_info
from baseline_models import BasePredictor, evaluate_baseline, get_all_baseline_models


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_mae: List[float] = field(default_factory=list)
    val_mae: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    best_val_loss: float = float('inf')
    best_epoch: int = 0
    training_time: float = 0.0


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Monitors validation loss and stops training when no improvement.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """Check if training should stop."""
        if self.mode == 'min':
            is_improvement = score < (self.best_score or float('inf')) - self.min_delta
        else:
            is_improvement = score > (self.best_score or float('-inf')) + self.min_delta
        
        if is_improvement:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        """Reset the early stopping counter."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class HuberLoss(nn.Module):
    """Huber Loss - robust to outliers."""
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
        
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        return torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        ).mean()


class WorkloadTrainer:
    """
    Trainer class for workload prediction models.
    Handles training, validation, and evaluation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_config: 'ModelConfig' = None,
        exp_config: 'ExperimentConfig' = None,
        device: str = None,
        loss_type: str = 'mse'
    ):
        """
        Args:
            model: PyTorch model to train
            model_config: Model configuration
            exp_config: Experiment configuration
            device: Device to use ('cuda' or 'cpu')
            loss_type: Loss function type ('mse', 'huber', 'mae')
        """
        self.model_config = model_config or MODEL_CONFIG
        self.exp_config = exp_config or EXPERIMENT_CONFIG
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() and 
                self.model_config.device == 'cuda' else 'cpu'
            )
        
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Loss function - Huber loss is more robust for workload prediction
        if loss_type == 'huber':
            self.criterion = HuberLoss(delta=0.5)
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.model_config.scheduler_factor,
            patience=self.model_config.scheduler_patience
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.model_config.early_stopping_patience
        )
        
        # Metrics
        self.metrics = TrainingMetrics()
        
        # Best model state
        self.best_model_state = None
        
        # Create output directories
        create_output_dirs()
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x)
            
            # Compute loss
            loss = self.criterion(output.squeeze(), y.squeeze())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.model_config.gradient_clip
            )
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(output.squeeze() - y.squeeze())).item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_mae = total_mae / max(num_batches, 1)
        
        return avg_loss, avg_mae
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                loss = self.criterion(output.squeeze(), y.squeeze())
                
                total_loss += loss.item()
                total_mae += torch.mean(torch.abs(output.squeeze() - y.squeeze())).item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_mae = total_mae / max(num_batches, 1)
        
        return avg_loss, avg_mae
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None,
        verbose: bool = True
    ) -> TrainingMetrics:
        """Full training loop."""
        if num_epochs is None:
            num_epochs = self.model_config.num_epochs
        
        if verbose:
            print("=" * 60)
            print("Starting Training")
            print(f"  Epochs: {num_epochs}")
            print(f"  Batch size: {self.model_config.batch_size}")
            print(f"  Learning rate: {self.model_config.learning_rate}")
            print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_mae = self.train_epoch(train_loader)
            
            # Validation
            if val_loader is not None:
                val_loss, val_mae = self.validate(val_loader)
            else:
                val_loss, val_mae = train_loss, train_mae
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record metrics
            self.metrics.train_loss.append(train_loss)
            self.metrics.val_loss.append(val_loss)
            self.metrics.train_mae.append(train_mae)
            self.metrics.val_mae.append(val_mae)
            self.metrics.learning_rates.append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Check for best model
            if val_loss < self.metrics.best_val_loss:
                self.metrics.best_val_loss = val_loss
                self.metrics.best_epoch = epoch
                
                # Save best model state
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                
                # Save to file
                if self.exp_config.save_model:
                    self.save_model('best_model.pt')
            
            # Logging
            epoch_time = time.time() - epoch_start
            if verbose and ((epoch + 1) % self.exp_config.log_interval == 0 or epoch == 0):
                print(f"Epoch [{epoch + 1}/{num_epochs}] "
                      f"Train Loss: {train_loss:.6f} "
                      f"Val Loss: {val_loss:.6f} "
                      f"Train MAE: {train_mae:.6f} "
                      f"Val MAE: {val_mae:.6f} "
                      f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if self.early_stopping(val_loss):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        self.metrics.training_time = time.time() - start_time
        
        if verbose:
            print(f"\nTraining completed in {self.metrics.training_time:.2f}s")
            print(f"Best validation loss: {self.metrics.best_val_loss:.6f} "
                  f"at epoch {self.metrics.best_epoch + 1}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.metrics
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluate model on test set."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                output = self.model(x)
                
                all_predictions.append(output.cpu().numpy())
                all_targets.append(y.numpy())
        
        predictions = np.concatenate(all_predictions).flatten()
        targets = np.concatenate(all_targets).flatten()
        
        # Compute metrics
        metrics = compute_metrics(predictions, targets)
        
        return metrics, predictions, targets
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on data."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                output = self.model(x)
                
                all_predictions.append(output.cpu().numpy())
                all_targets.append(y.numpy())
        
        predictions = np.concatenate(all_predictions).flatten()
        targets = np.concatenate(all_targets).flatten()
        
        return predictions, targets
    
    def save_model(self, filename: str = 'model.pt'):
        """Save model checkpoint."""
        path = os.path.join(self.exp_config.model_path, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
        }, path)
    
    def load_model(self, filename: str = 'model.pt'):
        """Load model checkpoint."""
        path = os.path.join(self.exp_config.model_path, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'metrics' in checkpoint:
            self.metrics = checkpoint['metrics']


# ==================== Evaluation Functions ====================

def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    predictions = predictions.flatten()
    targets = targets.flatten()
    
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
    
    # Symmetric MAPE (better for values close to 0)
    smape = np.mean(2 * np.abs(predictions - targets) / 
                    (np.abs(predictions) + np.abs(targets) + 1e-8)) * 100
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'smape': float(smape),
        'r2': float(r2)
    }


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """Pretty print evaluation metrics."""
    print(f"\n{model_name} Evaluation Results:")
    print("-" * 40)
    print(f"  MSE:   {metrics['mse']:.6f}")
    print(f"  RMSE:  {metrics['rmse']:.6f}")
    print(f"  MAE:   {metrics['mae']:.6f}")
    print(f"  MAPE:  {metrics['mape']:.2f}%")
    print(f"  SMAPE: {metrics['smape']:.2f}%")
    print(f"  R²:    {metrics['r2']:.6f}")


# ==================== Experiment Functions ====================

def run_deep_learning_experiment(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_size: int,
    output_size: int = 1,
    seq_length: int = None,
    num_epochs: int = None,
    verbose: bool = True,
    loss_type: str = 'mse',
    **model_kwargs
) -> Dict:
    """
    Run a complete experiment with a deep learning model.
    
    Args:
        model_type: Type of model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        input_size: Number of input features
        output_size: Number of output values (prediction horizon)
        seq_length: Input sequence length
        num_epochs: Number of training epochs
        verbose: Whether to print progress
        loss_type: Loss function ('mse', 'huber', 'mae'). 
                   'huber' is recommended for workload prediction.
    """
    if verbose:
        print("\n" + "=" * 60)
        print(f"Running Experiment: {model_type.upper()}")
        print("=" * 60)
    
    # Create model
    model = create_model(
        model_type,
        input_size=input_size,
        output_size=output_size,
        seq_length=seq_length,
        **model_kwargs
    )
    
    # Get model info
    info = get_model_info(model)
    if verbose:
        print(f"Model parameters: {info['trainable_params']:,}")
    
    # Create trainer with configurable loss
    trainer = WorkloadTrainer(model, loss_type=loss_type)
    
    # Train
    metrics = trainer.train(train_loader, val_loader, num_epochs, verbose=verbose)
    
    # Evaluate
    if test_loader is not None:
        eval_metrics, predictions, targets = trainer.evaluate(test_loader)
        if verbose:
            print_metrics(eval_metrics, model_type.upper())
    else:
        eval_metrics = {}
        predictions, targets = None, None
    
    # Prepare results
    results = {
        'model_type': model_type,
        'num_params': info['trainable_params'],
        'best_val_loss': metrics.best_val_loss,
        'best_epoch': metrics.best_epoch,
        'training_time': metrics.training_time,
        'eval_metrics': eval_metrics,
        'predictions': predictions,
        'targets': targets
    }
    
    return results, trainer


def run_baseline_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    include_ml: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """Run experiments with baseline models."""
    if verbose:
        print("\n" + "=" * 60)
        print("BASELINE MODEL EXPERIMENTS")
        print("=" * 60)
    
    models = get_all_baseline_models(include_ml=include_ml)
    results = []
    
    for model in models:
        if verbose:
            print(f"\nEvaluating {model.name}...", end=" ")
        
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            training_time = time.time() - start_time
            
            targets = y_test.flatten()
            predictions = predictions.flatten()
            
            metrics = compute_metrics(predictions, targets)
            metrics['model'] = model.name
            metrics['training_time'] = training_time
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


def compare_all_models(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_size: int,
    output_size: int = 1,
    seq_length: int = None,
    num_epochs: int = 50,
    include_baselines: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """Compare deep learning models and baselines."""
    all_results = []
    
    # Extract numpy arrays for baselines
    if include_baselines and test_loader is not None:
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []
        
        for x, y in train_loader:
            X_train_list.append(x.numpy())
            y_train_list.append(y.numpy())
        
        for x, y in test_loader:
            X_test_list.append(x.numpy())
            y_test_list.append(y.numpy())
        
        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test = np.concatenate(X_test_list)
        y_test = np.concatenate(y_test_list)
        
        # Run baseline experiments
        baseline_results = run_baseline_experiment(
            X_train, y_train, X_test, y_test, 
            include_ml=True, verbose=verbose
        )
        
        for _, row in baseline_results.iterrows():
            all_results.append({
                'model_type': row['model'],
                'model_class': 'Baseline',
                'rmse': row['rmse'],
                'mae': row['mae'],
                'mape': row['mape'],
                'smape': row['smape'],
                'r2': row['r2'],
                'training_time': row['training_time'],
                'num_params': 0
            })
    
    # Deep learning models - including new SOTA models
    dl_models = [
        # RNN-based
        'lstm', 'gru', 'attention_lstm',
        # Transformer-based
        'transformer', 'patchtst', 'informer', 'autoformer',
        # CNN-based
        'tcn', 'timesnet',
        # Linear (strong baselines)
        'nlinear', 'dlinear'
    ]
    
    for model_type in dl_models:
        try:
            results, _ = run_deep_learning_experiment(
                model_type,
                train_loader, val_loader, test_loader,
                input_size, output_size, seq_length,
                num_epochs, verbose
            )
            
            all_results.append({
                'model_type': model_type.upper(),
                'model_class': 'Deep Learning',
                'rmse': results['eval_metrics'].get('rmse', float('inf')),
                'mae': results['eval_metrics'].get('mae', float('inf')),
                'mape': results['eval_metrics'].get('mape', float('inf')),
                'smape': results['eval_metrics'].get('smape', float('inf')),
                'r2': results['eval_metrics'].get('r2', float('-inf')),
                'training_time': results['training_time'],
                'num_params': results['num_params']
            })
        except Exception as e:
            if verbose:
                print(f"Error with {model_type}: {e}")
            continue
    
    df = pd.DataFrame(all_results)
    if len(df) > 0:
        df = df.sort_values('rmse')
    
    return df


# ==================== Visualization Functions ====================

def plot_training_history(metrics: TrainingMetrics, save_path: str = None):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
    
    epochs = range(1, len(metrics.train_loss) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, metrics.train_loss, label='Train', linewidth=2)
    axes[0, 0].plot(epochs, metrics.val_loss, label='Validation', linewidth=2)
    axes[0, 0].axvline(
        x=metrics.best_epoch + 1,
        color='red',
        linestyle='--',
        label=f'Best epoch ({metrics.best_epoch + 1})'
    )
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(epochs, metrics.train_mae, label='Train', linewidth=2)
    axes[0, 1].plot(epochs, metrics.val_mae, label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Training and Validation MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(epochs, metrics.learning_rates, linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss comparison (log scale)
    axes[1, 1].semilogy(epochs, metrics.train_loss, label='Train', linewidth=2)
    axes[1, 1].semilogy(epochs, metrics.val_loss, label='Validation', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss (log scale)')
    axes[1, 1].set_title('Loss (Log Scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str = "Model",
    num_samples: int = 200,
    save_path: str = None
):
    """Plot predictions vs actual values."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=120)
    
    # Time series comparison
    n = min(num_samples, len(predictions))
    x = range(n)
    
    axes[0].plot(x, targets[:n], label='Actual', linewidth=1.5, alpha=0.8)
    axes[0].plot(x, predictions[:n], label='Predicted', linewidth=1.5, alpha=0.8)
    axes[0].fill_between(x, targets[:n], predictions[:n], alpha=0.2)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'{model_name}: Prediction vs Actual')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(targets, predictions, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                 label='Perfect prediction')
    
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title(f'{model_name}: Prediction Scatter Plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'rmse',
    save_path: str = None
):
    """Plot model comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    
    # Sort by metric
    df = comparison_df.sort_values(metric)
    
    # Bar chart for main metric
    colors = ['#2ecc71' if c == 'Baseline' else '#3498db' 
              for c in df['model_class']]
    
    axes[0].barh(df['model_type'], df[metric], color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel(metric.upper())
    axes[0].set_ylabel('Model')
    axes[0].set_title(f'Model Comparison by {metric.upper()}')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Baseline'),
        Patch(facecolor='#3498db', label='Deep Learning')
    ]
    axes[0].legend(handles=legend_elements, loc='lower right')
    
    # Multiple metrics comparison
    metrics_to_plot = ['rmse', 'mae', 'smape']
    metrics_data = []
    
    for m in metrics_to_plot:
        if m in df.columns:
            # Normalize to [0, 1] for comparison
            values = df[m].values
            if values.max() > values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
            else:
                normalized = np.zeros_like(values)
            metrics_data.append(normalized)
    
    x = np.arange(len(df))
    width = 0.25
    
    for i, (m, data) in enumerate(zip(metrics_to_plot, metrics_data)):
        axes[1].bar(x + i * width, data, width, label=m.upper())
    
    axes[1].set_ylabel('Normalized Score (lower is better)')
    axes[1].set_xlabel('Model')
    axes[1].set_title('Multi-Metric Comparison (Normalized)')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(df['model_type'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comprehensive_comparison(
    comparison_df: pd.DataFrame,
    save_dir: str = None
):
    """
    Generate comprehensive comparison plots for all models.
    Creates multiple visualization types for publication-quality figures.
    """
    if save_dir is None:
        save_dir = EXPERIMENT_CONFIG.figures_path
    os.makedirs(save_dir, exist_ok=True)
    
    df = comparison_df.copy()
    
    # 1. Radar Chart for multi-metric comparison
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True), dpi=120)
    
    metrics = ['rmse', 'mae', 'smape']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if available_metrics and len(df) > 0:
        # Normalize metrics (inverse for "lower is better")
        normalized_df = df.copy()
        for m in available_metrics:
            max_val = df[m].max()
            min_val = df[m].min()
            if max_val > min_val:
                normalized_df[m] = 1 - (df[m] - min_val) / (max_val - min_val)
            else:
                normalized_df[m] = 0.5
        
        # Add R² (higher is better, already correct direction)
        if 'r2' in df.columns:
            available_metrics.append('r2')
            r2_vals = df['r2'].values
            normalized_df['r2'] = (r2_vals - r2_vals.min()) / (r2_vals.max() - r2_vals.min() + 1e-8)
        
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot top 5 models
        top_models = normalized_df.nsmallest(5, 'rmse' if 'rmse' in df.columns else available_metrics[0])
        colors = plt.cm.Set2(np.linspace(0, 1, len(top_models)))
        
        for idx, (_, row) in enumerate(top_models.iterrows()):
            values = [row[m] for m in available_metrics]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model_type'], color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in available_metrics])
        ax.set_title('Model Performance Radar Chart (Higher is Better)', fontsize=14, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'radar_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Box plot for model categories
    if 'model_class' in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
        
        for ax, metric in zip(axes, ['rmse', 'mae', 'r2']):
            if metric in df.columns:
                df.boxplot(column=metric, by='model_class', ax=ax)
                ax.set_title(f'{metric.upper()} by Model Category')
                ax.set_xlabel('')
                ax.set_ylabel(metric.upper())
        
        plt.suptitle('Performance Distribution by Model Category', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'category_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Performance vs Training Time scatter
    if 'training_time' in df.columns and 'rmse' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        
        # Color by model class
        colors = {'Baseline': '#2ecc71', 'Deep Learning': '#3498db', 'deep_learning': '#3498db', 'baseline': '#2ecc71'}
        
        for _, row in df.iterrows():
            color = colors.get(row.get('model_class', 'unknown'), '#95a5a6')
            ax.scatter(row['training_time'], row['rmse'], c=color, s=100, alpha=0.7)
            ax.annotate(row['model_type'], (row['training_time'], row['rmse']), 
                       fontsize=8, ha='center', va='bottom')
        
        ax.set_xlabel('Training Time (seconds)', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('Model Performance vs Training Time', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_vs_time.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. Ranking table visualization
    fig, ax = plt.subplots(figsize=(12, 8), dpi=120)
    ax.axis('off')
    
    # Create ranking table
    ranking_df = df.sort_values('rmse' if 'rmse' in df.columns else df.columns[0])
    ranking_df['Rank'] = range(1, len(ranking_df) + 1)
    
    cols_to_show = ['Rank', 'model_type']
    for m in ['rmse', 'mae', 'mape', 'r2']:
        if m in ranking_df.columns:
            cols_to_show.append(m)
    
    display_df = ranking_df[cols_to_show].head(15)
    
    # Format numeric columns
    for col in display_df.columns:
        if col not in ['Rank', 'model_type']:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}' if isinstance(x, float) else x)
    
    table = ax.table(
        cellText=display_df.values,
        colLabels=[c.upper() if c != 'model_type' else 'Model' for c in display_df.columns],
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * len(display_df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Highlight best model
    table[(1, 0)].set_facecolor('#d4edda')
    table[(1, 1)].set_facecolor('#d4edda')
    
    ax.set_title('Model Ranking Table', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ranking_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive comparison plots saved to: {save_dir}")


def generate_latex_table(comparison_df: pd.DataFrame) -> str:
    """
    Generate LaTeX table code for publication.
    """
    df = comparison_df.sort_values('rmse' if 'rmse' in comparison_df.columns else comparison_df.columns[0])
    
    metrics = ['rmse', 'mae', 'mape', 'r2']
    available = [m for m in metrics if m in df.columns]
    
    # Build LaTeX table
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Model Performance Comparison}\n"
    latex += "\\label{tab:model_comparison}\n"
    latex += "\\begin{tabular}{l" + "c" * len(available) + "}\n"
    latex += "\\toprule\n"
    latex += "Model & " + " & ".join([m.upper() for m in available]) + " \\\\\n"
    latex += "\\midrule\n"
    
    for _, row in df.iterrows():
        values = [f"{row[m]:.4f}" if m in row else "-" for m in available]
        latex += f"{row['model_type']} & " + " & ".join(values) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


def save_results(
    results: Dict,
    filename: str,
    output_dir: str = None
):
    """Save experiment results to JSON."""
    if output_dir is None:
        output_dir = EXPERIMENT_CONFIG.results_path
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy types to Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    # Test trainer with sample data
    from data_loader import DataModule
    
    print("Testing trainer...")
    
    # Load data
    dm = DataModule()
    dm.load_data(num_msmetrics_files=4, load_msrtmcr=False)
    
    # Get top service
    top_services = dm.get_top_services(1)
    if top_services:
        msname = top_services[0][0]
        
        # Prepare data
        ts_data = dm.prepare_service_data(msname, add_features=False)
        
        if ts_data is not None and len(ts_data) > 50:
            # Create dataloaders
            train_loader, val_loader, test_loader = dm.create_dataloaders(
                ts_data.values,
                target_idx=0
            )
            
            if train_loader:
                # Run quick experiment
                results, trainer = run_deep_learning_experiment(
                    'lstm',
                    train_loader, val_loader, test_loader,
                    input_size=ts_data.shape[1],
                    num_epochs=10
                )
                
                print("\nExperiment completed!")
                print(f"Best val loss: {results['best_val_loss']:.6f}")
        else:
            print("Insufficient data for training")
    else:
        print("No services found")
