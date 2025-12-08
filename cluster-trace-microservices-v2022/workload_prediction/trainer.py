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
        device: str = None
    ):
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
        
        # Loss function
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
    **model_kwargs
) -> Dict:
    """Run a complete experiment with a deep learning model."""
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
    
    # Create trainer
    trainer = WorkloadTrainer(model)
    
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
    
    # Deep learning models
    dl_models = ['lstm', 'gru', 'attention_lstm', 'tcn', 'nlinear', 'dlinear']
    
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
