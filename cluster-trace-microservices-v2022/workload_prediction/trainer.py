"""
Training and evaluation module for workload prediction models.
Provides comprehensive training loops, metrics, and utilities.
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
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_CONFIG, EXPERIMENT_CONFIG, create_output_dirs
from models import create_model


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
        """
        Check if training should stop.
        Returns True if should stop, False otherwise.
        """
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
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.model_config.early_stopping_patience
        )
        
        # Metrics
        self.metrics = TrainingMetrics()
        
        # Create output directories
        create_output_dirs()
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        Returns: (average loss, average MAE)
        """
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(output.squeeze() - y.squeeze())).item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        Returns: (average loss, average MAE)
        """
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
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None
    ) -> TrainingMetrics:
        """
        Full training loop.
        """
        if num_epochs is None:
            num_epochs = self.model_config.num_epochs
        
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
            val_loss, val_mae = self.validate(val_loader)
            
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
                
                # Save best model
                if self.exp_config.save_model:
                    self.save_model('best_model.pt')
            
            # Logging
            epoch_time = time.time() - epoch_start
            if (epoch + 1) % self.exp_config.log_interval == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] "
                      f"Train Loss: {train_loss:.6f} "
                      f"Val Loss: {val_loss:.6f} "
                      f"Train MAE: {train_mae:.6f} "
                      f"Val MAE: {val_mae:.6f} "
                      f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Best validation loss: {self.metrics.best_val_loss:.6f} "
              f"at epoch {self.metrics.best_epoch + 1}")
        
        return self.metrics
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.
        Returns dictionary of evaluation metrics.
        """
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
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
        
        print("\nTest Evaluation Results:")
        print("-" * 40)
        for name, value in metrics.items():
            print(f"  {name.upper()}: {value:.6f}")
        
        return metrics, predictions, targets
    
    def predict(
        self,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on data.
        Returns: (predictions, targets)
        """
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
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, filename: str = 'model.pt'):
        """Load model checkpoint."""
        path = os.path.join(self.exp_config.model_path, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'metrics' in checkpoint:
            self.metrics = checkpoint['metrics']
        print(f"Model loaded from {path}")
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
        
        epochs = range(1, len(self.metrics.train_loss) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.metrics.train_loss, label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.metrics.val_loss, label='Validation', linewidth=2)
        axes[0, 0].axvline(
            x=self.metrics.best_epoch + 1,
            color='red',
            linestyle='--',
            label=f'Best epoch ({self.metrics.best_epoch + 1})'
        )
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        axes[0, 1].plot(epochs, self.metrics.train_mae, label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.metrics.val_mae, label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Training and Validation MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.metrics.learning_rates, linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss comparison (log scale)
        axes[1, 1].semilogy(epochs, self.metrics.train_loss, label='Train', linewidth=2)
        axes[1, 1].semilogy(epochs, self.metrics.val_loss, label='Validation', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss (log scale)')
        axes[1, 1].set_title('Loss (Log Scale)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(
                self.exp_config.figures_path,
                'training_history.png'
            )
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to {save_path}")
    
    def plot_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
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
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Value')
        axes[0].set_title('Prediction vs Actual')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1].scatter(targets, predictions, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                     label='Perfect prediction')
        
        axes[1].set_xlabel('Actual')
        axes[1].set_ylabel('Predicted')
        axes[1].set_title('Prediction Scatter Plot')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(
                self.exp_config.figures_path,
                'predictions.png'
            )
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Predictions plot saved to {save_path}")


def run_experiment(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_size: int,
    output_size: int = 1,
    num_epochs: int = None,
    **model_kwargs
) -> Dict:
    """
    Run a complete experiment with a specific model.
    
    Args:
        model_type: Type of model to use
        train_loader, val_loader, test_loader: Data loaders
        input_size: Number of input features
        output_size: Number of output values
        num_epochs: Number of training epochs
        **model_kwargs: Additional model arguments
    
    Returns:
        Dictionary with experiment results
    """
    print("\n" + "=" * 60)
    print(f"Running Experiment: {model_type.upper()}")
    print("=" * 60)
    
    # Create model
    model = create_model(
        model_type,
        input_size=input_size,
        output_size=output_size,
        **model_kwargs
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = WorkloadTrainer(model)
    
    # Train
    metrics = trainer.train(train_loader, val_loader, num_epochs)
    
    # Evaluate
    eval_metrics, predictions, targets = trainer.evaluate(test_loader)
    
    # Plot results
    trainer.plot_training_history()
    trainer.plot_predictions(predictions, targets)
    
    # Save results
    results = {
        'model_type': model_type,
        'num_params': num_params,
        'best_val_loss': metrics.best_val_loss,
        'best_epoch': metrics.best_epoch,
        'eval_metrics': eval_metrics
    }
    
    # Save to JSON
    results_path = os.path.join(
        EXPERIMENT_CONFIG.results_path,
        f'{model_type}_results.json'
    )
    
    # Convert numpy values to Python types for JSON serialization
    json_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            json_results[k] = {kk: float(vv) for kk, vv in v.items()}
        elif isinstance(v, (np.floating, np.integer)):
            json_results[k] = float(v)
        else:
            json_results[k] = v
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return results


def compare_models(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_size: int,
    output_size: int = 1,
    num_epochs: int = 30
) -> pd.DataFrame:
    """
    Compare multiple models on the same dataset.
    
    Returns:
        DataFrame with comparison results
    """
    model_types = ['lstm', 'gru', 'attention_lstm', 'tcn']
    results = []
    
    for model_type in model_types:
        try:
            result = run_experiment(
                model_type,
                train_loader,
                val_loader,
                test_loader,
                input_size,
                output_size,
                num_epochs
            )
            results.append(result)
        except Exception as e:
            print(f"Error with {model_type}: {e}")
            continue
    
    # Create comparison DataFrame
    comparison = pd.DataFrame(results)
    
    # Extract evaluation metrics
    for metric in ['mse', 'rmse', 'mae', 'mape', 'r2']:
        comparison[metric] = comparison['eval_metrics'].apply(lambda x: x[metric])
    
    comparison = comparison.drop(columns=['eval_metrics'])
    
    # Save comparison
    comparison_path = os.path.join(
        EXPERIMENT_CONFIG.results_path,
        'model_comparison.csv'
    )
    comparison.to_csv(comparison_path, index=False)
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison.to_string())
    
    return comparison


if __name__ == "__main__":
    # Test training with dummy data
    from data_loader import DataModule
    
    print("Testing trainer with sample data...")
    
    # Load data
    dm = DataModule()
    dm.load_data(num_msmetrics_files=2)
    
    # Get top service
    top_services = dm.get_top_services(1)
    msname = top_services[0][0]
    
    # Prepare data
    ts_data = dm.prepare_service_data(msname)
    
    if ts_data is not None and len(ts_data) > 100:
        # Create dataloaders
        train_loader, val_loader, test_loader = dm.create_dataloaders(
            ts_data.values,
            target_idx=0
        )
        
        # Run experiment
        results = run_experiment(
            'lstm',
            train_loader,
            val_loader,
            test_loader,
            input_size=ts_data.shape[1],
            num_epochs=5
        )
        
        print("\nExperiment completed!")
    else:
        print("Insufficient data for training")

