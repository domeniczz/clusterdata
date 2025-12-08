"""
Main experiment script for microservices workload prediction.
Runs comprehensive experiments comparing baselines and deep learning models.

Usage:
    python run_experiments.py --num_services 5 --num_epochs 50
    python run_experiments.py --quick  # Quick test with fewer services/epochs
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project path
project_path = os.path.dirname(os.path.abspath(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from config import (
    DATA_CONFIG, MODEL_CONFIG, EXPERIMENT_CONFIG, 
    create_output_dirs, get_all_features
)
from data_loader import DataModule
from models import create_model, get_model_info
from baseline_models import get_all_baseline_models, evaluate_baseline
from trainer import (
    WorkloadTrainer, run_deep_learning_experiment, run_baseline_experiment,
    compare_all_models, compute_metrics, print_metrics,
    plot_training_history, plot_predictions, plot_model_comparison,
    save_results
)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_service_experiment(
    dm: DataModule,
    service_name: str,
    num_epochs: int = 50,
    verbose: bool = True
) -> dict:
    """
    Run experiments for a single microservice.
    
    Returns:
        Dictionary with results for all models
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"EXPERIMENTS FOR SERVICE: {service_name}")
        print(f"{'='*60}")
    
    # Prepare time series data
    ts_data = dm.prepare_service_data(
        service_name,
        add_features=EXPERIMENT_CONFIG.use_lag_features
    )
    
    if ts_data is None:
        print(f"Skipping {service_name}: insufficient data")
        return None
    
    if len(ts_data) < MODEL_CONFIG.seq_length + MODEL_CONFIG.pred_length + 20:
        print(f"Skipping {service_name}: only {len(ts_data)} data points")
        return None
    
    if verbose:
        print(f"Time series shape: {ts_data.shape}")
        print(f"Features: {list(ts_data.columns[:5])}...")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = dm.create_dataloaders(
        ts_data.values,
        target_idx=0  # Predict first feature (target variable)
    )
    
    if train_loader is None or len(train_loader) == 0:
        print(f"Skipping {service_name}: cannot create dataloaders")
        return None
    
    input_size = ts_data.shape[1]
    output_size = MODEL_CONFIG.pred_length
    seq_length = MODEL_CONFIG.seq_length
    
    results = {
        'service_name': service_name,
        'num_samples': len(ts_data),
        'num_features': input_size,
        'models': {}
    }
    
    # ==================== Baseline Models ====================
    if verbose:
        print(f"\n--- Baseline Models ---")
    
    # Extract numpy arrays for baselines
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    for x, y in train_loader:
        X_train_list.append(x.numpy())
        y_train_list.append(y.numpy())
    
    if test_loader:
        for x, y in test_loader:
            X_test_list.append(x.numpy())
            y_test_list.append(y.numpy())
    
    if X_train_list and X_test_list:
        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test = np.concatenate(X_test_list)
        y_test = np.concatenate(y_test_list)
        
        baseline_results = run_baseline_experiment(
            X_train, y_train, X_test, y_test,
            include_ml=True, verbose=verbose
        )
        
        for _, row in baseline_results.iterrows():
            results['models'][row['model']] = {
                'model_class': 'baseline',
                'rmse': row['rmse'],
                'mae': row['mae'],
                'mape': row['mape'],
                'smape': row['smape'],
                'r2': row['r2'],
                'training_time': row['training_time']
            }
    
    # ==================== Deep Learning Models ====================
    if verbose:
        print(f"\n--- Deep Learning Models ---")
    
    # All available deep learning models for comprehensive comparison
    dl_models = [
        # RNN-based
        'lstm', 'gru', 'attention_lstm',
        # Transformer-based (SOTA)
        'transformer', 'patchtst', 'informer', 'autoformer',
        # CNN-based
        'tcn', 'timesnet',
        # Linear (strong baselines)
        'nlinear', 'dlinear'
    ]
    
    for model_type in dl_models:
        try:
            dl_results, trainer = run_deep_learning_experiment(
                model_type,
                train_loader, val_loader, test_loader,
                input_size, output_size, seq_length,
                num_epochs, verbose=verbose
            )
            
            results['models'][model_type.upper()] = {
                'model_class': 'deep_learning',
                'num_params': dl_results['num_params'],
                'best_val_loss': dl_results['best_val_loss'],
                'best_epoch': dl_results['best_epoch'],
                'training_time': dl_results['training_time'],
                **dl_results['eval_metrics']
            }
            
        except Exception as e:
            if verbose:
                print(f"Error with {model_type}: {e}")
            continue
    
    return results


def run_multi_service_experiments(
    num_services: int = 5,
    num_msmetrics_files: int = 24,
    num_msrtmcr_files: int = 100,
    num_epochs: int = 50,
    verbose: bool = True
) -> dict:
    """
    Run experiments across multiple microservices.
    
    Returns:
        Dictionary with all experiment results
    """
    print("=" * 70)
    print("MICROSERVICES WORKLOAD PREDICTION - COMPREHENSIVE EXPERIMENTS")
    print("=" * 70)
    
    set_seed(MODEL_CONFIG.random_seed)
    create_output_dirs()
    
    # Initialize data module
    dm = DataModule()
    
    # Load data
    print("\n[1/4] Loading data...")
    dm.load_data(
        num_msmetrics_files=num_msmetrics_files,
        num_msrtmcr_files=num_msrtmcr_files,
        load_msrtmcr=EXPERIMENT_CONFIG.use_msrtmcr,
        verbose=verbose
    )
    
    # Merge data sources if MSRTMCR is loaded
    if dm.msrtmcr_data is not None:
        print("\n[2/4] Merging data sources...")
        dm.merge_data_sources()
    else:
        print("\n[2/4] Using MSMetrics data only")
    
    # Get top services
    print(f"\n[3/4] Selecting top {num_services} microservices...")
    top_services = dm.get_top_services(n=num_services, min_points=100)
    
    if not top_services:
        print("ERROR: No suitable microservices found!")
        return None
    
    print(f"Selected services:")
    for i, (svc, count) in enumerate(top_services, 1):
        print(f"  {i}. {svc}: {count:,} records")
    
    # Run experiments for each service
    print(f"\n[4/4] Running experiments ({num_epochs} epochs per model)...")
    
    all_results = {
        'experiment_config': {
            'num_services': num_services,
            'num_epochs': num_epochs,
            'seq_length': MODEL_CONFIG.seq_length,
            'pred_length': MODEL_CONFIG.pred_length,
            'target_variable': EXPERIMENT_CONFIG.target_variable
        },
        'services': {}
    }
    
    for svc_name, svc_count in top_services:
        results = run_single_service_experiment(
            dm, svc_name, num_epochs, verbose
        )
        if results:
            all_results['services'][svc_name] = results
    
    return all_results


def aggregate_results(all_results: dict) -> pd.DataFrame:
    """Aggregate results from multiple services into a summary DataFrame."""
    summary_data = []
    
    for svc_name, svc_results in all_results.get('services', {}).items():
        for model_name, model_results in svc_results.get('models', {}).items():
            summary_data.append({
                'service': svc_name,
                'model': model_name,
                'model_class': model_results.get('model_class', 'unknown'),
                'rmse': model_results.get('rmse', float('inf')),
                'mae': model_results.get('mae', float('inf')),
                'mape': model_results.get('mape', float('inf')),
                'smape': model_results.get('smape', float('inf')),
                'r2': model_results.get('r2', float('-inf')),
                'training_time': model_results.get('training_time', 0)
            })
    
    if not summary_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(summary_data)
    return df


def generate_summary_report(all_results: dict, output_dir: str = None):
    """Generate summary report with tables and figures."""
    if output_dir is None:
        output_dir = EXPERIMENT_CONFIG.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY REPORT")
    print("=" * 70)
    
    # Aggregate results
    df = aggregate_results(all_results)
    
    if df.empty:
        print("No results to summarize")
        return
    
    # ==================== Summary Statistics ====================
    print("\n[1] Model Performance Summary (Averaged Across Services)")
    print("-" * 60)
    
    model_summary = df.groupby('model').agg({
        'rmse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'training_time': 'mean'
    }).round(4)
    
    model_summary.columns = ['RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std', 
                            'R2_mean', 'R2_std', 'Time_mean']
    model_summary = model_summary.sort_values('RMSE_mean')
    
    print(model_summary.to_string())
    
    # Save summary table
    summary_path = os.path.join(output_dir, 'results', 'model_summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    model_summary.to_csv(summary_path)
    print(f"\nSummary saved to: {summary_path}")
    
    # ==================== Best Model Per Service ====================
    print("\n[2] Best Model Per Service (by RMSE)")
    print("-" * 60)
    
    best_per_service = df.loc[df.groupby('service')['rmse'].idxmin()]
    print(best_per_service[['service', 'model', 'rmse', 'mae', 'r2']].to_string(index=False))
    
    # ==================== Model Win Counts ====================
    print("\n[3] Model Rankings (How often each model ranks best)")
    print("-" * 60)
    
    win_counts = best_per_service['model'].value_counts()
    print(win_counts.to_string())
    
    # ==================== Visualization ====================
    print("\n[4] Generating visualizations...")
    
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # Figure 1: Model comparison bar chart
    plt.figure(figsize=(12, 6), dpi=120)
    
    model_order = model_summary.index.tolist()
    colors = ['#2ecc71' if 'Naive' in m or 'Moving' in m or 'Exp' in m or 
              'XGBoost' in m or 'LightGBM' in m or 'Random' in m or 
              'Linear' in m or 'SVR' in m or 'Seasonal' in m
              else '#3498db' for m in model_order]
    
    plt.barh(model_order, model_summary['RMSE_mean'], 
             xerr=model_summary['RMSE_std'], 
             color=colors, edgecolor='black', linewidth=0.5, capsize=3)
    plt.xlabel('RMSE (Mean ± Std)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('Model Comparison: RMSE Across All Services', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'model_comparison_rmse.png'), dpi=150)
    plt.close()
    
    # Figure 2: Heatmap of model performance across services
    if len(df['service'].unique()) > 1 and len(df['model'].unique()) > 1:
        pivot_rmse = df.pivot(index='service', columns='model', values='rmse')
        
        plt.figure(figsize=(14, 8), dpi=120)
        import seaborn as sns
        sns.heatmap(pivot_rmse, annot=True, fmt='.3f', cmap='RdYlGn_r',
                   linewidths=0.5, cbar_kws={'label': 'RMSE'})
        plt.title('RMSE Heatmap: Services vs Models', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'performance_heatmap.png'), dpi=150)
        plt.close()
    
    # Figure 3: Training time comparison
    plt.figure(figsize=(12, 6), dpi=120)
    time_data = df.groupby('model')['training_time'].mean().sort_values()
    plt.barh(time_data.index, time_data.values, color='steelblue', edgecolor='black')
    plt.xlabel('Training Time (seconds)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('Average Training Time by Model', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'training_time.png'), dpi=150)
    plt.close()
    
    print(f"Figures saved to: {fig_dir}")
    
    # Save full results
    results_path = os.path.join(output_dir, 'results', 'full_results.json')
    
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
    
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\nFull results saved to: {results_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total services evaluated: {len(all_results.get('services', {}))}")
    print(f"Total models compared: {len(model_summary)}")
    print(f"\nBest overall model: {model_summary.index[0]} (RMSE: {model_summary['RMSE_mean'].iloc[0]:.4f})")
    
    return model_summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Microservices Workload Prediction Experiments'
    )
    parser.add_argument(
        '--num_services', type=int, default=5,
        help='Number of top microservices to evaluate (default: 5)'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--num_msmetrics', type=int, default=24,
        help='Number of MSMetrics files to load (default: 24, all available)'
    )
    parser.add_argument(
        '--num_msrtmcr', type=int, default=100,
        help='Number of MSRTMCR files to load (default: 100)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test mode (2 services, 10 epochs)'
    )
    parser.add_argument(
        '--verbose', action='store_true', default=True,
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.num_services = 2
        args.num_epochs = 10
        args.num_msmetrics = 4
        args.num_msrtmcr = 20
        print("Running in QUICK TEST mode")
    
    # Run experiments
    start_time = time.time()
    
    all_results = run_multi_service_experiments(
        num_services=args.num_services,
        num_msmetrics_files=args.num_msmetrics,
        num_msrtmcr_files=args.num_msrtmcr,
        num_epochs=args.num_epochs,
        verbose=args.verbose
    )
    
    if all_results:
        # Generate report
        generate_summary_report(all_results)
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Total experiment time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

