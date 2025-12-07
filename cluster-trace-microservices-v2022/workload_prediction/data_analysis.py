"""
Data analysis and visualization module for Alibaba Microservices Trace v2022.
Provides comprehensive analysis of workload patterns and characteristics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import DATA_CONFIG, EXPERIMENT_CONFIG, create_output_dirs


# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class WorkloadAnalyzer:
    """
    Analyzer for microservices workload data.
    Provides statistical analysis and visualization capabilities.
    """
    
    def __init__(self, config: 'ExperimentConfig' = None):
        self.config = config or EXPERIMENT_CONFIG
        create_output_dirs()
        
    def basic_statistics(self, df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
        """
        Compute basic statistics for the dataset.
        """
        if features is None:
            features = ['cpu_utilization', 'memory_utilization']
        
        valid_features = [f for f in features if f in df.columns]
        
        stats = df[valid_features].describe()
        
        # Add additional statistics
        for feature in valid_features:
            stats.loc['skewness', feature] = df[feature].skew()
            stats.loc['kurtosis', feature] = df[feature].kurtosis()
            stats.loc['non_zero_ratio', feature] = (df[feature] > 0).mean()
        
        return stats
    
    def service_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistics per microservice.
        """
        service_stats = df.groupby('msname').agg({
            'cpu_utilization': ['mean', 'std', 'min', 'max', 'count'],
            'memory_utilization': ['mean', 'std', 'min', 'max']
        })
        
        # Flatten column names
        service_stats.columns = [
            '_'.join(col).strip() for col in service_stats.columns.values
        ]
        
        return service_stats.sort_values('cpu_utilization_count', ascending=False)
    
    def temporal_analysis(
        self,
        df: pd.DataFrame,
        time_col: str = 'timestamp',
        feature: str = 'cpu_utilization'
    ) -> pd.DataFrame:
        """
        Analyze temporal patterns in the data.
        """
        df_copy = df.copy()
        
        # Convert timestamp (ms) to datetime
        df_copy['datetime'] = pd.to_datetime(df_copy[time_col], unit='ms')
        df_copy['minute'] = df_copy['datetime'].dt.minute
        df_copy['hour'] = df_copy['datetime'].dt.hour
        
        # Aggregate by time
        temporal_agg = df_copy.groupby(time_col)[feature].agg(['mean', 'std', 'count'])
        temporal_agg = temporal_agg.reset_index()
        
        return temporal_agg
    
    def correlation_analysis(
        self,
        df: pd.DataFrame,
        features: List[str] = None
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for features.
        """
        if features is None:
            # Use all numeric columns
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        valid_features = [f for f in features if f in df.columns]
        
        return df[valid_features].corr()
    
    def detect_patterns(
        self,
        series: pd.Series,
        window_sizes: List[int] = None
    ) -> Dict:
        """
        Detect patterns in time series using rolling statistics.
        """
        if window_sizes is None:
            window_sizes = [5, 10, 30, 60]
        
        patterns = {
            'trend': {},
            'volatility': {},
            'autocorrelation': []
        }
        
        # Rolling statistics
        for ws in window_sizes:
            patterns['trend'][f'ma_{ws}'] = series.rolling(ws).mean()
            patterns['volatility'][f'std_{ws}'] = series.rolling(ws).std()
        
        # Autocorrelation
        for lag in range(1, min(21, len(series) // 2)):
            patterns['autocorrelation'].append(series.autocorr(lag=lag))
        
        return patterns


class WorkloadVisualizer:
    """
    Visualizer for microservices workload data.
    Creates publication-quality figures.
    """
    
    def __init__(self, config: 'ExperimentConfig' = None, figsize: Tuple[int, int] = (12, 6)):
        self.config = config or EXPERIMENT_CONFIG
        self.figsize = figsize
        create_output_dirs()
        
    def plot_time_series(
        self,
        df: pd.DataFrame,
        time_col: str = 'timestamp',
        feature: str = 'cpu_utilization',
        title: str = None,
        save_path: str = None
    ):
        """
        Plot time series of a feature.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=120)
        
        # Convert timestamp to relative time in minutes
        min_time = df[time_col].min()
        x = (df[time_col] - min_time) / 60000  # Convert ms to minutes
        
        ax.plot(x, df[feature], linewidth=0.8, alpha=0.8)
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title or f'{feature.replace("_", " ").title()} Over Time', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, ax
    
    def plot_distribution(
        self,
        df: pd.DataFrame,
        features: List[str] = None,
        title: str = None,
        save_path: str = None
    ):
        """
        Plot distribution of features using histograms and KDE.
        """
        if features is None:
            features = ['cpu_utilization', 'memory_utilization']
        
        valid_features = [f for f in features if f in df.columns]
        n_features = len(valid_features)
        
        fig, axes = plt.subplots(1, n_features, figsize=(6 * n_features, 5), dpi=120)
        if n_features == 1:
            axes = [axes]
        
        for ax, feature in zip(axes, valid_features):
            data = df[feature].dropna()
            
            # Histogram with KDE
            ax.hist(data, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # KDE
            if len(data) > 10:
                data.plot.kde(ax=ax, linewidth=2, color='red')
            
            ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(f'Distribution of {feature.replace("_", " ").title()}', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title or 'Feature Distributions', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, axes
    
    def plot_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        title: str = None,
        save_path: str = None
    ):
        """
        Plot correlation heatmap.
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax,
            annot_kws={'size': 8}
        )
        
        ax.set_title(title or 'Feature Correlation Matrix', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, ax
    
    def plot_service_comparison(
        self,
        df: pd.DataFrame,
        services: List[str],
        feature: str = 'cpu_utilization',
        time_col: str = 'timestamp',
        title: str = None,
        save_path: str = None
    ):
        """
        Plot comparison of workload across multiple services.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=120)
        
        for svc in services:
            svc_data = df[df['msname'] == svc].sort_values(time_col)
            if len(svc_data) > 0:
                min_time = svc_data[time_col].min()
                x = (svc_data[time_col] - min_time) / 60000
                ax.plot(x, svc_data[feature], label=svc, linewidth=0.8, alpha=0.8)
        
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title or f'{feature.replace("_", " ").title()} Comparison', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, ax
    
    def plot_boxplot_by_service(
        self,
        df: pd.DataFrame,
        services: List[str] = None,
        feature: str = 'cpu_utilization',
        title: str = None,
        save_path: str = None
    ):
        """
        Plot boxplot of feature distribution by service.
        """
        if services is None:
            # Get top 10 services by count
            top_services = df['msname'].value_counts().head(10).index.tolist()
            services = top_services
        
        plot_data = df[df['msname'].isin(services)]
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
        
        sns.boxplot(
            data=plot_data,
            x='msname',
            y=feature,
            ax=ax,
            palette='husl'
        )
        
        ax.set_xlabel('Microservice', fontsize=12)
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title or f'{feature.replace("_", " ").title()} by Service', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, ax
    
    def plot_temporal_pattern(
        self,
        temporal_agg: pd.DataFrame,
        feature_mean: str = 'mean',
        feature_std: str = 'std',
        time_col: str = 'timestamp',
        title: str = None,
        save_path: str = None
    ):
        """
        Plot temporal pattern with confidence interval.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=120)
        
        # Convert timestamp to relative time in minutes
        min_time = temporal_agg[time_col].min()
        x = (temporal_agg[time_col] - min_time) / 60000
        
        mean_values = temporal_agg[feature_mean]
        std_values = temporal_agg[feature_std].fillna(0)
        
        ax.plot(x, mean_values, linewidth=2, label='Mean')
        ax.fill_between(
            x,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.3,
            label='±1 Std'
        )
        
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title or 'Temporal Pattern', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, ax
    
    def plot_autocorrelation(
        self,
        autocorr: List[float],
        title: str = None,
        save_path: str = None
    ):
        """
        Plot autocorrelation function.
        """
        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        
        lags = range(1, len(autocorr) + 1)
        
        ax.bar(lags, autocorr, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        # Confidence interval (approximate)
        n = len(autocorr) * 10  # Approximate sample size
        ci = 1.96 / np.sqrt(n)
        ax.axhline(y=ci, color='red', linestyle='--', linewidth=1, label=f'95% CI (±{ci:.3f})')
        ax.axhline(y=-ci, color='red', linestyle='--', linewidth=1)
        
        ax.set_xlabel('Lag', fontsize=12)
        ax.set_ylabel('Autocorrelation', fontsize=12)
        ax.set_title(title or 'Autocorrelation Function', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, ax
    
    def plot_cdf(
        self,
        df: pd.DataFrame,
        feature: str,
        title: str = None,
        save_path: str = None
    ):
        """
        Plot Cumulative Distribution Function (CDF).
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        
        data = df[feature].dropna().sort_values()
        cdf = np.arange(1, len(data) + 1) / len(data)
        
        ax.plot(data, cdf, linewidth=2)
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('CDF', fontsize=12)
        ax.set_title(title or f'CDF of {feature.replace("_", " ").title()}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, ax


def analyze_dataset(df: pd.DataFrame, output_dir: str = None):
    """
    Perform comprehensive analysis of the dataset.
    """
    analyzer = WorkloadAnalyzer()
    visualizer = WorkloadVisualizer()
    
    if output_dir is None:
        output_dir = EXPERIMENT_CONFIG.figures_path
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("WORKLOAD DATA ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    print("\n1. Basic Statistics")
    print("-" * 40)
    stats = analyzer.basic_statistics(df)
    print(stats.to_string())
    
    # Service statistics
    print("\n2. Service Statistics (Top 10)")
    print("-" * 40)
    service_stats = analyzer.service_statistics(df)
    print(service_stats.head(10).to_string())
    
    # Temporal analysis
    print("\n3. Temporal Analysis")
    print("-" * 40)
    temporal = analyzer.temporal_analysis(df)
    print(f"Time range: {temporal['timestamp'].min()} - {temporal['timestamp'].max()} ms")
    print(f"Number of unique timestamps: {len(temporal)}")
    
    # Visualizations
    print("\n4. Creating Visualizations...")
    print("-" * 40)
    
    # Distribution plot
    visualizer.plot_distribution(
        df,
        features=['cpu_utilization', 'memory_utilization'],
        save_path=os.path.join(output_dir, 'distribution.png')
    )
    print("  - Distribution plot saved")
    
    # Temporal pattern
    visualizer.plot_temporal_pattern(
        temporal,
        title='CPU Utilization Temporal Pattern',
        save_path=os.path.join(output_dir, 'temporal_pattern.png')
    )
    print("  - Temporal pattern plot saved")
    
    # CDF
    visualizer.plot_cdf(
        df,
        'cpu_utilization',
        save_path=os.path.join(output_dir, 'cpu_cdf.png')
    )
    print("  - CDF plot saved")
    
    # Box plot by service
    visualizer.plot_boxplot_by_service(
        df,
        save_path=os.path.join(output_dir, 'service_boxplot.png')
    )
    print("  - Service boxplot saved")
    
    plt.close('all')
    print("\nAnalysis complete!")
    print(f"Figures saved to: {output_dir}")
    
    return stats, service_stats, temporal


if __name__ == "__main__":
    # Test with sample data
    from data_loader import DataModule
    
    print("Loading data for analysis...")
    dm = DataModule()
    dm.load_data(num_msmetrics_files=2)
    
    # Run analysis
    stats, service_stats, temporal = analyze_dataset(dm.msmetrics_data)


