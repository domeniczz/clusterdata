"""
Data analysis and visualization module for Alibaba Microservices Trace v2022.
Provides comprehensive analysis of workload patterns and characteristics.

OPTIMIZED VERSION:
- Uses sampling for large datasets (>1M rows)
- Parallel processing with numpy vectorized operations
- Pre-aggregation strategies for service statistics
- Memory-efficient analysis pipelines
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
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
    
    OPTIMIZED: Uses sampling and efficient aggregation for large datasets.
    """
    
    def __init__(self, config: 'ExperimentConfig' = None, sample_size: int = 1_000_000):
        """
        Args:
            config: Experiment configuration
            sample_size: Maximum number of samples for analysis (default 1M)
        """
        self.config = config or EXPERIMENT_CONFIG
        self.sample_size = sample_size
        create_output_dirs()
    
    def _smart_sample(self, df: pd.DataFrame, preserve_services: bool = True) -> pd.DataFrame:
        """
        Smart sampling that preserves data distribution.
        Uses stratified sampling to maintain service diversity.
        """
        if len(df) <= self.sample_size:
            return df
        
        print(f"  [Sampling] Original size: {len(df):,}, sampling to ~{self.sample_size:,}")
        
        if preserve_services and 'msname' in df.columns:
            # Stratified sampling by service
            n_services = df['msname'].nunique()
            samples_per_service = max(100, self.sample_size // n_services)
            
            sampled = df.groupby('msname', group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), samples_per_service), random_state=42)
            )
            return sampled.reset_index(drop=True)
        else:
            # Simple random sampling
            return df.sample(n=self.sample_size, random_state=42)
        
    def basic_statistics(
        self, 
        df: pd.DataFrame, 
        features: List[str] = None,
        use_sampling: bool = True
    ) -> pd.DataFrame:
        """
        Compute basic statistics for the dataset.
        OPTIMIZED: Uses vectorized operations and optional sampling.
        """
        if features is None:
            features = ['cpu_utilization', 'memory_utilization']
        
        valid_features = [f for f in features if f in df.columns]
        
        # Sample for expensive operations
        df_sample = self._smart_sample(df, preserve_services=False) if use_sampling else df
        
        # Basic stats using describe (already optimized in pandas)
        stats = df_sample[valid_features].describe()
        
        # Vectorized additional statistics
        for feature in valid_features:
            data = df_sample[feature].dropna().values
            n = len(data)
            if n > 0:
                mean = np.mean(data)
                std = np.std(data)
                
                # Skewness (vectorized)
                if std > 0:
                    skew = np.mean(((data - mean) / std) ** 3)
                    kurt = np.mean(((data - mean) / std) ** 4) - 3
                else:
                    skew = 0
                    kurt = 0
                
                stats.loc['skewness', feature] = skew
                stats.loc['kurtosis', feature] = kurt
                stats.loc['non_zero_ratio', feature] = np.mean(data > 0)
        
        return stats
    
    def service_statistics(
        self, 
        df: pd.DataFrame,
        top_n: int = 100,
        use_agg_first: bool = True
    ) -> pd.DataFrame:
        """
        Compute statistics per microservice.
        OPTIMIZED: Pre-aggregation and limiting to top N services.
        
        Args:
            df: Input DataFrame
            top_n: Number of top services to return detailed stats
            use_agg_first: If True, first aggregate then compute stats (faster)
        """
        print(f"  [Service Stats] Processing {df['msname'].nunique():,} services...")
        
        if use_agg_first:
            # FAST PATH: Use named aggregation with optimized functions
            service_stats = df.groupby('msname', observed=True).agg(
                cpu_utilization_mean=('cpu_utilization', 'mean'),
                cpu_utilization_std=('cpu_utilization', 'std'),
                cpu_utilization_min=('cpu_utilization', 'min'),
                cpu_utilization_max=('cpu_utilization', 'max'),
                cpu_utilization_count=('cpu_utilization', 'count'),
                memory_utilization_mean=('memory_utilization', 'mean'),
                memory_utilization_std=('memory_utilization', 'std'),
                memory_utilization_min=('memory_utilization', 'min'),
                memory_utilization_max=('memory_utilization', 'max')
            )
        else:
            # Original method
            service_stats = df.groupby('msname').agg({
                'cpu_utilization': ['mean', 'std', 'min', 'max', 'count'],
                'memory_utilization': ['mean', 'std', 'min', 'max']
            })
            service_stats.columns = [
                '_'.join(col).strip() for col in service_stats.columns.values
            ]
        
        # Sort and return top N
        service_stats = service_stats.sort_values('cpu_utilization_count', ascending=False)
        
        if top_n and top_n < len(service_stats):
            print(f"  [Service Stats] Returning top {top_n} services")
            return service_stats.head(top_n)
        
        return service_stats
    
    def temporal_analysis(
        self,
        df: pd.DataFrame,
        time_col: str = 'timestamp',
        feature: str = 'cpu_utilization',
        use_sampling: bool = True
    ) -> pd.DataFrame:
        """
        Analyze temporal patterns in the data.
        OPTIMIZED: Direct aggregation without intermediate DataFrame copies.
        """
        # Sample if needed
        if use_sampling and len(df) > self.sample_size:
            df_sample = self._smart_sample(df, preserve_services=False)
        else:
            df_sample = df
        
        # Direct aggregation (no copy)
        temporal_agg = df_sample.groupby(time_col, observed=True)[feature].agg(
            ['mean', 'std', 'count']
        ).reset_index()
        
        return temporal_agg
    
    def correlation_analysis(
        self,
        df: pd.DataFrame,
        features: List[str] = None,
        use_sampling: bool = True
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for features.
        OPTIMIZED: Uses sampling for large datasets.
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        valid_features = [f for f in features if f in df.columns]
        
        # Sample for correlation
        if use_sampling and len(df) > self.sample_size:
            df_sample = df.sample(n=min(100000, len(df)), random_state=42)
        else:
            df_sample = df
        
        return df_sample[valid_features].corr()
    
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
        
        # Rolling statistics (pandas is already optimized for this)
        for ws in window_sizes:
            patterns['trend'][f'ma_{ws}'] = series.rolling(ws).mean()
            patterns['volatility'][f'std_{ws}'] = series.rolling(ws).std()
        
        # Autocorrelation
        max_lag = min(21, len(series) // 2)
        for lag in range(1, max_lag):
            patterns['autocorrelation'].append(series.autocorr(lag=lag))
        
        return patterns
    
    def quick_overview(self, df: pd.DataFrame) -> Dict:
        """
        Fast overview analysis for large datasets.
        Returns key statistics without expensive computations.
        """
        overview = {
            'total_records': len(df),
            'unique_services': df['msname'].nunique() if 'msname' in df.columns else 0,
            'unique_nodes': df['nodeid'].nunique() if 'nodeid' in df.columns else 0,
            'time_range': {
                'min': df['timestamp'].min(),
                'max': df['timestamp'].max(),
                'duration_hours': (df['timestamp'].max() - df['timestamp'].min()) / 3600000
            } if 'timestamp' in df.columns else None,
            'cpu_summary': {
                'mean': df['cpu_utilization'].mean(),
                'median': df['cpu_utilization'].median(),
                'std': df['cpu_utilization'].std()
            } if 'cpu_utilization' in df.columns else None,
            'memory_summary': {
                'mean': df['memory_utilization'].mean(),
                'median': df['memory_utilization'].median(),
                'std': df['memory_utilization'].std()
            } if 'memory_utilization' in df.columns else None
        }
        return overview


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


def analyze_dataset(
    df: pd.DataFrame, 
    output_dir: str = None,
    fast_mode: bool = True,
    sample_size: int = 1_000_000
):
    """
    Perform comprehensive analysis of the dataset.
    
    OPTIMIZED VERSION:
    - fast_mode=True: Uses sampling and efficient aggregation (recommended for >1M rows)
    - fast_mode=False: Full analysis (slow for large datasets)
    
    Args:
        df: Input DataFrame
        output_dir: Directory to save figures
        fast_mode: If True, use optimized analysis with sampling
        sample_size: Maximum samples for analysis (when fast_mode=True)
    
    Returns:
        Tuple of (basic_stats, service_stats, temporal_analysis)
    """
    import time
    start_time = time.time()
    
    analyzer = WorkloadAnalyzer(sample_size=sample_size)
    visualizer = WorkloadVisualizer()
    
    if output_dir is None:
        output_dir = EXPERIMENT_CONFIG.figures_path
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("WORKLOAD DATA ANALYSIS" + (" [OPTIMIZED]" if fast_mode else ""))
    print("=" * 60)
    print(f"Dataset size: {len(df):,} records")
    
    # Quick overview (always fast)
    print("\n0. Quick Overview")
    print("-" * 40)
    overview = analyzer.quick_overview(df)
    print(f"  Total records: {overview['total_records']:,}")
    print(f"  Unique services: {overview['unique_services']:,}")
    if overview['time_range']:
        print(f"  Time duration: {overview['time_range']['duration_hours']:.1f} hours")
    if overview['cpu_summary']:
        print(f"  CPU utilization: mean={overview['cpu_summary']['mean']:.4f}, "
              f"std={overview['cpu_summary']['std']:.4f}")
    
    # Basic statistics
    print("\n1. Basic Statistics")
    print("-" * 40)
    stats = analyzer.basic_statistics(df, use_sampling=fast_mode)
    print(stats.to_string())
    print(f"  Time: {time.time() - start_time:.1f}s")
    
    # Service statistics
    print("\n2. Service Statistics (Top 10)")
    print("-" * 40)
    step_time = time.time()
    service_stats = analyzer.service_statistics(df, top_n=100 if fast_mode else None)
    print(service_stats.head(10).to_string())
    print(f"  Time: {time.time() - step_time:.1f}s")
    
    # Temporal analysis
    print("\n3. Temporal Analysis")
    print("-" * 40)
    step_time = time.time()
    temporal = analyzer.temporal_analysis(df, use_sampling=fast_mode)
    print(f"Time range: {temporal['timestamp'].min()} - {temporal['timestamp'].max()} ms")
    print(f"Number of unique timestamps: {len(temporal)}")
    print(f"  Time: {time.time() - step_time:.1f}s")
    
    # Visualizations (use sampling for speed)
    print("\n4. Creating Visualizations...")
    print("-" * 40)
    step_time = time.time()
    
    # Sample for visualization if needed
    if fast_mode and len(df) > sample_size:
        df_viz = df.sample(n=sample_size, random_state=42)
        print(f"  Using {sample_size:,} samples for visualization")
    else:
        df_viz = df
    
    # Distribution plot
    visualizer.plot_distribution(
        df_viz,
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
        df_viz,
        'cpu_utilization',
        save_path=os.path.join(output_dir, 'cpu_cdf.png')
    )
    print("  - CDF plot saved")
    
    # Box plot by service (limit to top services)
    top_services = service_stats.head(10).index.tolist()
    visualizer.plot_boxplot_by_service(
        df_viz,
        services=top_services,
        save_path=os.path.join(output_dir, 'service_boxplot.png')
    )
    print("  - Service boxplot saved")
    print(f"  Time: {time.time() - step_time:.1f}s")
    
    plt.close('all')
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Analysis complete! Total time: {total_time:.1f}s")
    print(f"Figures saved to: {output_dir}")
    print("=" * 60)
    
    return stats, service_stats, temporal


def analyze_dataset_parallel(
    df: pd.DataFrame,
    output_dir: str = None,
    n_workers: int = 4
):
    """
    Parallel version of dataset analysis for maximum speed.
    Runs different analyses concurrently.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    start_time = time.time()
    analyzer = WorkloadAnalyzer(sample_size=500_000)
    
    if output_dir is None:
        output_dir = EXPERIMENT_CONFIG.figures_path
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("WORKLOAD DATA ANALYSIS [PARALLEL MODE]")
    print("=" * 60)
    print(f"Dataset size: {len(df):,} records, using {n_workers} workers")
    
    results = {}
    
    # Define analysis tasks
    def task_basic_stats():
        return ('basic_stats', analyzer.basic_statistics(df, use_sampling=True))
    
    def task_service_stats():
        return ('service_stats', analyzer.service_statistics(df, top_n=100))
    
    def task_temporal():
        return ('temporal', analyzer.temporal_analysis(df, use_sampling=True))
    
    def task_overview():
        return ('overview', analyzer.quick_overview(df))
    
    # Run tasks in parallel
    tasks = [task_basic_stats, task_service_stats, task_temporal, task_overview]
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(task): task.__name__ for task in tasks}
        
        for future in as_completed(futures):
            try:
                name, result = future.result()
                results[name] = result
                print(f"  Completed: {name}")
            except Exception as e:
                print(f"  Error in {futures[future]}: {e}")
    
    # Print results
    print("\n" + "-" * 40)
    print("Results Summary:")
    
    if 'overview' in results:
        ov = results['overview']
        print(f"  Records: {ov['total_records']:,}, Services: {ov['unique_services']:,}")
    
    if 'basic_stats' in results:
        print("\nBasic Statistics:")
        print(results['basic_stats'].to_string())
    
    if 'service_stats' in results:
        print("\nTop 10 Services:")
        print(results['service_stats'].head(10).to_string())
    
    print(f"\nTotal time: {time.time() - start_time:.1f}s")
    
    return (
        results.get('basic_stats'),
        results.get('service_stats'),
        results.get('temporal')
    )


if __name__ == "__main__":
    # Test with sample data
    from data_loader import DataModule
    
    print("Loading data for analysis...")
    dm = DataModule()
    dm.load_data(num_msmetrics_files=2)
    
    # Run analysis
    stats, service_stats, temporal = analyze_dataset(dm.msmetrics_data)


