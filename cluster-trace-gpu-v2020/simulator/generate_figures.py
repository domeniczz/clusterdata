#!/usr/bin/env python3
"""
Generate figures from existing .npy data files in the logs directory.

Usage:
    python generate_figures.py                    # Generate from all .npy files
    python generate_figures.py --pattern "1206*" # Generate from files matching pattern
"""

import argparse
from pathlib import Path
from utils import (plot_job_stats, plot_cluster_util,
                   plot_multi_job_stats, plot_multi_cluster_util)

LOG_DIR = Path(__file__).parent / 'logs'
FIGURES_DIR = Path(__file__).parent / 'figures'


def generate_figures(pattern="*"):
    """Generate figures from .npy files matching the pattern."""
    
    # Create figures directory if needed
    FIGURES_DIR.mkdir(exist_ok=True)
    
    # Find all matching files
    job_stats_files = sorted(LOG_DIR.glob(f"{pattern}job_stats.npy"))
    cluster_util_files = sorted(LOG_DIR.glob(f"{pattern}cluster_util.npy"))
    
    print(f"Searching in: {LOG_DIR}")
    print(f"Pattern: {pattern}")
    print(f"Found {len(job_stats_files)} job_stats files")
    print(f"Found {len(cluster_util_files)} cluster_util files")
    print()
    
    if not job_stats_files and not cluster_util_files:
        print("No .npy files found! Run the simulator first with EXPORT_JOB_STATS=True")
        return
    
    # Generate individual job stats plots
    if job_stats_files:
        print("Generating job stats plots...")
        for npyfile in job_stats_files:
            try:
                plot_job_stats(str(npyfile))
                png_path = str(npyfile).replace('.npy', '.png')
                print(f"  ✓ {Path(png_path).name}")
            except Exception as e:
                print(f"  ✗ Error plotting {npyfile.name}: {e}")
    
    # Generate individual cluster utilization plots
    if cluster_util_files:
        print("\nGenerating cluster utilization plots...")
        for npyfile in cluster_util_files:
            try:
                plot_cluster_util(str(npyfile))
                png_path = str(npyfile).replace('.npy', '.png')
                print(f"  ✓ {Path(png_path).name}")
            except Exception as e:
                print(f"  ✗ Error plotting {npyfile.name}: {e}")
    
    # Generate comparison plots if multiple files exist
    if len(job_stats_files) > 1:
        print("\nGenerating multi-policy job stats comparison...")
        try:
            plot_multi_job_stats([str(f) for f in job_stats_files])
            # Get base name for output file
            base_name = str(job_stats_files[0]).split('.log.')[0]
            print(f"  ✓ {Path(base_name).name}-job_stats.png")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    if len(cluster_util_files) > 1:
        print("\nGenerating multi-policy cluster utilization comparison...")
        try:
            plot_multi_cluster_util([str(f) for f in cluster_util_files])
            base_name = str(cluster_util_files[0]).split('.log.')[0]
            print(f"  ✓ {Path(base_name).name}-cluster_util.png")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n==========")
    print(f"Figures saved to: {LOG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate figures from simulation data.')
    parser.add_argument("-p", "--pattern", help="File pattern to match (default: *)", 
                        type=str, default="*")
    args = parser.parse_args()
    
    generate_figures(args.pattern)

