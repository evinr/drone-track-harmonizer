#!/usr/bin/env python3
"""
OcuSync 4 ID Pattern Visualization
----------------------------------
This script creates visualizations of OcuSync 4 ID patterns to help
understand the relationships between IDs across different bandwidths.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_id_relationships(relationships_file, output_dir='visualizations'):
    """Plot relationships between IDs across different bandwidths"""
    if not os.path.exists(relationships_file):
        logger.warning(f"File not found: {relationships_file}")
        return
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(relationships_file)
    
    if df.empty:
        logger.warning(f"No data in file: {relationships_file}")
        return
    
    # Ensure 'ratio' column exists
    if 'ratio' not in df.columns:
        logger.warning(f"'ratio' column not found in {relationships_file}")
        return
    
    # Add bandwidth pair column if it doesn't exist
    if 'bandwidth_pair' not in df.columns:
        df['bandwidth_pair'] = df['bandwidth_from'] + '_to_' + df['bandwidth_to']
    
    # Plot histogram of ratios for each bandwidth pair
    bandwidth_pairs = df['bandwidth_pair'].unique()
    
    for pair in bandwidth_pairs:
        pair_data = df[df['bandwidth_pair'] == pair]
        
        # Skip if too few data points
        if len(pair_data) < 5:
            continue
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(pair_data['ratio'].dropna(), bins=50, kde=True)
        
        # Add mean and median lines
        mean_ratio = pair_data['ratio'].mean()
        median_ratio = pair_data['ratio'].median()
        
        plt.axvline(mean_ratio, color='r', linestyle='--', label=f'Mean: {mean_ratio:.2f}')
        plt.axvline(median_ratio, color='g', linestyle='--', label=f'Median: {median_ratio:.2f}')
        
        # Check for specific ratio patterns (1:2, 1:4, etc.)
        common_ratios = [1, 2, 4, 10, 20]
        for ratio in common_ratios:
            if 0.5 <= ratio <= 10:  # Only plot within a reasonable range
                plt.axvline(ratio, color='b', alpha=0.3, linestyle=':', label=f'Ratio {ratio}:1')
        
        plt.title(f'ID Ratio Distribution: {pair}')
        plt.xlabel('ID Ratio (id_to / id_from)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Adjust x-axis to focus on the most common ratios
        q1, q3 = np.percentile(pair_data['ratio'].dropna(), [5, 95])
        plt.xlim(max(0, q1-1), min(q3+1, 10))
        
        # Save figure
        plt.savefig(f'{output_dir}/ratio_histogram_{pair}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create scatter plots of id_from vs id_to
    for pair in bandwidth_pairs:
        pair_data = df[df['bandwidth_pair'] == pair]
        
        # Skip if too few data points
        if len(pair_data) < 5:
            continue
        
        # Plot scatter
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=pair_data, x='id_from', y='id_to', alpha=0.6)
        
        # Add potential linear relationship lines
        x_min, x_max = pair_data['id_from'].min(), pair_data['id_from'].max()
        
        # Try to fit a line
        if len(pair_data) >= 10:
            try:
                # Linear regression
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    pair_data['id_from'].dropna(), 
                    pair_data['id_to'].dropna()
                )
                
                # Only plot if reasonable fit
                if r_value**2 >= 0.5:
                    x = np.array([x_min, x_max])
                    y = slope * x + intercept
                    plt.plot(x, y, 'r--', 
                            label=f'y = {slope:.2f}x + {intercept:.2f} (R²: {r_value**2:.2f})')
            except:
                pass
        
        plt.title(f'ID Relationship: {pair}')
        plt.xlabel(f'ID from {pair.split("_to_")[0]}')
        plt.ylabel(f'ID to {pair.split("_to_")[1]}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        plt.savefig(f'{output_dir}/id_scatter_{pair}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_drone_correlations(correlations_file, output_dir='visualizations'):
    """Plot correlations between RF detections and DJI drones"""
    if not os.path.exists(correlations_file):
        logger.warning(f"File not found: {correlations_file}")
        return
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(correlations_file)
    
    if df.empty:
        logger.warning(f"No data in file: {correlations_file}")
        return
    
    # Ensure required columns exist
    required_cols = ['serial_number', 'distance', 'rssi', 'bandwidth']
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"'{col}' column not found in {correlations_file}")
            return
    
    # Add timestamp if needed
    if 'detection_time' in df.columns:
        df['detection_time'] = pd.to_datetime(df['detection_time'])
    
    # Plot RSSI vs distance for each bandwidth
    bandwidths = df['bandwidth'].unique()
    
    plt.figure(figsize=(12, 8))
    
    for bw in bandwidths:
        bw_data = df[df['bandwidth'] == bw]
        
        # Skip if too few data points
        if len(bw_data) < 5:
            continue
        
        # Plot
        plt.scatter(bw_data['distance'], bw_data['rssi'], alpha=0.5, label=bw)
    
    # Try to fit log curve (RSSI = A*log10(distance) + B)
    try:
        from scipy.optimize import curve_fit
        
        def log_func(x, a, b):
            return a * np.log10(x) + b
        
        # Combine all data
        valid_data = df[['distance', 'rssi']].dropna()
        valid_data = valid_data[valid_data['distance'] > 0]  # Avoid log(0)
        
        if len(valid_data) >= 10:
            # Fit curve
            popt, _ = curve_fit(log_func, valid_data['distance'], valid_data['rssi'])
            
            # Plot fitted curve
            x_range = np.linspace(valid_data['distance'].min(), valid_data['distance'].max(), 100)
            plt.plot(x_range, log_func(x_range, *popt), 'r-', 
                    label=f'Fitted curve: {popt[0]:.2f}*log10(d) + {popt[1]:.2f}')
    except:
        pass
    
    plt.title('RSSI vs Distance by Bandwidth')
    plt.xlabel('Distance')
    plt.ylabel('RSSI (dBm)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save figure
    plt.savefig(f'{output_dir}/rssi_vs_distance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot ID correlations by serial number
    serial_numbers = df['serial_number'].unique()
    
    for serial in serial_numbers:
        serial_data = df[df['serial_number'] == serial]
        
        # Skip if too few data points or only one bandwidth
        if len(serial_data) < 5 or serial_data['bandwidth'].nunique() <= 1:
            continue
        
        # Group by bandwidth
        id_mapping = defaultdict(list)
        
        for bw in serial_data['bandwidth'].unique():
            bw_data = serial_data[serial_data['bandwidth'] == bw]
            id_list = bw_data['id_number'].unique()
            
            for id_num in id_list:
                id_mapping[bw].append(id_num)
        
        # Create ID combinations
        bandwidths = list(id_mapping.keys())
        combinations = []
        
        for i, bw1 in enumerate(bandwidths):
            for j, bw2 in enumerate(bandwidths[i+1:], i+1):
                for id1 in id_mapping[bw1]:
                    for id2 in id_mapping[bw2]:
                        combinations.append({
                            'bandwidth_from': bw1,
                            'id_from': id1,
                            'bandwidth_to': bw2,
                            'id_to': id2,
                            'ratio': id2 / id1 if id1 != 0 else None
                        })
        
        if combinations:
            # Convert to DataFrame
            combo_df = pd.DataFrame(combinations)
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            
            # Group by bandwidth pair
            combo_df['bandwidth_pair'] = combo_df['bandwidth_from'] + '_to_' + combo_df['bandwidth_to']
            
            for pair in combo_df['bandwidth_pair'].unique():
                pair_data = combo_df[combo_df['bandwidth_pair'] == pair]
                plt.scatter(pair_data['id_from'], pair_data['id_to'], alpha=0.7, label=pair)
            
            plt.title(f'ID Relationships for Drone {serial}')
            plt.xlabel('ID From')
            plt.ylabel('ID To')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save figure
            plt.savefig(f'{output_dir}/id_relationships_drone_{serial}.png', dpi=300, bbox_inches='tight')
            plt.close()

def plot_spatial_clustering(clusters_file, output_dir='visualizations'):
    """Visualize spatial clusters of RF detections"""
    if not os.path.exists(clusters_file):
        logger.warning(f"File not found: {clusters_file}")
        return
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(clusters_file)
    
    if df.empty:
        logger.warning(f"No data in file: {clusters_file}")
        return
    
    # Ensure required columns exist
    required_cols = ['cluster_id', 'latitude', 'longitude', 'bandwidth', 'id_number']
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"'{col}' column not found in {clusters_file}")
            return
    
    # Plot each cluster on a map
    clusters = df['cluster_id'].unique()
    
    for cluster_id in clusters:
        cluster_data = df[df['cluster_id'] == cluster_id]
        
        # Skip if too few points
        if len(cluster_data) < 3:
            continue
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot points by bandwidth
        for bw in cluster_data['bandwidth'].unique():
            bw_data = cluster_data[cluster_data['bandwidth'] == bw]
            plt.scatter(bw_data['longitude'], bw_data['latitude'], alpha=0.7, 
                      label=f"{bw} (IDs: {', '.join(map(str, bw_data['id_number'].unique()))})")
        
        plt.title(f'Spatial Cluster {cluster_id}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        plt.savefig(f'{output_dir}/spatial_cluster_{cluster_id}.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_visualization(pattern_stats_file, output_dir='visualizations'):
    """Create summary visualization of pattern statistics"""
    if not os.path.exists(pattern_stats_file):
        logger.warning(f"File not found: {pattern_stats_file}")
        return
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(pattern_stats_file)
    
    if df.empty:
        logger.warning(f"No data in file: {pattern_stats_file}")
        return
    
    # Ensure required columns exist
    required_cols = ['bandwidth_pair', 'mean_ratio', 'median_ratio', 'std_ratio', 'count']
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"'{col}' column not found in {pattern_stats_file}")
            return
    
    # Create bar chart of mean ratios
    plt.figure(figsize=(12, 8))
    
    # Sort by count
    df = df.sort_values('count', ascending=False)
    
    # Create bar chart
    bars = plt.bar(df['bandwidth_pair'], df['mean_ratio'], yerr=df['std_ratio'], alpha=0.7)
    
    # Add count labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + df['std_ratio'].iloc[i] + 0.1,
                f'n={df["count"].iloc[i]}', ha='center')
    
    # Add horizontal lines for common ratios
    for ratio in [1, 2, 4]:
        plt.axhline(ratio, color='r', alpha=0.3, linestyle='--', label=f'Ratio {ratio}:1')
    
    plt.title('Mean ID Ratios by Bandwidth Pair')
    plt.xlabel('Bandwidth Pair')
    plt.ylabel('Mean Ratio (id_to / id_from)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    
    # Save figure
    plt.savefig(f'{output_dir}/summary_ratios.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main visualization function"""
    # Create directories
    os.makedirs('visualizations', exist_ok=True)
    
    # Check if results directory exists
    if not os.path.exists('results'):
        logger.warning("Results directory not found. Run analysis script first.")
        return
    
    # Visualize ID relationships
    logger.info("Visualizing ID relationships from clustering...")
    plot_id_relationships('results/cluster_relationships.csv')
    
    logger.info("Visualizing ID relationships from DJI correlations...")
    plot_id_relationships('results/dji_id_relationships.csv')
    
    # Visualize drone correlations
    logger.info("Visualizing drone correlations...")
    plot_drone_correlations('results/dji_correlations.csv')
    
    # Visualize spatial clusters
    logger.info("Visualizing spatial clusters...")
    plot_spatial_clustering('results/spatial_clusters.csv')
    
    # Create summary visualizations
    logger.info("Creating summary visualizations...")
    create_summary_visualization('results/cluster_pattern_stats.csv')
    create_summary_visualization('results/dji_pattern_stats.csv')
    
    logger.info("Visualization complete. Results saved to 'visualizations' directory.")

if __name__ == "__main__":
    main()