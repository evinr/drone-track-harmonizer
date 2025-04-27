#!/usr/bin/env python3
"""
OcuSync 4 ID Sample Analysis
---------------------------
This script performs a quick analysis on a small sample of the data
to verify the approach before running the full analysis.
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import logging
import os
from collections import defaultdict
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Regular expression to extract OcuSync 4 IDs
OCUSYNC_PATTERN = r'V4_(\d+MHz)_(\d+)'

def extract_ocusync_info(rf_uuid):
    """Extract bandwidth and ID number from OcuSync 4 ID string"""
    match = re.search(OCUSYNC_PATTERN, rf_uuid, re.IGNORECASE)
    if match:
        bandwidth = match.group(1)
        id_number = int(match.group(2))
        return bandwidth, id_number
    return None, None

def load_sample_data(rf_file, dji_file, sample_size=10000):
    """Load a sample of the data for quick analysis"""
    logger.info(f"Loading {sample_size} rows from {rf_file}...")
    rf_data = pd.read_csv(rf_file, nrows=sample_size)
    
    # Parse timestamps - convert all to timezone naive
    rf_data['time'] = pd.to_datetime(rf_data['time'], format='ISO8601').dt.tz_localize(None)
    
    # Extract bandwidth and ID from rf_uuid
    rf_data['bandwidth'] = None
    rf_data['id_number'] = None
    
    ocusync_entries = rf_data['rf_uuid'].str.contains('V4_', case=False, na=False)
    
    # Apply only to OcuSync entries
    if ocusync_entries.any():
        ocusync_df = rf_data[ocusync_entries].copy()
        
        # Extract bandwidth and ID number
        results = ocusync_df['rf_uuid'].apply(extract_ocusync_info)
        ocusync_df['bandwidth'] = results.apply(lambda x: x[0])
        ocusync_df['id_number'] = results.apply(lambda x: x[1])
        
        # Update original dataframe
        rf_data.loc[ocusync_entries, 'bandwidth'] = ocusync_df['bandwidth']
        rf_data.loc[ocusync_entries, 'id_number'] = ocusync_df['id_number']
    
    logger.info(f"Found {ocusync_entries.sum()} OcuSync 4 entries")
    
    # Load DJI data
    logger.info(f"Loading DJI data from {dji_file}...")
    dji_data = pd.read_csv(dji_file)
    
    # Parse timestamps - convert all to timezone naive
    dji_data['DetectionTime'] = pd.to_datetime(dji_data['DetectionTime']).dt.tz_localize(None)
    
    return rf_data, dji_data

def find_bandwidth_transitions(rf_data, time_window=1):
    """Find bandwidth transitions within the same track"""
    logger.info("Finding bandwidth transitions...")
    
    # Group by track_id and sort by time
    track_groups = rf_data.groupby('track_id')
    
    # Store transitions
    transitions = []
    
    for track_id, group in track_groups:
        if 'OcuSync' not in str(track_id):
            continue
            
        # Sort by time
        group = group.sort_values('time')
        
        # Look for bandwidth transitions
        prev_time = None
        prev_bandwidth = None
        prev_id = None
        
        for _, row in group.iterrows():
            if row['bandwidth'] is None:
                continue
                
            current_time = row['time']
            current_bandwidth = row['bandwidth']
            current_id = row['id_number']
            
            if prev_time is not None and prev_bandwidth != current_bandwidth:
                # Check if within time window
                time_diff = (current_time - prev_time).total_seconds()
                
                if time_diff <= time_window:
                    # Found a bandwidth transition
                    transitions.append({
                        'track_id': track_id,
                        'time': current_time,
                        'bandwidth_from': prev_bandwidth,
                        'id_from': prev_id,
                        'bandwidth_to': current_bandwidth,
                        'id_to': current_id,
                        'time_diff': time_diff,
                        'ratio': current_id / prev_id if prev_id != 0 else None
                    })
            
            prev_time = current_time
            prev_bandwidth = current_bandwidth
            prev_id = current_id
    
    transitions_df = pd.DataFrame(transitions)
    logger.info(f"Found {len(transitions_df)} bandwidth transitions")
    
    return transitions_df

def correlate_with_dji(rf_data, dji_data, time_window=10):
    """Correlate RF detections with DJI positional data"""
    logger.info("Correlating RF detections with DJI data...")
    
    # Focus on OcuSync entries
    ocusync_data = rf_data[rf_data['bandwidth'].notna()].copy()
    
    # Store correlations
    correlations = []
    
    # Process each DJI detection
    for _, dji_row in dji_data.iterrows():
        detection_time = dji_row['DetectionTime']
        
        # Find RF detections within time window
        time_min = detection_time - pd.Timedelta(seconds=time_window)
        time_max = detection_time + pd.Timedelta(seconds=time_window)
        
        nearby_rf = ocusync_data[
            (ocusync_data['time'] >= time_min) & 
            (ocusync_data['time'] <= time_max)
        ]
        
        # Calculate distance from drone to each RF sensor
        if len(nearby_rf) > 0:
            for _, rf_row in nearby_rf.iterrows():
                # Calculate distance
                dist = np.sqrt(
                    (rf_row['sensor_latitude'] - dji_row['DroneLatitude'])**2 + 
                    (rf_row['sensor_longitude'] - dji_row['DroneLongitude'])**2
                )
                
                # Record correlation
                correlations.append({
                    'serial_number': dji_row['SerialNumber'],
                    'drone_type': dji_row['DroneType'],
                    'detection_time': detection_time,
                    'rf_time': rf_row['time'],
                    'time_diff': (rf_row['time'] - detection_time).total_seconds(),
                    'bandwidth': rf_row['bandwidth'],
                    'id_number': rf_row['id_number'],
                    'rssi': rf_row['rf_rssi'],
                    'distance': dist
                })
    
    corr_df = pd.DataFrame(correlations)
    logger.info(f"Found {len(corr_df)} RF-DJI correlations")
    
    return corr_df

def analyze_id_patterns(transitions_df):
    """Analyze patterns between ID numbers across different bandwidths"""
    logger.info("Analyzing ID patterns across bandwidths...")
    
    # Add bandwidth pair column if it doesn't exist
    if 'bandwidth_pair' not in transitions_df.columns:
        transitions_df['bandwidth_pair'] = transitions_df['bandwidth_from'] + '_to_' + transitions_df['bandwidth_to']
    
    # Group by bandwidth pair
    bandwidth_pairs = transitions_df.groupby('bandwidth_pair')
    
    # Analyze each bandwidth pair
    pattern_results = {}
    
    for pair_name, pair_group in bandwidth_pairs:
        # Calculate ratios between IDs if not already calculated
        if 'ratio' not in pair_group.columns:
            pair_group = pair_group.copy()
            pair_group['ratio'] = pair_group['id_to'] / pair_group['id_from']
        
        # Calculate statistics
        mean_ratio = pair_group['ratio'].mean()
        median_ratio = pair_group['ratio'].median()
        std_ratio = pair_group['ratio'].std()
        
        # Check for linear relationship
        pattern_results[pair_name] = {
            'count': len(pair_group),
            'mean_ratio': mean_ratio,
            'median_ratio': median_ratio,
            'std_ratio': std_ratio,
            'samples': pair_group.head(5).to_dict('records')
        }
        
        logger.info(f"{pair_name}: {len(pair_group)} transitions, mean ratio: {mean_ratio:.2f}, std: {std_ratio:.2f}")
    
    return pattern_results

def create_quick_visualization(transitions_df, output_dir='sample_results'):
    """Create a quick visualization of the transitions"""
    os.makedirs(output_dir, exist_ok=True)
    
    if transitions_df.empty:
        logger.warning("No transitions to visualize")
        return
    
    # Add bandwidth pair column if it doesn't exist
    if 'bandwidth_pair' not in transitions_df.columns:
        transitions_df['bandwidth_pair'] = transitions_df['bandwidth_from'] + '_to_' + transitions_df['bandwidth_to']
    
    # Plot histogram of ratios for each bandwidth pair
    bandwidth_pairs = transitions_df['bandwidth_pair'].unique()
    
    for pair in bandwidth_pairs:
        pair_data = transitions_df[transitions_df['bandwidth_pair'] == pair]
        
        # Skip if too few data points
        if len(pair_data) < 5:
            continue
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(pair_data['ratio'].dropna(), bins=30, alpha=0.7)
        
        # Add mean and median lines
        mean_ratio = pair_data['ratio'].mean()
        median_ratio = pair_data['ratio'].median()
        
        plt.axvline(mean_ratio, color='r', linestyle='--', label=f'Mean: {mean_ratio:.2f}')
        plt.axvline(median_ratio, color='g', linestyle='--', label=f'Median: {median_ratio:.2f}')
        
        # Check for specific ratio patterns
        for ratio in [1, 2, 4, 10]:
            plt.axvline(ratio, color='b', alpha=0.3, linestyle=':', label=f'Ratio {ratio}:1')
        
        plt.title(f'ID Ratio Distribution: {pair}')
        plt.xlabel('ID Ratio (id_to / id_from)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        plt.savefig(f'{output_dir}/sample_ratio_histogram_{pair}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create scatter plot of id_from vs id_to for one bandwidth pair
    if bandwidth_pairs.size > 0:
        pair = bandwidth_pairs[0]
        pair_data = transitions_df[transitions_df['bandwidth_pair'] == pair]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(pair_data['id_from'], pair_data['id_to'], alpha=0.7)
        
        plt.title(f'ID Relationship: {pair}')
        plt.xlabel(f'ID from {pair.split("_to_")[0]}')
        plt.ylabel(f'ID to {pair.split("_to_")[1]}')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(f'{output_dir}/sample_id_scatter_{pair}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OcuSync 4 ID Sample Analysis')
    parser.add_argument('--rf_file', type=str, default='Site1.csv', help='Path to RF data file')
    parser.add_argument('--dji_file', type=str, default='Site1_DJI_Data.csv', help='Path to DJI data file')
    parser.add_argument('--sample', type=int, default=50000, help='Number of rows to sample')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs('sample_results', exist_ok=True)
    
    # Load sample data
    rf_data, dji_data = load_sample_data(args.rf_file, args.dji_file, args.sample)
    
    # Find bandwidth transitions
    transitions = find_bandwidth_transitions(rf_data)
    
    # Analyze patterns
    if not transitions.empty:
        patterns = analyze_id_patterns(transitions)
        
        # Save results
        with open('sample_results/sample_patterns.json', 'w') as f:
            json.dump(patterns, f, indent=2)
        
        # Create quick visualization
        create_quick_visualization(transitions)
    
    # Correlate with DJI data
    correlations = correlate_with_dji(rf_data, dji_data)
    
    if not correlations.empty:
        # Save correlations
        correlations.to_csv('sample_results/sample_correlations.csv', index=False)
        
        # Group by serial number and bandwidth
        drone_bandwidth = correlations.groupby(['serial_number', 'bandwidth'])['id_number'].agg(['min', 'max', 'count']).reset_index()
        
        # Save drone-bandwidth summary
        drone_bandwidth.to_csv('sample_results/sample_drone_bandwidth.csv', index=False)
        
        # Create ID mappings for each drone
        drone_ids = defaultdict(dict)
        
        for serial in correlations['serial_number'].unique():
            drone_data = correlations[correlations['serial_number'] == serial]
            
            # Get unique bandwidth-id pairs
            for bw in drone_data['bandwidth'].unique():
                ids = drone_data[drone_data['bandwidth'] == bw]['id_number'].unique()
                if len(ids) > 0:
                    drone_ids[serial][bw] = ids.tolist()
        
        # Save drone ID mappings
        with open('sample_results/sample_drone_ids.json', 'w') as f:
            json.dump(drone_ids, f, indent=2)
    
    logger.info("Sample analysis complete. Results saved to 'sample_results' directory.")

if __name__ == "__main__":
    main()