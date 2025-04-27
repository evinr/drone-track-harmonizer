#!/usr/bin/env python3
"""
OcuSync 4 ID Harmonization Analysis
-----------------------------------
This script analyzes OcuSync 4 IDs across different bandwidths to test the 
linear mapping hypothesis and develop a harmonization method.
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import datetime
from tqdm import tqdm
import os
from scipy.spatial.distance import cdist
import logging

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

def load_rf_data(file_path, sample=None):
    """Load RF data, optionally using a sample for faster processing"""
    logger.info(f"Loading RF data from {file_path}")
    
    if sample:
        # Use a smaller sample during development
        df = pd.read_csv(file_path, nrows=sample)
    else:
        # Read the whole file
        df = pd.read_csv(file_path)
    
    # Parse timestamps - convert all to timezone naive
    df['time'] = pd.to_datetime(df['time'], format='ISO8601').dt.tz_localize(None)
    
    # Extract bandwidth and ID from rf_uuid
    df['bandwidth'] = None
    df['id_number'] = None
    
    logger.info("Extracting bandwidth and ID numbers...")
    ocusync_entries = df['rf_uuid'].str.contains('V4_', case=False, na=False)
    
    # Apply only to OcuSync entries to avoid processing non-matching entries
    if ocusync_entries.any():
        ocusync_df = df[ocusync_entries].copy()
        
        # Extract bandwidth and ID number
        results = ocusync_df['rf_uuid'].apply(extract_ocusync_info)
        ocusync_df['bandwidth'] = results.apply(lambda x: x[0])
        ocusync_df['id_number'] = results.apply(lambda x: x[1])
        
        # Update original dataframe
        df.loc[ocusync_entries, 'bandwidth'] = ocusync_df['bandwidth']
        df.loc[ocusync_entries, 'id_number'] = ocusync_df['id_number']
    
    logger.info(f"Found {ocusync_entries.sum()} OcuSync 4 entries")
    return df

def load_dji_data(file_path):
    """Load DJI positional data"""
    logger.info(f"Loading DJI data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Parse timestamps - convert all to timezone naive
    df['DetectionTime'] = pd.to_datetime(df['DetectionTime']).dt.tz_localize(None)
    
    logger.info(f"Loaded {len(df)} DJI entries")
    return df

def analyze_bandwidth_transitions(rf_data):
    """
    Analyze transitions between bandwidths for the same track_id
    to find patterns between ID numbers
    """
    logger.info("Analyzing bandwidth transitions...")
    
    # Group by track_id and sort by time
    track_groups = rf_data.groupby('track_id')
    
    # Store ID relationships
    id_relationships = []
    
    # Time window for considering transitions (in seconds)
    time_window = 1
    
    # Process each track
    for track_id, group in tqdm(track_groups):
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
                    id_relationships.append({
                        'track_id': track_id,
                        'time': current_time,
                        'bandwidth_from': prev_bandwidth,
                        'id_from': prev_id,
                        'bandwidth_to': current_bandwidth,
                        'id_to': current_id,
                        'time_diff': time_diff
                    })
            
            prev_time = current_time
            prev_bandwidth = current_bandwidth
            prev_id = current_id
    
    # Convert to DataFrame
    transitions_df = pd.DataFrame(id_relationships)
    
    logger.info(f"Found {len(transitions_df)} bandwidth transitions")
    return transitions_df

def analyze_id_patterns(transitions_df):
    """Analyze patterns between ID numbers across different bandwidths"""
    logger.info("Analyzing ID patterns across bandwidths...")
    
    # Create bandwidth pairs
    transitions_df['bandwidth_pair'] = transitions_df['bandwidth_from'] + '_to_' + transitions_df['bandwidth_to']
    
    # Group by bandwidth pair
    bandwidth_pairs = transitions_df.groupby('bandwidth_pair')
    
    # Analyze each bandwidth pair
    pattern_results = {}
    
    for pair_name, pair_group in bandwidth_pairs:
        # Calculate ratios between IDs
        pair_group = pair_group.copy()
        pair_group['id_ratio'] = pair_group['id_to'] / pair_group['id_from']
        
        # Calculate statistics
        mean_ratio = pair_group['id_ratio'].mean()
        median_ratio = pair_group['id_ratio'].median()
        std_ratio = pair_group['id_ratio'].std()
        
        # Check for linear relationship (id_to = factor * id_from)
        pattern_results[pair_name] = {
            'count': len(pair_group),
            'mean_ratio': mean_ratio,
            'median_ratio': median_ratio,
            'std_ratio': std_ratio,
            'samples': pair_group.head(5).to_dict('records')
        }
        
        logger.info(f"{pair_name}: {len(pair_group)} transitions, mean ratio: {mean_ratio:.2f}, std: {std_ratio:.2f}")
    
    return pattern_results

def spatial_temporal_clustering(rf_data, time_window=0.5, distance_threshold=0.01):
    """
    Cluster RF detections by spatial and temporal proximity to identify
    potential unique drones across different bandwidth IDs
    """
    logger.info("Performing spatial-temporal clustering...")
    
    # Focus on OcuSync entries
    ocusync_data = rf_data[rf_data['bandwidth'].notna()].copy()
    
    # Convert time to numeric for clustering
    ocusync_data['time_numeric'] = (ocusync_data['time'] - ocusync_data['time'].min()).dt.total_seconds()
    
    # Prepare for clustering
    clusters = []
    processed = set()
    
    # Process each unique detection time (within window)
    time_groups = ocusync_data.groupby(pd.cut(ocusync_data['time_numeric'], bins=int(ocusync_data['time_numeric'].max() / time_window)))
    
    for _, time_group in tqdm(time_groups):
        if len(time_group) <= 1:
            continue
            
        # For each bandwidth within this time window
        bandwidth_groups = time_group.groupby('bandwidth')
        
        bandwidth_representatives = {}
        for bandwidth, bw_group in bandwidth_groups:
            # For each ID within this bandwidth
            id_groups = bw_group.groupby('id_number')
            
            for id_num, id_group in id_groups:
                key = f"{bandwidth}_{id_num}"
                if key in processed:
                    continue
                    
                processed.add(key)
                
                # Use average position as representative
                avg_lat = id_group['sensor_latitude'].mean()
                avg_lon = id_group['sensor_longitude'].mean()
                avg_rssi = id_group['rf_rssi'].mean()
                avg_time = id_group['time_numeric'].mean()
                
                bandwidth_representatives[key] = {
                    'bandwidth': bandwidth,
                    'id_number': id_num,
                    'latitude': avg_lat,
                    'longitude': avg_lon,
                    'rssi': avg_rssi,
                    'time': avg_time,
                    'count': len(id_group)
                }
        
        # Skip if too few representatives
        if len(bandwidth_representatives) <= 1:
            continue
            
        # Extract representatives
        reps = list(bandwidth_representatives.values())
        
        # Create a new cluster
        current_cluster = []
        
        # Check spatial proximity
        for i, rep1 in enumerate(reps):
            for j, rep2 in enumerate(reps[i+1:], i+1):
                # Skip same bandwidth
                if rep1['bandwidth'] == rep2['bandwidth']:
                    continue
                    
                # Calculate distance
                dist = np.sqrt((rep1['latitude'] - rep2['latitude'])**2 + 
                             (rep1['longitude'] - rep2['longitude'])**2)
                
                if dist <= distance_threshold:
                    # Add to cluster if close enough
                    if rep1 not in current_cluster:
                        current_cluster.append(rep1)
                    if rep2 not in current_cluster:
                        current_cluster.append(rep2)
        
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
    
    logger.info(f"Found {len(clusters)} potential drone clusters")
    return clusters

def analyze_clusters(clusters):
    """Analyze ID relationships within spatial-temporal clusters"""
    logger.info("Analyzing ID relationships within clusters...")
    
    # Extract bandwidth-ID relationships
    relationships = []
    
    for i, cluster in enumerate(clusters):
        # Skip single-element clusters
        if len(cluster) <= 1:
            continue
            
        # Extract all bandwidth-ID pairs
        bandwidths = {}
        for item in cluster:
            bandwidths[item['bandwidth']] = item['id_number']
        
        # Need at least two different bandwidths
        if len(bandwidths) <= 1:
            continue
            
        # Create relationships
        for bw1, id1 in bandwidths.items():
            for bw2, id2 in bandwidths.items():
                if bw1 != bw2:
                    relationships.append({
                        'cluster_id': i,
                        'bandwidth_from': bw1,
                        'id_from': id1,
                        'bandwidth_to': bw2,
                        'id_to': id2,
                        'ratio': id2 / id1 if id1 != 0 else None
                    })
    
    # Convert to DataFrame
    rel_df = pd.DataFrame(relationships)
    
    # Analyze ratios by bandwidth pair
    if not rel_df.empty:
        rel_df['bandwidth_pair'] = rel_df['bandwidth_from'] + '_to_' + rel_df['bandwidth_to']
        
        # Group by bandwidth pair
        pair_stats = rel_df.groupby('bandwidth_pair').agg({
            'ratio': ['count', 'mean', 'median', 'std']
        }).reset_index()
        
        pair_stats.columns = ['bandwidth_pair', 'count', 'mean_ratio', 'median_ratio', 'std_ratio']
        
        # Display results
        for _, row in pair_stats.iterrows():
            logger.info(f"{row['bandwidth_pair']}: {row['count']} pairs, mean ratio: {row['mean_ratio']:.2f}, std: {row['std_ratio']:.2f}")
        
        return rel_df, pair_stats
    
    return None, None

def correlate_with_dji_data(rf_data, dji_data, time_window=10):
    """
    Correlate RF detections with DJI positional data to verify
    ID harmonization
    """
    logger.info("Correlating RF detections with DJI data...")
    
    # Focus on OcuSync entries
    ocusync_data = rf_data[rf_data['bandwidth'].notna()].copy()
    
    # Group DJI data by drone (serial number)
    dji_groups = dji_data.groupby('SerialNumber')
    
    # Store correlations
    correlations = []
    
    # Process each DJI drone
    for serial, dji_group in tqdm(dji_groups):
        # Process each detection time
        for _, dji_row in dji_group.iterrows():
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
                        'serial_number': serial,
                        'drone_type': dji_row['DroneType'],
                        'detection_time': detection_time,
                        'rf_time': rf_row['time'],
                        'time_diff': (rf_row['time'] - detection_time).total_seconds(),
                        'bandwidth': rf_row['bandwidth'],
                        'id_number': rf_row['id_number'],
                        'rssi': rf_row['rf_rssi'],
                        'distance': dist
                    })
    
    # Convert to DataFrame
    corr_df = pd.DataFrame(correlations)
    
    logger.info(f"Found {len(corr_df)} RF-DJI correlations")
    return corr_df

def analyze_dji_correlations(correlations):
    """Analyze correlations between DJI drones and RF IDs"""
    if correlations.empty:
        logger.warning("No correlations found")
        return None
        
    logger.info("Analyzing DJI-RF correlations...")
    
    # Group by serial number
    serial_groups = correlations.groupby('serial_number')
    
    # Analyze each drone
    drone_results = []
    
    for serial, group in serial_groups:
        # Get unique bandwidths for this drone
        bandwidths = group['bandwidth'].unique()
        
        # Skip if only one bandwidth
        if len(bandwidths) <= 1:
            continue
            
        # Get all ID combinations
        bandwidth_ids = {}
        for bw in bandwidths:
            ids = group[group['bandwidth'] == bw]['id_number'].unique()
            if len(ids) > 0:
                bandwidth_ids[bw] = ids
        
        # Create ID relationships
        for bw1, ids1 in bandwidth_ids.items():
            for bw2, ids2 in bandwidth_ids.items():
                if bw1 != bw2:
                    for id1 in ids1:
                        for id2 in ids2:
                            drone_results.append({
                                'serial_number': serial,
                                'drone_type': group['drone_type'].iloc[0],
                                'bandwidth_from': bw1,
                                'id_from': id1,
                                'bandwidth_to': bw2,
                                'id_to': id2,
                                'ratio': id2 / id1 if id1 != 0 else None
                            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(drone_results)
    
    # Analyze by bandwidth pair
    if not results_df.empty:
        results_df['bandwidth_pair'] = results_df['bandwidth_from'] + '_to_' + results_df['bandwidth_to']
        
        pair_stats = results_df.groupby(['bandwidth_pair']).agg({
            'ratio': ['count', 'mean', 'median', 'std']
        }).reset_index()
        
        pair_stats.columns = ['bandwidth_pair', 'count', 'mean_ratio', 'median_ratio', 'std_ratio']
        
        # Display results
        for _, row in pair_stats.iterrows():
            logger.info(f"{row['bandwidth_pair']}: {row['count']} pairs, mean ratio: {row['mean_ratio']:.2f}, std: {row['std_ratio']:.2f}")
        
        # Also group by drone type
        drone_stats = results_df.groupby(['drone_type', 'bandwidth_pair']).agg({
            'ratio': ['count', 'mean', 'median', 'std']
        }).reset_index()
        
        return results_df, pair_stats, drone_stats
    
    return None, None, None

def main(rf_file, dji_file, sample=None):
    """Main analysis function"""
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Load data
    rf_data = load_rf_data(rf_file, sample)
    dji_data = load_dji_data(dji_file)
    
    # Save summary of loaded data
    with open('results/data_summary.txt', 'w') as f:
        f.write(f"RF Data Summary:\n")
        f.write(f"Total rows: {len(rf_data)}\n")
        f.write(f"OcuSync entries: {rf_data['bandwidth'].notna().sum()}\n")
        f.write(f"Unique bandwidths: {rf_data['bandwidth'].nunique()}\n")
        f.write(f"Time range: {rf_data['time'].min()} to {rf_data['time'].max()}\n\n")
        
        f.write(f"DJI Data Summary:\n")
        f.write(f"Total rows: {len(dji_data)}\n")
        f.write(f"Unique drones: {dji_data['SerialNumber'].nunique()}\n")
        f.write(f"Drone types: {', '.join(dji_data['DroneType'].unique())}\n")
        f.write(f"Time range: {dji_data['DetectionTime'].min()} to {dji_data['DetectionTime'].max()}\n")
    
    # Analyze bandwidth transitions
    transitions = analyze_bandwidth_transitions(rf_data)
    if not transitions.empty:
        transitions.to_csv('results/bandwidth_transitions.csv', index=False)
        
        # Analyze patterns in transitions
        patterns = analyze_id_patterns(transitions)
        
        # Save results
        pattern_df = []
        for pair, stats in patterns.items():
            pattern_df.append({
                'bandwidth_pair': pair,
                'count': stats['count'],
                'mean_ratio': stats['mean_ratio'],
                'median_ratio': stats['median_ratio'],
                'std_ratio': stats['std_ratio']
            })
        
        pattern_df = pd.DataFrame(pattern_df)
        pattern_df.to_csv('results/bandwidth_patterns.csv', index=False)
    
    # Spatial-temporal clustering
    clusters = spatial_temporal_clustering(rf_data)
    
    # Save cluster data
    cluster_items = []
    for i, cluster in enumerate(clusters):
        for item in cluster:
            cluster_items.append({
                'cluster_id': i,
                **item
            })
    
    if cluster_items:
        cluster_df = pd.DataFrame(cluster_items)
        cluster_df.to_csv('results/spatial_clusters.csv', index=False)
        
        # Analyze clusters
        cluster_relationships, pair_stats = analyze_clusters(clusters)
        
        if cluster_relationships is not None:
            cluster_relationships.to_csv('results/cluster_relationships.csv', index=False)
            pair_stats.to_csv('results/cluster_pattern_stats.csv', index=False)
    
    # Correlate with DJI data
    correlations = correlate_with_dji_data(rf_data, dji_data)
    
    if not correlations.empty:
        correlations.to_csv('results/dji_correlations.csv', index=False)
        
        # Analyze correlations
        dji_relationships, dji_pair_stats, drone_stats = analyze_dji_correlations(correlations)
        
        if dji_relationships is not None:
            dji_relationships.to_csv('results/dji_id_relationships.csv', index=False)
            dji_pair_stats.to_csv('results/dji_pattern_stats.csv', index=False)
            drone_stats.to_csv('results/drone_type_patterns.csv', index=False)
    
    logger.info("Analysis complete. Results saved to 'results' directory.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OcuSync 4 ID Harmonization Analysis')
    parser.add_argument('--rf_file', type=str, default='Site1.csv', help='Path to RF data file')
    parser.add_argument('--dji_file', type=str, default='Site1_DJI_Data.csv', help='Path to DJI data file')
    parser.add_argument('--sample', type=int, help='Number of rows to sample (for testing)')
    
    args = parser.parse_args()
    
    main(args.rf_file, args.dji_file, args.sample)