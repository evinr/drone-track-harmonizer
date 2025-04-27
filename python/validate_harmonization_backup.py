#!/usr/bin/env python3
"""
Drone Track Harmonizer Validation Visualization Tool

This script provides visualization and validation of correlations between
DJI drone data and SDR detections to help verify the quality of harmonization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import folium
from folium.plugins import HeatMap, MarkerCluster
import math
import json
from pathlib import Path

# Configure output directories
RESULTS_DIR = Path("results")
EXPORTS_DIR = Path("../exports")
VALIDATION_DIR = Path("../validation")

# Ensure directories exist
VALIDATION_DIR.mkdir(exist_ok=True)

def load_correlations():
    """Load the correlation data from the results directory"""
    correlations_file = RESULTS_DIR / "dji_correlations.csv"
    
    if not correlations_file.exists():
        print(f"Error: Could not find {correlations_file}")
        return None
        
    print(f"Loading correlations from {correlations_file}...")
    df = pd.read_csv(correlations_file)
    print(f"Loaded {len(df)} correlation records")
    
    # Convert timestamps to datetime
    for col in ['detection_time', 'rf_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df

def load_pattern_stats():
    """Load the bandwidth pattern statistics"""
    pattern_file = RESULTS_DIR / "dji_pattern_stats.csv"
    
    if not pattern_file.exists():
        print(f"Warning: Could not find {pattern_file}")
        return None
        
    print(f"Loading pattern statistics from {pattern_file}...")
    df = pd.read_csv(pattern_file)
    print(f"Loaded {len(df)} pattern statistics")
    
    return df

def analyze_correlations(df):
    """Analyze the correlations to determine quality of data"""
    if df is None or df.empty:
        print("No correlations to analyze")
        return {}
    
    results = {}
    
    # Basic statistics
    results['total_correlations'] = len(df)
    results['unique_drones'] = df['serial_number'].nunique()
    results['unique_bandwidths'] = df['bandwidth'].nunique()
    results['unique_rf_ids'] = df['id_number'].nunique()
    
    # Time range
    results['time_range_start'] = df['detection_time'].min()
    results['time_range_end'] = df['detection_time'].max()
    
    # Distance statistics
    if 'distance' in df.columns:
        distances = df['distance'].dropna()
        results['avg_distance'] = distances.mean()
        results['min_distance'] = distances.min()
        results['max_distance'] = distances.max()
    
    # RSSI statistics
    if 'rssi' in df.columns:
        rssi = df['rssi'].dropna()
        results['avg_rssi'] = rssi.mean()
        results['min_rssi'] = rssi.min()
        results['max_rssi'] = rssi.max()
    
    # Correlation quality
    # (calculated as percentage of correlations within reasonable distance)
    if 'distance' in df.columns:
        good_distance = df[df['distance'] < 0.1].shape[0]  # Within ~100m
        results['correlation_quality'] = good_distance / len(df) if len(df) > 0 else 0
    
    # Bandwidth distribution
    if 'bandwidth' in df.columns:
        bandwidth_counts = df['bandwidth'].value_counts().to_dict()
        results['bandwidth_distribution'] = bandwidth_counts
    
    # Drone type distribution
    if 'drone_type' in df.columns:
        drone_counts = df['drone_type'].value_counts().to_dict()
        results['drone_type_distribution'] = drone_counts
    
    # Time difference statistics (between RF and DJI detections)
    if 'time_diff' in df.columns:
        time_diffs = df['time_diff'].dropna()
        results['avg_time_diff'] = time_diffs.mean()
        results['min_time_diff'] = time_diffs.min()
        results['max_time_diff'] = time_diffs.max()
    
    return results

def create_correlation_plots(df, analysis_results):
    """Create visualization plots to understand the correlation data"""
    if df is None or df.empty:
        print("No data for plots")
        return
    
    print("Creating validation plots...")
    
    # Plot 1: Correlation counts by drone type
    if 'drone_type' in df.columns:
        plt.figure(figsize=(15, 8))
        drone_counts = df['drone_type'].value_counts().sort_values(ascending=False)
        drone_counts.plot(kind='bar')
        plt.title('Correlation Count by Drone Type')
        plt.ylabel('Number of Correlations')
        plt.xlabel('Drone Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(VALIDATION_DIR / 'drone_type_correlations.png')
        plt.close()
    
    # Plot 2: RF detection RSSI distribution
    if 'rssi' in df.columns:
        plt.figure(figsize=(12, 6))
        df['rssi'].hist(bins=50)
        plt.title('RF Signal Strength (RSSI) Distribution')
        plt.xlabel('RSSI (dB)')
        plt.ylabel('Count')
        plt.savefig(VALIDATION_DIR / 'rssi_distribution.png')
        plt.close()
    
    # Plot 3: Distance between RF detection and DJI position
    if 'distance' in df.columns:
        plt.figure(figsize=(12, 6))
        # Convert distance to meters for better visualization (assuming distance is in degrees)
        df['distance_meters'] = df['distance'] * 111000  # Rough conversion
        df['distance_meters'].hist(bins=50, range=(0, 1000))
        plt.title('Distance Between RF Detection and DJI Position')
        plt.xlabel('Distance (meters)')
        plt.ylabel('Count')
        plt.savefig(VALIDATION_DIR / 'distance_distribution.png')
        plt.close()
    
    # Plot 4: Time difference between RF and DJI detection
    if 'time_diff' in df.columns:
        plt.figure(figsize=(12, 6))
        df['time_diff'].hist(bins=50, range=(-10, 10))
        plt.title('Time Difference Between RF and DJI Detection')
        plt.xlabel('Time Difference (seconds)')
        plt.ylabel('Count')
        plt.savefig(VALIDATION_DIR / 'time_diff_distribution.png')
        plt.close()
    
    # Plot 5: Correlation scatter plot by distance and RSSI
    if 'distance' in df.columns and 'rssi' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.scatter(df['distance_meters'], df['rssi'], alpha=0.3)
        plt.title('RF Signal Strength vs Distance')
        plt.xlabel('Distance (meters)')
        plt.ylabel('RSSI (dB)')
        plt.savefig(VALIDATION_DIR / 'rssi_vs_distance.png')
        plt.close()
    
    # Plot 6: Correlation counts over time
    plt.figure(figsize=(15, 6))
    df['detection_date'] = df['detection_time'].dt.date
    daily_counts = df.groupby('detection_date').size()
    daily_counts.plot(kind='line')
    plt.title('Correlations by Date')
    plt.ylabel('Number of Correlations')
    plt.xlabel('Date')
    plt.grid(True)
    plt.savefig(VALIDATION_DIR / 'correlations_by_date.png')
    plt.close()
    
    # Plot 7: Bandwidth distribution
    if 'bandwidth' in df.columns:
        plt.figure(figsize=(12, 6))
        df['bandwidth'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Distribution of Bandwidth in Correlations')
        plt.ylabel('')
        plt.savefig(VALIDATION_DIR / 'bandwidth_distribution.png')
        plt.close()

def create_map_visualization(df):
    """Create an interactive map showing correlations between DJI and RF data"""
    if df is None or df.empty:
        print("No data for map visualization")
        return
    
    # Check if we have the necessary columns
    required_cols = ['serial_number', 'detection_time', 'latitude', 'longitude', 
                     'bandwidth', 'id_number', 'rssi']
    
    # Adapt to either DJI coordinate naming or RF coordinate naming
    if 'DroneLatitude' in df.columns and 'DroneLongitude' in df.columns:
        df['latitude'] = df['DroneLatitude']
        df['longitude'] = df['DroneLongitude']
    
    if not all(col in df.columns for col in required_cols):
        print("Missing required columns for map visualization")
        return
    
    print("Creating interactive map visualization...")
    
    # Create base map centered on average coordinates
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Create marker clusters for DJI positions
    dji_cluster = MarkerCluster(name="DJI Drone Positions")
    
    # Sample data to avoid overwhelming the map (max 1000 points)
    if len(df) > 1000:
        sample_size = 1000
        print(f"Sampling {sample_size} points for map visualization")
        sample_df = df.sample(sample_size, random_state=42)
    else:
        sample_df = df
    
    # Add markers for each correlation
    for idx, row in sample_df.iterrows():
        # Format popup information
        popup_html = f"""
        <b>Drone:</b> {row['serial_number']}<br>
        <b>Time:</b> {row['detection_time']}<br>
        <b>RF ID:</b> {row['id_number']} ({row['bandwidth']})<br>
        <b>RSSI:</b> {row['rssi']} dB<br>
        """
        
        # Add marker to cluster
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(dji_cluster)
    
    # Add heatmap for RF signal strength
    if 'rssi' in df.columns:
        # Normalize RSSI values (make negative values positive for heatmap)
        df['rssi_norm'] = df['rssi'].apply(lambda x: abs(x) if x < 0 else x)
        
        # Create heatmap data
        heat_data = [[row['latitude'], row['longitude'], row['rssi_norm']] 
                    for idx, row in sample_df.iterrows()]
        
        # Add heatmap layer
        HeatMap(heat_data, name="RF Signal Strength", radius=15).add_to(m)
    
    # Add marker cluster to map
    dji_cluster.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    map_file = VALIDATION_DIR / 'correlation_map.html'
    m.save(str(map_file))
    print(f"Interactive map saved to {map_file}")

def create_harmonized_gpx(df):
    """Create a harmonized GPX file that includes both DJI and SDR data"""
    if df is None or df.empty:
        print("No data for harmonized GPX")
        return
    
    print("Creating harmonized GPX file...")
    
    # Check if we have coordinate columns
    if 'DroneLatitude' in df.columns and 'DroneLongitude' in df.columns:
        df['latitude'] = df['DroneLatitude']
        df['longitude'] = df['DroneLongitude']
    
    # If we still don't have coordinates, create a basic GPX file
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("Warning: No coordinate data found. Creating a basic summary GPX file.")
        
        # Create a basic GPX file with metadata
        gpx_file = VALIDATION_DIR / 'harmonized_drones_summary.gpx'
        
        with open(gpx_file, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<gpx version="1.1" creator="Drone Track Harmonizer - Validation">\n')
            f.write('  <metadata>\n')
            f.write('    <name>Drone Track Summary</name>\n')
            f.write('    <desc>Summary of drone correlations without coordinate data</desc>\n')
            f.write(f'    <time>{datetime.now().isoformat()}</time>\n')
            f.write('  </metadata>\n')
            
            # Add waypoints with drone serial numbers
            for serial in df['serial_number'].unique():
                group = df[df['serial_number'] == serial]
                drone_type = group['drone_type'].iloc[0] if 'drone_type' in group.columns else 'Unknown'
                
                f.write('  <wpt lat="0.0" lon="0.0">\n')
                f.write(f'    <name>{drone_type} - {serial}</name>\n')
                f.write(f'    <desc>RF ID: {group["id_number"].iloc[0]} ({group["bandwidth"].iloc[0]})</desc>\n')
                f.write('  </wpt>\n')
            
            f.write('</gpx>\n')
        
        print(f"Basic summary GPX file created at {gpx_file}")
        return
    
    # Group by drone serial number
    gpx_file = VALIDATION_DIR / 'harmonized_drones.gpx'
    
    # Create GPX content
    gpx_header = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Drone Track Harmonizer - Validation">
"""
    gpx_footer = """</gpx>"""
    
    with open(gpx_file, 'w') as f:
        f.write(gpx_header)
        
        # Group by drone
        drone_groups = df.groupby('serial_number')
        
        # Add tracks for each drone
        for serial, group in drone_groups:
            # Sort by time
            group = group.sort_values('detection_time')
            
            # Only create tracks with at least 2 points
            if len(group) < 2:
                continue
            
            # Get drone type
            drone_type = group['drone_type'].iloc[0] if 'drone_type' in group.columns else 'Unknown'
            
            # Create track
            f.write(f'  <trk>\n')
            f.write(f'    <name>{drone_type} - {serial}</name>\n')
            f.write(f'    <desc>RF ID: {group["id_number"].iloc[0]} ({group["bandwidth"].iloc[0]})</desc>\n')
            f.write(f'    <trkseg>\n')
            
            # Add points
            for _, row in group.iterrows():
                f.write(f'      <trkpt lat="{row["latitude"]}" lon="{row["longitude"]}">\n')
                f.write(f'        <time>{row["detection_time"].isoformat()}</time>\n')
                f.write(f'        <extensions>\n')
                f.write(f'          <rssi>{row["rssi"]}</rssi>\n')
                f.write(f'          <rf_id>{row["id_number"]}</rf_id>\n')
                f.write(f'          <bandwidth>{row["bandwidth"]}</bandwidth>\n')
                f.write(f'        </extensions>\n')
                f.write(f'      </trkpt>\n')
            
            f.write(f'    </trkseg>\n')
            f.write(f'  </trk>\n')
        
        f.write(gpx_footer)
    
    print(f"Harmonized GPX file created at {gpx_file}")

def create_harmonized_kml(df):
    """Create a harmonized KML file that includes both DJI and SDR data"""
    if df is None or df.empty:
        print("No data for harmonized KML")
        return
    
    print("Creating harmonized KML file...")
    
    # Prepare coordinates
    if 'DroneLatitude' in df.columns and 'DroneLongitude' in df.columns:
        df['latitude'] = df['DroneLatitude']
        df['longitude'] = df['DroneLongitude']
    
    # Group by drone serial number
    kml_file = VALIDATION_DIR / 'harmonized_drones.kml'
    
    # Create KML content
    kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
    <Document>
        <Style id="djiStyle">
            <LineStyle>
                <color>ff0000ff</color>
                <width>4</width>
            </LineStyle>
            <IconStyle>
                <Icon>
                    <href>http://maps.google.com/mapfiles/kml/paddle/blu-circle.png</href>
                </Icon>
            </IconStyle>
        </Style>
        <Style id="sdrStyle">
            <LineStyle>
                <color>ff0000ff</color>
                <width>4</width>
            </LineStyle>
            <IconStyle>
                <Icon>
                    <href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href>
                </Icon>
            </IconStyle>
        </Style>
        <Folder>
            <name>Harmonized Drone Tracks</name>
"""
    kml_footer = """
        </Folder>
    </Document>
</kml>"""
    
    with open(kml_file, 'w') as f:
        f.write(kml_header)
        
        # Group by drone
        drone_groups = df.groupby('serial_number')
        
        # Add placemarks for each drone
        for serial, group in drone_groups:
            # Sort by time
            group = group.sort_values('detection_time')
            
            # Only create tracks with at least 2 points
            if len(group) < 2:
                continue
            
            # Get drone type
            drone_type = group['drone_type'].iloc[0] if 'drone_type' in group.columns else 'Unknown'
            
            # Create placemark
            f.write(f'            <Placemark>\n')
            f.write(f'                <name>{drone_type} - {serial}</name>\n')
            f.write(f'                <description>RF ID: {group["id_number"].iloc[0]} ({group["bandwidth"].iloc[0]})</description>\n')
            f.write(f'                <styleUrl>#djiStyle</styleUrl>\n')
            f.write(f'                <LineString>\n')
            f.write(f'                    <coordinates>\n')
            
            # Add coordinates
            for _, row in group.iterrows():
                f.write(f'                        {row["longitude"]},{row["latitude"]},0\n')
            
            f.write(f'                    </coordinates>\n')
            f.write(f'                </LineString>\n')
            f.write(f'            </Placemark>\n')
        
        f.write(kml_footer)
    
    print(f"Harmonized KML file created at {kml_file}")

def create_validation_summary(analysis_results):
    """Create a summary report of the validation findings"""
    if not analysis_results:
        print("No analysis results for summary")
        return
    
    print("Creating validation summary...")
    
    summary_file = VALIDATION_DIR / 'validation_summary.md'
    
    with open(summary_file, 'w') as f:
        f.write("# Drone Track Harmonization Validation Summary\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Data Overview\n\n")
        f.write(f"- Total correlations: {analysis_results.get('total_correlations', 'N/A')}\n")
        f.write(f"- Unique drones: {analysis_results.get('unique_drones', 'N/A')}\n")
        f.write(f"- Unique RF IDs: {analysis_results.get('unique_rf_ids', 'N/A')}\n")
        f.write(f"- Unique bandwidths: {analysis_results.get('unique_bandwidths', 'N/A')}\n")
        f.write(f"- Time range: {analysis_results.get('time_range_start', 'N/A')} to {analysis_results.get('time_range_end', 'N/A')}\n\n")
        
        f.write("## Correlation Quality\n\n")
        if 'correlation_quality' in analysis_results:
            quality_pct = analysis_results['correlation_quality'] * 100
            f.write(f"- Correlation quality score: {quality_pct:.2f}%\n")
            
            # Add interpretation
            if quality_pct > 80:
                f.write("- Interpretation: **Excellent** - Very strong correlation between DJI and RF data\n")
            elif quality_pct > 60:
                f.write("- Interpretation: **Good** - Strong correlation, suitable for most applications\n")
            elif quality_pct > 40:
                f.write("- Interpretation: **Moderate** - Acceptable correlation, may need filtering\n")
            elif quality_pct > 20:
                f.write("- Interpretation: **Fair** - Weak correlation, use with caution\n")
            else:
                f.write("- Interpretation: **Poor** - Very weak correlation, not recommended for use\n")
        
        f.write("\n## Distance Statistics\n\n")
        if all(k in analysis_results for k in ['avg_distance', 'min_distance', 'max_distance']):
            # Convert distances to meters
            avg_dist_m = analysis_results['avg_distance'] * 111000
            min_dist_m = analysis_results['min_distance'] * 111000
            max_dist_m = analysis_results['max_distance'] * 111000
            
            f.write(f"- Average distance: {avg_dist_m:.2f} meters\n")
            f.write(f"- Minimum distance: {min_dist_m:.2f} meters\n")
            f.write(f"- Maximum distance: {max_dist_m:.2f} meters\n\n")
        
        f.write("## RSSI Statistics\n\n")
        if all(k in analysis_results for k in ['avg_rssi', 'min_rssi', 'max_rssi']):
            f.write(f"- Average RSSI: {analysis_results['avg_rssi']:.2f} dB\n")
            f.write(f"- Minimum RSSI: {analysis_results['min_rssi']:.2f} dB\n")
            f.write(f"- Maximum RSSI: {analysis_results['max_rssi']:.2f} dB\n\n")
        
        f.write("## Time Difference Statistics\n\n")
        if all(k in analysis_results for k in ['avg_time_diff', 'min_time_diff', 'max_time_diff']):
            f.write(f"- Average time difference: {analysis_results['avg_time_diff']:.2f} seconds\n")
            f.write(f"- Minimum time difference: {analysis_results['min_time_diff']:.2f} seconds\n")
            f.write(f"- Maximum time difference: {analysis_results['max_time_diff']:.2f} seconds\n\n")
        
        f.write("## Bandwidth Distribution\n\n")
        if 'bandwidth_distribution' in analysis_results:
            f.write("| Bandwidth | Count | Percentage |\n")
            f.write("|-----------|-------|------------|\n")
            
            total = analysis_results['total_correlations']
            for bw, count in analysis_results['bandwidth_distribution'].items():
                pct = (count / total) * 100 if total > 0 else 0
                f.write(f"| {bw} | {count} | {pct:.2f}% |\n")
            
            f.write("\n")
        
        f.write("## Validation Files\n\n")
        f.write("The following validation files have been generated:\n\n")
        f.write("1. **Interactive Map**: `correlation_map.html`\n")
        f.write("2. **Harmonized GPX**: `harmonized_drones.gpx`\n")
        f.write("3. **Harmonized KML**: `harmonized_drones.kml`\n")
        f.write("4. **Visualization Plots**:\n")
        f.write("   - Drone type correlations: `drone_type_correlations.png`\n")
        f.write("   - RSSI distribution: `rssi_distribution.png`\n")
        f.write("   - Distance distribution: `distance_distribution.png`\n")
        f.write("   - Time difference distribution: `time_diff_distribution.png`\n")
        f.write("   - RSSI vs Distance: `rssi_vs_distance.png`\n")
        f.write("   - Correlations by date: `correlations_by_date.png`\n")
        f.write("   - Bandwidth distribution: `bandwidth_distribution.png`\n\n")
        
        f.write("## Next Steps\n\n")
        
        # Provide recommendations based on quality
        if 'correlation_quality' in analysis_results:
            quality_pct = analysis_results['correlation_quality'] * 100
            
            if quality_pct > 60:
                f.write("Based on the validation results, the data appears to be of good quality and ready for use in the ATAK plugin. You can proceed with confidence.\n\n")
                f.write("Recommended actions:\n")
                f.write("1. Use the harmonized GPX/KML files for ATAK integration\n")
                f.write("2. Implement the ID correlation formulas from the pattern stats\n")
            elif quality_pct > 30:
                f.write("Based on the validation results, the data shows moderate correlation. Some filtering may be needed for optimal results.\n\n")
                f.write("Recommended actions:\n")
                f.write("1. Filter out correlations with distances > 100m\n")
                f.write("2. Focus on the strongest signal bands (see bandwidth distribution)\n")
                f.write("3. Test with a subset of the data first\n")
            else:
                f.write("Based on the validation results, the correlation quality is low. Further data collection or processing may be needed.\n\n")
                f.write("Recommended actions:\n")
                f.write("1. Review the data collection methodology\n")
                f.write("2. Apply strict filtering to use only the highest quality matches\n")
                f.write("3. Consider collecting additional data\n")
    
    print(f"Validation summary created at {summary_file}")

def create_json_metadata(analysis_results):
    """Create JSON metadata file with analysis results"""
    if not analysis_results:
        return
    
    # Convert datetime objects to strings
    results_copy = dict(analysis_results)
    for key, value in results_copy.items():
        if isinstance(value, (datetime, pd.Timestamp)):
            results_copy[key] = value.isoformat()
    
    json_file = VALIDATION_DIR / 'validation_metadata.json'
    with open(json_file, 'w') as f:
        json.dump(results_copy, f, indent=2)
    
    print(f"Validation metadata saved to {json_file}")

def main():
    """Main function to run the validation and visualization"""
    print("\n===== Drone Track Harmonizer Validation =====\n")
    
    # Load data
    correlations_df = load_correlations()
    pattern_stats_df = load_pattern_stats()
    
    if correlations_df is None:
        print("Error: Could not load correlation data. Exiting.")
        return
    
    # Analyze correlations
    print("\nAnalyzing correlation data...")
    analysis_results = analyze_correlations(correlations_df)
    
    # Create visualizations
    create_correlation_plots(correlations_df, analysis_results)
    create_map_visualization(correlations_df)
    
    # Create harmonized outputs
    create_harmonized_gpx(correlations_df)
    create_harmonized_kml(correlations_df)
    
    # Create summary reports
    create_validation_summary(analysis_results)
    create_json_metadata(analysis_results)
    
    print("\n===== Validation Complete =====")
    print(f"\nValidation files can be found in: {VALIDATION_DIR}")

if __name__ == "__main__":
    main()