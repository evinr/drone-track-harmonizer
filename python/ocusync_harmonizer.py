#!/usr/bin/env python3
"""
OcuSync 4 ID Harmonizer
----------------------
This script implements an algorithm to harmonize OcuSync 4 IDs across different
bandwidths based on the identified patterns.
"""

import pandas as pd
import numpy as np
import re
import json
import os
import logging
from collections import defaultdict
import gpxpy
import gpxpy.gpx
import simplekml
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Regular expression to extract OcuSync 4 IDs
OCUSYNC_PATTERN = r'V4_(\d+MHz)_(\d+)'

# Bandwidth conversion factors (will be learned from data)
BANDWIDTH_FACTORS = {
    '10MHz_to_20MHz': None,
    '10MHz_to_40MHz': None,
    '20MHz_to_40MHz': None,
    '20MHz_to_10MHz': None,
    '40MHz_to_10MHz': None,
    '40MHz_to_20MHz': None
}

class OcuSyncHarmonizer:
    """
    A class to harmonize OcuSync 4 IDs across different bandwidths
    """
    def __init__(self, factor_map=None, tolerance=0.1):
        """
        Initialize the harmonizer
        
        Parameters:
        - factor_map: Dictionary of bandwidth conversion factors
        - tolerance: Tolerance for ratio matching (0.1 = 10%)
        """
        self.factors = factor_map or BANDWIDTH_FACTORS.copy()
        self.tolerance = tolerance
        
        # Dictionary to store unique drone identifiers
        self.drone_map = {}
        self.next_drone_id = 1
    
    def load_factors_from_file(self, file_path):
        """Load bandwidth conversion factors from a file"""
        try:
            with open(file_path, 'r') as f:
                self.factors = json.load(f)
            logger.info(f"Loaded conversion factors from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading factors from {file_path}: {e}")
            return False
    
    def learn_factors_from_data(self, data_file):
        """
        Learn bandwidth conversion factors from data
        
        Parameters:
        - data_file: Path to CSV file with ID relationships
        """
        try:
            # Load data
            df = pd.read_csv(data_file)
            
            # Ensure required columns exist
            required_cols = ['bandwidth_from', 'id_from', 'bandwidth_to', 'id_to', 'ratio']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Required column '{col}' not found in {data_file}")
                    return False
            
            # Add bandwidth pair column if it doesn't exist
            if 'bandwidth_pair' not in df.columns:
                df['bandwidth_pair'] = df['bandwidth_from'] + '_to_' + df['bandwidth_to']
            
            # Calculate statistics for each bandwidth pair
            for pair in df['bandwidth_pair'].unique():
                pair_data = df[df['bandwidth_pair'] == pair]
                
                # Skip if too few data points
                if len(pair_data) < 10:
                    logger.warning(f"Too few data points for {pair}, skipping")
                    continue
                
                # Calculate median ratio (more robust than mean)
                median_ratio = pair_data['ratio'].median()
                
                # Check if ratio is close to common values (1, 2, 4, etc.)
                common_ratios = [1, 2, 4, 8, 10, 20]
                closest_ratio = min(common_ratios, key=lambda x: abs(x - median_ratio))
                
                # Use the closest common ratio if it's within tolerance
                if abs(closest_ratio - median_ratio) / median_ratio <= self.tolerance:
                    self.factors[pair] = closest_ratio
                else:
                    # Otherwise use the exact ratio
                    self.factors[pair] = median_ratio
                
                logger.info(f"Learned factor for {pair}: {self.factors[pair]}")
            
            # Save factors to file
            with open('results/bandwidth_factors.json', 'w') as f:
                json.dump(self.factors, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error learning factors from {data_file}: {e}")
            return False
    
    def extract_ocusync_info(self, rf_uuid):
        """Extract bandwidth and ID number from OcuSync 4 ID string"""
        match = re.search(OCUSYNC_PATTERN, rf_uuid, re.IGNORECASE)
        if match:
            bandwidth = match.group(1)
            id_number = int(match.group(2))
            return bandwidth, id_number
        return None, None
    
    def convert_id(self, from_bandwidth, from_id, to_bandwidth):
        """
        Convert an ID from one bandwidth to another
        
        Parameters:
        - from_bandwidth: Source bandwidth (e.g., '10MHz')
        - from_id: Source ID number
        - to_bandwidth: Target bandwidth (e.g., '20MHz')
        
        Returns:
        - Converted ID number or None if conversion not possible
        """
        # Check if direct conversion factor exists
        pair_key = f"{from_bandwidth}_to_{to_bandwidth}"
        
        if pair_key in self.factors and self.factors[pair_key] is not None:
            # Direct conversion
            return round(from_id * self.factors[pair_key])
        
        # Try reverse conversion
        reverse_key = f"{to_bandwidth}_to_{from_bandwidth}"
        
        if reverse_key in self.factors and self.factors[reverse_key] is not None:
            # Reverse conversion
            return round(from_id / self.factors[reverse_key])
        
        # Try indirect conversion through another bandwidth
        bandwidths = set([key.split('_to_')[0] for key in self.factors.keys()] + 
                       [key.split('_to_')[1] for key in self.factors.keys()])
        
        for intermediate_bw in bandwidths:
            if intermediate_bw == from_bandwidth or intermediate_bw == to_bandwidth:
                continue
                
            # Check if both conversions exist
            step1_key = f"{from_bandwidth}_to_{intermediate_bw}"
            step2_key = f"{intermediate_bw}_to_{to_bandwidth}"
            
            if (step1_key in self.factors and self.factors[step1_key] is not None and
                step2_key in self.factors and self.factors[step2_key] is not None):
                
                # Two-step conversion
                intermediate_id = round(from_id * self.factors[step1_key])
                return round(intermediate_id * self.factors[step2_key])
        
        # No conversion path found
        return None
    
    def normalize_to_base_bandwidth(self, bandwidth, id_number, base_bandwidth='10MHz'):
        """
        Normalize any bandwidth ID to the base bandwidth
        
        Parameters:
        - bandwidth: Source bandwidth
        - id_number: Source ID number
        - base_bandwidth: Base bandwidth to normalize to
        
        Returns:
        - Normalized ID number or None if conversion not possible
        """
        if bandwidth == base_bandwidth:
            return id_number
            
        return self.convert_id(bandwidth, id_number, base_bandwidth)
    
    def assign_drone_id(self, ocusync_id):
        """
        Assign a unique drone ID based on OcuSync ID
        
        Parameters:
        - ocusync_id: OcuSync ID string (e.g., 'V4_10MHz_100')
        
        Returns:
        - Unique drone identifier
        """
        bandwidth, id_number = self.extract_ocusync_info(ocusync_id)
        
        if bandwidth is None or id_number is None:
            return None
            
        # Normalize to base bandwidth
        base_id = self.normalize_to_base_bandwidth(bandwidth, id_number)
        
        if base_id is None:
            return None
            
        # Generate key based on normalized ID
        key = f"normalized_{base_id}"
        
        # Assign drone ID if not already assigned
        if key not in self.drone_map:
            self.drone_map[key] = f"drone_{self.next_drone_id}"
            self.next_drone_id += 1
            
        return self.drone_map[key]
    
    def process_rf_data(self, input_file, output_file=None):
        """
        Process RF data and add harmonized drone IDs
        
        Parameters:
        - input_file: Path to input CSV file or DataFrame
        - output_file: Path to output CSV file (default: input_file with '_harmonized' suffix)
        
        Returns:
        - Path to output file or DataFrame if input was DataFrame
        """
        try:
            # Load data
            if isinstance(input_file, pd.DataFrame):
                df = input_file
                is_dataframe_input = True
            else:
                df = pd.read_csv(input_file)
                is_dataframe_input = False
            
            # Ensure required columns exist
            if 'rf_uuid' not in df.columns:
                logger.error("Required column 'rf_uuid' not found in input data")
                return None
            
            # Extract bandwidth and ID from rf_uuid
            df['bandwidth'] = None
            df['id_number'] = None
            
            logger.info("Extracting bandwidth and ID numbers...")
            ocusync_entries = df['rf_uuid'].str.contains('V4_', case=False, na=False)
            
            # Apply only to OcuSync entries to avoid processing non-matching entries
            if ocusync_entries.any():
                ocusync_df = df[ocusync_entries].copy()
                
                # Extract bandwidth and ID number
                results = ocusync_df['rf_uuid'].apply(self.extract_ocusync_info)
                ocusync_df['bandwidth'] = results.apply(lambda x: x[0])
                ocusync_df['id_number'] = results.apply(lambda x: x[1])
                
                # Update original dataframe
                df.loc[ocusync_entries, 'bandwidth'] = ocusync_df['bandwidth']
                df.loc[ocusync_entries, 'id_number'] = ocusync_df['id_number']
            
            # Add normalized ID
            df['normalized_id'] = None
            
            logger.info("Normalizing IDs...")
            for index, row in df[ocusync_entries].iterrows():
                bandwidth = row['bandwidth']
                id_number = row['id_number']
                
                if bandwidth is not None and id_number is not None:
                    df.at[index, 'normalized_id'] = self.normalize_to_base_bandwidth(bandwidth, id_number)
            
            # Add harmonized drone ID
            df['harmonized_drone_id'] = None
            
            logger.info("Assigning harmonized drone IDs...")
            for index, row in df[ocusync_entries].iterrows():
                df.at[index, 'harmonized_drone_id'] = self.assign_drone_id(row['rf_uuid'])
            
            # Generate summary
            summary = {
                'total_records': len(df),
                'ocusync_records': ocusync_entries.sum(),
                'unique_bandwidths': df['bandwidth'].nunique(),
                'unique_raw_ids': df[ocusync_entries]['id_number'].nunique() if ocusync_entries.any() else 0,
                'unique_normalized_ids': df['normalized_id'].nunique(),
                'unique_harmonized_drones': df['harmonized_drone_id'].nunique()
            }
            
            # If input was DataFrame, return the DataFrame
            if is_dataframe_input:
                logger.info("Processed DataFrame in memory")
                return df
            
            # Otherwise save to file
            if output_file is None:
                base, ext = os.path.splitext(input_file)
                output_file = f"{base}_harmonized{ext}"
                
            df.to_csv(output_file, index=False)
            logger.info(f"Saved harmonized data to {output_file}")
            
            # Save drone map
            try:
                with open('results/drone_id_map.json', 'w') as f:
                    json.dump(self.drone_map, f, indent=2)
                
                with open('results/harmonization_summary.json', 'w') as f:
                    json.dump(summary, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save results metadata: {e}")
            
            return output_file
        except Exception as e:
            logger.error(f"Error processing data from {input_file}: {e}")
            return None
    
    def export_to_gpx(self, harmonized_data, output_file, source_type="SDR"):
        """
        Export harmonized data to GPX format for ATAK
        
        Parameters:
        - harmonized_data: DataFrame with harmonized data
        - output_file: Path to output GPX file
        - source_type: Source type for labeling (SDR or Aeroscope)
        
        Returns:
        - Path to output file
        """
        try:
            logger.info(f"Exporting harmonized data to GPX: {output_file}")
            
            gpx = gpxpy.gpx.GPX()
            
            # Create a track for each unique drone
            if 'harmonized_drone_id' in harmonized_data.columns:
                # Use harmonized IDs if available
                drone_groups = harmonized_data.groupby('harmonized_drone_id')
            elif 'SerialNumber' in harmonized_data.columns:
                # Use DJI serial numbers for Aeroscope data
                drone_groups = harmonized_data.groupby('SerialNumber')
            else:
                # Fall back to track_id
                drone_groups = harmonized_data.groupby('track_id')
            
            for drone_id, drone_data in drone_groups:
                if pd.isna(drone_id):
                    continue
                    
                # Create GPX track
                gpx_track = gpxpy.gpx.GPXTrack(name=f"{source_type} - Drone {drone_id}")
                gpx.tracks.append(gpx_track)
                
                # Create track segment
                gpx_segment = gpxpy.gpx.GPXTrackSegment()
                gpx_track.segments.append(gpx_segment)
                
                # Sort data by time
                if 'time' in drone_data.columns:
                    time_col = 'time'
                elif 'DetectionTime' in drone_data.columns:
                    time_col = 'DetectionTime'
                else:
                    # If no time column, just use the order
                    time_col = None
                    
                if time_col:
                    sorted_data = drone_data.sort_values(time_col)
                else:
                    sorted_data = drone_data
                
                # Add track points with timestamps
                for _, row in sorted_data.iterrows():
                    # Get coordinates
                    if 'sensor_latitude' in row and 'sensor_longitude' in row:
                        lat = row['sensor_latitude']
                        lon = row['sensor_longitude']
                    elif 'DroneLatitude' in row and 'DroneLongitude' in row:
                        lat = row['DroneLatitude']
                        lon = row['DroneLongitude']
                    else:
                        continue
                    
                    # Get timestamp
                    if time_col:
                        dt = row[time_col]
                        if isinstance(dt, str):
                            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                    else:
                        dt = None
                    
                    # Create point
                    point = gpxpy.gpx.GPXTrackPoint(
                        latitude=lat,
                        longitude=lon,
                        time=dt
                    )
                    
                    # Add RSSI as a custom attribute if available
                    if 'rf_rssi' in row and not pd.isna(row['rf_rssi']):
                        # Simply store as XML string, which GPX parsers will ignore but preserve
                        point.extensions = ["<rssi>{}</rssi>".format(row['rf_rssi'])]
                    
                    gpx_segment.points.append(point)
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write(gpx.to_xml())
                
            logger.info(f"Successfully exported GPX to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error exporting to GPX: {e}")
            return None
    
    def export_to_kml(self, harmonized_data, output_file, source_type="SDR"):
        """
        Export harmonized data to KML format for ATAK
        
        Parameters:
        - harmonized_data: DataFrame with harmonized data
        - output_file: Path to output KML file
        - source_type: Source type for labeling (SDR or Aeroscope)
        
        Returns:
        - Path to output file
        """
        try:
            logger.info(f"Exporting harmonized data to KML: {output_file}")
            
            kml = simplekml.Kml()
            
            # Create a folder for this source
            folder = kml.newfolder(name=f"{source_type} Detections")
            
            # Create a track for each unique drone
            if 'harmonized_drone_id' in harmonized_data.columns:
                # Use harmonized IDs if available
                drone_groups = harmonized_data.groupby('harmonized_drone_id')
            elif 'SerialNumber' in harmonized_data.columns:
                # Use DJI serial numbers for Aeroscope data
                drone_groups = harmonized_data.groupby('SerialNumber')
            else:
                # Fall back to track_id
                drone_groups = harmonized_data.groupby('track_id')
            
            for drone_id, drone_data in drone_groups:
                if pd.isna(drone_id):
                    continue
                    
                # Create KML track
                track = folder.newlinestring(name=f"{source_type} - Drone {drone_id}")
                
                # Style based on source
                if source_type == "SDR":
                    track.style.linestyle.color = simplekml.Color.red
                    icon_style = "drone_red"
                else:  # Aeroscope
                    track.style.linestyle.color = simplekml.Color.blue
                    icon_style = "drone_blue"
                
                track.style.linestyle.width = 4
                
                # Sort data by time
                if 'time' in drone_data.columns:
                    time_col = 'time'
                elif 'DetectionTime' in drone_data.columns:
                    time_col = 'DetectionTime'
                else:
                    # If no time column, just use the order
                    time_col = None
                    
                if time_col:
                    sorted_data = drone_data.sort_values(time_col)
                else:
                    sorted_data = drone_data
                
                # Extract coordinates
                coords = []
                for _, row in sorted_data.iterrows():
                    # Get coordinates
                    if 'sensor_longitude' in row and 'sensor_latitude' in row:
                        lon = row['sensor_longitude']
                        lat = row['sensor_latitude']
                    elif 'DroneLongitude' in row and 'DroneLatitude' in row:
                        lon = row['DroneLongitude']
                        lat = row['DroneLatitude']
                    else:
                        continue
                        
                    coords.append((lon, lat))
                
                # Add coordinates to track
                if coords:
                    track.coords = coords
                
                # Add timestamps if available
                if time_col:
                    times = [row[time_col] for _, row in sorted_data.iterrows() if not pd.isna(row[time_col])]
                    if times:
                        time_str = "\n".join([str(t) for t in times])
                        track.extendeddata.newdata(name="Times", value=time_str)
                
                # Add RSSI data if available
                if 'rf_rssi' in sorted_data.columns:
                    rssi_values = sorted_data['rf_rssi'].dropna().tolist()
                    if rssi_values:
                        avg_rssi = sum(rssi_values) / len(rssi_values)
                        track.extendeddata.newdata(name="AverageRSSI", value=str(avg_rssi))
                
                # Add other extended data
                track.extendeddata.newdata(name="Source", value=source_type)
                track.extendeddata.newdata(name="DroneID", value=str(drone_id))
                
                # Add starting point placemark with icon
                if coords:
                    start_point = folder.newpoint(name=f"{source_type} - Drone {drone_id} Start")
                    start_point.coords = [coords[0]]
                    start_point.style.iconstyle.icon.href = f"http://maps.google.com/mapfiles/kml/paddle/{icon_style}.png"
                    
                    # Add end point placemark
                    end_point = folder.newpoint(name=f"{source_type} - Drone {drone_id} End")
                    end_point.coords = [coords[-1]]
                    end_point.style.iconstyle.icon.href = f"http://maps.google.com/mapfiles/kml/paddle/{icon_style}.png"
            
            # Write to file
            kml.save(output_file)
            
            logger.info(f"Successfully exported KML to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error exporting to KML: {e}")
            return None
            
    def export_to_cot_xml(self, harmonized_data, output_file, source_type="SDR"):
        """
        Export harmonized data to Cursor on Target (CoT) XML format for ATAK
        
        Parameters:
        - harmonized_data: DataFrame with harmonized data
        - output_file: Path to output XML file
        - source_type: Source type for labeling (SDR or Aeroscope)
        
        Returns:
        - Path to output file
        """
        try:
            logger.info(f"Exporting harmonized data to CoT XML: {output_file}")
            
            # Create XML header
            xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
            xml_lines.append('<events>')
            
            # Create a track for each unique drone
            if 'harmonized_drone_id' in harmonized_data.columns:
                # Use harmonized IDs if available
                drone_groups = harmonized_data.groupby('harmonized_drone_id')
            elif 'SerialNumber' in harmonized_data.columns:
                # Use DJI serial numbers for Aeroscope data
                drone_groups = harmonized_data.groupby('SerialNumber')
            else:
                # Fall back to track_id
                drone_groups = harmonized_data.groupby('track_id')
            
            for drone_id, drone_data in drone_groups:
                if pd.isna(drone_id):
                    continue
                
                # Sort data by time
                if 'time' in drone_data.columns:
                    time_col = 'time'
                elif 'DetectionTime' in drone_data.columns:
                    time_col = 'DetectionTime'
                else:
                    # If no time column, just use the order
                    time_col = None
                    
                if time_col:
                    sorted_data = drone_data.sort_values(time_col)
                else:
                    sorted_data = drone_data
                
                # Process each point
                for _, row in sorted_data.iterrows():
                    # Get coordinates
                    if 'sensor_latitude' in row and 'sensor_longitude' in row:
                        lat = row['sensor_latitude']
                        lon = row['sensor_longitude']
                    elif 'DroneLatitude' in row and 'DroneLongitude' in row:
                        lat = row['DroneLatitude']
                        lon = row['DroneLongitude']
                    else:
                        continue
                    
                    # Get timestamp
                    if time_col and not pd.isna(row[time_col]):
                        dt = row[time_col]
                        if isinstance(dt, str):
                            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                        time_str = dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                    else:
                        time_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                    
                    # Create unique ID for this point
                    point_uid = f"{source_type}_{drone_id}_{hash(str(lat) + str(lon) + time_str) & 0xffffffff}"
                    
                    # Create CoT event
                    xml_lines.append(f'  <event version="2.0" uid="{point_uid}" type="a-f-G-U-C" time="{time_str}" start="{time_str}" stale="{time_str}" how="m-g">')
                    xml_lines.append(f'    <point lat="{lat}" lon="{lon}" hae="0" ce="9999999" le="9999999"/>')
                    xml_lines.append('    <detail>')
                    xml_lines.append(f'      <contact callsign="{source_type}_Drone_{drone_id}"/>')
                    xml_lines.append(f'      <remarks>Source: {source_type}, Drone ID: {drone_id}</remarks>')
                    
                    # Add RSSI if available
                    if 'rf_rssi' in row and not pd.isna(row['rf_rssi']):
                        xml_lines.append(f'      <rssi value="{row["rf_rssi"]}"/>')
                    
                    # Add bandwidth/ID info if available
                    if 'bandwidth' in row and 'id_number' in row:
                        if not pd.isna(row['bandwidth']) and not pd.isna(row['id_number']):
                            xml_lines.append(f'      <ocusync bandwidth="{row["bandwidth"]}" id="{row["id_number"]}"/>')
                    
                    xml_lines.append('    </detail>')
                    xml_lines.append('  </event>')
            
            # Close XML
            xml_lines.append('</events>')
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write('\n'.join(xml_lines))
            
            logger.info(f"Successfully exported CoT XML to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error exporting to CoT XML: {e}")
            return None

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OcuSync 4 ID Harmonizer')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, help='Path to output CSV file')
    parser.add_argument('--factors', type=str, help='Path to JSON file with bandwidth factors')
    parser.add_argument('--learn', type=str, help='Path to CSV file with ID relationships to learn factors from')
    parser.add_argument('--tolerance', type=float, default=0.1, help='Tolerance for ratio matching (0.1 = 10%)')
    parser.add_argument('--export-kml', type=str, help='Export to KML file')
    parser.add_argument('--export-gpx', type=str, help='Export to GPX file')
    parser.add_argument('--export-cot', type=str, help='Export to CoT XML file')
    parser.add_argument('--source-type', type=str, default='SDR', help='Source type label (SDR or Aeroscope)')
    parser.add_argument('--dji-input', type=str, help='Path to DJI data CSV for export')
    parser.add_argument('--dji-kml', type=str, help='Export DJI data to KML file')
    parser.add_argument('--dji-gpx', type=str, help='Export DJI data to GPX file')
    parser.add_argument('--dji-cot', type=str, help='Export DJI data to CoT XML file')
    parser.add_argument('--sample', type=int, help='Sample size for processing (optional)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('exports', exist_ok=True)
    
    # Initialize harmonizer
    harmonizer = OcuSyncHarmonizer(tolerance=args.tolerance)
    
    # Load factors if provided
    if args.factors and os.path.exists(args.factors):
        harmonizer.load_factors_from_file(args.factors)
    
    # Learn factors if provided
    if args.learn and os.path.exists(args.learn):
        try:
            harmonizer.learn_factors_from_data(args.learn)
        except Exception as e:
            logger.warning(f"Could not learn factors, using defaults: {e}")
            # Set some default factors for demo purposes
            harmonizer.factors = {
                '10MHz_to_20MHz': 2.0,
                '10MHz_to_40MHz': 4.0,
                '20MHz_to_40MHz': 2.0,
                '20MHz_to_10MHz': 0.5,
                '40MHz_to_10MHz': 0.25,
                '40MHz_to_20MHz': 0.5
            }
    
    # Process data - use a sample to speed up demo if specified
    try:
        # Read only a sample of the data for faster processing
        if args.sample:
            input_data = pd.read_csv(args.input, nrows=args.sample)
            sample_output = args.output + ".sample" if args.output else None
            output_file = harmonizer.process_rf_data(input_data, sample_output)
        else:
            output_file = harmonizer.process_rf_data(args.input, args.output)
        
        # Check if we got a return value
        if output_file is not None:
            logger.info("Harmonization complete")
            
            # Export data
            try:
                # If it's a DataFrame, use it directly, otherwise load from file
                if isinstance(output_file, pd.DataFrame):
                    harmonized_data = output_file
                else:
                    harmonized_data = pd.read_csv(output_file)
                
                # Export to requested formats
                if args.export_cot:
                    harmonizer.export_to_cot_xml(harmonized_data, args.export_cot, args.source_type)
                
                if args.export_kml and 'gpxpy' in globals() and 'simplekml' in globals():
                    harmonizer.export_to_kml(harmonized_data, args.export_kml, args.source_type)
                    
                if args.export_gpx and 'gpxpy' in globals():
                    harmonizer.export_to_gpx(harmonized_data, args.export_gpx, args.source_type)
            except Exception as e:
                logger.error(f"Error during export: {e}")
        else:
            logger.error("Harmonization failed")
    except Exception as e:
        logger.error(f"Error in harmonization: {e}")
    
    # Process DJI data if provided
    if args.dji_input and os.path.exists(args.dji_input):
        try:
            # Load DJI data (with sample if specified)
            if args.sample:
                dji_data = pd.read_csv(args.dji_input, nrows=args.sample)
            else:
                dji_data = pd.read_csv(args.dji_input)
                
            logger.info(f"Loaded DJI data from {args.dji_input}: {len(dji_data)} records")
            
            # Export to CoT XML (always available)
            if args.dji_cot:
                harmonizer.export_to_cot_xml(dji_data, args.dji_cot, "Aeroscope")
                
            # Export to other formats if libraries are available
            if args.dji_kml and 'simplekml' in globals():
                harmonizer.export_to_kml(dji_data, args.dji_kml, "Aeroscope")
                
            if args.dji_gpx and 'gpxpy' in globals():
                harmonizer.export_to_gpx(dji_data, args.dji_gpx, "Aeroscope")
        except Exception as e:
            logger.error(f"Error processing DJI data: {e}")

if __name__ == "__main__":
    main()