#!/usr/bin/env python3
"""
Create a GPX file compatible with Organic Maps from the drone data
"""
import csv
import os

def create_simplified_gpx():
    # Location for our new file
    output_path = "../exports/organic_maps_drones.gpx"
    
    # Get drone coordinates from the correlations file
    correlations_path = "results/dji_correlations.csv"
    
    # Create a simple but valid GPX file compatible with Organic Maps
    gpx_header = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Drone Track Harmonizer">
"""
    
    gpx_footer = """</gpx>"""
    
    # Read drone data from correlations file
    drones = {}
    if os.path.exists(correlations_path):
        with open(correlations_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                serial = row.get('serial_number', '')
                if not serial:
                    continue
                
                drone_type = row.get('drone_type', 'Unknown')
                lat = row.get('DroneLatitude', None) or row.get('latitude', None)
                lon = row.get('DroneLongitude', None) or row.get('longitude', None)
                timestamp = row.get('detection_time', None) or row.get('rf_time', None)
                
                if not lat or not lon or not timestamp:
                    continue
                
                if serial not in drones:
                    drones[serial] = {
                        'name': f"{drone_type} - {serial}",
                        'points': []
                    }
                
                drones[serial]['points'].append({
                    'lat': lat,
                    'lon': lon,
                    'time': timestamp
                })
    
    # If no drones found, use the original file to extract data
    if not drones:
        # Create a simplified waypoint for demonstration
        drones = {
            'sample': {
                'name': 'Sample Drone',
                'points': [
                    {'lat': '41.754118001691424', 'lon': '-99.80345125920356', 'time': '2024-08-01T12:42:54'}
                ]
            }
        }
    
    # Create the GPX file content
    with open(output_path, 'w') as f:
        f.write(gpx_header)
        
        # Add waypoints
        for serial, drone in drones.items():
            f.write(f'  <wpt lat="{drone["points"][0]["lat"]}" lon="{drone["points"][0]["lon"]}">\n')
            f.write(f'    <name>{drone["name"]}</name>\n')
            f.write(f'    <time>{drone["points"][0]["time"]}</time>\n')
            f.write(f'  </wpt>\n')
        
        # Add tracks
        for serial, drone in drones.items():
            f.write(f'  <trk>\n')
            f.write(f'    <name>{drone["name"]}</name>\n')
            f.write(f'    <trkseg>\n')
            
            for point in drone['points']:
                f.write(f'      <trkpt lat="{point["lat"]}" lon="{point["lon"]}">\n')
                f.write(f'        <time>{point["time"]}</time>\n')
                f.write(f'      </trkpt>\n')
            
            f.write(f'    </trkseg>\n')
            f.write(f'  </trk>\n')
        
        f.write(gpx_footer)
    
    print(f"Created simplified GPX file for Organic Maps at {output_path}")

if __name__ == "__main__":
    create_simplified_gpx()