#!/usr/bin/env python3
"""
Fix GPX file by creating a simplified version that should work with Organic Maps
"""

def create_fixed_gpx():
    input_path = "../exports/dji_drones.gpx"
    output_path = "../exports/dji_drones_fixed.gpx"
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        if "<n>" in line:
            line = line.replace("<n>", "<name>")
        if "</n>" in line:
            line = line.replace("</n>", "</name>")
        fixed_lines.append(line)
    
    with open(output_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed GPX file created at {output_path}")

if __name__ == "__main__":
    create_fixed_gpx()