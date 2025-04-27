#!/usr/bin/env python3
"""
Fix GPX files by replacing <n> tags with <name> tags
"""

import sys
import re

def fix_gpx_file(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Fix using regex to be more precise
    fixed_content = re.sub(r'<n>(.*?)</n>', r'<name>\1</name>', content)
    
    with open(output_file, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed GPX file saved to {output_file}")

if __name__ == "__main__":
    input_file = "../exports/dji_drones.gpx"
    output_file = "../exports/dji_drones_fixed.gpx"
    
    fix_gpx_file(input_file, output_file)