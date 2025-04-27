#!/usr/bin/env python3
"""
Create a new GPX file with proper name tags from the original file
"""
import re
from xml.etree import ElementTree as ET

def fix_gpx():
    input_file = "../exports/dji_drones.gpx"
    output_file = "../exports/dji_drones_fixed.gpx"
    
    # Read the file as text to manually replace the tag
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Replace <n> with <name> and </n> with </name>
    content = content.replace("<n>", "<name>")
    content = content.replace("</n>", "</name>")
    
    # Write the new file
    with open(output_file, 'w') as f:
        f.write(content)
    
    # Verify the replacement
    with open(output_file, 'r') as f:
        first_100_chars = f.read(200)
    
    print(f"Fixed file created at {output_file}")
    print(f"First part of file: {first_100_chars}")

if __name__ == "__main__":
    fix_gpx()