#!/usr/bin/env python3
"""
Fix GPX files by replacing <n> tags with <name> tags using basic string replacement
"""

def fix_gpx_file():
    input_file = "../exports/dji_drones.gpx"
    output_file = "../exports/dji_drones_fixed.gpx"
    
    # Read the entire file content
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Simple string replacement
    content = content.replace("<n>", "<name>")
    content = content.replace("</n>", "</name>")
    
    # Write the fixed content
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Fixed GPX file saved to {output_file}")

if __name__ == "__main__":
    fix_gpx_file()