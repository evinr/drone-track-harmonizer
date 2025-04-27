#!/bin/bash
# Setup script for Drone Track Harmonizer

# Create necessary directories if they don't exist
mkdir -p exports
mkdir -p sample_results
mkdir -p results

# Make Python scripts executable
chmod +x python/*.py
chmod +x python/*.sh

# Install Python dependencies (uncomment if needed)
# pip install pandas numpy matplotlib seaborn tqdm gpxpy simplekml

echo "Setup complete!"
echo "To run the demo, place your data files in the project root:"
echo "- Site1.csv (OcuSync 4 RF data)"
echo "- Site1_DJI_Data.csv (DJI Aeroscope data)"
echo ""
echo "Then run: cd python && ./run_hackathon_demo.sh"