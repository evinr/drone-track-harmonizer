#!/bin/bash
# OcuSync 4 ID Harmonization Demo Script for Hackathon
# This script processes the data and sends it to ATAK

# Configuration
ATAK_IP="192.168.1.100"  # Replace with your ATAK device IP
ATAK_PORT="4242"         # Default ATAK UDP port
OUTPUT_DIR="../exports"

# Create output directory
mkdir -p $OUTPUT_DIR

# Step 1: Run the sample analysis to verify patterns
echo "Step 1: Running sample analysis to verify patterns..."
python sample_analysis.py --rf_file ../Site1.csv --dji_file ../Site1_DJI_Data.csv

# Step 2: Process the full dataset to get complete results
echo "Step 2: Processing full dataset..."
python analyze_ocusync_ids.py --rf_file ../Site1.csv --dji_file ../Site1_DJI_Data.csv

# Step 3: Harmonize the data and export to CoT format only (since we installed gpxpy and simplekml)
echo "Step 3: Harmonizing and exporting data..."
python ocusync_harmonizer.py --input ../Site1.csv \
    --output $OUTPUT_DIR/harmonized_data.csv \
    --learn ../results/dji_id_relationships.csv \
    --export-cot $OUTPUT_DIR/sdr_drones.xml \
    --export-kml $OUTPUT_DIR/sdr_drones.kml \
    --export-gpx $OUTPUT_DIR/sdr_drones.gpx \
    --source-type SDR \
    --dji-input ../Site1_DJI_Data.csv \
    --dji-cot $OUTPUT_DIR/dji_drones.xml \
    --dji-kml $OUTPUT_DIR/dji_drones.kml \
    --dji-gpx $OUTPUT_DIR/dji_drones.gpx

# Step 4: Send to ATAK
echo "Step 4: Sending data to ATAK at $ATAK_IP:$ATAK_PORT..."
echo "NOTE: Make sure your ATAK device is connected and listening for UDP messages."
# Automatically continue without requiring user input
# read -p "Press Enter to start sending data to ATAK..." 

python atak_udp_sender.py --host $ATAK_IP --port $ATAK_PORT \
    --sdr-cot $OUTPUT_DIR/sdr_drones.xml \
    --dji-cot $OUTPUT_DIR/dji_drones.xml \
    --delay 0.5

echo "Demo complete! Data has been sent to ATAK."
echo "You can find all output files in the '$OUTPUT_DIR' directory."