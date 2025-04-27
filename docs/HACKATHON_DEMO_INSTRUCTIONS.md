# Drone Track Harmonizer - Hackathon Demo Instructions

## Project Overview

This project harmonizes drone tracking data from two sources:
1. SDR/Axon RF data (OcuSync 4 protocol)
2. DJI Aeroscope high-resolution positional data

Our solution addresses the challenge of identifying the same drone across different bandwidth transmissions in the OcuSync 4 protocol, and displays the data in ATAK for maritime security operations.

## Demo Setup

### Prerequisites
- Python 3.7+ with packages: pandas, numpy, matplotlib, seaborn, gpxpy, simplekml
- ATAK installed on Android device (tablet or phone)
- Network connectivity between your computer and ATAK device

### Installation

1. Install required Python packages:
```bash
pip install pandas numpy matplotlib seaborn tqdm gpxpy simplekml
```

2. Ensure the CSV data files are in the project directory:
   - `Site1.csv` - OcuSync 4 RF data
   - `Site1_DJI_Data.csv` - DJI Aeroscope data

3. Make sure demo scripts are executable:
```bash
chmod +x run_hackathon_demo.sh
chmod +x atak_udp_sender.py
```

## Running the Demo

### Option 1: Full Automated Demo

1. Edit the `run_hackathon_demo.sh` script to set your ATAK device IP address:
```bash
ATAK_IP="192.168.1.100"  # Replace with your ATAK device IP
```

2. Run the demo script:
```bash
./run_hackathon_demo.sh
```

3. Follow the prompts and wait for the script to process the data and send it to ATAK.

### Option 2: Step-by-Step Demo

#### Step 1: Process the data
```bash
# Run sample analysis to quickly verify patterns
python sample_analysis.py --rf_file Site1.csv --dji_file Site1_DJI_Data.csv --sample 50000

# Process the full dataset
python analyze_ocusync_ids.py --rf_file Site1.csv --dji_file Site1_DJI_Data.csv

# Harmonize the data and export to various formats
python ocusync_harmonizer.py --input Site1.csv \
    --output harmonized_data.csv \
    --learn results/dji_id_relationships.csv \
    --export-cot sdr_drones.xml \
    --source-type SDR \
    --dji-input Site1_DJI_Data.csv \
    --dji-cot dji_drones.xml
```

#### Step 2: Send data to ATAK
```bash
# Replace 192.168.1.100 with your ATAK device IP
python atak_udp_sender.py --host 192.168.1.100 --port 4242 \
    --sdr-cot sdr_drones.xml \
    --dji-cot dji_drones.xml
```

## ATAK Configuration

1. Configure ATAK to receive UDP CoT messages:
   - Open ATAK settings
   - Go to "Network Preferences" > "Inputs"
   - Add a new input with protocol "UDP" and port "4242"
   - Save the configuration

2. If you have the Drone Track Harmonizer plugin installed:
   - A new tool icon will appear in the ATAK toolbar
   - Click to open the plugin interface
   - Use the toggles to show/hide different data sources

3. Without the plugin, the data will appear as:
   - Red tracks: SDR drone detections
   - Blue tracks: Aeroscope drone detections

## Key Features to Demonstrate

1. **ID Harmonization**: Show how the system identifies the same drone across different bandwidths.

2. **Multi-Source Integration**: Demonstrate the correlation between SDR and Aeroscope data.

3. **ATAK Visualization**: Show how the data appears on the map, with togglable layers and drone information.

4. **Maritime Security Application**: Explain how this technology applies to maritime surveillance and security.

## Troubleshooting

- **Data not appearing in ATAK**: Verify network connectivity and UDP input configuration.
- **Processing errors**: Check that all required packages are installed and CSV files are present.
- **Plugin not working**: The plugin is optional; the CoT data will still appear in ATAK without it.

## Resources

- Visualization outputs are saved in the `visualizations` directory
- Analysis results are saved in the `results` directory
- Exported files are saved according to the paths specified in commands