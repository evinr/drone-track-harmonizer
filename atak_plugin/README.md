# Drone Track Harmonizer ATAK Plugin

This plugin for Android Tactical Assault Kit (ATAK) displays harmonized drone tracking data from multiple sources:
- SDR/Axon RF detections with harmonized IDs
- DJI Aeroscope high-resolution positional data

## Features

- Displays drone tracks from SDR and Aeroscope as separate togglable layers
- Color-coding by source (SDR vs Aeroscope)
- Automatically processes incoming UDP CoT messages
- Highlights when the same drone is detected by both systems
- Shows RSSI values from RF detection

## Setup

### Prerequisites

- ATAK installed on Android device
- Python environment for data processing
- Gradle and Android SDK for building the plugin

### Building the Plugin

1. Clone this repository
2. Open the project in Android Studio
3. Build the APK
4. Install on your ATAK device

### Using with Processed Data

This plugin works with the OcuSync Harmonizer Python tool chain:

1. Process the RF and DJI data:
```bash
python ocusync_harmonizer.py --input Site1.csv --output harmonized_data.csv --export-cot sdr_drones.xml --source-type SDR --dji-input Site1_DJI_Data.csv --dji-cot dji_drones.xml
```

2. Send the data to ATAK:
```bash
python atak_udp_sender.py --host ATAK_IP_ADDRESS --port 4242 --sdr-cot sdr_drones.xml --dji-cot dji_drones.xml
```

## Building from Source

This plugin uses the ATAK-CIV Plugin Template. To build:

1. Set up your development environment following ATAK SDK instructions
2. Configure the gradle properties in `gradle.properties`
3. Run `./gradlew assembleDebug` to build the plugin
4. Install on your ATAK device

## Project Structure

- `app/src/main/java/com/dronetrack/atak/` - Java source code
- `app/src/main/res/` - Resources (layouts, icons, strings)
- `app/src/main/assets/` - Static assets
- `cot/` - Sample CoT XML files

## Screens

(Screenshots will be added after development)