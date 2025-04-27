# Drone Track Harmonizer

A comprehensive solution for harmonizing drone tracking data from multiple sources and visualizing on an interactive map.

## Overview

This project addresses the challenge of harmonizing drone tracking data from different sources:
- SDR/Axon RF detections of OcuSync 4 protocol
- DJI Aeroscope high-resolution positional data

The solution identifies the same drone across different bandwidth transmissions in the OcuSync 4 protocol and visualizes the harmonized data on an interactive map.

## Interactive Visualization

The harmonized drone tracks can be viewed in an interactive map:
- Local access: [View Interactive Map](validation/working_map.html)
- Online access: Available through GitHub Pages

## Project Structure

```
drone-track-harmonizer/
├── python/                    # Python data processing scripts
│   ├── analyze_ocusync_ids.py    # Analyzes OcuSync 4 ID patterns
│   ├── ocusync_harmonizer.py     # Harmonizes drone IDs across bandwidths
│   ├── visualize_ocusync_patterns.py # Creates visualizations of patterns
│   ├── sample_analysis.py        # Quick analysis on sample data
│   └── run_hackathon_demo.sh     # Complete end-to-end demo script
│
├── docs/                      # Documentation
│   ├── SOLUTION_OVERVIEW.md       # High-level solution overview
│   ├── HACKATHON_DEMO_INSTRUCTIONS.md # Step-by-step demo guide
│   └── QUICK_START.md             # Quick start guide
│
├── validation/                # Visualization and validation
│   ├── working_map.html          # Interactive visualization of drone data
│   └── validation_summary.md     # Summary of validation results
│
├── exports/                   # Output directory for processed data
└── results/                   # Results from analysis
```

## Quick Start

1. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn tqdm gpxpy simplekml
```

2. Process data and export:
```bash
cd python
python ocusync_harmonizer.py --input ../Site1.csv \
    --output ../exports/harmonized_data.csv \
    --export-gpx ../exports/sdr_drones.gpx \
    --source-type SDR \
    --dji-input ../Site1_DJI_Data.csv \
    --dji-gpx ../exports/dji_drones.gpx
```

3. View visualization:
Open `validation/working_map.html` in your web browser to view the harmonized drone tracking visualization.

## Documentation

- For complete setup instructions, see [HACKATHON_DEMO_INSTRUCTIONS.md](docs/HACKATHON_DEMO_INSTRUCTIONS.md)
- For a high-level overview of the solution, see [SOLUTION_OVERVIEW.md](docs/SOLUTION_OVERVIEW.md)
- For quick testing, see [QUICK_START.md](docs/QUICK_START.md)

## Data Requirements

- `Site1.csv`: OcuSync 4 RF data from SDR/Axon
- `Site1_DJI_Data.csv`: DJI Aeroscope positional data

Place these files in the project root directory before running the processing scripts.

## License

This project is licensed under the MIT License - see the LICENSE file for details.