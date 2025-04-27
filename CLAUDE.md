# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Setup: `./setup.sh`
- Run demo: `./run_hackathon_demo.sh`
- Run analysis: `python analyze_ocusync_ids.py --rf_file Site1.csv --dji_file Site1_DJI_Data.csv`
- Run visualization: `python visualize_ocusync_patterns.py --results_file results/dji_id_relationships.csv`
- Send to ATAK: `python atak_udp_sender.py --input harmonized_data.csv --host 127.0.0.1 --port 4242`

## Code Style Guidelines
- Python: 4-space indentation, snake_case for functions/variables, CamelCase for classes
- Java: CamelCase for classes, camelCase for methods/variables
- Docstrings use triple quotes with description and parameters
- Imports grouped by standard library, third-party, and local modules
- Exception handling includes descriptive log messages
- Command-line tools use argparse with helpful descriptions
- Output directories: "results/" for analysis output, "exports/" for processed data

## Project Structure
- Python code in `python/` focuses on data processing and analysis
- ATAK plugin in `atak_plugin/` handles Android integration
- Documentation in `docs/` explains usage and architecture
- Input data should be CSV files in the root directory