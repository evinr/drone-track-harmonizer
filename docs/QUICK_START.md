# OcuSync 4 ID Harmonizer - Quick Start Guide

This guide helps you get started with a smaller sample of data to test the OcuSync 4 ID harmonization approach.

## Prerequisites

- Python 3.7+
- Required packages: pandas, numpy, matplotlib, seaborn

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn tqdm
```

## Quick Test with Sample Data

For a quick test on a sample of the data (useful for initial validation):

```bash
python sample_analysis.py --rf_file Site1.csv --dji_file Site1_DJI_Data.csv --sample 50000
```

This will:
1. Load a sample of 50,000 rows from the RF data
2. Find bandwidth transitions
3. Analyze ID patterns across bandwidths
4. Correlate with DJI data
5. Create visualizations
6. Save results to the `sample_results` directory

## Examine Results

After running the sample analysis, check these key files:

- `sample_results/sample_patterns.json`: Contains statistics on ID relationships across bandwidths
- `sample_results/sample_drone_ids.json`: Shows ID mappings for each drone across bandwidths 
- `sample_results/sample_ratio_histogram_*.png`: Visualizes the distribution of ID ratios
- `sample_results/sample_id_scatter_*.png`: Shows the relationship between IDs

## Run Full Analysis

When you're ready to run the complete analysis:

```bash
./run_ocusync_analysis.sh
```

To run with a smaller sample size for the full pipeline:

```bash
./run_ocusync_analysis.sh --sample 100000
```

## Expected Patterns

If the linear mapping hypothesis is correct, you should observe:
- The ratio of `id_to / id_from` clustering around values like 1, 2, and 4
- A clear linear relationship in scatter plots of `id_from` vs `id_to`
- Consistent patterns across different drones

## Troubleshooting

- **"No transitions found"**: Try increasing the sample size or time window
- **"No DJI correlations found"**: Check that the time ranges in both datasets overlap
- **High variation in ratios**: This could indicate the linear mapping hypothesis doesn't hold, or that there are multiple different mapping patterns

## Next Steps

After confirming the patterns with the sample analysis:
1. Run the full analysis to learn the conversion factors
2. Apply the harmonization algorithm to your data
3. Verify the effectiveness by checking if unique drones are consistently identified