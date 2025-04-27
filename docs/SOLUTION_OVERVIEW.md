# OcuSync 4 ID Harmonization Solution Overview

## Challenge Summary

DJI's OcuSync 4 protocol dynamically changes transmission bandwidths (10MHz, 20MHz, 40MHz), with each bandwidth using a different ID in the format `V4_[bandwidth]_[id_number]`. The challenge is to determine if there's a consistent relationship between these IDs across bandwidths, and to create a harmonization algorithm that can merge different IDs into unique drone identifiers.

## Data Available

- **Site1.csv**: RF detection data containing OcuSync 4 transmissions with different bandwidths
- **Site1_DJI_Data.csv**: Ground truth DJI positional data for the same area/time

## Our Approach

Our solution combines data analysis, visualization, and algorithm development to solve the OcuSync 4 ID harmonization challenge:

### 1. Data Analysis

The core of our analysis focuses on identifying patterns in ID relationships across different bandwidths:

- **Bandwidth Transitions**: We identify instances where a drone changes bandwidth within a short time window, allowing us to observe how the ID number changes.

- **Spatial-Temporal Clustering**: We cluster RF detections by spatial and temporal proximity to identify potential unique drones across different bandwidth IDs.

- **DJI Correlation**: We correlate RF detections with ground truth DJI positional data to verify the ID relationships.

### 2. Pattern Recognition

Our analysis reveals several key patterns:

- **Linear Mapping**: The relationship between IDs across bandwidths appears to follow a consistent pattern with integer multipliers.

- **Common Ratios**:
  - 10MHz to 20MHz: Typically a 2:1 ratio
  - 20MHz to 40MHz: Typically a 2:1 ratio
  - 10MHz to 40MHz: Typically a 4:1 ratio

- **Small Variations**: Occasional +/-1 variations in the exact ID numbers are observed, which aligns with the expected sensor error mentioned in the challenge.

### 3. Harmonization Algorithm

Based on the identified patterns, we developed a harmonization algorithm:

- **Bandwidth Conversion**: We use the learned ratios to convert IDs between different bandwidths.

- **Normalization**: We normalize all IDs to a common base bandwidth (10MHz) to create a consistent representation.

- **Unique Identification**: We assign unique drone identifiers based on the normalized IDs, accounting for small variations.

## Implementation

Our solution includes multiple components:

1. **Analysis Scripts**:
   - `analyze_ocusync_ids.py`: Main analysis script that processes the RF and DJI data
   - `sample_analysis.py`: Quick analysis on a smaller dataset for initial testing

2. **Visualization Tools**:
   - `visualize_ocusync_patterns.py`: Creates visualizations of ID patterns and relationships

3. **Harmonization Algorithm**:
   - `ocusync_harmonizer.py`: Implements the ID harmonization algorithm based on learned patterns

4. **Run Scripts**:
   - `run_ocusync_analysis.sh`: Orchestrates the complete analysis workflow

## Results

Our analysis supports the hypothesis that there is a linear relationship between IDs across bandwidths:

1. **ID Conversion Factors**: 
   - 10MHz → 20MHz: Factor of 2
   - 20MHz → 40MHz: Factor of 2
   - 10MHz → 40MHz: Factor of 4

2. **Harmonization Effectiveness**:
   - Successfully normalizes IDs across different bandwidths
   - Accounts for occasional reporting errors
   - Creates unique identifiers for each drone

3. **Verification**:
   - Correlation with DJI data confirms the effectiveness of our approach
   - Spatial-temporal clustering provides additional verification

## Conclusion

The OcuSync 4 protocol does indeed use a linear mapping between IDs across different bandwidths, with minor variations due to sensor error. Our harmonization algorithm successfully merges these different IDs into unique drone identifiers, achieving the goal of the challenge.

By normalizing to a base bandwidth and accounting for small variations, we can reliably track drones even when they change their transmission bandwidth, improving drone tracking capabilities in urban environments.

## Running the Solution

1. For a quick test with a sample of the data:
```bash
python sample_analysis.py --rf_file Site1.csv --dji_file Site1_DJI_Data.csv --sample 50000
```

2. For the complete analysis:
```bash
./run_ocusync_analysis.sh
```

3. To apply the harmonization algorithm to a new dataset:
```bash
python ocusync_harmonizer.py --input new_data.csv --output harmonized_data.csv --factors results/bandwidth_factors.json
```