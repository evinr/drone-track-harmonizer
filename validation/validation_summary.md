# Drone Track Harmonization Validation Summary

Generated on: 2025-04-26 23:53:59

## Data Overview

- Total correlations: 2543055
- Unique drones: 220
- Unique RF IDs: 68
- Unique bandwidths: 3
- Time range: 2024-07-01 14:33:57 to 2024-08-05 00:02:52

## Correlation Quality

- Correlation quality score: 91.95%
- Interpretation: **Excellent** - Very strong correlation between DJI and RF data

## Distance Statistics

- Average distance: 20080.56 meters
- Minimum distance: 45.93 meters
- Maximum distance: 12882434.67 meters

## RSSI Statistics

- Average RSSI: -17.45 dB
- Minimum RSSI: -33.88 dB
- Maximum RSSI: -1.09 dB

## Time Difference Statistics

- Average time difference: 0.01 seconds
- Minimum time difference: -10.00 seconds
- Maximum time difference: 10.00 seconds

## Bandwidth Distribution

| Bandwidth | Count | Percentage |
|-----------|-------|------------|
| 10MHz | 2337791 | 91.93% |
| 40MHz | 155842 | 6.13% |
| 20MHz | 49422 | 1.94% |

## Validation Files

The following validation files have been generated:

1. **Interactive Map**: `correlation_map.html`
2. **Harmonized GPX**: `harmonized_drones.gpx`
3. **Harmonized KML**: `harmonized_drones.kml`
4. **Visualization Plots**:
   - Drone type correlations: `drone_type_correlations.png`
   - RSSI distribution: `rssi_distribution.png`
   - Distance distribution: `distance_distribution.png`
   - Time difference distribution: `time_diff_distribution.png`
   - RSSI vs Distance: `rssi_vs_distance.png`
   - Correlations by date: `correlations_by_date.png`
   - Bandwidth distribution: `bandwidth_distribution.png`

## Next Steps

Based on the validation results, the data appears to be of good quality and ready for use in the ATAK plugin. You can proceed with confidence.

Recommended actions:
1. Use the harmonized GPX/KML files for ATAK integration
2. Implement the ID correlation formulas from the pattern stats
