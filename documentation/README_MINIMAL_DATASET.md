# README - Emergency Health Services Accessibility Minimal Dataset

## Quick Start Guide

This minimal dataset contains all data and code necessary to reproduce the findings in our paper on emergency health services accessibility in Melbourne, Victoria.

### Requirements

**Software:**
- Python 3.8+
- Required packages: pandas, numpy, geopandas, matplotlib, seaborn, scipy, shapely, contextily

**Hardware:**
- Minimum 8GB RAM recommended
- ~500MB disk space

### Installation

```bash
# Install required packages
pip install pandas numpy geopandas matplotlib seaborn scipy shapely contextily

# Or use the requirements file (if provided)
pip install -r requirements.txt
```

### Running the Analysis

1. **Main Analysis** - Generates all figures and primary results:
   ```bash
   python Scripts/final_advance_script.py
   ```

2. **Generate Tables** - Creates the three key tables from the paper:
   ```bash
   python Scripts/access_tables.py
   ```

3. **Catchment Statistics** - Analyzes hospital service areas:
   ```bash
   python Scripts/stats_v.py
   ```

### Output Locations

- **Figures**: `Victoria/Enhanced_Analysis_Results/figures/`
- **Tables**: `Victoria/Enhanced_Analysis_Results/tables/`
- **Processed Data**: `Victoria/Enhanced_Analysis_Results/data/`
- **Reports**: `Victoria/Enhanced_Analysis_Results/reports/`

## Dataset Contents

### Input Data
- **Census Demographics** (`2021Census_G01_VIC_SA1.csv`): Age distribution by area
- **Census Income** (`2021Census_G33_VIC_SA1.csv`): Household income data
- **Hospital Locations** (`Victoria/spatial/hospital_Points.shp`): 30 hospitals
- **Mesh Blocks** (`Victoria/Melbourne/Res_melbourne/Residential_mesh_blocks.shp`): Residential areas

### Key Outputs
- **Equity Analysis**: Income-based accessibility disparities
- **Priority Areas**: Top 10 underserved mesh blocks
- **Coverage Statistics**: Population coverage by distance bands
- **Catchment Analysis**: Hospital service area demographics

## Reproducibility Notes

1. All random seeds are fixed in the scripts for reproducibility
2. Coordinate system: GDA2020 MGA Zone 55 (EPSG:7855)
3. Distance calculations use Euclidean (straight-line) distances
4. Population data is down-scaled from SA1 to mesh block level

## Variable Definitions

### Demographic Variables
- `vulnerable_population`: Residents under 4 or over 85 years old
- `low_income_pct`: Percentage of households earning <$500/week
- `priority_score`: Composite index combining distance, vulnerability, and income

### Distance Metrics
- All distances in meters
- 15-minute ambulance catchment = 15km radius
- Distance bands: 0-5km, 5-10km, 10-15km, 15-20km, >20km

## Known Limitations

1. **Straight-line distances**: Does not account for road networks
2. **2021 Census data**: May not reflect current population distribution
3. **Hospital capacity**: Analysis does not consider bed numbers or services
4. **Private hospitals**: Limited to emergency department availability

## Troubleshooting

**Common Issues:**

1. **Import Error**: Ensure all packages are installed
2. **File Not Found**: Check working directory is project root
3. **Memory Error**: Process data in chunks or increase system RAM
4. **CRS Warning**: Ignore warnings about CRS axis order

## Data Sources

- Census Data: Australian Bureau of Statistics (2021)
- Hospital Locations: Victorian Department of Health (2023)
- Mesh Blocks: Australian Bureau of Statistics (2021)

## Support

For questions about this dataset or analysis:
- Review the detailed metadata in `MINIMAL_DATASET_METADATA.md`
- Check script comments for implementation details
- Contact: [Your email]

## License and Citation

Please see `MINIMAL_DATASET_METADATA.md` for licensing information and citation guidelines.