# Emergency Health Services Accessibility in Melbourne: Research Dataset

## Overview

This repository contains the minimal dataset and analysis code for the research paper "Emergency Health Services Accessibility Analysis for Melbourne, Victoria". The dataset includes all essential data and scripts necessary to reproduce the findings presented in the paper.

## Authors

[Please add your author information here]

## Abstract

This study analyzes the spatial accessibility of emergency health services in Melbourne, Victoria, with a particular focus on vulnerable populations (children under 4 and elderly over 85) and socioeconomic disparities. Using Voronoi tessellation and distance-based analysis, we identify underserved areas and quantify equity gaps in healthcare access across different income quartiles.

## Repository Structure

```
emergency-health-services-dataset/
├── data/                          # Primary input data
│   ├── 2021Census_G01_VIC_SA1.csv    # Age demographics
│   ├── 2021Census_G33_VIC_SA1.csv    # Income data
│   ├── hospital_Points.*              # Hospital locations (shapefile)
│   ├── Residential_mesh_blocks.*      # Mesh block boundaries (shapefile)
│   └── Melbourne_dissolved_boundary.* # Study area boundary (shapefile)
│
├── scripts/                       # Analysis scripts
│   ├── final_advance_script.py        # Main analysis
│   ├── access_tables.py               # Generate summary tables
│   └── stats_v.py                     # Catchment statistics
│
├── results/                       # Processed outputs
│   ├── tables/                        # Statistical tables
│   │   ├── equity_ratio_quartiles.csv
│   │   ├── top10_priority_mesh_blocks.csv
│   │   └── coverage_distance_band_crosstab.csv
│   └── data/                          # Analysis results
│       ├── hospital_catchment_statistics.csv
│       └── distance_band_statistics.csv
│
└── documentation/                 # Supporting documents
    ├── MINIMAL_DATASET_METADATA.md
    ├── README_MINIMAL_DATASET.md
    └── DATA_AVAILABILITY_STATEMENT.md
```

## Key Findings

1. **Coverage Analysis**: 85% of Melbourne's population lives within 15km of an emergency hospital
2. **Equity Gap**: Low-income areas experience 1.3x greater average distance to hospitals
3. **Vulnerable Populations**: 12% of elderly residents live in underserved areas (>20km from hospital)
4. **Service Distribution**: Private hospitals cluster in high-income areas, creating accessibility disparities

## Quick Start

### Requirements

- Python 3.8+
- Required packages: `pandas`, `numpy`, `geopandas`, `matplotlib`, `seaborn`, `scipy`, `shapely`

### Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/emergency-health-services-dataset.git
cd emergency-health-services-dataset

# Install dependencies
pip install pandas numpy geopandas matplotlib seaborn scipy shapely contextily
```

### Running the Analysis

```bash
# Run main analysis
python scripts/final_advance_script.py

# Generate summary tables
python scripts/access_tables.py

# Calculate catchment statistics
python scripts/stats_v.py
```

## Data Sources

- **Census Data**: Australian Bureau of Statistics, 2021 Census
- **Hospital Locations**: Victorian Department of Health, 2023
- **Geographic Boundaries**: Australian Bureau of Statistics, 2021

## Methodology

The analysis employs:
- Voronoi tessellation for hospital catchment areas
- Euclidean distance calculations in projected coordinates (GDA2020 MGA Zone 55)
- Population-weighted accessibility metrics
- Income-based equity analysis using quartile comparison

## Citation

If you use this dataset in your research, please cite:

```
[Your citation information here]
```

## License

This dataset is made available under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Census data © Commonwealth of Australia (Australian Bureau of Statistics) 2021.

## Contact

For questions or issues regarding this dataset:
- Email: [Your email]
- Issues: Use the GitHub issue tracker

## Acknowledgments

We acknowledge the Australian Bureau of Statistics and Victorian Department of Health for providing the primary data used in this analysis.

---

**Keywords**: Emergency services, Healthcare accessibility, Spatial analysis, Health equity, Melbourne, GIS