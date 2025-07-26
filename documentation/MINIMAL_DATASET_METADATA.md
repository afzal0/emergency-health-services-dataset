# Minimal Dataset for Emergency Health Services Accessibility Study

## Overview
This minimal dataset contains all essential data necessary to reproduce the findings presented in "Emergency Services Accessibility Analysis for Melbourne, Victoria". The dataset includes primary data, processed outputs, analysis scripts, and comprehensive documentation.

## Dataset Components

### 1. Primary Input Data

#### Census Data (Australian Bureau of Statistics 2021)
- **2021Census_G01_VIC_SA1.csv** (4.7 MB)
  - Description: Age demographics by Statistical Area Level 1 (SA1)
  - Key Variables:
    - `SA1_CODE_2021`: Unique SA1 identifier
    - `Tot_P_P`: Total population count
    - `Age_0_4_yr_P`: Population aged 0-4 years
    - `Age_85ov_P`: Population aged 85 years and over
  - Format: CSV with headers
  - Source: Australian Bureau of Statistics Census 2021

- **2021Census_G33_VIC_SA1.csv** (2.2 MB)
  - Description: Household income data by SA1
  - Key Variables:
    - `SA1_CODE_2021`: Unique SA1 identifier
    - `HI_1_149_Tot`: Households with weekly income $1-$149
    - `HI_150_299_Tot`: Households with weekly income $150-$299
    - `HI_300_399_Tot`: Households with weekly income $300-$399
    - `HI_400_499_Tot`: Households with weekly income $400-$499
    - `Tot_Tot`: Total household count
  - Format: CSV with headers
  - Source: Australian Bureau of Statistics Census 2021

#### Spatial Data
- **Victoria/spatial/hospital_Points.shp** (with .dbf, .prj, .shx, .cpg)
  - Description: Hospital locations in Melbourne
  - Features: 30 hospitals
  - Key Attributes:
    - `Name`: Hospital name
    - `Type`: Public or Private
    - `Latitude`: WGS84 latitude
    - `Longitude`: WGS84 longitude
  - Coordinate System: GDA94 (EPSG:4283)

- **Victoria/Melbourne/Res_melbourne/Residential_mesh_blocks.shp** (with associated files)
  - Description: Residential mesh blocks in Melbourne metropolitan area
  - Features: 59,483 mesh blocks
  - Key Attributes:
    - `MB_CODE20`: Mesh block identifier
    - `SA1_CODE20`: Statistical Area 1 code
    - `Area_SQM`: Area in square meters
  - Coordinate System: GDA94 (EPSG:4283)

- **Victoria/spatial/Melbourne_dissolved_boundary/Melbourne_dissolved_boundary.shp**
  - Description: Study area boundary
  - Features: 1 multipolygon
  - Coordinate System: GDA94 (EPSG:4283)

### 2. Processed Output Data

#### Statistical Tables (Victoria/Enhanced_Analysis_Results/tables/)
- **equity_ratio_quartiles.csv**
  - Description: Equity analysis by income quartiles
  - Variables:
    - `Income_Quartile`: Q1 (lowest) to Q4 (highest)
    - `Avg_Distance_to_Hospital`: Mean distance in meters
    - `Equity_Ratio`: Ratio compared to Q4 baseline
    - `Population_Count`: Total population in quartile

- **top10_priority_mesh_blocks.csv**
  - Description: High-priority areas for intervention
  - Variables:
    - `MB_CODE`: Mesh block identifier
    - `Priority_Score`: Composite priority index (0-100)
    - `Vulnerable_Pop`: Count of vulnerable residents
    - `Distance_to_Hospital`: Meters to nearest hospital
    - `Low_Income_Pct`: Percentage of low-income households

- **coverage_distance_band_crosstab.csv**
  - Description: Population coverage by distance bands
  - Variables:
    - `Distance_Band`: 0-5km, 5-10km, 10-15km, 15-20km, >20km
    - `Public_Hospital_Coverage`: Population with public hospital access
    - `Private_Hospital_Coverage`: Population with private hospital access
    - `Any_Hospital_Coverage`: Total coverage

#### Catchment Analysis (Victoria/Enhanced_Analysis_Results/data/)
- **hospital_catchment_statistics.csv**
  - Description: Voronoi catchment area statistics per hospital
  - Variables:
    - `index_right`: Hospital ID
    - `total_popu`: Total catchment population
    - `under4`: Population under 4 years
    - `over85`: Population over 85 years
    - `vulnerable`: Total vulnerable population
    - `mesh_block_count`: Number of mesh blocks served
    - `hospital_type`: Public or Private

- **distance_band_statistics.csv**
  - Description: Population distribution by distance from hospitals
  - Variables:
    - `distance_band`: Distance range (e.g., "0-5km")
    - `total_population`: Population in band
    - `vulnerable_population`: Vulnerable population in band
    - `percentage_of_total`: Percentage of total population

### 3. Analysis Scripts

#### Main Analysis Script
- **Scripts/final_advance_script.py**
  - Description: Primary analysis script generating all key findings
  - Functions:
    - Voronoi tessellation for hospital catchments
    - Distance analysis to nearest hospitals
    - Vulnerable population mapping
    - Equity analysis by income levels
    - Generation of all figures and tables in manuscript

#### Supporting Scripts
- **Scripts/access_tables.py**
  - Description: Generates the three key tables for the paper
  - Output: Equity ratios, priority areas, coverage statistics

- **Scripts/stats_v.py**
  - Description: Statistical analysis of Voronoi catchments
  - Output: Catchment comparison between public/private hospitals

### 4. Data Processing Notes

#### Population Down-scaling
- SA1-level census data is down-scaled to mesh blocks
- Each mesh block receives an equal share of its SA1's population
- Prevents double-counting in aggregations

#### Distance Calculations
- All distances computed as Euclidean (straight-line)
- Projected to GDA2020 MGA Zone 55 (EPSG:7855) for accuracy
- 15-minute ambulance radius assumed at 60 km/h = 15 km

#### Vulnerable Population Definition
- Under 4 years: Higher emergency service needs
- Over 85 years: Increased health service utilization
- Combined as "vulnerable population" metric

## Data Quality and Limitations

1. **Temporal Coverage**: All data from 2021 Australian Census
2. **Spatial Resolution**: Mesh block level (finest available)
3. **Hospital Data**: Current as of 2023, public hospitals only
4. **Distance Metric**: Straight-line distance (not road network)
5. **Privacy**: No individual-level data included

## File Structure
```
/
├── 2021Census_G01_VIC_SA1.csv
├── 2021Census_G33_VIC_SA1.csv
├── Scripts/
│   ├── final_advance_script.py
│   ├── access_tables.py
│   └── stats_v.py
└── Victoria/
    ├── spatial/
    │   └── hospital_Points.shp (.dbf, .prj, .shx, .cpg)
    ├── Melbourne/
    │   └── Res_melbourne/
    │       └── Residential_mesh_blocks.shp (+ files)
    └── Enhanced_Analysis_Results/
        ├── data/
        │   ├── hospital_catchment_statistics.csv
        │   └── distance_band_statistics.csv
        └── tables/
            ├── equity_ratio_quartiles.csv
            ├── top10_priority_mesh_blocks.csv
            └── coverage_distance_band_crosstab.csv
```

## Citation
If using this dataset, please cite:
[Your paper citation here]

## License
Census data: © Commonwealth of Australia (Australian Bureau of Statistics) 2021
Analysis outputs: [Your preferred license]

## Contact
[Your contact information]