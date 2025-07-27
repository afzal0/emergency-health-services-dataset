# Data Availability Statement

## Minimal Dataset Access

The minimal dataset supporting the findings of this study has been compiled according to journal requirements and includes all essential data necessary to reproduce our results. The dataset consists of:

1. **Primary Data Sources**
   - Australian Bureau of Statistics 2021 Census data (SA1 level)
   - Victorian hospital location data (30 emergency departments)
   - Melbourne residential mesh block boundaries
   
2. **Processed Data Outputs**
   - Hospital catchment area statistics
   - Distance-based population coverage analysis
   - Equity metrics by income quartile
   - Priority area identification results

3. **Analysis Code**
   - Complete Python scripts for all analyses
   - Documentation of methods and parameters
   - Requirements for software dependencies

## Data Repository

The complete minimal dataset package is available at:
[Repository URL to be added upon acceptance]

## File Organization

```
Minimal_Dataset/
├── Primary_Data/
│   ├── Census_Demographics/
│   ├── Census_Income/
│   └── Spatial_Data/
├── Processed_Results/
│   ├── Tables/
│   └── Statistical_Outputs/
├── Analysis_Scripts/
├── Documentation/
│   ├── MINIMAL_DATASET_METADATA.md
│   └── README_MINIMAL_DATASET.md
└── DATA_AVAILABILITY_STATEMENT.md
```

## Access Restrictions

- **Census Data**: Publicly available from Australian Bureau of Statistics under Creative Commons Attribution 4.0 International licence
- **Hospital Data**: Publicly available from Victorian Department of Health
- **Analysis Outputs**: Available under Creative Commons Attribution 4.0 International License (CC BY 4.0)

No restrictions apply to the data. All personally identifiable information has been removed, and data is aggregated to mesh block level to ensure privacy.

## Reproducibility

To reproduce our findings:
1. Download the minimal dataset package
2. Install required Python packages (see README)
3. Run `final_advance_script.py` for main analysis
4. Run `access_tables.py` for summary tables

Expected computation time: ~10-15 minutes on standard hardware

## Contact for Data Queries

For questions about data access or technical issues:
- Corresponding Author: Theophilus I. Emeto (theophilus.emeto@jcu.edu.au)
- Data Repository Issues: https://github.com/afzal0/emergency-health-services-dataset/issues

## Compliance Statement

This dataset complies with journal requirements for:
- Minimal dataset provision
- Removal of sensitive information
- Clear documentation and metadata
- Reproducible analysis code
- Appropriate data licensing

Last Updated: July 27, 2025