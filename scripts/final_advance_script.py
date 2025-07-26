#!/usr/bin/env python3
"""
Fixed Emergency Service Accessibility Analysis
Analyzes hospital accessibility for vulnerable populations (under 4 and over 85 years)
With corrected Voronoi file paths and proper population calculation logic
Enhanced with contour maps and bivariate visualizations
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon, Patch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
import seaborn as sns
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
# UPDATED Voronoi paths
PUBLIC_VORONOI = "./Victoria/Output-qgis/clipped-public-voronoi.shp"
PRIVATE_VORONOI = "./Victoria/Output-qgis/clipped-private-voronoi.shp"
ALL_HOSPITALS = "Victoria/Output-qgis/All-hospital.shp"
HOSPITAL_POINTS = './Victoria/spatial/hospital_Points.shp'
RESIDENTIAL_MESH = './Victoria/Melbourne/Res_melbourne/Residential_mesh_blocks.shp'
MELBOURNE_BOUNDARY = "./Victoria/spatial/Melbourne_dissolved_boundary/Melbourne_dissolved_boundary.shp"

# Census data paths
CENSUS_DEMOGRAPHICS = './2021Census_G01_VIC_SA1.csv'
CENSUS_INCOME = './2021Census_G33_VIC_SA1.csv'

# Ambulance response time configuration
AMBULANCE_CODE1_MINUTES = 15  # Code 1 response time
AMBULANCE_SPEED_KMH = 60  # Average ambulance speed in urban areas
RADIUS_KM = (AMBULANCE_CODE1_MINUTES / 60) * AMBULANCE_SPEED_KMH  # Distance in 15 minutes

# Create output directory
output_dir = './output_new/accessibility_analysis-improved_2'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'maps'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)

print(f"Output directory: {output_dir}")

# Load data
print("\nLoading spatial data...")
try:
    # Load Voronoi catchments - UPDATED paths
    voronoi_all = gpd.read_file(ALL_HOSPITALS)
    voronoi_public = gpd.read_file(PUBLIC_VORONOI)
    voronoi_private = gpd.read_file(PRIVATE_VORONOI)
except Exception as e:
    print(f"Error loading Voronoi shapefiles: {e}")
    print("Please ensure the Voronoi shapefiles exist at the specified paths.")
    sys.exit(1)

# Load hospitals
try:
    hospitals = gpd.read_file(HOSPITAL_POINTS)
except FileNotFoundError:
    print(f"Error: Hospital shapefile not found at {HOSPITAL_POINTS}")
    sys.exit(1)

# Load residential mesh blocks
try:
    residential_mesh = gpd.read_file(RESIDENTIAL_MESH)
except FileNotFoundError:
    print(f"Error: Residential mesh blocks shapefile not found at {RESIDENTIAL_MESH}")
    sys.exit(1)

# Load Melbourne boundary
try:
    melbourne_boundary = gpd.read_file(MELBOURNE_BOUNDARY)
except FileNotFoundError:
    print(f"Warning: Melbourne boundary shapefile not found at {MELBOURNE_BOUNDARY}")
    print("Proceeding without boundary.")
    melbourne_boundary = None

# Ensure all data is in the same CRS
print("Checking and harmonizing coordinate reference systems...")
crs_to_use = "EPSG:7855"  # GDA2020 / MGA zone 55 (common for Victoria)

datasets = {
    "voronoi_all": voronoi_all,
    "voronoi_public": voronoi_public,
    "voronoi_private": voronoi_private,
    "hospitals": hospitals,
    "residential_mesh": residential_mesh,
}

if melbourne_boundary is not None:
    datasets["melbourne_boundary"] = melbourne_boundary

# Reproject all datasets to the same CRS if needed
for name, dataset in datasets.items():
    if dataset.crs != crs_to_use:
        print(f"Reprojecting {name} from {dataset.crs} to {crs_to_use}")
        datasets[name] = dataset.to_crs(crs_to_use)

# Update variables with reprojected data
voronoi_all = datasets["voronoi_all"]
voronoi_public = datasets["voronoi_public"]
voronoi_private = datasets["voronoi_private"]
hospitals = datasets["hospitals"]
residential_mesh = datasets["residential_mesh"]
if melbourne_boundary is not None:
    melbourne_boundary = datasets["melbourne_boundary"]

print(f"Loaded {len(voronoi_all)} total hospital catchments")
print(f"Loaded {len(residential_mesh)} residential mesh blocks")

# Check if voronoi data has the expected columns
print("\nChecking Voronoi data structure...")
print(f"Voronoi_all columns: {voronoi_all.columns.tolist()}")
if 'type' in voronoi_all.columns:
    print(f"Hospital types in data: {voronoi_all['type'].unique()}")
elif 'Type' in voronoi_all.columns:
    print(f"Hospital types in data: {voronoi_all['Type'].unique()}")

# Load census data
print("\nLoading census data...")
try:
    census_demo = pd.read_csv(CENSUS_DEMOGRAPHICS)
    census_income = pd.read_csv(CENSUS_INCOME)
    print(f"Census demographics: {len(census_demo)} SA1 areas")
    print(f"Census income: {len(census_income)} SA1 areas")
    census_data_available = True
except FileNotFoundError as e:
    print(f"Error: Census data files not found. {e}")
    print("Please ensure the following files exist:")
    print(f"  - {CENSUS_DEMOGRAPHICS}")
    print(f"  - {CENSUS_INCOME}")
    print("Proceeding with population data from mesh blocks instead.")
    census_data_available = False

# Process census data if available
if census_data_available:
    # Merge census data
    print("Merging census datasets...")
    census_data = census_demo.merge(census_income, on='SA1_CODE_2021', how='inner')
    
    # Calculate key demographics
    census_data['total_population'] = census_data['Tot_P_P']
    census_data['pop_under_4'] = census_data['Age_0_4_yr_P']
    census_data['pop_over_85'] = census_data['Age_85ov_P']
    census_data['vulnerable_population'] = census_data['pop_under_4'] + census_data['pop_over_85']
    census_data['vulnerable_percentage'] = (census_data['vulnerable_population'] / 
                                           census_data['total_population'] * 100).fillna(0)
    
    # Calculate income brackets
    income_brackets = [
        ('low_income', ['HI_1_149_Tot', 'HI_150_299_Tot', 'HI_300_399_Tot', 'HI_400_499_Tot']),
        ('middle_income', ['HI_500_649_Tot', 'HI_650_799_Tot', 'HI_800_999_Tot', 'HI_1000_1249_Tot']),
        ('high_income', ['HI_1250_1499_Tot', 'HI_1500_1749_Tot', 'HI_1750_1999_Tot', 
                         'HI_2000_2499_Tot', 'HI_2500_2999_Tot', 'HI_3000_3499_Tot', 
                         'HI_3500_3999_Tot', 'HI_4000_more_Tot'])
    ]
    
    for bracket_name, columns in income_brackets:
        census_data[bracket_name] = census_data[columns].sum(axis=1)
    
    census_data['total_households'] = census_data['Tot_Tot']
    census_data['low_income_percentage'] = (census_data['low_income'] / 
                                            census_data['total_households'] * 100).fillna(0)
    
    print("\nMerging census data with mesh blocks...")
    
    # Find the correct SA1 field name in residential mesh
    print("Checking field names in residential mesh...")
    print(f"First few columns: {residential_mesh.columns[:10].tolist()}")
    
    sa1_field = None
    possible_fields = ['SA1_CODE21', 'SA1_CODE_21', 'SA1_CODE_2021', 'SA1_7DIG21', 'SA1_7DIGIT', 
                     'SA1_MAIN16', 'SA1_MAIN21', 'MB_CODE21', 'MB_CODE_21', 'MESHBLOCK']
    
    for field in possible_fields:
        if field in residential_mesh.columns:
            sa1_field = field
            print(f"Found SA1 field: {sa1_field}")
            break
    
    if sa1_field is None:
        # Try to find any column with SA1 in the name
        for col in residential_mesh.columns:
            if 'SA1' in col.upper():
                sa1_field = col
                print(f"Found SA1 field: {sa1_field}")
                break
    
    if sa1_field is None:
        print("Warning: Could not find SA1 field in mesh blocks.")
        print("Available columns:", residential_mesh.columns.tolist())
        
        # Check for mesh block codes
        mb_field = None
        for col in residential_mesh.columns:
            if 'MB' in col.upper() or 'MESH' in col.upper():
                mb_field = col
                print(f"\nFound mesh block field: {mb_field}")
                break
        
        print("\nContinuing with population data from shapefile.")
        census_data_available = False
    else:
        # Convert SA1 codes to string in both datasets to ensure matching
        census_data['SA1_CODE_2021'] = census_data['SA1_CODE_2021'].astype(str)
        residential_mesh[sa1_field] = residential_mesh[sa1_field].astype(str)
        
        # Merge census data with mesh blocks
        try:
            residential_mesh = residential_mesh.merge(
                census_data[['SA1_CODE_2021', 'total_population', 'pop_under_4', 'pop_over_85', 
                            'vulnerable_population', 'vulnerable_percentage', 
                            'low_income', 'middle_income', 'high_income', 
                            'total_households', 'low_income_percentage']],
                left_on=sa1_field,
                right_on='SA1_CODE_2021',
                how='left'
            )
            print("Census data merged successfully")
            
            # Remove duplicate SA1 column if it exists
            if sa1_field != 'SA1_CODE_2021' and 'SA1_CODE_2021' in residential_mesh.columns:
                residential_mesh = residential_mesh.drop('SA1_CODE_2021', axis=1)
            
            # ------------------------------------------------------------------
            # IMPORTANT FIX #1 – de-duplicate SA1 totals copied to multiple mesh-blocks
            # ------------------------------------------------------------------
            print("Applying population de-duplication to avoid inflated counts...")
            
            # How many mesh-blocks belong to each SA1 in the mesh layer?
            mb_counts = residential_mesh.groupby(sa1_field)['geometry'].transform('count')
            
            # Columns that came from the SA1 census tables
            cols_to_scale = [
                'total_population', 'pop_under_4', 'pop_over_85', 'vulnerable_population',
                'low_income', 'middle_income', 'high_income', 'total_households'
            ]
            
            for col in cols_to_scale:
                # If the column exists, down-scale it
                if col in residential_mesh.columns:
                    residential_mesh[col] = residential_mesh[col] / mb_counts
            
            # ------------------------------------------------------------------
            # IMPORTANT FIX #2 – Recalculate ALL percentage fields after scaling raw counts
            # ------------------------------------------------------------------
            # Re-compute the derived fields so they stay valid
            residential_mesh['vulnerable_population'] = (
                residential_mesh['pop_under_4'] + residential_mesh['pop_over_85']
            )
            
            # Recalculate vulnerability percentage
            residential_mesh['vulnerable_percentage'] = (
                residential_mesh['vulnerable_population'] / residential_mesh['total_population']
            ).replace([np.inf, np.nan], 0) * 100
            
            # Recalculate low income percentage
            if all(col in residential_mesh.columns for col in ['low_income', 'total_households']):
                residential_mesh['low_income_percentage'] = (
                    residential_mesh['low_income'] / residential_mesh['total_households']
                ).replace([np.inf, np.nan], 0) * 100
            
        except Exception as e:
            print(f"Warning: Failed to merge census data: {e}")
            print("Proceeding with population data from shapefile.")
            census_data_available = False

# Set field names based on available data
if census_data_available and 'total_population' in residential_mesh.columns:
    population_field = 'total_population'
    under4_field = 'pop_under_4'
    over85_field = 'pop_over_85'
    vulnerable_field = 'vulnerable_population'
    vulnerable_pct_field = 'vulnerable_percentage'
    print("Using population data from census files")
else:
    # Use fields from the shapefile if census data not available
    print("Using population data from shapefile fields")
    if 'total_popu' in residential_mesh.columns:
        population_field = 'total_popu'
    else:
        pop_fields = [col for col in residential_mesh.columns if 'pop' in col.lower()]
        if pop_fields:
            print(f"Found potential population fields: {pop_fields}")
            population_field = pop_fields[0]
        else:
            print("No population field found. Using default values.")
            residential_mesh['total_popu'] = 100  # Default population
            population_field = 'total_popu'
    
    # Check for vulnerable population fields
    if all(field in residential_mesh.columns for field in ['under4', 'over85']):
        print("Found vulnerable population fields: 'under4' and 'over85'")
        under4_field = 'under4'
        over85_field = 'over85'
        vulnerable_field = 'vulnerable'
        vulnerable_pct_field = 'vulnerable_pct'
        
        # Calculate vulnerable population if not already done
        if 'vulnerable' not in residential_mesh.columns:
            residential_mesh['vulnerable'] = residential_mesh['under4'] + residential_mesh['over85']
        
        if 'vulnerable_pct' not in residential_mesh.columns:
            residential_mesh['vulnerable_pct'] = (residential_mesh['vulnerable'] / 
                                                 residential_mesh[population_field] * 100).fillna(0)
    else:
        print("Vulnerable population fields not found. Creating default values.")
        residential_mesh['under4'] = residential_mesh[population_field] * 0.05  # Estimate 5% under 4
        residential_mesh['over85'] = residential_mesh[population_field] * 0.05  # Estimate 5% over 85
        residential_mesh['vulnerable'] = residential_mesh['under4'] + residential_mesh['over85']
        residential_mesh['vulnerable_pct'] = 10.0
        under4_field = 'under4'
        over85_field = 'over85'
        vulnerable_field = 'vulnerable'
        vulnerable_pct_field = 'vulnerable_pct'

# Fill NaN values with 0
fill_columns = [population_field, under4_field, over85_field, vulnerable_field,
                'low_income', 'middle_income', 'high_income', 'total_households', 
                'low_income_percentage', vulnerable_pct_field]

for col in fill_columns:
    if col in residential_mesh.columns:
        residential_mesh[col] = residential_mesh[col].fillna(0)

print(f"\nPopulation summary:")
print(f"Total mesh blocks: {len(residential_mesh)}")
print(f"Mesh blocks with population data: {(residential_mesh[population_field] > 0).sum()}")
print(f"Mesh blocks without population data: {(residential_mesh[population_field] == 0).sum()}")
print(f"Total population in study area: {residential_mesh[population_field].sum():,.0f}")

# 1. DISTANCE ANALYSIS
print("\n1. ANALYZING DISTANCES TO HOSPITALS...")

def calculate_distance_to_nearest_hospital(mesh_blocks, hospitals):
    """Calculate distance from each mesh block centroid to nearest hospital"""
    distances = []
    nearest_hospital = []
    hospital_types = []
    
    for idx, block in mesh_blocks.iterrows():
        centroid = block.geometry.centroid
        min_distance = float('inf')
        nearest = None
        nearest_type = None
        
        for hidx, hospital in hospitals.iterrows():
            # Calculate Euclidean distance in projection units (usually meters)
            dist = centroid.distance(hospital.geometry)
            
            # Convert to kilometers if necessary
            if mesh_blocks.crs.is_projected:
                dist_km = dist / 1000  # Convert meters to kilometers
            else:
                # Approximate conversion for geographic CRS (less accurate)
                dist_km = dist * 111.32 * np.cos(np.radians(centroid.y))
            
            if dist_km < min_distance:
                min_distance = dist_km
                nearest = hospital['Name'] if 'Name' in hospital else f"Hospital_{hidx}"
                nearest_type = hospital['Type'] if 'Type' in hospital else 'Unknown'
        
        distances.append(min_distance)
        nearest_hospital.append(nearest)
        hospital_types.append(nearest_type)
    
    mesh_blocks['distance_to_nearest_hospital_km'] = distances
    mesh_blocks['nearest_hospital'] = nearest_hospital
    mesh_blocks['nearest_hospital_type'] = hospital_types
    
    return mesh_blocks

# Check if hospital type information is available
if 'Type' in hospitals.columns:
    print(f"Hospital types found: {hospitals['Type'].unique()}")
    # Separate hospitals by type
    public_hospitals = hospitals[hospitals['Type'] == 'Public']
    private_hospitals = hospitals[hospitals['Type'] == 'Private']
    print(f"Public hospitals: {len(public_hospitals)}")
    print(f"Private hospitals: {len(private_hospitals)}")
else:
    print("Warning: Hospital type information not available. Treating all hospitals as one type.")
    public_hospitals = hospitals  # Treat all as public for analysis
    private_hospitals = pd.DataFrame()  # Empty dataframe

# Calculate overall distances
print("Calculating distances to nearest hospitals...")
residential_mesh = calculate_distance_to_nearest_hospital(residential_mesh, hospitals)

# Calculate distances to public hospitals only
if len(public_hospitals) > 0:
    print("Calculating distances to public hospitals...")
    temp_mesh = residential_mesh.copy()
    temp_mesh = calculate_distance_to_nearest_hospital(temp_mesh, public_hospitals)
    residential_mesh['distance_to_public_hospital_km'] = temp_mesh['distance_to_nearest_hospital_km']
else:
    residential_mesh['distance_to_public_hospital_km'] = np.nan

# Calculate distances to private hospitals only
if len(private_hospitals) > 0:
    print("Calculating distances to private hospitals...")
    temp_mesh = residential_mesh.copy()
    temp_mesh = calculate_distance_to_nearest_hospital(temp_mesh, private_hospitals)
    residential_mesh['distance_to_private_hospital_km'] = temp_mesh['distance_to_nearest_hospital_km']
else:
    residential_mesh['distance_to_private_hospital_km'] = np.nan

# 2. VORONOI CATCHMENT ANALYSIS
print("\n2. ANALYZING VORONOI CATCHMENTS...")

# Find the hospital name column (might be truncated due to shapefile limitations)
hospital_name_col = None
for col in voronoi_all.columns:
    if 'hosp' in col.lower() and 'name' in col.lower():
        hospital_name_col = col
        break
    elif col.lower() in ['name', 'hospital_n', 'hosp_name']:
        hospital_name_col = col
        break

if hospital_name_col is None:
    print("Warning: Could not find hospital name column in Voronoi data")
    print("Available columns:", voronoi_all.columns.tolist())
    # Try to proceed without hospital names
    hospital_name_col = 'hospital_id'  # fallback
    voronoi_all[hospital_name_col] = range(len(voronoi_all))

print(f"Using hospital name column: {hospital_name_col}")

# Spatial join mesh blocks with Voronoi cells
print("Assigning mesh blocks to hospital catchments...")
mesh_with_catchment = gpd.sjoin(residential_mesh, voronoi_all, how='left', predicate='within')

print(f"Spatial join complete. Result shape: {mesh_with_catchment.shape}")

# Find the hospital name column in the joined data (might have _right suffix)
hospital_name_col_joined = hospital_name_col
if hospital_name_col not in mesh_with_catchment.columns:
    # Check for suffix versions
    if f"{hospital_name_col}_right" in mesh_with_catchment.columns:
        hospital_name_col_joined = f"{hospital_name_col}_right"
    else:
        # Find any column with hospital/name
        for col in mesh_with_catchment.columns:
            if 'hosp' in col.lower() and 'name' in col.lower() and col != hospital_name_col:
                hospital_name_col_joined = col
                break

print(f"Using joined hospital name column: {hospital_name_col_joined}")
print(f"Number of mesh blocks with catchment assignment: {mesh_with_catchment[hospital_name_col_joined].notna().sum()}")

# Calculate statistics for each catchment
catchment_stats = []

# Get unique hospital names from the joined column
unique_hospitals = mesh_with_catchment[hospital_name_col_joined].dropna().unique()
print(f"Found {len(unique_hospitals)} unique hospital catchments")

for hospital_name in unique_hospitals:
    catchment_mesh = mesh_with_catchment[mesh_with_catchment[hospital_name_col_joined] == hospital_name]
    
    # Check for potential population duplication
    if len(catchment_mesh) > 0:
        # Handle potentially missing columns gracefully
        stats = {
            'hospital_name': hospital_name,
            'num_mesh_blocks': len(catchment_mesh),
            'total_population': catchment_mesh[population_field].sum(),
            'pop_under_4': catchment_mesh[under4_field].sum(),
            'pop_over_85': catchment_mesh[over85_field].sum(),
            'vulnerable_population': catchment_mesh[vulnerable_field].sum(),
            'avg_distance_km': catchment_mesh['distance_to_nearest_hospital_km'].mean() if 'distance_to_nearest_hospital_km' in catchment_mesh.columns else 0,
            'max_distance_km': catchment_mesh['distance_to_nearest_hospital_km'].max() if 'distance_to_nearest_hospital_km' in catchment_mesh.columns else 0,
        }
        
        # Add income data if available
        if 'low_income' in catchment_mesh.columns:
            stats['low_income_households'] = catchment_mesh['low_income'].sum()
        if 'middle_income' in catchment_mesh.columns:
            stats['middle_income_households'] = catchment_mesh['middle_income'].sum()
        if 'high_income' in catchment_mesh.columns:
            stats['high_income_households'] = catchment_mesh['high_income'].sum()
        if 'total_households' in catchment_mesh.columns:
            stats['total_households'] = catchment_mesh['total_households'].sum()
        if 'low_income_percentage' in catchment_mesh.columns:
            stats['avg_low_income_percentage'] = catchment_mesh['low_income_percentage'].mean()
            
        catchment_stats.append(stats)

catchment_stats_df = pd.DataFrame(catchment_stats)

# Check for population duplication in catchment statistics
if len(catchment_stats_df) > 0:
    catchment_total_pop = catchment_stats_df['total_population'].sum()
    residential_total_pop = residential_mesh[population_field].sum()
    
    print(f"Calculated statistics for {len(catchment_stats_df)} catchments")
    print(f"Total population across all catchments: {catchment_total_pop:,.0f}")
    print(f"Total population in residential blocks: {residential_total_pop:,.0f}")
    
    # If there's significant population inflation, apply a correction factor
    if catchment_total_pop > (residential_total_pop * 1.1):
        print(f"WARNING: Population inflation detected in catchment statistics")
        print(f"Applying global correction factor")
        
        correction_factor = residential_total_pop / catchment_total_pop
        catchment_stats_df['total_population'] *= correction_factor
        catchment_stats_df['pop_under_4'] *= correction_factor
        catchment_stats_df['pop_over_85'] *= correction_factor
        catchment_stats_df['vulnerable_population'] *= correction_factor
        
        print(f"After correction: Total catchment population = {catchment_stats_df['total_population'].sum():,.0f}")
else:
    print("Warning: No catchment statistics calculated. Check spatial join.")

# 3. RADIUS CATCHMENT ANALYSIS (15-minute ambulance response)
print(f"\n3. ANALYZING {AMBULANCE_CODE1_MINUTES}-MINUTE AMBULANCE CATCHMENTS...")

# Create buffer around all hospitals
hospital_buffers = hospitals.copy()

# Create buffers in projected coordinates (usually meters)
if hospitals.crs.is_projected:  # Fixed from hospital_points to hospitals
    buffer_distance = RADIUS_KM * 1000  # Convert km to meters
    hospital_buffers['geometry'] = hospitals.geometry.buffer(buffer_distance)
else:
    # Approximate buffer in degrees if not projected
    buffer_degrees = RADIUS_KM / (111.32 * np.cos(np.radians(hospitals.geometry.y.mean())))
    hospital_buffers['geometry'] = hospitals.geometry.buffer(buffer_degrees)

# Create buffers for public hospitals
if len(public_hospitals) > 0:
    public_buffers = public_hospitals.copy()
    if public_hospitals.crs.is_projected:
        public_buffers['geometry'] = public_hospitals.geometry.buffer(RADIUS_KM * 1000)
    else:
        public_buffers['geometry'] = public_hospitals.geometry.buffer(buffer_degrees)

# Create buffers for private hospitals
if len(private_hospitals) > 0:
    private_buffers = private_hospitals.copy()
    if private_hospitals.crs.is_projected:
        private_buffers['geometry'] = private_hospitals.geometry.buffer(RADIUS_KM * 1000)
    else:
        private_buffers['geometry'] = private_hospitals.geometry.buffer(buffer_degrees)

# Find mesh blocks within ambulance response radius - ALL hospitals
residential_mesh['within_ambulance_radius'] = False
for idx, buffer in hospital_buffers.iterrows():
    within = residential_mesh.geometry.within(buffer.geometry)
    residential_mesh.loc[within, 'within_ambulance_radius'] = True

# Find mesh blocks within radius - PUBLIC hospitals
residential_mesh['within_public_radius'] = False
if len(public_hospitals) > 0:
    for idx, buffer in public_buffers.iterrows():
        within = residential_mesh.geometry.within(buffer.geometry)
        residential_mesh.loc[within, 'within_public_radius'] = True

# Find mesh blocks within radius - PRIVATE hospitals
residential_mesh['within_private_radius'] = False
if len(private_hospitals) > 0:
    for idx, buffer in private_buffers.iterrows():
        within = residential_mesh.geometry.within(buffer.geometry)
        residential_mesh.loc[within, 'within_private_radius'] = True

# Calculate coverage statistics
total_pop = residential_mesh[population_field].sum()
covered_pop = residential_mesh[residential_mesh['within_ambulance_radius']][population_field].sum()
coverage_percentage = (covered_pop / total_pop * 100) if total_pop > 0 else 0

vulnerable_total = residential_mesh[vulnerable_field].sum()
vulnerable_covered = residential_mesh[residential_mesh['within_ambulance_radius']][vulnerable_field].sum()
vulnerable_coverage = (vulnerable_covered / vulnerable_total * 100) if vulnerable_total > 0 else 0

# Public hospital coverage
public_covered_pop = residential_mesh[residential_mesh['within_public_radius']][population_field].sum()
public_coverage_percentage = (public_covered_pop / total_pop * 100) if total_pop > 0 else 0
public_vulnerable_covered = residential_mesh[residential_mesh['within_public_radius']][vulnerable_field].sum()
public_vulnerable_coverage = (public_vulnerable_covered / vulnerable_total * 100) if vulnerable_total > 0 else 0

# Private hospital coverage
private_covered_pop = residential_mesh[residential_mesh['within_private_radius']][population_field].sum()
private_coverage_percentage = (private_covered_pop / total_pop * 100) if total_pop > 0 else 0
private_vulnerable_covered = residential_mesh[residential_mesh['within_private_radius']][vulnerable_field].sum()
private_vulnerable_coverage = (private_vulnerable_covered / vulnerable_total * 100) if vulnerable_total > 0 else 0

print(f"\nOVERALL COVERAGE:")
print(f"Population within {AMBULANCE_CODE1_MINUTES}-minute response: {covered_pop:,.0f} ({coverage_percentage:.1f}%)")
print(f"Vulnerable population covered: {vulnerable_covered:,.0f} ({vulnerable_coverage:.1f}%)")

print(f"\nPUBLIC HOSPITAL COVERAGE:")
print(f"Population covered: {public_covered_pop:,.0f} ({public_coverage_percentage:.1f}%)")
print(f"Vulnerable population covered: {public_vulnerable_covered:,.0f} ({public_vulnerable_coverage:.1f}%)")

print(f"\nPRIVATE HOSPITAL COVERAGE:")
print(f"Population covered: {private_covered_pop:,.0f} ({private_coverage_percentage:.1f}%)")
print(f"Vulnerable population covered: {private_vulnerable_covered:,.0f} ({private_vulnerable_coverage:.1f}%)")

# 4. IDENTIFY UNDERSERVED AREAS
print("\n4. IDENTIFYING UNDERSERVED AREAS...")

# Areas outside ambulance radius
underserved_mesh = residential_mesh[~residential_mesh['within_ambulance_radius']].copy()
underserved_mesh = underserved_mesh[underserved_mesh[population_field] > 0]

# Sort by vulnerable population
underserved_mesh = underserved_mesh.sort_values(vulnerable_field, ascending=False)

print(f"Found {len(underserved_mesh)} populated mesh blocks outside {AMBULANCE_CODE1_MINUTES}-minute ambulance radius")
print(f"Total underserved population: {underserved_mesh[population_field].sum():,.0f}")
print(f"Total underserved vulnerable population: {underserved_mesh[vulnerable_field].sum():,.0f}")

# Create priority score for underserved areas
underserved_mesh['priority_score'] = (
    (underserved_mesh[vulnerable_field] / underserved_mesh[vulnerable_field].max()) * 0.5 +
    (underserved_mesh['distance_to_nearest_hospital_km'] / underserved_mesh['distance_to_nearest_hospital_km'].max()) * 0.3 +
    (underserved_mesh[population_field] / underserved_mesh[population_field].max()) * 0.2
)

# Identify high priority areas (top 10%)
high_priority_threshold = underserved_mesh['priority_score'].quantile(0.9)
high_priority_areas = underserved_mesh[underserved_mesh['priority_score'] >= high_priority_threshold]

print(f"High priority underserved areas: {len(high_priority_areas)}")
print(f"Population in high priority areas: {high_priority_areas[population_field].sum():,.0f}")
print(f"Vulnerable population in high priority areas: {high_priority_areas[vulnerable_field].sum():,.0f}")

# IMPORTANT FIX: Build coverage categories on original data for report generation
residential_mesh['coverage_type'] = 'No Coverage'
residential_mesh.loc[residential_mesh['within_private_radius'] & 
                     ~residential_mesh['within_public_radius'], 
                     'coverage_type'] = 'Private Only'
residential_mesh.loc[~residential_mesh['within_private_radius'] & 
                     residential_mesh['within_public_radius'], 
                     'coverage_type'] = 'Public Only'
residential_mesh.loc[residential_mesh['within_private_radius'] & 
                     residential_mesh['within_public_radius'], 
                     'coverage_type'] = 'Both'

print(f"\nCoverage type distribution:")
print(residential_mesh['coverage_type'].value_counts())

# 5. CREATE VISUALIZATIONS (Updated Version with Lat/Lon)
print("\n5. CREATING VISUALIZATIONS...")

# Set up color schemes
vuln_cmap = LinearSegmentedColormap.from_list(
    'vulnerability_cmap', 
    ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026']
)

distance_cmap = LinearSegmentedColormap.from_list(
    'distance_cmap',
    ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d73027']
)

# Convert all data to WGS84 (lat/lon) for display if needed
if residential_mesh.crs.is_projected:
    print("Converting to WGS84 for map display...")
    residential_mesh_wgs84 = residential_mesh.to_crs('EPSG:4326')
    hospitals_wgs84 = hospitals.to_crs('EPSG:4326')
    hospital_buffers_wgs84 = hospital_buffers.to_crs('EPSG:4326')
    if melbourne_boundary is not None:
        melbourne_boundary_wgs84 = melbourne_boundary.to_crs('EPSG:4326')
    if len(public_hospitals) > 0:
        public_hospitals_wgs84 = public_hospitals.to_crs('EPSG:4326')
    if len(private_hospitals) > 0:
        private_hospitals_wgs84 = private_hospitals.to_crs('EPSG:4326')
    if len(underserved_mesh) > 0:
        underserved_mesh_wgs84 = underserved_mesh.to_crs('EPSG:4326')
    if 'public_buffers' in locals() and len(public_buffers) > 0:
        public_buffers_wgs84 = public_buffers.to_crs('EPSG:4326')
    if 'private_buffers' in locals() and len(private_buffers) > 0:
        private_buffers_wgs84 = private_buffers.to_crs('EPSG:4326')
else:
    residential_mesh_wgs84 = residential_mesh
    hospitals_wgs84 = hospitals
    hospital_buffers_wgs84 = hospital_buffers
    if melbourne_boundary is not None:
        melbourne_boundary_wgs84 = melbourne_boundary
    if len(public_hospitals) > 0:
        public_hospitals_wgs84 = public_hospitals
    if len(private_hospitals) > 0:
        private_hospitals_wgs84 = private_hospitals
    if len(underserved_mesh) > 0:
        underserved_mesh_wgs84 = underserved_mesh
    if 'public_buffers' in locals() and len(public_buffers) > 0:
        public_buffers_wgs84 = public_buffers
    if 'private_buffers' in locals() and len(private_buffers) > 0:
        private_buffers_wgs84 = private_buffers

# Map 1: Vulnerable Population Density
fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
ax.set_facecolor('#FAFAFA')

# Plot Melbourne boundary if available
if melbourne_boundary is not None:
    melbourne_boundary_wgs84.boundary.plot(ax=ax, color='#333333', linewidth=2, zorder=4)

# Plot mesh blocks colored by vulnerable population percentage
mesh_plot = residential_mesh_wgs84.copy()
mesh_plot = mesh_plot[mesh_plot[population_field] > 0]

# Create normalization for vulnerability percentage
min_vuln = mesh_plot[vulnerable_pct_field].min()
max_vuln = mesh_plot[vulnerable_pct_field].max()
print(f"Vulnerability percentage range: {min_vuln:.2f}% to {max_vuln:.2f}%")

norm_max = min(max_vuln, 30)  # Cap at 30% or actual max
norm = Normalize(vmin=0, vmax=norm_max)

mesh_plot.plot(column=vulnerable_pct_field, 
               ax=ax, 
               cmap=vuln_cmap, 
               norm=norm,
               edgecolor='none',
               legend=True,
               legend_kwds={'label': 'Vulnerable Population %', 
                           'orientation': 'vertical',
                           'shrink': 0.7,
                           'extend': 'max',
                           'format': '%.1f%%'})

# Plot hospitals
hospitals_wgs84.plot(ax=ax, color='darkblue', markersize=100, marker='h', 
               edgecolor='white', linewidth=1.5, zorder=5, label='Hospitals')

ax.set_title('Vulnerable Population Distribution and Hospital Locations', 
             fontsize=16, pad=20, weight='bold')
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

# Format tick labels
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}°'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.3f}°'))
plt.xticks(rotation=45, ha='right')

# Add subtle grid
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Add north arrow
x_pos = ax.get_xlim()[1] - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05
y_pos = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1
ax.annotate('N', xy=(x_pos, y_pos), xytext=(x_pos, y_pos - 0.01),
            arrowprops=dict(arrowstyle='->', lw=2), 
            ha='center', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'maps', 'vulnerable_population_map.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Map 2: Distance to Nearest Hospital
fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
ax.set_facecolor('#FAFAFA')

# Plot Melbourne boundary if available
if melbourne_boundary is not None:
    melbourne_boundary_wgs84.boundary.plot(ax=ax, color='#333333', linewidth=2, zorder=4)

mesh_plot.plot(column='distance_to_nearest_hospital_km', 
               ax=ax, 
               cmap=distance_cmap, 
               edgecolor='none',
               legend=True,
               legend_kwds={'label': 'Distance to Hospital (km)', 
                           'orientation': 'vertical',
                           'shrink': 0.7,
                           'format': '%.1f'})

hospitals_wgs84.plot(ax=ax, color='darkblue', markersize=100, marker='h', 
               edgecolor='white', linewidth=1.5, zorder=5, label='Hospitals')

ax.set_title('Distance to Nearest Hospital', fontsize=16, pad=20, weight='bold')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

# Format tick labels
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}°'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.3f}°'))
plt.xticks(rotation=45, ha='right')

# Add grid
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)

# Add north arrow
x_pos = ax.get_xlim()[1] - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05
y_pos = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1
ax.annotate('N', xy=(x_pos, y_pos), xytext=(x_pos, y_pos - 0.01),
            arrowprops=dict(arrowstyle='->', lw=2), 
            ha='center', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'maps', 'distance_to_hospital_map.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Map 3: Underserved Areas (Cleaner version)
fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
ax.set_facecolor('#FAFAFA')

# Plot Melbourne boundary if available
if melbourne_boundary is not None:
    melbourne_boundary_wgs84.boundary.plot(ax=ax, color='#333333', linewidth=2, zorder=4)

# Plot all residential blocks in very light gray
residential_mesh_wgs84.plot(ax=ax, color='#f5f5f5', edgecolor='none', alpha=0.5)

# Highlight underserved areas with vulnerable population count
if len(underserved_mesh_wgs84) > 0:
    # Use vulnerable population for coloring
    underserved_mesh_wgs84.plot(column=vulnerable_field, 
                         ax=ax, 
                         cmap='Reds',  # Simple red gradient
                         edgecolor='none',  # No edges for cleaner look
                         alpha=0.8,
                         legend=True,
                         legend_kwds={'label': 'Vulnerable Population', 
                                     'orientation': 'vertical',
                                     'shrink': 0.7,
                                     'format': '%.0f'})

# Plot hospitals
hospitals_wgs84.plot(ax=ax, color='darkblue', markersize=100, marker='h', 
               edgecolor='white', linewidth=1.5, zorder=5, label='Hospitals')

# Plot ambulance radius boundaries with very light lines
for idx, buffer in hospital_buffers_wgs84.iterrows():
    ax.plot(*buffer.geometry.exterior.xy, color='green', linewidth=1, 
             linestyle='--', alpha=0.3, zorder=2)

ax.set_title(f'Underserved Areas - Outside {AMBULANCE_CODE1_MINUTES}-minute Emergency Response', 
             fontsize=16, pad=20, weight='bold')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

# Format tick labels
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}°'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.3f}°'))
plt.xticks(rotation=45, ha='right')

# Add grid
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)

# Add subtle annotation
ax.text(
    0.02, 0.98, 
    f'{AMBULANCE_CODE1_MINUTES}-min response radius', 
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='top',
    color='green',
    alpha=0.7,
    weight='light'
)

# Add north arrow
x_pos = ax.get_xlim()[1] - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05
y_pos = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1
ax.annotate('N', xy=(x_pos, y_pos), xytext=(x_pos, y_pos - 0.01),
            arrowprops=dict(arrowstyle='->', lw=2), 
            ha='center', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'maps', 'underserved_areas_map.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Map 4: Public vs Private Hospital Coverage Comparison
fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
ax.set_facecolor('#FAFAFA')

# Plot Melbourne boundary if available
if melbourne_boundary is not None:
    melbourne_boundary_wgs84.boundary.plot(ax=ax, color='#333333', linewidth=2, zorder=4)

# Build coverage categories (need to rebuild for WGS84 data)
residential_mesh_wgs84['coverage_type'] = 'No Coverage'
residential_mesh_wgs84.loc[residential_mesh_wgs84['within_private_radius'] &
                     ~residential_mesh_wgs84['within_public_radius'],
                     'coverage_type'] = 'Private Only'
residential_mesh_wgs84.loc[~residential_mesh_wgs84['within_private_radius'] &
                     residential_mesh_wgs84['within_public_radius'],
                     'coverage_type'] = 'Public Only'
residential_mesh_wgs84.loc[residential_mesh_wgs84['within_private_radius'] &
                     residential_mesh_wgs84['within_public_radius'],
                     'coverage_type'] = 'Both'

coverage_colors = {
    'No Coverage' : '#E57373',  # Light red
    'Private Only': '#FFB74D',  # Orange
    'Public Only' : '#64B5F6',  # Light blue
    'Both'        : '#81C784'   # Light green
}

# Plot coverage polygons
for cov_type, color in coverage_colors.items():
    subset = residential_mesh_wgs84[residential_mesh_wgs84['coverage_type'] == cov_type]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor='none', alpha=0.8)

# Plot hospitals by type
if len(public_hospitals) > 0:
    public_hospitals_wgs84.plot(ax=ax, color='#1976D2', marker='s',
                          markersize=80, edgecolor='white', linewidth=2, zorder=5)

if len(private_hospitals) > 0:
    private_hospitals_wgs84.plot(ax=ax, color='#D32F2F', marker='^',
                           markersize=80, edgecolor='white', linewidth=2, zorder=5)

# Custom legend
coverage_patches = [
    Patch(facecolor='#E57373', label='No Coverage'),
    Patch(facecolor='#FFB74D', label='Private Only'),
    Patch(facecolor='#64B5F6', label='Public Only'),
    Patch(facecolor='#81C784', label='Both')
]

hospital_handles = [
    Line2D([0], [0], marker='s', color='none',
           markerfacecolor='#1976D2', markersize=10, label='Public Hospitals'),
    Line2D([0], [0], marker='^', color='none',
           markerfacecolor='#D32F2F', markersize=10, label='Private Hospitals')
]

ax.set_title('Emergency Service Coverage by Hospital Type', fontsize=16, pad=20, weight='bold')
ax.legend(handles=coverage_patches + hospital_handles,
          loc='upper right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

# Format tick labels
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}°'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.3f}°'))
plt.xticks(rotation=45, ha='right')

# Add grid
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Add north arrow
x_pos = ax.get_xlim()[1] - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05
y_pos = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1
ax.annotate('N', xy=(x_pos, y_pos), xytext=(x_pos, y_pos - 0.01),
            arrowprops=dict(arrowstyle='->', lw=2), 
            ha='center', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'maps', 'coverage_by_hospital_type.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Map 5: Underserved Areas with Distance Contours
print("Creating underserved areas map with distance contours...")

fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
ax.set_facecolor('#FAFAFA')

# Plot Melbourne boundary if available
if melbourne_boundary is not None:
    melbourne_boundary_wgs84.boundary.plot(ax=ax, color='#333333', linewidth=2.5, zorder=10)

# Plot all residential blocks in a light color
residential_mesh_wgs84[residential_mesh_wgs84['within_ambulance_radius']].plot(
    ax=ax,
    color='#e5f5e0',  # Light green for served areas
    edgecolor='none',
    alpha=0.4,
    label='Within 15-minute Response'
)

# Plot underserved areas with vulnerable population coloring
if len(underserved_mesh_wgs84) > 0:
    underserved_mesh_wgs84.plot(
        column=vulnerable_field,
        ax=ax,
        cmap='Reds',
        edgecolor='none',
        alpha=0.8,
        legend=True,
        legend_kwds={
            'label': 'Vulnerable Population',
            'orientation': 'vertical',
            'shrink': 0.7,
            'format': '%.0f'
        }
    )

# Create distance contours using the original projected data for accuracy
contours = []
contour_distances = [5, 10, 15, 20, 25, 30]
for i, distance in enumerate(contour_distances):
    # Find blocks within a distance range
    if i == 0:
        contour_blocks = residential_mesh[
            residential_mesh['distance_to_nearest_hospital_km'] <= distance
        ]
    else:
        contour_blocks = residential_mesh[
            (residential_mesh['distance_to_nearest_hospital_km'] > contour_distances[i-1]) & 
            (residential_mesh['distance_to_nearest_hospital_km'] <= distance)
        ]
    
    if len(contour_blocks) > 0:
        # Create a boundary around all blocks in this distance range
        boundary = unary_union(contour_blocks.geometry).boundary
        # Convert to WGS84 for display
        boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=residential_mesh.crs)
        boundary_gdf_wgs84 = boundary_gdf.to_crs('EPSG:4326')
        contours.append((distance, boundary_gdf_wgs84))

# Plot contour lines
contour_colors = ['#2b8cbe', '#7bccc4', '#bae4bc', '#f0f9ff', '#fee0d2', '#fc9272', '#de2d26']
for i, (distance, boundary_gdf) in enumerate(contours):
    if i < len(contour_colors):
        color = contour_colors[i]
    else:
        color = '#de2d26'  # Default for any additional contours
    
    boundary_gdf.plot(
        ax=ax,
        color=color,
        linewidth=2,
        linestyle='-',
        alpha=0.7,
        label=f'{distance} km'
    )

# Plot hospitals
hospitals_wgs84.plot(
    ax=ax,
    color='darkblue',
    markersize=100,
    marker='h',
    edgecolor='white',
    linewidth=1.5,
    zorder=10,
    label='Hospital'
)

# Add title and labels
ax.set_title('Underserved Areas with Distance Contours', fontsize=18, pad=20, weight='bold')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Format tick labels
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}°'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.3f}°'))
plt.xticks(rotation=45, ha='right')

# Legend
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)

# Add north arrow
x_pos = ax.get_xlim()[1] - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05
y_pos = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1
ax.annotate('N', xy=(x_pos, y_pos), xytext=(x_pos, y_pos - 0.01),
            arrowprops=dict(arrowstyle='->', lw=2), 
            ha='center', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'maps', 'underserved_areas_with_contours.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Bivariate Maps
print("\nCreating bivariate maps...")

# Helper function to create bivariate color scheme
def create_bivariate_colormap():
    """Create a 3x3 bivariate color scheme from green to red"""
    # Define colors for a 3x3 grid (low-low to high-high)
    colors = [
        ['#e8e8e8', '#ace4aa', '#5ac864'],  # Low Y (bottom row)
        ['#fdb863', '#dfb89b', '#85c67c'],  # Medium Y (middle row)
        ['#e66101', '#d85b3a', '#ca4323']   # High Y (top row)
    ]
    return colors

def classify_bivariate_data(data1, data2, n_classes=3):
    """Classify two variables into a bivariate scheme"""
    # Calculate quantiles for both variables
    quantiles1 = np.quantile(data1[data1 > 0], np.linspace(0, 1, n_classes + 1))
    quantiles2 = np.quantile(data2[data2 > 0], np.linspace(0, 1, n_classes + 1))
    
    # Classify each variable
    class1 = np.digitize(data1, quantiles1[1:-1])
    class2 = np.digitize(data2, quantiles2[1:-1])
    
    # Combine into bivariate classes (0-8 for 3x3)
    bivariate_class = class2 * n_classes + class1
    
    return bivariate_class, quantiles1, quantiles2

# Prepare data for bivariate maps
mesh_plot_bv = residential_mesh_wgs84[residential_mesh_wgs84[population_field] > 0].copy()

# Calculate population density (population per square km)
# First calculate area in square kilometers
if residential_mesh.crs.is_projected:
    mesh_plot_bv['area_sqkm'] = residential_mesh[residential_mesh[population_field] > 0].geometry.area / 1e6
else:
    # For geographic CRS, use a rough approximation
    mesh_plot_bv['area_sqkm'] = mesh_plot_bv.geometry.area * (111.32 ** 2)

mesh_plot_bv['pop_density'] = mesh_plot_bv[population_field] / mesh_plot_bv['area_sqkm']

# Get color scheme (define once and reuse)
colors = create_bivariate_colormap()
color_map = {}
for i in range(3):
    for j in range(3):
        color_map[i * 3 + j] = colors[i][j]

# Map 6: Bivariate Map - Population Density vs Income
if 'low_income_percentage' in mesh_plot_bv.columns:
    print("Creating bivariate map: Population Density vs Low Income Percentage...")
    
    # Create figure with tight layout
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
    ax.set_facecolor('#FAFAFA')
    
    # Plot Melbourne boundary
    if melbourne_boundary is not None:
        melbourne_boundary_wgs84.boundary.plot(ax=ax, color='#333333', linewidth=2, zorder=4)
    
    # Create bivariate classification
    bv_class, q_density, q_income = classify_bivariate_data(
        mesh_plot_bv['pop_density'].values,
        mesh_plot_bv['low_income_percentage'].values
    )
    mesh_plot_bv['bv_class'] = bv_class
    
    # Plot each bivariate class
    for class_num, color in color_map.items():
        subset = mesh_plot_bv[mesh_plot_bv['bv_class'] == class_num]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor='none', alpha=0.8)
    
    # Plot hospitals
    hospitals_wgs84.plot(ax=ax, color='darkblue', markersize=60, marker='h', 
                   edgecolor='white', linewidth=1, zorder=5)
    
    # Title and labels
    ax.set_title('Population Density vs Low Income Percentage', fontsize=16, pad=20, weight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Format tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}°'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.3f}°'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Create inset axes for legend in the bottom right corner of the map
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    legend_ax = inset_axes(ax, width="20%", height="20%", loc='lower right', 
                          bbox_to_anchor=(0.25, 0, 1.05, 1), bbox_transform=ax.transAxes)
    legend_ax.set_xlim(0, 3)
    legend_ax.set_ylim(0, 3)
    
    # Draw the color grid
    for i in range(3):
        for j in range(3):
            rect = Rectangle((j, i), 1, 1, facecolor=colors[i][j], edgecolor='white', linewidth=1)
            legend_ax.add_patch(rect)
    
    # Remove ticks and spines
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)
    
    # Add compact labels
    legend_ax.text(1.5, 3.3, 'Pop. Density vs Low Income %', ha='center', va='bottom', 
                   fontsize=10, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Bottom labels
    legend_ax.text(0.5, -0.3, 'Low', ha='center', va='top', fontsize=8)
    legend_ax.text(2.5, -0.3, 'High', ha='center', va='top', fontsize=8)
    legend_ax.text(1.5, -0.5, 'Pop. Density →', ha='center', va='top', fontsize=8, style='italic')
    
    # Left labels  
    legend_ax.text(-0.3, 0.5, 'Low', ha='right', va='center', fontsize=8, rotation=90)
    legend_ax.text(-0.3, 2.5, 'High', ha='right', va='center', fontsize=8, rotation=90)
    legend_ax.text(-0.5, 1.5, 'Income % →', ha='center', va='center', fontsize=8, style='italic', rotation=90)
    
    # Add north arrow
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    x_pos = ax.get_xlim()[1] - x_range * 0.05
    y_pos = ax.get_ylim()[1] - y_range * 0.05
    ax.annotate('N', xy=(x_pos, y_pos), xytext=(x_pos, y_pos - y_range * 0.02),
                arrowprops=dict(arrowstyle='->', lw=2), 
                ha='center', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'maps', 'bivariate_density_income.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# Map 7: Bivariate Map - Population Density vs Vulnerable Population
print("Creating bivariate map: Population Density vs Vulnerable Population Percentage...")

fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
ax.set_facecolor('#FAFAFA')

# Plot Melbourne boundary
if melbourne_boundary is not None:
    melbourne_boundary_wgs84.boundary.plot(ax=ax, color='#333333', linewidth=2, zorder=4)

# Create bivariate classification
bv_class, q_density, q_vulnerable = classify_bivariate_data(
    mesh_plot_bv['pop_density'].values,
    mesh_plot_bv[vulnerable_pct_field].values
)
mesh_plot_bv['bv_class_2'] = bv_class

# Plot each bivariate class
for class_num, color in color_map.items():
    subset = mesh_plot_bv[mesh_plot_bv['bv_class_2'] == class_num]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor='none', alpha=0.8)

# Plot hospitals
hospitals_wgs84.plot(ax=ax, color='darkblue', markersize=60, marker='h', 
               edgecolor='white', linewidth=1, zorder=5)

# Title and labels
ax.set_title('Population Density vs Vulnerable Population Percentage', fontsize=16, pad=20, weight='bold')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Format tick labels
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}°'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.3f}°'))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Create inset axes for legend
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
legend_ax = inset_axes(ax, width="20%", height="20%", loc='lower right', 
                      bbox_to_anchor=(0.25, 0, 1.05, 1), bbox_transform=ax.transAxes)
legend_ax.set_xlim(0, 3)
legend_ax.set_ylim(0, 3)

# Draw the color grid
for i in range(3):
    for j in range(3):
        rect = Rectangle((j, i), 1, 1, facecolor=colors[i][j], edgecolor='white', linewidth=1)
        legend_ax.add_patch(rect)

# Remove ticks and spines
legend_ax.set_xticks([])
legend_ax.set_yticks([])
for spine in legend_ax.spines.values():
    spine.set_visible(False)

# Add compact labels
legend_ax.text(1.5, 3.3, 'Pop. Density vs Vulnerable %', ha='center', va='bottom', 
               fontsize=10, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Bottom labels
legend_ax.text(0.5, -0.3, 'Low', ha='center', va='top', fontsize=8)
legend_ax.text(2.5, -0.3, 'High', ha='center', va='top', fontsize=8)
legend_ax.text(1.5, -0.5, 'Pop. Density →', ha='center', va='top', fontsize=8, style='italic')

# Left labels  
legend_ax.text(-0.3, 0.5, 'Low', ha='right', va='center', fontsize=8, rotation=90)
legend_ax.text(-0.3, 2.5, 'High', ha='right', va='center', fontsize=8, rotation=90)
legend_ax.text(-0.5, 1.5, 'Vulnerable % →', ha='center', va='center', fontsize=8, style='italic', rotation=90)

# Add north arrow
x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
x_pos = ax.get_xlim()[1] - x_range * 0.05
y_pos = ax.get_ylim()[1] - y_range * 0.05
ax.annotate('N', xy=(x_pos, y_pos), xytext=(x_pos, y_pos - y_range * 0.02),
            arrowprops=dict(arrowstyle='->', lw=2), 
            ha='center', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'maps', 'bivariate_density_vulnerable.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Map 8: Bivariate Map - Vulnerable Population vs Income
if 'low_income_percentage' in mesh_plot_bv.columns:
    print("Creating bivariate map: Vulnerable Population Percentage vs Low Income Percentage...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
    ax.set_facecolor('#FAFAFA')
    
    # Plot Melbourne boundary
    if melbourne_boundary is not None:
        melbourne_boundary_wgs84.boundary.plot(ax=ax, color='#333333', linewidth=2, zorder=4)
    
    # Create bivariate classification
    bv_class, q_vulnerable, q_income = classify_bivariate_data(
        mesh_plot_bv[vulnerable_pct_field].values,
        mesh_plot_bv['low_income_percentage'].values
    )
    mesh_plot_bv['bv_class_3'] = bv_class
    
    # Plot each bivariate class
    for class_num, color in color_map.items():
        subset = mesh_plot_bv[mesh_plot_bv['bv_class_3'] == class_num]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor='none', alpha=0.8)
    
    # Plot hospitals
    hospitals_wgs84.plot(ax=ax, color='darkblue', markersize=60, marker='h', 
                   edgecolor='white', linewidth=1, zorder=5)
    
    # Title and labels
    ax.set_title('Vulnerable Population Percentage vs Low Income Percentage', fontsize=16, pad=20, weight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Format tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}°'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.3f}°'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Create inset axes for legend
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    legend_ax = inset_axes(ax, width="20%", height="20%", loc='lower right', 
                          bbox_to_anchor=(0.25, 0, 1.05, 1), bbox_transform=ax.transAxes)
    legend_ax.set_xlim(0, 3)
    legend_ax.set_ylim(0, 3)
    
    # Draw the color grid
    for i in range(3):
        for j in range(3):
            rect = Rectangle((j, i), 1, 1, facecolor=colors[i][j], edgecolor='white', linewidth=1)
            legend_ax.add_patch(rect)
    
    # Remove ticks and spines
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)
    
    # Add compact labels
    legend_ax.text(1.5, 3.3, 'Vulnerable % vs Low Income %', ha='center', va='bottom', 
                   fontsize=10, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Bottom labels
    legend_ax.text(0.5, -0.3, 'Low', ha='center', va='top', fontsize=8)
    legend_ax.text(2.5, -0.3, 'High', ha='center', va='top', fontsize=8)
    legend_ax.text(1.5, -0.5, 'Vulnerable % →', ha='center', va='top', fontsize=8, style='italic')
    
    # Left labels  
    legend_ax.text(-0.3, 0.5, 'Low', ha='right', va='center', fontsize=8, rotation=90)
    legend_ax.text(-0.3, 2.5, 'High', ha='right', va='center', fontsize=8, rotation=90)
    legend_ax.text(-0.5, 1.5, 'Income % →', ha='center', va='center', fontsize=8, style='italic', rotation=90)
    
    # Add north arrow
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    x_pos = ax.get_xlim()[1] - x_range * 0.05
    y_pos = ax.get_ylim()[1] - y_range * 0.05
    ax.annotate('N', xy=(x_pos, y_pos), xytext=(x_pos, y_pos - y_range * 0.02),
                arrowprops=dict(arrowstyle='->', lw=2), 
                ha='center', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'maps', 'bivariate_vulnerable_income.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

print("All bivariate maps created successfully with properly positioned inset legends!")

# 6. GENERATE DETAILED REPORT
print("\n6. GENERATING ENHANCED REPORT...")

# Main summary report
report_path = os.path.join(output_dir, 'reports', 'accessibility_analysis_report.txt')
with open(report_path, 'w') as f:
    f.write("ENHANCED EMERGENCY SERVICE ACCESSIBILITY ANALYSIS\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*70 + "\n\n")
    
    f.write("EXECUTIVE SUMMARY\n")
    f.write("-"*70 + "\n")
    f.write(f"Total Population Analyzed: {total_pop:,.0f}\n")
    f.write(f"Vulnerable Population (Under 4 & Over 85): {vulnerable_total:,.0f} ({vulnerable_total/total_pop*100:.1f}%)\n")
    f.write(f"  - Under 4 years: {residential_mesh[under4_field].sum():,.0f}\n")
    f.write(f"  - Over 85 years: {residential_mesh[over85_field].sum():,.0f}\n\n")
    
    f.write(f"AMBULANCE COVERAGE ({AMBULANCE_CODE1_MINUTES}-MINUTE RESPONSE TIME)\n")
    f.write("-"*70 + "\n")
    f.write(f"Response radius: {RADIUS_KM:.1f} km (at {AMBULANCE_SPEED_KMH} km/h average speed)\n\n")
    
    f.write("ALL HOSPITALS:\n")
    f.write(f"Population Covered: {covered_pop:,.0f} ({coverage_percentage:.1f}%)\n")
    f.write(f"Population Not Covered: {total_pop - covered_pop:,.0f} ({100-coverage_percentage:.1f}%)\n")
    f.write(f"Vulnerable Population Covered: {vulnerable_covered:,.0f} ({vulnerable_coverage:.1f}%)\n")
    f.write(f"Vulnerable Population Not Covered: {vulnerable_total - vulnerable_covered:,.0f} ({100-vulnerable_coverage:.1f}%)\n\n")
    
    f.write("PUBLIC HOSPITALS ONLY:\n")
    f.write(f"Population Covered: {public_covered_pop:,.0f} ({public_coverage_percentage:.1f}%)\n")
    f.write(f"Vulnerable Population Covered: {public_vulnerable_covered:,.0f} ({public_vulnerable_coverage:.1f}%)\n\n")
    
    f.write("PRIVATE HOSPITALS ONLY:\n")
    f.write(f"Population Covered: {private_covered_pop:,.0f} ({private_coverage_percentage:.1f}%)\n")
    f.write(f"Vulnerable Population Covered: {private_vulnerable_covered:,.0f} ({private_vulnerable_coverage:.1f}%)\n\n")
    
    f.write("DISTANCE ANALYSIS\n")
    f.write("-"*70 + "\n")
    f.write("ALL HOSPITALS:\n")
    f.write(f"Average Distance to Nearest Hospital: {residential_mesh['distance_to_nearest_hospital_km'].mean():.2f} km\n")
    f.write(f"Maximum Distance to Nearest Hospital: {residential_mesh['distance_to_nearest_hospital_km'].max():.2f} km\n")
    f.write(f"Mesh blocks > 10km from hospital: {(residential_mesh['distance_to_nearest_hospital_km'] > 10).sum()}\n")
    f.write(f"Mesh blocks > 20km from hospital: {(residential_mesh['distance_to_nearest_hospital_km'] > 20).sum()}\n\n")
    
    if 'distance_to_public_hospital_km' in residential_mesh.columns and not residential_mesh['distance_to_public_hospital_km'].isna().all():
        f.write("PUBLIC HOSPITALS:\n")
        f.write(f"Average Distance: {residential_mesh['distance_to_public_hospital_km'].mean():.2f} km\n")
        f.write(f"Maximum Distance: {residential_mesh['distance_to_public_hospital_km'].max():.2f} km\n")
        f.write(f"Mesh blocks > 20km: {(residential_mesh['distance_to_public_hospital_km'] > 20).sum()}\n\n")
    
    if 'distance_to_private_hospital_km' in residential_mesh.columns and not residential_mesh['distance_to_private_hospital_km'].isna().all():
        f.write("PRIVATE HOSPITALS:\n")
        f.write(f"Average Distance: {residential_mesh['distance_to_private_hospital_km'].mean():.2f} km\n")
        f.write(f"Maximum Distance: {residential_mesh['distance_to_private_hospital_km'].max():.2f} km\n")
        f.write(f"Mesh blocks > 20km: {(residential_mesh['distance_to_private_hospital_km'] > 20).sum()}\n\n")
    
    f.write("HOSPITAL CATCHMENT ANALYSIS (VORONOI CELLS)\n")
    f.write("-"*70 + "\n")
    
    if len(catchment_stats_df) > 0:
        f.write(f"Number of Hospital Catchments: {len(catchment_stats_df)}\n")
        f.write(f"Average Population per Catchment: {catchment_stats_df['total_population'].mean():,.0f}\n")
        f.write(f"Average Vulnerable Population per Catchment: {catchment_stats_df['vulnerable_population'].mean():,.0f}\n\n")
        
        f.write("Top 5 Catchments by Vulnerable Population:\n")
        top_vulnerable = catchment_stats_df.nlargest(5, 'vulnerable_population')
        for idx, row in top_vulnerable.iterrows():
            f.write(f"  Catchment {row['hospital_name']}: {row['vulnerable_population']:,.0f} vulnerable residents\n")
    else:
        f.write("No catchment statistics available (spatial join may have failed)\n")
    
    f.write("\nUNDERSERVED AREAS ANALYSIS\n")
    f.write("-"*70 + "\n")
    f.write(f"Total Underserved Areas: {len(underserved_mesh)}\n")
    f.write(f"Total Underserved Population: {underserved_mesh[population_field].sum():,.0f}\n")
    f.write(f"Total Underserved Vulnerable Population: {underserved_mesh[vulnerable_field].sum():,.0f}\n\n")
    
    f.write("HIGH PRIORITY UNDERSERVED AREAS\n")
    f.write(f"High Priority Areas (Top 10% by Priority Score): {len(high_priority_areas)}\n")
    f.write(f"Population in High Priority Areas: {high_priority_areas[population_field].sum():,.0f}\n")
    f.write(f"Vulnerable Population in High Priority Areas: {high_priority_areas[vulnerable_field].sum():,.0f}\n")
    f.write(f"Average Distance to Hospital in High Priority Areas: {high_priority_areas['distance_to_nearest_hospital_km'].mean():.2f} km\n\n")
    
    f.write("SOCIO-ECONOMIC ANALYSIS\n")
    f.write("-"*70 + "\n")
    
    if 'low_income_percentage' in residential_mesh.columns:
        low_income_areas = residential_mesh[residential_mesh['low_income_percentage'] > 50]
        f.write(f"Mesh blocks with >50% low-income households: {len(low_income_areas)}\n")
        f.write(f"Population in high low-income areas: {low_income_areas[population_field].sum():,.0f}\n")
        
        # Low-income areas outside ambulance radius
        if 'low_income_percentage' in underserved_mesh.columns:
            low_income_underserved = underserved_mesh[underserved_mesh['low_income_percentage'] > 50]
            f.write(f"\nLow-income areas outside ambulance radius: {len(low_income_underserved)}\n")
            f.write(f"Population affected: {low_income_underserved[population_field].sum():,.0f}\n")
    else:
        f.write("Income data not available for socio-economic analysis\n")
    
    f.write("\nHOSPITAL TYPE COMPARISON\n")
    f.write("-"*70 + "\n")
    f.write("PUBLIC VS PRIVATE COVERAGE:\n")
    f.write(f"Public only coverage: {public_coverage_percentage - (residential_mesh[residential_mesh['within_public_radius'] & residential_mesh['within_private_radius']][population_field].sum() / total_pop * 100):.1f}%\n")
    f.write(f"Private only coverage: {private_coverage_percentage - (residential_mesh[residential_mesh['within_public_radius'] & residential_mesh['within_private_radius']][population_field].sum() / total_pop * 100):.1f}%\n")
    f.write(f"Overlapping coverage: {(residential_mesh[residential_mesh['within_public_radius'] & residential_mesh['within_private_radius']][population_field].sum() / total_pop * 100):.1f}%\n")
    f.write(f"No coverage: {100 - coverage_percentage:.1f}%\n\n")
    
    f.write("POPULATION DISTRIBUTION BY COVERAGE TYPE:\n")
    coverage_distribution = residential_mesh.groupby('coverage_type')[population_field].sum()
    for coverage_type, pop in coverage_distribution.items():
        f.write(f"  {coverage_type}: {pop:,.0f} people ({pop/total_pop*100:.1f}%)\n")
    f.write("\n")
    
    f.write("VULNERABLE POPULATION BY COVERAGE TYPE:\n")
    vulnerable_distribution = residential_mesh.groupby('coverage_type')[vulnerable_field].sum()
    for coverage_type, vuln_pop in vulnerable_distribution.items():
        f.write(f"  {coverage_type}: {vuln_pop:,.0f} people ({vuln_pop/vulnerable_total*100:.1f}%)\n")
    f.write("\n")
    
    f.write("\nKEY FINDINGS AND RECOMMENDATIONS\n")
    f.write("-"*70 + "\n")
    
    # Generate insights based on the analysis
    insights = []
    
    if coverage_percentage < 80:
        insights.append(f"⚠️  Only {coverage_percentage:.1f}% of population is within {AMBULANCE_CODE1_MINUTES}-minute ambulance response time")
    
    if vulnerable_coverage < 85:
        insights.append(f"⚠️  Only {vulnerable_coverage:.1f}% of vulnerable population is within {AMBULANCE_CODE1_MINUTES}-minute ambulance response time")
    
    if len(high_priority_areas) > 0:
        insights.append(f"⚠️  {len(high_priority_areas)} high priority areas identified with {high_priority_areas[population_field].sum():,.0f} total population")
    
    # Identify critical underserved areas
    if vulnerable_pct_field in underserved_mesh.columns and 'distance_to_nearest_hospital_km' in underserved_mesh.columns:
        critical_areas = underserved_mesh[
            (underserved_mesh[vulnerable_pct_field] > 10) & 
            (underserved_mesh['distance_to_nearest_hospital_km'] > 15)
        ]
        if len(critical_areas) > 0:
            insights.append(f"⚠️  CRITICAL: {len(critical_areas)} areas have high vulnerable population AND are >15km from hospital")
            insights.append(f"   Total population in critical areas: {critical_areas[population_field].sum():,.0f}")
    
    # Public vs private coverage comparison
    public_only_pct = public_coverage_percentage - (residential_mesh[residential_mesh['within_public_radius'] & residential_mesh['within_private_radius']][population_field].sum() / total_pop * 100)
    private_only_pct = private_coverage_percentage - (residential_mesh[residential_mesh['within_public_radius'] & residential_mesh['within_private_radius']][population_field].sum() / total_pop * 100)
    overlap_pct = (residential_mesh[residential_mesh['within_public_radius'] & residential_mesh['within_private_radius']][population_field].sum() / total_pop * 100)
    
    if public_only_pct > private_only_pct * 2:
        insights.append(f"Public hospitals provide significantly more unique coverage ({public_only_pct:.1f}%) than private hospitals ({private_only_pct:.1f}%)")
    elif private_only_pct > public_only_pct * 2:
        insights.append(f"Private hospitals provide significantly more unique coverage ({private_only_pct:.1f}%) than public hospitals ({public_only_pct:.1f}%)")
    
    if overlap_pct < 20:
        insights.append(f"Low overlap between public and private hospital coverage ({overlap_pct:.1f}%) suggests complementary service areas")
    elif overlap_pct > 50:
        insights.append(f"High overlap between public and private hospital coverage ({overlap_pct:.1f}%) suggests potential service duplication")
    
    # Add recommendations based on insights
    recommendations = [
        "1. Establish new emergency service points in high priority underserved areas to improve coverage",
        "2. Increase mobile emergency response units in areas beyond the 15-minute response radius",
        "3. Develop telehealth capabilities for remote assessment in areas far from hospitals",
        "4. Implement community paramedicine programs in underserved areas",
        "5. Create patient transfer agreements between hospitals to balance service loads",
        f"6. Focus on improving access for the {underserved_mesh[vulnerable_field].sum():,.0f} vulnerable individuals outside response range"
    ]
    
    # Write insights and recommendations
    f.write("INSIGHTS:\n")
    for insight in insights:
        f.write(f"- {insight}\n")
    
    f.write("\nRECOMMENDATIONS:\n")
    for recommendation in recommendations:
        f.write(f"{recommendation}\n")
    
    f.write("\n\nMETHODOLOGY NOTES\n")
    f.write("-"*70 + "\n")
    f.write("This analysis combined multiple spatial datasets to evaluate emergency healthcare accessibility:\n")
    f.write("- Voronoi polygons to define theoretical hospital service areas\n")
    f.write(f"- {AMBULANCE_CODE1_MINUTES}-minute ambulance response buffer ({RADIUS_KM:.1f} km) based on {AMBULANCE_SPEED_KMH} km/h average speed\n")
    f.write("- Distance calculations from residential areas to nearest hospitals\n")
    f.write("- Vulnerability analysis based on population under 4 and over 85 years old\n")
    f.write("- Priority scoring incorporating vulnerability, distance, and population density\n")
    f.write("- Separate analysis of public and private hospital coverage\n")
    f.write("- Bivariate analysis showing relationships between demographic and socio-economic factors\n\n")
    
    f.write("LIMITATIONS:\n")
    f.write("- Analysis assumes straight-line distances; actual road network travel times may vary\n")
    f.write("- Ambulance speed may vary by time of day and traffic conditions\n")
    f.write("- Voronoi polygons represent theoretical catchments that may differ from actual service patterns\n")
    f.write("- Population data is allocated to mesh blocks which may not perfectly represent actual residence patterns\n")

# Save detailed data files
print("\n7. SAVING DATA FILES...")

# Save catchment statistics
if len(catchment_stats_df) > 0:
    catchment_stats_df.to_csv(os.path.join(output_dir, 'data', 'catchment_statistics.csv'), index=False)
else:
    print("Warning: No catchment statistics to save")

# Save public/private hospital analysis
hospital_type_analysis = pd.DataFrame({
    'metric': ['Total Population', 'Covered by Any Hospital', 'Covered by Public Only', 
               'Covered by Private Only', 'Covered by Both', 'Not Covered',
               'Vulnerable Population Total', 'Vulnerable Covered (Any)', 
               'Vulnerable Covered (Public)', 'Vulnerable Covered (Private)'],
    'value': [
        total_pop,
        covered_pop,
        residential_mesh[(residential_mesh['within_public_radius']) & (~residential_mesh['within_private_radius'])][population_field].sum(),
        residential_mesh[(~residential_mesh['within_public_radius']) & (residential_mesh['within_private_radius'])][population_field].sum(),
        residential_mesh[residential_mesh['within_public_radius'] & residential_mesh['within_private_radius']][population_field].sum(),
        total_pop - covered_pop,
        vulnerable_total,
        vulnerable_covered,
        public_vulnerable_covered,
        private_vulnerable_covered
    ],
    'percentage': [
        100,
        coverage_percentage,
        (residential_mesh[(residential_mesh['within_public_radius']) & (~residential_mesh['within_private_radius'])][population_field].sum() / total_pop * 100) if total_pop > 0 else 0,
        (residential_mesh[(~residential_mesh['within_public_radius']) & (residential_mesh['within_private_radius'])][population_field].sum() / total_pop * 100) if total_pop > 0 else 0,
        (residential_mesh[residential_mesh['within_public_radius'] & residential_mesh['within_private_radius']][population_field].sum() / total_pop * 100) if total_pop > 0 else 0,
        100 - coverage_percentage,
        100,
        vulnerable_coverage,
        public_vulnerable_coverage,
        private_vulnerable_coverage
    ]
})

hospital_type_analysis.to_csv(os.path.join(output_dir, 'data', 'hospital_type_coverage_analysis.csv'), index=False)

# Save underserved areas data
# Build column list based on what's available
underserved_cols = [population_field, under4_field, over85_field, vulnerable_field, vulnerable_pct_field, 
                   'distance_to_nearest_hospital_km', 'priority_score']

underserved_summary = underserved_mesh[underserved_cols].copy()
underserved_summary.to_csv(os.path.join(output_dir, 'data', 'underserved_areas.csv'), index=False)

# Save GeoDataFrame of underserved areas
underserved_mesh.to_file(os.path.join(output_dir, 'data', 'underserved_areas.shp'))

# Save high priority areas
high_priority_areas.to_csv(os.path.join(output_dir, 'data', 'high_priority_areas.csv'), index=False)
high_priority_areas.to_file(os.path.join(output_dir, 'data', 'high_priority_areas.shp'))

print(f"\nAnalysis complete! Enhanced results saved to: {output_dir}")
print("\nKey files generated:")
print(f"  - Enhanced Report: {os.path.join(output_dir, 'reports', 'accessibility_analysis_report.txt')}")
print(f"  - Enhanced Maps: {os.path.join(output_dir, 'maps')}/*.png")
print(f"    - Vulnerable population distribution map")
print(f"    - Distance to hospital map") 
print(f"    - Underserved areas map")
print(f"    - Coverage by hospital type map")
print(f"    - Underserved areas with contours map")
print(f"    - Bivariate maps:")
print(f"      * Population density vs low income percentage")
print(f"      * Population density vs vulnerable population percentage")
print(f"      * Vulnerable population vs low income percentage")
print(f"  - Data: {os.path.join(output_dir, 'data')}/*.csv and *.shp")

print("\n" + "="*70)
print("ENHANCED ANALYSIS COMPLETE!")
print("="*70)