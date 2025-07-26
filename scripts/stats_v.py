#!/usr/bin/env python3
"""
Emergency Service Accessibility Analysis with Enhanced Voronoi Analysis
Analyzes hospital accessibility for vulnerable populations (under 4 and over 85 years)
Includes comprehensive Voronoi catchment analysis for public, private, and all hospitals
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
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
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
AMBULANCE_CODE1_MINUTES = 15
AMBULANCE_SPEED_KMH = 60
RADIUS_KM = (AMBULANCE_CODE1_MINUTES / 60) * AMBULANCE_SPEED_KMH

# Create output directory
output_dir = './output_new/accessibility_analysis_enhanced'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'maps'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)

print(f"Output directory: {output_dir}")

# Load spatial data
print("\nLoading spatial data...")
try:
    voronoi_all = gpd.read_file(ALL_HOSPITALS)
    voronoi_public = gpd.read_file(PUBLIC_VORONOI)
    voronoi_private = gpd.read_file(PRIVATE_VORONOI)
    hospitals = gpd.read_file(HOSPITAL_POINTS)
    residential_mesh = gpd.read_file(RESIDENTIAL_MESH)
    melbourne_boundary = gpd.read_file(MELBOURNE_BOUNDARY) if os.path.exists(MELBOURNE_BOUNDARY) else None
except Exception as e:
    print(f"Error loading shapefiles: {e}")
    sys.exit(1)

# Ensure all data is in the same CRS
print("Harmonizing coordinate reference systems...")
crs_to_use = "EPSG:7855"  # GDA2020 / MGA zone 55

datasets = {
    "voronoi_all": voronoi_all,
    "voronoi_public": voronoi_public,
    "voronoi_private": voronoi_private,
    "hospitals": hospitals,
    "residential_mesh": residential_mesh,
}

if melbourne_boundary is not None:
    datasets["melbourne_boundary"] = melbourne_boundary

for name, dataset in datasets.items():
    if dataset.crs != crs_to_use:
        print(f"Reprojecting {name} to {crs_to_use}")
        datasets[name] = dataset.to_crs(crs_to_use)

voronoi_all = datasets["voronoi_all"]
voronoi_public = datasets["voronoi_public"]
voronoi_private = datasets["voronoi_private"]
hospitals = datasets["hospitals"]
residential_mesh = datasets["residential_mesh"]
if melbourne_boundary is not None:
    melbourne_boundary = datasets["melbourne_boundary"]

print(f"Loaded {len(voronoi_all)} total hospital catchments")
print(f"Loaded {len(residential_mesh)} residential mesh blocks")
print(f"Loaded {len(hospitals)} hospitals")

# Load and process census data
print("\nLoading census data...")
census_data_available = False
try:
    census_demo = pd.read_csv(CENSUS_DEMOGRAPHICS)
    census_income = pd.read_csv(CENSUS_INCOME)
    census_data = census_demo.merge(census_income, on='SA1_CODE_2021', how='inner')
    
    # Calculate demographics
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
    
    # Find SA1 field in mesh blocks
    sa1_field = None
    possible_fields = ['SA1_CODE21', 'SA1_CODE_21', 'SA1_CODE_2021', 'SA1_7DIG21', 'SA1_7DIGIT']
    
    for field in possible_fields:
        if field in residential_mesh.columns:
            sa1_field = field
            break
    
    if sa1_field is not None:
        census_data['SA1_CODE_2021'] = census_data['SA1_CODE_2021'].astype(str)
        residential_mesh[sa1_field] = residential_mesh[sa1_field].astype(str)
        
        residential_mesh = residential_mesh.merge(
            census_data[['SA1_CODE_2021', 'total_population', 'pop_under_4', 'pop_over_85', 
                        'vulnerable_population', 'vulnerable_percentage', 
                        'low_income', 'middle_income', 'high_income', 
                        'total_households', 'low_income_percentage']],
            left_on=sa1_field,
            right_on='SA1_CODE_2021',
            how='left'
        )
        
        # De-duplicate population counts
        mb_counts = residential_mesh.groupby(sa1_field)['geometry'].transform('count')
        cols_to_scale = ['total_population', 'pop_under_4', 'pop_over_85', 'vulnerable_population',
                        'low_income', 'middle_income', 'high_income', 'total_households']
        
        for col in cols_to_scale:
            if col in residential_mesh.columns:
                residential_mesh[col] = residential_mesh[col] / mb_counts
        
        # Recalculate percentages
        residential_mesh['vulnerable_population'] = (
            residential_mesh['pop_under_4'] + residential_mesh['pop_over_85']
        )
        residential_mesh['vulnerable_percentage'] = (
            residential_mesh['vulnerable_population'] / residential_mesh['total_population']
        ).replace([np.inf, np.nan], 0) * 100
        
        if 'low_income' in residential_mesh.columns and 'total_households' in residential_mesh.columns:
            residential_mesh['low_income_percentage'] = (
                residential_mesh['low_income'] / residential_mesh['total_households']
            ).replace([np.inf, np.nan], 0) * 100
        
        census_data_available = True
        print("Census data merged successfully")
    else:
        print("Warning: Could not find SA1 field in mesh blocks")
        
except Exception as e:
    print(f"Warning: Failed to load census data: {e}")

# Set field names
if census_data_available and 'total_population' in residential_mesh.columns:
    population_field = 'total_population'
    under4_field = 'pop_under_4'
    over85_field = 'pop_over_85'
    vulnerable_field = 'vulnerable_population'
    vulnerable_pct_field = 'vulnerable_percentage'
else:
    population_field = 'total_popu' if 'total_popu' in residential_mesh.columns else 'population'
    residential_mesh['under4'] = residential_mesh[population_field] * 0.05
    residential_mesh['over85'] = residential_mesh[population_field] * 0.05
    residential_mesh['vulnerable'] = residential_mesh['under4'] + residential_mesh['over85']
    residential_mesh['vulnerable_pct'] = 10.0
    under4_field = 'under4'
    over85_field = 'over85'
    vulnerable_field = 'vulnerable'
    vulnerable_pct_field = 'vulnerable_pct'

# Fill NaN values
fill_columns = [population_field, under4_field, over85_field, vulnerable_field, vulnerable_pct_field]
for col in fill_columns:
    if col in residential_mesh.columns:
        residential_mesh[col] = residential_mesh[col].fillna(0)

print(f"\nTotal population in study area: {residential_mesh[population_field].sum():,.0f}")
print(f"Total vulnerable population: {residential_mesh[vulnerable_field].sum():,.0f}")

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
            dist = centroid.distance(hospital.geometry)
            dist_km = dist / 1000 if mesh_blocks.crs.is_projected else dist * 111.32 * np.cos(np.radians(centroid.y))
            
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

# Separate hospitals by type
if 'Type' in hospitals.columns:
    public_hospitals = hospitals[hospitals['Type'] == 'Public']
    private_hospitals = hospitals[hospitals['Type'] == 'Private']
    print(f"Public hospitals: {len(public_hospitals)}")
    print(f"Private hospitals: {len(private_hospitals)}")
else:
    public_hospitals = hospitals
    private_hospitals = gpd.GeoDataFrame()

# Calculate distances
residential_mesh = calculate_distance_to_nearest_hospital(residential_mesh, hospitals)

if len(public_hospitals) > 0:
    temp_mesh = residential_mesh.copy()
    temp_mesh = calculate_distance_to_nearest_hospital(temp_mesh, public_hospitals)
    residential_mesh['distance_to_public_hospital_km'] = temp_mesh['distance_to_nearest_hospital_km']

if len(private_hospitals) > 0:
    temp_mesh = residential_mesh.copy()
    temp_mesh = calculate_distance_to_nearest_hospital(temp_mesh, private_hospitals)
    residential_mesh['distance_to_private_hospital_km'] = temp_mesh['distance_to_nearest_hospital_km']

# 2. RADIUS CATCHMENT ANALYSIS (moved before Voronoi to calculate coverage)
print(f"\n2. ANALYZING {AMBULANCE_CODE1_MINUTES}-MINUTE AMBULANCE CATCHMENTS...")

# Create buffers
buffer_distance = RADIUS_KM * 1000 if hospitals.crs.is_projected else RADIUS_KM / 111.32

hospital_buffers = hospitals.copy()
hospital_buffers['geometry'] = hospitals.geometry.buffer(buffer_distance)

# Find mesh blocks within radius
residential_mesh['within_ambulance_radius'] = False
for idx, buffer in hospital_buffers.iterrows():
    within = residential_mesh.geometry.within(buffer.geometry)
    residential_mesh.loc[within, 'within_ambulance_radius'] = True

# Public hospital coverage
residential_mesh['within_public_radius'] = False
if len(public_hospitals) > 0:
    public_buffers = public_hospitals.copy()
    public_buffers['geometry'] = public_hospitals.geometry.buffer(buffer_distance)
    for idx, buffer in public_buffers.iterrows():
        within = residential_mesh.geometry.within(buffer.geometry)
        residential_mesh.loc[within, 'within_public_radius'] = True

# Private hospital coverage
residential_mesh['within_private_radius'] = False
if len(private_hospitals) > 0:
    private_buffers = private_hospitals.copy()
    private_buffers['geometry'] = private_hospitals.geometry.buffer(buffer_distance)
    for idx, buffer in private_buffers.iterrows():
        within = residential_mesh.geometry.within(buffer.geometry)
        residential_mesh.loc[within, 'within_private_radius'] = True

# Build coverage categories
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

# 3. ENHANCED VORONOI CATCHMENT ANALYSIS (now with ambulance coverage data)
print("\n3. ENHANCED VORONOI CATCHMENT ANALYSIS...")

def calculate_catchment_statistics(mesh_blocks, voronoi_polygons, hospital_name_col, catchment_type="all"):
    """Calculate comprehensive statistics for each hospital catchment"""
    # Spatial join mesh blocks with Voronoi cells
    mesh_with_catchment = gpd.sjoin(mesh_blocks, voronoi_polygons, how='left', predicate='within')
    
    # Find the hospital name column in the joined data
    hospital_name_col_joined = hospital_name_col
    if hospital_name_col not in mesh_with_catchment.columns:
        if f"{hospital_name_col}_right" in mesh_with_catchment.columns:
            hospital_name_col_joined = f"{hospital_name_col}_right"
        else:
            for col in mesh_with_catchment.columns:
                if 'hosp' in col.lower() and 'name' in col.lower() and col != hospital_name_col:
                    hospital_name_col_joined = col
                    break
    
    print(f"  Processing {catchment_type} catchments using column: {hospital_name_col_joined}")
    
    # Calculate area for each Voronoi polygon
    voronoi_areas = {}
    for idx, row in voronoi_polygons.iterrows():
        hospital_name = row[hospital_name_col] if hospital_name_col in row else f"Hospital_{idx}"
        if voronoi_polygons.crs.is_projected:
            area_sqkm = row.geometry.area / 1e6
        else:
            area_sqkm = row.geometry.area * (111.32 ** 2)
        voronoi_areas[hospital_name] = area_sqkm
    
    # Calculate statistics for each catchment
    catchment_stats = []
    unique_hospitals = mesh_with_catchment[hospital_name_col_joined].dropna().unique()
    
    for hospital_name in unique_hospitals:
        catchment_mesh = mesh_with_catchment[mesh_with_catchment[hospital_name_col_joined] == hospital_name]
        
        if len(catchment_mesh) > 0:
            # Basic population statistics
            total_pop = catchment_mesh[population_field].sum()
            pop_under_4 = catchment_mesh[under4_field].sum()
            pop_over_85 = catchment_mesh[over85_field].sum()
            vulnerable_pop = catchment_mesh[vulnerable_field].sum()
            
            # Calculate area-based metrics
            catchment_area = voronoi_areas.get(hospital_name, 0)
            
            # Density calculations
            pop_density = total_pop / catchment_area if catchment_area > 0 else 0
            vulnerable_density = vulnerable_pop / catchment_area if catchment_area > 0 else 0
            
            # Percentage calculations
            vulnerable_pct = (vulnerable_pop / total_pop * 100) if total_pop > 0 else 0
            under4_pct = (pop_under_4 / total_pop * 100) if total_pop > 0 else 0
            over85_pct = (pop_over_85 / total_pop * 100) if total_pop > 0 else 0
            
            stats = {
                'hospital_name': hospital_name,
                'catchment_type': catchment_type,
                'num_mesh_blocks': len(catchment_mesh),
                'catchment_area_sqkm': catchment_area,
                'total_population': total_pop,
                'pop_density_per_sqkm': pop_density,
                'pop_under_4': pop_under_4,
                'pop_over_85': pop_over_85,
                'vulnerable_population': vulnerable_pop,
                'vulnerable_density_per_sqkm': vulnerable_density,
                'vulnerable_percentage': vulnerable_pct,
                'under4_percentage': under4_pct,
                'over85_percentage': over85_pct,
                'avg_distance_km': catchment_mesh['distance_to_nearest_hospital_km'].mean() if 'distance_to_nearest_hospital_km' in catchment_mesh.columns else 0,
                'max_distance_km': catchment_mesh['distance_to_nearest_hospital_km'].max() if 'distance_to_nearest_hospital_km' in catchment_mesh.columns else 0,
            }
            
            # Add ambulance coverage metrics if available
            if 'within_ambulance_radius' in catchment_mesh.columns:
                covered_pop = catchment_mesh[catchment_mesh['within_ambulance_radius']][population_field].sum()
                covered_vulnerable = catchment_mesh[catchment_mesh['within_ambulance_radius']][vulnerable_field].sum()
                stats['pop_within_15min'] = covered_pop
                stats['pop_within_15min_pct'] = (covered_pop / total_pop * 100) if total_pop > 0 else 0
                stats['vulnerable_within_15min'] = covered_vulnerable
                stats['vulnerable_within_15min_pct'] = (covered_vulnerable / vulnerable_pop * 100) if vulnerable_pop > 0 else 0
            
            # Add income data if available
            if 'low_income_percentage' in catchment_mesh.columns:
                stats['avg_low_income_percentage'] = catchment_mesh['low_income_percentage'].mean()
            
            # Calculate vulnerability index
            vulnerability_index = (
                (vulnerable_pct / 100) * 0.4 +
                (1 - stats.get('pop_within_15min_pct', 0) / 100) * 0.3 +
                (stats.get('avg_low_income_percentage', 0) / 100) * 0.3
            )
            stats['vulnerability_index'] = vulnerability_index
            
            catchment_stats.append(stats)
    
    return pd.DataFrame(catchment_stats), mesh_with_catchment

# Find hospital name column
hospital_name_col = None
for col in voronoi_all.columns:
    if 'hosp' in col.lower() and 'name' in col.lower():
        hospital_name_col = col
        break
    elif col.lower() in ['name', 'hospital_n', 'hosp_name']:
        hospital_name_col = col
        break

if hospital_name_col is None:
    hospital_name_col = 'hospital_id'
    voronoi_all[hospital_name_col] = range(len(voronoi_all))
    voronoi_public[hospital_name_col] = range(len(voronoi_public))
    voronoi_private[hospital_name_col] = range(len(voronoi_private))

print(f"Using hospital name column: {hospital_name_col}")

# Process all hospital types
all_catchment_stats, mesh_with_all_catchment = calculate_catchment_statistics(
    residential_mesh, voronoi_all, hospital_name_col, "all"
)

public_catchment_stats, mesh_with_public_catchment = calculate_catchment_statistics(
    residential_mesh, voronoi_public, hospital_name_col, "public"
)

private_catchment_stats, mesh_with_private_catchment = calculate_catchment_statistics(
    residential_mesh, voronoi_private, hospital_name_col, "private"
)

# Store the joined hospital name column for later use in report
hospital_name_col_joined = None
if hospital_name_col != 'hospital_id':  # Only look for joined column if we have a real name column
    for col in mesh_with_all_catchment.columns:
        if hospital_name_col in col and col != hospital_name_col and '_right' in col:
            hospital_name_col_joined = col
            break
        elif 'hosp' in col.lower() and 'name' in col.lower() and col != hospital_name_col:
            hospital_name_col_joined = col
            break
else:
    # If we're using hospital_id, look for any column with hospital_id
    for col in mesh_with_all_catchment.columns:
        if 'hospital_id' in col and col != 'hospital_id':
            hospital_name_col_joined = col
            break

print(f"Joined hospital column for report: {hospital_name_col_joined}")

# Combine all statistics
all_voronoi_stats = pd.concat([all_catchment_stats, public_catchment_stats, private_catchment_stats], 
                               ignore_index=True)

# 4. CALCULATE COVERAGE STATISTICS
print("\n4. CALCULATING COVERAGE STATISTICS...")

# Calculate coverage statistics
total_pop = residential_mesh[population_field].sum()
covered_pop = residential_mesh[residential_mesh['within_ambulance_radius']][population_field].sum()
coverage_percentage = (covered_pop / total_pop * 100) if total_pop > 0 else 0

vulnerable_total = residential_mesh[vulnerable_field].sum()
vulnerable_covered = residential_mesh[residential_mesh['within_ambulance_radius']][vulnerable_field].sum()
vulnerable_coverage = (vulnerable_covered / vulnerable_total * 100) if vulnerable_total > 0 else 0

# 5. IDENTIFY UNDERSERVED AREAS
print("\n5. IDENTIFYING UNDERSERVED AREAS...")

underserved_mesh = residential_mesh[~residential_mesh['within_ambulance_radius']].copy()
underserved_mesh = underserved_mesh[underserved_mesh[population_field] > 0]

# Create priority score
if len(underserved_mesh) > 0:
    max_vuln = underserved_mesh[vulnerable_field].max()
    max_dist = underserved_mesh['distance_to_nearest_hospital_km'].max()
    max_pop = underserved_mesh[population_field].max()
    
    underserved_mesh['priority_score'] = 0
    if max_vuln > 0:
        underserved_mesh['priority_score'] += (underserved_mesh[vulnerable_field] / max_vuln) * 0.5
    if max_dist > 0:
        underserved_mesh['priority_score'] += (underserved_mesh['distance_to_nearest_hospital_km'] / max_dist) * 0.3
    if max_pop > 0:
        underserved_mesh['priority_score'] += (underserved_mesh[population_field] / max_pop) * 0.2
    
    high_priority_threshold = underserved_mesh['priority_score'].quantile(0.9)
    high_priority_areas = underserved_mesh[underserved_mesh['priority_score'] >= high_priority_threshold]
else:
    high_priority_areas = underserved_mesh  # Empty dataframe

# 6. CREATE VISUALIZATIONS
print("\n6. CREATING VISUALIZATIONS...")

# Convert to WGS84 for display
residential_mesh_wgs84 = residential_mesh.to_crs('EPSG:4326')
hospitals_wgs84 = hospitals.to_crs('EPSG:4326')
voronoi_all_wgs84 = voronoi_all.to_crs('EPSG:4326')
if melbourne_boundary is not None:
    melbourne_boundary_wgs84 = melbourne_boundary.to_crs('EPSG:4326')

# Map 1: Vulnerable Population Density by Voronoi Catchment
fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
ax.set_facecolor('#FAFAFA')

# Merge Voronoi polygons with statistics
voronoi_all_with_stats = voronoi_all_wgs84.merge(
    all_catchment_stats[['hospital_name', 'vulnerable_density_per_sqkm']], 
    left_on=hospital_name_col, 
    right_on='hospital_name', 
    how='left'
)

voronoi_all_with_stats.plot(
    column='vulnerable_density_per_sqkm',
    ax=ax,
    cmap='YlOrRd',
    edgecolor='black',
    linewidth=0.5,
    legend=True,
    legend_kwds={'label': 'Vulnerable Population Density (per km²)', 
                 'orientation': 'vertical',
                 'shrink': 0.7}
)

hospitals_wgs84.plot(ax=ax, color='darkblue', markersize=80, marker='h', 
                     edgecolor='white', linewidth=1.5, zorder=5)

ax.set_title('Vulnerable Population Density by Hospital Catchment', fontsize=16, pad=20, weight='bold')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'maps', 'voronoi_vulnerable_density.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Map 2: Public vs Private Hospital Coverage
fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white')
ax.set_facecolor('#FAFAFA')

if melbourne_boundary is not None:
    melbourne_boundary_wgs84.boundary.plot(ax=ax, color='#333333', linewidth=2, zorder=4)

coverage_colors = {
    'No Coverage': '#E57373',
    'Private Only': '#FFB74D',
    'Public Only': '#64B5F6',
    'Both': '#81C784'
}

for cov_type, color in coverage_colors.items():
    subset = residential_mesh_wgs84[residential_mesh_wgs84['coverage_type'] == cov_type]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor='none', alpha=0.8)

if len(public_hospitals) > 0:
    public_hospitals.to_crs('EPSG:4326').plot(ax=ax, color='#1976D2', marker='s',
                                               markersize=80, edgecolor='white', linewidth=2, zorder=5)

if len(private_hospitals) > 0:
    private_hospitals.to_crs('EPSG:4326').plot(ax=ax, color='#D32F2F', marker='^',
                                               markersize=80, edgecolor='white', linewidth=2, zorder=5)

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
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'maps', 'coverage_by_hospital_type.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Map 3: Comparative bar chart - Public vs Private catchments
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
fig.suptitle('Public vs Private Hospital Catchment Comparison', fontsize=16, weight='bold')

catchment_types = ['public', 'private']

# Plot 1: Average vulnerable population
ax1 = axes[0, 0]
avg_vulnerable = [
    all_voronoi_stats[all_voronoi_stats['catchment_type'] == t]['vulnerable_population'].mean() 
    for t in catchment_types
]
bars1 = ax1.bar(catchment_types, avg_vulnerable, color=['#1976D2', '#D32F2F'])
ax1.set_ylabel('Average Vulnerable Population')
ax1.set_title('Average Vulnerable Population per Catchment')
ax1.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,.0f}', ha='center', va='bottom')

# Plot 2: Average vulnerable density
ax2 = axes[0, 1]
avg_density = [
    all_voronoi_stats[all_voronoi_stats['catchment_type'] == t]['vulnerable_density_per_sqkm'].mean() 
    for t in catchment_types
]
bars2 = ax2.bar(catchment_types, avg_density, color=['#1976D2', '#D32F2F'])
ax2.set_ylabel('Vulnerable per km²')
ax2.set_title('Average Vulnerable Population Density')
ax2.grid(axis='y', alpha=0.3)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,.0f}', ha='center', va='bottom')

# Plot 3: Average catchment area
ax3 = axes[1, 0]
avg_area = [
    all_voronoi_stats[all_voronoi_stats['catchment_type'] == t]['catchment_area_sqkm'].mean() 
    for t in catchment_types
]
bars3 = ax3.bar(catchment_types, avg_area, color=['#1976D2', '#D32F2F'])
ax3.set_ylabel('Area (km²)')
ax3.set_title('Average Catchment Area')
ax3.grid(axis='y', alpha=0.3)

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,.0f}', ha='center', va='bottom')

# Plot 4: Average vulnerability index
ax4 = axes[1, 1]
avg_vuln_index = [
    all_voronoi_stats[all_voronoi_stats['catchment_type'] == t]['vulnerability_index'].mean() 
    for t in catchment_types
]
bars4 = ax4.bar(catchment_types, avg_vuln_index, color=['#1976D2', '#D32F2F'])
ax4.set_ylabel('Vulnerability Index')
ax4.set_title('Average Vulnerability Index')
ax4.set_ylim(0, 1)
ax4.grid(axis='y', alpha=0.3)

for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'maps', 'voronoi_public_private_comparison.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 7. GENERATE COMPREHENSIVE DETAILED REPORT
print("\n7. GENERATING COMPREHENSIVE DETAILED REPORT...")

report_path = os.path.join(output_dir, 'reports', 'detailed_accessibility_analysis_report.txt')
with open(report_path, 'w') as f:
    f.write("="*100 + "\n")
    f.write("COMPREHENSIVE EMERGENCY SERVICE ACCESSIBILITY ANALYSIS WITH ENHANCED VORONOI CATCHMENTS\n")
    f.write("="*100 + "\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Study Area: Melbourne, Victoria, Australia\n")
    f.write(f"Total Mesh Blocks Analyzed: {len(residential_mesh):,}\n")
    f.write(f"Total Hospitals: {len(hospitals)}\n")
    f.write(f"  - Public Hospitals: {len(public_hospitals)}\n")
    f.write(f"  - Private Hospitals: {len(private_hospitals)}\n")
    f.write("\n")
    
    # SECTION 1: POPULATION OVERVIEW
    f.write("="*100 + "\n")
    f.write("SECTION 1: POPULATION DEMOGRAPHICS OVERVIEW\n")
    f.write("="*100 + "\n\n")
    
    f.write("TOTAL POPULATION STATISTICS:\n")
    f.write("-"*50 + "\n")
    f.write(f"Total Population: {total_pop:,.0f}\n")
    f.write(f"Population Density: {total_pop / (residential_mesh.geometry.area.sum() / 1e6):.1f} per km²\n")
    f.write(f"Total Mesh Blocks with Population: {(residential_mesh[population_field] > 0).sum():,}\n")
    f.write(f"Average Population per Mesh Block: {residential_mesh[population_field].mean():.1f}\n")
    f.write(f"Median Population per Mesh Block: {residential_mesh[population_field].median():.1f}\n")
    f.write(f"Standard Deviation: {residential_mesh[population_field].std():.1f}\n\n")
    
    f.write("VULNERABLE POPULATION BREAKDOWN:\n")
    f.write("-"*50 + "\n")
    f.write(f"Total Vulnerable Population: {vulnerable_total:,.0f} ({vulnerable_total/total_pop*100:.2f}%)\n")
    f.write(f"  - Under 4 years: {residential_mesh[under4_field].sum():,.0f} ({residential_mesh[under4_field].sum()/total_pop*100:.2f}%)\n")
    f.write(f"  - Over 85 years: {residential_mesh[over85_field].sum():,.0f} ({residential_mesh[over85_field].sum()/total_pop*100:.2f}%)\n")
    f.write(f"Average Vulnerable % per Mesh Block: {residential_mesh[vulnerable_pct_field].mean():.2f}%\n")
    f.write(f"Median Vulnerable % per Mesh Block: {residential_mesh[vulnerable_pct_field].median():.2f}%\n")
    f.write(f"Max Vulnerable % in a Mesh Block: {residential_mesh[vulnerable_pct_field].max():.2f}%\n\n")
    
    # Add income statistics if available
    if 'low_income_percentage' in residential_mesh.columns:
        f.write("SOCIO-ECONOMIC STATISTICS:\n")
        f.write("-"*50 + "\n")
        f.write(f"Total Households: {residential_mesh['total_households'].sum():,.0f}\n")
        f.write(f"Low Income Households: {residential_mesh['low_income'].sum():,.0f} ({residential_mesh['low_income'].sum()/residential_mesh['total_households'].sum()*100:.1f}%)\n")
        f.write(f"Middle Income Households: {residential_mesh['middle_income'].sum():,.0f} ({residential_mesh['middle_income'].sum()/residential_mesh['total_households'].sum()*100:.1f}%)\n")
        f.write(f"High Income Households: {residential_mesh['high_income'].sum():,.0f} ({residential_mesh['high_income'].sum()/residential_mesh['total_households'].sum()*100:.1f}%)\n")
        f.write(f"Average Low Income %: {residential_mesh['low_income_percentage'].mean():.1f}%\n")
        f.write(f"Mesh Blocks with >50% Low Income: {(residential_mesh['low_income_percentage'] > 50).sum():,}\n\n")
    
    # SECTION 2: DISTANCE ANALYSIS
    f.write("="*100 + "\n")
    f.write("SECTION 2: DISTANCE TO HOSPITAL ANALYSIS\n")
    f.write("="*100 + "\n\n")
    
    f.write("OVERALL DISTANCE STATISTICS:\n")
    f.write("-"*50 + "\n")
    f.write(f"Average Distance to Nearest Hospital: {residential_mesh['distance_to_nearest_hospital_km'].mean():.2f} km\n")
    f.write(f"Median Distance to Nearest Hospital: {residential_mesh['distance_to_nearest_hospital_km'].median():.2f} km\n")
    f.write(f"Minimum Distance: {residential_mesh['distance_to_nearest_hospital_km'].min():.2f} km\n")
    f.write(f"Maximum Distance: {residential_mesh['distance_to_nearest_hospital_km'].max():.2f} km\n")
    f.write(f"Standard Deviation: {residential_mesh['distance_to_nearest_hospital_km'].std():.2f} km\n\n")
    
    f.write("DISTANCE DISTRIBUTION:\n")
    f.write("-"*50 + "\n")
    distance_bins = [0, 2, 5, 10, 15, 20, 30, float('inf')]
    distance_labels = ['0-2 km', '2-5 km', '5-10 km', '10-15 km', '15-20 km', '20-30 km', '30+ km']
    residential_mesh['distance_category'] = pd.cut(residential_mesh['distance_to_nearest_hospital_km'], 
                                                   bins=distance_bins, labels=distance_labels)
    
    for category in distance_labels:
        category_data = residential_mesh[residential_mesh['distance_category'] == category]
        pop_in_category = category_data[population_field].sum()
        vuln_in_category = category_data[vulnerable_field].sum()
        f.write(f"{category}: {len(category_data):,} mesh blocks, {pop_in_category:,.0f} people ({pop_in_category/total_pop*100:.1f}%), {vuln_in_category:,.0f} vulnerable ({vuln_in_category/vulnerable_total*100:.1f}%)\n")
    
    f.write("\nDISTANCE BY HOSPITAL TYPE:\n")
    f.write("-"*50 + "\n")
    if 'distance_to_public_hospital_km' in residential_mesh.columns:
        f.write(f"Average Distance to Public Hospital: {residential_mesh['distance_to_public_hospital_km'].mean():.2f} km\n")
        f.write(f"Average Distance to Private Hospital: {residential_mesh['distance_to_private_hospital_km'].mean():.2f} km\n")
        f.write(f"Difference (Public - Private): {residential_mesh['distance_to_public_hospital_km'].mean() - residential_mesh['distance_to_private_hospital_km'].mean():.2f} km\n\n")
    
    # SECTION 3: VORONOI CATCHMENT DETAILED ANALYSIS
    f.write("="*100 + "\n")
    f.write("SECTION 3: DETAILED VORONOI CATCHMENT ANALYSIS\n")
    f.write("="*100 + "\n\n")
    
    # Overall Voronoi Statistics
    f.write("CATCHMENT SUMMARY STATISTICS:\n")
    f.write("-"*80 + "\n")
    
    for catchment_type in ['all', 'public', 'private']:
        type_stats = all_voronoi_stats[all_voronoi_stats['catchment_type'] == catchment_type]
        if len(type_stats) > 0:
            f.write(f"\n{catchment_type.upper()} HOSPITAL CATCHMENTS:\n")
            f.write(f"  Number of catchments: {len(type_stats)}\n")
            f.write(f"  Total area covered: {type_stats['catchment_area_sqkm'].sum():,.1f} km²\n")
            f.write(f"  Average catchment area: {type_stats['catchment_area_sqkm'].mean():.1f} km² (±{type_stats['catchment_area_sqkm'].std():.1f})\n")
            f.write(f"  Median catchment area: {type_stats['catchment_area_sqkm'].median():.1f} km²\n")
            f.write(f"  Smallest catchment: {type_stats['catchment_area_sqkm'].min():.1f} km²\n")
            f.write(f"  Largest catchment: {type_stats['catchment_area_sqkm'].max():.1f} km²\n")
            f.write(f"\n  POPULATION SERVED:\n")
            f.write(f"  Total population: {type_stats['total_population'].sum():,.0f}\n")
            f.write(f"  Average population per catchment: {type_stats['total_population'].mean():,.0f} (±{type_stats['total_population'].std():,.0f})\n")
            f.write(f"  Median population per catchment: {type_stats['total_population'].median():,.0f}\n")
            f.write(f"  Min population in catchment: {type_stats['total_population'].min():,.0f}\n")
            f.write(f"  Max population in catchment: {type_stats['total_population'].max():,.0f}\n")
            f.write(f"\n  POPULATION DENSITY:\n")
            f.write(f"  Average density: {type_stats['pop_density_per_sqkm'].mean():.1f} per km² (±{type_stats['pop_density_per_sqkm'].std():.1f})\n")
            f.write(f"  Median density: {type_stats['pop_density_per_sqkm'].median():.1f} per km²\n")
            f.write(f"  Min density: {type_stats['pop_density_per_sqkm'].min():.1f} per km²\n")
            f.write(f"  Max density: {type_stats['pop_density_per_sqkm'].max():.1f} per km²\n")
            f.write(f"\n  VULNERABLE POPULATION:\n")
            f.write(f"  Total vulnerable: {type_stats['vulnerable_population'].sum():,.0f}\n")
            f.write(f"  Average vulnerable per catchment: {type_stats['vulnerable_population'].mean():,.0f} (±{type_stats['vulnerable_population'].std():,.0f})\n")
            f.write(f"  Average vulnerable percentage: {type_stats['vulnerable_percentage'].mean():.2f}% (±{type_stats['vulnerable_percentage'].std():.2f}%)\n")
            f.write(f"  Average vulnerable density: {type_stats['vulnerable_density_per_sqkm'].mean():.1f} per km² (±{type_stats['vulnerable_density_per_sqkm'].std():.1f})\n")
            f.write(f"  Max vulnerable density: {type_stats['vulnerable_density_per_sqkm'].max():.1f} per km²\n")
            f.write(f"\n  AGE DISTRIBUTION:\n")
            f.write(f"  Total under 4: {type_stats['pop_under_4'].sum():,.0f}\n")
            f.write(f"  Average under 4 percentage: {type_stats['under4_percentage'].mean():.2f}%\n")
            f.write(f"  Total over 85: {type_stats['pop_over_85'].sum():,.0f}\n")
            f.write(f"  Average over 85 percentage: {type_stats['over85_percentage'].mean():.2f}%\n")
            f.write(f"\n  VULNERABILITY INDEX:\n")
            f.write(f"  Average vulnerability index: {type_stats['vulnerability_index'].mean():.3f} (±{type_stats['vulnerability_index'].std():.3f})\n")
            f.write(f"  Median vulnerability index: {type_stats['vulnerability_index'].median():.3f}\n")
            f.write(f"  Max vulnerability index: {type_stats['vulnerability_index'].max():.3f}\n")
            f.write(f"  Catchments with index > 0.5: {(type_stats['vulnerability_index'] > 0.5).sum()}\n")
            f.write(f"  Catchments with index > 0.7: {(type_stats['vulnerability_index'] > 0.7).sum()}\n")
    
    # Individual Catchment Details
    f.write("\n\nINDIVIDUAL CATCHMENT DETAILS (ALL HOSPITALS):\n")
    f.write("-"*100 + "\n")
    f.write("Sorted by Vulnerability Index (Highest First)\n\n")
    
    all_hospital_stats = all_voronoi_stats[all_voronoi_stats['catchment_type'] == 'all'].sort_values('vulnerability_index', ascending=False)
    
    for idx, row in all_hospital_stats.iterrows():
        f.write(f"CATCHMENT: {row['hospital_name']}\n")
        f.write(f"  Area: {row['catchment_area_sqkm']:.1f} km²\n")
        f.write(f"  Total Population: {row['total_population']:,.0f} (Density: {row['pop_density_per_sqkm']:.1f} per km²)\n")
        f.write(f"  Vulnerable Population: {row['vulnerable_population']:,.0f} ({row['vulnerable_percentage']:.1f}%) (Density: {row['vulnerable_density_per_sqkm']:.1f} per km²)\n")
        f.write(f"    - Under 4: {row['pop_under_4']:,.0f} ({row['under4_percentage']:.1f}%)\n")
        f.write(f"    - Over 85: {row['pop_over_85']:,.0f} ({row['over85_percentage']:.1f}%)\n")
        f.write(f"  Average Distance to Hospital: {row['avg_distance_km']:.1f} km (Max: {row['max_distance_km']:.1f} km)\n")
        if 'pop_within_15min_pct' in row:
            f.write(f"  15-min Coverage: {row['pop_within_15min_pct']:.1f}% of population\n")
            f.write(f"  15-min Vulnerable Coverage: {row.get('vulnerable_within_15min_pct', 0):.1f}% of vulnerable\n")
        if 'avg_low_income_percentage' in row:
            f.write(f"  Average Low Income: {row['avg_low_income_percentage']:.1f}%\n")
        f.write(f"  Vulnerability Index: {row['vulnerability_index']:.3f}\n")
        f.write("-"*50 + "\n")
    
    # SECTION 4: AMBULANCE RESPONSE COVERAGE
    f.write("\n="*100 + "\n")
    f.write(f"SECTION 4: {AMBULANCE_CODE1_MINUTES}-MINUTE AMBULANCE RESPONSE COVERAGE\n")
    f.write("="*100 + "\n\n")
    
    f.write("OVERALL AMBULANCE COVERAGE:\n")
    f.write("-"*50 + "\n")
    f.write(f"Response Time Target: {AMBULANCE_CODE1_MINUTES} minutes\n")
    f.write(f"Average Ambulance Speed: {AMBULANCE_SPEED_KMH} km/h\n")
    f.write(f"Coverage Radius: {RADIUS_KM:.1f} km\n\n")
    
    f.write("POPULATION COVERAGE:\n")
    f.write("-"*50 + "\n")
    f.write(f"Total Population Covered: {covered_pop:,.0f} ({coverage_percentage:.2f}%)\n")
    f.write(f"Total Population NOT Covered: {total_pop - covered_pop:,.0f} ({100-coverage_percentage:.2f}%)\n")
    f.write(f"Mesh Blocks Covered: {residential_mesh['within_ambulance_radius'].sum():,} ({residential_mesh['within_ambulance_radius'].sum()/len(residential_mesh)*100:.1f}%)\n\n")
    
    f.write("VULNERABLE POPULATION COVERAGE:\n")
    f.write("-"*50 + "\n")
    f.write(f"Vulnerable Population Covered: {vulnerable_covered:,.0f} ({vulnerable_coverage:.2f}%)\n")
    f.write(f"Vulnerable Population NOT Covered: {vulnerable_total - vulnerable_covered:,.0f} ({100-vulnerable_coverage:.2f}%)\n")
    
    # Coverage by age group
    under4_covered = residential_mesh[residential_mesh['within_ambulance_radius']][under4_field].sum()
    over85_covered = residential_mesh[residential_mesh['within_ambulance_radius']][over85_field].sum()
    under4_total = residential_mesh[under4_field].sum()
    over85_total = residential_mesh[over85_field].sum()
    
    f.write(f"Under 4 Coverage: {under4_covered:,.0f} / {under4_total:,.0f} ({under4_covered/under4_total*100:.1f}%)\n")
    f.write(f"Over 85 Coverage: {over85_covered:,.0f} / {over85_total:,.0f} ({over85_covered/over85_total*100:.1f}%)\n\n")
    
    # Public vs Private Coverage
    f.write("COVERAGE BY HOSPITAL TYPE:\n")
    f.write("-"*50 + "\n")
    
    # Calculate detailed coverage statistics
    public_covered_pop = residential_mesh[residential_mesh['within_public_radius']][population_field].sum()
    private_covered_pop = residential_mesh[residential_mesh['within_private_radius']][population_field].sum()
    both_covered_pop = residential_mesh[residential_mesh['within_public_radius'] & residential_mesh['within_private_radius']][population_field].sum()
    
    f.write(f"PUBLIC HOSPITALS:\n")
    f.write(f"  Population Covered: {public_covered_pop:,.0f} ({public_covered_pop/total_pop*100:.2f}%)\n")
    f.write(f"  Vulnerable Covered: {residential_mesh[residential_mesh['within_public_radius']][vulnerable_field].sum():,.0f}\n")
    f.write(f"  Mesh Blocks Covered: {residential_mesh['within_public_radius'].sum():,}\n")
    
    f.write(f"\nPRIVATE HOSPITALS:\n")
    f.write(f"  Population Covered: {private_covered_pop:,.0f} ({private_covered_pop/total_pop*100:.2f}%)\n")
    f.write(f"  Vulnerable Covered: {residential_mesh[residential_mesh['within_private_radius']][vulnerable_field].sum():,.0f}\n")
    f.write(f"  Mesh Blocks Covered: {residential_mesh['within_private_radius'].sum():,}\n")
    
    f.write(f"\nCOVERAGE OVERLAP:\n")
    f.write(f"  Population Covered by Both: {both_covered_pop:,.0f} ({both_covered_pop/total_pop*100:.2f}%)\n")
    f.write(f"  Population Covered by Public Only: {public_covered_pop - both_covered_pop:,.0f}\n")
    f.write(f"  Population Covered by Private Only: {private_covered_pop - both_covered_pop:,.0f}\n\n")
    
    # Coverage Type Distribution
    f.write("POPULATION BY COVERAGE TYPE:\n")
    f.write("-"*50 + "\n")
    coverage_stats = residential_mesh.groupby('coverage_type').agg({
        population_field: ['sum', 'count'],
        vulnerable_field: 'sum',
        vulnerable_pct_field: 'mean'
    })
    
    for coverage_type in ['No Coverage', 'Public Only', 'Private Only', 'Both']:
        if coverage_type in coverage_stats.index:
            pop_sum = coverage_stats.loc[coverage_type, (population_field, 'sum')]
            mesh_count = coverage_stats.loc[coverage_type, (population_field, 'count')]
            vuln_sum = coverage_stats.loc[coverage_type, (vulnerable_field, 'sum')]
            vuln_avg = coverage_stats.loc[coverage_type, (vulnerable_pct_field, 'mean')]
            
            f.write(f"{coverage_type}:\n")
            f.write(f"  Population: {pop_sum:,.0f} ({pop_sum/total_pop*100:.1f}%)\n")
            f.write(f"  Mesh Blocks: {mesh_count:,}\n")
            f.write(f"  Vulnerable: {vuln_sum:,.0f} ({vuln_sum/vulnerable_total*100:.1f}%)\n")
            f.write(f"  Average Vulnerable %: {vuln_avg:.1f}%\n\n")
    
    # SECTION 5: UNDERSERVED AREAS DETAILED ANALYSIS
    f.write("="*100 + "\n")
    f.write("SECTION 5: UNDERSERVED AREAS DETAILED ANALYSIS\n")
    f.write("="*100 + "\n\n")
    
    f.write("UNDERSERVED AREAS SUMMARY:\n")
    f.write("-"*50 + "\n")
    f.write(f"Total Underserved Mesh Blocks: {len(underserved_mesh):,}\n")
    f.write(f"Total Underserved Population: {underserved_mesh[population_field].sum():,.0f} ({underserved_mesh[population_field].sum()/total_pop*100:.1f}%)\n")
    f.write(f"Total Underserved Vulnerable: {underserved_mesh[vulnerable_field].sum():,.0f} ({underserved_mesh[vulnerable_field].sum()/vulnerable_total*100:.1f}%)\n")
    f.write(f"  - Under 4: {underserved_mesh[under4_field].sum():,.0f}\n")
    f.write(f"  - Over 85: {underserved_mesh[over85_field].sum():,.0f}\n\n")
    
    f.write("UNDERSERVED DISTANCE STATISTICS:\n")
    f.write("-"*50 + "\n")
    if len(underserved_mesh) > 0:
        f.write(f"Average Distance to Hospital: {underserved_mesh['distance_to_nearest_hospital_km'].mean():.1f} km\n")
        f.write(f"Median Distance to Hospital: {underserved_mesh['distance_to_nearest_hospital_km'].median():.1f} km\n")
        f.write(f"Max Distance to Hospital: {underserved_mesh['distance_to_nearest_hospital_km'].max():.1f} km\n\n")
    else:
        f.write("No underserved areas to analyze.\n\n")
    
    if 'low_income_percentage' in underserved_mesh.columns and len(underserved_mesh) > 0:
        f.write("UNDERSERVED SOCIO-ECONOMIC PROFILE:\n")
        f.write("-"*50 + "\n")
        f.write(f"Average Low Income %: {underserved_mesh['low_income_percentage'].mean():.1f}%\n")
        f.write(f"Underserved with >50% Low Income: {(underserved_mesh['low_income_percentage'] > 50).sum():,}\n\n")
    
    f.write("HIGH PRIORITY UNDERSERVED AREAS:\n")
    f.write("-"*50 + "\n")
    f.write(f"Number of High Priority Areas (Top 10%): {len(high_priority_areas):,}\n")
    if len(high_priority_areas) > 0:
        f.write(f"Population in High Priority Areas: {high_priority_areas[population_field].sum():,.0f}\n")
        f.write(f"Vulnerable in High Priority Areas: {high_priority_areas[vulnerable_field].sum():,.0f}\n")
        f.write(f"Average Distance in High Priority: {high_priority_areas['distance_to_nearest_hospital_km'].mean():.1f} km\n")
        f.write(f"Average Priority Score: {high_priority_areas['priority_score'].mean():.3f}\n\n")
    else:
        f.write("No high priority areas identified.\n\n")
    
    # Top 10 Priority Areas
    f.write("TOP 10 PRIORITY UNDERSERVED AREAS:\n")
    f.write("-"*80 + "\n")
    if len(high_priority_areas) > 0:
        top_priority = high_priority_areas.nlargest(min(10, len(high_priority_areas)), 'priority_score')
        for idx, (_, area) in enumerate(top_priority.iterrows(), 1):
            f.write(f"{idx}. Priority Score: {area['priority_score']:.3f}\n")
            f.write(f"   Population: {area[population_field]:,.0f}, Vulnerable: {area[vulnerable_field]:,.0f} ({area[vulnerable_pct_field]:.1f}%)\n")
            f.write(f"   Distance to Hospital: {area['distance_to_nearest_hospital_km']:.1f} km\n")
            if 'low_income_percentage' in area:
                f.write(f"   Low Income: {area['low_income_percentage']:.1f}%\n")
            f.write("\n")
    else:
        f.write("No high priority underserved areas identified.\n\n")
    
    # SECTION 6: COMPARATIVE ANALYSIS
    f.write("="*100 + "\n")
    f.write("SECTION 6: PUBLIC VS PRIVATE HOSPITAL COMPARATIVE ANALYSIS\n")
    f.write("="*100 + "\n\n")
    
    if len(public_hospitals) > 0 and len(private_hospitals) > 0:
        public_stats = all_voronoi_stats[all_voronoi_stats['catchment_type'] == 'public']
        private_stats = all_voronoi_stats[all_voronoi_stats['catchment_type'] == 'private']
        
        f.write("CATCHMENT COMPARISON:\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Metric':<40} {'Public':>15} {'Private':>15} {'Difference':>15}\n")
        f.write("-"*85 + "\n")
        
        metrics = [
            ('Number of Hospitals', len(public_stats), len(private_stats)),
            ('Avg Catchment Area (km²)', public_stats['catchment_area_sqkm'].mean(), private_stats['catchment_area_sqkm'].mean()),
            ('Avg Population per Catchment', public_stats['total_population'].mean(), private_stats['total_population'].mean()),
            ('Avg Pop Density (per km²)', public_stats['pop_density_per_sqkm'].mean(), private_stats['pop_density_per_sqkm'].mean()),
            ('Avg Vulnerable Population', public_stats['vulnerable_population'].mean(), private_stats['vulnerable_population'].mean()),
            ('Avg Vulnerable %', public_stats['vulnerable_percentage'].mean(), private_stats['vulnerable_percentage'].mean()),
            ('Avg Vulnerable Density (per km²)', public_stats['vulnerable_density_per_sqkm'].mean(), private_stats['vulnerable_density_per_sqkm'].mean()),
            ('Avg Distance to Hospital (km)', public_stats['avg_distance_km'].mean(), private_stats['avg_distance_km'].mean()),
            ('Avg Vulnerability Index', public_stats['vulnerability_index'].mean(), private_stats['vulnerability_index'].mean()),
        ]
        
        for metric_name, public_val, private_val in metrics:
            diff = public_val - private_val
            diff_pct = (diff / private_val * 100) if private_val != 0 else 0
            f.write(f"{metric_name:<40} {public_val:>15.1f} {private_val:>15.1f} {diff:>15.1f} ({diff_pct:+.1f}%)\n")
    
    # SECTION 7: KEY FINDINGS AND INSIGHTS
    f.write("\n="*100 + "\n")
    f.write("SECTION 7: KEY FINDINGS, INSIGHTS AND RECOMMENDATIONS\n")
    f.write("="*100 + "\n\n")
    
    f.write("KEY FINDINGS:\n")
    f.write("-"*50 + "\n")
    
    # Generate data-driven findings
    findings = []
    
    # Coverage findings
    if coverage_percentage < 70:
        findings.append(f"CRITICAL: Only {coverage_percentage:.1f}% of population is within {AMBULANCE_CODE1_MINUTES}-minute emergency response")
    elif coverage_percentage < 85:
        findings.append(f"WARNING: {coverage_percentage:.1f}% coverage falls short of 85% target for emergency response")
    
    if vulnerable_coverage < coverage_percentage - 5:
        findings.append(f"Vulnerable population coverage ({vulnerable_coverage:.1f}%) is significantly lower than general population")
    
    # Distance findings
    far_hospitals = (residential_mesh['distance_to_nearest_hospital_km'] > 20).sum()
    if far_hospitals > 0:
        findings.append(f"{far_hospitals:,} mesh blocks are more than 20km from nearest hospital")
    
    # Voronoi findings
    high_vuln_catchments = all_voronoi_stats[all_voronoi_stats['vulnerability_index'] > 0.7]
    if len(high_vuln_catchments) > 0:
        findings.append(f"{len(high_vuln_catchments)} hospital catchments have critical vulnerability index (>0.7)")
    
    # Density findings
    if len(underserved_mesh) > 0:
        high_density_underserved = underserved_mesh[underserved_mesh[population_field] > underserved_mesh[population_field].quantile(0.75)]
        if len(high_density_underserved) > 0:
            findings.append(f"{len(high_density_underserved)} high-density areas lack emergency coverage")
    
    # Public vs Private findings
    if len(public_hospitals) > 0 and len(private_hospitals) > 0:
        public_avg_area = all_voronoi_stats[all_voronoi_stats['catchment_type'] == 'public']['catchment_area_sqkm'].mean()
        private_avg_area = all_voronoi_stats[all_voronoi_stats['catchment_type'] == 'private']['catchment_area_sqkm'].mean()
        
        if public_avg_area > private_avg_area * 2:
            findings.append(f"Public hospitals serve areas {public_avg_area/private_avg_area:.1f}x larger than private hospitals")
    
    for finding in findings:
        f.write(f"• {finding}\n")
    
    f.write("\nSPATIAL INSIGHTS:\n")
    f.write("-"*50 + "\n")
    
    # Identify geographic patterns
    if melbourne_boundary is not None:
        f.write("• Analysis covers the Melbourne metropolitan area\n")
    
    # Identify concentrated vulnerability areas
    vuln_hotspots = all_voronoi_stats[all_voronoi_stats['vulnerable_density_per_sqkm'] > 
                                      all_voronoi_stats['vulnerable_density_per_sqkm'].quantile(0.9)]
    if len(vuln_hotspots) > 0:
        f.write(f"• {len(vuln_hotspots)} catchments have extreme vulnerable population density (top 10%)\n")
        f.write(f"  Hospitals: {', '.join(vuln_hotspots['hospital_name'].head(5).tolist())}\n")
    
    f.write("\nRECOMMENDATIONS:\n")
    f.write("-"*50 + "\n")
    
    recommendations = []
    
    # Priority-based recommendations
    if len(high_priority_areas) > 20:
        recommendations.append(f"URGENT: Establish emergency facilities in {len(high_priority_areas)} high-priority underserved areas")
    
    if vulnerable_coverage < 80:
        recommendations.append("Implement mobile emergency units specifically for areas with high vulnerable populations")
    
    if len(high_vuln_catchments) > 0:
        hospitals_list = high_vuln_catchments['hospital_name'].head(3).tolist()
        recommendations.append(f"Enhance resources at high-vulnerability hospitals: {', '.join(hospitals_list)}")
    
    # Distance-based recommendations
    if len(underserved_mesh) > 0 and 'distance_to_nearest_hospital_km' in underserved_mesh.columns:
        max_distance = underserved_mesh['distance_to_nearest_hospital_km'].max()
        if max_distance > 30:
            recommendations.append("Consider air ambulance services for remote areas beyond 30km from hospitals")
    
    # Type-based recommendations
    no_coverage_pop = residential_mesh[residential_mesh['coverage_type'] == 'No Coverage'][population_field].sum()
    if no_coverage_pop > total_pop * 0.15:
        recommendations.append("Develop public-private partnerships to extend coverage to unserved areas")
    
    recommendations.extend([
        "Implement real-time ambulance tracking to optimize response times",
        "Establish community first-responder programs in underserved areas",
        "Develop telemedicine capabilities for initial assessment in remote locations",
        "Create inter-hospital transfer protocols to balance patient loads",
        "Regular review of catchment boundaries based on population growth"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        f.write(f"{i}. {rec}\n")
    
    # SECTION 8: METHODOLOGY AND DATA QUALITY
    f.write("\n="*100 + "\n")
    f.write("SECTION 8: METHODOLOGY AND DATA QUALITY NOTES\n")
    f.write("="*100 + "\n\n")
    
    f.write("DATA SOURCES:\n")
    f.write("-"*50 + "\n")
    f.write(f"• Hospital Locations: {len(hospitals)} facilities from hospital_Points.shp\n")
    f.write(f"• Residential Areas: {len(residential_mesh)} mesh blocks from Residential_mesh_blocks_pop.shp\n")
    f.write(f"• Voronoi Catchments: Pre-calculated using QGIS spatial analysis\n")
    f.write(f"• Census Data: {'2021 Australian Census (SA1 level)' if census_data_available else 'Not available - using estimates'}\n\n")
    
    f.write("ANALYTICAL METHODS:\n")
    f.write("-"*50 + "\n")
    f.write("• Distance Calculations: Euclidean distance in projected coordinates (GDA2020 MGA Zone 55)\n")
    f.write(f"• Ambulance Coverage: {RADIUS_KM:.1f}km radius based on {AMBULANCE_CODE1_MINUTES}-minute response at {AMBULANCE_SPEED_KMH}km/h\n")
    f.write("• Voronoi Tessellation: Defines theoretical service areas based on nearest hospital\n")
    f.write("• Vulnerability Index: Composite score combining population vulnerability, coverage, and income\n")
    f.write("• Priority Scoring: Weighted combination of vulnerability, distance, and population\n\n")
    
    f.write("LIMITATIONS AND ASSUMPTIONS:\n")
    f.write("-"*50 + "\n")
    f.write("• Distances are straight-line; actual road travel times will vary\n")
    f.write("• Ambulance speed assumes average conditions without traffic consideration\n")
    f.write("• Hospital capacity and specialization not considered\n")
    f.write("• Population distribution assumed uniform within mesh blocks\n")
    f.write("• Emergency response times may vary by time of day and weather\n")
    f.write("• Private hospital accessibility may be limited by insurance/payment\n\n")
    
    f.write("DATA QUALITY INDICATORS:\n")
    f.write("-"*50 + "\n")
    f.write(f"• Mesh blocks with population data: {(residential_mesh[population_field] > 0).sum():,} / {len(residential_mesh):,}\n")
    
    if hospital_name_col_joined and hospital_name_col_joined in mesh_with_all_catchment.columns:
        f.write(f"• Mesh blocks successfully assigned to catchments: {mesh_with_all_catchment[hospital_name_col_joined].notna().sum():,}\n")
    else:
        f.write(f"• Mesh blocks successfully assigned to catchments: Unable to determine (column not found)\n")
    
    if len(all_catchment_stats) > 0:
        pop_variance = abs(total_pop - all_catchment_stats['total_population'].sum()) / total_pop * 100
        f.write(f"• Population variance check: {pop_variance:.2f}% difference\n")
    else:
        f.write(f"• Population variance check: No catchment statistics available\n")
    
    f.write("\n" + "="*100 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*100 + "\n")

# 8. SAVE ALL DATA FILES
print("\n8. SAVING DATA FILES...")

all_voronoi_stats.to_csv(os.path.join(output_dir, 'data', 'voronoi_catchment_statistics.csv'), index=False)
catchment_rankings = all_voronoi_stats.sort_values('vulnerability_index', ascending=False)
catchment_rankings.to_csv(os.path.join(output_dir, 'data', 'catchment_vulnerability_rankings.csv'), index=False)

if len(underserved_mesh) > 0:
    underserved_cols = [col for col in [population_field, under4_field, over85_field, vulnerable_field, 
                       vulnerable_pct_field, 'distance_to_nearest_hospital_km', 'priority_score'] 
                       if col in underserved_mesh.columns]
    underserved_summary = underserved_mesh[underserved_cols].copy()
    underserved_summary.to_csv(os.path.join(output_dir, 'data', 'underserved_areas.csv'), index=False)

print(f"\nAnalysis complete! Results saved to: {output_dir}")
print("\nKey outputs:")
print(f"  - Comprehensive Detailed Report: {report_path}")
print(f"  - Voronoi Maps: {os.path.join(output_dir, 'maps')}/*.png")
print(f"  - Data Files: {os.path.join(output_dir, 'data')}/*.csv")
print("\n" + "="*80)
print("ENHANCED ANALYSIS WITH DETAILED REPORT COMPLETE!")
print("="*80)