#!/usr/bin/env python3
# ------------------------------------------------------------------
#  generate_access_tables_with_census.py
#  -----------------------------------------------------------------
#  * Loads mesh‑block shapefile and census SA1 tables (G01 & G33)
#  * Down‑scales SA1 counts into mesh‑blocks  ➞ avoids double‑counting
#  * Computes:
#      ① Equity ratio by income quartile
#      ② Top‑10 high‑priority mesh blocks (priority_score formula)
#      ③ Coverage × Distance‑band cross‑tab
#  * Saves results as CSV in  ./Victoria/Enhanced_Analysis_Results/tables/
# ------------------------------------------------------------------

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ───────────────────────── 1. CONFIG  ────────────────────────────
DATA_ROOT = Path("./Victoria")
MESH_SHP   = DATA_ROOT / "Melbourne/Res_melbourne/Residential_mesh_blocks.shp"
PUBLIC_VOR = DATA_ROOT / "Output-qgis/clipped-public-voronoi.shp"
PRIVATE_VOR= DATA_ROOT / "Output-qgis/clipped-private-voronoi.shp"
HOSPITALS  = DATA_ROOT / "spatial/hospital_Points.shp"

CENSUS_G01 = Path("./2021Census_G01_VIC_SA1.csv")
CENSUS_G33 = Path("./2021Census_G33_VIC_SA1.csv")

OUT_DIR    = DATA_ROOT / "Enhanced_Analysis_Results/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CRS = "EPSG:7855"          # GDA2020 / MGA 55
AMB_MIN, AMB_KMH = 15, 60
RADIUS_M  = (AMB_MIN/60)*AMB_KMH*1000  # 15‑min straight‑line

# ────────────────────── 2. LOAD GEO DATA ─────────────────────────
print("Loading spatial layers …")
try:
    mesh  = gpd.read_file(MESH_SHP).to_crs(TARGET_CRS)
    public_vor  = gpd.read_file(PUBLIC_VOR).to_crs(TARGET_CRS)
    private_vor = gpd.read_file(PRIVATE_VOR).to_crs(TARGET_CRS)
    hospitals   = gpd.read_file(HOSPITALS).to_crs(TARGET_CRS)
except Exception as e:
    sys.exit(f"❌  Failed to load shapefiles: {e}")

# ────────────────────── 3. LOAD CENSUS CSVs ──────────────────────
print("Reading census SA1 tables …")
if not (CENSUS_G01.exists() and CENSUS_G33.exists()):
    sys.exit("❌  Census CSV files not found. "
             "Place G01 & G33 files in script directory.")

g01 = pd.read_csv(CENSUS_G01, low_memory=False)
g33 = pd.read_csv(CENSUS_G33, low_memory=False)

# merge on SA1 code
census = g01.merge(g33, on="SA1_CODE_2021")

# essential counts
census["total_population"]   = census["Tot_P_P"]
census["pop_under_4"]        = census["Age_0_4_yr_P"]
census["pop_over_85"]        = census["Age_85ov_P"]
census["vulnerable_population"] = census["pop_under_4"] + census["pop_over_85"]

# income buckets → low‑income households
LOW_BUCKETS = [
    'HI_1_149_Tot','HI_150_299_Tot','HI_300_399_Tot','HI_400_499_Tot'
]
census["low_income"]       = census[LOW_BUCKETS].sum(axis=1)
census["total_households"] = census["Tot_Tot"]
census["low_income_pct"]   = census["low_income"] / census["total_households"] * 100

# ────────────────────── 4. MERGE  SA1 → MESH ─────────────────────
print("Merging census into mesh blocks …")
# detect SA1 field in mesh file
sa1_field = next((c for c in mesh.columns if "SA1" in c.upper()), None)
if sa1_field is None:
    sys.exit("❌  SA1 code not found in mesh shapefile columns.")

mesh[sa1_field]          = mesh[sa1_field].astype(str)
census["SA1_CODE_2021"]  = census["SA1_CODE_2021"].astype(str)

mesh = mesh.merge(
    census[["SA1_CODE_2021","total_population","pop_under_4",
            "pop_over_85","vulnerable_population","low_income_pct"]],
    left_on=sa1_field, right_on="SA1_CODE_2021", how="left"
)

# down‑scale SA1 counts so each mesh‑block holds an equal share
mb_per_sa1 = mesh.groupby(sa1_field)["geometry"].transform("count")
for col in ["total_population","pop_under_4","pop_over_85","vulnerable_population"]:
    mesh[col] = mesh[col] / mb_per_sa1

mesh["low_income_pct"] = mesh["low_income_pct"].fillna(0)
mesh = mesh.rename(columns={
    "total_population"     : "pop",
    "vulnerable_population": "vulnerable",
})

# ────────────────────── 5. DISTANCE & COVERAGE ───────────────────
print("Calculating hospital distances …")
mesh["distance_km"] = mesh.geometry.centroid.apply(
    lambda c: hospitals.distance(c).min()/1000)

mesh["within15"] = False
for _, h in hospitals.iterrows():
    mesh.loc[mesh.geometry.distance(h.geometry) <= RADIUS_M, "within15"] = True

# public/private flags
mesh["public_cov"], mesh["private_cov"] = False, False
for _, h in hospitals[hospitals["Type"]=="Public"].iterrows():
    mesh.loc[mesh.geometry.distance(h.geometry) <= RADIUS_M, "public_cov"]=True
for _, h in hospitals[hospitals["Type"]=="Private"].iterrows():
    mesh.loc[mesh.geometry.distance(h.geometry) <= RADIUS_M, "private_cov"]=True

mesh["coverage_type"] = np.select(
    [mesh["public_cov"]&mesh["private_cov"],
     mesh["public_cov"]&~mesh["private_cov"],
     ~mesh["public_cov"]&mesh["private_cov"]],
    ["Both","Public Only","Private Only"],
    default="None")
mesh["distance_band"] = pd.cut(
    mesh["distance_km"], bins=[0,5,10,15,20,np.inf],
    labels=["0–5 km","5–10 km","10–15 km","15–20 km","20 km+"],
    right=False)

# ────────────────────── 6. TABLE 1  (EQUITY) ─────────────────────
mesh["quartile"] = pd.qcut(mesh["low_income_pct"],4,
                           labels=["Q1 (Low)","Q2","Q3","Q4 (High)"])
mesh["far15"] = mesh["distance_km"]>15
equity = (mesh.groupby("quartile")
             .apply(lambda g: pd.Series({
                 "Total_Pop": g["pop"].sum(),
                 "Vuln_Pop" : g["vulnerable"].sum(),
                 "Total_>15km": g.loc[g["far15"],"pop"].sum(),
                 "Vuln_>15km" : g.loc[g["far15"],"vulnerable"].sum()})))
equity["Total_Far_%"] = equity["Total_>15km"]/equity["Total_Pop"]*100
equity["Vuln_Far_%"]  = equity["Vuln_>15km"]/equity["Vuln_Pop"]*100
equity["Equity_Ratio"]= equity["Vuln_Far_%"]/equity["Total_Far_%"]
equity.round(2).to_csv(OUT_DIR/"equity_ratio_quartiles.csv")
print("✔ equity_ratio_quartiles.csv saved")

# ────────────────────── 7. TABLE 2  (PRIORITY TOP‑10) ─────────────
mesh["priority_score"] = (
      mesh["vulnerable"]/mesh["vulnerable"].max()*0.5
    + mesh["distance_km"]/mesh["distance_km"].max()*0.3
    + mesh["pop"]/mesh["pop"].max()*0.2 )
top10 = (mesh.sort_values("priority_score",ascending=False)
           .head(10)[["MB_CODE21","pop","vulnerable","distance_km","priority_score"]]
           .rename(columns={"pop":"Population",
                            "vulnerable":"Vulnerable",
                            "distance_km":"Distance_km",
                            "priority_score":"Priority_Score"}))
top10.to_csv(OUT_DIR/"top10_priority_mesh_blocks.csv",index=False)
print("✔ top10_priority_mesh_blocks.csv saved")

# ────────────────────── 8. TABLE 3  (CROSS‑TAB) ──────────────────
ctab = (mesh.pivot_table(index="distance_band", columns="coverage_type",
                         values="pop", aggfunc="sum", fill_value=0).astype(int))
ctab.to_csv(OUT_DIR/"coverage_distance_band_crosstab.csv")
print("✔ coverage_distance_band_crosstab.csv saved")

print("\nAll tables exported to", OUT_DIR.resolve())
