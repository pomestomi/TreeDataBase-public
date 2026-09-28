#!/usr/bin/env python3
"""
CityCelle pipeline completion script.
Run this after closing QGIS — it patches the fresh terrain and SVF values
into treeLocations.shp, then re-runs land use and rebuilds all_tree_locations.csv.

Steps executed:
  1. Patch terrain columns (tpi, twi, slope, elevation, flowAcc) from .gpkg → .shp
  2. Patch SVF and elevation from SVF/trees_svf.shp → .shp
  3. Re-run compute_landuse.py --city CityCelle (writes land use to .shp)
  4. Re-run urban_tree_report.py (rebuilds all_tree_locations.csv)
"""
import subprocess
import sys
from pathlib import Path

import geopandas as gpd

PYTHON = sys.executable
REPO   = Path(__file__).resolve().parent.parent.parent
GIS    = REPO / "GISData"
VL     = GIS / "CityCelle" / "VectorLayers"
SHP    = VL  / "treeLocations.shp"
GPKG   = VL  / "treeLocations.gpkg"
SVF    = GIS / "CityCelle" / "SVF" / "trees_svf.shp"


def run(script: Path, *args):
    print(f"\n=== {script.name} {' '.join(args)} ===")
    r = subprocess.run([PYTHON, str(script), *args])
    if r.returncode != 0:
        print(f"ERROR: {script.name} exited {r.returncode}")
        sys.exit(r.returncode)


# ── Step 1: patch terrain from .gpkg ──────────────────────────────────────────
print("=== Step 1: Patch terrain columns from .gpkg into .shp ===")
gpkg_gdf = gpd.read_file(GPKG)
shp_gdf  = gpd.read_file(SHP)

TERRAIN_COLS = ["tpi2m5", "tpi5m", "tpi7m5", "twi", "slope", "elevation",
                "tpiCD", "flowAcc"]
patched = []
for col in TERRAIN_COLS:
    if col in gpkg_gdf.columns:
        shp_gdf[col] = gpkg_gdf[col].values
        patched.append(col)
# flowAcc is a new column (previously named flotAcc with typo); keep old one too
if "flowAcc" in gpkg_gdf.columns and "flotAcc" in shp_gdf.columns:
    shp_gdf["flotAcc"] = gpkg_gdf["flowAcc"].values
    if "flotAcc" not in patched:
        patched.append("flotAcc←flowAcc")

shp_gdf.to_file(SHP, driver="ESRI Shapefile")
print(f"  Patched: {patched}")

# ── Step 2: patch SVF + elevation from trees_svf.shp ──────────────────────────
print("\n=== Step 2: Patch SVF and elevation from trees_svf.shp ===")
svf_gdf = gpd.read_file(SVF)
shp_gdf = gpd.read_file(SHP)

shp_gdf["SVF"]       = svf_gdf["SVF_mod"].values
shp_gdf["elevation"] = svf_gdf["DEM"].values

shp_gdf.to_file(SHP, driver="ESRI Shapefile")
print(f"  SVF range : {svf_gdf['SVF_mod'].min():.3f} – {svf_gdf['SVF_mod'].max():.3f}")
print(f"  Elev range: {svf_gdf['DEM'].min():.1f} – {svf_gdf['DEM'].max():.1f} m")

# ── Step 3: re-run compute_landuse.py ─────────────────────────────────────────
run(REPO / "src" / "compute_landuse.py", "--city", "CityCelle")

# ── Step 4: rebuild all_tree_locations.csv ────────────────────────────────────
run(REPO / "DatasetStatistics" / "urban_tree_report.py")

print("\nDone — CityCelle pipeline complete.")
