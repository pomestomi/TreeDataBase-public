from pathlib import Path
import yaml
import subprocess
import os

# paths
# project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_dir = str(Path(__file__).resolve().parents[2])
config_path = os.path.join(project_dir, "GISData", "APythonCodes", "config.yaml")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cities = cfg["cities"]
bundeslaender = cfg["bundeslaender"]

# OPTIONS
QUIET = False          # True: suppress subscript output
ONLY_MISSING = False    # True: skip existing outputs, False: run all (if not filtered out)
# CITY_FILTER = None     # None = all cities
# examples:
# CITY_FILTER = ["CityKassel", "CityCelle"]
CITY_FILTER = ["CityCelle"]

# --------------------------------------------------
for city, bundesland in cities.items():

    if CITY_FILTER is not None:
        if city not in CITY_FILTER:
            continue

    city_dir = os.path.join(project_dir, "GISData", city)
    tree_path = os.path.join(city_dir, "VectorLayers", "treeLocations.shp")
    dsm_dir = os.path.join(city_dir, "DSM")
    dem_dir = os.path.join(city_dir, "DEM")
    output_tree_path = os.path.join(city_dir, "SVF", "trees_svf.shp")

    # skip existing
    if ONLY_MISSING and os.path.exists(output_tree_path):
        print(f"Skipping existing: {city}")
        continue

    # prerequisite check
    if not os.path.exists(tree_path):
        print(f"Missing tree layer: {city}")
        continue

    if not os.path.exists(dsm_dir):
        print(f"Missing DSM: {city}")
        continue

    if not os.path.exists(dem_dir):
        print(f"Missing DEM: {city}")
        continue

    print("-----------------------------------")
    print(f"Processing SVF: {city}")

    cmd = [
        "python",
        os.path.join(project_dir, "GISData", "APythonCodes", "calculateSVFs_treebuffers.py"),
        "--city", city]

    if QUIET:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    else:
        result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Failed: {city}")
    else:
        print(f"Done: {city}")