# Download Digital Surface Model and Digital Elevation Model via WCS
#---------------------------------------

import os
import geopandas as gpd
import requests
from owslib.wcs import WebCoverageService
import rasterio
import math
import numpy as np
from shapely.geometry import mapping
from rasterio.mask import mask
import yaml
import argparse

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--city", required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()

tree_loc = args.city
model = args.model

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

config_path = os.path.join(project_dir, "GISData", "APythonCodes", "config.yaml")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

bundesland = cfg["cities"][tree_loc]
bl_cfg = cfg["bundeslaender"][bundesland]
crs_dom = bl_cfg["crs"]
model_cfg = bl_cfg[model]

wcs_url = model_cfg.get("wcs_url")
coverage_id = model_cfg.get("coverage_id")
wcs_version = bl_cfg.get("wcs_version")

# paths
city_dir = os.path.join(project_dir, "GISData", tree_loc)
tree_path = os.path.join(city_dir, "VectorLayers", "treeLocations.shp")
output_dir = os.path.join(city_dir, model)
output_path = os.path.join(output_dir, f"{model}_{tree_loc}.tif")

os.makedirs(output_dir, exist_ok=True)

if model == "DSM":
    g_name = "dom"
if model == "DEM":
    g_name = "dgm"

# 2. LOAD TREE DATA
# --------------------------------------------------
trees = gpd.read_file(tree_path)

# Ensure correct CRS for Bundesland
trees = trees.to_crs(crs_dom)

# --------------------------------------------------
# 3. BUFFER TREES (SVF CONTEXT AREA)
# --------------------------------------------------
buffer_m = 200  # adjust if needed
tree_buffers = trees.copy()
tree_buffers["geometry"] = tree_buffers.geometry.buffer(buffer_m)

# union geometry
aoi = tree_buffers.geometry.union_all()
aoi = aoi.buffer(5) # meters
# --------------------------------------------------
# 4. UNION GEOMETRY → SINGLE BBOX
# --------------------------------------------------
minx, miny, maxx, maxy = aoi.bounds

# --------------------------------------------------
# 5. CONNECT TO WCS
# --------------------------------------------------
wcs = WebCoverageService(wcs_url, version= wcs_version)
coverage_id = list(wcs.contents)[0]

# --------------------------------------------------
# 6. Output grid setup
# --------------------------------------------------
target_res = 1.0  # meters
nodata_val = -9999

# Snap to resolution
minx = math.floor(minx / target_res) * target_res
miny = math.floor(miny / target_res) * target_res
maxx = math.ceil(maxx / target_res) * target_res
maxy = math.ceil(maxy / target_res) * target_res

# Define output grid width, height
width = int(round((maxx - minx) / target_res))
height = int(round((maxy - miny) / target_res))

# --------------------------------------------------
# 7. BUILD WCS REQUEST
# --------------------------------------------------
if bundesland in ["berlin", "brandenburg"]:
    params = {
        "SERVICE": "WCS",
        "VERSION": "1.0.0",
        "REQUEST": "GetCoverage",
        "COVERAGE": coverage_id,
        "CRS": crs_dom,
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/tiff"}

if bundesland in ["bw", "hessen"]:

    epsg_uri = f"http://www.opengis.net/def/crs/EPSG/0/{crs_dom.split(':')[1]}"

    params = [
        ("SERVICE", "WCS"),
        ("VERSION", "2.0.1"),
        ("REQUEST", "GetCoverage"),
        ("COVERAGEID", coverage_id),
        ("FORMAT", "image/tiff"),
        # axis-specific subsets with CRS URI
        ("SUBSET", f"N,{epsg_uri}({miny},{maxy})"),
        ("SUBSET", f"E,{epsg_uri}({minx},{maxx})"),
    ]

print(requests.get(wcs_url, params=params).url)

if bundesland == "bw":
    width = int(round((maxx - minx) / target_res))
    height = int(round((maxy - miny) / target_res))
    print(f"Expected size: {width} x {height} pixels \n"
      "max permitted for bw: 25.000 x 25.000 pixel")

# --------------------------------------------------
# 8. DOWNLOAD FILE
# --------------------------------------------------
response = requests.get(wcs_url, params=params)

if response.status_code == 200:
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"{model} successfully downloaded:")
    print(output_path)
# assign coordinate system and nodata to raster
    with rasterio.open(output_path) as src:
        data = src.read(1).astype("float32")

        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            nodata=-9999.0,
            crs=crs_dom)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)
else:
    print("Download failed!")
    print("Status code:", response.status_code)
    print(response.text)

# --------------------------------------------------
# Process trees
# --------------------------------------------------

buffers_src = tree_buffers.copy()

with rasterio.open(output_path) as src:

    for idx, row in buffers_src.iterrows():

        geom = row.geometry
        tree_output = os.path.join(output_dir, f"{model}_tree_{idx:03d}.tif")

        masked, masked_transform = mask(src, [mapping(geom)], crop=True, nodata=nodata_val)

        data = masked[0].astype("float32")

        if model == "DSM":
            data[data <= 0] = nodata_val

        # output profile
        profile = src.profile.copy()

        profile.update({
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": masked_transform,
            "dtype": "float32",
            "nodata": nodata_val,
            "compress": "lzw",
            "tiled": False
        })

        # save tree raster
        with rasterio.open(tree_output, "w", **profile) as dst:
            dst.write(data, 1)

# metadata check:

        with rasterio.open(tree_output) as tree_src:
            print(f"\n--- {model} METADATA CHECK {tree_loc}, tree {idx}---")
            print("CRS:", tree_src.crs)
            print("Resolution:", tree_src.res)
            print("Width x Height:", tree_src.width, "x", tree_src.height)
            print("NoData value:", tree_src.nodata)

            data = tree_src.read(1).astype("float32")

            valid = (data != tree_src.nodata)
            data_valid = data[valid]

            if data_valid.size == 0:
                print("No valid data found!")
            else:
                print("Min:", np.min(data_valid))
                print("Max:", np.max(data_valid))
                print("Mean:", np.mean(data_valid))
                print("Median:", np.median(data_valid))

