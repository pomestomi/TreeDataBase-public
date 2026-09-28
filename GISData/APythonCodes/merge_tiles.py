# Merge tif tiles from folder
#--------------------------------------------
import os
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import math
# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# tree_loc = "CityBamberg" #-> bavaria
# tree_loc = "CityGarbsen" #-> niedersachsen
tree_loc = "CityBerlinFrhXBerg"
# output_folder = "DEMBamberg"
output_folder = "DEMBerlin"
output_model = "DEM"
model_keyword = "dgm"

city_dir = os.path.join(project_dir, "GISData", tree_loc)
tree_path = os.path.join(city_dir, "VectorLayers", "treeLocations.shp")
output_dir = os.path.join(city_dir, output_folder)
output_path = os.path.join(output_dir, f"{output_model}_{tree_loc}.tif")

if tree_loc in ["CityBamberg", "CityErlangen"]:
    bundesland = "bavaria"
    crs_dom = "EPSG:25832"
if tree_loc in ["CityGarbsen", "CityHanover"]:
    bundesland = "niedersachsen"
    crs_dom = "EPSG:25832"
if tree_loc in ["CityBerlinFrhXBerg"]:
    crs_dom = "EPSG:25833"

# --------------------------------------------------
# 2. Prepare tree data
# --------------------------------------------------
trees = gpd.read_file(tree_path)
# Ensure correct CRS for Bundesland
trees = trees.to_crs(crs_dom)

# buffer trees
buffer_m = 200  # adjust if needed (SVF influence radius)
trees["geometry"] = trees.geometry.buffer(buffer_m)

# union geometry
aoi = trees.union_all()
# Compute AOI bounds
minx, miny, maxx, maxy = aoi.bounds

# --------------------------------------------------
# 3. Output grid setup
# --------------------------------------------------
target_res = 1.0 #1m
nodata_val = -9999

# Snap to resolution
minx = math.floor(minx / target_res) * target_res
miny = math.floor(miny / target_res) * target_res
maxx = math.ceil(maxx / target_res) * target_res
maxy = math.ceil(maxy / target_res) * target_res

# Define output grid width, height
width = int((maxx - minx) / target_res)
height = int((maxy - miny) / target_res)

# initialize output raster
dst_array = np.full((height, width), nodata_val, dtype="float32")

# define CRS
dst_crs = crs_dom

transform = rasterio.transform.from_origin(
    minx, maxy, target_res, target_res)

# --------------------------------------------------
# 4. Mosaic tiles
# --------------------------------------------------
tile_files = [
    os.path.join(output_dir, f)
    for f in os.listdir(output_dir)
    if f.endswith(".tif") and model_keyword in f.lower()]

for fp in tile_files:
    print("Processing:", os.path.basename(fp))

    with rasterio.open(fp) as src:

        # fast overlap check
        if not (
            src.bounds.left < maxx and
            src.bounds.right > minx and
            src.bounds.bottom < maxy and
            src.bounds.top > miny
        ):
            print("Skipping (no overlap):", os.path.basename(fp))
            continue

        data = src.read(1).astype("float32")

        data[data <= 0] = np.nan  # DSM-safe rule

        temp = np.full((height, width), nodata_val, dtype="float32")

        reproject(
            source=data,
            destination=temp,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            src_nodata=nodata_val,
            dst_nodata=nodata_val,
            resampling=Resampling.max
        )

        mask_valid = temp != nodata_val

        dst_array[mask_valid] = np.maximum(
            dst_array[mask_valid],
            temp[mask_valid]
        )

# write once at the end
with rasterio.open(
    output_path,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype="float32",
    crs=dst_crs,
    transform=transform,
    compress="lzw",
    nodata=nodata_val
) as dst:
    dst.write(dst_array, 1)

print("Done: raster created at 1m resolution")

with rasterio.open(output_path) as src:
    print("\n--- raster METADATA CHECK ---")
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Width x Height:", src.width, "x", src.height)
    print("NoData value:", src.nodata)

    data = src.read(1).astype("float32")

    valid = (data != src.nodata)
    data_valid = data[valid]

    if data_valid.size == 0:
        print("No valid data found!")
    else:
        print("Min:", np.min(data_valid))
        print("Max:", np.max(data_valid))
        print("Mean:", np.mean(data_valid))
        print("Median:", np.median(data_valid))