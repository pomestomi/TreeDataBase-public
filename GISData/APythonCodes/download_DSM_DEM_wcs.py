# Download Digital Surface Model and Digital Elevation Model via WCS
#---------------------------------------

import os
import geopandas as gpd
import requests
from owslib.wcs import WebCoverageService
import rasterio
import math
import numpy as np

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tree_loc = "CityPotsdam"
# tree_loc = "CityBerlinFrhXBerg"
# tree_loc = "CityBiberach"
# tree_loc = "CityKarben"

# model = "DEM"
model = "DSM"

city_dir = os.path.join(project_dir, "GISData", tree_loc)
tree_path = os.path.join(city_dir, "VectorLayers", "treeLocations.shp")
output_dir = os.path.join(city_dir, model)
output_path = os.path.join(output_dir, f"{model}_{tree_loc}.tif")

os.makedirs(output_dir, exist_ok=True)

if model == "DSM":
    g_name = "dom"
if model == "DEM":
    g_name = "dgm"

if tree_loc in ["CityPotsdam", "CityBerlinFrhXBerg"]:
    bundesland = "brandenburg"
    wcs_version = "1.0.0"
    crs_dom = "EPSG:25833"
    if model == "DEM":
        wcs_url = "https://isk.geobasis-bb.de/ows/dgm_wcs?"
    if model == "DSM":
        wcs_url = "https://isk.geobasis-bb.de/ows/bdom_wcs?"

if tree_loc in ["CityBiberach", "CityLüdenscheid"]:
    bundesland = "bw"
    wcs_version = "2.0.1"
    crs_dom = "EPSG:25832"
    if model == "DEM":
        wcs_url = "https://owsproxy.lgl-bw.de/owsproxy/wcs/WCS_INSP_BW_Hoehe_Coverage_DGM1?"

if tree_loc in ["CityKarben", "CityOffenbach"]:
    bundesland = "hessen"
    wcs_version = "2.0.1"
    crs_dom = "EPSG:25832"
    if model == "DEM":
        wcs_url = "https://inspire-hessen.de/raster/dgm1/ows?"
    if model == "DSM":
        wcs_url = "https://inspire-hessen.de/raster/dom1/ows?"

# --------------------------------------------------
# 2. LOAD TREE DATA
# --------------------------------------------------
# as soon as there are tree location shapefiles, this code block until else (included) can be taken out
if tree_loc == "CityKarben":
    # Create Karben dummy tree locations
    from shapely.geometry import Point

    # Google Maps coordinates (lat, lon)
    coords = [
        (50.241166957808126, 8.763262728835233),
        (50.23331163497982, 8.770854390198943),
        (50.23321512011581, 8.773794186505699)]
    # Convert to Point geometry
    geometry = [Point(lon, lat) for lat, lon in coords]
    trees = gpd.GeoDataFrame(geometry=geometry)
    trees.set_crs(epsg=4326, inplace=True)

else:
    trees = gpd.read_file(tree_path)

# Ensure correct CRS for Bundesland
trees = trees.to_crs(crs_dom)

# --------------------------------------------------
# 3. BUFFER TREES (SVF CONTEXT AREA)
# --------------------------------------------------
buffer_m = 200  # adjust if needed
trees["geometry"] = trees.geometry.buffer(buffer_m)

# --------------------------------------------------
# 4. UNION GEOMETRY → SINGLE BBOX
# --------------------------------------------------
union_geom = trees.union_all()
minx, miny, maxx, maxy = union_geom.bounds

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
if bundesland == "brandenburg":
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

# metadata check
with rasterio.open(output_path) as src:
    print(f"\n--- {model} METADATA CHECK ---")
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