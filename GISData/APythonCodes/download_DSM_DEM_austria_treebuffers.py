import os
import geopandas as gpd
import requests
import xml.etree.ElementTree as ET
import rasterio
import numpy as np
import re
from shapely.geometry import Polygon, mapping
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.warp import reproject
import math
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

script = bl_cfg["script"]
method = bl_cfg.get("method", None)
wcs_url = model_cfg.get("wcs_url")
coverage_id = model_cfg.get("coverage_id")
wcs_version = bl_cfg.get("wcs_version")
atom_url = model_cfg.get("atom_url")
g_name = model_cfg.get("g_name")

# paths
city_dir = os.path.join(project_dir, "GISData", tree_loc)
tree_path = os.path.join(city_dir, "VectorLayers", "treeLocations.shp")
output_dir = os.path.join(city_dir, model)

os.makedirs(output_dir, exist_ok=True)

if model == "DSM":
    g_name = "als_dsm"
if model == "DEM":
    g_name = "als_dtm"

# --------------------------------------------------
# trees
# --------------------------------------------------
trees = gpd.read_file(tree_path)

# Ensure correct CRS for Bundesland
trees = trees.to_crs(crs_dom)

# buffer trees
buffer_m = 200  # adjust if needed (SVF influence radius)
tree_buffers = trees.copy()
tree_buffers["geometry"] = tree_buffers.geometry.buffer(buffer_m)

search_geom = tree_buffers.geometry.union_all()

search_geom_wgs = (
    gpd.GeoSeries([search_geom], crs=crs_dom)
    .to_crs("EPSG:4326")
    .iloc[0])

# --------------------------------------------------
# ATOM feed
# --------------------------------------------------
root = ET.fromstring(requests.get(atom_url).content)

tile_candidates = []

# --------------------------------------------------
# collect intersecting tiles
# --------------------------------------------------

for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):

    title = entry.findtext("{http://www.w3.org/2005/Atom}title")

    if model == "DEM" and "DTM" not in title:
        continue

    if model == "DSM" and "DSM" not in title:
        continue

    poly = entry.find("{http://www.georss.org/georss}polygon")

    if poly is None:
        continue

    coords = list(map(float, poly.text.split()))
    pts = [(coords[i + 1], coords[i]) for i in range(0, len(coords), 2)]

    tile_geom = Polygon(pts)

    # intersect AOI
    if not tile_geom.intersects(search_geom_wgs):
        continue

    # dataset link
    ds_url = None

    for link in entry.findall("{http://www.w3.org/2005/Atom}link"):

        if link.attrib.get("rel") == "alternate":
            ds_url = link.attrib["href"]
            break

    if ds_url is None:
        continue

    # extract spatial tile id
    tile_match = re.search(
        r"N\d+E\d+",
        title
    )

    if tile_match is None:
        continue

    tile_id = tile_match.group(0)

    # extract date from title
    date_match = re.search(
        r"(\d{2})\.(\d{2})\.(\d{4})",
        title
    )

    if date_match:
        day, month, year = date_match.groups()
        date = int(f"{year}{month}{day}")

    else:
        date = -1

    tile_candidates.append({
        "tile_id": tile_id,
        "date": date,
        "ds_url": ds_url
    })

# --------------------------------------------------
# keep newest version of each tile
# --------------------------------------------------

latest_tiles = {}

for tile in tile_candidates:

    tid = tile["tile_id"]

    if tid not in latest_tiles:
        latest_tiles[tid] = tile

    elif tile["date"] > latest_tiles[tid]["date"]:
        latest_tiles[tid] = tile

if not latest_tiles:
    raise ValueError("No matching tiles found")

print(f"Using {len(latest_tiles)} tile(s)")

# --------------------------------------------------
# PROCESS Trees ONE-BY-ONE
# --------------------------------------------------
buffers_src = tree_buffers.copy()
target_res = 1.0
nodata_val = -9999

for idx, row in buffers_src.iterrows():

    geom = row.geometry

    # build output raster
    minx, miny, maxx, maxy = geom.bounds

    minx = math.floor(minx / target_res) * target_res
    miny = math.floor(miny / target_res) * target_res
    maxx = math.ceil(maxx / target_res) * target_res
    maxy = math.ceil(maxy / target_res) * target_res

    width = int(round((maxx - minx) / target_res))
    height = int(round((maxy - miny) / target_res))

    transform = rasterio.transform.from_origin(
        minx, maxy, target_res, target_res
    )

    tree_output = os.path.join(
        output_dir,
        f"{model}_tree_{idx:03d}.tif"
    )

    acc = np.full((height, width), nodata_val, dtype="float32")

    for tid, tile in latest_tiles.items():

        ds_root = ET.fromstring(requests.get(tile["ds_url"]).content)
        print(tile["ds_url"])

        tif_url = None

        for e in ds_root.findall(".//{http://www.w3.org/2005/Atom}entry"):
            for l in e.findall("{http://www.w3.org/2005/Atom}link"):
                href = l.attrib.get("href", "")
                if href.lower().endswith(".tif"):
                    tif_url = href
                    break

        if tif_url is None:
            continue

        with rasterio.open(tif_url) as src:

            tile_box = Polygon([
                (src.bounds.left, src.bounds.bottom),
                (src.bounds.left, src.bounds.top),
                (src.bounds.right, src.bounds.top),
                (src.bounds.right, src.bounds.bottom)
            ])

            if not geom.intersects(tile_box):
                continue

            masked, masked_transform = mask(
                src,
                [mapping(geom)],
                crop=True,
                nodata=nodata_val
            )

            data = masked[0].astype("float32")

            if model == "DSM":
                data[data <= 0] = nodata_val

            if np.all(data == nodata_val):
                continue

            tmp = np.full((height, width), nodata_val, dtype="float32")

            reproject(
                source=data,
                destination=tmp,
                src_transform=masked_transform,
                # src_crs=src.crs,
                src_crs=crs_dom,
                dst_transform=transform,
                dst_crs=crs_dom,
                src_nodata=nodata_val,
                dst_nodata=nodata_val,
                resampling=Resampling.max,
                init_dest_nodata=False
            )

            # MERGE instead of overwrite
            valid = tmp != nodata_val
            acc[valid] = np.maximum(acc[valid], tmp[valid])

    dst = rasterio.open(
        tree_output,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs_dom,
        transform=transform,
        compress="lzw",
        nodata=nodata_val,
        tiled=True
    )

    dst.write(acc, 1)
    dst.close()

#---------------------------------------------------------
# metadata check:

    with rasterio.open(tree_output) as src:
        print(f"\n--- {model} METADATA CHECK {tree_loc}, tree {idx}---")
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