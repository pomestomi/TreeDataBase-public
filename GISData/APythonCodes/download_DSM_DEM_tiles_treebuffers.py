# Download Digital Surface Model via Metalink
#--------------------------------------------

import os
import math
import re
import sys
import requests
import rasterio
import zipfile
import numpy as np
import geopandas as gpd
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, parse_qs
from pyproj import Transformer
from shapely.geometry import box, mapping
from rasterio.mask import mask
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import yaml
import argparse

# https problem...
import ssl
import certifi
# Override the default SSL context for all urllib-based HTTPS requests
ssl._create_default_https_context = (
    lambda: ssl.create_default_context(cafile=certifi.where()))


# os.environ["SHAPE_RESTORE_SHX"] = "YES"

parser = argparse.ArgumentParser()
parser.add_argument("--city", required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()

tree_loc = args.city
model = args.model

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
config_path = os.path.join(project_dir, "GISData", "APythonCodes", "config.yaml")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

bundesland = cfg["cities"][tree_loc]
bl_cfg = cfg["bundeslaender"][bundesland]
crs_dom = bl_cfg["crs"]
model_cfg = bl_cfg[model]
method = model_cfg.get("method") or bl_cfg["method"]


# paths
city_dir = os.path.join(project_dir, "GISData", tree_loc)
tree_path = os.path.join(city_dir, "VectorLayers", "treeLocations.shp")
output_dir = os.path.join(city_dir, model)
output_path = os.path.join(output_dir, f"{model}_{tree_loc}.tif")
tmp_dir = os.path.join(output_dir, "tiles")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

if model == "DSM":
    g_name = "dom"
if model == "DEM":
    g_name = "dgm"


# define function to select needed tiles from rheinlandpfalz, sachsen
def tile_geom_from_name(name, bundesland):
    if bundesland == "rlp":
        m = re.search(r"(dom1|dgm1)_\d+_(\d+)_(\d+)_", name)
        tile_size = 1000  # 1 km

    elif bundesland == "sachsen":
        m = re.search(r"(dgm1|dom1)_\d{2}(\d+)_(\d+)_\d+_", name)
        tile_size = 2000  # 2 km Kacheln

    elif bundesland == "bremen":
        m = re.search(r"(dgm1|dom1)_\d{2}(\d{3})_(\d{4})_", name)
        tile_size = 1000

    elif bundesland == "bw":
        m = re.search(r"(dgm1|dom1)_\d{2}_(\d{3})_(\d{4})_", name)
        tile_size = 2000

    elif bundesland == "saarland":
        m = re.search(r"dom1_\d+_\d+_(\d+)_(\d+)_", name)
        tile_size = 1000

    if not m:
        return None
    x, y = map(int, m.groups()[1:])

    # 100m raster -> meter
    xmin = x * 1000
    ymin = y * 1000
    xmax = xmin + tile_size
    ymax = ymin + tile_size

    return box(xmin, ymin, xmax, ymax)

# define function to transform xyz to raster
def xyz_to_tif(xyz_path, tif_path, crs):
    with open(xyz_path, "r") as _fh:
        first_line = _fh.readline().strip().split()
    _skip = 0 if first_line[0].replace(".", "").replace("-", "").isdigit() else 1

    # Special handling for Schleswig-Holstein (SH) files
    if bundesland == "sh":
        data = np.loadtxt(xyz_path, skiprows=_skip, max_rows=1000000)
    else:
        data = np.loadtxt(xyz_path, skiprows=_skip)

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    x_unique = np.unique(x)
    y_unique = np.unique(y)

    nx = len(x_unique)
    ny = len(y_unique)

    grid = z.reshape(ny, nx)

    flip_needed = not (bundesland == "sh" and model == "DEM")
    if flip_needed:
        grid = np.flipud(grid)

    res = x_unique[1] - x_unique[0]

    transform = from_origin(
        x_unique.min(), y_unique.max(), res, res)

    with rasterio.open(
        tif_path,
        "w", driver="GTiff",
        height=ny, width=nx,
        count=1, dtype="float32",
        crs=crs, transform=transform,
        compress="lzw"
    ) as dst:
        dst.write(grid.astype("float32"), 1)

#--------------------------------------------------------------------------------
# Prepare tree data
trees = gpd.read_file(tree_path)

# Ensure correct CRS for Bundesland
trees = trees.to_crs(crs_dom)

# buffer trees
buffer_m = 200  # adjust if needed (SVF influence radius)
tree_buffers = trees.copy()
tree_buffers["geometry"] = tree_buffers.geometry.buffer(buffer_m)

# union geometry
aoi = tree_buffers.geometry.union_all()
# Compute AOI bounds
minx, miny, maxx, maxy = aoi.bounds
# aoi.is_valid # option to check if geometry is valid, apply next line and recheck if not
# aoi = aoi.buffer(0)

# --------------------------------------------------
# a) meta4
# --------------------------------------------------
# Create EWKT string
if method == "meta4":
    if bundesland == "bayern":
        ewkt = f"SRID={trees.crs.to_epsg()};{aoi.wkt}"

    # Save to file for reproducibility
        ewkt_path = os.path.join(output_dir, "ewkt_trees_buffer200.txt")
        with open(ewkt_path, "w") as f:
            f.write(ewkt)

#-----------------------------------------------------------------------
# if bavaria, DOM: manually create .meta4 file on this homepage: https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=dom20
# if bavaria, DGM: manually create .meta4 file on this homepage:     https: // geodaten.bayern.de / opengeodata / OpenDataDetail.html?pn = dgm1
# if rheinlandpfalz: download .meta4 file for all the Bundesland here: https://geoshop.rlp.de/opendata-dom1.html
#-----------------------------------------------------------------------

# load meta4 file
    ns = {"m": "urn:ietf:params:xml:ns:metalink"}
    meta4_path = model_cfg["meta4_path"]
    tree = ET.parse(os.path.join(output_dir, meta4_path))
    root = tree.getroot()

    tiles = []

    for f in root.findall("m:file", ns):
        name = f.attrib["name"]

        if not name.lower().endswith(".tif"):
            continue

        urls = [u.text for u in f.findall("m:url", ns)]
        url = urls[0]  # first mirror

        if bundesland == "rlp":
            geom = tile_geom_from_name(name, bundesland)

            if geom is None:
                continue

            # IMPORTANT: AOI must be in same CRS (UTM32)
            if not geom.intersects(aoi.buffer(1)):
                continue

        tiles.append((name, url))

# --------------------------------------------------
# b) geojson
# --------------------------------------------------
if method == "geojson_index":
    # load tile index
    tile_index_url = model_cfg["tile_index_url"]
    tiles_gdf = gpd.read_file(tile_index_url)
    tiles_gdf = tiles_gdf.to_crs(crs_dom)

    # select tiles intersecting aoi
    tiles_sel = tiles_gdf[tiles_gdf.intersects(aoi)]

    if bundesland == "nds":
        row_name = model_cfg["row_name"]
        tiles = [
        (os.path.basename(row[row_name]), row[row_name])
        for _, row in tiles_sel.iterrows()]

    elif bundesland == "sh":
        tiles = []

        for _, row in tiles_sel.iterrows():
            url = row["link_data"]

            parsed = urlparse(url)
            qs = parse_qs(parsed.query)

            if "file" in qs:
                name = qs["file"][0]
            else:
                raise ValueError("No file parameter in URL")

            tiles.append((name, url))


# --------------------------------------------------
# c) WCS tiling (nrw)
# --------------------------------------------------
if method == "wcs_tiles":
    coverage_id = model_cfg["coverage_id"]

    if bundesland == "nrw":
        tile_size = 2000  # meters (nrw wcs pixel limit)

    x_tiles = np.arange(minx, maxx, tile_size)
    y_tiles = np.arange(miny, maxy, tile_size)

    tiles = []

    for x0 in x_tiles:
        for y0 in y_tiles:
            x1 = min(x0 + tile_size, maxx)
            y1 = min(y0 + tile_size, maxy)

            name = f"{g_name}_{int(x0)}_{int(y0)}.tif"

            params = [
                ("SERVICE", "WCS"),
                ("VERSION", "2.0.1"),
                ("REQUEST", "GetCoverage"),
                ("COVERAGEID", coverage_id),
                ("FORMAT", "image/tiff"),
                ("SUBSET", f"x({x0},{x1})"),
                ("SUBSET", f"y({y0},{y1})"),
                ("SCALEFACTOR", "1")]

            tiles.append((name, params))
# --------------------------------------------------
# d) Sachsen tiles
# --------------------------------------------------
# manual step (only once): download tile urls for all the state of Sachsen from
# https://www.geodaten.sachsen.de/batch-download-4719.html

if method == "sachsen_tiles":
    url_file = model_cfg["url_file"]
    url_path = os.path.join(output_dir, url_file)
    if not os.path.exists(url_path):
        sys.exit(
            f"Missing file: {url_path}\n"
            "Download it manually from:\n"
            "https://www.geodaten.sachsen.de/batch-download-4719.html"
        )
    with open(url_path, "r") as f:
        tile_urls = [line.strip() for line in f if line.strip()]

    tiles = []

    for url in tile_urls:

        name = os.path.basename(url)
        geom = tile_geom_from_name(name, bundesland)

        if geom is None:
            continue

        if not geom.intersects(aoi.buffer(1)):
            continue

        tiles.append((name, url))

# --------------------------------------------------
# e) Thüringen ATOM feed
# --------------------------------------------------
if method == "atom_feed_th":

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    atom_url = model_cfg["atom_url"]
    root = ET.fromstring(requests.get(atom_url).content)
    transformer = Transformer.from_crs("EPSG:4326", crs_dom, always_xy=True)

    tiles = []

    for entry in root.findall("atom:entry", ns):
        for link in entry.findall("atom:link", ns):

            if link.attrib.get("rel") == "section":
                url = link.attrib["href"]
                name = os.path.basename(url)
                if "2020-2025" not in url:
                    continue
                bbox_str = link.attrib.get("bbox")
                if not bbox_str:
                    continue

                # bbox format: "lat_min lon_min lat_max lon_max"
                lat_min, lon_min, lat_max, lon_max = map(float, bbox_str.split())

                # convert to projected CRS
                xmin, ymin = transformer.transform(lon_min, lat_min)
                xmax, ymax = transformer.transform(lon_max, lat_max)

                tile_geom = box(xmin, ymin, xmax, ymax)

                if not tile_geom.intersects(aoi.buffer(1)):
                    continue

                tiles.append((name, url))

# --------------------------------------------------
# f) Bremen single Zip xyz
# --------------------------------------------------
if method == "bremen_xyz_zip":
    zip_url = model_cfg["zip_url"]
    zip_name = os.path.basename(zip_url)
    zip_path = os.path.join(tmp_dir, zip_name)

    # download once
    if not os.path.exists(zip_path):

        print("Downloading Bremen ZIP...")

        r = requests.get(zip_url, stream=True)
        r.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)

    tiles = []

    with zipfile.ZipFile(zip_path, "r") as zip_ref:

        members = zip_ref.namelist()

        for member in members:
            if not member.lower().endswith(".xyz"):
                continue
            name = os.path.basename(member)
            geom = tile_geom_from_name(name, bundesland)
            if geom is None:
                continue
            if not geom.intersects(aoi.buffer(1)):
                continue

            print("Extracting:", name)
            out_xyz = os.path.join(tmp_dir, name)
            if not os.path.exists(out_xyz):
                with zip_ref.open(member) as src, open(out_xyz, "wb") as dst:
                    dst.write(src.read())
            tif_path = out_xyz.replace(".xyz", ".tif")

            if not os.path.exists(tif_path):
                print("Converting XYZ -> TIF:", name)
                xyz_to_tif(out_xyz, tif_path, crs_dom)

            tiles.append((name, tif_path))

# --------------------------------------------------
# g) BW DSM Zip tifs
# --------------------------------------------------
if method == "bw_DSM_zip":

    zip_files = [
        f for f in os.listdir(tmp_dir)
        if f.lower().endswith(".zip")
    ]

    tiles = []

    for zip_name in zip_files:

        geom = tile_geom_from_name(zip_name, bundesland)

        if geom is None:
            continue

        # skip ZIPs outside AOI
        if not geom.intersects(aoi.buffer(1)):
            continue

        zip_path = os.path.join(tmp_dir, zip_name)

        print("Using BW ZIP:", zip_name)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                if not member.lower().endswith(".tif"):
                    continue
                tif_name = os.path.basename(member)
                out_path = os.path.join(tmp_dir, tif_name)

                # extract once
                if not os.path.exists(out_path):
                    print("Extracting:", tif_name)
                    with zip_ref.open(member) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())

                tiles.append((tif_name, out_path))


# --------------------------------------------------
# h) Saarland DSM Zip tifs
# --------------------------------------------------
if method == "sl_zip":

    tiles = []
    if tree_loc == "CityNeunkirchen":
        lk = "NK"
    if tree_loc == "CompanyHomburg":
        lk = "SPK"
    if tree_loc == "CompanySaarbrücken":
        lk = "SB"

    zip_template = model_cfg["zip_url"]

    zip_url_lk = zip_template.format(lk=lk)
    zip_name = f"DOM1_tif_{lk}_EPSG-25832_Entstehung-2025.zip"
    zip_path = os.path.join(tmp_dir, zip_name)

    # -----------------------------
    # download each Landkreis ZIP
    # -----------------------------
    if not os.path.exists(zip_path):

        print(f"Downloading Saarland DSM ({lk})")

        r = requests.get(zip_url_lk, stream=True)
        r.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)

    # -----------------------------
    # extract tif tiles
    # -----------------------------
    print(f"extracting Landkreis {lk}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            if not member.lower().endswith(".tif"):
                continue
            name = os.path.basename(member)
            out_path = os.path.join(tmp_dir, name)
            if not os.path.exists(out_path):
                with zip_ref.open(member) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())

            tiles.append((name, out_path))

# --------------------------------------------------
# i) BW DEM XYZ tiles placed manually in subfolders
# --------------------------------------------------
if method == "bw_xyz_subfolder":
    from pathlib import Path as _Path
    tiles = []
    for xyz_path in sorted(_Path(tmp_dir).rglob("*.xyz")):
        tif_path = xyz_path.with_suffix(".tif")
        geom = tile_geom_from_name(xyz_path.name, bundesland)
        if geom is not None and not geom.intersects(aoi.buffer(1)):
            continue
        if not tif_path.exists():
            print("Converting XYZ -> TIF:", xyz_path.name)
            xyz_to_tif(str(xyz_path), str(tif_path), crs_dom)
        tiles.append((tif_path.name, str(tif_path)))

#---------------------------------------------------------
# Download tiles
print(f"Tiles found: {len(tiles)}")

if method in ["bremen_xyz_zip", "bw_DSM_zip", "sl_zip", "bw_xyz_subfolder"]:
    print(f"{tree_loc} tiles already extracted")
else:
    for name, source in tiles:
        tile_output_path = os.path.join(tmp_dir, name)

        if os.path.exists(tile_output_path):
            # xyz was downloaded before but tif conversion may have been missed
            if tile_output_path.lower().endswith(".xyz"):
                tif_path = tile_output_path.replace(".xyz", ".tif")
                if not os.path.exists(tif_path):
                    print("Converting existing XYZ -> TIF:", name)
                    xyz_to_tif(tile_output_path, tif_path, crs_dom)
            continue

        print("Downloading:", name)

        if method == "wcs_tiles":
            wcs_url = model_cfg["wcs_url"]
            r = requests.get(wcs_url, params=source)

        else:
            url = source
            r = requests.get(url, stream=True)

        r.raise_for_status()

        with open(tile_output_path, "wb") as f:
            if method == "wcs_tiles":
                f.write(r.content)
            else:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
    # zip handling
        if tile_output_path.lower().endswith(".zip"):
            with zipfile.ZipFile(tile_output_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if member.lower().endswith(".tif"):
                        out_path = os.path.join(tmp_dir, os.path.basename(member))
                        with zip_ref.open(member) as src, open(out_path, "wb") as dst:
                            dst.write(src.read())

    # xyz handling
        elif tile_output_path.lower().endswith(".xyz"):

                tif_path = tile_output_path.replace(".xyz", ".tif")
                print("Converting XYZ -> TIF:", name)
                xyz_to_tif(tile_output_path, tif_path, crs_dom)

    print("Download complete")

# --------------------------------------------------
# Process trees
buffers_src = tree_buffers.copy()
target_res = 1.0
nodata_val = -9999

for idx, row in buffers_src.iterrows():
    print("\n-----------------------------------")
    print(f"processing tree {idx}")

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

    tree_output = os.path.join(output_dir, f"{model}_tree_{idx:03d}.tif")

    # accumulator raster
    acc = np.full((height, width), nodata_val, dtype="float32")

    from pathlib import Path as _Path
    for _tif in sorted(_Path(tmp_dir).rglob("*.tif")):
        f = _tif.name
        fp = str(_tif)

        with rasterio.open(fp) as src:

            tile_geom = box(*src.bounds)

            if not geom.intersects(tile_geom):
                continue

            print("Using tile:", os.path.basename(fp))

            src_crs = src.crs
            if src_crs is None:
                src_crs = crs_dom

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
                src_crs=src_crs,
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