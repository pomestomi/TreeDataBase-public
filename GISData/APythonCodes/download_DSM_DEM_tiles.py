# Download Digital Surface Model via Metalink
#--------------------------------------------

import os
import geopandas as gpd
import requests
import xml.etree.ElementTree as ET
import rasterio
from rasterio.warp import reproject, Resampling
import math
import numpy as np
import re
from shapely.geometry import box
import zipfile
from urllib.parse import urlparse, parse_qs
from rasterio.transform import from_origin
from pyproj import Transformer


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------
project_dir = r"D:\TreeDataBase"
tree_loc = "CityErlangen" #-> bavaria
# bavaria: "CityBamberg", "CityErlangen", "CityHassfurt", "CityIngolstadt", "CityNürnberg", "CityStein", "CityWeisendorf"
# bavaria: "CompanyEffeltrich", "CompanyErlangen", "CompanyPappenheim", "UniversityErlangen"
# tree_loc = "CityGarbsen" #-> niedersachsen
# tree_loc = "CityHanover" #-> niedersachsen
# tree_loc = "BotanicalGardenBremen" # bremen
# tree_loc = "CompanyBremen" # bremen
# tree_loc = "CityPirmasens" #-> rheinlandpfalz
# tree_loc = "CityHagen" #-> nrw
# tree_loc = "CityLeipzig" #-> sachsen
# tree_loc = "CompanyPlön" #-> sh
# tree_loc = "CompanyErfurt" #-> thueringen
# tree_loc = "CastleAdminSaxony" #-> sachsen
# tree_loc = "CityBiberach" # -> bw, DSM; process manual tile download
# tree_loc = "CompanySaarbrücken"

model = "DEM"
# model = "DSM"

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

if tree_loc in ["CityBamberg", "CityErlangen", "CityHassfurt", "CityIngolstadt", "CityNürnberg", "CityStein", "CityWeisendorf",
                "CompanyEffeltrich", "CompanyErlangen", "CompanyPappenheim", "UniversityErlangen"]:
    bundesland = "bayern"
    crs_dom = "EPSG:25832"
    method = "meta4"
    if model == "DSM":
        meta4_path = os.path.join(output_dir, "dom20dom.meta4")
    if model == "DEM":
        meta4_path = os.path.join(output_dir, "dgm1.meta4")

if tree_loc in ["CityPirmasens"]:
    bundesland = "rlp"
    crs_dom = "EPSG:25832"
    method = "meta4"
    if model == "DSM":
        meta4_path = os.path.join(output_dir, "dom1_tif_07.meta4")
    if model == "DEM":
        meta4_path = os.path.join(output_dir, "dgm1_tif_07.meta4")

if tree_loc in ["CityGarbsen", "CityHanover"]:
    bundesland = "nds"
    crs_dom = "EPSG:25832"
    method = "geojson_index"
    if model == "DSM":
        tile_index_url = "https://arcgis-geojson.s3.eu-de.cloud-object-storage.appdomain.cloud/dom1/lgln-opengeodata-dom1.geojson"
        row_name = "dom1"
    if model == "DEM":
        tile_index_url = "https://arcgis-geojson.s3.eu-de.cloud-object-storage.appdomain.cloud/dgm1/lgln-opengeodata-dgm1.geojson"
        row_name = "dgm1"

if tree_loc in ["CompanyPlön"]:
    bundesland = "sh"
    crs_dom = "EPSG:25832"
    method = "geojson_index"
    if model == "DSM":
        tile_index_url = "https://geodaten.schleswig-holstein.de/gaialight-sh/_apps/dladownload/single.php?file=bDOM_SH_Massendownload.geojson&id=4"
    if model == "DEM":
        tile_index_url = "https://geodaten.schleswig-holstein.de/gaialight-sh/_apps/dladownload/single.php?file=DGM1_SH__Massendownload.geojson&id=4"

if tree_loc in ["CityHagen"]:
    bundesland = "nrw"
    crs_dom = "EPSG:25832"
    method = "wcs_tiles"
    if model == "DEM":
        wcs_url = "https://www.wcs.nrw.de/geobasis/wcs_nw_dgm"
        coverage_id = "nw_dgm"
    if model == "DSM":
        wcs_url = "https://www.wcs.nrw.de/geobasis/wcs_nw_dom"
        coverage_id = "nw_dom"

if tree_loc in ["CompanySaarbrücken"]:
    bundesland = "saarland"
    crs_dom = "EPSG:25832"
    if model == "DEM":
        method = "wcs_tiles"
        wcs_url = "https://geoportal.saarland.de/gdi-sl/inspireraster/inspirewcsel?"
        coverage_id = "EL.GridCoverage"

    if model == "DSM":
        method = "sl_dsm_zip"

if tree_loc in ["CityLeipzig", "CastleAdminSaxony"]:
    bundesland = "sachsen"
    crs_dom = "EPSG:25833"
    method = "sachsen_tiles"
    url_file = os.path.join(output_dir, f"{g_name}_index.txt")
    with open(url_file, "r") as f:
        tile_urls = [line.strip() for line in f if line.strip()]

if tree_loc in ["CompanyErfurt"]:
    bundesland = "thueringen"
    crs_dom = "EPSG:25832"
    method = "atom_feed_th"
    if model == "DSM":
        atom_url = "https://geoportal.geoportal-th.de/dienste/atom_th_hoehendaten_dom?type=dataset&amp;id=3b5d8d9c-775d-4617-8dfe-71480d6472a6"
    if model == "DEM":
        atom_url = "https://geoportal.geoportal-th.de/dienste/atom_th_hoehendaten_dgm?type=dataset&amp;id=14418d25-fcd7-4a3f-99a9-e3059a2772af"

if tree_loc in ["BotanicalGardenBremen", "CompanyBremen"]:
    bundesland = "bremen"
    crs_dom = "EPSG:25832"
    method = "bremen_xyz_zip"
    if model == "DSM":
        zip_url = "https://gdi2.geo.bremen.de/inspire/download/DOM/data/Gitternetz_DOM1_2017_HB_ASCII_XYZ.zip"
    if model == "DEM":
        zip_url = "https://gdi2.geo.bremen.de/inspire/download/DGM/data/Gitternetz_DGM1_2017_HB_ASCII_XYZ.zip"

if tree_loc in ["CityBiberach"]:
    bundesland = "bw"
    crs_dom = "EPSG:25832"
    method = "bw_DSM_zip"

# define function to select needed tiles from rheinlandpfalz, sachsen
def tile_geom_from_name(name, bundesland):
    if bundesland == "rlp":
        m = re.search(r"(dom1|dgm01)_\d+_(\d+)_(\d+)_", name)
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

    # 100m raster → meter
    xmin = x * 1000
    ymin = y * 1000
    xmax = xmin + tile_size
    ymax = ymin + tile_size

    return box(xmin, ymin, xmax, ymax)

# define function to transform xyz to raster
def xyz_to_tif(xyz_path, tif_path, crs):
    # detect header automatically
    with open(xyz_path, "r") as f:
        first_line = f.readline().strip().lower()

    has_header = any(c.isalpha() for c in first_line)

    data = np.loadtxt(xyz_path, skiprows=1 if has_header else 0)

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    x_unique = np.unique(x)
    y_unique = np.unique(y)

    nx = len(x_unique)
    ny = len(y_unique)

    grid = z.reshape(ny, nx)
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

# --------------------------------------------------
# 2. Prepare tree data
# --------------------------------------------------
# as soon as there are tree location shapefiles, this code block until else (included) can be taken out
# if tree_loc == "CompanySaarbrücken":
    # Create Saarland dummy tree locations
  #  from shapely.geometry import Point

    # Google Maps coordinates (lat, lon)
   # coords = [
    #    (49.230286, 6.987292),
    #    (49.230251, 6.990157),
    #    (49.230250, 6.990167)]
    # Convert to Point geometry
   # geometry = [Point(lon, lat) for lat, lon in coords]
   # trees = gpd.GeoDataFrame(geometry=geometry)
   # trees.set_crs(epsg=4326, inplace=True)

# else:

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
# aoi.is_valid # option to check if geometry is valid, apply next line and recheck if not
# aoi = aoi.buffer(0)

# --------------------------------------------------
# 3. a) meta4
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
# if bavaria, DSM: manually create .meta4 file on this homepage: https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=dom20
# if bavaria, DEM: manually create .meta4 file on this homepage:     https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=dgm1
# if rheinlandpfalz: download .meta4 file for all the Bundesland from: https://geoshop.rlp.de/
# rheinlandpfalz DGM meta4 file: https://geobasis-rlp.de/data/dgm1/current/meta4/dgm1_tif_07.meta4
# rheinlandpfalz DOM meta4 file: https://geobasis-rlp.de/data/dom1/current/meta4/dom1_tif_07.meta4
#-----------------------------------------------------------------------

# load meta4 file
    ns = {"m": "urn:ietf:params:xml:ns:metalink"}
    tree = ET.parse(meta4_path)
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
# 3. b) geojson
# --------------------------------------------------
if method == "geojson_index":
    # load tile index
    tiles_gdf = gpd.read_file(tile_index_url)
    tiles_gdf = tiles_gdf.to_crs(crs_dom)

    # select tiles intersecting aoi
    tiles_sel = tiles_gdf[tiles_gdf.intersects(aoi)]

    if bundesland == "nds":
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
# 3. c) WCS tiles (nrw/saarland DEM)
# --------------------------------------------------
if method == "wcs_tiles":

    if bundesland == "nrw":
        tile_size = 2000  # meters (nrw wcs pixel limit)

    if bundesland == "saarland":
        tile_size = 1000  # meters

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
# 3. d) Sachsen tiles
# --------------------------------------------------
# manual step (only once): download tile urls for all the state of Sachsen from
# https://www.geodaten.sachsen.de/batch-download-4719.html

if method == "sachsen_tiles":

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
# 3. e) Thüringen ATOM feed
# --------------------------------------------------
if method == "atom_feed_th":

    ns = {"atom": "http://www.w3.org/2005/Atom"}
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
# 3. f) Bremen single Zip xyz
# --------------------------------------------------
if method == "bremen_xyz_zip":

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

                print("Converting XYZ → TIF:", name)

                xyz_to_tif(out_xyz, tif_path, crs_dom)

            tiles.append((name, tif_path))

# --------------------------------------------------
# 3. g) BW DSM Zip tifs
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
# 3. h) Saarland DSM Zip tifs
# --------------------------------------------------
if method == "sl_dsm_zip":

    tiles = []

    if tree_loc == "CityNeunkirchen":
        saarland_lk = ["NK"]
    if tree_loc == "CompanyHomburg":
        saarland_lk = ["SPK"]
    if tree_loc == "CompanySaarbrücken":
        saarland_lk = ["SB"]
    else:
        saarland_lk = ["MZG", "NK", "SB", "SLS", "SPK", "WND"]

    saarland_zip_template = "https://www.shop.lvgl.saarland.de/cloud/public.php/dav/files/NK8ndP55qAqGEZD/OD_DOM1_2025_tif_LK/DOM1_tif_{lk}_EPSG-25832_Entstehung-2025.zip"

    for lk in saarland_lk:

        zip_url_lk = saarland_zip_template.format(lk=lk)
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
#---------------------------------------------------------
# 4. Download tiles
print(f"Tiles found: {len(tiles)}")

if method in ["bremen_xyz_zip", "bw_DSM_zip", "sl_dsm_zip"]:
    print(f"{tree_loc} tiles already extracted")
else:
    for name, source in tiles:
        tile_output_path = os.path.join(tmp_dir, name)

        if os.path.exists(tile_output_path):
            continue

        print("Downloading:", name)

        if method == "wcs_tiles":
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
                print("Converting XYZ → TIF:", name)
                xyz_to_tif(tile_output_path, tif_path, crs_dom)
                # optional cleanup (recommended)
               # os.remove(tile_output_path)

    print("Download complete")

#---------------------------------------------------------
# 5. Output grid setup
target_res = 1.0 # meters
nodata_val = -9999

# Snap to resolution
minx = math.floor(minx / target_res) * target_res
miny = math.floor(miny / target_res) * target_res
maxx = math.ceil(maxx / target_res) * target_res
maxy = math.ceil(maxy / target_res) * target_res

# Define output grid width, height
width = int(round((maxx - minx) / target_res))
height = int(round((maxy - miny) / target_res))

# initialize output raster
dst_array = np.full((height, width), nodata_val, dtype="float32")

# define CRS
dst_crs = crs_dom

transform = rasterio.transform.from_origin(
    minx, maxy, target_res, target_res
)

#---------------------------------------------------------
# 6. Mosaic tiles
tile_files = [
    os.path.join(tmp_dir, f)
    for f in os.listdir(tmp_dir)
    if f.lower().endswith(".tif")
]

for fp in tile_files:
    print("Processing:", os.path.basename(fp))

    with rasterio.open(fp) as src:

        # fast overlap check (replaces mask try/except)
        if not (
            src.bounds.left < maxx and
            src.bounds.right > minx and
            src.bounds.bottom < maxy and
            src.bounds.top > miny
        ):
            print("Skipping (no overlap):", os.path.basename(fp))
            continue

        src_crs = src.crs
        if src_crs is None:
            src_crs = crs_dom
        data = src.read(1).astype("float32")

        if bundesland == "saarland" and model == "DEM":
            data = data/100.0

        if model == "DSM":
            data[data <= 0] = np.nan  # DSM-safe rule

        temp = np.full((height, width), nodata_val, dtype="float32")

        reproject(
            source=data,
            destination=temp,
            src_transform=src.transform,
            src_crs=src_crs,
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

print(f"Done: {model} created at 1m resolution")

with rasterio.open(output_path) as src:
    print(f"\n--- {model} METADATA CHECK {tree_loc}---")
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


import requests
import rasterio

wcs_url = "https://geoportal.saarland.de/gdi-sl/inspireraster/inspirewcsel?"

# Small 100 x 100 m test area
x0 = 352100
y0 = 5454400
x1 = x0 + 100
y1 = y0 + 100

params = [
    ("SERVICE", "WCS"),
    ("VERSION", "2.0.1"),
    ("REQUEST", "GetCoverage"),
    ("COVERAGEID", "EL.GridCoverage"),
    ("FORMAT", "image/tiff"),
    ("SUBSET", f"x({x0},{x1})"),
    ("SUBSET", f"y({y0},{y1})"),

    # NEW:
    ("SCALESIZE", "i(100)"),
    ("SCALESIZE", "j(100)")
]

r = requests.get(wcs_url, params=params)

print("HTTP:", r.status_code)
print(r.headers.get("Content-Type"))
print(r.text)