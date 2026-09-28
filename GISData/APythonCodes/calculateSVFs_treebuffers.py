# Calculate Sky View Factor
#----------------------------

import os
import geopandas as gpd
import rasterio
from rvt.vis import sky_view_factor
import math
import matplotlib.pyplot as plt
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--city", required=True)
args = parser.parse_args()
tree_loc = args.city

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
city_dir = os.path.join(project_dir, "GISData", tree_loc)
tree_path = os.path.join(city_dir, "VectorLayers", "treeLocations.shp")
dsm_dir = os.path.join(city_dir, "DSM")
dem_dir = os.path.join(city_dir, "DEM")
svf_dir = os.path.join(city_dir, "SVF")
output_tree_path = os.path.join(svf_dir, "trees_svf.shp")

os.makedirs(svf_dir, exist_ok=True)

# --------------------------------------------------
# 2. COMPUTE SVF PER TREE
# --------------------------------------------------
# derive CRS automatically from DEM
sample_dem = os.path.join(dem_dir, "DEM_tree_000.tif")
with rasterio.open(sample_dem) as src:
    crs_dom = src.crs

trees = gpd.read_file(tree_path).to_crs(crs_dom)

buffer_size = 100
resolution = 1.0

print(f"Processing {len(trees)} trees...")

for idx, row in trees.iterrows():

    print(f"... tree {idx}")

    # file paths for THIS tree
    dem_tree_path = os.path.join(dem_dir, f"DEM_tree_{idx:03d}.tif")
    dsm_tree_path = os.path.join(dsm_dir, f"DSM_tree_{idx:03d}.tif")
    svf_raw_tree_path = os.path.join(svf_dir, f"SVF_raw_tree_{idx:03d}.tif")
    svf_mod_tree_path = os.path.join(svf_dir, f"SVF_mod_tree_{idx:03d}.tif")

    # skip tree if DEM or DSM is missing
    if not (os.path.exists(dem_tree_path) and os.path.exists(dsm_tree_path)):
        print(f"Skipping tree {idx}: missing DEM or DSM.")
        continue

    # read DSM
    with rasterio.open(dsm_tree_path) as src:
        dsm = src.read(1).astype("float32")
        profile = src.profile.copy()

    # read DEM
    with rasterio.open(dem_tree_path) as src:
        dem = src.read(1).astype("float32")

    # tree coordinates
    if row.geometry.geom_type == "Point":
        point = row.geometry
    else:
        point = row.geometry.geoms[0]

    x = point.x
    y = point.y
    row_pix, col_pix = rasterio.transform.rowcol(
        profile["transform"], x, y)

    # build DSM_mod
    dsm_mod = dsm.copy()
    tree_height = row["height"]
    if pd.notna(tree_height):
        new_height = dem[row_pix, col_pix] + tree_height
        if new_height > dsm[row_pix, col_pix] + 0.01:
            dsm_mod[row_pix, col_pix] = new_height

    # compute SVF from original DSM
    svf_raw = sky_view_factor(
        dem=dsm,
        resolution=resolution,
        compute_svf=True,
        compute_asvf=False,
        compute_opns=False,
        svf_r_max=buffer_size,
        svf_n_dir=16
    )["svf"]

    # compute SVF from modified DSM
    svf_mod = sky_view_factor(
        dem=dsm_mod,
        resolution=resolution,
        compute_svf=True,
        compute_asvf=False,
        compute_opns=False,
        svf_r_max=buffer_size,
        svf_n_dir=16
    )["svf"]

    # save SVF rasters
    profile.update(dtype="float32", count=1)

    with rasterio.open(svf_raw_tree_path, "w", **profile) as dst:
        dst.write(svf_raw.astype("float32"), 1)

    with rasterio.open(svf_mod_tree_path, "w", **profile) as dst:
        dst.write(svf_mod.astype("float32"), 1)

    # sample SVF
    with rasterio.open(svf_raw_tree_path) as src:
        svf_raw_val = list(src.sample([(x, y)]))[0][0]
    trees.loc[idx, "SVF_raw"] = svf_raw_val

    # sample SVF_mod
    with rasterio.open(svf_mod_tree_path) as src:
        svf_mod_val = list(src.sample([(x, y)]))[0][0]
    trees.loc[idx, "SVF_mod"] = svf_mod_val

    # sample DSM
    with rasterio.open(dsm_tree_path) as src:
        dsm_val = list(src.sample([(x, y)]))[0][0]
    trees.loc[idx, "DSM"] = dsm_val

    # sample DEM
    with rasterio.open(dem_tree_path) as src:
        dem_val = list(src.sample([(x, y)]))[0][0]
    trees.loc[idx, "DEM"] = dem_val

print("SVF computation complete.")

# --------------------------------------------------
# save result
# derived height
trees["height_dev"] = trees["DSM"]-trees["DEM"]
trees.to_file(output_tree_path)

# add SVF_mod and elevation to original tree shapefile
trees_orig = gpd.read_file(tree_path)
trees_orig["SVF"] = trees["SVF_mod"]
trees_orig["elevation"] = trees["DEM"]
trees_orig.to_file(tree_path)

# plot measured vs calculated tree height
df = trees[["height", "height_dev"]].dropna()
if len(df) > 0:
    max_val = max(df["height"].max(), df["height_dev"].max())+0.3
    max_val = math.ceil(max_val)
    min_val = -1

    plt.figure(figsize=(6,6))
    plt.scatter(df["height"], df["height_dev"], s=10, alpha=0.8, color = "green",  zorder=2)

    # same axis range for x and y
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    # 1:1 reference line
    plt.plot([min_val, max_val], [min_val, max_val], alpha = 0.5, color="gray",  zorder=1)

    plt.xlabel("Measured tree height [m]")
    plt.ylabel("DSM-DEM [m]")
    plt.title(tree_loc)
    plt.grid(True)

    plt.savefig(os.path.join(svf_dir, "fig_height_scatter.pdf"), bbox_inches="tight")
    plt.show()
else:
    print("No measured tree height available")

#---------------------
# plot SVF vs. SVFmod
#---------------------
df = trees[["height", "SVF_raw", "SVF_mod"]]
max_val = 1.03
min_val = -0.03

plt.figure(figsize=(6, 6))
# 1:1 reference line
plt.plot([min_val, max_val], [min_val, max_val], alpha=0.5, color="gray",  zorder=1)

plt.scatter(df["SVF_raw"], df["SVF_mod"], s=10, alpha=0.8, color = "blue", label = "Height available",  zorder=2)
mask_na = df["height"].isna()
plt.scatter(df.loc[mask_na, "SVF_raw"], df.loc[mask_na, "SVF_mod"], s=10, alpha=0.8, color="orange", label="Height = NA",  zorder=3)

# same axis range for x and y
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

plt.xlabel("SVF raw")
plt.ylabel("SVF modified")
plt.title(tree_loc)
if mask_na.any():
    plt.legend(loc = "lower right")
plt.grid(True)

plt.savefig(os.path.join(svf_dir, "fig_SVF_scatter.pdf"), bbox_inches="tight")
plt.show()
