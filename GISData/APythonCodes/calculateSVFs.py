# Calculate Sky View Factor
#----------------------------

import os
import geopandas as gpd
import rasterio
import numpy as np
from rvt.vis import sky_view_factor
import math
from rasterio.windows import Window
import matplotlib.pyplot as plt
from rasterio.features import rasterize

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# tree_loc = "CityBerlinFrhXBerg"
# tree_loc = "CityPotsdam"
# tree_loc = "CityBamberg"
# tree_loc = "CityGarbsen"
tree_loc = "UniversitySalzburg"
compute_svf_mod = True

print(f"processing: {tree_loc}, compute_svf_mod = {compute_svf_mod}")

city_dir = os.path.join(project_dir, "GISData", tree_loc)
tree_path = os.path.join(city_dir, "VectorLayers", "treeLocations.shp")
dsm_dir = os.path.join(city_dir, "DSM")
dem_dir = os.path.join(city_dir, "DEM")
dsm_path = os.path.join(dsm_dir, f"DSM_{tree_loc}.tif")
dem_path = os.path.join(dem_dir, f"DEM_{tree_loc}.tif")
dsm_mod_path = os.path.join(dsm_dir, f"DSM_mod_{tree_loc}.tif")

output_tree_path = os.path.join(dsm_dir, "trees_svf.shp")

if compute_svf_mod == True:
    svf_path = os.path.join(dsm_dir, f"SVFmod_{tree_loc}.tif")
else:
    svf_path = os.path.join(dsm_dir, f"SVF_{tree_loc}.tif")

if tree_loc in ["CityBamberg", "CityErlangen"]:
    bundesland = "bavaria"
    crs_dom = "EPSG:25832"
if tree_loc in ["CityGarbsen", "CityHanover"]:
    bundesland = "niedersachsen"
    crs_dom = "EPSG:25832"
if tree_loc in ["CityBerlinFrhXBerg", "CityPotsdam"]:
    crs_dom = "EPSG:25833"
if tree_loc in ["CityHagen"]:
    bundesland = "nrw"
    crs_dom = "EPSG:25832"

# --------------------------------------------------
# 2. Prepare dsm_mod for svf_mod = True
# --------------------------------------------------
if compute_svf_mod:
    print("Building DSM_mod with conditional tree replacement...")

    trees = gpd.read_file(tree_path).to_crs(crs_dom)

    with rasterio.open(dsm_path) as dsm_src:
        dsm = dsm_src.read(1).astype("float32")
        transform = dsm_src.transform
        out_shape = dsm.shape
        profile_dsm = dsm_src.profile.copy()

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1).astype("float32")

    # rasterize tree heights (0 where no tree)
    tree_height_raster = rasterize(
        [(geom, h) for geom, h in zip(trees.geometry, trees["height"])],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="float32"
    )

    # rasterize tree presence mask (1 where tree exists)
    tree_mask = rasterize(
        [(geom, 1) for geom in trees.geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    # compute tree-based surface
    tree_surface = dem + tree_height_raster

    # CONDITIONAL replacement
    dsm_mod = np.where(
        (tree_mask == 1) & (tree_surface > dsm +0.01),
        tree_surface, dsm)

    profile_dsm.update(dtype="float32", count=1)
    with rasterio.open(dsm_mod_path, "w", **profile_dsm) as dst:
        dst.write(dsm_mod.astype("float32"), 1)
    print("DSM_mod created successfully.")

# define raster source
if compute_svf_mod:
    svf_source_raster = dsm_mod
else:
    svf_source_raster = None  # will be read from file later
# --------------------------------------------------
# 3. COMPUTE SVF in tiles
# --------------------------------------------------
tile_size = 1024          # core tile size (pixels)
buffer_size = 100         # MUST match svf_r_max
resolution = 1.0

# --------------------------------------------------
# OPEN INPUT + PREP OUTPUT
# --------------------------------------------------
with rasterio.open(dsm_path) as src_base:
    profile_base = src_base.profile.copy()
    transform = src_base.transform
    width = src_base.width
    height = src_base.height

    profile_base.update(dtype="float32", count=1)

    with rasterio.open(svf_path, "w", **profile_base) as dst:

        n_tiles_x = math.ceil(width / tile_size)
        n_tiles_y = math.ceil(height / tile_size)

        print(f"Processing {n_tiles_x * n_tiles_y} tiles...")

        for i in range(n_tiles_x):
            for j in range(n_tiles_y):

                # -----------------------------
                # 1. Define core tile
                # -----------------------------
                col_off = i * tile_size
                row_off = j * tile_size

                core_width = min(tile_size, width - col_off)
                core_height = min(tile_size, height - row_off)

                # -----------------------------
                # 2. Expand with buffer
                # -----------------------------
                col_off_buf = max(0, col_off - buffer_size)
                row_off_buf = max(0, row_off - buffer_size)

                col_end_buf = min(width, col_off + core_width + buffer_size)
                row_end_buf = min(height, row_off + core_height + buffer_size)

                win_width = col_end_buf - col_off_buf
                win_height = row_end_buf - row_off_buf

                window = Window(col_off_buf, row_off_buf, win_width, win_height)

                # -----------------------------
                # 3. Read DSM tile
                # -----------------------------
                if compute_svf_mod:
                    dsm_tile = svf_source_raster[
                        window.row_off:window.row_off + win_height,
                        window.col_off:window.col_off + win_width
                    ]
                else:
                    dsm_tile = src_base.read(1, window=window).astype("float32")

                # -----------------------------
                # 4. Compute SVF on buffered tile
                # -----------------------------
                svf_tile = sky_view_factor(
                    dem=dsm_tile,
                    resolution=resolution,
                    compute_svf=True,
                    compute_asvf=False,
                    compute_opns=False,
                    svf_r_max=buffer_size,
                    svf_n_dir=16
                )["svf"]

                # -----------------------------
                # 5. Crop buffer away
                # -----------------------------
                crop_col_start = col_off - col_off_buf
                crop_row_start = row_off - row_off_buf

                svf_core = svf_tile[
                    crop_row_start:crop_row_start + core_height,
                    crop_col_start:crop_col_start + core_width
                ]

                # -----------------------------
                # 6. Write to output raster
                # -----------------------------
                out_window = Window(col_off, row_off, core_width, core_height)

                dst.write(svf_core.astype("float32"), 1, window=out_window)

                print(f"Tile ({i},{j}) done")

print("SVF tiling complete.")
print(f"CRS SVF: {profile_base['crs']}, CRS {tree_loc}: {crs_dom}")

# --------------------------------------------------
# 5. extract tree SVF
# --------------------------------------------------
if compute_svf_mod == True:
    trees = gpd.read_file(output_tree_path).to_crs(crs_dom)

trees = trees.explode(index_parts=False)
coords = [(geom.x, geom.y) for geom in trees.geometry]

with rasterio.open(svf_path) as src:
    sampled = list(src.sample(coords))
if compute_svf_mod == True:
    trees["SVF_mod"] = [val[0] if val[0] is not None else np.nan for val in sampled]
    trees["SVF_diff"] = trees.SVF_mod - trees.SVF
    with rasterio.open(dsm_mod_path) as src:
        trees["DSM_mod"] = [v[0] for v in src.sample(coords)]
    # save result
    trees.to_file(output_tree_path)
else:
    trees["SVF"] = [val[0] if val[0] is not None else np.nan for val in sampled]

# --------------------------------------------------
# 6. Check precision: Calc tree height from DSM-DEM
# --------------------------------------------------

# sample DSM
    with rasterio.open(dsm_path) as src:
        sampled = list(src.sample(coords))

    trees["DSM"] = [val[0] if val[0] is not None else np.nan for val in sampled]

    # sample DEM
    with rasterio.open(dem_path) as src:
        sampled = list(src.sample(coords))

    trees["DEM"] = [val[0] if val[0] is not None else np.nan for val in sampled]

    trees["height_mod"] = trees["DSM"]-trees["DEM"]

    # save result
    trees.to_file(output_tree_path)

    # plot measured vs calculated tree height
    df = trees[["height", "height_mod"]].dropna()
    max_val = max(df["height"].max(), df["height_mod"].max())+0.3
    max_val = math.ceil(max_val)
    min_val = -1

    plt.figure(figsize=(6,6))
    plt.scatter(df["height"], df["height_mod"], s=10, alpha=0.5)

    # same axis range for x and y
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    # 1:1 reference line
    plt.plot([min_val, max_val], [min_val, max_val], alpha = 0.5, color = "red")

    plt.xlabel("Measured tree height [m]")
    plt.ylabel("DSM-DEM [m]")
    plt.title(tree_loc)
    plt.grid(True)

    plt.savefig(os.path.join(dsm_dir, "height_scatter.pdf"), bbox_inches="tight")
    plt.show()

    #---------------------
    # plot SVF vs. SVFmod
    #---------------------
if compute_svf_mod == True:
#   df = trees[["DSM", "DEM", "height", "height_mod", "SVF", "SVF_mod"]].dropna()
    df = trees[["SVF", "SVF_mod"]].dropna()
    max_val = 1
    min_val = 0

    plt.figure(figsize=(6, 6))
    plt.scatter(df["SVF"], df["SVF_mod"], s=10, alpha=0.5, color = "orange")

    # same axis range for x and y
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    # 1:1 reference line
    plt.plot([min_val, max_val], [min_val, max_val], alpha=0.5, color="red")

    plt.xlabel("SVF")
    plt.ylabel("SVF modified")
    plt.title(tree_loc)
    plt.grid(True)

    plt.savefig(os.path.join(dsm_dir, "SVF_scatter.pdf"), bbox_inches="tight")
    plt.show()
