# Extract Depth to Groundwater
#-----------------------------
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------
# define paths
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tree_loc = "CityPotsdam"
# tree_loc = "CityBerlinFrhXBerg"

city_dir = os.path.join(project_dir, "GISData", tree_loc)
tree_path = os.path.join(city_dir, "VectorLayers", "treeLocations.shp")
tree_svf_path = os.path.join(city_dir, "SVF", "trees_svf.shp")
output_dir = os.path.join(city_dir, "GW")
output_tree_path = os.path.join(output_dir, "trees_gw.shp")

if tree_loc in ["CityPotsdam", "CityBerlinFrhXBerg"]:
    gw_path_2011 = os.path.join(output_dir, "grundwasserflurabstand_20130620/GW_Flurabstand.shp")
    gw_path_iso = os.path.join(output_dir, "grundwasserisohypsen_2011", "shp", "GWD_2011F.shp")
    crs_dom = "EPSG:25833"  # Brandenburg/Berlin

# load data and reproject
trees = gpd.read_file(tree_svf_path)
trees = trees.to_crs(crs_dom)

# GWFA 2011 polygon (published 2013)
gw = gpd.read_file(gw_path_2011)
gw = gw.to_crs(crs_dom)
trees = gpd.sjoin(trees, gw[["FA", "KLASSE", "geometry"]], how="left", predicate="within")
trees = trees.drop(columns=["index_right"], errors="ignore")

# GWFA from isohypsen 2011F
gw = gpd.read_file(gw_path_iso)
gw = gw.to_crs(crs_dom)
trees = gpd.sjoin_nearest(trees, gw[["ZLEVEL", "geometry"]], how="left", distance_col='distance_to_line')
trees = trees.drop(columns=["index_right"], errors="ignore")
trees["dtgw_iso"] = trees.DEM - trees.ZLEVEL
bins = [-np.inf, 1, 2, 3, 4, 5, 7.5, 10, 15, 20, 30]
labels_fa = ['<= 1', '> 1 - 2', '> 2 - 3', '> 3 - 4', '> 4 - 5', '> 5 - 7,5', '> 7,5 - 10', '> 10 - 15', '> 15 - 20', '> 20 - 30']
labels_kl = range(1, 11)
trees["FA_iso"] = pd.cut(trees['dtgw_iso'], bins=bins, labels=labels_fa, include_lowest=True)
trees["KLASSE_iso"] = pd.cut(trees['dtgw_iso'], bins=bins, labels=labels_kl, include_lowest=True)

# Plot DTGW from isolines
plt.boxplot(trees["dtgw_iso"].dropna())
plt.ylabel("DTGW from isolines [m]")
plt.savefig(os.path.join(output_dir, "DTGW_iso_2011_boxplot.pdf"), bbox_inches="tight")
plt.show()

# Plot DTGW 2013
# Get order based on numeric column
order = trees.sort_values("KLASSE")["FA"].unique()
# Count and align to that order
counts = trees["FA"].value_counts().reindex(order)

plt.bar(counts.index, counts.values)
plt.xlabel("Depth to groundwater class")
plt.ylabel("Number of trees")
plt.title(tree_loc)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)

plt.savefig(os.path.join(output_dir, "DTGW_2011_hist.pdf"), bbox_inches="tight")
plt.show()

# Scatterplot dtgw 2011 vs dtgw from isolines 2011
df = trees[["KLASSE", "FA", "KLASSE_iso", "FA_iso"]].dropna()

max_val = 11
min_val = 0

plt.figure(figsize=(6,6))
plt.scatter(df["KLASSE"], df["KLASSE_iso"], s=10, alpha=0.3, color = "blue",  zorder=2)

# same axis range for x and y
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

# 1:1 reference line
plt.plot([min_val, max_val], [min_val, max_val], alpha = 0.5, color="gray",  zorder=1)

plt.xlabel("DTGW class 2011")
plt.ylabel("DTGW class from isolines 2011")
plt.title(tree_loc)
plt.grid(True)

plt.show()
