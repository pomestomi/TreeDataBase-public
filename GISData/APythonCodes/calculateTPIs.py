from pathlib import Path
"""
Point-based Topographic Position Index (TPI) calculation in QGIS

- Reads a DEM (GeoTIFF or similar) with 1 m resolution
- Reads a point layer with tree locations
- For each point, computes TPI with circular neighbourhoods of radius 5 m, 10 m, 15 m
- Equal weights for all neighbours; centre cell is excluded (weight = 0)
- Writes results into new fields: TPI_5m, TPI_10m, TPI_15m

Based on Zoran Čučković's TPI implementation (landscapearchaeology.org),
adapted to work point-wise instead of full convolution.

Paste into the QGIS Python console and adjust the paths / layer names below.
"""

from qgis.core import QgsVectorLayer, QgsField, QgsProject, edit
from osgeo import gdal
import numpy as np
from qgis.PyQt.QtCore import QVariant

# -------------------------------------------------------------------------
# USER INPUTS
# -------------------------------------------------------------------------
# 1) DEM raster (must be in same CRS as tree points; 1 m resolution)
dem_path = str(Path(__file__).resolve().parents[1] / "CityBamberg" / "DEMBamberg" / "DEMBamberg.tif")

# 2) Tree point layer (existing shapefile; will be edited in place)
tree_path = str(Path(__file__).resolve().parents[1] / "CityBamberg" / "VectorLayers" / "treeLocations.shp")

tree_name_field = "treeName"

# Radii (in metres) and corresponding attribute field names in the shapefile
radii_m = [2.5, 5.0, 7.5]
field_map = {
    2.5: "tpi2m5",
    5.0: "tpi5m",
    7.5: "tpi7m5",
}

# Only debug some trees by name, or set to None for all
debug_tree_names = None  # e.g. ["Tree_001", "Oak45"] or None

# -------------------------------------------------------------------------
# LOAD DEM WITH GDAL
# -------------------------------------------------------------------------
dem_ds = gdal.Open(dem_path)
if dem_ds is None:
    raise RuntimeError(f"Could not open DEM: {dem_path}")

band = dem_ds.GetRasterBand(1)
dem_array = band.ReadAsArray().astype(float)
nodata = band.GetNoDataValue()
gt = dem_ds.GetGeoTransform()

n_rows, n_cols = dem_array.shape
cell_size_x = gt[1]
cell_size_y = abs(gt[5])

if not np.isclose(cell_size_x, cell_size_y):
    raise RuntimeError("Non-square pixels are not supported. Check DEM resolution.")

cell_size = cell_size_x
print(f"DEM loaded: {n_cols} x {n_rows}, cell size={cell_size}, nodata={nodata}")

# -------------------------------------------------------------------------
# BUILD CIRCULAR WINDOWS
# -------------------------------------------------------------------------
def make_circular_mask(radius_m, cell_size):
    """
    Create a circular mask (centre excluded) for given radius in metres.
    Radius in pixels = round(radius_m / cell_size).
    """
    radius_pix = int(round(radius_m / cell_size))
    if radius_pix < 1:
        raise ValueError(f"Radius {radius_m} m too small for cell size {cell_size} m.")

    size = 2 * radius_pix + 1
    cy, cx = radius_pix, radius_pix

    yy, xx = np.indices((size, size))
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    mask = dist < radius_pix + 1e-9   # strictly < radius (in pixel units)
    mask[cy, cx] = False              # exclude centre cell
    return mask, radius_pix

radius_definitions = {}
for r in radii_m:
    mask, r_pix = make_circular_mask(r, cell_size)
    radius_definitions[r] = (mask, r_pix)
    print(f"Radius {r} m -> radius_pix={r_pix}, window size={(2*r_pix+1)}x{(2*r_pix+1)}")

# -------------------------------------------------------------------------
# LOAD EXISTING TREE SHAPEFILE AS QGIS VECTOR LAYER
# -------------------------------------------------------------------------
tree_layer = QgsVectorLayer(tree_path, "treeLocations", "ogr")
if not tree_layer.isValid():
    raise RuntimeError(f"Could not load tree layer: {tree_path}")

# Optionally add to project (so you see it in the Layers panel)
QgsProject.instance().addMapLayer(tree_layer)

fields = tree_layer.fields()
tree_name_idx = fields.indexOf(tree_name_field)
if tree_name_idx < 0:
    raise RuntimeError(f"Field '{tree_name_field}' not found in tree layer.")

# Ensure TPI fields exist (in the SAME shapefile)
provider = tree_layer.dataProvider()
existing_field_names = [f.name() for f in fields]

new_fields = []
for r in radii_m:
    fname = field_map[r]
    if fname not in existing_field_names:
        new_fields.append(QgsField(fname, QVariant.Double))
        print(f"Will create attribute field: {fname}")

if new_fields:
    provider.addAttributes(new_fields)
    tree_layer.updateFields()
    fields = tree_layer.fields()  # refresh

# Get indices for TPI fields
field_indices = {r: fields.indexOf(field_map[r]) for r in radii_m}
print("Field indices:", field_indices)

# -------------------------------------------------------------------------
# WORLD → PIXEL
# -------------------------------------------------------------------------
def world_to_pixel(x, y, gt):
    """
    Convert map coordinates (x, y) to raster indices (col, row) using GeoTransform.
    """
    col = int((x - gt[0]) / gt[1])
    row = int((y - gt[3]) / gt[5])  # gt[5] is usually negative
    return col, row

# -------------------------------------------------------------------------
# MAIN LOOP: DEBUG + WRITE INTO SAME SHAPEFILE
# -------------------------------------------------------------------------
feat_count = tree_layer.featureCount()
print(f"Starting TPI computation for {feat_count} features...\n")

with edit(tree_layer):  # edits are written back to treeLocations.shp
    for feat in tree_layer.getFeatures():
        fid = feat.id()
        name_value = feat[tree_name_field]

        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            print(f"\n=== FID {fid}, treeName={name_value} ===")
            print("  -> NO_GEOMETRY")
            for r in radii_m:
                tree_layer.changeAttributeValue(fid, field_indices[r], None)
            continue

        # Use centroid (works for points and polygons)
        centroid = geom.centroid()
        if centroid is None or centroid.isEmpty():
            print(f"\n=== FID {fid}, treeName={name_value} ===")
            print("  -> NO_CENTROID")
            for r in radii_m:
                tree_layer.changeAttributeValue(fid, field_indices[r], None)
            continue

        x = centroid.asPoint().x()
        y = centroid.asPoint().y()

        col, row = world_to_pixel(x, y, gt)

        # Full debug output for all / or subset
        if (debug_tree_names is None) or (name_value in debug_tree_names):
            print(f"\n=== Feature FID {fid}, treeName={name_value} ===")
            print(f"  Tree coordinates: x={x:.3f}, y={y:.3f}")
            print(f"  Raster index:     row={row}, col={col}")

        # Outside DEM?
        if not (0 <= col < n_cols and 0 <= row < n_rows):
            if (debug_tree_names is None) or (name_value in debug_tree_names):
                print("  -> OUTSIDE_DEM")
            for r in radii_m:
                tree_layer.changeAttributeValue(fid, field_indices[r], None)
            continue

        centre_val = dem_array[row, col]
        if (debug_tree_names is None) or (name_value in debug_tree_names):
            print(f"  DEM centre value: {centre_val}")

        if nodata is not None and centre_val == nodata:
            if (debug_tree_names is None) or (name_value in debug_tree_names):
                print("  -> NODATA_AT_LOCATION")
            for r in radii_m:
                tree_layer.changeAttributeValue(fid, field_indices[r], None)
            continue

        # Per-radius computation (debug + write)
        for r in radii_m:
            mask, r_pix = radius_definitions[r]
            idx = field_indices[r]
            fname = field_map[r]

            if (debug_tree_names is None) or (name_value in debug_tree_names):
                print(f"\n  --- Radius {r} m (field {fname}) ---")
                print(f"    radius_pix = {r_pix}")

            # DEM window bounds
            row_min = max(0, row - r_pix)
            row_max = min(n_rows - 1, row + r_pix)
            col_min = max(0, col - r_pix)
            col_max = min(n_cols - 1, col + r_pix)

            if (debug_tree_names is None) or (name_value in debug_tree_names):
                print(f"    DEM window rows: {row_min}..{row_max}")
                print(f"    DEM window cols: {col_min}..{col_max}")

            sub = dem_array[row_min:row_max+1, col_min:col_max+1]

            # Mask subset indices
            mr_min = r_pix - (row - row_min)
            mr_max = mr_min + sub.shape[0]
            mc_min = r_pix - (col - col_min)
            mc_max = mc_min + sub.shape[1]

            mask_sub = mask[mr_min:mr_max, mc_min:mc_max]

            if (debug_tree_names is None) or (name_value in debug_tree_names):
                print("    DEM window (subarray):")
                print(sub)
                print("    Mask window (1=included, 0=excluded):")
                print(mask_sub.astype(int))

            # Valid neighbours = inside mask and not nodata
            valid = mask_sub
            if nodata is not None:
                valid = valid & (sub != nodata)

            neigh = sub[valid]

            if (debug_tree_names is None) or (name_value in debug_tree_names):
                print("    Neighbour DEM values:")
                print(neigh)

            if neigh.size == 0:
                if (debug_tree_names is None) or (name_value in debug_tree_names):
                    print("    -> No neighbours -> writing NULL")
                tree_layer.changeAttributeValue(fid, idx, None)
            else:
                mean_neigh = float(neigh.mean())
                tpi_val = float(centre_val - mean_neigh)
                if (debug_tree_names is None) or (name_value in debug_tree_names):
                    print(f"    Neighbour mean: {mean_neigh}")
                    print(f"    TPI = centre - mean = {tpi_val}")
                tree_layer.changeAttributeValue(fid, idx, tpi_val)

print("\nFinished: TPI values written to tpi2m5, tpi5m, tpi7m5 in:")
print(f"  {tree_path}")
