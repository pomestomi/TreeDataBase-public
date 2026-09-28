from pathlib import Path
from qgis.core import QgsVectorLayer, QgsProject, edit
from osgeo import gdal
import numpy as np
import math

# -------------------------------------------------------------------------
# USER INPUTS
# -------------------------------------------------------------------------
# 1) DEM raster
dem_path = str(Path(__file__).resolve().parents[1] / "UniversityErlangen" / "DEMErlangen" / "DEMErlangen.tif")

# 2) Tree point layer
tree_path = str(Path(__file__).resolve().parents[1] / "CityErlangen" / "VectorLayers" / "treeLocations.shp")

# Attribute field names in treeLocations.shp that should receive the values
twi_field = "twi"           # field to store TWI
flowacc_field = "flowAcc"   # field to store flow accumulation (unitless count)
slope_field = "slope"       # field to store slope in degrees

# Name of the attribute field identifying trees (for logging only)
tree_name_field = "treeName"

# Local DEM window radius around each tree for the TWI calculation [m]
window_radius_m = 200.0   # same as you used before; can be changed

# -------------------------------------------------------------------------
# HELPER: world → pixel
# -------------------------------------------------------------------------
def world_to_pixel(x, y, gt):
    """
    Convert map coordinates (x, y) to raster indices (col, row) using GeoTransform.
    """
    col = int((x - gt[0]) / gt[1])
    row = int((y - gt[3]) / gt[5])  # gt[5] is usually negative
    return col, row

# -------------------------------------------------------------------------
# HELPER: local D8 flow accumulation + slope + TWI
# -------------------------------------------------------------------------
def compute_d8_flow_accum_twi(dem, nodata=None, cell_size=1.0,
                              tree_row=None, tree_col=None):
    """
    dem: 2D numpy array (local DEM window)
    nodata: nodata value or None
    cell_size: raster cell size (m)
    tree_row, tree_col: indices of the tree cell inside this window

    Returns: (twi, flow_acc_at_tree, slope_rad_at_tree)
             or (None, None, None) if tree cell is nodata.
    """
    H, W = dem.shape

    # nodata mask
    if nodata is None:
        nodata_mask = np.zeros_like(dem, dtype=bool)
    else:
        nodata_mask = (dem == nodata)

    # Flow direction: -1 = pit/flat or nodata, 0..7 = N, NE, E, SE, S, SW, W, NW
    flow_dir = np.full((H, W), -1, dtype=np.int8)
    dr = np.array([-1,-1, 0, 1, 1, 1, 0,-1])
    dc = np.array([ 0, 1, 1, 1, 0,-1,-1,-1])
    dist = np.array([1, math.sqrt(2), 1, math.sqrt(2),
                     1, math.sqrt(2), 1, math.sqrt(2)]) * cell_size

    # 1) D8 flow directions (steepest descent)
    for r in range(H):
        for c in range(W):
            if nodata_mask[r, c]:
                continue
            z = dem[r, c]
            max_drop = 0.0
            max_dir = -1
            for d in range(8):
                rr = r + dr[d]
                cc = c + dc[d]
                if rr < 0 or rr >= H or cc < 0 or cc >= W:
                    continue
                if nodata_mask[rr, cc]:
                    continue
                drop = (z - dem[rr, cc]) / dist[d]
                if drop > max_drop:
                    max_drop = drop
                    max_dir = d
            flow_dir[r, c] = max_dir

    # 2) Flow accumulation
    flow_acc = np.ones((H, W), dtype=float)  # each cell contributes 1 unit
    # Process cells from high to low elevation to push flow downslope
    order = np.argsort(dem, axis=None)[::-1]
    for idx in order:
        r = idx // W
        c = idx % W
        if nodata_mask[r, c]:
            continue
        d = flow_dir[r, c]
        if d == -1:
            continue  # no downslope neighbor
        rr = r + dr[d]
        cc = c + dc[d]
        if 0 <= rr < H and 0 <= cc < W and not nodata_mask[rr, cc]:
            flow_acc[rr, cc] += flow_acc[r, c]

    # 3) Slope at tree cell (central difference approximation)
    if tree_row is None or tree_col is None:
        raise ValueError("tree_row, tree_col must be specified")
    r = tree_row
    c = tree_col
    if nodata_mask[r, c]:
        return None, None, None

    def safe_z(rr, cc):
        # If neighbor is out of bounds or nodata, use the tree cell elevation
        if rr < 0 or rr >= H or cc < 0 or cc >= W or nodata_mask[rr, cc]:
            return dem[r, c]
        return dem[rr, cc]

    dzdx = (safe_z(r, c+1) - safe_z(r, c-1)) / (2 * cell_size)
    dzdy = (safe_z(r-1, c) - safe_z(r+1, c)) / (2 * cell_size)
    slope_rad = math.atan(math.sqrt(dzdx**2 + dzdy**2))

    # 4) TWI = ln( A / tan(beta) )
    eps = 1e-6
    a = flow_acc[r, c] * (cell_size * cell_size)  # upslope area (m²)
    twi = math.log((a + eps) / math.tan(slope_rad + eps))

    return twi, flow_acc[r, c], slope_rad

# -------------------------------------------------------------------------
# LOAD DEM
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
    raise RuntimeError("DEM pixels are not square. This script assumes square cells.")

cell_size = cell_size_x
print(f"DEM loaded: {n_cols} x {n_rows}, cell_size = {cell_size} m, nodata = {nodata}")

window_radius_pix = int(round(window_radius_m / cell_size))
print(f"Using local DEM window radius: {window_radius_m} m (~{window_radius_pix} pixels)")

# -------------------------------------------------------------------------
# LOAD TREE LAYER AS QGIS VECTOR LAYER
# -------------------------------------------------------------------------
tree_layer = QgsVectorLayer(tree_path, "treeLocations", "ogr")
if not tree_layer.isValid():
    raise RuntimeError(f"Could not load tree layer: {tree_path}")

QgsProject.instance().addMapLayer(tree_layer)

fields = tree_layer.fields()
idx_twi = fields.indexOf(twi_field)
idx_flowacc = fields.indexOf(flowacc_field)
idx_slope = fields.indexOf(slope_field)

if idx_twi < 0 or idx_flowacc < 0 or idx_slope < 0:
    raise RuntimeError(
        f"One or more attribute fields not found. "
        f"Expected: '{twi_field}', '{flowacc_field}', '{slope_field}'."
    )

idx_name = fields.indexOf(tree_name_field) if fields.indexOf(tree_name_field) >= 0 else None

feat_count = tree_layer.featureCount()
print(f"Starting TWI computation for {feat_count} trees...\n")

# -------------------------------------------------------------------------
# MAIN LOOP: compute local TWI per tree and write attributes
# -------------------------------------------------------------------------
with edit(tree_layer):
    for feat in tree_layer.getFeatures():
        fid = feat.id()
        if idx_name is not None:
            tree_name = feat[tree_name_field]
        else:
            tree_name = f"FID_{fid}"

        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            print(f"{tree_name}: NO_GEOMETRY -> writing NULLs")
            tree_layer.changeAttributeValue(fid, idx_twi, None)
            tree_layer.changeAttributeValue(fid, idx_flowacc, None)
            tree_layer.changeAttributeValue(fid, idx_slope, None)
            continue

        # Use centroid (works for points and polygons)
        centroid = geom.centroid()
        if centroid is None or centroid.isEmpty():
            print(f"{tree_name}: NO_CENTROID -> writing NULLs")
            tree_layer.changeAttributeValue(fid, idx_twi, None)
            tree_layer.changeAttributeValue(fid, idx_flowacc, None)
            tree_layer.changeAttributeValue(fid, idx_slope, None)
            continue

        x = centroid.asPoint().x()
        y = centroid.asPoint().y()

        col, row = world_to_pixel(x, y, gt)

        # Outside DEM?
        if not (0 <= col < n_cols and 0 <= row < n_rows):
            print(f"{tree_name}: OUTSIDE_DEM -> writing NULLs")
            tree_layer.changeAttributeValue(fid, idx_twi, None)
            tree_layer.changeAttributeValue(fid, idx_flowacc, None)
            tree_layer.changeAttributeValue(fid, idx_slope, None)
            continue

        centre_val = dem_array[row, col]
        if nodata is not None and centre_val == nodata:
            print(f"{tree_name}: NODATA_AT_LOCATION -> writing NULLs")
            tree_layer.changeAttributeValue(fid, idx_twi, None)
            tree_layer.changeAttributeValue(fid, idx_flowacc, None)
            tree_layer.changeAttributeValue(fid, idx_slope, None)
            continue

        # Determine local window around tree
        row_min = max(0, row - window_radius_pix)
        row_max = min(n_rows - 1, row + window_radius_pix)
        col_min = max(0, col - window_radius_pix)
        col_max = min(n_cols - 1, col + window_radius_pix)

        sub_dem = dem_array[row_min:row_max+1, col_min:col_max+1]
        local_row = row - row_min
        local_col = col - col_min

        # Compute local D8 / TWI
        twi_val, flowacc_val, slope_rad = compute_d8_flow_accum_twi(
            sub_dem,
            nodata=nodata,
            cell_size=cell_size,
            tree_row=local_row,
            tree_col=local_col
        )

        if twi_val is None:
            print(f"{tree_name}: computation failed (nodata) -> writing NULLs")
            tree_layer.changeAttributeValue(fid, idx_twi, None)
            tree_layer.changeAttributeValue(fid, idx_flowacc, None)
            tree_layer.changeAttributeValue(fid, idx_slope, None)
        else:
            slope_deg = math.degrees(slope_rad)
            tree_layer.changeAttributeValue(fid, idx_twi, float(twi_val))
            tree_layer.changeAttributeValue(fid, idx_flowacc, float(flowacc_val))
            tree_layer.changeAttributeValue(fid, idx_slope, float(slope_deg))
            print(f"{tree_name}: TWI={twi_val:.4f}, FlowAcc={flowacc_val:.1f}, Slope={slope_deg:.2f}°")

print("\nFinished computing TWI / FlowAcc / Slope and writing to attribute table.")