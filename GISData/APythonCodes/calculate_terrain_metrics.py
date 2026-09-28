#!/usr/bin/env python3
"""
calculate_terrain_metrics.py
============================
Standalone script (no QGIS required) that computes TWI, flow accumulation,
slope, and TPI at three radii for every tree in every city under GISData/,
then writes the results back into each city's VectorLayers/treeLocations.shp.

DEM discovery (checked in order for each city):
  1. {city}/DEM/DEM_tree_000.tif          — mosaicked output from run_dem_dsm.py
  2. {city}/DEM/*.tif                     — any single tif in DEM/
  3. {city}/DEM*/*.tif                    — tile collections (DEMBamberg/, DEMBerlin/, …)

Multiple tiles are mosaicked on-the-fly via a GDAL VRT (no extra disk writes).
Tree coordinates are reprojected to the DEM CRS automatically.

Usage:
    python GISData/APythonCodes/calculate_terrain_metrics.py [--city CityErlangen ...]
    python GISData/APythonCodes/calculate_terrain_metrics.py --skip-filled

Options:
    --city NAME [NAME ...]   process only these cities (default: all)
    --skip-filled            skip trees that already have a non-null twi value
    --window-m FLOAT         local TWI window radius in metres (default: 200)
"""

import argparse
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.merge import merge as rio_merge

try:
    from pyproj import Transformer
except ImportError:
    print("ERROR: pyproj is required. Install with: pip install pyproj")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[1]
GISDATA    = REPO_ROOT / "GISData"
SKIP_DIRS  = {"AEmptyLayers", "APythonCodes"}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WINDOW_RADIUS_M      = 200.0
TPI_RADII_M          = [2.5, 5.0, 7.5]
TPI_FIELDS           = {2.5: "tpi2m5", 5.0: "tpi5m", 7.5: "tpi7m5"}
TWI_FIELD            = "twi"
FLOWACC_FIELD        = "flowAcc"
SLOPE_FIELD          = "slope"
AMSL_FIELD           = "amsl"
TPICD_FIELD          = "tpiCD"
NODATA_FALLBACK_MAX_M = 20  # max search radius (m) when tree pixel is nodata


# ---------------------------------------------------------------------------
# DEM discovery: find all .tif files for a city, build a VRT if needed
# ---------------------------------------------------------------------------
def find_dem_tiles(city_dir: Path) -> list[Path]:
    """Return list of DEM .tif paths for a city, empty list if none found."""
    # 1. Per-tree tiles from run_dem_dsm.py (DEM_tree_000.tif, DEM_tree_001.tif, …)
    dem_dir = city_dir / "DEM"
    if dem_dir.is_dir():
        per_tree = sorted(dem_dir.glob("DEM_tree_*.tif"))
        if per_tree:
            return per_tree
        # 2. Single city-wide raster (e.g. DEM_CityBerlin.tif from WCS download)
        tifs = sorted(dem_dir.glob("*.tif"))
        if tifs:
            return tifs

    # 3. Legacy tile collections in DEM*/ subfolders (DEMBamberg/, DEMBerlin/, …)
    candidates = []
    for sub in sorted(city_dir.iterdir()):
        if sub.is_dir() and sub.name.upper().startswith("DEM"):
            tifs = sorted(sub.glob("*.tif"))
            if tifs:
                candidates.extend(tifs)
    return candidates


def load_dem(tiles: list[Path]):
    """
    Load DEM from one or more tiles via rasterio.
    Returns (dem_array, nodata, gt_tuple, crs_wkt, cell_size).
    gt_tuple follows GDAL convention: (x_origin, pixel_w, rot, y_origin, rot, -pixel_h).
    """
    if len(tiles) == 1:
        with rasterio.open(tiles[0]) as src:
            arr    = src.read(1).astype(np.float64)
            nodata = src.nodata
            tf     = src.transform
            crs_wkt = src.crs.to_wkt() if src.crs else ""
    else:
        srcs = [rasterio.open(t) for t in tiles]
        try:
            arr_3d, tf = rio_merge(srcs, method="first")
        finally:
            for s in srcs:
                s.close()
        arr    = arr_3d[0].astype(np.float64)
        nodata = srcs[0].nodata
        crs_wkt = srcs[0].crs.to_wkt() if srcs[0].crs else ""

    # Convert rasterio Affine to GDAL-style 6-tuple
    gt = (tf.c, tf.a, tf.b, tf.f, tf.d, tf.e)

    cell_x = gt[1]
    cell_y = abs(gt[5])
    if not np.isclose(cell_x, cell_y, rtol=0.05):
        raise RuntimeError(f"Non-square pixels: {cell_x} vs {cell_y}")
    cell_size = (cell_x + cell_y) / 2

    return arr, nodata, gt, crs_wkt, cell_size


# ---------------------------------------------------------------------------
# Coordinate transformation
# ---------------------------------------------------------------------------
def get_epsg(crs_wkt: str) -> int | None:
    try:
        return rasterio.crs.CRS.from_wkt(crs_wkt).to_epsg()
    except Exception:
        return None


def make_transformer(from_crs, to_epsg: int):
    """Return a pyproj Transformer (x/y = lon/lat order)."""
    return Transformer.from_crs(from_crs, f"EPSG:{to_epsg}", always_xy=True)


# ---------------------------------------------------------------------------
# Raster index conversion
# ---------------------------------------------------------------------------
def world_to_pixel(x: float, y: float, gt):
    col = int((x - gt[0]) / gt[1])
    row = int((y - gt[3]) / gt[5])
    return col, row


# ---------------------------------------------------------------------------
# TWI: D8 flow accumulation + slope at tree cell
# ---------------------------------------------------------------------------
def compute_twi(dem: np.ndarray, nodata, cell_size: float,
                tree_row: int, tree_col: int):
    """
    Local D8 flow accumulation and TWI for one tree cell.
    Returns (twi, flow_acc, slope_deg) or (None, None, None) on failure.
    """
    H, W = dem.shape
    eps  = 1e-6

    # nodata mask
    nd_mask = np.zeros((H, W), dtype=bool)
    if nodata is not None:
        nd_mask = (dem == nodata) | np.isnan(dem)

    if nd_mask[tree_row, tree_col]:
        return None, None, None

    # D8 direction offsets
    DR = np.array([-1, -1,  0,  1,  1,  1,  0, -1], dtype=np.int32)
    DC = np.array([ 0,  1,  1,  1,  0, -1, -1, -1], dtype=np.int32)
    DIST = np.array([1, math.sqrt(2), 1, math.sqrt(2),
                     1, math.sqrt(2), 1, math.sqrt(2)]) * cell_size

    # ---- vectorised D8 flow direction ----
    flow_dir = np.full((H, W), -1, dtype=np.int8)

    # Pad the dem with a large value at borders so border cells flow inward
    pad = np.pad(dem, 1, mode="edge")
    nd_pad = np.pad(nd_mask, 1, mode="constant", constant_values=True)

    for d in range(8):
        dr, dc, dist = DR[d], DC[d], DIST[d]
        # slope from (r,c) toward neighbor
        z_neigh = pad[1 + DR[d]: H + 1 + DR[d], 1 + DC[d]: W + 1 + DC[d]]
        valid   = ~nd_pad[1 + DR[d]: H + 1 + DR[d], 1 + DC[d]: W + 1 + DC[d]]
        drop    = (dem - z_neigh) / dist
        # Update flow_dir only where this direction has the steepest drop so far
        current_best = np.where(flow_dir >= 0,
                                (dem - pad[1 + DR[flow_dir.clip(0,7).ravel()].reshape(H,W),
                                           1 + DC[flow_dir.clip(0,7).ravel()].reshape(H,W)])
                                / DIST[flow_dir.clip(0,7)],
                                -np.inf)
        # Simpler approach: iterate (small loop, 8 iters)
        pass  # handled below

    # Simpler scalar approach for flow_dir (8 iterations are fast enough)
    flow_dir[:] = -1
    best_drop = np.full((H, W), 0.0)
    for d in range(8):
        r_off, c_off = int(DR[d]), int(DC[d])
        dist_d = DIST[d]
        # shifted arrays
        r0s = max(0, -r_off);  r0e = H - max(0, r_off)
        c0s = max(0, -c_off);  c0e = W - max(0, c_off)
        r1s = max(0,  r_off);  r1e = H + min(0, r_off) if r_off < 0 else H - max(0, r_off) + r_off
        c1s = max(0,  c_off);  c1e = W + min(0, c_off) if c_off < 0 else W - max(0, c_off) + c_off
        # Clamp
        r1e = min(H, r1s + (r0e - r0s))
        c1e = min(W, c1s + (c0e - c0s))
        rows_h = r0e - r0s
        cols_w = c0e - c0s

        center = dem[r0s:r0s+rows_h, c0s:c0s+cols_w]
        neigh  = dem[r1s:r1s+rows_h, c1s:c1s+cols_w]
        nd_n   = nd_mask[r1s:r1s+rows_h, c1s:c1s+cols_w]
        drop   = (center - neigh) / dist_d

        update = (~nd_mask[r0s:r0s+rows_h, c0s:c0s+cols_w] &
                  ~nd_n & (drop > best_drop[r0s:r0s+rows_h, c0s:c0s+cols_w]))
        best_drop[r0s:r0s+rows_h, c0s:c0s+cols_w] = np.where(update, drop, best_drop[r0s:r0s+rows_h, c0s:c0s+cols_w])
        flow_dir[r0s:r0s+rows_h, c0s:c0s+cols_w]  = np.where(update, d, flow_dir[r0s:r0s+rows_h, c0s:c0s+cols_w])

    # ---- flow accumulation (high-to-low order) ----
    flow_acc = np.ones((H, W), dtype=np.float64)
    flow_acc[nd_mask] = 0.0
    order = np.argsort(dem.ravel())[::-1]
    for flat_idx in order:
        r = flat_idx // W
        c = flat_idx % W
        if nd_mask[r, c]:
            continue
        d = int(flow_dir[r, c])
        if d < 0:
            continue
        rr = r + int(DR[d])
        cc = c + int(DC[d])
        if 0 <= rr < H and 0 <= cc < W and not nd_mask[rr, cc]:
            flow_acc[rr, cc] += flow_acc[r, c]

    # ---- slope at tree cell (central difference) ----
    r, c = tree_row, tree_col

    def safe_z(rr, cc):
        if rr < 0 or rr >= H or cc < 0 or cc >= W or nd_mask[rr, cc]:
            return dem[r, c]
        return dem[rr, cc]

    dzdx = (safe_z(r, c + 1) - safe_z(r, c - 1)) / (2 * cell_size)
    dzdy = (safe_z(r - 1, c) - safe_z(r + 1, c)) / (2 * cell_size)
    slope_rad = math.atan(math.sqrt(dzdx ** 2 + dzdy ** 2))

    # ---- TWI ----
    a   = flow_acc[r, c] * (cell_size ** 2)     # upslope contributing area (m²)
    twi = math.log((a + eps) / math.tan(slope_rad + eps))

    return float(twi), float(flow_acc[r, c]), math.degrees(slope_rad)


# ---------------------------------------------------------------------------
# TPI: circular neighbourhood, centre excluded
# ---------------------------------------------------------------------------
def make_circular_mask(radius_m: float, cell_size: float):
    """Circular boolean mask (centre=False) for given radius in metres."""
    r_pix = max(1, int(round(radius_m / cell_size)))
    size  = 2 * r_pix + 1
    yy, xx = np.indices((size, size))
    dist = np.sqrt((xx - r_pix) ** 2 + (yy - r_pix) ** 2)
    mask = dist < r_pix + 1e-9
    mask[r_pix, r_pix] = False
    return mask, r_pix


def _nearest_valid_pixel(dem: np.ndarray, nodata, row_px: int, col_px: int,
                         max_radius_px: int):
    """
    Search outward from (row_px, col_px) for the nearest pixel that is neither
    nodata nor NaN.  Returns (row, col, distance_px) or (None, None, None) if
    nothing is found within max_radius_px.  Uses Chebyshev (L∞) distance so
    the search expands as a square ring — good enough for a small fallback.
    """
    H, W = dem.shape
    for radius in range(1, max_radius_px + 1):
        r0 = max(0, row_px - radius)
        r1 = min(H - 1, row_px + radius)
        c0 = max(0, col_px - radius)
        c1 = min(W - 1, col_px + radius)
        # iterate only the outer ring of the current square
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if abs(r - row_px) != radius and abs(c - col_px) != radius:
                    continue
                v = dem[r, c]
                if (nodata is None or v != nodata) and not np.isnan(float(v)):
                    return r, c, radius
    return None, None, None


def compute_tpi(dem: np.ndarray, nodata, row: int, col: int,
                circular_masks: dict):
    """
    Returns dict {radius_m: tpi_value or None} for all radii.
    circular_masks: {radius_m: (mask_array, r_pix)}
    """
    H, W = dem.shape
    results = {}
    for r_m, (mask, r_pix) in circular_masks.items():
        row_min = max(0, row - r_pix)
        row_max = min(H - 1, row + r_pix)
        col_min = max(0, col - r_pix)
        col_max = min(W - 1, col + r_pix)

        sub = dem[row_min:row_max + 1, col_min:col_max + 1]

        # Align mask to the potentially edge-clipped window
        mr_min = r_pix - (row - row_min)
        mr_max = mr_min + sub.shape[0]
        mc_min = r_pix - (col - col_min)
        mc_max = mc_min + sub.shape[1]
        mask_sub = mask[mr_min:mr_max, mc_min:mc_max]

        valid = mask_sub.copy()
        if nodata is not None:
            valid &= (sub != nodata)
        valid &= ~np.isnan(sub)

        neigh = sub[valid]
        if neigh.size == 0:
            results[r_m] = None
        else:
            centre = dem[row, col]
            results[r_m] = float(centre - neigh.mean())

    return results


# ---------------------------------------------------------------------------
# Per-city processing
# ---------------------------------------------------------------------------
def process_city(city_dir: Path, skip_filled: bool, window_m: float) -> bool:
    city = city_dir.name
    vec_dir = city_dir / "VectorLayers"
    gpkg = vec_dir / "treeLocations.gpkg"
    shp  = vec_dir / "treeLocations.shp"
    # .shp is the authoritative source; write to .gpkg (avoids Windows file locking)
    src_file = shp if shp.exists() else gpkg
    out_file = gpkg
    if not src_file.exists():
        print(f"  [{city}] No treeLocations.shp/.gpkg — skipping")
        return False

    tiles = find_dem_tiles(city_dir)
    if not tiles:
        print(f"  [{city}] No DEM tiles found — skipping")
        return False

    print(f"\n{'='*60}")
    print(f"  {city}: {len(tiles)} DEM tile(s)")

    # Load vector layer
    gdf = gpd.read_file(src_file)
    n_trees = len(gdf)
    if n_trees == 0:
        print(f"  [{city}] Shapefile is empty — skipping")
        return False

    # Ensure output columns exist
    for col in ([TWI_FIELD, FLOWACC_FIELD, SLOPE_FIELD, AMSL_FIELD, TPICD_FIELD]
                + list(TPI_FIELDS.values())):
        if col not in gdf.columns:
            gdf[col] = np.nan

    # Determine which trees need processing
    if skip_filled:
        needs = gdf[TWI_FIELD].isna() | (gdf[TWI_FIELD] == 0)
    else:
        needs = pd.Series(True, index=gdf.index)

    n_to_process = needs.sum()
    if n_to_process == 0:
        print(f"  [{city}] All trees already have TWI — skipping (use --no-skip to recompute)")
        return True

    print(f"  Trees to process: {n_to_process} / {n_trees}")

    # Load DEM
    dem, nodata, gt, crs_wkt, cell_size = load_dem(tiles)
    n_rows, n_cols = dem.shape
    print(f"  DEM: {n_cols}×{n_rows} px, cell={cell_size} m, nodata={nodata}")
    epsg_dem = get_epsg(crs_wkt)
    if epsg_dem is None:
        print(f"  WARNING: cannot identify DEM EPSG — assuming EPSG:25832")
        epsg_dem = 25832
    print(f"  DEM CRS: EPSG:{epsg_dem}")

    # Transformer: shapefile CRS → DEM CRS
    shp_crs = gdf.crs
    transformer = make_transformer(shp_crs, epsg_dem) if shp_crs else None

    # Build TPI circular masks (only once per DEM cell size)
    circ_masks = {r: make_circular_mask(r, cell_size) for r in TPI_RADII_M}
    cd_mask_cache = {}  # crown_radius_m (rounded to 0.1 m) → (mask, r_pix)

    window_pix = int(round(window_m / cell_size))
    print(f"  TWI window: {window_m} m = {window_pix} px radius")

    # Counters
    n_ok = n_null = n_outside = 0

    for idx in gdf.index[needs]:
        row_data = gdf.loc[idx]
        tree_name = str(row_data.get("treeName", f"idx_{idx}"))

        geom = row_data.geometry
        if geom is None or geom.is_empty:
            for col in ([TWI_FIELD, FLOWACC_FIELD, SLOPE_FIELD, AMSL_FIELD, TPICD_FIELD]
                        + list(TPI_FIELDS.values())):
                gdf.at[idx, col] = np.nan
            n_null += 1
            continue

        pt = geom.centroid
        x, y = pt.x, pt.y

        # Reproject to DEM CRS
        if transformer is not None:
            x, y = transformer.transform(x, y)

        col_px, row_px = world_to_pixel(x, y, gt)

        if not (0 <= col_px < n_cols and 0 <= row_px < n_rows):
            for col in ([TWI_FIELD, FLOWACC_FIELD, SLOPE_FIELD, AMSL_FIELD, TPICD_FIELD]
                        + list(TPI_FIELDS.values())):
                gdf.at[idx, col] = np.nan
            n_outside += 1
            print(f"    {tree_name}: OUTSIDE_DEM (x={x:.0f}, y={y:.0f})")
            continue

        if nodata is not None and dem[row_px, col_px] == nodata:
            fb_row, fb_col, fb_dist = _nearest_valid_pixel(
                dem, nodata, row_px, col_px,
                int(round(NODATA_FALLBACK_MAX_M / cell_size)))
            if fb_row is None:
                for col in ([TWI_FIELD, FLOWACC_FIELD, SLOPE_FIELD, AMSL_FIELD, TPICD_FIELD]
                            + list(TPI_FIELDS.values())):
                    gdf.at[idx, col] = np.nan
                n_null += 1
                print(f"    {tree_name}: NODATA — no valid pixel within "
                      f"{NODATA_FALLBACK_MAX_M} m → skipped")
                continue
            print(f"    {tree_name}: NODATA at tree pixel → "
                  f"fallback to nearest valid pixel ({fb_dist} m away)")
            row_px, col_px = fb_row, fb_col

        # --- AMSL (ground elevation from DEM at tree location) ---
        amsl_raw = dem[row_px, col_px]
        if np.isnan(amsl_raw):
            gdf.at[idx, AMSL_FIELD] = np.nan
        else:
            gdf.at[idx, AMSL_FIELD] = round(float(amsl_raw), 2)

        # --- TWI ---
        r_min = max(0, row_px - window_pix)
        r_max = min(n_rows - 1, row_px + window_pix)
        c_min = max(0, col_px - window_pix)
        c_max = min(n_cols - 1, col_px + window_pix)
        sub_dem   = dem[r_min:r_max + 1, c_min:c_max + 1]
        local_row = row_px - r_min
        local_col = col_px - c_min

        twi_val, flowacc_val, slope_deg = compute_twi(
            sub_dem, nodata, cell_size, local_row, local_col)

        if twi_val is None:
            gdf.at[idx, TWI_FIELD]      = np.nan
            gdf.at[idx, FLOWACC_FIELD]  = np.nan
            gdf.at[idx, SLOPE_FIELD]    = np.nan
        else:
            gdf.at[idx, TWI_FIELD]     = round(twi_val, 5)
            gdf.at[idx, FLOWACC_FIELD] = round(flowacc_val, 2)
            gdf.at[idx, SLOPE_FIELD]   = round(slope_deg, 4)

        # --- TPI (fixed radii) ---
        tpi_vals = compute_tpi(dem, nodata, row_px, col_px, circ_masks)
        for r_m, tpi_v in tpi_vals.items():
            fname = TPI_FIELDS[r_m]
            gdf.at[idx, fname] = round(tpi_v, 5) if tpi_v is not None else np.nan

        # --- tpiCD (TPI at crown-diameter radius, per tree) ---
        tpicd_val = np.nan
        try:
            if "crownDiam" in gdf.columns:
                cd = gdf.at[idx, "crownDiam"]
                cd = float(cd)
                if not pd.isna(cd) and cd > 0:
                    cd_r = cd / 2.0
                    cd_key = round(cd_r, 1)
                    if cd_key not in cd_mask_cache:
                        cd_mask_cache[cd_key] = make_circular_mask(cd_r, cell_size)
                    cd_tpi = compute_tpi(dem, nodata, row_px, col_px,
                                         {cd_r: cd_mask_cache[cd_key]})
                    v = cd_tpi.get(cd_r)
                    tpicd_val = round(v, 5) if v is not None else np.nan
        except (TypeError, ValueError):
            pass
        gdf.at[idx, TPICD_FIELD] = tpicd_val

        n_ok += 1
        if n_ok % 10 == 0 or n_ok == n_to_process:
            print(f"    {n_ok}/{n_to_process} done  "
                  f"(last: {tree_name} | amsl={amsl_raw:.1f}, "
                  f"TWI={twi_val:.3f}, slope={slope_deg:.2f}°, "
                  f"TPI2.5={tpi_vals.get(2.5, 'n/a')!r:.4}, "
                  f"tpiCD={tpicd_val!r:.6})")

    print(f"  Results: {n_ok} computed, {n_outside} outside DEM, {n_null} nodata/null")

    # Write back as GeoPackage (single file, avoids Windows .shp locking)
    # Recreate the file each time so stale extra layers don't accumulate
    out_gdf = gdf.drop(columns=["fid"], errors="ignore")
    if out_file.exists():
        out_file.unlink()
    out_gdf.to_file(out_file, driver="GPKG", layer="treeLocations")
    print(f"  Written -> {out_file}")

    # Also sync terrain values back to .shp (source of truth for other tools)
    # flowAcc (internal name) → flotAcc (pre-existing .shp column stub)
    if shp.exists():
        try:
            shp_gdf = gdf.copy()
            shp_gdf["flotAcc"] = shp_gdf[FLOWACC_FIELD]
            shp_gdf = shp_gdf.drop(columns=[FLOWACC_FIELD, "fid"], errors="ignore")
            shp_gdf.to_file(shp, driver="ESRI Shapefile")
            print(f"  Written -> {shp.name}")
        except PermissionError:
            print(f"  WARNING: {shp.name} locked (QGIS open?) — terrain values only in .gpkg")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute TWI + TPI for all GISData cities")
    parser.add_argument("--city", nargs="+", default=None,
                        help="Process only these city folder names (default: all)")
    parser.add_argument("--skip-filled", action="store_true",
                        help="Skip trees that already have a non-null twi value")
    parser.add_argument("--window-m", type=float, default=WINDOW_RADIUS_M,
                        help=f"TWI local window radius in metres (default: {WINDOW_RADIUS_M})")
    args = parser.parse_args()

    city_dirs = sorted(
        d for d in GISDATA.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS
    )

    if args.city:
        city_dirs = [d for d in city_dirs if d.name in args.city]
        if not city_dirs:
            print(f"ERROR: none of the specified cities found under {GISDATA}")
            sys.exit(1)

    print(f"TreeDataBase — Terrain Metrics Calculator")
    print(f"Repo:    {REPO_ROOT}")
    print(f"GISData: {GISDATA}")
    print(f"Cities:  {len(city_dirs)}  |  window={args.window_m} m  "
          f"|  skip_filled={args.skip_filled}")

    ok = skipped = 0
    for city_dir in city_dirs:
        result = process_city(city_dir, args.skip_filled, args.window_m)
        if result:
            ok += 1
        else:
            skipped += 1

    print(f"\n{'='*60}")
    print(f"Done. Processed: {ok}  |  Skipped (no DEM / no shp): {skipped}")


if __name__ == "__main__":
    main()
