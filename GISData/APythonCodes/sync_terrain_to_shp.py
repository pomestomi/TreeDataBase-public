"""
sync_terrain_to_shp.py
One-time backfill: copies terrain metrics from treeLocations.gpkg → treeLocations.shp
for all city folders under GISData/.

Column mapping (gpkg name → shp name):
  twi      → twi
  flowAcc  → flotAcc   (flotAcc is the pre-existing .shp stub)
  slope    → slope
  tpi2m5   → tpi2m5
  tpi5m    → tpi5m
  tpi7m5   → tpi7m5
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GISDATA   = REPO_ROOT / "GISData"
SKIP_DIRS = {"APythonCodes", "Figures", "GeneralData", "QGIS"}

TERRAIN_MAP = {
    "amsl":    "amsl",
    "twi":     "twi",
    "flowAcc": "flotAcc",
    "slope":   "slope",
    "tpi2m5":  "tpi2m5",
    "tpi5m":   "tpi5m",
    "tpi7m5":  "tpi7m5",
}


def read_gpkg_best_layer(gpkg_p: Path) -> gpd.GeoDataFrame:
    """Read the layer that has terrain data; fall back to first layer."""
    import pyogrio
    layers = [row[0] for row in pyogrio.list_layers(str(gpkg_p))]
    # Prefer layer named 'treeLocations'; then pick first with non-null twi
    preferred = ["treeLocations"] + [l for l in layers if l != "treeLocations"]
    for layer in preferred:
        gdf = gpd.read_file(gpkg_p, layer=layer)
        if "twi" in gdf.columns and gdf["twi"].notna().any():
            return gdf
    # Last resort: return first layer as-is
    return gpd.read_file(gpkg_p, layer=layers[0])


def sync_city(city_dir: Path) -> str:
    city     = city_dir.name
    vec_dir  = city_dir / "VectorLayers"
    gpkg_p   = vec_dir / "treeLocations.gpkg"
    shp_p    = vec_dir / "treeLocations.shp"

    if not shp_p.exists():
        return f"  [{city}] No .shp — skipped"
    if not gpkg_p.exists():
        return f"  [{city}] No .gpkg — skipped"

    gpkg = read_gpkg_best_layer(gpkg_p)
    shp  = gpd.read_file(shp_p)

    if len(gpkg) != len(shp):
        return (f"  [{city}] Row count mismatch "
                f"({len(gpkg)} gpkg vs {len(shp)} shp) — skipped")

    synced = []
    for gpkg_col, shp_col in TERRAIN_MAP.items():
        if gpkg_col not in gpkg.columns:
            continue
        vals = gpkg[gpkg_col]
        if vals.isna().all():
            continue
        if shp_col not in shp.columns:
            shp[shp_col] = pd.NA
        shp[shp_col] = vals.values
        synced.append(shp_col)

    if not synced:
        return f"  [{city}] No terrain values found in .gpkg — skipped"

    try:
        shp.drop(columns=["fid"], errors="ignore").to_file(shp_p, driver="ESRI Shapefile")
        return f"  {city}: {', '.join(synced)} -> {shp_p.name}  ({len(shp)} rows)"
    except PermissionError:
        return f"  [{city}] PermissionError writing .shp (QGIS open?)"


def main():
    city_dirs = sorted(
        d for d in GISDATA.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS
    )
    print(f"Syncing terrain metrics from .gpkg -> .shp  ({len(city_dirs)} candidate folders)\n")
    ok = skipped = errors = 0
    for city_dir in city_dirs:
        msg = sync_city(city_dir)
        print(msg)
        if "skipped" in msg or "No .shp" in msg or "No .gpkg" in msg:
            skipped += 1
        elif "PermissionError" in msg or "mismatch" in msg:
            errors += 1
        else:
            ok += 1
    print(f"\nDone. Synced: {ok}  |  Skipped: {skipped}  |  Errors: {errors}")


if __name__ == "__main__":
    main()
