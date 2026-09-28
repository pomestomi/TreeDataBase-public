#!/usr/bin/env python3
"""
Land Use Area Calculator for TreeDataBase
==========================================
Reads polygon shapefiles from VectorLayers/ subfolders, computes the
area of each polygon, sums areas per devEUI and land use category, and
writes the results back to the treeLocations shapefile.

Sealed surface area is not directly digitised but computed by subtracting
the labelled areas (greenAttached, greenDetached, buildings) from the
theoretical annulus area at each radius.

Folder structure expected per city:
    GISData/{City}/
    └── VectorLayers/
        ├── treeLocations.shp           <-- target: land use fields written here
        ├── greenAtta2m5.shp            <-- polygons: attached green, 0–2.5 m
        ├── greenAtta5m0.shp            <-- polygons: attached green, 2.5–5 m ring
        ├── greenAtta7m5.shp            <-- polygons: attached green, 5–7.5 m ring
        ├── greenDeta2m5.shp            <-- polygons: detached green, 0–2.5 m
        ├── greenDeta5m0.shp            <-- ...
        ├── greenDeta7m5.shp
        ├── buildings2m5.shp
        ├── buildings5m0.shp
        └── buildings7m5.shp

Radii and annulus geometry:
    2m5: full circle     r = 0 – 2.5 m   →  pi * 2.5^2            = 19.635 m^2
    5m0: annulus          r = 2.5 – 5.0 m →  pi * 5^2 - pi * 2.5^2 = 58.905 m^2
    7m5: annulus          r = 5.0 – 7.5 m →  pi * 7.5^2 - pi * 5^2 = 98.175 m^2

Plausibility rule:
    If the sum of labelled areas at a given radius exceeds the annulus
    area, all categories are proportionally scaled down so that their
    sum equals the annulus area exactly.

Usage:
    cd src
    python compute_landuse.py                     # process all cities
    python compute_landuse.py --city CityErlangen # process one city
    python compute_landuse.py --dry-run           # compute but don't write

Requirements:
    pip install geopandas pandas
"""

import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GIS_DIR = REPO_ROOT / "GISData"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RADII = {
    "2m5": {"r_inner": 0.0, "r_outer": 2.5},
    "5m0": {"r_inner": 2.5, "r_outer": 5.0},
    "7m5": {"r_inner": 5.0, "r_outer": 7.5},
}
ANNULUS_AREA = {
    k: math.pi * v["r_outer"]**2 - math.pi * v["r_inner"]**2
    for k, v in RADII.items()
}

CATEGORIES = ["greenAtta", "greenDeta", "buildings"]

# Shapefile field names (10-char limit)
SHP_FIELD = {
    ("greenAtta", "2m5"): "greenAtta2", ("greenAtta", "5m0"): "greenAtta5",
    ("greenAtta", "7m5"): "greenAtta7",
    ("greenDeta", "2m5"): "greenDeta2", ("greenDeta", "5m0"): "greenDeta5",
    ("greenDeta", "7m5"): "greenDeta7",
    ("buildings", "2m5"): "buildings2", ("buildings", "5m0"): "buildings5",
    ("buildings", "7m5"): "buildings7",
}
SEALED_SHP = {"2m5": "sealedSu2", "5m0": "sealedSu5", "7m5": "sealedSu7"}

# Crown-diameter based fields (10-char DBF limit)
CROWN_FIELDS = {
    "greenAtta": "greenAttCD",
    "greenDeta": "greenDetCD",
    "buildings": "buildingCD",
    "sealed":    "sealedSuCD",
}
CAP_RADIUS = max(v["r_outer"] for v in RADII.values())  # 7.5 m


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def compute_areas_for_city(city_dir: Path):
    """Read polygon layers, compute areas, return results per devEUI."""
    vl_dir = city_dir / "VectorLayers"
    if not vl_dir.is_dir():
        return {}

    raw = {}  # {devEUI: {(category, radius): sum_area}}

    for cat in CATEGORIES:
        for rad_key in RADII:
            shp_path = vl_dir / f"{cat}{rad_key}.shp"
            if not shp_path.exists():
                continue
            gdf = gpd.read_file(shp_path)
            if gdf.empty:
                continue
            gdf["_area"] = gdf.geometry.area
            for eui, grp in gdf.groupby("devEUI"):
                eui = str(eui).strip().upper()
                if eui not in raw:
                    raw[eui] = {}
                raw[eui][(cat, rad_key)] = grp["_area"].sum()

    # Compute sealed surface and apply plausibility correction
    results = {}
    for eui, layer_areas in raw.items():
        rec = {}
        for rad_key in RADII:
            max_area = ANNULUS_AREA[rad_key]
            ga = layer_areas.get(("greenAtta", rad_key), 0.0)
            gd = layer_areas.get(("greenDeta", rad_key), 0.0)
            bu = layer_areas.get(("buildings", rad_key), 0.0)
            total_labeled = ga + gd + bu

            if total_labeled > max_area and total_labeled > 0:
                scale = max_area / total_labeled
                ga *= scale
                gd *= scale
                bu *= scale
                total_labeled = max_area

            sealed = max(0.0, max_area - total_labeled)
            rec[("greenAtta", rad_key)] = round(ga, 2)
            rec[("greenDeta", rad_key)] = round(gd, 2)
            rec[("buildings", rad_key)] = round(bu, 2)
            rec[("sealed", rad_key)] = round(sealed, 2)

        results[eui] = rec
    return results


def write_to_shapefile(shp_path: Path, results: dict):
    """Write computed land use values to treeLocations.shp."""
    gdf = gpd.read_file(shp_path)
    # Ensure all expected columns exist (creates them as None if absent)
    all_expected = list(SHP_FIELD.values()) + list(SEALED_SHP.values())
    for col in all_expected:
        if col not in gdf.columns:
            gdf[col] = None
    updated = 0
    for idx, row in gdf.iterrows():
        eui = str(row.get("devEUI", "")).strip().upper()
        if eui not in results:
            continue
        rec = results[eui]
        for (cat, rad), val in rec.items():
            field = SEALED_SHP[rad] if cat == "sealed" else SHP_FIELD[(cat, rad)]
            gdf.at[idx, field] = val
        updated += 1
    gdf.to_file(shp_path, driver="ESRI Shapefile")
    return updated


def update_polygon_area_fields(city_dir: Path):
    """Write computed area into the 'area' field of each polygon shapefile."""
    vl_dir = city_dir / "VectorLayers"
    for cat in CATEGORIES:
        for rad_key in RADII:
            shp_path = vl_dir / f"{cat}{rad_key}.shp"
            if not shp_path.exists():
                continue
            gdf = gpd.read_file(shp_path)
            if gdf.empty:
                continue
            gdf["area"] = gdf.geometry.area.round(2)
            gdf.to_file(shp_path, driver="ESRI Shapefile")


def compute_crown_areas_for_city(city_dir: Path) -> dict:
    """Compute land use areas within the crown-diameter circle of each tree.

    The radius used is crownDiam / 2, capped at CAP_RADIUS (7.5 m) because
    polygon layers are only digitised up to that distance.  Trees with a
    crown diameter > 15 m are flagged on the console and processed with the
    capped radius.  Trees without a valid crownDiam are skipped.

    Returns {devEUI: {"greenAtta": m², "greenDeta": m², "buildings": m²,
                      "sealed": m², "radius_used": m}}.
    """
    vl_dir = city_dir / "VectorLayers"
    if not vl_dir.is_dir():
        return {}

    shp_path = vl_dir / "treeLocations.shp"
    if not shp_path.exists():
        return {}

    trees = gpd.read_file(shp_path)
    if trees.empty:
        return {}

    # Merge all ring polygon layers per category into one combined GDF
    merged: dict[str, gpd.GeoDataFrame] = {}
    ref_crs = None
    for cat in CATEGORIES:
        frames = []
        for rad_key in RADII:
            p = vl_dir / f"{cat}{rad_key}.shp"
            if not p.exists():
                continue
            gdf = gpd.read_file(p)
            if gdf.empty:
                continue
            if ref_crs is None:
                ref_crs = gdf.crs
            sub = gdf[["geometry", "devEUI"]].copy()
            # Repair any invalid geometries (self-intersections, etc.)
            sub["geometry"] = sub["geometry"].buffer(0)
            frames.append(sub)
        if frames:
            combined = gpd.GeoDataFrame(
                pd.concat(frames, ignore_index=True),
                geometry="geometry", crs=ref_crs,
            )
            combined["_eui"] = (combined["devEUI"]
                                .astype(str).str.strip().str.upper())
            merged[cat] = combined

    # Align tree CRS with polygon CRS
    if ref_crs is not None and trees.crs is not None and trees.crs != ref_crs:
        trees = trees.to_crs(ref_crs)

    results = {}
    for _, row in trees.iterrows():
        eui = str(row.get("devEUI", "")).strip().upper()
        if not eui or eui == "NAN":
            continue

        crown_diam = row.get("crownDiam")
        try:
            crown_diam = float(crown_diam)
        except (TypeError, ValueError):
            continue
        if pd.isna(crown_diam) or crown_diam <= 0:
            continue

        radius = crown_diam / 2.0
        if radius > CAP_RADIUS:
            print(f"      [INFO] {eui}: crownDiam {crown_diam:.1f} m > "
                  f"{CAP_RADIUS * 2:.0f} m — labelled area only reaches "
                  f"{CAP_RADIUS} m, capping radius there.")
            radius = CAP_RADIUS

        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        # Support MultiPoint (take first component)
        if geom.geom_type == "MultiPoint":
            geom = list(geom.geoms)[0]

        crown_circle = geom.buffer(radius)
        circle_area  = crown_circle.area

        rec: dict[str, float] = {}
        total_labeled = 0.0

        for cat in CATEGORIES:
            if cat not in merged:
                rec[cat] = 0.0
                continue
            tree_polys = merged[cat][merged[cat]["_eui"] == eui]
            if tree_polys.empty:
                rec[cat] = 0.0
                continue
            # Per-geometry intersection with individual fallback so a single
            # invalid polygon does not abort the whole tree.
            area = 0.0
            for poly in tree_polys.geometry:
                try:
                    area += poly.intersection(crown_circle).area
                except Exception:
                    try:
                        area += poly.buffer(0).intersection(crown_circle).area
                    except Exception:
                        pass
            area = float(area)
            rec[cat] = round(area, 2)
            total_labeled += area

        # Proportional plausibility correction (same rule as fixed rings)
        if total_labeled > circle_area and total_labeled > 0:
            scale = circle_area / total_labeled
            for cat in CATEGORIES:
                rec[cat] = round(rec[cat] * scale, 2)
            total_labeled = circle_area

        rec["sealed"]      = round(max(0.0, circle_area - total_labeled), 2)
        rec["radius_used"] = round(radius, 3)
        results[eui] = rec

    return results


def write_crown_to_shapefile(shp_path: Path, results: dict) -> int:
    """Write crown-diameter land use fields to treeLocations.shp."""
    gdf = gpd.read_file(shp_path)
    for col in CROWN_FIELDS.values():
        if col not in gdf.columns:
            gdf[col] = None
    updated = 0
    for idx, row in gdf.iterrows():
        eui = str(row.get("devEUI", "")).strip().upper()
        if eui not in results:
            continue
        rec = results[eui]
        for cat_key, field in CROWN_FIELDS.items():
            gdf.at[idx, field] = rec.get(cat_key)
        updated += 1
    gdf.to_file(shp_path, driver="ESRI Shapefile")
    return updated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute land use areas from polygon layers")
    parser.add_argument("--city", help="Process only this city folder name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print without writing to files")
    args = parser.parse_args()

    if not GIS_DIR.is_dir():
        print(f"ERROR: GISData not found: {GIS_DIR}")
        sys.exit(1)

    print("=" * 65)
    print("  LAND USE AREA CALCULATOR")
    print("=" * 65)
    print(f"\n  Annulus areas:")
    for k, v in ANNULUS_AREA.items():
        r = RADII[k]
        print(f"    {r['r_inner']}–{r['r_outer']} m: {v:.3f} m²")

    # Discover city folders with VectorLayers/
    city_dirs = []
    for entry in sorted(GIS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if args.city and entry.name != args.city:
            continue
        if (entry / "VectorLayers").is_dir():
            city_dirs.append(entry)
        for sub in entry.iterdir():
            if sub.is_dir() and (sub / "VectorLayers").is_dir():
                if not args.city or sub.name == args.city:
                    city_dirs.append(sub)

    if not city_dirs:
        print("\n  No cities with VectorLayers/ found.")
        return

    total_updated = 0
    for city_dir in city_dirs:
        vl_dir = city_dir / "VectorLayers"
        city_name = city_dir.name
        print(f"\n  Processing: {city_name}")

        n_layers = sum(1 for c in CATEGORIES for r in RADII
                       if (vl_dir / f"{c}{r}.shp").exists())
        print(f"    Polygon layers found: {n_layers}")

        results = compute_areas_for_city(city_dir)
        print(f"    Trees with polygon data: {len(results)}")
        if not results:
            continue

        # Count plausibility corrections
        n_corr = 0
        for eui, rec in results.items():
            for rad_key in RADII:
                ga = rec.get(("greenAtta", rad_key), 0)
                gd = rec.get(("greenDeta", rad_key), 0)
                bu = rec.get(("buildings", rad_key), 0)
                se = rec.get(("sealed", rad_key), 0)
                # Sealed < 0 would indicate overflow (already corrected)
                if se == 0 and (ga + gd + bu) > ANNULUS_AREA[rad_key] * 0.999:
                    n_corr += 1
        if n_corr > 0:
            print(f"    Plausibility corrections applied: {n_corr}")

        # Crown-diameter based land use (computed regardless of dry-run)
        results_cd = compute_crown_areas_for_city(city_dir)
        n_cd_trees = len(results_cd)
        print(f"    Trees with crown diameter data: {n_cd_trees}")

        if args.dry_run:
            for eui in list(results.keys())[:5]:
                rec = results[eui]
                print(f"\n    {eui}:")
                for rad_key in ["2m5", "5m0", "7m5"]:
                    r = RADII[rad_key]
                    ga = rec.get(("greenAtta", rad_key), 0)
                    gd = rec.get(("greenDeta", rad_key), 0)
                    bu = rec.get(("buildings", rad_key), 0)
                    se = rec.get(("sealed", rad_key), 0)
                    print(f"      {r['r_inner']}–{r['r_outer']}m: "
                          f"gA={ga:.1f}  gD={gd:.1f}  "
                          f"bld={bu:.1f}  seal={se:.1f}  "
                          f"Σ={ga+gd+bu+se:.1f}")
                if eui in results_cd:
                    rcd = results_cd[eui]
                    r_used = rcd["radius_used"]
                    print(f"      crown (r={r_used:.2f} m): "
                          f"gA={rcd['greenAtta']:.1f}  "
                          f"gD={rcd['greenDeta']:.1f}  "
                          f"bld={rcd['buildings']:.1f}  "
                          f"seal={rcd['sealed']:.1f}  "
                          f"Σ={rcd['greenAtta']+rcd['greenDeta']+rcd['buildings']+rcd['sealed']:.1f}")
            continue

        # Write to shapefile only
        shp_path = vl_dir / "treeLocations.shp"
        if shp_path.exists():
            n = write_to_shapefile(shp_path, results)
            print(f"    Written to treeLocations.shp: {n} trees (fixed rings)")
            total_updated += n

            if results_cd:
                n_cd = write_crown_to_shapefile(shp_path, results_cd)
                print(f"    Written to treeLocations.shp: {n_cd} trees "
                      f"(crown-diameter fields)")

        update_polygon_area_fields(city_dir)
        print(f"    Updated 'area' field in polygon shapefiles")

    print(f"\n{'=' * 65}")
    print(f"  DONE — {total_updated} trees updated across "
          f"{len(city_dirs)} cities")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
