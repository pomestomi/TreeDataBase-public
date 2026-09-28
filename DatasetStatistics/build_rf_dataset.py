#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_rf_dataset.py — Join bsc_events_all.csv with treeLocations shapefile metadata.

Inputs:
  TreeTabularData/bsc_events_all.csv          (produced by src/_poc_irrigation_v3.py)
  GISData/**/VectorLayers/treeLocations.shp   (one per city/project folder)

Output:
  TreeTabularData/rf_dataset.csv
  — one row per rain event with all sensor-derived metrics AND all site attributes joined.

Column groups in the output:
  Rain event        ev_start, ev_end, total_mm, duration_h, max_intensity_mmh, stac,
                    huff_q, bsc, month, doy, rain_pre/post 24/72 h
  BSC bits/areas    bsc_s1..4, area_1..4
  VWC metrics       pre_vwc, peak_vwc, delta_vwc, delta_pct_dyn, t_peak_min, dry_h,
                    temp_sum, vwc_p05/p95, dyn_range  — all × 3 depths (-10/-30/-45 cm)
  Tree biology      genus, species, variety, height, crownDiam_m, stemDiam_m, vitality
  GIS / terrain     greenDeta/Atta 2.5/5/7.5 m, buildings 2.5/5/7.5 m,
                    sealedSurf 2.5/5/7.5 m, tpi 2.5/5/7.5 m, twi, slope_deg,
                    flowAccumulation, elevation_m, depthToGroundwater_m, skyViewFactor
  Soil chemistry    pH, conductivity, saltContent, nSoluble, ammNSoluble, nitrNSoluble,
                    mgSoluble, pHSoluble, kSoluble
  Soil texture      part1Perc .. part7Perc
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

from pathlib import Path
import pandas as pd
import geopandas as gpd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
BSC_CSV   = REPO_ROOT / "TreeTabularData" / "bsc_events_all.csv"
GIS_DIR   = REPO_ROOT / "GISData"
OUT_CSV   = REPO_ROOT / "TreeTabularData" / "rf_dataset.csv"

# ---------------------------------------------------------------------------
# Column renames: shapefile DBF names → readable equivalents
# DBF format limits field names to 10 chars; geopandas appends _1/_2 for dupes.
# ---------------------------------------------------------------------------
_COL_RENAME = {
    # Sealed surface (three radii — sealedSurf is already 10 chars so geopandas
    # renamed the 5m and 7.5m versions to sealedSu_1 / sealedSu_2)
    "sealedSurf": "sealedSurf_2m5",
    "sealedSu_1": "sealedSurf_5m",
    "sealedSu_2": "sealedSurf_7m5",
    # Green cover — detached / attached at 2.5 / 5 / 7.5 m radius
    "greenDeta2": "greenDeta_2m5",
    "greenDeta5": "greenDeta_5m",
    "greenDeta7": "greenDeta_7m5",
    "greenAtta2": "greenAtta_2m5",
    "greenAtta5": "greenAtta_5m",
    "greenAtta7": "greenAtta_7m5",
    # Building footprint area at 2.5 / 5 / 7.5 m
    "buildings2": "buildings_2m5",
    "buildings5": "buildings_5m",
    "buildings7": "buildings_7m5",
    # Terrain
    "slope":      "slope_deg",
    "flotAcc":    "flowAccumulation",
    "amsl":       "elevation_m",
    "DTGW":       "depthToGroundwater_m",
    "SVF":        "skyViewFactor",
    # Soil chemistry
    "conductiv":  "conductivity",
    "saltCont":   "saltContent",
    "ammNSolubl": "ammNSoluble",
    "nitrNSolub": "nitrNSoluble",
    "phSoluble":  "pHSoluble",
    # Tree measurements
    "crownDiam":  "crownDiam_m",
    "stemDiam":   "stemDiam_m",
    # Admin / date columns — kept but renamed for clarity
    "assesDate":  "assessDate",
    "senInsDate": "sensorInstallDate",
    "senRmvDate": "sensorRemoveDate",
    "cutDwnDate": "cutDownDate",
    "germDate":   "germinationDate",
    "plantDate":  "plantingDate",
    "soilDate":   "soilSampleDate",
}


def load_tree_metadata(gis_dir: Path) -> pd.DataFrame:
    """
    Read every treeLocations.shp under gis_dir, concatenate, rename columns,
    and deduplicate on devEUI (keeping the most recently assessed record).
    Returns a plain DataFrame (geometry dropped).
    """
    shp_files = sorted(gis_dir.glob("**/VectorLayers/treeLocations.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No treeLocations.shp found under {gis_dir}")

    frames = []
    for shp in shp_files:
        gdf = gpd.read_file(shp)
        if gdf.empty:
            continue
        df = gdf.drop(columns=["geometry"]).copy()
        # city_folder = the directory two levels above VectorLayers/
        df["city_folder"] = shp.parts[-3]
        frames.append(df)

    if not frames:
        raise ValueError("All treeLocations.shp files are empty.")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        meta = pd.concat(frames, ignore_index=True)
    meta = meta.rename(columns=_COL_RENAME)

    # Deduplicate: prefer the most recently assessed entry per EUI
    date_col = "assessDate" if "assessDate" in meta.columns else None
    if date_col:
        meta[date_col] = pd.to_datetime(meta[date_col], errors="coerce")
        meta = meta.sort_values(date_col, ascending=False, na_position="last")
    meta = meta.drop_duplicates(subset="devEUI", keep="first").reset_index(drop=True)

    print(f"  Tree metadata: {len(meta)} unique EUIs "
          f"from {len(frames)} shapefile(s)")
    return meta


def build_rf_dataset(
    bsc_csv: Path = BSC_CSV,
    gis_dir: Path = GIS_DIR,
    out_csv: Path = OUT_CSV,
) -> pd.DataFrame:
    """
    Left-join bsc_events_all.csv with tree/site metadata and write rf_dataset.csv.
    Returns the joined DataFrame.
    """
    if not bsc_csv.exists():
        raise FileNotFoundError(
            f"{bsc_csv} not found.\nRun src/_poc_irrigation_v3.py first."
        )

    df_events = pd.read_csv(bsc_csv)
    print(f"  Events loaded:  {len(df_events)}  from  {df_events['eui'].nunique()} sensors")

    meta = load_tree_metadata(gis_dir)

    df_rf = df_events.merge(
        meta.rename(columns={"devEUI": "eui"}),
        on="eui",
        how="left",
        suffixes=("", "_meta"),
    )

    # Report match rate
    n_matched = int(df_rf["project"].notna().sum()) if "project" in df_rf.columns else 0
    n_unmatched = len(df_events) - n_matched
    if n_unmatched > 0:
        missing = sorted(df_rf.loc[df_rf["project"].isna(), "eui"].unique())
        print(f"  [WARN] {n_unmatched} events ({len(missing)} EUIs) "
              f"have no shapefile entry")
        if len(missing) <= 10:
            print(f"         EUIs: {missing}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_rf.to_csv(out_csv, index=False)
    print(f"  RF dataset: {len(df_rf)} rows × {len(df_rf.columns)} cols  ->  {out_csv}")

    _print_column_summary(df_rf)
    return df_rf


def _print_column_summary(df: pd.DataFrame):
    grps = {
        "Rain event": [c for c in df if c in {
            "ev_start", "ev_end", "total_mm", "duration_h", "max_intensity_mmh",
            "stac", "huff_q", "bsc", "month", "doy",
            "rain_pre24h_mm", "rain_post24h_mm",
            "rain_pre72h_mm", "rain_post72h_mm",
        }],
        "BSC bits/areas": [c for c in df if c.startswith("bsc_s") or c.startswith("area_")],
        "VWC metrics": [c for c in df if any(c.startswith(p) for p in [
            "pre_vwc", "peak_vwc", "delta_vwc", "delta_pct",
            "t_peak_min", "dry_h", "temp_sum", "vwc_p", "dyn_range",
        ])],
        "Tree biology": [c for c in df if c in {
            "genus", "species", "variety", "height",
            "crownDiam_m", "stemDiam_m", "vitality", "treeName", "project",
        }],
        "GIS / terrain": [c for c in df if any(c.startswith(p) for p in [
            "green", "buildings", "sealed", "tpi", "twi", "slope",
            "flow", "elevation", "depth", "skyView", "DTGW", "SVF",
        ])],
        "Soil chemistry": [c for c in df if any(c.startswith(p) for p in [
            "pH", "salt", "conduct", "nSol", "ammN", "nitrN", "mg", "kSol", "pHS",
        ])],
        "Soil texture": [c for c in df if c.startswith("part")],
    }
    print("\n  Column groups:")
    for grp, cols in grps.items():
        if cols:
            print(f"    {grp} ({len(cols)}): {', '.join(cols)}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Build RF-ready dataset")
    print("=" * 60)
    build_rf_dataset()
    print("\n  Done.")
