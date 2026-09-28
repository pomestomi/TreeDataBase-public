#!/usr/bin/env python3
"""
add_sensor_dates.py
===================
Adds two date attribute fields to every treeLocations.shp in the dataset:

  senInsDate  - first date on which any VWC channel read above
                VWC_MIN_THRESHOLD_PCT.  Readings at or below that value
                are treated as "sensor attached to transmitter but not yet
                buried" (hardware minimum ~1.47 %).  Always recalculated
                on every run so corrections to the threshold propagate.

  senRmvDate  - date the sensor was physically removed from this location.
                Populated from the REMOVAL_DATES table below; other entries
                are left empty.

Both fields follow ISO 8601 (YYYY-MM-DD).  The script writes the dates as
proper date fields in the DBF so that QGIS and geopandas can filter on them.

Usage:
    cd src
    python add_sensor_dates.py                      # process all cities
    python add_sensor_dates.py --city CityErlangen  # one city only
    python add_sensor_dates.py --dry-run            # print changes, no write

Requirements:
    pip install geopandas pandas
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# VWC threshold: readings at or below this value are treated as "not yet
# installed" (sensor clipped at hardware minimum ~1.47 %).
# VWC_MIN_COUNT: minimum number of above-threshold readings required before
# the sensor is considered installed.  A single spike (e.g. during bench
# testing or transit) therefore does not trigger an early insertion date.
# ---------------------------------------------------------------------------
VWC_MIN_THRESHOLD_PCT = 2.0   # readings at or below this are treated as hardware minimum
VWC_MIN_DEPTHS        = 2    # how many depth channels must exceed the threshold per reading
VWC_MAX_GAP_H         = 168  # gap longer than this (7 days) splits readings into separate runs
VWC_MIN_COUNT         = 10   # a run must have at least this many qualifying readings
VWC_MIN_RUN_DAYS      = 10   # a run must span at least this many calendar days

VWC_COLUMNS = (
    "-10|ENV__SOIL__VWC",
    "-30|ENV__SOIL__VWC",
    "-45|ENV__SOIL__VWC",
)

# ---------------------------------------------------------------------------
# Sensors removed from their original location.
# Add rows here whenever a sensor is decommissioned or relocated.
# ---------------------------------------------------------------------------
REMOVAL_DATES: dict[str, date] = {
    "0201002336000000": date(2024,  2, 13),
    "0201002336000013": date(2025, 11, 23),
    "0201002336000018": date(2023, 10, 18),
    "0201002336000021": date(2025,  8, 27),
    "8C1F640980000073": date(2025,  3, 31),
    "8C1F640980000131": date(2025,  9, 12),
    "8C1F64098000008E": date(2025, 10,  7),
    "0201002336000008": date(2024,  9, 12),
    "0201002336000007": date(2023, 11,  5),
    "0201002336000001": date(2024,  1, 21),
    "0201002512000222": date(2026,  5, 20),   # Bamberg - Hain Bootshaus C
    "8C1F6409800000E9": date(2026,  2, 20),   # Erlangen - Moenau Nelderrad
    "0201002512000292": date(2026,  6, 26),   # Garbsen - GSG Ludwigstrasse
}

SHAPEFILE_NAME = "treeLocations.shp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_shapefiles(gis_dir: Path, city_filter: str | None = None) -> list[Path]:
    """Return all treeLocations.shp paths under gis_dir."""
    results = []
    for entry in sorted(gis_dir.iterdir()):
        if not entry.is_dir():
            continue
        if city_filter and entry.name.lower() != city_filter.lower():
            continue
        for candidate in (entry / SHAPEFILE_NAME,):
            if candidate.exists():
                results.append(candidate)
                break
        else:
            for sub in sorted(entry.iterdir()):
                if sub.is_dir():
                    shp = sub / SHAPEFILE_NAME
                    if shp.exists():
                        results.append(shp)
    return results


def first_installed_date(trees_dir: Path, eui: str) -> date | None:
    """Return the first date on which the sensor is considered installed.

    Qualifying readings: >= VWC_MIN_DEPTHS depth channels above VWC_MIN_THRESHOLD_PCT.
    Qualifying readings are grouped into continuous runs: a new run starts whenever
    the gap to the previous qualifying reading exceeds VWC_MAX_GAP_H (7 days).
    The installation date is the first timestamp of the first run that contains
    at least VWC_MIN_COUNT readings.

    This correctly handles sensors that transmit a handful of readings during
    transport/bench-testing (producing a short run that fails the count or
    duration check) and then go silent for months before real installation.

    Falls back to the earliest timestamp if the CSV contains no VWC columns.
    """
    csv_path = trees_dir / eui / "sensor_data.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        vwc_cols = [c for c in VWC_COLUMNS if c in df.columns]
        if not vwc_cols:
            valid = df["datetime"].dropna()
            return valid.min().date() if not valid.empty else None

        depths_above = df[vwc_cols].gt(VWC_MIN_THRESHOLD_PCT).sum(axis=1)
        above = depths_above >= min(VWC_MIN_DEPTHS, len(vwc_cols))
        ts = df.loc[above, "datetime"].dropna().sort_values().reset_index(drop=True)

        if ts.empty:
            return None

        # Split into runs: gap > VWC_MAX_GAP_H starts a new run
        gap_h = ts.diff().dt.total_seconds().div(3600).fillna(0)
        run_id = (gap_h > VWC_MAX_GAP_H).cumsum()

        for _, run_ts in ts.groupby(run_id):
            run_duration_days = (
                (run_ts.iloc[-1] - run_ts.iloc[0]).total_seconds() / 86400
            )
            if len(run_ts) >= VWC_MIN_COUNT and run_duration_days >= VWC_MIN_RUN_DAYS:
                return run_ts.iloc[0].date()

        return None
    except Exception as exc:
        print(f"    [WARN] Could not read {csv_path.relative_to(trees_dir.parent)}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Per-shapefile processing
# ---------------------------------------------------------------------------

def process_shapefile(shp_path: Path, trees_dir: Path,
                      dry_run: bool) -> tuple[int, int, set[str]]:
    """Add/fill senInsDate and senRmvDate in *shp_path*.

    Returns (n_ins_updated, n_rmv_updated, euis_seen).
    """
    gdf = gpd.read_file(shp_path)
    n_ins = n_rmv = 0
    euis_seen: set[str] = set()

    for col in ("senInsDate", "senRmvDate"):
        if col not in gdf.columns:
            gdf[col] = pd.NaT
            print(f"    + Column '{col}' added")

    for col in ("senInsDate", "senRmvDate"):
        gdf[col] = pd.to_datetime(gdf[col], errors="coerce")

    for idx, row in gdf.iterrows():
        eui = str(row.get("devEUI", "")).strip().upper()
        if not eui or eui in ("", "NAN"):
            continue
        euis_seen.add(eui)

        # senInsDate — always recalculate using VWC threshold
        d = first_installed_date(trees_dir, eui)
        if d is not None:
            new_ts = pd.Timestamp(d)
            old_ts = gdf.at[idx, "senInsDate"]
            if pd.isna(old_ts) or old_ts != new_ts:
                if not pd.isna(old_ts):
                    print(f"    {eui}: senInsDate {old_ts.date()} -> {d}")
                gdf.at[idx, "senInsDate"] = new_ts
                n_ins += 1

        # senRmvDate — always apply the authoritative removal table
        if eui in REMOVAL_DATES:
            target = pd.Timestamp(REMOVAL_DATES[eui])
            if pd.isna(gdf.at[idx, "senRmvDate"]) or gdf.at[idx, "senRmvDate"] != target:
                gdf.at[idx, "senRmvDate"] = target
                n_rmv += 1

    print(f"    senInsDate updated : {n_ins} sensor(s)")
    print(f"    senRmvDate updated : {n_rmv} sensor(s)")

    if not dry_run:
        gdf.to_file(shp_path)
        print(f"    Saved  -> {shp_path}")
    else:
        print(f"    [DRY-RUN] Would save -> {shp_path}")

    return n_ins, n_rmv, euis_seen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add senInsDate / senRmvDate to treeLocations shapefiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--city", metavar="FOLDER",
        help="Process only the city folder with this name (e.g. CityErlangen)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute changes and print a summary without writing any files",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root  = script_dir.parent
    gis_dir    = repo_root / "GISData"
    trees_dir  = repo_root / "TreeTabularData" / "trees"

    if not gis_dir.is_dir():
        print(f"ERROR: GIS directory not found: {gis_dir}")
        sys.exit(1)
    if not trees_dir.is_dir():
        print(f"WARN: Timeseries directory not found: {trees_dir}")
        print("      senInsDate will be left empty for all sensors.")

    print("=" * 60)
    print("  SENSOR DATE ATTRIBUTE UPDATER")
    print(f"  VWC threshold : > {VWC_MIN_THRESHOLD_PCT} %  "
          f"(>={VWC_MIN_DEPTHS} depths, >={VWC_MIN_COUNT} readings, >={VWC_MIN_RUN_DAYS} day run)")
    if args.dry_run:
        print("  (DRY-RUN — no files will be modified)")
    print("=" * 60)

    shapefiles = discover_shapefiles(gis_dir, args.city)
    if not shapefiles:
        msg = f"No {SHAPEFILE_NAME} files found"
        print(f"{msg}{f' for city {args.city!r}' if args.city else ''}.")
        sys.exit(1)
    print(f"\nFound {len(shapefiles)} shapefile(s).\n")

    total_ins = total_rmv = 0
    all_euis: set[str] = set()

    for shp_path in shapefiles:
        label = "/".join(shp_path.parts[-3:])
        print(f"  {label}")
        n_ins, n_rmv, euis = process_shapefile(shp_path, trees_dir, args.dry_run)
        total_ins += n_ins
        total_rmv += n_rmv
        all_euis |= euis
        print()

    missing = {e for e in REMOVAL_DATES if e not in all_euis}
    if missing:
        print(f"[WARN] {len(missing)} EUI(s) in REMOVAL_DATES not found in any shapefile:")
        for eui in sorted(missing):
            print(f"       {eui}  ({REMOVAL_DATES[eui]})")
        print()

    print("=" * 60)
    print(f"  Total senInsDate updated : {total_ins}")
    print(f"  Total senRmvDate updated : {total_rmv}")
    if args.dry_run:
        print("  DRY-RUN complete — no files modified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
