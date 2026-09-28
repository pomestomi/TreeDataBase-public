#!/usr/bin/env python3
"""
Urban Tree Dataset Processor & Reporter
========================================
Reads QGIS shapefiles (treeLocations.shp) from city subfolders, exports
per-tree JSON files, and generates a PDF statistics report.

Expected folder structure:
    TreeDataBase/
    ├── DatasetStatistics/
    │   └── urban_tree_report.py        <-- this script
    ├── GISData/                         <-- QGIS city subfolders
    │   ├── CityErlangen/
    │   │   └── treeLocations.shp
    │   ├── CityBamberg/VectorLayers/
    │   │   └── treeLocations.shp
    │   └── ...
    └── TreeTabularData/                 <-- auto-created output
        ├── trees/{devEUI}/tree_data.json
        └── dataset_report.pdf

Usage:
    python urban_tree_report.py

Requirements:
    pip install geopandas matplotlib numpy reportlab contextily Pillow
"""

import json
import hashlib
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    PageBreak, Table, TableStyle,
)
from reportlab.lib import colors
import pandas as pd

# We use Pillow to read actual PNG dimensions for proportional PDF scaling
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHAPEFILE_NAME = "treeLocations.shp"

CROWN_LU_FIELDS = {
    "greenAttCD": "Green attached (crown)",
    "greenDetCD": "Green detached (crown)",
    "buildingCD": "Buildings (crown)",
    "sealedSuCD": "Sealed surface (crown)",
}

LAND_USE_FIELDS = {
    "greenAtta2": "Green attached 2.5 m",
    "greenAtta5": "Green attached 5 m",
    "greenAtta7": "Green attached 7.5 m",
    "greenDeta2": "Green detached 2.5 m",
    "greenDeta5": "Green detached 5 m",
    "greenDeta7": "Green detached 7.5 m",
    "buildings2": "Buildings 2.5 m",
    "buildings5": "Buildings 5 m",
    "buildings7": "Buildings 7.5 m",
    "sealedSu2": "Sealed surface 2.5 m",
    "sealedSu5": "Sealed surface 5 m",
    "sealedSu7": "Sealed surface 7.5 m",
}

TOPO_FIELDS = {
    "twi": "TWI",
    "flotAcc": "Flow Accumulation",
    "slope": "Slope",
}

TPI_FIELDS = {
    "tpi2m5": "TPI 2.5 m",
    "tpi5m": "TPI 5 m",
    "tpi7m5": "TPI 7.5 m",
}

CHEMICAL_SOIL_FIELDS = {
    "pH": "pH",
    "saltCont": "Salt content",
    "conductiv": "Conductivity",
    "nSoluble": "N soluble",
    "ammNSolubl": "NH4-N soluble",
    "nitrNSolub": "NO3-N soluble",
    "mgSoluble": "Mg soluble",
    "phSoluble": "P soluble",
    "kSoluble": "K soluble",
}

PHYSICAL_SOIL_FIELDS = {
    "part1Perc": "Particle frac. 1 [%]",
    "part2Perc": "Particle frac. 2 [%]",
    "part3Perc": "Particle frac. 3 [%]",
    "part4Perc": "Particle frac. 4 [%]",
    "part5Perc": "Particle frac. 5 [%]",
    "part6Perc": "Particle frac. 6 [%]",
    "part7Perc": "Particle frac. 7 [%]",
}

# Variety values that should be treated as "not specified"
VARIETY_IGNORE = {"nan", "none", "1", "", "n/a", "na", "unbekannt", "unknown"}

GENUS_BASE_COLORS = {
    "tilia":        "#2E86AB",
    "acer":         "#A23B72",
    "quercus":      "#F18F01",
    "castanea":     "#C73E1D",
    "fagus":        "#355070",
    "fraxinus":     "#44BBA4",
    "prunus":       "#E94F37",
    "ostrya":       "#6D597A",
    "platanus":     "#D4A373",
    "carpinus":     "#B56576",
    "ulmus":        "#8DB580",
    "populus":      "#F7B267",
    "alnus":        "#3B1F2B",
    "magnolia":     "#E56B6F",
    "liquidambar":  "#EAAC8B",
    "corylus":      "#F4845F",
    "malus":        "#F27059",
    "ginkgo":       "#F25C54",
    "parrotia":     "#393E41",
    "sophora":      "#7B2D8E",
    "zelkova":      "#1B998B",
    "celtis":       "#5C677D",
    "sorbus":       "#FF6B6B",
    "chitalpa":     "#4ECDC4",
}

COLORS_PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
    "#44BBA4", "#E94F37", "#393E41", "#D4A373", "#6D597A",
    "#B56576", "#355070", "#EAAC8B", "#E56B6F", "#8DB580",
    "#F7B267", "#F79D65", "#F4845F", "#F27059", "#F25C54",
    "#7B2D8E", "#1B998B", "#5C677D", "#FF6B6B", "#4ECDC4",
    "#A5668B", "#0F4C5C", "#FB6107", "#36C9C6", "#C1666B",
]

# Maximum usable width inside the A4 page (with 18 mm margins each side)
PDF_MAX_WIDTH = A4[0] - 2 * 18 * mm       # ≈ 159 mm
PDF_MAX_HEIGHT = A4[1] - (20 + 15) * mm - 55 * mm   # conservative: leaves room for caption + frame margins

# BSC (Binary Shape Code) colour palette — shared with _poc_irrigation_v3.py
_BSC_PALETTE = {
    "0000": "#a6cee3", "0001": "#1f78b4", "0010": "#b2df8a", "0011": "#33a02c",
    "0100": "#fb9a99", "0101": "#e31a1c", "0110": "#fdbf6f", "0111": "#ff7f00",
    "1000": "#cab2d6", "1001": "#6a3d9a", "1010": "#ffff99", "1011": "#b15928",
    "1100": "#8dd3c7", "1101": "#ffffb3", "1110": "#bebada", "1111": "#fb8072",
}
_BSC_LITERATURE_MM   = 12.7   # Wischmeier & Smith (1978) erosivity threshold
_BSC_MIN_EVENT_MM    = 5.0    # lower detection threshold used in this project
_BSC_DRY_GAP_H       = 6
_BSC_IMPACT_DELTA    = 20.0   # % of dynamic range — minimum VWC rise to count as "impactful"
_BSC_IMPACT_PEAK_H   = 6.0    # hours after event end within which the VWC peak must occur
_BSC_VWC_THRESHOLDS  = [5.0, 10.0, 20.0, 30.0]   # % dyn. range thresholds shown in section 13e

# ── City lookup keyed by shapefile 'project' attribute value ──────────────────
_SHAPEFILE_PROJECT_TO_CITY = {
    "Botanical Garden Bremen":          "Bremen",
    "Botanical Garden Vienna":          "Vienna",
    "City Aachen":                      "Aachen",
    "City Bamberg":                     "Bamberg",
    "City Berlin Charlottenburg":       "Berlin",
    "City Berlin Frh XBerg":            "Berlin",
    "City Berlin Neukölln":             "Berlin",
    "City Biberach":                    "Biberach",
    "City Celle":                       "Celle",
    "City Crailsheim":                  "Crailsheim",
    "City Erlangen":                    "Erlangen",
    "City Garbsen":                     "Garbsen",
    "City Grünwald":                    "Grünwald",
    "City Hagen":                       "Hagen",
    "City Hanover":                     "Hannover",
    "City Hassfurt":                    "Hassfurt",
    "City Hildesheim":                  "Hildesheim",
    "City Ingolstadt":                  "Ingolstadt",
    "City Kassel":                      "Kassel",
    "City Leipzig":                     "Leipzig",
    "City Lemgo":                       "Lemgo",
    "City Luedenscheid":                "Lüdenscheid",
    "City Neunkirchen":                 "Neunkirchen",
    "City Nuernberg":                   "Nürnberg",
    "City Pforzheim":                   "Pforzheim",
    "City Pirmasens":                   "Pirmasens",
    "City Potsdam":                     "Potsdam",
    "City Stein":                       "Nürnberg",
    "City Vienna":                      "Vienna",
    "City Weisendorf":                  "Weisendorf",
    "Company Bremen":                   "Bremen",
    "Company Effeltrich":               "Effeltrich",
    "Company Erfurt":                   "Erfurt",
    "Company Erlangen":                 "Erlangen",
    "Company Heidelberg":               "Heidelberg",
    "Company Homburg":                  "Homburg",
    "Company Laax":                     "Laax",
    "Company Nörvenich":                "Aachen",
    "Company Pillnitz":                 "Pillnitz",
    "Company Plön":                     "Plön",
    "Company Saarbrücken":              "Saarbrücken",
    "Company Vienna":                   "Vienna",
    "Schlösserland Sachsen":            "Dresden",
    "University Erlangen Green Office": "Erlangen",
    "University Salzburg":              "Salzburg",
}

# ── Coord-based override for sensors spanning multiple cities in one project ──
_COORD_CITY_MAP = {
    "11.40,48.78": "Ingolstadt",
    "11.36,49.04": "Greding",
    "10.84,48.38": "Augsburg",
    "16.26,47.83": "Wiener Neustadt",
    "13.87,51.01": "Pillnitz",
}

_CITY_STATE = {
    "Aachen":          ("Nordrhein-Westfalen",  "Germany"),
    "Augsburg":        ("Bayern",               "Germany"),
    "Bamberg":         ("Bayern",               "Germany"),
    "Berlin":          ("Berlin",               "Germany"),
    "Biberach":        ("Baden-Württemberg",    "Germany"),
    "Bremen":          ("Bremen",               "Germany"),
    "Celle":           ("Niedersachsen",        "Germany"),
    "Crailsheim":      ("Baden-Württemberg",    "Germany"),
    "Dresden":         ("Sachsen",              "Germany"),
    "Effeltrich":      ("Bayern",               "Germany"),
    "Erlangen":        ("Bayern",               "Germany"),
    "Erfurt":          ("Thüringen",            "Germany"),
    "Garbsen":         ("Niedersachsen",        "Germany"),
    "Greding":         ("Bayern",               "Germany"),
    "Grünwald":        ("Bayern",               "Germany"),
    "Hagen":           ("Nordrhein-Westfalen",  "Germany"),
    "Hannover":        ("Niedersachsen",        "Germany"),
    "Hassfurt":        ("Bayern",               "Germany"),
    "Heidelberg":      ("Baden-Württemberg",    "Germany"),
    "Hildesheim":      ("Niedersachsen",        "Germany"),
    "Homburg":         ("Saarland",             "Germany"),
    "Ingolstadt":      ("Bayern",               "Germany"),
    "Kassel":          ("Hessen",               "Germany"),
    "Leipzig":         ("Sachsen",              "Germany"),
    "Lemgo":           ("Nordrhein-Westfalen",  "Germany"),
    "Lüdenscheid":     ("Nordrhein-Westfalen",  "Germany"),
    "Neunkirchen":     ("Saarland",             "Germany"),
    "Nürnberg":        ("Bayern",               "Germany"),
    "Pforzheim":       ("Baden-Württemberg",    "Germany"),
    "Pillnitz":        ("Sachsen",              "Germany"),
    "Pirmasens":       ("Rheinland-Pfalz",      "Germany"),
    "Plön":            ("Schleswig-Holstein",   "Germany"),
    "Potsdam":         ("Brandenburg",          "Germany"),
    "Saarbrücken":     ("Saarland",             "Germany"),
    "Weisendorf":      ("Bayern",               "Germany"),
    "Vienna":          ("Wien",                 "Austria"),
    "Wiener Neustadt": ("Niederösterreich",     "Austria"),
    "Salzburg":        ("Salzburg",             "Austria"),
    "Laax":            ("Graubünden",           "Switzerland"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def json_serializer(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def row_to_dict(row):
    d = {}
    for col, val in row.items():
        if col == "geometry":
            geom = val
            if geom is not None:
                coords = (list(geom.geoms)[0].coords[0]
                          if geom.geom_type == "MultiPoint" else geom.coords[0])
                d["geometry_x"] = coords[0]
                d["geometry_y"] = coords[1]
            continue
        if val is None:
            d[col] = None
        elif isinstance(val, float) and math.isnan(val):
            d[col] = None
        elif hasattr(val, "isoformat"):
            d[col] = val.isoformat() if not (hasattr(val, "year") and val != val) else None
        else:
            d[col] = val
    return d


def content_hash(data: dict) -> str:
    raw = json.dumps(data, sort_keys=True, default=json_serializer)
    return hashlib.md5(raw.encode()).hexdigest()


def compute_age_years(plant_date, ref_date=None):
    if plant_date is None:
        return None
    ref = ref_date or datetime.now()
    try:
        if isinstance(plant_date, str):
            plant_date = plant_date.strip()
            if not plant_date or plant_date.lower() in ("nat", "none", "null", "nan"):
                return None
            # Try common formats found in the dataset
            for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S",
                        "%Y/%m/%d", "%Y-%m-%d", "%d.%m.%Y"):
                try:
                    plant_date = datetime.strptime(plant_date, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
        # Handle pandas NaT or float NaN
        if hasattr(plant_date, "year") and plant_date != plant_date:
            return None
        return (ref - plant_date).days / 365.25
    except Exception:
        return None


def _shade_color(hex_color, factor):
    r, g, b = mcolors.to_rgb(hex_color)
    if factor > 1:
        r = r + (1 - r) * (factor - 1)
        g = g + (1 - g) * (factor - 1)
        b = b + (1 - b) * (factor - 1)
    else:
        r, g, b = r * factor, g * factor, b * factor
    return mcolors.to_hex((min(r, 1), min(g, 1), min(b, 1)))


def _proportional_image(chart_path, max_w=PDF_MAX_WIDTH, max_h=PDF_MAX_HEIGHT):
    """Return a ReportLab Image scaled proportionally to fit within max_w × max_h."""
    img = PILImage.open(str(chart_path))
    pw, ph = img.size
    ratio = pw / ph
    # Try fitting to width first
    w = max_w
    h = w / ratio
    if h > max_h:
        h = max_h
        w = h * ratio
    return RLImage(str(chart_path), width=w, height=h)


def _export_sensor_csv(gdf, output_path: Path) -> None:
    """Write CSV with EUI, shapefile project, state/Bundesland, and map city for each sensor."""
    gdf_wgs = gdf.copy()
    if gdf_wgs.crs and gdf_wgs.crs.to_epsg() != 4326:
        gdf_wgs = gdf_wgs.to_crs(epsg=4326)

    rows = []
    for _, row in gdf_wgs.iterrows():
        _raw_eui = row.get("devEUI")
        eui = "" if (_raw_eui is None or (isinstance(_raw_eui, float) and math.isnan(_raw_eui))) \
                 else str(_raw_eui).strip()
        _raw_proj = row.get("project")
        project = "" if (_raw_proj is None or (isinstance(_raw_proj, float) and math.isnan(_raw_proj))) \
                     else str(_raw_proj).strip()

        geom = row.geometry
        if geom is not None:
            pt = list(geom.geoms)[0] if geom.geom_type == "MultiPoint" else geom
            lon, lat = pt.x, pt.y
            coord_key = f"{lon:.2f},{lat:.2f}"
        else:
            lon, lat, coord_key = None, None, ""

        # Coord override takes priority (handles multi-city projects like Company Pappenheim)
        city = _COORD_CITY_MAP.get(coord_key)
        # Shapefile project attribute lookup
        if city is None:
            city = _SHAPEFILE_PROJECT_TO_CITY.get(project)
        # Fallback: strip standard prefix words from a non-empty project value
        if city is None and project:
            for pfx in ("Botanical Garden ", "City ", "Company ",
                        "University ", "Castle ", "Schloss "):
                if project.startswith(pfx):
                    city = project[len(pfx):].strip()
                    break

        state, country = _CITY_STATE.get(city or "", ("", ""))
        rows.append({
            "eui":     eui,
            "project": project,
            "state":   state,
            "country": country,
            "city":    city if city else float("nan"),
        })

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    print(f"  Sensor location CSV -> {output_path}")


# ---------------------------------------------------------------------------
# Data discovery & JSON export
# ---------------------------------------------------------------------------

def discover_city_shapefiles(project_dir: Path):
    results = []
    for entry in sorted(project_dir.iterdir()):
        if entry.is_dir():
            shp = entry / SHAPEFILE_NAME
            if shp.exists():
                results.append((entry.name, shp))
                print(f"  [FOUND] {entry.name}: {shp}")
            else:
                for sub in sorted(entry.iterdir()):
                    if sub.is_dir():
                        shp2 = sub / SHAPEFILE_NAME
                        if shp2.exists():
                            results.append((entry.name, shp2))
                            print(f"  [FOUND] {entry.name}: {shp2}")
    return results


def load_all_trees(city_shapefiles):
    frames = []
    for city, shp_path in city_shapefiles:
        gdf = gpd.read_file(shp_path)
        gdf["_source_city_folder"] = city
        frames.append(gdf)
    if not frames:
        print("ERROR: No tree data found.")
        sys.exit(1)
    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
    print(f"\n  Total trees loaded: {len(combined)}")
    return combined


def export_tree_jsons(gdf, output_dir):
    """Export one tree_data.json per devEUI as a list of attribute snapshots.

    On every run the current shapefile attributes are compared (via content
    hash) against the most recent snapshot.  If anything has changed a new
    snapshot is appended, timestamped with the current script execution time.
    The full history of all detected changes is therefore preserved in order.
    Legacy single-dict JSON files are migrated to the list format on first
    write.
    """
    trees_dir = output_dir / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)
    stats = {"created": 0, "versioned": 0, "skipped": 0, "errors": 0}
    run_ts = datetime.now().isoformat()

    eui_counts = gdf["devEUI"].value_counts()
    dupes = eui_counts[eui_counts > 1]
    if len(dupes) > 0:
        print("  [WARN] Duplicate devEUI(s) – last occurrence wins:")
        for eui, cnt in dupes.items():
            print(f"         {eui} appears {cnt} times")

    for idx, row in gdf.iterrows():
        dev_eui = row.get("devEUI")
        if not dev_eui or (isinstance(dev_eui, float) and math.isnan(dev_eui)):
            stats["errors"] += 1
            continue
        dev_eui = str(dev_eui).strip()
        tree_dir = trees_dir / dev_eui
        json_path = tree_dir / "tree_data.json"
        data = row_to_dict(row)
        new_hash = content_hash(data)
        data["_content_hash"] = new_hash
        data["_snapshot_ts"] = run_ts

        if json_path.exists():
            snapshots = []
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                # Migrate legacy single-dict format to list
                snapshots = existing if isinstance(existing, list) else [existing]
            except Exception:
                pass

            if snapshots and snapshots[-1].get("_content_hash", "") == new_hash:
                stats["skipped"] += 1
                continue

            # Any field change → append a new timestamped snapshot
            snapshots.append(data)
            stats["versioned"] += 1
        else:
            tree_dir.mkdir(parents=True, exist_ok=True)
            snapshots = [data]
            stats["created"] += 1

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(snapshots, f, indent=2, default=json_serializer,
                      ensure_ascii=False)

    print(f"\n  JSON export: {stats['created']} created, "
          f"{stats['versioned']} updated (new snapshot), "
          f"{stats['skipped']} unchanged, {stats['errors']} errors")
    return stats


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _make_cluster_map(gdf, filepath, min_cluster_radius_km=8.0):
    """Sensor locations on a basemap with adaptive clustering.

    Two-pass approach:
      1. Grid-based initial clustering (fast, coarse).
      2. Iterative merge of closest cluster-pair until no rendered dots
         would overlap in the final figure.
    """
    from shapely.geometry import Point

    gdf_wgs = gdf.copy()
    if gdf_wgs.crs and gdf_wgs.crs.to_epsg() != 4326:
        gdf_wgs = gdf_wgs.to_crs(epsg=4326)

    lons, lats = [], []
    for geom in gdf_wgs.geometry:
        if geom is None:
            continue
        pt = list(geom.geoms)[0] if geom.geom_type == "MultiPoint" else geom
        lons.append(pt.x)
        lats.append(pt.y)
    lons, lats = np.array(lons), np.array(lats)

    # --- Pass 1: grid-based initial clustering ---
    lat_span_km = (lats.max() - lats.min()) * 111.0 if len(lats) > 1 else 20
    lon_span_km = ((lons.max() - lons.min()) * 111.0
                   * np.cos(np.radians(lats.mean()))) if len(lons) > 1 else 20
    extent_km = max(lat_span_km, lon_span_km, 20)
    cluster_km = max(min_cluster_radius_km, extent_km * 0.04)

    deg_r = cluster_km / 111.0
    grid_lats = np.round(lats / deg_r) * deg_r
    grid_lons = np.round(lons / deg_r) * deg_r
    grid_clusters = defaultdict(list)
    for i in range(len(lats)):
        grid_clusters[(grid_lats[i], grid_lons[i])].append(i)

    # Build mutable cluster list: [(mean_lat, mean_lon, count), ...]
    cl = []
    for idxs in grid_clusters.values():
        cl.append([np.mean(lats[idxs]), np.mean(lons[idxs]), len(idxs)])

    # --- Pass 2: iterative merge of closest pair until no overlap ---
    # "Overlap" means two dot centres are closer than a fraction of the
    # map extent.  At ~5 % of extent, two large bubbles won't touch.
    merge_threshold_deg = extent_km * 0.05 / 111.0

    def _closest_pair(clusters):
        """Return (i, j, dist) of the closest pair, or None."""
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = math.hypot(clusters[i][0] - clusters[j][0],
                               clusters[i][1] - clusters[j][1])
                if best is None or d < best[2]:
                    best = (i, j, d)
        return best

    while len(cl) > 1:
        pair = _closest_pair(cl)
        if pair is None or pair[2] > merge_threshold_deg:
            break
        i, j, _ = pair
        a, b = cl[i], cl[j]
        total = a[2] + b[2]
        merged = [(a[0]*a[2] + b[0]*b[2]) / total,
                  (a[1]*a[2] + b[1]*b[2]) / total,
                  total]
        cl[i] = merged
        cl.pop(j)

    clat = np.array([c[0] for c in cl])
    clon = np.array([c[1] for c in cl])
    ccount = np.array([c[2] for c in cl])

    cluster_gdf = gpd.GeoDataFrame(
        {"count": ccount},
        geometry=[Point(x, y) for x, y in zip(clon, clat)],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    # Bubble sizes – sqrt-scaled so area is proportional to count
    min_sz, max_sz = 150, 1400
    if ccount.max() > ccount.min():
        normed = (np.sqrt(ccount) - np.sqrt(ccount.min())) / (
            np.sqrt(ccount.max()) - np.sqrt(ccount.min()))
        sizes = min_sz + normed * (max_sz - min_sz)
    else:
        sizes = np.full_like(ccount, 300, dtype=float)

    # Compute extent with padding; force landscape if portrait data
    xmin, ymin, xmax, ymax = cluster_gdf.total_bounds
    dx = (xmax - xmin) or 10000
    dy = (ymax - ymin) or 10000
    if dy > dx:
        extra = (dy - dx) / 2
        xmin -= extra
        xmax += extra
        dx = xmax - xmin

    pad_x = dx * 0.15
    pad_y = dy * 0.15

    fig_w = 14
    padded_ratio = (dy + 2*pad_y) / (dx + 2*pad_x)
    fig_h = max(6, min(fig_w * padded_ratio, 11))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    # Basemap tiles
    try:
        import contextily as ctx
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                        zoom="auto", attribution_size=8)
    except Exception as e:
        print(f"  [WARN] Basemap tiles failed ({e}), plotting without.")

    # Plot dots on top
    cluster_gdf.plot(
        ax=ax, markersize=sizes, color="#2E86AB", alpha=0.70,
        edgecolor="white", linewidth=1.2, zorder=5,
    )

    # Count labels
    for i, row in cluster_gdf.iterrows():
        cnt = int(row["count"])
        fs = 9 if cnt < 100 else 10
        ax.annotate(
            str(cnt), xy=(row.geometry.x, row.geometry.y),
            ha="center", va="center", fontsize=fs,
            fontweight="bold", color="white", zorder=6,
        )

    ax.set_axis_off()
    ax.set_aspect("auto")
    ax.set_title(
        f"Sensor Locations  ({len(lats)} sensors, {len(cl)} clusters)",
        fontsize=14, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_project_pie(labels, sizes, title, filepath,
                      status_per_project=None, totals=None):
    """Pie chart coloured by online/offline/removed ratio per project.

    status_per_project : {project: {STATUS_ONLINE: n, STATUS_OFFLINE: n,
                                     STATUS_REMOVED: n}}
    totals             : {STATUS_ONLINE: n, STATUS_OFFLINE: n,
                          STATUS_REMOVED: n}
    Slice colour logic (applied to active – i.e. non-removed – sensors):
      all removed       → grey #888888
      none online       → heavily desaturated base colour
      some online       → partially desaturated, proportional to online fraction
      all online        → full base colour
    """
    from matplotlib.patches import Patch

    pairs = sorted(zip(sizes, labels), reverse=True)
    sizes  = [s for s, _ in pairs]
    labels = [lb for _, lb in pairs]
    n = len(labels)
    base_col = (COLORS_PALETTE * (n // len(COLORS_PALETTE) + 1))[:n]

    use_colors = []
    for i, lbl in enumerate(labels):
        if status_per_project is not None:
            sp     = status_per_project.get(lbl, {})
            n_on   = sp.get(STATUS_ONLINE,  0)
            n_off  = sp.get(STATUS_OFFLINE, 0)
            n_rmv  = sp.get(STATUS_REMOVED, 0)
            active = n_on + n_off
            if active == 0:
                use_colors.append("#888888")          # all removed
            elif n_on == 0:
                use_colors.append(_shade_color(base_col[i], 0.50))  # none online
            else:
                frac = n_on / active
                use_colors.append(_shade_color(base_col[i], 0.60 + 0.40 * frac))
        else:
            use_colors.append(base_col[i])

    fig, ax = plt.subplots(figsize=(9, 6.5))
    wedges, _, autotexts = ax.pie(
        sizes, labels=None,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        colors=use_colors, startangle=140, pctdistance=0.8,
    )
    for at in autotexts:
        at.set_fontsize(8)

    legend_labels = []
    for lbl, cnt in zip(labels, sizes):
        if status_per_project is not None:
            sp    = status_per_project.get(lbl, {})
            n_on  = sp.get(STATUS_ONLINE,  0)
            n_off = sp.get(STATUS_OFFLINE, 0)
            n_rmv = sp.get(STATUS_REMOVED, 0)
            legend_labels.append(
                f"{lbl}  ({cnt}: {n_on} \u25cf on, {n_off} \u25cf off, "
                f"{n_rmv} \u25a0 rmv)")
        else:
            legend_labels.append(f"{lbl}  ({cnt})")

    if status_per_project is not None and totals is not None:
        t_on  = totals.get(STATUS_ONLINE,  0)
        t_off = totals.get(STATUS_OFFLINE, 0)
        t_rmv = totals.get(STATUS_REMOVED, 0)
        summary_patch = Patch(
            facecolor="none", edgecolor="none",
            label=(f"Total:  {t_on} online  |  {t_off} offline  |  "
                   f"{t_rmv} removed"))
        ax.legend(wedges + [summary_patch],
                  legend_labels + [summary_patch.get_label()],
                  loc="center left", bbox_to_anchor=(1, 0.5),
                  fontsize=7.5, framealpha=0.9)
    else:
        ax.legend(wedges, legend_labels, loc="center left",
                  bbox_to_anchor=(1, 0.5), fontsize=7.5, framealpha=0.9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Misspelled taxon names as they occur in the source shapefiles.
GENUS_TYPO_FIX = {
    "acre": "acer",
    "carpinius": "carpinus",
    "parottia": "parrotia",
    "eleagnus": "elaeagnus",
    "pauwlonia tomentosa": "paulownia",
}
SPECIES_TYPO_FIX = {
    "satvia": "sativa",
    "termula": "tremula",
    "platyhpyllos": "platyphyllos",
    "platyphyllus": "platyphyllos",
    "saccherum": "saccharum",
    "robus": "robur",
    "pennsylvania": "pennsylvanica",
    "edulnis": "edulis",
}
# Non-taxonomic placeholders in the species column; dropped from the label.
SPECIES_DROP = {"", "nan", "5"}


def _build_species_label(genus, species, variety):
    """Build a clean species label, ignoring meaningless variety values."""
    genus = str(genus).strip().lower()
    species = str(species).strip().lower()
    variety = str(variety).strip().lower()

    if genus in ("", "nan"):
        genus = "unknown"
    genus = GENUS_TYPO_FIX.get(genus, genus)
    species = SPECIES_TYPO_FIX.get(species, species)
    # Botanical convention for a tree identified only to genus.
    if species == "spec":
        species = "sp."

    parts = [genus.capitalize()]
    if species not in SPECIES_DROP:
        parts.append(species)
    if variety not in VARIETY_IGNORE:
        parts.append(f"'{variety}'")
    return genus, " ".join(parts)


def _make_species_pie(gdf, filepath):
    """Pie chart: every species shown, grouped by genus colour family,
    sorted largest-genus-first, absolute counts in legend.
    Varieties that are 'none'/'1'/NaN are merged with the base species."""
    records = []
    for _, row in gdf.iterrows():
        genus, label = _build_species_label(
            row.get("genus", ""), row.get("species", ""), row.get("variety", ""))
        records.append({"genus": genus, "label": label})

    df = pd.DataFrame(records)
    species_counts = df["label"].value_counts()
    genus_of = df.drop_duplicates("label").set_index("label")["genus"]
    genus_totals = df["genus"].value_counts()

    # Order: largest genus first, within genus largest species first
    ordered_labels, ordered_sizes = [], []
    for g in genus_totals.index:
        members = [l for l in species_counts.index if genus_of.get(l) == g]
        members.sort(key=lambda l: species_counts[l], reverse=True)
        for m in members:
            ordered_labels.append(m)
            ordered_sizes.append(species_counts[m])

    # Colour shading within each genus
    genus_member_n = defaultdict(int)
    for l in ordered_labels:
        genus_member_n[genus_of.get(l, "unknown")] += 1
    genus_idx = defaultdict(int)
    color_list = []
    for l in ordered_labels:
        g = genus_of.get(l, "unknown")
        base = GENUS_BASE_COLORS.get(
            g, COLORS_PALETTE[hash(g) % len(COLORS_PALETTE)])
        n_m = genus_member_n[g]
        i = genus_idx[g]
        genus_idx[g] += 1
        if n_m == 1:
            color_list.append(base)
        else:
            factor = 0.65 + 0.7 * (i / max(n_m - 1, 1))
            color_list.append(_shade_color(base, factor))

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, _, autotexts = ax.pie(
        ordered_sizes, labels=None,
        autopct=lambda p: f"{p:.1f}%" if p > 2.5 else "",
        colors=color_list, startangle=140, pctdistance=0.82,
    )
    for at in autotexts:
        at.set_fontsize(6.5)

    legend_labels = [f"{l}  ({c})" for l, c in
                     zip(ordered_labels, ordered_sizes)]
    ncol = 2 if len(legend_labels) > 25 else 1
    ax.legend(wedges, legend_labels, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=6.5,
              framealpha=0.9, ncol=ncol)
    # Two-line title to avoid collision with legend
    ax.set_title("Species Distribution\n(grouped by genus family)",
                 fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_histogram(data_series, title, xlabel, filepath, bins=20,
                    clip_percentile=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    clean = pd.to_numeric(data_series, errors="coerce").dropna()
    if len(clean) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
    else:
        if clip_percentile is not None and len(clean) > 5:
            upper = np.percentile(clean, clip_percentile)
            n_clip = (clean > upper).sum()
            clean = clean[clean <= upper]
            if n_clip > 0:
                ax.annotate(f"{n_clip} outlier(s) > {upper:.3f} clipped",
                            xy=(0.98, 0.95), xycoords="axes fraction",
                            ha="right", va="top", fontsize=8, color="grey")
        ax.hist(clean, bins=bins, color=COLORS_PALETTE[0],
                edgecolor="white", alpha=0.85)
        mean_v = clean.mean()
        med_v = clean.median()
        ax.axvline(mean_v, color=COLORS_PALETTE[1], ls="--", lw=1.5,
                   label=f"Mean: {mean_v:.3f}")
        ax.axvline(med_v, color=COLORS_PALETTE[2], ls="-.", lw=1.5,
                   label=f"Median: {med_v:.3f}")
        ax.legend(fontsize=9)
        ax.annotate(f"n = {len(clean):,}", xy=(0.02, 0.95),
                    xycoords="axes fraction", ha="left", va="top",
                    fontsize=9, color="grey")
        ax.set_ylabel("Count")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_boxplot(data_dict, title, ylabel, filepath):
    fig, ax = plt.subplots(figsize=(10, 5))
    filtered = {}
    for k, v in data_dict.items():
        numeric = pd.to_numeric(v, errors="coerce").dropna()
        if len(numeric) > 0:
            filtered[k] = numeric
    if not filtered:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
    else:
        bp = ax.boxplot(filtered.values(), patch_artist=True,
                        tick_labels=list(filtered.keys()),
                        medianprops=dict(color="black", linewidth=1.5))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS_PALETTE[i % len(COLORS_PALETTE)])
            patch.set_alpha(0.7)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_coverage_bar(field_counts, total, title, filepath):
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(field_counts.keys())
    counts = list(field_counts.values())
    bars = ax.bar(names, counts, color=COLORS_PALETTE[:len(names)],
                  edgecolor="white")
    ax.axhline(total, color="grey", ls="--", lw=1, alpha=0.5,
               label=f"Total: {total}")
    for bar, count in zip(bars, counts):
        pct = count / total * 100 if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{count}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Number of trees")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_offline_project_map(project_name, offline_gdf, ts_metrics,
                               telemetry_cache, filepath):
    """Map of offline sensors for one project with a detail table below.

    offline_gdf : GeoDataFrame (WGS-84) of offline trees for this project.
    telemetry_cache : {eui_upper: {battery_pct, battery_mv, rssi}} dict.
    """
    # Collect per-sensor display info
    sensors = []
    gdf_wgs = offline_gdf.copy()
    if gdf_wgs.crs and gdf_wgs.crs.to_epsg() != 4326:
        gdf_wgs = gdf_wgs.to_crs(epsg=4326)

    for _, row in gdf_wgs.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        pt = list(geom.geoms)[0] if geom.geom_type == "MultiPoint" else geom
        eui       = str(row.get("devEUI", "")).strip().upper()
        tree_name = str(row.get("treeName", "")).strip() or "—"
        tm        = ts_metrics.get(eui, {})
        last_date = tm.get("last_date")
        last_str  = last_date.strftime("%Y-%m-%d") if last_date else "never"
        tel       = telemetry_cache.get(eui, {})
        bat       = tel.get("battery_pct")
        bat_str   = f"{bat:.0f} %" if bat is not None else "—"
        rssi      = tel.get("rssi")
        rssi_str  = f"{rssi:.0f} dBm" if rssi is not None else "—"
        sensors.append({
            "eui": eui, "tree_name": tree_name,
            "last_online": last_str, "battery": bat_str, "rssi": rssi_str,
            "lon": pt.x, "lat": pt.y,
        })

    if not sensors:
        return

    n_sensors = len(sensors)

    # Build Web-Mercator GDF for plotting
    sensor_gdf_wm = gpd.GeoDataFrame(
        sensors,
        geometry=gpd.GeoSeries(
            [gpd.points_from_xy([s["lon"]], [s["lat"]])[0] for s in sensors],
            crs="EPSG:4326",
        ).to_crs(epsg=3857),
        crs="EPSG:3857",
    )

    # Figure: map top, table bottom
    table_row_h  = 0.26   # inches per table row
    table_h      = max(1.2, (n_sensors + 1) * table_row_h + 0.4)
    map_h        = 6.5
    fig_w        = 13.0
    fig, gs_fig  = plt.subplots(2, 1,
                                figsize=(fig_w, map_h + table_h),
                                gridspec_kw={"height_ratios": [map_h, table_h],
                                             "hspace": 0.03})
    ax_map, ax_tbl = gs_fig

    # Map extent with padding
    xs = np.array([r.geometry.x for _, r in sensor_gdf_wm.iterrows()])
    ys = np.array([r.geometry.y for _, r in sensor_gdf_wm.iterrows()])
    dx = max(xs.max() - xs.min(), 400)
    dy = max(ys.max() - ys.min(), 400)
    pad_x = max(dx * 0.35, 250)
    pad_y = max(dy * 0.35, 250)
    ax_map.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
    ax_map.set_ylim(ys.min() - pad_y, ys.max() + pad_y)

    try:
        import contextily as ctx
        ctx.add_basemap(ax_map, source=ctx.providers.CartoDB.Positron,
                        zoom="auto", attribution_size=7)
    except Exception as exc:
        print(f"  [WARN] Basemap failed for {project_name}: {exc}")

    # Red markers
    sensor_gdf_wm.plot(ax=ax_map, markersize=90, color="#D63030",
                       edgecolor="white", linewidth=1.0, zorder=5, alpha=0.9)
    # Numbered white labels
    for i, (_, srow) in enumerate(sensor_gdf_wm.iterrows(), start=1):
        ax_map.annotate(
            str(i),
            xy=(srow.geometry.x, srow.geometry.y),
            ha="center", va="center", fontsize=6.5,
            fontweight="bold", color="white", zorder=6,
        )

    ax_map.set_axis_off()
    ax_map.set_title(
        f"Offline Sensors – {project_name}  ({n_sensors} offline)",
        fontsize=13, fontweight="bold", pad=10,
    )

    # Detail table
    ax_tbl.axis("off")
    col_labels = ["#", "Serial (devEUI)", "Tree Name", "Last Online", "Battery", "RSSI"]
    cell_text  = [
        [str(i+1), s["eui"], s["tree_name"][:32], s["last_online"],
         s["battery"], s["rssi"]]
        for i, s in enumerate(sensors)
    ]
    tbl = ax_tbl.table(
        cellText=cell_text, colLabels=col_labels,
        cellLoc="left", loc="upper center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    # Header style
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#D63030")
        cell.set_text_props(color="white", fontweight="bold")
    # Alternating row colours
    for i in range(1, n_sensors + 1):
        bg = "#fff0f0" if i % 2 == 1 else "#ffffff"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def _bsc_impact_mask(df: pd.DataFrame,
                     delta_pct: float = _BSC_IMPACT_DELTA,
                     peak_h:    float = _BSC_IMPACT_PEAK_H) -> "pd.Series":
    """
    Boolean mask: keep events where any VWC depth rose >= delta_pct % of its
    dynamic range, with the peak occurring within peak_h hours of event end.
    """
    mask = pd.Series(False, index=df.index)
    for depth in ["-10", "-30", "-45"]:
        dcol = f"delta_pct_dyn{depth}"
        pcol = f"t_peak_min{depth}"
        if dcol not in df.columns or pcol not in df.columns:
            continue
        max_peak_min = df["duration_h"] * 60 + peak_h * 60
        mask = mask | (
            df[dcol].notna() &
            (df[dcol] >= delta_pct) &
            df[pcol].notna() &
            (df[pcol] <= max_peak_min)
        )
    return mask


def _make_bsc_quartile_distribution(df_all: pd.DataFrame, out_path: Path,
                                    title_suffix: str = ""):
    """Stacked bar chart: % of events per BSC code per Huff quartile."""
    if df_all.empty or "bsc" not in df_all.columns or "huff_q" not in df_all.columns:
        return
    df_all = df_all.copy()
    df_all["bsc"] = df_all["bsc"].astype(str).str.zfill(4)

    total   = len(df_all)
    codes   = df_all["bsc"].value_counts().index.tolist()
    q_sizes = [df_all["huff_q"].eq(q).sum() for q in range(1, 5)]
    q_labels = [f"Quartile {r} ({q_sizes[r-1]/total*100:.0f} %)" for r in range(1, 5)]

    matrix = np.zeros((len(codes), 4))
    for i, code in enumerate(codes):
        for q in range(1, 5):
            m = (df_all["bsc"] == code) & (df_all["huff_q"] == q)
            matrix[i, q - 1] = m.sum() / total * 100

    fig, ax = plt.subplots(figsize=(11, 5))
    x      = np.arange(4)
    bottom = np.zeros(4)
    for i, code in enumerate(codes):
        color = _BSC_PALETTE.get(code, "#dddddd")
        vals  = matrix[i]
        bars  = ax.bar(x, vals, bottom=bottom, color=color,
                       label=code, edgecolor="white", linewidth=0.4, zorder=2)
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if v >= 1.5:
                ax.text(bar.get_x() + bar.get_width() / 2, bottom[j] + v / 2,
                        f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 3 else "#222222")
        bottom += vals
    for j in range(4):
        ax.text(x[j], bottom[j] + 0.4, f"{bottom[j]:.1f} %",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, fontsize=10)
    ax.set_ylabel("Percentage of events [%]", fontsize=10)
    ax.set_title(
        f"BSC Distribution by Huff Quartile  (n = {total}){title_suffix}",
        fontsize=11,
    )
    ax.legend(title="BSC", fontsize=8, title_fontsize=8,
              ncol=min(len(codes), 8),
              bbox_to_anchor=(0.5, -0.18), loc="upper center")
    ax.set_ylim(0, bottom.max() * 1.12)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart -> {out_path}")


def _make_bsc_scatter(df_all: pd.DataFrame, out_path: Path):
    """Scatter plots coloured by BSC code: event total_mm vs duration_h and max intensity."""
    if df_all.empty or "bsc" not in df_all.columns or "duration_h" not in df_all.columns:
        return
    df_all = df_all.copy()
    df_all["bsc"] = df_all["bsc"].astype(str).str.zfill(4)   # pandas may read as int (0001→1)
    has_int   = "max_intensity_mmh" in df_all.columns
    n_total   = len(df_all)
    codes_ord = df_all["bsc"].value_counts().index.tolist()

    n_plots = 2 if has_int else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(16 if has_int else 9, 7))
    if n_plots == 1:
        axes = [axes]
    ax_dur = axes[0]
    ax_int = axes[1] if has_int else None

    for code in codes_ord:
        sub   = df_all[df_all["bsc"] == code]
        color = _BSC_PALETTE.get(code, "#cccccc")
        kw    = dict(color=color, s=8, alpha=0.65, linewidths=0,
                     label=f"{code} (n={len(sub)})", zorder=3)
        ax_dur.scatter(sub["total_mm"], sub["duration_h"], **kw)
        if has_int and ax_int is not None:
            ax_int.scatter(sub["total_mm"], sub["max_intensity_mmh"], **kw)

    for ax in axes[:n_plots]:
        ax.axvline(_BSC_LITERATURE_MM, color="#222222", lw=1.2, ls="--", zorder=4,
                   label=f"{_BSC_LITERATURE_MM} mm (literature threshold)")
        ax.set_xlabel("Event total rainfall [mm]", fontsize=10)
        ax.grid(alpha=0.25, zorder=0)

    for ax in axes[:n_plots]:
        ax.set_xlim(left=0, right=75)
    ax_dur.set_ylabel("Event duration [h]", fontsize=10)
    ax_dur.set_ylim(bottom=0, top=70)
    ax_dur.set_title("Event sum vs. Duration", fontsize=11)
    if has_int and ax_int is not None:
        ax_int.set_ylabel("Max hourly intensity [mm/h]", fontsize=10)
        ax_int.set_ylim(bottom=0, top=20)
        ax_int.set_title("Event sum vs. Max intensity", fontsize=11)

    handles, labels = ax_dur.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="BSC code", fontsize=7.5, title_fontsize=8,
        ncol=min(len(codes_ord) + 1, 9),
        bbox_to_anchor=(0.5, -0.02), loc="upper center",
    )
    fig.suptitle(
        f"BSC Rain Events — Rainfall Characteristics  "
        f"(n = {n_total}, >= {_BSC_MIN_EVENT_MM} mm)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.10, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart -> {out_path}")


def _make_bsc_vwc_distribution(df_all: pd.DataFrame, out_path: Path):
    """
    Two-panel distribution of the per-event VWC response (max across all depth channels).
    Left : absolute VWC increase (Δ% VWC), y capped at 1 000 events/bin.
    Right: VWC increase as % of local dynamic range, x capped at 300 %, y capped at 1 000.
    Bars that exceed the y cap are annotated with their true count in white text.
    Both panels include a cumulative-% overlay on the right y-axis.
    """
    _Y_CAP    = 1_000
    _X_PCT_MAX = 300.0

    depths   = ["-10", "-30", "-45"]
    abs_cols = [f"delta_vwc{d}"     for d in depths if f"delta_vwc{d}"     in df_all.columns]
    pct_cols = [f"delta_pct_dyn{d}" for d in depths if f"delta_pct_dyn{d}" in df_all.columns]
    if not abs_cols and not pct_cols:
        return

    df = df_all.copy()
    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(14, 5.5))

    thr_styles = [
        (5.0,  "#4dac26", "--", 1.5),
        (10.0, "#f1b300", "--", 1.5),
        (20.0, "#e31a1c", "-",  2.2),
        (30.0, "#7b0055", "--", 1.5),
    ]

    def _draw_panel(ax, vals_raw, xlabel, title, bar_color, cdf_color, xlim=None):
        vals_raw = vals_raw.dropna()
        vals_raw = vals_raw[vals_raw > 0]
        if vals_raw.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="grey")
            return 0
        # Histogram uses only values within xlim so bins are distributed usefully
        vals_hist = vals_raw[vals_raw <= xlim] if xlim is not None else vals_raw
        n_bins = min(60, max(20, len(vals_hist) // 10))
        counts, edges, patches = ax.hist(
            vals_hist, bins=n_bins, color=bar_color, alpha=0.70, ec="white", lw=0.3, zorder=2
        )
        # Y cap: add 15 % headroom so count labels sit above the cutoff line
        ax.set_ylim(0, _Y_CAP * 1.15)
        ax.set_yticks(range(0, _Y_CAP + 1, 200))
        ax.axhline(_Y_CAP, color="#888888", lw=0.6, ls="--", alpha=0.5, zorder=1)
        for cnt, patch in zip(counts, patches):
            if cnt > _Y_CAP:
                x_mid = patch.get_x() + patch.get_width() / 2
                ax.text(x_mid, _Y_CAP * 1.02, f"{int(cnt):,}",
                        ha="center", va="bottom", fontsize=6.5, rotation=90,
                        color="#222222", fontweight="bold", zorder=6)
        # CDF overlay — based on all values (not just those ≤ xlim) for a true CDF
        ax2 = ax.twinx()
        sv = np.sort(vals_raw.values)
        ax2.plot(sv, np.arange(1, len(sv) + 1) / len(sv) * 100,
                 color=cdf_color, lw=1.8, alpha=0.75, zorder=3)
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("Cumulative %", fontsize=9, color=cdf_color)
        ax2.tick_params(axis="y", labelcolor=cdf_color, labelsize=8)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(f"Events / bin  (y capped at {_Y_CAP:,})", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, xlim if xlim is not None else vals_hist.max() * 1.02)
        ax.text(0.98, 0.97, f"n = {len(vals_raw):,}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)
        ax.grid(alpha=0.2, zorder=0)
        return len(vals_raw)

    # Left panel — absolute
    if abs_cols:
        max_abs = df[abs_cols].max(axis=1)
        _draw_panel(ax_abs, max_abs,
                    "Max VWC increase in any channel (Δ% VWC)",
                    "Absolute soil moisture response\n(maximum across depths)",
                    "#6baed6", "#08519c")
    else:
        ax_abs.text(0.5, 0.5, "no delta_vwc data", transform=ax_abs.transAxes,
                    ha="center", va="center", fontsize=10, color="grey")

    # Right panel — % of dynamic range
    if pct_cols:
        max_pct = df[pct_cols].max(axis=1)
        _draw_panel(ax_pct, max_pct,
                    "Max VWC increase as % of dynamic range",
                    "Relative soil moisture response\n(maximum across depths)",
                    "#74c476", "#006d2c",
                    xlim=_X_PCT_MAX)
        # Threshold lines (draw after panel so xlim is already set)
        vals_valid = max_pct.dropna()
        vals_valid = vals_valid[vals_valid > 0]
        n_total = len(vals_valid)
        for thr, col, ls, lw in thr_styles:
            n_ge = int((vals_valid >= thr).sum())
            pct_ge = n_ge / n_total * 100 if n_total > 0 else 0
            ax_pct.axvline(thr, color=col, lw=lw, ls=ls, zorder=5, alpha=0.9,
                           label=f"≥{thr:.0f}%: {n_ge} ({pct_ge:.0f}%)")
        ax_pct.legend(fontsize=8, title="Threshold (events ≥)",
                      title_fontsize=8, loc="upper right")
    else:
        ax_pct.text(0.5, 0.5, "no delta_pct_dyn data", transform=ax_pct.transAxes,
                    ha="center", va="center", fontsize=10, color="grey")

    fig.suptitle(
        "Section 13e — VWC Response Distribution across all Rain Events",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart -> {out_path}")


# ---------------------------------------------------------------------------
# Event distribution per city chart
# ---------------------------------------------------------------------------

def _make_events_per_city(ev_df: pd.DataFrame, gdf, out_path: Path):
    """Grouped horizontal bar chart: rain and irrigation event counts per city.

    Left panel: absolute event counts (RAIN_ALL, IRRIGATION, IRRIGATION_STRONG as stacked overlay).
    Right panel: same counts normalised by the number of sensors in that city.
    Cities sorted by total events descending (most events at top of chart).
    Y-axis labels include mean elevation above sea level (amsl) when available.
    """
    # Build EUI → city from gdf -------------------------------------------
    def _proj_to_city(proj):
        if pd.isna(proj):
            return None
        p = str(proj).strip()
        city = _SHAPEFILE_PROJECT_TO_CITY.get(p)
        if city is None:
            for pfx in ("City ", "Company ", "Botanical Garden ",
                        "University ", "Castle ", "Schloss "):
                if p.startswith(pfx):
                    city = p[len(pfx):].strip()
                    break
        return city or p

    eui_col = "devEUI"
    if eui_col not in gdf.columns:
        return

    _gdf2 = gdf.copy()
    _gdf2["_city"] = _gdf2["project"].map(_proj_to_city)
    _gdf2["_eui_up"] = _gdf2[eui_col].astype(str).str.strip().str.upper()

    eui_to_city = dict(zip(_gdf2["_eui_up"], _gdf2["_city"]))
    city_n_sensors = (
        _gdf2[_gdf2["_city"].notna()]
        .groupby("_city")
        .size()
        .to_dict()
    )

    has_amsl = "amsl" in _gdf2.columns
    city_elev = {}
    if has_amsl:
        _gdf2["amsl"] = pd.to_numeric(_gdf2["amsl"], errors="coerce")
        city_elev = (
            _gdf2[_gdf2["_city"].notna() & _gdf2["amsl"].notna()]
            .groupby("_city")["amsl"]
            .mean()
            .to_dict()
        )

    # Map events to cities ------------------------------------------------
    ev = ev_df.copy()
    ev["_eui_up"] = ev["eui"].astype(str).str.strip().str.upper()
    ev["_city"] = ev["_eui_up"].map(eui_to_city)
    ev = ev[ev["_city"].notna()]
    if ev.empty:
        return

    city_rain   = ev[ev["event_type"] == "RAIN_ALL"].groupby("_city").size()
    city_irrig  = ev[ev["event_type"] == "IRRIGATION"].groupby("_city").size()
    city_strong = ev[ev["event_type"] == "IRRIGATION_STRONG"].groupby("_city").size()

    all_cities = sorted(set(city_rain.index) | set(city_irrig.index))
    df_c = pd.DataFrame({
        "city":   all_cities,
        "rain":   [int(city_rain.get(c, 0))   for c in all_cities],
        "irrig":  [int(city_irrig.get(c, 0))  for c in all_cities],
        "strong": [int(city_strong.get(c, 0)) for c in all_cities],
        "n_sens": [city_n_sensors.get(c, 0)   for c in all_cities],
    })
    # Sort: highest total events at the top of the chart (ascending = top rendered last)
    df_c = df_c.sort_values("rain", ascending=True).reset_index(drop=True)

    n_c  = len(df_c)
    y    = np.arange(n_c)
    bw   = 0.38
    fig_h = max(6.0, 0.50 * n_c + 2.5)

    CLR_RAIN   = "#2E86AB"
    CLR_IRRIG  = "#D63030"
    CLR_STRONG = "#8B0000"

    fig, (ax_abs, ax_nrm) = plt.subplots(
        1, 2, figsize=(16, fig_h), sharey=True,
        gridspec_kw={"wspace": 0.08})

    # ── Left panel: absolute counts ───────────────────────────────────────
    ax_abs.barh(y + bw / 2, df_c["rain"],   bw, color=CLR_RAIN,
                alpha=0.85, label="RAIN_ALL")
    ax_abs.barh(y - bw / 2, df_c["irrig"],  bw, color=CLR_IRRIG,
                alpha=0.85, label="IRRIGATION")
    ax_abs.barh(y - bw / 2, df_c["strong"], bw, color=CLR_STRONG,
                alpha=0.90, label="IRRIGATION_STRONG (subset)")

    # Annotate sensor count per city
    x_max_abs = max(df_c["rain"].max(), df_c["irrig"].max()) or 1
    for i, row in df_c.iterrows():
        n = row["n_sens"]
        if n:
            ax_abs.text(x_max_abs * 0.01, i, f"n={n}",
                        va="center", ha="left", fontsize=6.5, color="#555")

    ax_abs.set_xlabel("Number of events", fontsize=9)
    ax_abs.set_title("Absolute event counts", fontsize=10, fontweight="bold")
    ax_abs.legend(fontsize=7.5, loc="lower right")
    ax_abs.grid(axis="x", alpha=0.25, zorder=0)
    ax_abs.set_xlim(left=0)

    # ── Right panel: normalised per sensor ───────────────────────────────
    n_sens_arr = df_c["n_sens"].replace(0, np.nan)
    nrm_rain   = df_c["rain"]   / n_sens_arr
    nrm_irrig  = df_c["irrig"]  / n_sens_arr
    nrm_strong = df_c["strong"] / n_sens_arr

    ax_nrm.barh(y + bw / 2, nrm_rain,   bw, color=CLR_RAIN,   alpha=0.85)
    ax_nrm.barh(y - bw / 2, nrm_irrig,  bw, color=CLR_IRRIG,  alpha=0.85)
    ax_nrm.barh(y - bw / 2, nrm_strong, bw, color=CLR_STRONG, alpha=0.90)
    ax_nrm.set_xlabel("Events per sensor", fontsize=9)
    ax_nrm.set_title("Normalised by sensor count", fontsize=10, fontweight="bold")
    ax_nrm.grid(axis="x", alpha=0.25, zorder=0)
    ax_nrm.set_xlim(left=0)

    # Y-tick labels (shared axis) — include elevation if available
    if has_amsl:
        ylabels = []
        for c in df_c["city"]:
            elev = city_elev.get(c)
            if elev is not None and not np.isnan(elev):
                ylabels.append(f"{c}  ({elev:.0f} m a.s.l.)")
            else:
                ylabels.append(c)
    else:
        ylabels = df_c["city"].tolist()

    ax_abs.set_yticks(y)
    ax_abs.set_yticklabels(ylabels, fontsize=8)

    total_rain  = int(df_c["rain"].sum())
    total_irrig = int(df_c["irrig"].sum())
    fig.suptitle(
        f"Event Distribution by City — {n_c} cities\n"
        f"RAIN_ALL: {total_rain:,} events | IRRIGATION: {total_irrig:,} events",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart -> {out_path}")


# ---------------------------------------------------------------------------
# Timeseries analysis (reads sensor_data.csv from TreeTabularData)
# ---------------------------------------------------------------------------

VWC_KEYS = ["-10|ENV__SOIL__VWC", "-30|ENV__SOIL__VWC", "-45|ENV__SOIL__VWC"]
# A sensor counts as "actively transmitting" on a day only when it has >= this
# many depth channels above this threshold — matching add_sensor_dates.py logic.
VWC_ACTIVE_THRESHOLD_PCT = 2.0
VWC_ACTIVE_MIN_DEPTHS    = 2
ONLINE_THRESHOLD_DAYS = 14
BATTERY_PCT_KEY = "DEVICE|DEV__ENERGY__VCAP"
BATTERY_MV_KEY  = "DEVICE|DEV__ENERGY__VBAT"
RSSI_KEY        = "DEVICE|DEV__RF__RSSI"

# Sensor installation-location status values
STATUS_ONLINE  = "online"
STATUS_OFFLINE = "offline"
STATUS_REMOVED = "removed"


def load_sensor_telemetry(output_dir: Path, eui: str) -> dict:
    """Return last known battery % and RSSI for a sensor.

    Reads latest_attributes.json first; falls back to the tail of
    sensor_data.csv if the JSON is absent.
    Result keys: battery_pct, battery_mv, rssi (float or None each).
    """
    result = {"battery_pct": None, "battery_mv": None, "rssi": None}

    json_path = output_dir / "trees" / eui / "latest_attributes.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                attrs = json.load(f)
            for dest, key in [("battery_pct", BATTERY_PCT_KEY),
                               ("battery_mv",  BATTERY_MV_KEY),
                               ("rssi",        RSSI_KEY)]:
                if key in attrs:
                    try:
                        result[dest] = float(attrs[key]["value"])
                    except (TypeError, ValueError, KeyError):
                        pass
            return result
        except Exception:
            pass

    # Fallback: last non-null row in sensor_data.csv
    csv_path = output_dir / "trees" / eui / "sensor_data.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(
                csv_path,
                usecols=lambda c: c in {BATTERY_PCT_KEY, BATTERY_MV_KEY, RSSI_KEY},
            )
            for dest, key in [("battery_pct", BATTERY_PCT_KEY),
                               ("battery_mv",  BATTERY_MV_KEY),
                               ("rssi",        RSSI_KEY)]:
                if key in df.columns:
                    valid = df[key].dropna()
                    if len(valid):
                        result[dest] = float(valid.iloc[-1])
        except Exception:
            pass

    return result


def analyse_timeseries(output_dir: Path, dev_euis: list):
    """Analyse sensor_data.csv files and return per-sensor metrics.

    Returns a dict keyed by devEUI (upper-case) with:
      first_date   – first VWC transmission (date)
      last_date    – last VWC transmission (date)
      is_online    – bool, transmitted VWC in last 14 days
      coverage_pct – % of days with ≥1 VWC sample since first_date
      daily_dates  – set of dates with VWC data (for aggregation)
    Also returns a combined daily_counts Series (date → n sensors online).
    """
    trees_dir = output_dir / "trees"
    metrics = {}
    all_daily = {}  # devEUI → set of dates

    for eui in dev_euis:
        csv_path = trees_dir / eui / "sensor_data.csv"
        if not csv_path.exists():
            metrics[eui] = {"first_date": None, "last_date": None,
                            "is_online": False, "coverage_pct": 0.0,
                            "daily_dates": set(), "row_count": 0}
            continue

        try:
            df = pd.read_csv(csv_path, parse_dates=["datetime"],
                             usecols=lambda c: c in (
                                 ["datetime"] + VWC_KEYS))
        except Exception:
            metrics[eui] = {"first_date": None, "last_date": None,
                            "is_online": False, "coverage_pct": 0.0,
                            "daily_dates": set(), "row_count": 0}
            continue

        # Keep only rows where at least one VWC column has data
        vwc_cols = [c for c in df.columns if c in VWC_KEYS]
        if not vwc_cols:
            metrics[eui] = {"first_date": None, "last_date": None,
                            "is_online": False, "coverage_pct": 0.0,
                            "daily_dates": set(), "row_count": 0}
            continue

        df["_has_vwc"] = df[vwc_cols].notna().any(axis=1)
        df_vwc = df[df["_has_vwc"]].copy()
        if df_vwc.empty:
            metrics[eui] = {"first_date": None, "last_date": None,
                            "is_online": False, "coverage_pct": 0.0,
                            "daily_dates": set(), "row_count": 0}
            continue

        df_vwc["_date"] = df_vwc["datetime"].dt.date
        dates_with_data = set(df_vwc["_date"].unique())
        first = min(dates_with_data)
        last = max(dates_with_data)
        today = date.today()
        total_days = (today - first).days + 1
        coverage = len(dates_with_data) / total_days * 100 if total_days > 0 else 0
        is_online = (today - last).days <= ONLINE_THRESHOLD_DAYS

        # Qualifying active days: >= VWC_ACTIVE_MIN_DEPTHS channels above threshold
        _min_d = min(VWC_ACTIVE_MIN_DEPTHS, len(vwc_cols))
        _n_above = df_vwc[vwc_cols].gt(VWC_ACTIVE_THRESHOLD_PCT).sum(axis=1)
        active_dates = set(df_vwc.loc[_n_above >= _min_d, "_date"].unique())

        metrics[eui] = {
            "first_date": first,
            "last_date": last,
            "is_online": is_online,
            "coverage_pct": coverage,
            "daily_dates": dates_with_data,
            "row_count": len(df_vwc),
        }
        all_daily[eui] = active_dates  # qualifying days only for the "transmitting" chart line

    # Build daily sensor count: for each day, how many sensors had VWC data
    if all_daily:
        all_dates = set()
        for ds in all_daily.values():
            all_dates |= ds
        if all_dates:
            date_range = pd.date_range(min(all_dates), max(all_dates), freq="D")
            counts = []
            for d in date_range:
                d_date = d.date()
                n = sum(1 for ds in all_daily.values() if d_date in ds)
                counts.append(n)
            daily_counts = pd.Series(counts, index=date_range)
        else:
            daily_counts = pd.Series(dtype=int)
    else:
        daily_counts = pd.Series(dtype=int)

    return metrics, daily_counts


# ---------------------------------------------------------------------------
# Plausibility check
# ---------------------------------------------------------------------------

def _run_plausibility_check(gdf) -> list:
    """Detect devEUIs that are simultaneously active at more than one location.

    Two entries for the same EUI overlap when their [senInsDate, eff_rmv)
    windows intersect.  senInsDate=None → beginning of time;
    eff_rmv=None → entry is still active (open-ended).
    eff_rmv = min(senRmvDate, cutDwnDate), whichever is earlier.

    Returns a list of issue dicts:
      {eui, issue_type, description, rows: [row-info dicts]}
    """
    eui_rows = defaultdict(list)
    for idx, row in gdf.iterrows():
        eui = str(row.get("devEUI", "")).strip().upper()
        if not eui or eui == "NAN":
            continue
        ins_raw = row.get("senInsDate")
        # Use pd.notna() as the sole guard so that pd.NaT (which is
        # isinstance(pd.NaT, date)==True in some pandas versions) always
        # resolves to None instead of being stored in the dict.
        ins_d = None
        if pd.notna(ins_raw):
            ins_d = ins_raw.date() if hasattr(ins_raw, "date") else ins_raw
        # Effective removal: earliest of senRmvDate and cutDwnDate
        rmv_candidates = []
        for _col in ("senRmvDate", "cutDwnDate"):
            raw = row.get(_col)
            if pd.notna(raw):
                d = raw.date() if hasattr(raw, "date") else raw
                rmv_candidates.append(d)
        rmv_d = min(rmv_candidates) if rmv_candidates else None
        eui_rows[eui].append({
            "idx":      idx,
            "treeName": str(row.get("treeName", "")).strip() or "—",
            "project":  str(row.get("project", "Unknown")).strip(),
            "ins_date": ins_d,
            "rmv_date": rmv_d,
        })

    issues = []
    for eui, rows in eui_rows.items():
        if len(rows) < 2:
            continue

        # Multiple entries without a removal date → currently active at
        # several locations at once
        active = [r for r in rows if r["rmv_date"] is None]
        if len(active) > 1:
            issues.append({
                "eui":        eui,
                "issue_type": "multiple_active",
                "description": (
                    f"{len(active)} entries have no senRmvDate/cutDwnDate "
                    f"(sensor assigned to multiple active locations)"),
                "rows": rows,
            })
            continue  # Overlap check is redundant when this fires

        # Pairwise time-window overlap
        def _overlaps(a, b):
            a0 = a["ins_date"] or date.min
            b0 = b["ins_date"] or date.min
            if b["rmv_date"] is not None and a0 >= b["rmv_date"]:
                return False
            if a["rmv_date"] is not None and b0 >= a["rmv_date"]:
                return False
            return True

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if _overlaps(rows[i], rows[j]):
                    issues.append({
                        "eui":        eui,
                        "issue_type": "overlap",
                        "description": (
                            f"Active periods overlap between "
                            f"'{rows[i]['treeName']}' ({rows[i]['project']}) "
                            f"and '{rows[j]['treeName']}' ({rows[j]['project']})"),
                        "rows": [rows[i], rows[j]],
                    })

    return issues


# ---------------------------------------------------------------------------
# Outlier detection helpers
# ---------------------------------------------------------------------------

def _find_outliers(gdf, field, multiplier=3.0, abs_min=None, abs_max=None):
    """Return extreme outliers in gdf[field] as (devEUI, treeName, project, value).

    Uses the IQR × multiplier fence.  abs_min / abs_max add hard absolute bounds
    that are always flagged regardless of the IQR result (e.g. percentages > 100).
    Requires at least 8 non-null values; returns [] for constant or sparse columns.
    """
    numeric = pd.to_numeric(gdf[field], errors="coerce")
    col = numeric.dropna()
    if len(col) < 8:
        return []
    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        mask = (numeric < q1 - multiplier * iqr) | (numeric > q3 + multiplier * iqr)
    else:
        mask = pd.Series(False, index=gdf.index)
    if abs_min is not None:
        mask = mask | (numeric < abs_min)
    if abs_max is not None:
        mask = mask | (numeric > abs_max)
    result = []
    for idx, row in gdf[mask & numeric.notna()].iterrows():
        val  = float(numeric.loc[idx])
        eui  = str(row.get("devEUI",   "")).strip() or "—"
        name = str(row.get("treeName", "")).strip() or "—"
        proj = str(row.get("project",  "Unknown")).strip()
        result.append((eui, name, proj, val))
    result.sort(key=lambda x: abs(x[3]), reverse=True)
    return result


def _fmt_outlier_text(groups):
    """Format outlier groups as an HTML-compatible string for a ReportLab Paragraph.

    groups: list of (display_label, unit_str, [(eui, name, proj, val)])
    Returns "" when there are no outliers across all groups.
    """
    lines = []
    for label, unit, rows in groups:
        if not rows:
            continue
        lines.append(f"<b>{label} [{unit}]:</b>")
        for eui, name, proj, val in rows:
            lines.append(f"&nbsp;&nbsp;{eui} / {name} / {proj}: {val:.4g} {unit}")
    if not lines:
        return ""
    return (
        "<b>Possible data-entry errors (IQR × 3 rule):</b><br/>"
        + "<br/>".join(lines)
    )


# ---------------------------------------------------------------------------
# PDF report assembly
# ---------------------------------------------------------------------------
# Schematic classification diagram
# ---------------------------------------------------------------------------
def _build_schematic(out_path):
    """Draw a two-panel schematic explaining the v2 event classification."""
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor("#fafafa")

    # ── Left panel: hierarchy tree ────────────────────────────────────────
    ax_tree = fig.add_axes([0.01, 0.02, 0.46, 0.95])
    ax_tree.set_xlim(0, 10); ax_tree.set_ylim(0, 10); ax_tree.axis("off")
    ax_tree.set_title("Classification Hierarchy", fontsize=13,
                       fontweight="bold", pad=6)

    _NODE_STYLE = dict(ha="center", va="center")
    _CLR = {
        "IRRIGATION":        ("#D63030", "#fff"),
        "IRRIGATION_STRONG": ("#8B0000", "#fff"),
        "RAIN_ALL":          ("#9DC3E6", "#222"),
        "RAIN_HEAD":         ("#2E86AB", "#fff"),
        "RAIN_TAIL":         ("#9B59B6", "#fff"),
        "RAIN_HEAD_IMPACT":  ("#1b7837", "#fff"),
        "RAIN_STRONG":       ("#003300", "#fff"),
        "RAIN_TAIL_DRY":     ("#F18F01", "#fff"),
        "RAIN_FULL":         ("#003f7f", "#fff"),
    }
    # (name, x, y, short_desc)
    _nodes = [
        ("All detections",      5.0, 9.5, "sensor_data.csv"),
        ("IRRIGATION",          2.5, 8.2, "Simultaneous rise ≥20%dyn\nin all 3 depths within 4 h"),
        ("IRRIGATION_STRONG",   2.5, 6.9, "Same, but rise ≥80%dyn\nin all 3 depths"),
        ("RAIN_ALL",            7.2, 8.2, "24 h rolling sum ≥4 mm"),
        ("RAIN_HEAD",           5.2, 7.0, "Pre-peak: 68% of\n36h rain within ±12h"),
        ("RAIN_TAIL",           8.9, 7.0, "Post-peak: 68% of\n36h rain within ±12h"),
        ("RAIN_HEAD_IMPACT",    4.2, 5.7, "≥1 depth: ΔVWC ≥5%dyn\nAND ≥1%-pt"),
        ("RAIN_STRONG",         4.2, 4.4, "ALL 3 depths: ΔVWC ≥5%dyn\nAND ≥1%-pt"),
        ("RAIN_TAIL_DRY",       8.2, 5.7, "≥1 depth: ΔVWC ≥5%dyn\nAND drying return"),
        ("RAIN_FULL",           6.3, 4.4, "HEAD_IMPACT ∩ TAIL_DRY"),
    ]
    node_pos = {}
    for name, x, y, desc in _nodes:
        fc, tc = _CLR.get(name, ("#dddddd", "#000"))
        label   = f"{name}\n{desc}" if name != "All detections" else name
        ax_tree.text(x, y, label, color=tc, fontsize=7.5 if name != "All detections" else 9,
                     fontweight="bold" if name != "All detections" else "normal",
                     **_NODE_STYLE,
                     bbox=dict(boxstyle="round,pad=0.45", facecolor=fc,
                               edgecolor="#333", linewidth=1.2,
                               alpha=1.0 if name != "All detections" else 0.15))
        node_pos[name] = (x, y)

    # Arrows
    _arrows = [
        ("All detections", "IRRIGATION"),
        ("All detections", "RAIN_ALL"),
        ("IRRIGATION",     "IRRIGATION_STRONG"),
        ("RAIN_ALL",       "RAIN_HEAD"),
        ("RAIN_ALL",       "RAIN_TAIL"),
        ("RAIN_HEAD",      "RAIN_HEAD_IMPACT"),
        ("RAIN_HEAD_IMPACT","RAIN_STRONG"),
        ("RAIN_TAIL",      "RAIN_TAIL_DRY"),
        ("RAIN_HEAD_IMPACT","RAIN_FULL"),
        ("RAIN_TAIL_DRY",  "RAIN_FULL"),
    ]
    for src, dst in _arrows:
        x1, y1 = node_pos[src]; x2, y2 = node_pos[dst]
        ax_tree.annotate("", xy=(x2, y2+0.35), xytext=(x1, y1-0.35),
                         arrowprops=dict(arrowstyle="-|>", color="#555",
                                         lw=1.3, mutation_scale=12))
    # RAIN_FULL note
    ax_tree.text(6.3, 3.8, "⊂ HEAD_IMPACT   AND   ⊂ TAIL_DRY",
                 ha="center", fontsize=7, color="#003f7f", style="italic")

    # ── Right panel: annotated schematic timeseries ───────────────────────
    ax_r = fig.add_axes([0.52, 0.10, 0.46, 0.83])
    t = np.linspace(0, 144, 500)   # 144 h = 6 days

    # Stylised rain pulse centred at t=72 with pre/post tails
    rain = (np.exp(-((t-66)**2)/8) * 12 +
            np.exp(-((t-72)**2)/4) * 18 +
            np.exp(-((t-78)**2)/10) * 6 +
            np.exp(-((t-52)**2)/60) * 2.5 +
            np.exp(-((t-100)**2)/120) * 1.8)
    vwc  = 8 + 5*(1 - np.exp(-np.maximum(t - 68, 0)/6)) * np.exp(-np.maximum(t-80,0)/28)

    ax_r.fill_between(t, rain, color="steelblue", alpha=0.35, label="Hourly rain")
    ax_r.plot(t, rain, color="steelblue", lw=0.8, alpha=0.7)
    ax_r2 = ax_r.twinx()
    ax_r2.plot(t, vwc, color="#1b7837", lw=2.5, label="VWC at depth")
    ax_r2.set_ylabel("VWC (%)", color="#1b7837", fontsize=9)
    ax_r2.tick_params(axis="y", labelcolor="#1b7837", labelsize=8)
    ax_r2.set_ylim(5, 16)

    # Key reference lines and annotations
    t_onset = 72
    t_peak  = 80
    t_dry   = 110
    dyn_lo, dyn_hi = 8.2, 13.5
    pre_vwc = 8.2
    peak_vwc = vwc[np.argmin(np.abs(t - t_peak))]

    ax_r.axvline(t_onset, color="red", lw=2, linestyle="--", alpha=0.8)
    ax_r.axvline(t_peak,  color="#1b7837", lw=1.5, linestyle=":", alpha=0.7)
    ax_r.axvline(t_dry,   color="#F18F01", lw=1.5, linestyle=":", alpha=0.7)

    # Window shadings
    for h, alpha, clr in [(12, 0.12, "steelblue"), (36, 0.06, "steelblue")]:
        ax_r.axvspan(t_onset-h, t_onset+h, color=clr, alpha=alpha, zorder=0)

    # Annotations
    _ann_kw = dict(fontsize=8, ha="center",
                   bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                             edgecolor="#aaa", alpha=0.85))
    ax_r.text(t_onset,  27, "Onset\n(max hourly rain)", color="red",      **_ann_kw)
    ax_r.text(t_peak+4, 27, "VWC peak",                 color="#1b7837",  **_ann_kw)
    ax_r.text(t_dry+4,  27, "Drying\nreturn",           color="#F18F01",  **_ann_kw)
    ax_r.text(t_onset,  24, "±12 h inner\nwindow",      color="steelblue",**_ann_kw)
    ax_r.text(t_onset+18, 21, "±36 h outer\nwindow",    color="#5599bb",  **_ann_kw)

    ax_r2.annotate("", xy=(t_peak, peak_vwc), xytext=(t_peak, pre_vwc),
                   arrowprops=dict(arrowstyle="<->", color="#1b7837", lw=1.5,
                                   mutation_scale=12))
    ax_r2.text(t_peak+3, (pre_vwc+peak_vwc)/2,
               f"ΔVWC\n≥5%dyn for\nHEAD_IMPACT\n≥80%dyn for\nIRRIG_STRONG",
               color="#1b7837", fontsize=7, va="center")
    ax_r2.annotate("", xy=(t_dry, pre_vwc+0.3), xytext=(t_peak, pre_vwc+0.3),
                   arrowprops=dict(arrowstyle="<->", color="#F18F01", lw=1.3,
                                   mutation_scale=10))
    ax_r2.text((t_peak+t_dry)/2, pre_vwc+1.0, "Drying time",
               color="#F18F01", fontsize=8, ha="center")
    ax_r2.axhline(dyn_lo, color="#888", lw=0.9, linestyle=":")
    ax_r2.axhline(dyn_hi, color="#888", lw=0.9, linestyle=":")
    ax_r2.text(2, dyn_hi+0.2, "p95 (dyn. range max)", color="#888", fontsize=7)
    ax_r2.text(2, dyn_lo-0.5, "p05 (dyn. range min)", color="#888", fontsize=7)

    # frac_pre / frac_post annotation
    pre_rain  = np.trapz(rain[t<=t_onset][-25:], t[t<=t_onset][-25:])
    post_rain = np.trapz(rain[t>=t_onset][:25],  t[t>=t_onset][:25])
    outer_pre = np.trapz(rain[t<=t_onset][-73:], t[t<=t_onset][-73:])
    ax_r.text(t_onset-25, 5, f"frac_pre = inner12h_rain / 36h_pre_rain\n"
              f"→ RAIN_HEAD if ≥ 68 %", color="#2E86AB", fontsize=7.5,
              ha="center", bbox=dict(boxstyle="round,pad=0.25",
              facecolor="white", edgecolor="#2E86AB", alpha=0.9))
    ax_r.text(t_onset+25, 5, f"frac_post = inner12h_rain / 36h_post_rain\n"
              f"→ RAIN_TAIL if ≥ 68 %", color="#9B59B6", fontsize=7.5,
              ha="center", bbox=dict(boxstyle="round,pad=0.25",
              facecolor="white", edgecolor="#9B59B6", alpha=0.9))

    ax_r.set_xlabel("Hours around onset", fontsize=9)
    ax_r.set_ylabel("Precipitation (mm/h)", fontsize=9)
    ax_r.set_xlim(0, 144)
    ax_r.set_xticks(np.arange(0, 150, 24))
    ax_r.set_xticklabels([f"{int(v-72):+d}h" for v in np.arange(0, 150, 24)],
                          fontsize=8)
    ax_r.set_ylim(0, 32)
    ax_r.grid(alpha=0.2)
    ax_r.set_title("Schematic: key metrics and thresholds on one event",
                   fontsize=11, fontweight="bold")
    lines1, labels1 = ax_r.get_legend_handles_labels()
    lines2, labels2 = ax_r2.get_legend_handles_labels()
    ax_r.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="upper left")

    fig.suptitle("Rain & Irrigation Event Classification — Principles and Metrics",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------

def generate_report(gdf, output_dir, json_stats):
    report_dir = output_dir / "_report_assets"
    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "dataset_report.pdf"

    total = len(gdf)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    gdf["_genus_clean"] = (gdf["genus"].str.strip().str.lower()
                           .fillna("unknown").replace(GENUS_TYPO_FIX))
    gdf["_species_label"] = gdf.apply(
        lambda r: _build_species_label(
            r.get("genus", ""), r.get("species", ""), r.get("variety", "")
        )[1],
        axis=1,
    )
    def _age_from_row(row):
        age = compute_age_years(row.get("germDate"))
        if age is None:
            age = compute_age_years(row.get("plantDate"))
        return age

    gdf["_age_years"] = gdf.apply(_age_from_row, axis=1)

    print("\n  Generating charts...")

    # ── Timeseries freshness helpers ─────────────────────────────────────────
    def _ts_src_mtime() -> float:
        """Return the mtime of the newest source file (sensor data or shapefiles)."""
        trees_dir = output_dir / "trees"
        newest = 0.0
        if trees_dir.exists():
            for p in trees_dir.rglob("sensor_data.csv"):
                try: newest = max(newest, p.stat().st_mtime)
                except OSError: pass
        for extra in (output_dir / "irrigation_events_v2_all.csv",):
            if extra.exists():
                try: newest = max(newest, extra.stat().st_mtime)
                except OSError: pass
        # Track shapefile changes so terrain/availability charts regenerate after a sync
        gis_dir = output_dir.parent / "GISData"
        if gis_dir.exists():
            for p in gis_dir.rglob("treeLocations.shp"):
                try: newest = max(newest, p.stat().st_mtime)
                except OSError: pass
        return newest

    def _chart_fresh(chart_path: Path) -> bool:
        """True if the chart PNG exists and is newer than all timeseries sources."""
        try:
            return chart_path.exists() and chart_path.stat().st_mtime > _ts_src_mtime_val
        except OSError:
            return False

    _ts_src_mtime_val = _ts_src_mtime()

    # Charts that depend on sensor_data.csv / irrigation CSV
    _ts_charts = [
        report_dir / "chart_sensors_online.png",
        report_dir / "chart_completeness_trend.png",
        report_dir / "chart_irrigation_overview.png",
        report_dir / "chart_irrigation_seasonal.png",
        report_dir / "chart_irrigation_extended.png",
        report_dir / "chart_irrigation_min_vwc.png",
    ]
    _ts_all_fresh = all(_chart_fresh(c) for c in _ts_charts)

    # ── Timeseries analysis (reads sensor_data.csv files) ──
    all_euis = [str(r.get("devEUI", "")).strip().upper()
                for _, r in gdf.iterrows()
                if str(r.get("devEUI", "")).strip().upper() not in ("", "NAN")]

    # Always run the data scan — ts_metrics is needed for correct online/offline
    # sensor status regardless of whether the chart PNGs are already up to date.
    # (Only the matplotlib rendering steps are skipped when _ts_all_fresh is True.)
    ts_metrics, daily_sensor_counts = analyse_timeseries(output_dir, all_euis)
    if _ts_all_fresh:
        print("  [cache] Timeseries charts are current — skipping PNG regeneration.")

    # ── Three-state sensor status: online / offline / removed ──────────────
    today_date = date.today()

    # Ensure date columns exist (added by add_sensor_dates.py; absent on older
    # shapefiles).  cutDwnDate is set manually in QGIS for felled trees.
    for _col in ("senRmvDate", "senInsDate", "cutDwnDate"):
        if _col not in gdf.columns:
            gdf[_col] = pd.NaT
        else:
            gdf[_col] = pd.to_datetime(gdf[_col], errors="coerce")

    def _eff_rmv_date(row) -> date | None:
        """Return the earliest applicable removal date for a GDF row, or None."""
        candidates = []
        for _col in ("senRmvDate", "cutDwnDate"):
            raw = row.get(_col)
            if pd.notna(raw):
                d = raw.date() if hasattr(raw, "date") else raw
                candidates.append(d)
        return min(candidates) if candidates else None

    # Derive status and effective-removal date for every GDF row.
    # eff_rmv_map stores the effective removal date (date object or None).
    status_map:  dict = {}
    eff_rmv_map: dict = {}
    for idx, row in gdf.iterrows():
        eui   = str(row.get("devEUI", "")).strip().upper()
        eff_r = _eff_rmv_date(row)
        eff_rmv_map[idx] = eff_r
        if eff_r is not None and eff_r <= today_date:
            status_map[idx] = STATUS_REMOVED
            continue
        status_map[idx] = (
            STATUS_ONLINE if ts_metrics.get(eui, {}).get("is_online")
            else STATUS_OFFLINE
        )

    n_online  = sum(1 for s in status_map.values() if s == STATUS_ONLINE)
    n_offline = sum(1 for s in status_map.values() if s == STATUS_OFFLINE)
    n_removed = sum(1 for s in status_map.values() if s == STATUS_REMOVED)

    # Coverage / ts stats restricted to active (non-removed) sensors
    active_euis = {
        str(row.get("devEUI", "")).strip().upper()
        for idx, row in gdf.iterrows()
        if status_map[idx] != STATUS_REMOVED
        and str(row.get("devEUI", "")).strip().upper() not in ("", "NAN")
    }
    n_with_ts = sum(
        1 for e in active_euis if ts_metrics.get(e, {}).get("first_date"))
    avg_coverage = (
        np.mean([ts_metrics[e]["coverage_pct"] for e in active_euis
                 if ts_metrics.get(e, {}).get("first_date")])
        if n_with_ts > 0 else 0
    )
    # Total unique VWC datapoints: count rows with any VWC reading per sensor,
    # multiply by 3 to account for all three measurement depths (-10, -30, -45 cm)
    total_vwc_datapoints = sum(m.get("row_count", 0) for m in ts_metrics.values()) * 3

    print(f"  Timeseries: {n_with_ts} active sensors with data, "
          f"{n_online} online / {n_offline} offline / {n_removed} removed, "
          f"avg coverage {avg_coverage:.1f}%, "
          f"total VWC datapoints {total_vwc_datapoints:,}")

    # Plausibility check: each EUI must be active at ≤1 location at a time
    plausibility_issues = _run_plausibility_check(gdf)
    if plausibility_issues:
        print(f"  [WARN] {len(plausibility_issues)} plausibility issue(s) detected:")
        for _iss in plausibility_issues:
            print(f"         EUI {_iss['eui']}: {_iss['description']}")
    else:
        print("  Plausibility check: OK")

    # Per-project status counts for the pie chart
    status_per_project = defaultdict(
        lambda: {STATUS_ONLINE: 0, STATUS_OFFLINE: 0, STATUS_REMOVED: 0})
    for idx, row in gdf.iterrows():
        proj = str(row.get("project", "Unknown")).strip()
        status_per_project[proj][status_map[idx]] += 1

    # 0 – Sensor map: prefer the publication-quality study area map; fall back to cluster map
    _study_area_map = output_dir.parent / "figures" / "paper" / "fig_study_area_map.png"
    chart0 = report_dir / "chart_location_map.png"
    if not _study_area_map.exists():
        _make_cluster_map(gdf, chart0, min_cluster_radius_km=8.0)

    # 1 – Project pie (with online/offline colouring)
    proj_c = gdf["project"].fillna("Unknown").value_counts()
    chart1 = report_dir / "chart_project_pie.png"
    _make_project_pie(proj_c.index.tolist(), proj_c.values.tolist(),
                      "Trees per Project", chart1,
                      status_per_project=dict(status_per_project),
                      totals={STATUS_ONLINE: n_online, STATUS_OFFLINE: n_offline,
                              STATUS_REMOVED: n_removed})

    # 2 – Species pie
    chart2 = report_dir / "chart_species_pie.png"
    _make_species_pie(gdf, chart2)

    # 3-6 – Histograms
    chart3 = report_dir / "chart_height_hist.png"
    _make_histogram(gdf["height"], "Tree Height Distribution",
                    "Height [m]", chart3)

    chart4 = report_dir / "chart_age_hist.png"
    _make_histogram(gdf["_age_years"],
                    "Tree Age Distribution (germDate, fallback: plantDate)",
                    "Age [years]", chart4)

    chart5 = report_dir / "chart_crown_hist.png"
    _make_histogram(gdf["crownDiam"], "Crown Diameter Distribution",
                    "Crown Diameter [m]", chart5)

    chart6 = report_dir / "chart_stem_hist.png"
    _make_histogram(gdf["stemDiam"], "Stem Diameter Distribution",
                    "Stem Diameter [m]", chart6, clip_percentile=99)

    # 7 – Land use coverage & boxplot
    lu_present = {LAND_USE_FIELDS.get(f, f): gdf[f].notna().sum()
                  for f in LAND_USE_FIELDS if f in gdf.columns}
    chart7a = report_dir / "chart_landuse_coverage.png"
    _make_coverage_bar(lu_present, total,
                       "Land Use Attribute Coverage", chart7a)

    lu_data = {LAND_USE_FIELDS.get(f, f): gdf[f]
               for f in LAND_USE_FIELDS if f in gdf.columns}
    chart7b = report_dir / "chart_landuse_boxplot.png"
    _make_boxplot(lu_data, "Land Use Values Distribution [m²]",
                  "Area [m²]", chart7b)

    # 7 – Land use coverage per city — all fields heatmap
    _LU_CITY_COLS = [
        ("greenAtta2", "gAtta\n2.5m"), ("greenAtta5", "gAtta\n5m"),  ("greenAtta7", "gAtta\n7.5m"),
        ("greenDeta2", "gDeta\n2.5m"), ("greenDeta5", "gDeta\n5m"),  ("greenDeta7", "gDeta\n7.5m"),
        ("buildings2", "bldg\n2.5m"),  ("buildings5", "bldg\n5m"),   ("buildings7", "bldg\n7.5m"),
        ("sealedSu2",  "seal\n2.5m"),  ("sealedSu5",  "seal\n5m"),   ("sealedSu7",  "seal\n7.5m"),
        ("greenAttCD", "gAtta\nCD"),   ("greenDetCD", "gDeta\nCD"),
        ("buildingCD", "bldg\nCD"),    ("sealedSuCD", "seal\nCD"),
    ]
    chart7_city = report_dir / "chart_lu_city_coverage.png"
    if not _chart_fresh(chart7_city):
        def _geo_city_lu(proj):
            if pd.isna(proj):
                return None
            p = str(proj).strip()
            city = _SHAPEFILE_PROJECT_TO_CITY.get(p)
            if city is None:
                for pfx in ("City ", "Company ", "Botanical Garden ",
                            "University ", "Castle ", "Schloss "):
                    if p.startswith(pfx):
                        city = p[len(pfx):].strip()
                        break
            return city

        _gdf_lu2 = gdf.copy()
        _gdf_lu2["_geo_city"] = _gdf_lu2["project"].map(_geo_city_lu)
        _gdf_lu2 = _gdf_lu2[_gdf_lu2["_geo_city"].notna()]
        _lu_city_n = _gdf_lu2["_geo_city"].value_counts()
        _lu_cities = _lu_city_n.index.tolist()
        _lu_nc, _lu_nm = len(_lu_cities), len(_LU_CITY_COLS)

        _lu_cnt = np.zeros((_lu_nc, _lu_nm), dtype=int)
        _lu_pct = np.zeros((_lu_nc, _lu_nm), dtype=float)
        for _ci, _city in enumerate(_lu_cities):
            _sub = _gdf_lu2[_gdf_lu2["_geo_city"] == _city]
            _nt = len(_sub)
            for _mi, (col, _) in enumerate(_LU_CITY_COLS):
                _cnt = int(_sub[col].notna().sum()) if col in _sub.columns else 0
                _lu_cnt[_ci, _mi] = _cnt
                _lu_pct[_ci, _mi] = 100.0 * _cnt / _nt if _nt else 0.0

        _fig_h7 = max(5.0, 2.0 + _lu_nc * 0.42)
        _fig7, _ax7 = plt.subplots(figsize=(16, _fig_h7))
        _im7 = _ax7.imshow(_lu_pct, aspect="auto", cmap="YlGn",
                           vmin=0, vmax=100, interpolation="nearest")
        _ax7.set_xticks(np.arange(_lu_nm))
        _ax7.set_xticklabels([lbl for _, lbl in _LU_CITY_COLS], fontsize=8)
        _ax7.set_yticks(np.arange(_lu_nc))
        _ax7.set_yticklabels(
            [f"{c}  (n={_lu_city_n[c]})" for c in _lu_cities], fontsize=8)
        _ax7.set_title(
            f"Land use field availability per city  (total: {total} sensors)",
            fontsize=11, fontweight="bold")
        # Vertical separator between fixed-ring and crown-diameter groups
        _ax7.axvline(x=11.5, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
        for _ci in range(_lu_nc):
            for _mi in range(_lu_nm):
                _v  = _lu_pct[_ci, _mi]
                _cn = _lu_cnt[_ci, _mi]
                _tc = "white" if _v > 55 else ("black" if _cn > 0 else "#AAAAAA")
                _ax7.text(_mi, _ci,
                          f"{_cn}\n({_v:.0f}%)" if _cn > 0 else "—",
                          ha="center", va="center", fontsize=6, color=_tc)
        _fig7.colorbar(_im7, ax=_ax7, shrink=0.55, label="% of trees with data")
        _fig7.tight_layout(pad=1.5)
        _fig7.savefig(chart7_city, dpi=150, bbox_inches="tight")
        plt.close(_fig7)

    # Crown-diameter boxplot (page 8)
    cd_data = {CROWN_LU_FIELDS[f]: gdf[f]
               for f in CROWN_LU_FIELDS if f in gdf.columns}
    chart_cd_box = report_dir / "chart_crown_lu_boxplot.png"
    _make_boxplot(cd_data, "Crown-Diameter Land Use Values [m²]",
                  "Area [m²]", chart_cd_box)

    # 9 – Topographic indices  (was 8 before crown-diameter sections were added)
    topo_all = {**TOPO_FIELDS, **TPI_FIELDS}
    topo_data = {topo_all.get(f, f): gdf[f]
                 for f in topo_all if f in gdf.columns}
    chart8 = report_dir / "chart_topo_boxplot.png"
    _make_boxplot(topo_data, "Topographic Index Distributions",
                  "Value", chart8)

    # 9 – Soil sample coverage (bar chart)
    chem_fields_list = list(CHEMICAL_SOIL_FIELDS.keys())
    phys_fields_list = list(PHYSICAL_SOIL_FIELDS.keys())
    chem_count = sum(
        1 for _, r in gdf.iterrows()
        if any(pd.notna(r.get(f)) for f in chem_fields_list
               if f in gdf.columns))
    phys_count = sum(
        1 for _, r in gdf.iterrows()
        if any(pd.notna(r.get(f)) for f in phys_fields_list
               if f in gdf.columns))

    chart9 = report_dir / "chart_soil_coverage.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    soil_l = ["Chemical soil\nsamples", "Physical soil\nsamples"]
    soil_c = [chem_count, phys_count]
    bars = ax.bar(soil_l, soil_c,
                  color=[COLORS_PALETTE[0], COLORS_PALETTE[1]],
                  edgecolor="white", width=0.5)
    ax.axhline(total, color="grey", ls="--", lw=1, alpha=0.5,
               label=f"Total: {total}")
    for bar, cnt in zip(bars, soil_c):
        pct = cnt / total * 100 if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.02,
                f"{cnt} ({pct:.0f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of trees")
    ax.set_title("Soil Sample Availability", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(chart9, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 10 – Chemical soil properties boxplot
    chem_data = {CHEMICAL_SOIL_FIELDS[f]: gdf[f]
                 for f in CHEMICAL_SOIL_FIELDS if f in gdf.columns}
    chart10 = report_dir / "chart_soil_chemical_boxplot.png"
    _make_boxplot(chem_data,
                  "Chemical Soil Properties", "Value", chart10)

    # 11 – Physical soil properties boxplot
    phys_data = {PHYSICAL_SOIL_FIELDS[f]: gdf[f]
                 for f in PHYSICAL_SOIL_FIELDS if f in gdf.columns}
    chart11 = report_dir / "chart_soil_physical_boxplot.png"
    _make_boxplot(phys_data,
                  "Physical Soil Properties (Particle Size Fractions)",
                  "Percentage [%]", chart11)

    # ===================================================================
    # Assemble PDF – all images are placed with proportional scaling
    # ===================================================================
    print("  Assembling PDF report...")
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            topMargin=20*mm, bottomMargin=15*mm,
                            leftMargin=18*mm, rightMargin=18*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", parent=styles["Title"],
               fontSize=22, spaceAfter=6*mm,
               textColor=HexColor("#1a1a2e")))
    styles.add(ParagraphStyle("SubHead", parent=styles["Heading2"],
               fontSize=14, spaceAfter=4*mm,
               textColor=HexColor("#2E86AB")))
    styles.add(ParagraphStyle("Body", parent=styles["Normal"],
               fontSize=10, spaceAfter=3*mm, leading=14))
    styles.add(ParagraphStyle("Note", parent=styles["Normal"],
               fontSize=7.5, leading=10.5, spaceAfter=1*mm,
               textColor=HexColor("#444444"),
               fontName="Helvetica-Oblique"))

    story = []

    # --- Title page ---
    story.append(Spacer(1, 30*mm))
    story.append(Paragraph("Urban Tree Dataset Report", styles["ReportTitle"]))
    story.append(Paragraph(f"Generated: {now_str}", styles["Body"]))
    story.append(Spacer(1, 8*mm))

    unique_proj = gdf["project"].fillna("Unknown").nunique()
    unique_sp = gdf["_species_label"].nunique()
    n_active = n_online + n_offline
    summary = [
        ["Metric", "Value"],
        ["Total trees / sensors", str(total)],
        ["Projects / cities", str(unique_proj)],
        ["Unique species", str(unique_sp)],
        ["Sensors online (last 14 d)",
         f"{n_online} ({n_online/total*100:.1f}%)" if total > 0 else "0"],
        ["Sensors offline",
         f"{n_offline} ({n_offline/total*100:.1f}%)" if total > 0 else "0"],
        ["Sensors removed",
         f"{n_removed} ({n_removed/total*100:.1f}%)" if total > 0 else "0"],
        ["Chemical soil data",
         f"{chem_count} ({chem_count/total*100:.1f}%)"],
        ["Physical soil data",
         f"{phys_count} ({phys_count/total*100:.1f}%)"],
        ["JSON created", str(json_stats.get("created", 0))],
        ["JSON snapshots added", str(json_stats.get("versioned", 0))],
        ["JSON unchanged", str(json_stats.get("skipped", 0))],
    ]
    tbl = Table(summary, colWidths=[55*mm, 45*mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), HexColor("#2E86AB")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [HexColor("#f4f4f4"), colors.white]),
        ("ALIGN", (1,0), (1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(tbl)
    story.append(PageBreak())

    # Helper: add a proportionally-scaled chart on its own page
    def add_chart_page(path, caption, note=None, outlier_text=None):
        if path.exists():
            story.append(Paragraph(caption, styles["SubHead"]))
            extra_h = (22 * mm if note else 0) + (20 * mm if outlier_text else 0)
            img_max_h = PDF_MAX_HEIGHT - extra_h
            story.append(_proportional_image(path, max_h=img_max_h))
            if note:
                story.append(Spacer(1, 2 * mm))
                story.append(Paragraph(note, styles["Note"]))
            if outlier_text:
                story.append(Spacer(1, 2 * mm))
                story.append(Paragraph(outlier_text, styles["Note"]))
            story.append(Spacer(1, 4 * mm))
            story.append(PageBreak())

    # Helper: add two histograms per page (stacked) with an optional note
    def add_hist_pair(path_a, path_b, note=None, outlier_text=None):
        note_h   = 14 * mm if note else 0
        out_h    = 18 * mm if outlier_text else 0
        half_h   = (A4[1] - (20 + 15) * mm - 25 * mm - note_h - out_h) / 2 - 5 * mm
        for p in (path_a, path_b):
            if p.exists():
                story.append(_proportional_image(p, max_h=half_h))
                story.append(Spacer(1, 3 * mm))
        if note:
            story.append(Paragraph(note, styles["Note"]))
        if outlier_text:
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(outlier_text, styles["Note"]))
        story.append(PageBreak())

    # Helper: two charts on one page with a shared section heading and per-chart notes
    def add_chart_pair(path_a, path_b, caption, note_a=None, note_b=None):
        if not path_a.exists() and not path_b.exists():
            return
        story.append(Paragraph(caption, styles["SubHead"]))
        note_h = (18 * mm if note_a else 0) + (18 * mm if note_b else 0)
        each_h = (PDF_MAX_HEIGHT - note_h - 8 * mm) / 2
        if path_a.exists():
            story.append(_proportional_image(path_a, max_h=each_h))
            if note_a:
                story.append(Spacer(1, 1.5 * mm))
                story.append(Paragraph(note_a, styles["Note"]))
            story.append(Spacer(1, 3 * mm))
        if path_b.exists():
            story.append(_proportional_image(path_b, max_h=each_h))
            if note_b:
                story.append(Spacer(1, 1.5 * mm))
                story.append(Paragraph(note_b, styles["Note"]))
        story.append(Spacer(1, 4 * mm))
        story.append(PageBreak())

    if _study_area_map.exists():
        _n_geo_cities = (
            gdf["project"]
            .map(lambda p: _SHAPEFILE_PROJECT_TO_CITY.get(str(p).strip()) if pd.notna(p) else None)
            .dropna()
            .nunique()
        )
        add_chart_page(
            _study_area_map, "1 &ndash; Study Area Map",
            note=(
                f"{total} sensors deployed across {_n_geo_cities} cities "
                "in Germany, Austria, and Switzerland. "
                "Bubble size scales with the square root of the sensor count per city; "
                "the number inside each bubble is the raw sensor count. "
                "Country and state/canton borders from Natural Earth 1:10m. "
                "Map generated by figures/generate_study_area_map.py."
            ),
        )
    else:
        add_chart_page(
            chart0, "1 &ndash; Sensor Location Map",
            note=(
                "Sensor coordinates are extracted from the GIS shapefiles and "
                "projected to Web Mercator (EPSG:3857). Initial clustering assigns "
                "each sensor to a grid cell whose size is ≈ 4 % of the total "
                "geographic extent; a second pass iteratively merges the two closest "
                "cluster centres until no pair is within 5 % of the extent. Bubble "
                "area scales with the square root of sensor count so that small "
                "clusters remain visually distinguishable. The number inside each "
                "bubble is the raw sensor count for that cluster. "
                "Basemap: CartoDB Positron (contextily)."
            ),
        )
    add_chart_page(
        chart1, "2 &ndash; Trees per Project",
        note=(
            "Slice area is proportional to the total number of sensor locations "
            "per project (including removed ones). Slice colour encodes the "
            "online fraction among active (non-removed) sensors: full project "
            "colour = all active sensors online; progressively desaturated = "
            "increasing offline fraction; grey (#888888) = no active locations "
            "remain. A sensor is counted as online if its sensor_data.csv "
            "contains at least one valid VWC reading (any depth: \u221210, "
            "\u221230, or \u221245 cm) within the last "
            f"{ONLINE_THRESHOLD_DAYS} days. The legend for each project shows "
            "the three-way split: \u25cf online / \u25cf offline / "
            "\u25a0 removed."
        ),
    )
    add_chart_page(
        chart2, "3 &ndash; Species Distribution (by genus)",
        note=(
            "Each slice represents a distinct species label (genus + species + "
            "variety). Intra-genus shading groups related species under a shared "
            "hue; multiple species in the same genus are rendered as a range of "
            "lighter and darker shades of the genus base colour. Variety strings "
            "matching generic tokens (nan, 1, n/a, none, unbekannt, unknown) "
            "are stripped and merged with the base species entry. Slices are "
            "ordered by decreasing genus total, then by decreasing species count "
            "within each genus. Percentage labels are suppressed for slices "
            "smaller than 2.5 % to prevent overlap."
        ),
    )

    # Dimension histograms: two per page
    story.append(Paragraph("4 &ndash; Tree Dimension Distributions",
                           styles["SubHead"]))
    _ot_height = _fmt_outlier_text([
        ("Height",   "m",     _find_outliers(gdf, "height",     abs_min=0, abs_max=45,  multiplier=float("inf"))),
        ("Tree age", "years", _find_outliers(gdf, "_age_years", abs_min=0, abs_max=300, multiplier=float("inf"))),
    ])
    add_hist_pair(
        chart3, chart4,
        note=(
            "Each histogram uses 20 equal-width bins. Dashed vertical line = "
            "arithmetic mean; dash-dot line = median; both computed from "
            "non-null values only. Tree age (bottom) is calculated in decimal "
            "years from germDate to today; plantDate is used as a fallback when "
            "germDate is absent. Trees with no valid date in either field are "
            "excluded from the age histogram."
        ),
        outlier_text=_ot_height or None,
    )
    _ot_diam = _fmt_outlier_text([
        ("Crown diameter", "m", _find_outliers(gdf, "crownDiam", abs_min=0, abs_max=30, multiplier=float("inf"))),
        ("Stem diameter",  "m", _find_outliers(gdf, "stemDiam",  abs_min=0, abs_max=2,  multiplier=float("inf"))),
    ])
    add_hist_pair(
        chart5, chart6,
        note=(
            "Same histogram encoding as height and age above. Stem diameter "
            "(bottom): values above the 99th percentile are clipped to limit "
            "the influence of extreme outliers; the number of clipped values is "
            "annotated in the top-right corner of the subplot."
        ),
        outlier_text=_ot_diam or None,
    )

    add_chart_page(
        chart7a, "5 &ndash; Land Use Attribute Coverage",
        note=(
            "Each bar counts sensor records with a non-null value in that land "
            "use field; the dashed horizontal line is the total sensor count. "
            "Field name structure: category + radius ring. Categories: "
            "greenAtta = vegetated area directly connected to the root zone; "
            "greenDeta = vegetated area hydraulically disconnected (e.g. "
            "separated by a kerb); buildings = building footprint; "
            "sealedSu2/Su5/Su7 = impervious surface derived by subtracting "
            "all labelled categories from the theoretical annulus area. "
            "Radius rings: 2 = 0\u20132.5 m, 5 = 2.5\u20135.0 m, "
            "7 = 5.0\u20137.5 m from the trunk centre."
        ),
    )
    add_chart_page(
        chart7b,
        "6 &ndash; Land Use Values (m<super>2</super>) &ndash; Boxplot",
        note=(
            "Box = interquartile range (IQR); centre line = median; whiskers "
            "extend to 1.5 \u00d7 IQR; individual points beyond the whiskers "
            "are outliers. Values are in m\u00b2. Theoretical annulus areas: "
            "0\u20132.5 m ring = 19.6 m\u00b2; 2.5\u20135.0 m ring = "
            "58.9 m\u00b2; 5.0\u20137.5 m ring = 98.2 m\u00b2. Per-ring "
            "category sums exceeding the theoretical area are rescaled "
            "proportionally by compute_landuse.py."
        ),
    )

    add_chart_page(
        chart7_city, "7 &ndash; Land Use Field Coverage per City",
        note=(
            "Heatmap showing how many sensor locations in each geographic city "
            "have a non-null value for each land use field. Colour encodes the "
            "fraction of trees with data (0–100 %). Each cell shows the "
            "absolute count and percentage. The dashed vertical line separates "
            "fixed-radius ring fields (left, rings at 2.5/5/7.5 m from trunk) "
            "from crown-diameter fields (right, circle with radius = crownDiam/2). "
            "Categories: gAtta = greenAttached; gDeta = greenDetached; "
            "bldg = buildings; seal = sealed surface (residual area). "
            "Cities are sorted by total sensor count descending."
        ),
    )
    _ot_cd = _fmt_outlier_text([
        (lbl, "m²", _find_outliers(gdf, f, abs_min=0))
        for f, lbl in CROWN_LU_FIELDS.items() if f in gdf.columns
    ])
    add_chart_page(
        chart_cd_box,
        "8 &ndash; Crown-Diameter Land Use Values (m<super>2</super>) "
        "&ndash; Boxplot",
        note=(
            "Same boxplot encoding as section 6. Each value represents the "
            "area [m\u00b2] of the given land use category within a circle of "
            "radius = crownDiam / 2 centred on the tree, capped at 7.5 m. "
            "The sealed surface is derived by subtracting all labelled "
            "categories from the total circle area (\u03c0 r\u00b2). "
            "This metric is tree-size-dependent and intended for feature "
            "analysis alongside the fixed-radius ring values."
        ),
        outlier_text=_ot_cd or None,
    )

    _ot_topo = _fmt_outlier_text([
        (lbl, "",  _find_outliers(gdf, f))
        for f, lbl in {**TOPO_FIELDS, **TPI_FIELDS}.items() if f in gdf.columns
    ])
    add_chart_page(
        chart8, "9 &ndash; Topographic Index Distributions",
        note=(
            "Same boxplot encoding as land use values. "
            "TWI (Topographic Wetness Index) = ln(upslope catchment area / "
            "tan(slope)); higher values indicate wetter, more topographically "
            "flat positions. TPI (Topographic Position Index) = cell elevation "
            "minus mean elevation of a surrounding window; negative = valley "
            "bottom, positive = local ridge. flotAcc = count of upslope DEM "
            "cells draining through a pixel. slope is in degrees. Radius "
            "suffixes (2m5, 5m, 7m5) indicate the TPI neighbourhood radius."
        ),
        outlier_text=_ot_topo or None,
    )
    add_chart_page(
        chart9, "10 &ndash; Soil Sample Availability",
        note=(
            "A tree is counted once per bar if at least one field in the "
            "corresponding group is non-null. Chemical group (9 fields): pH, "
            "saltCont, conductiv, nSoluble, ammNSolubl, nitrNSolub, mgSoluble, "
            "phSoluble, kSoluble. Physical group (7 fields): part1Perc \u2013 "
            "part7Perc (particle-size distribution fractions). The dashed "
            "horizontal line marks the total sensor count."
        ),
    )
    _ot_chem = _fmt_outlier_text([
        (lbl, "",  _find_outliers(gdf, f))
        for f, lbl in CHEMICAL_SOIL_FIELDS.items() if f in gdf.columns
    ])
    add_chart_page(
        chart10, "11 &ndash; Chemical Soil Properties",
        note=(
            "Same boxplot encoding as land use values. Only records with a "
            "non-null value contribute to each individual box. All measurements "
            "are from laboratory analyses of soil samples collected at the tree "
            "site. Units follow the reporting convention of the originating "
            "laboratory and are stored as recorded in the original field data."
        ),
        outlier_text=_ot_chem or None,
    )
    _ot_phys = _fmt_outlier_text([
        (lbl, "%", _find_outliers(gdf, f, abs_min=0, abs_max=100))
        for f, lbl in PHYSICAL_SOIL_FIELDS.items() if f in gdf.columns
    ])
    add_chart_page(
        chart11, "12 &ndash; Physical Soil Properties",
        note=(
            "Same boxplot encoding. Each fraction [%] represents a particle-"
            "size class from a sieve or hydrometer analysis. For a complete "
            "granulometric analysis, fractions for one sample should sum to "
            "approximately 100 %; values outside this range indicate a partial "
            "analysis or laboratory rounding."
        ),
        outlier_text=_ot_phys or None,
    )

    # ===================================================================
    # Section: Per-tree data completeness / gap analysis
    # ===================================================================
    print("  Building data completeness tables...")

    # Determine reportable columns (skip geometry, fid, internal _* cols)
    skip_cols = {"geometry", "fid"}
    data_cols = [c for c in gdf.columns
                 if c not in skip_cols and not c.startswith("_")]
    n_fields = len(data_cols)

    # Compute per-tree completeness
    tree_rows = []
    for idx, row in gdf.iterrows():
        filled = []
        missing = []
        for c in data_cols:
            val = row.get(c)
            if pd.notna(val) and str(val).strip() not in ("", "nan"):
                filled.append(c)
            else:
                missing.append(c)
        ins_raw = row.get("senInsDate")
        ins_d = None
        if pd.notna(ins_raw):
            ins_d = ins_raw.date() if hasattr(ins_raw, "date") else ins_raw
        tree_rows.append({
            "idx":      idx,
            "devEUI":   str(row.get("devEUI", "??")).strip(),
            "treeName": str(row.get("treeName", "")).strip(),
            "project":  str(row.get("project", "Unknown")).strip(),
            "n_filled": len(filled),
            "n_missing": len(missing),
            "missing":  missing,
            "ins_date": ins_d,
            "eff_rmv":  eff_rmv_map.get(idx),
        })

    # Sort: fewest filled first → most data gaps on top
    tree_rows.sort(key=lambda r: r["n_filled"])

    # --- Completeness chart (survival / at-least-N curve) ---
    chart_comp = report_dir / "chart_completeness_hist.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fill_counts = [r["n_filled"] for r in tree_rows]
    mean_v = np.mean(fill_counts)
    std_v = np.std(fill_counts)

    # For each integer threshold t, count trees with n_filled >= t.
    # This gives a monotone non-increasing curve: the bar at x=t answers
    # "how many trees have data in at least t fields?"
    t_min, t_max = min(fill_counts), max(fill_counts)
    thresholds = list(range(t_min, t_max + 1))
    counts_at_least = [sum(1 for c in fill_counts if c >= t) for t in thresholds]

    ax.bar(thresholds, counts_at_least,
           color=COLORS_PALETTE[0], edgecolor="white", alpha=0.85, width=1.0)
    ax.axvline(mean_v, color=COLORS_PALETTE[1], ls="--", lw=1.5,
               label=f"Mean: {mean_v:.1f} / {n_fields}")
    ax.set_xlabel(f"Minimum fields with data (out of {n_fields})")
    ax.set_ylabel("Number of trees")
    ax.set_title("Data Completeness — Trees with At Least N Fields Filled",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(chart_comp, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Historical completeness log ---
    # Append current run to a persistent TSV file next to the report
    history_path = output_dir / "completeness_history.tsv"
    run_date = datetime.now().strftime("%d.%m.%Y")
    history_header = "date\ttotal_trees\tmean_filled\tstd_filled\tmax_fields\tn_online\tavg_coverage\ttotal_vwc_datapoints\n"
    history_line = (f"{run_date}\t{total}\t{mean_v:.2f}\t"
                    f"{std_v:.2f}\t{n_fields}\t{n_online}\t{avg_coverage:.2f}\t{total_vwc_datapoints}\n")

    # Read existing history (if any)
    history_rows = []
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:  # skip header
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    # Pad legacy rows (5 cols) that predate the sensor columns
                    while len(parts) < 8:
                        parts.append("")
                    history_rows.append(parts)

    # Check if today's entry already exists — replace it
    today_replaced = False
    for i, row in enumerate(history_rows):
        if row[0] == run_date:
            history_rows[i] = history_line.strip().split("\t")
            today_replaced = True
            break
    if not today_replaced:
        history_rows.append(history_line.strip().split("\t"))

    # Write back
    with open(history_path, "w", encoding="utf-8") as f:
        f.write(history_header)
        for row in history_rows:
            f.write("\t".join(row) + "\n")

    print(f"  Completeness history updated: {history_path}")

    # --- History trend chart (if more than 1 data point) ---
    chart_hist_trend = report_dir / "chart_completeness_trend.png"
    if len(history_rows) > 1 and not _chart_fresh(chart_hist_trend):
        h_dates = [r[0] for r in history_rows]
        h_trees = [int(r[1]) for r in history_rows]
        h_mean = [float(r[2]) for r in history_rows]
        h_std = [float(r[3]) for r in history_rows]
        h_max = [int(r[4]) for r in history_rows]
        h_pct = [m / mx * 100 if mx > 0 else 0
                 for m, mx in zip(h_mean, h_max)]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        x = range(len(h_dates))

        # Left axis: total trees
        color_trees = COLORS_PALETTE[0]
        ax1.bar(x, h_trees, color=color_trees, alpha=0.35,
                label="Total trees", zorder=2)
        ax1.set_ylabel("Total trees", color=color_trees)
        ax1.tick_params(axis="y", labelcolor=color_trees)
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(h_dates, rotation=45, ha="right", fontsize=8)

        # Right axis: mean completeness %
        ax2 = ax1.twinx()
        color_pct = COLORS_PALETTE[1]
        ax2.plot(x, h_pct, color=color_pct, marker="o", linewidth=2,
                 label="Mean completeness %", zorder=3)
        ax2.set_ylabel("Mean completeness [%]", color=color_pct)
        ax2.tick_params(axis="y", labelcolor=color_pct)
        ax2.set_ylim(0, 105)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper left", fontsize=9)

        ax1.set_title("Dataset Growth & Completeness Over Time",
                      fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(chart_hist_trend, dpi=150, bbox_inches="tight")
        plt.close(fig)

    add_chart_page(
        chart_comp, "13 &ndash; Data Completeness Overview",
        note=(
            f"Each sensor location is scored by counting non-null, non-empty "
            f"attribute columns out of all {n_fields} reportable fields "
            f"(geometry, fid, and internal _* columns are excluded). For each "
            f"value N on the x-axis the bar height shows how many of the "
            f"{total} sensor locations have data in at least N fields — bars "
            f"are therefore monotonically non-increasing from left to right "
            f"and a tree counted at x=60 is also counted at x=30. The "
            f"leftmost bar equals {total} (every tree). Dashed line = "
            f"arithmetic mean ({mean_v:.1f} / {n_fields} = "
            f"{mean_v/n_fields*100:.1f}%). Locations at the lower end are "
            f"typically from projects where only identification and coordinate "
            f"fields have been entered so far."
        ),
    )

    # --- Terrain & SVF availability chart (13b) ---
    chart_terrain = report_dir / "chart_terrain_availability.png"
    _TERRAIN_METRICS = [
        ("SVF",       ["SVF"]),
        ("TWI",       ["twi", "twi2m5", "twi5m", "twi7m5"]),
        ("TPI",       ["tpi", "tpi2m5", "tpi5m", "tpi7m5", "TPI_5m", "TPI_10m", "TPI_15m"]),
        ("Slope",     ["slope"]),
        ("Flow Acc.", ["flowAcc", "flotAcc"]),
    ]

    def _metric_mask(df, cols):
        mask = pd.Series(False, index=df.index)
        for c in cols:
            if c in df.columns:
                mask |= pd.to_numeric(df[c], errors="coerce").notna()
        return mask

    if not _chart_fresh(chart_terrain):
        overall_counts = {name: int(_metric_mask(gdf, cols).sum())
                          for name, cols in _TERRAIN_METRICS}

        proj_col = gdf["project"].fillna("Unknown").str.strip()
        city_order = proj_col.value_counts().head(20).index.tolist()
        city_data = {}
        for _city in city_order:
            _sub = gdf[proj_col == _city]
            city_data[_city] = {
                "_total": len(_sub),
                **{name: int(_metric_mask(_sub, cols).sum())
                   for name, cols in _TERRAIN_METRICS}
            }

        _mnames = [m[0] for m in _TERRAIN_METRICS]
        _n_cities = len(city_order)
        _mcolors = ["#1b7837", "#2166ac", "#762a83", "#e08214", "#d6604d"]

        _fig, (_ax_top, _ax_bot) = plt.subplots(
            2, 1,
            figsize=(12, 4.5 + _n_cities * 0.48),
            gridspec_kw={"height_ratios": [1, max(1, _n_cities * 0.55)]},
        )

        # Top panel — overall bar chart
        _yp = np.arange(len(_mnames))
        _cnts = [overall_counts[m] for m in _mnames]
        _bars = _ax_top.barh(_yp, _cnts, color=_mcolors, alpha=0.85, edgecolor="white")
        _ax_top.set_yticks(_yp)
        _ax_top.set_yticklabels(_mnames, fontsize=10)
        _ax_top.set_xlabel("Number of trees with data available", fontsize=9)
        _ax_top.set_title(
            f"Terrain & Sky-View Factor Availability  (total sensor locations: {total})",
            fontsize=11, fontweight="bold",
        )
        _ax_top.set_xlim(0, total * 1.15)
        _ax_top.axvline(total, color="gray", lw=0.8, linestyle="--", alpha=0.45,
                        label=f"Total ({total})")
        _ax_top.legend(fontsize=8, loc="lower right")
        for _bar, _cnt in zip(_bars, _cnts):
            _pct = 100 * _cnt / total if total else 0
            _ax_top.text(
                _cnt + total * 0.01,
                _bar.get_y() + _bar.get_height() / 2,
                f"{_cnt}  ({_pct:.0f} %)",
                va="center", ha="left", fontsize=9,
            )
        _ax_top.grid(axis="x", alpha=0.3)

        # Bottom panel — per-city heatmap
        _mat = np.zeros((_n_cities, len(_mnames)))
        for _ci, _city in enumerate(city_order):
            _nt = city_data[_city]["_total"]
            for _mi, _mn in enumerate(_mnames):
                _mat[_ci, _mi] = 100.0 * city_data[_city][_mn] / _nt if _nt else 0.0

        _im = _ax_bot.imshow(_mat, aspect="auto", cmap="YlGn",
                             vmin=0, vmax=100, interpolation="nearest")
        _ax_bot.set_xticks(np.arange(len(_mnames)))
        _ax_bot.set_xticklabels(_mnames, fontsize=9)
        _ax_bot.set_yticks(np.arange(_n_cities))
        _ax_bot.set_yticklabels(
            [f"{c}  (n={city_data[c]['_total']})" for c in city_order],
            fontsize=8,
        )
        _ax_bot.set_title(
            "Per-project breakdown  (colour = % of trees in that project with data)",
            fontsize=10,
        )
        for _ci in range(_n_cities):
            for _mi in range(len(_mnames)):
                _v = _mat[_ci, _mi]
                _n_abs = city_data[city_order[_ci]][_mnames[_mi]]
                _tc = "white" if _v > 55 else "black"
                _ax_bot.text(
                    _mi, _ci,
                    f"{int(_v)} %\nn={_n_abs}",
                    ha="center", va="center", fontsize=6.5,
                    color=_tc,
                    fontweight="bold" if _n_abs > 0 else "normal",
                )
        _fig.colorbar(_im, ax=_ax_bot, shrink=0.55,
                      label="% of trees in project with data")
        _fig.tight_layout(pad=1.5)
        _fig.savefig(chart_terrain, dpi=150, bbox_inches="tight")
        plt.close(_fig)

    _twi_n  = int(_metric_mask(gdf, ["twi","twi2m5","twi5m","twi7m5"]).sum())
    _tpi_n  = int(_metric_mask(gdf, ["tpi","tpi2m5","tpi5m","tpi7m5",
                                     "TPI_5m","TPI_10m","TPI_15m"]).sum())
    _slp_n  = int(_metric_mask(gdf, ["slope"]).sum())
    _svf_n  = int(_metric_mask(gdf, ["SVF"]).sum())
    _fac_n  = int(_metric_mask(gdf, ["flowAcc","flotAcc"]).sum())

    add_chart_page(
        chart_terrain,
        "13b &ndash; Terrain &amp; Sky-View Factor Availability",
        note=(
            f"Availability of terrain and sky-view factor (SVF) attributes. "
            f"SVF: {_svf_n} trees ({100*_svf_n//total if total else 0} %). "
            f"TWI: {_twi_n} ({100*_twi_n//total if total else 0} %). "
            f"TPI: {_tpi_n} ({100*_tpi_n//total if total else 0} %). "
            f"Slope: {_slp_n} ({100*_slp_n//total if total else 0} %). "
            f"Flow Acc.: {_fac_n} ({100*_fac_n//total if total else 0} %). "
            f"TWI = Topographic Wetness Index; TPI = Topographic Position Index; "
            f"SVF = Sky View Factor (from LiDAR / DSM). "
            f"Multiple scale variants are grouped (any non-null variant counts as present). "
            f"SVF is the most broadly available terrain metric, computed for "
            f"{100*_svf_n//total if total else 0} % of sensor locations across "
            f"most projects. TWI, Slope, and Flow Accumulation are currently limited "
            f"to projects where a high-resolution DEM was available."
        ),
    )

    # Resolve BSC events CSV: prefer bsc_events_all.csv, fall back to rf_dataset.csv
    _bsc_events_csv = next(
        (p for p in [output_dir / "bsc_events_all.csv",
                     output_dir / "rf_dataset.csv"] if p.exists()),
        None,
    )

    # --- 13c + 13c-ii: BSC quartile distribution — both on one page ---
    chart_bsc        = report_dir / "chart_bsc_quartiles.png"
    chart_bsc_impact = report_dir / "chart_bsc_quartiles_impact.png"
    _bsc_csv         = _bsc_events_csv
    _note_13c = _note_13c_ii = None

    if _bsc_csv is not None and _bsc_csv.exists():
        try:
            _bsc_df    = pd.read_csv(_bsc_csv)
            _bsc_n     = len(_bsc_df)
            _bsc_trees = _bsc_df["eui"].nunique() if "eui" in _bsc_df.columns else 0
            _bsc_thr   = _bsc_df.get("total_mm", pd.Series(dtype=float)).min()
            _bsc_thr_s = f">= {_bsc_thr:.1f} mm" if pd.notna(_bsc_thr) else ">= 5 mm"
            _note_13c  = (
                f"13c \u2014 Binary Shape Code (BSC) distribution by Huff quartile "
                f"(Terranova & Iaquinta, 2011). "
                f"Events with {_bsc_thr_s} total rainfall and >= 6 h dry gap. "
                f"BSC = 4-digit binary code; S\u2096 = 1 if the trapezoidal area of the "
                f"cumulative rainfall curve (\u03c0) over time-quartile k exceeds the area "
                f"of the uniform diagonal (USRP) over the same interval. "
                f"Stacked bars: % of all events (n = {_bsc_n}) per BSC code "
                f"within each Huff quartile. Covers {_bsc_trees} sensor trees. \u2014 "
                f"BSC code interpretation: because \u03c0 is monotone non-decreasing "
                f"(cumulative), a front-loaded storm keeps \u03c0 high throughout all "
                f"four quartiles, yielding BSC = 1111; a back-loaded storm (rain "
                f"concentrated in Q4) keeps \u03c0 near zero until the very end, "
                f"yielding BSC = 0000. The natural dominant sequence is: "
                f"1111 (Q1-heavy) \u2192 0111 (Q2) \u2192 0001 (Q3) \u2192 0000 (Q4 / uniform). "
                f"Codes such as 1100 or 1010 are rare because the monotone constraint "
                f"prevents \u03c0 from falling after an early rise. "
                f"For soil-moisture impulse analysis use XX11 (bits 3\u20134 = 1, "
                f"\u03c0 already high by Q3/Q4 \u2192 rain finished early); "
                f"for drying-time analysis use 00XX (bits 1\u20132 = 0, "
                f"\u03c0 still low in Q1/Q2 \u2192 back-loaded, peak comes late)."
            )
            _make_bsc_quartile_distribution(_bsc_df, chart_bsc)
            _imp_mask = _bsc_impact_mask(_bsc_df)
            _df_imp   = _bsc_df[_imp_mask]
            _n_imp    = len(_df_imp)
            if not _df_imp.empty:
                _make_bsc_quartile_distribution(
                    _df_imp, chart_bsc_impact,
                    title_suffix=(
                        f"  \u2014  VWC \u2265 {_BSC_IMPACT_DELTA:.0f}% dyn. range"
                        f" within {_BSC_IMPACT_PEAK_H:.0f} h  (n = {_n_imp})"
                    ),
                )
            _note_13c_ii = (
                f"13c-ii \u2014 Same chart restricted to events with a VWC rise of at least "
                f"{_BSC_IMPACT_DELTA:.0f}% of the local dynamic range in at least one depth "
                f"within {_BSC_IMPACT_PEAK_H:.0f} h of event end "
                f"(n = {_n_imp} of {_bsc_n}, "
                f"{100*_n_imp//_bsc_n if _bsc_n else 0}%). "
                f"Isolates events with a measurable hydrological impact on soil moisture."
            )
        except Exception as _e:
            print(f"  [WARN] 13c/13c-ii: {_e}")
            _note_13c    = "13c \u2014 BSC distribution by Huff quartile."
            _note_13c_ii = (f"13c-ii \u2014 VWC-impact events only "
                            f"(>= {_BSC_IMPACT_DELTA:.0f}% dyn. range).")

    add_chart_pair(chart_bsc, chart_bsc_impact,
                   "13c &ndash; BSC Rain Event Distribution",
                   note_a=_note_13c, note_b=_note_13c_ii)

    # --- 13d + 13d-ii: BSC scatter plots — both on one page ---
    chart_bsc_scatter        = report_dir / "chart_bsc_scatter.png"
    chart_bsc_scatter_impact = report_dir / "chart_bsc_scatter_impact.png"
    _bsc_scatter_csv         = _bsc_events_csv
    _note_13d = _note_13d_ii = None

    if _bsc_scatter_csv is not None and _bsc_scatter_csv.exists():
        try:
            _df_sc     = pd.read_csv(_bsc_scatter_csv)
            _make_bsc_scatter(_df_sc, chart_bsc_scatter)
            _df_sc_imp = _df_sc[_bsc_impact_mask(_df_sc)]
            if not _df_sc_imp.empty:
                _make_bsc_scatter(_df_sc_imp, chart_bsc_scatter_impact)
            _bsc_n2    = len(_df_sc)
            _n_sc_imp  = len(_df_sc_imp)
            _dur_range = (
                f"{_df_sc['duration_h'].min():.0f}–{_df_sc['duration_h'].max():.0f} h"
                if "duration_h" in _df_sc.columns else "n/a"
            )
            _has_int = "max_intensity_mmh" in _df_sc.columns
            _note_13d = (
                f"13d — Scatter plots of rain event characteristics coloured by BSC code "
                f"(n = {_bsc_n2} events). "
                f"Left: event total rainfall [mm] vs. event duration [h] "
                f"(range: {_dur_range}). "
                + (f"Right: event total rainfall [mm] vs. max hourly intensity [mm/h]. "
                   if _has_int else "") +
                f"Dashed vertical line at 12.7 mm marks the Wischmeier & Smith (1978) "
                f"erosivity threshold."
            )
            _note_13d_ii = (
                f"13d-ii — Same scatter restricted to events with a VWC rise of at least "
                f"{_BSC_IMPACT_DELTA:.0f}% of the local dynamic range in at least one depth "
                f"within {_BSC_IMPACT_PEAK_H:.0f} h of event end "
                f"(n = {_n_sc_imp} events, "
                f"{100*_n_sc_imp//_bsc_n2 if _bsc_n2 else 0}%). "
                f"Axes clipped to 75 mm × 70 h / 20 mm/h."
            )
        except Exception as _e:
            print(f"  [WARN] 13d/13d-ii: {_e}")
            _note_13d    = "13d — BSC event scatter plots."
            _note_13d_ii = (f"13d-ii — VWC-impact events only "
                            f"(>= {_BSC_IMPACT_DELTA:.0f}% dyn. range).")

    add_chart_pair(chart_bsc_scatter, chart_bsc_scatter_impact,
                   "13d &ndash; BSC Event Characteristics",
                   note_a=_note_13d, note_b=_note_13d_ii)

    # --- 13e: VWC response distribution — all rain events ---
    chart_bsc_vwc_dist = report_dir / "chart_bsc_vwc_distribution.png"
    _bsc_vwc_csv = _bsc_events_csv
    if _bsc_vwc_csv is not None and _bsc_vwc_csv.exists():
        try:
            _df_vwc = pd.read_csv(_bsc_vwc_csv)
            _make_bsc_vwc_distribution(_df_vwc, chart_bsc_vwc_dist)
        except Exception as _vwc_err:
            print(f"  [WARN] VWC distribution chart failed: {_vwc_err}")
    if chart_bsc_vwc_dist.exists():
        try:
            _df_vwc2   = pd.read_csv(_bsc_vwc_csv)
            _n_vwc     = len(_df_vwc2)
            _pct_cols  = [c for c in _df_vwc2.columns if c.startswith("delta_pct_dyn")]
            if _pct_cols:
                _max_pct = _df_vwc2[_pct_cols].max(axis=1).dropna()
                _max_pct = _max_pct[_max_pct > 0]
                _thr_parts = []
                for _t in _BSC_VWC_THRESHOLDS:
                    _n_ge = int((_max_pct >= _t).sum())
                    _thr_parts.append(
                        f"{_t:.0f}%: {_n_ge} events "
                        f"({_n_ge / len(_max_pct) * 100:.0f}%)"
                    )
                _thr_summary = "; ".join(_thr_parts)
            else:
                _thr_summary = "VWC data not available."
            _vwc_note = (
                f"Distribution of the maximum VWC increase per event across all three "
                f"measurement depths (−10, −30, −45 cm), based on {_n_vwc} rain events "
                f"(≥ {_BSC_MIN_EVENT_MM} mm). "
                f"<b>Left:</b> absolute VWC increase (Δ% VWC). "
                f"<b>Right:</b> VWC increase as a percentage of the local sensor dynamic "
                f"range (p05–p95). Dashed threshold lines at "
                + ", ".join(f"{t:.0f}%" for t in _BSC_VWC_THRESHOLDS) +
                f" of dynamic range; the solid red line at "
                f"{_BSC_IMPACT_DELTA:.0f}% marks the impact threshold used in sections "
                f"13c-ii and 13d-ii. The cumulative curve (right axis) shows the "
                f"fraction of events below each threshold. "
                f"Counts above thresholds: {_thr_summary}"
            )
        except Exception:
            _vwc_note = (
                "Distribution of maximum per-event VWC response across depths. "
                "Left: absolute (Δ% VWC); Right: relative (% of dynamic range). "
                f"Threshold lines at {', '.join(str(int(t)) for t in _BSC_VWC_THRESHOLDS)}% "
                "of dynamic range."
            )
        add_chart_page(chart_bsc_vwc_dist,
                       "13e &ndash; VWC Response Distribution",
                       note=_vwc_note)

    # --- 13f: DEM-derived terrain metric coverage per city ---
    chart_dem_city = report_dir / "chart_dem_city_coverage.png"
    _DEM_COLS = [
        ("amsl",    "Elevation\n(amsl)"),
        ("slope",   "Slope"),
        ("twi",     "TWI"),
        ("flotAcc", "Flow Acc."),
        ("tpi2m5",  "TPI\n2.5 m"),
        ("tpi5m",   "TPI\n5 m"),
        ("tpi7m5",  "TPI\n7.5 m"),
        ("SVF",     "SVF"),
        ("DTGW",    "DTGW"),
    ]
    if not _chart_fresh(chart_dem_city):
        def _proj_to_geo_city(proj):
            if pd.isna(proj):
                return None
            p = str(proj).strip()
            city = _SHAPEFILE_PROJECT_TO_CITY.get(p)
            if city is None:
                for pfx in ("City ", "Company ", "Botanical Garden ",
                            "University ", "Castle ", "Schloss "):
                    if p.startswith(pfx):
                        city = p[len(pfx):].strip()
                        break
            return city

        _gdf_dem = gdf.copy()
        _gdf_dem["_geo_city"] = _gdf_dem["project"].map(_proj_to_geo_city)
        _gdf_dem = _gdf_dem[_gdf_dem["_geo_city"].notna()]

        _city_n   = _gdf_dem["_geo_city"].value_counts()
        _cities   = _city_n.index.tolist()   # sorted by tree count descending
        _nc, _nm  = len(_cities), len(_DEM_COLS)

        _mat_cnt = np.zeros((_nc, _nm), dtype=int)
        _mat_pct = np.zeros((_nc, _nm), dtype=float)
        for _ci, _city in enumerate(_cities):
            _sub = _gdf_dem[_gdf_dem["_geo_city"] == _city]
            _nt  = len(_sub)
            for _mi, (col, _) in enumerate(_DEM_COLS):
                _cnt = int(pd.to_numeric(_sub[col], errors="coerce").notna().sum()) \
                       if col in _sub.columns else 0
                _mat_cnt[_ci, _mi] = _cnt
                _mat_pct[_ci, _mi] = 100.0 * _cnt / _nt if _nt else 0.0

        _col_labels = [lbl for _, lbl in _DEM_COLS]
        _row_labels = [f"{c}  (n={_city_n[c]})" for c in _cities]

        _fig_h = max(5.0, 2.0 + _nc * 0.42)
        _fig, _ax = plt.subplots(figsize=(13, _fig_h))
        _im = _ax.imshow(_mat_pct, aspect="auto", cmap="YlGn",
                         vmin=0, vmax=100, interpolation="nearest")
        _ax.set_xticks(np.arange(_nm))
        _ax.set_xticklabels(_col_labels, fontsize=9)
        _ax.set_yticks(np.arange(_nc))
        _ax.set_yticklabels(_row_labels, fontsize=8)
        _ax.set_title(
            f"DEM-derived terrain metric availability per city  (total: {total} sensors)",
            fontsize=11, fontweight="bold",
        )
        for _ci in range(_nc):
            for _mi in range(_nm):
                _v   = _mat_pct[_ci, _mi]
                _cnt = _mat_cnt[_ci, _mi]
                _tc  = "white" if _v > 55 else ("black" if _cnt > 0 else "#AAAAAA")
                _ax.text(_mi, _ci,
                         f"{_cnt}\n({_v:.0f} %)" if _cnt > 0 else "—",
                         ha="center", va="center", fontsize=6.5, color=_tc)
        _fig.colorbar(_im, ax=_ax, shrink=0.55, label="% of trees with data")
        _fig.tight_layout(pad=1.5)
        _fig.savefig(chart_dem_city, dpi=150, bbox_inches="tight")
        plt.close(_fig)

    _dem_totals = {
        col: int(pd.to_numeric(gdf[col], errors="coerce").notna().sum())
        if col in gdf.columns else 0
        for col, _ in _DEM_COLS
    }
    add_chart_page(
        chart_dem_city,
        "13f &ndash; DEM Terrain Metric Coverage per City",
        note=(
            "Number of sensor locations with each DEM-derived terrain attribute, "
            "grouped by geographic city and sorted by sensor count. "
            f"Totals across all cities: "
            f"amsl {_dem_totals['amsl']}, "
            f"slope {_dem_totals['slope']}, "
            f"TWI {_dem_totals['twi']}, "
            f"Flow Acc. {_dem_totals['flotAcc']}, "
            f"TPI 2.5 m {_dem_totals['tpi2m5']}, "
            f"TPI 5 m {_dem_totals['tpi5m']}, "
            f"TPI 7.5 m {_dem_totals['tpi7m5']}, "
            f"SVF {_dem_totals['SVF']}, "
            f"DTGW {_dem_totals['DTGW']}. "
            "amsl, slope, TWI, Flow Accumulation, and TPI are derived from "
            "high-resolution DEMs. SVF (Sky View Factor) is derived from LiDAR DSMs. "
            "DTGW (depth to groundwater) is from regional hydrogeological layers. "
            "Computed by calculate_terrain_metrics.py and stored in the shapefiles."
        ),
    )

    # --- Sensor online per day chart ---
    chart_online = report_dir / "chart_sensors_online.png"
    if len(daily_sensor_counts) > 1 and not _chart_fresh(chart_online):
        # Build daily_installed: sensors with senInsDate <= date and not yet removed
        date_range = daily_sensor_counts.index
        sensor_intervals = []
        for _, _row in gdf.iterrows():
            ins_raw = _row.get("senInsDate")
            if not pd.notna(ins_raw):
                continue
            ins_d = pd.Timestamp(ins_raw).normalize()
            rmv_candidates = []
            for _col in ("senRmvDate", "cutDwnDate"):
                raw = _row.get(_col)
                if pd.notna(raw):
                    rmv_candidates.append(pd.Timestamp(raw).normalize())
            rmv_d = min(rmv_candidates) if rmv_candidates else None
            sensor_intervals.append((ins_d, rmv_d))

        daily_installed = pd.Series(
            [sum(1 for ins_d, rmv_d in sensor_intervals
                 if ins_d <= d and (rmv_d is None or d < rmv_d))
             for d in date_range],
            index=date_range,
        )

        fig, ax = plt.subplots(figsize=(12, 4.5))

        # Background: total installed (red) — fills the full installed height
        ax.fill_between(date_range, daily_installed.values,
                        color="#D63030", alpha=0.35, label="Installed (not transmitting)")
        ax.plot(date_range, daily_installed.values,
                color="#D63030", lw=0.8, alpha=0.7)

        # Foreground: VWC-transmitting (blue) — sits on top; red overflow = offline gap
        ax.fill_between(daily_sensor_counts.index, daily_sensor_counts.values,
                        color=COLORS_PALETTE[0], alpha=0.6, label="Transmitting VWC data")
        ax.plot(daily_sensor_counts.index, daily_sensor_counts.values,
                color=COLORS_PALETTE[0], lw=0.8, alpha=0.9)

        ax.set_xlim(left=pd.Timestamp("2023-04-01"))
        ax.set_ylabel("Number of sensors")
        ax.set_xlabel("Date")
        ax.set_title("Sensor Deployment and Activity Over Time",
                     fontsize=13, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        fig.tight_layout()
        fig.savefig(chart_online, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Table paragraph styles (used by both history table and gap listing)
    cell_style = ParagraphStyle("Cell", parent=styles["Normal"],
                                fontSize=6.5, leading=8)
    cell_bold = ParagraphStyle("CellBold", parent=cell_style,
                               fontName="Helvetica-Bold")
    missing_style = ParagraphStyle("Missing", parent=styles["Normal"],
                                   fontSize=5.5, leading=7,
                                   textColor=HexColor("#666666"))

    # --- Historical completeness table in PDF ---
    story.append(Paragraph(
        "14 &ndash; Completeness History &amp; Dataset Size",
        styles["SubHead"]))
    story.append(Paragraph(
        f"Log of each script run stored in "
        f"<i>completeness_history.tsv</i>. "
        f"Today: {total} sensor locations, mean {mean_v:.1f} / {n_fields} fields "
        f"({mean_v/n_fields*100:.1f}%), stdev {std_v:.1f}.<br/>"
        f"<b>Sensor status today:</b> "
        f"<font color='#22A722'>&#x25CF;</font> {n_online} online &nbsp; "
        f"<font color='#D63030'>&#x25CF;</font> {n_offline} offline &nbsp; "
        f"<font color='#D4890A'>&#x25A0;</font> {n_removed} removed<br/>"
        f"<b>Mean timeseries coverage: {avg_coverage:.1f}%</b> "
        f"(avg % of days with VWC data, active sensors only)",
        styles["Body"]))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph(
        "<b>Column guide:</b> "
        "<i>Date</i> — script execution date. "
        "<i>Total trees</i> — number of sensor-location rows in the shapefile. "
        "<i>Mean filled / Stdev</i> — average and standard deviation of the "
        "non-null attribute count per row. "
        "<i>Max fields</i> — total reportable columns in that run (may grow as "
        "new attributes are added). "
        "<i>Completeness</i> — Mean filled ÷ Max fields × 100 %. "
        "<i>Sensors online</i> — sensors with at least one VWC reading in the "
        "14 days before the run, excluding removed sensors. "
        "<i>Mean coverage</i> — average % of calendar days with ≥1 VWC reading "
        "since first_date, computed over active (non-removed) sensors only. "
        "<i>Total VWC pts</i> — for each sensor, the number of transmission "
        "events in sensor_data.csv that contain at least one non-null VWC value "
        "across any of the three measurement depths (−10, −30, −45 cm), summed "
        "across all sensors and multiplied by 3 to give an upper bound on "
        "individual depth readings; the true count per depth may be lower if a "
        "transmission only recorded 1 or 2 depths.",
        styles["Note"]))
    story.append(Spacer(1, 2*mm))

    hist_table_data = [[
        Paragraph("<b>Date</b>", cell_bold),
        Paragraph("<b>Total trees</b>", cell_bold),
        Paragraph("<b>Mean filled</b>", cell_bold),
        Paragraph("<b>Stdev</b>", cell_bold),
        Paragraph("<b>Max fields</b>", cell_bold),
        Paragraph("<b>Completeness</b>", cell_bold),
        Paragraph("<b>Sensors online</b>", cell_bold),
        Paragraph("<b>Mean coverage</b>", cell_bold),
        Paragraph("<b>Total VWC pts</b>", cell_bold),
    ]]
    for hr in history_rows:
        try:
            pct = float(hr[2]) / int(hr[4]) * 100 if int(hr[4]) > 0 else 0
        except (ValueError, ZeroDivisionError):
            pct = 0
        # Sensor online / coverage / VWC datapoints (may be empty for legacy rows)
        online_str = hr[5] if hr[5] != "" else "–"
        try:
            cov_str = f"{float(hr[6]):.1f}%" if hr[6] != "" else "–"
        except ValueError:
            cov_str = "–"
        try:
            vwc_val = int(hr[7]) if hr[7] != "" else None
            vwc_str = f"{vwc_val:,}" if vwc_val is not None else "–"
        except ValueError:
            vwc_str = "–"
        hist_table_data.append([
            Paragraph(hr[0], cell_style),
            Paragraph(hr[1], cell_style),
            Paragraph(hr[2], cell_style),
            Paragraph(hr[3], cell_style),
            Paragraph(hr[4], cell_style),
            Paragraph(f"{pct:.1f}%", cell_bold),
            Paragraph(online_str, cell_style),
            Paragraph(cov_str, cell_style),
            Paragraph(vwc_str, cell_style),
        ])

    hist_col_w = [22*mm, 18*mm, 18*mm, 16*mm, 18*mm, 20*mm, 18*mm, 18*mm, 20*mm]
    htbl = Table(hist_table_data, colWidths=hist_col_w, repeatRows=1)
    htbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2E86AB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, HexColor("#f8f8f8")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(htbl)
    story.append(Spacer(1, 4*mm))

    # Add trend chart if available
    if chart_hist_trend.exists():
        story.append(_proportional_image(chart_hist_trend))
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(
            "Each data point represents one execution of urban_tree_report.py; "
            "results are appended to completeness_history.tsv after every run. "
            "Bar chart (left axis): total number of sensor locations in the "
            "dataset on that date. Line (right axis): mean attribute "
            "completeness as a percentage of the maximum possible field count "
            "for that run. Older runs may show fewer fields if new attribute "
            "columns were added to the dataset after those runs.",
            styles["Note"]))
        story.append(Spacer(1, 4 * mm))
    # Add sensor online chart if available
    if chart_online.exists():
        story.append(_proportional_image(chart_online))
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(
            "<font color='#D63030'><b>Red area:</b></font> total sensors installed "
            "on each calendar day \u2014 a sensor contributes from its <i>senInsDate</i> "
            "until its effective removal date (earliest of <i>senRmvDate</i> and "
            "<i>cutDwnDate</i>); sensors without a recorded <i>senInsDate</i> are "
            "excluded. "
            "<font color='#1f77b4'><b>Blue area:</b></font> sensors that "
            "transmitted at least one non-null VWC value (\u221210, \u221230, or \u221245 cm "
            "depth) on that day according to sensor_data.csv. The visible red "
            "region above the blue area represents sensors that were installed but "
            "did not transmit on that day (offline gap).",
            styles["Note"]))
    story.append(PageBreak())

    # --- Per-tree gap listing (paginated tables) ---
    # Field groups: if ANY member is missing, show the group label instead
    # of listing every individual field name.
    FIELD_GROUPS = {
        "chemical soil parameters": set(CHEMICAL_SOIL_FIELDS.keys()),
        "physical soil parameters": set(PHYSICAL_SOIL_FIELDS.keys()),
        "land use layers": {
            "greenAtta2", "greenAtta5", "greenAtta7",
            "greenDeta2", "greenDeta5", "greenDeta7",
            "buildings2", "buildings5", "buildings7",
            "sealedSu2", "sealedSu5", "sealedSu7",
        },
        "TPI indices": {"tpi2m5", "tpi5m", "tpi7m5"},
    }

    def _collapse_missing(missing_fields):
        """Replace groups of related missing fields with a single label."""
        remaining = set(missing_fields)
        result = []
        for label, members in FIELD_GROUPS.items():
            overlap = remaining & members
            if overlap:
                result.append(label)
                remaining -= members  # remove ALL group members
        # Add any ungrouped fields individually
        # Sort them in original column order for consistency
        ungrouped_order = [f for f in data_cols if f in remaining]
        result.extend(ungrouped_order)
        return result

    story.append(Paragraph(
        "15 &ndash; Per-Tree Data Gap Listing",
        styles["SubHead"]))
    story.append(Paragraph(
        f"Sensor locations sorted by completeness (least data first). "
        f"Each location has {n_fields} reportable attribute fields. "
        f"Related fields are grouped (e.g. <i>chemical soil parameters</i> "
        f"covers all 9 chemistry fields). "
        f"A sensor is considered <b>removed</b> when either its "
        f"<i>senRmvDate</i> (physical sensor removal) or <i>cutDwnDate</i> "
        f"(tree felled) is set to a date in the past.<br/>"
        f"<font color='#22A722'>&#x25CF;</font> = online (VWC data in last "
        f"14 days) &nbsp; "
        f"<font color='#D63030'>&#x25CF;</font> = offline &nbsp; "
        f"<font color='#D4890A'>&#x25A0;</font> = removed &nbsp; "
        f"<font color='#999999'>&#x25CB;</font> = no timeseries data<br/>"
        f"TS cov. for removed sensors (marked <b>*</b>) is bounded to the "
        f"[senInsDate, removal date) window.",
        styles["Body"]))
    story.append(Spacer(1, 3*mm))

    # Status dot styles
    cell_green  = ParagraphStyle("CellGreen",  parent=cell_style,
                                 textColor=HexColor("#22A722"))
    cell_red    = ParagraphStyle("CellRed",    parent=cell_style,
                                 textColor=HexColor("#D63030"))
    cell_orange = ParagraphStyle("CellOrange", parent=cell_style,
                                 textColor=HexColor("#D4890A"))
    cell_grey   = ParagraphStyle("CellGrey",   parent=cell_style,
                                 textColor=HexColor("#999999"))

    # 7 columns: devEUI | Tree Name | Project | Filled | St | TS cov. | Missing
    _fixed = 25 + 28 + 21 + 13 + 7 + 10  # = 104 mm
    col_widths = [25*mm, 28*mm, 21*mm, 13*mm, 7*mm, 10*mm,
                  PDF_MAX_WIDTH - _fixed * mm]
    header = [
        Paragraph("<b>devEUI</b>",         cell_bold),
        Paragraph("<b>Tree Name</b>",      cell_bold),
        Paragraph("<b>Project</b>",        cell_bold),
        Paragraph("<b>Filled</b>",         cell_bold),
        Paragraph("<b>St</b>",             cell_bold),
        Paragraph("<b>TS cov.</b>",        cell_bold),
        Paragraph("<b>Missing fields</b>", cell_bold),
    ]

    ROWS_PER_PAGE = 22
    for batch_start in range(0, len(tree_rows), ROWS_PER_PAGE):
        batch = tree_rows[batch_start:batch_start + ROWS_PER_PAGE]
        table_data = [header]

        for tr in batch:
            pct = tr["n_filled"] / n_fields * 100
            fill_str = f"{tr['n_filled']}/{n_fields} ({pct:.0f}%)"
            collapsed = _collapse_missing(tr["missing"])
            if len(collapsed) > 8:
                miss_str = (", ".join(collapsed[:8])
                            + f"  ... +{len(collapsed)-8} more")
            else:
                miss_str = ", ".join(collapsed) if collapsed else "—"

            # Status cell + TS coverage
            row_status = status_map.get(tr["idx"], STATUS_OFFLINE)
            eui_upper  = tr["devEUI"].upper()
            tm         = ts_metrics.get(eui_upper, {})
            if row_status == STATUS_REMOVED:
                status_dot = Paragraph("&#x25A0;", cell_orange)  # orange square
                # Bounded coverage: count days with VWC data within [ins, eff_rmv]
                ins_d  = tr.get("ins_date")
                eff_r  = tr.get("eff_rmv")
                d_set  = tm.get("daily_dates", set())
                if ins_d and eff_r and d_set and eff_r > ins_d:
                    span_days = (eff_r - ins_d).days
                    in_window = sum(
                        1 for d in d_set
                        if ins_d <= (d.date() if hasattr(d, "date") else d) < eff_r
                    )
                    cov_str = Paragraph(
                        f"{in_window/span_days*100:.0f}%*", cell_style)
                else:
                    cov_str = Paragraph("—", cell_style)
            elif tm.get("first_date") is None:
                status_dot = Paragraph("&#x25CB;", cell_grey)    # empty circle
                cov_str    = Paragraph("—", cell_style)
            elif row_status == STATUS_ONLINE:
                status_dot = Paragraph("&#x25CF;", cell_green)   # green dot
                cov_str    = Paragraph(f"{tm['coverage_pct']:.0f}%", cell_style)
            else:
                status_dot = Paragraph("&#x25CF;", cell_red)     # red dot
                cov_str    = Paragraph(f"{tm['coverage_pct']:.0f}%", cell_style)

            table_data.append([
                Paragraph(tr["devEUI"],        cell_style),
                Paragraph(tr["treeName"][:26], cell_style),
                Paragraph(tr["project"][:18],  cell_style),
                Paragraph(fill_str,            cell_bold),
                status_dot,
                cov_str,
                Paragraph(miss_str,            missing_style),
            ])

        tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2E86AB")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, HexColor("#f8f8f8")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(tbl)
        story.append(PageBreak())

    # ===================================================================
    # Section 16: Offline Sensor Maps (STATUS_OFFLINE only, not removed)
    # ===================================================================
    offline_idx_by_project = defaultdict(list)
    for idx, row in gdf.iterrows():
        if status_map[idx] == STATUS_OFFLINE:
            proj = str(row.get("project", "Unknown")).strip()
            offline_idx_by_project[proj].append(idx)

    if offline_idx_by_project:
        n_off_total = sum(len(v) for v in offline_idx_by_project.values())
        print(f"  Building offline sensor maps: {n_off_total} offline sensor(s) "
              f"across {len(offline_idx_by_project)} project(s)...")

        telemetry_cache = {}
        for idxs in offline_idx_by_project.values():
            for idx in idxs:
                eui = str(gdf.loc[idx, "devEUI"]).strip().upper()
                if eui not in telemetry_cache:
                    telemetry_cache[eui] = load_sensor_telemetry(output_dir, eui)

        story.append(Paragraph(
            "16 &ndash; Offline Sensor Maps by Project",
            styles["SubHead"]))
        story.append(Paragraph(
            f"<b>{n_off_total}</b> sensor(s) across "
            f"<b>{len(offline_idx_by_project)}</b> project(s) are offline "
            f"(no VWC data in the last {ONLINE_THRESHOLD_DAYS} days) and have "
            f"not been removed from their location. "
            f"Each map shows their geographic position and last known diagnostics "
            f"(serial number, tree name, last transmission date, battery, RSSI). "
            f"Removed sensors are listed separately in section 17.",
            styles["Body"]))
        story.append(PageBreak())

        for proj_name in sorted(offline_idx_by_project.keys()):
            idxs          = offline_idx_by_project[proj_name]
            proj_gdf      = gdf.loc[idxs].copy()
            safe_name     = (proj_name.replace(" ", "_")
                                      .replace("/", "_")
                                      .replace("\\", "_"))
            chart_offline = report_dir / f"chart_offline_{safe_name}.png"
            _make_offline_project_map(
                proj_name, proj_gdf, ts_metrics, telemetry_cache, chart_offline)
            if chart_offline.exists():
                story.append(_proportional_image(chart_offline,
                                                  max_h=PDF_MAX_HEIGHT + 40*mm))
                story.append(PageBreak())

    # ===================================================================
    # Section 17: Removed Sensor Registry
    # ===================================================================
    removed_idx_by_project = defaultdict(list)
    for idx, row in gdf.iterrows():
        if status_map[idx] == STATUS_REMOVED:
            proj = str(row.get("project", "Unknown")).strip()
            removed_idx_by_project[proj].append(idx)

    story.append(Paragraph(
        "17 &ndash; Removed Sensor Registry",
        styles["SubHead"]))

    if not removed_idx_by_project:
        story.append(Paragraph(
            "No sensors have been marked as removed (senRmvDate is empty "
            "for all entries).",
            styles["Body"]))
        story.append(PageBreak())
    else:
        n_rmv_total = sum(len(v) for v in removed_idx_by_project.values())
        story.append(Paragraph(
            f"<b>{n_rmv_total}</b> sensor location(s) across "
            f"<b>{len(removed_idx_by_project)}</b> project(s) are classified as "
            f"removed. A location is removed when either <i>senRmvDate</i> "
            f"(sensor physically relocated or decommissioned) or "
            f"<i>cutDwnDate</i> (tree felled) is set to a past date. "
            f"No further data is expected from these locations. The same devEUI "
            f"may reappear in the dataset under a different tree name if the "
            f"device was redeployed.",
            styles["Body"]))
        story.append(Spacer(1, 3*mm))

        rmv_header = [
            Paragraph("<b>devEUI</b>",      cell_bold),
            Paragraph("<b>Tree Name</b>",   cell_bold),
            Paragraph("<b>Project</b>",     cell_bold),
            Paragraph("<b>senInsDate</b>",  cell_bold),
            Paragraph("<b>Removal date</b>", cell_bold),
            Paragraph("<b>Reason</b>",      cell_bold),
            Paragraph("<b>Days active</b>", cell_bold),
        ]
        rmv_col_w = [28*mm, 34*mm, 30*mm, 22*mm, 22*mm, 22*mm, 18*mm]

        rmv_table_data = [rmv_header]
        for proj_name in sorted(removed_idx_by_project.keys()):
            for idx in removed_idx_by_project[proj_name]:
                row     = gdf.loc[idx]
                eui     = str(row.get("devEUI", "")).strip().upper()
                tname   = str(row.get("treeName", "")).strip() or "—"
                ins_raw = row.get("senInsDate")
                ins_str = (pd.Timestamp(ins_raw).strftime("%Y-%m-%d")
                           if pd.notna(ins_raw) else "—")
                # Determine effective removal date and reason
                sen_raw = row.get("senRmvDate")
                cut_raw = row.get("cutDwnDate")
                sen_d = (pd.Timestamp(sen_raw).date()
                         if pd.notna(sen_raw) else None)
                cut_d = (pd.Timestamp(cut_raw).date()
                         if pd.notna(cut_raw) else None)
                if sen_d and cut_d:
                    eff_d  = min(sen_d, cut_d)
                    reason = "sensor removed" if sen_d <= cut_d else "tree felled"
                elif sen_d:
                    eff_d  = sen_d
                    reason = "sensor removed"
                elif cut_d:
                    eff_d  = cut_d
                    reason = "tree felled"
                else:
                    eff_d  = None
                    reason = "—"
                eff_str = eff_d.strftime("%Y-%m-%d") if eff_d else "—"
                if ins_str != "—" and eff_d:
                    days = (eff_d - pd.Timestamp(ins_raw).date()).days
                    days_str = str(days)
                else:
                    days_str = "—"
                rmv_table_data.append([
                    Paragraph(eui,           cell_style),
                    Paragraph(tname[:30],    cell_style),
                    Paragraph(proj_name[:26], cell_style),
                    Paragraph(ins_str,       cell_style),
                    Paragraph(eff_str,       cell_style),
                    Paragraph(reason,        cell_style),
                    Paragraph(days_str,      cell_style),
                ])

        rmv_tbl = Table(rmv_table_data, colWidths=rmv_col_w, repeatRows=1)
        rmv_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#D4890A")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTSIZE",   (0, 0), (-1, -1), 7.5),
            ("GRID",       (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, HexColor("#fff8f0")]),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING",   (0, 0), (-1, -1), 3),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
        ]))
        story.append(rmv_tbl)
        story.append(PageBreak())

    # ===================================================================
    # Section 18 (conditional): Plausibility Issues
    # ===================================================================
    if plausibility_issues:
        story.append(Paragraph(
            "18 &ndash; Plausibility Issues",
            styles["SubHead"]))
        story.append(Paragraph(
            f"<b>{len(plausibility_issues)} issue(s)</b> were detected where a "
            f"single devEUI appears to be active at more than one tree location "
            f"simultaneously. This means either <i>senRmvDate</i> or "
            f"<i>cutDwnDate</i> is missing for the old location, or the "
            f"<i>senInsDate</i> windows overlap. "
            f"Please correct the affected entries in QGIS and re-run "
            f"<i>add_sensor_dates.py</i>.",
            styles["Body"]))
        story.append(Spacer(1, 3*mm))

        pq_header = [
            Paragraph("<b>devEUI</b>",   cell_bold),
            Paragraph("<b>Issue</b>",    cell_bold),
            Paragraph("<b>Locations involved</b>", cell_bold),
        ]
        pq_col_w = [30*mm, 50*mm, PDF_MAX_WIDTH - 80*mm]
        pq_data  = [pq_header]
        for iss in plausibility_issues:
            loc_lines = "; ".join(
                f"{r['treeName']} ({r['project']}, "
                f"ins:{r['ins_date'] or '?'}, rmv:{r['rmv_date'] or 'open'})"
                for r in iss["rows"]
            )
            pq_data.append([
                Paragraph(iss["eui"],         cell_style),
                Paragraph(iss["description"], cell_style),
                Paragraph(loc_lines,          missing_style),
            ])

        pq_tbl = Table(pq_data, colWidths=pq_col_w, repeatRows=1)
        pq_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#C73E1D")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTSIZE",   (0, 0), (-1, -1), 7.5),
            ("GRID",       (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, HexColor("#fff5f5")]),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 3),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
        ]))
        story.append(pq_tbl)
        story.append(PageBreak())

    # ===================================================================
    # Section 19: Irrigation & Rain Event Statistics
    # ===================================================================
    MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    irrig_csv = output_dir / "irrigation_events_v2_all.csv"
    chart_irrig_overview  = report_dir / "chart_irrigation_overview.png"
    chart_irrig_pie       = report_dir / "chart_irrigation_pie.png"
    chart_irrig_seasonal  = report_dir / "chart_irrigation_seasonal.png"

    # Colours for all event types
    CLR_IRRIG        = "#D63030"   # red         — IRRIGATION
    CLR_IRRIG_STRONG = "#8B0000"   # dark red    — IRRIGATION_STRONG
    CLR_RAIN_ALL     = "#9DC3E6"   # pale blue   — RAIN_ALL
    CLR_HEAD         = "#2E86AB"   # steelblue   — RAIN_HEAD
    CLR_TAIL         = "#9B59B6"   # purple      — RAIN_TAIL
    CLR_HEAD_IMP     = "#1b7837"   # green       — RAIN_HEAD_IMPACT
    CLR_RAIN_STRONG  = "#003300"   # dark green  — RAIN_STRONG
    CLR_TAIL_DRY     = "#F18F01"   # orange      — RAIN_TAIL_DRY
    CLR_FULL         = "#003f7f"   # dark navy   — RAIN_FULL
    chart_irrig_schematic  = report_dir / "chart_irrigation_schematic.png"
    chart_irrig_min_vwc    = report_dir / "chart_irrigation_min_vwc.png"
    chart_events_per_city  = report_dir / "chart_events_per_city.png"

    _irrig_charts_fresh = (_chart_fresh(chart_irrig_overview) and
                           _chart_fresh(chart_irrig_pie) and
                           _chart_fresh(chart_irrig_seasonal) and
                           _chart_fresh(chart_irrig_schematic) and
                           _chart_fresh(chart_irrig_min_vwc))

    if irrig_csv.exists():
        # Always read event counts for the PDF narrative (cheap — one column only)
        _ev_types = pd.read_csv(irrig_csv, usecols=["eui", "event_type"],
                                dtype={"eui": str})
        n_ev               = len(_ev_types)
        n_irr_ev           = int((_ev_types["event_type"] == "IRRIGATION").sum())
        n_irr_strong_ev    = int((_ev_types["event_type"] == "IRRIGATION_STRONG").sum())
        n_rain_all_ev      = int((_ev_types["event_type"] == "RAIN_ALL").sum())
        n_head_ev          = int((_ev_types["event_type"] == "RAIN_HEAD").sum())
        n_tail_ev          = int((_ev_types["event_type"] == "RAIN_TAIL").sum())
        n_head_imp_ev      = int((_ev_types["event_type"] == "RAIN_HEAD_IMPACT").sum())
        n_rain_strong_ev   = int((_ev_types["event_type"] == "RAIN_STRONG").sum())
        n_tail_dry_ev      = int((_ev_types["event_type"] == "RAIN_TAIL_DRY").sum())
        n_full_ev          = int((_ev_types["event_type"] == "RAIN_FULL").sum())
        n_trees_w          = _ev_types["eui"].nunique()

        # City-level event distribution chart (uses only the cheap _ev_types read)
        if not _chart_fresh(chart_events_per_city):
            _make_events_per_city(_ev_types, gdf, chart_events_per_city)

    if irrig_csv.exists() and not _irrig_charts_fresh:
        ev = pd.read_csv(irrig_csv, parse_dates=["event_time"],
                         dtype={"eui": str}, low_memory=False)
        n_ev               = len(ev)
        n_irr_ev           = int((ev["event_type"] == "IRRIGATION").sum())
        n_irr_strong_ev    = int((ev["event_type"] == "IRRIGATION_STRONG").sum())
        n_rain_all_ev      = int((ev["event_type"] == "RAIN_ALL").sum())
        n_head_ev          = int((ev["event_type"] == "RAIN_HEAD").sum())
        n_tail_ev          = int((ev["event_type"] == "RAIN_TAIL").sum())
        n_head_imp_ev      = int((ev["event_type"] == "RAIN_HEAD_IMPACT").sum())
        n_rain_strong_ev   = int((ev["event_type"] == "RAIN_STRONG").sum())
        n_tail_dry_ev      = int((ev["event_type"] == "RAIN_TAIL_DRY").sum())
        n_full_ev          = int((ev["event_type"] == "RAIN_FULL").sum())
        n_trees_w          = ev["eui"].nunique()

        if "month" not in ev.columns and "event_time" in ev.columns:
            ev["month"] = pd.to_datetime(ev["event_time"]).dt.month

        dry_col = "dry_h-10" if "dry_h-10" in ev.columns else None

        m_irrig         = ev["event_type"] == "IRRIGATION"
        m_irrig_strong  = ev["event_type"] == "IRRIGATION_STRONG"
        m_rain_all      = ev["event_type"] == "RAIN_ALL"
        m_head          = ev["event_type"] == "RAIN_HEAD"
        m_tail         = ev["event_type"] == "RAIN_TAIL"
        m_head_imp     = ev["event_type"] == "RAIN_HEAD_IMPACT"
        m_rain_strong  = ev["event_type"] == "RAIN_STRONG"
        m_tail_dry     = ev["event_type"] == "RAIN_TAIL_DRY"
        m_full         = ev["event_type"] == "RAIN_FULL"

        # ── Pie + summary: separate, large, readable figure ────────────────
        n_unique_ev = n_irr_ev + n_rain_all_ev   # unique = top-level categories only
        _pct = lambda v: v / n_unique_ev * 100 if n_unique_ev > 0 else 0

        from matplotlib.patches import Patch as _Patch
        fig_pie, (ax_pie, ax_tbl) = plt.subplots(1, 2, figsize=(18, 9),
                                                   gridspec_kw={"wspace": 0.08})
        fig_pie.suptitle(
            f"Event Type Distribution — {n_unique_ev:,} unique events across {n_trees_w} trees\n"
            f"(IRRIGATION + RAIN_ALL are the root categories; all other types are nested subsets)",
            fontsize=12, fontweight="bold")

        # Pie — no text on slices, use legend instead
        pie_vals = [n_irr_ev, n_irr_strong_ev, n_rain_all_ev, n_head_ev, n_tail_ev,
                    n_head_imp_ev, n_rain_strong_ev, n_tail_dry_ev, n_full_ev]
        pie_clrs = [CLR_IRRIG, CLR_IRRIG_STRONG, CLR_RAIN_ALL, CLR_HEAD, CLR_TAIL,
                    CLR_HEAD_IMP, CLR_RAIN_STRONG, CLR_TAIL_DRY, CLR_FULL]
        pie_names = ["IRRIGATION", "IRRIGATION_STRONG", "RAIN_ALL", "RAIN_HEAD",
                     "RAIN_TAIL", "RAIN_HEAD_IMPACT", "RAIN_STRONG",
                     "RAIN_TAIL_DRY", "RAIN_FULL"]
        non_zero = [(v, n, c) for v, n, c in zip(pie_vals, pie_names, pie_clrs) if v > 0]
        if non_zero:
            _pv, _pn, _pc = zip(*non_zero)
            wedges, _ = ax_pie.pie(_pv, colors=_pc, startangle=90,
                                   wedgeprops=dict(edgecolor="white", linewidth=1.2))
            ax_pie.legend(
                wedges,
                [f"{n}  {v:,}  ({v/_pv[0]*100 if i==0 else v/_pv[2]*100 if i>=2 else v/_pv[0]*100:.0f}%)"
                 if False else f"{n} — {v:,}"
                 for v, n in zip(_pv, _pn)],
                loc="lower center", bbox_to_anchor=(0.5, -0.18),
                fontsize=10, ncol=2, framealpha=0.9,
            )
        ax_pie.set_title("Proportions by event count", fontsize=11)

        # Summary table in right panel
        ax_tbl.axis("off")
        rows = [
            ("── Top-level (unique events) ──────────────────────────", "", ""),
            ("IRRIGATION",        f"{n_irr_ev:>7,}", f"{_pct(n_irr_ev):5.1f} % of total"),
            ("RAIN_ALL",          f"{n_rain_all_ev:>7,}", f"{_pct(n_rain_all_ev):5.1f} % of total"),
            ("Total unique",      f"{n_unique_ev:>7,}", f"trees: {n_trees_w}"),
            ("── Subsets (% shown relative to parent) ───────────────", "", ""),
            ("IRRIGATION_STRONG", f"{n_irr_strong_ev:>7,}",
             f"{n_irr_strong_ev/max(n_irr_ev,1)*100:.0f} % of IRRIGATION"),
            ("RAIN_HEAD",         f"{n_head_ev:>7,}",
             f"{n_head_ev/max(n_rain_all_ev,1)*100:.0f} % of RAIN_ALL"),
            ("RAIN_TAIL",         f"{n_tail_ev:>7,}",
             f"{n_tail_ev/max(n_rain_all_ev,1)*100:.0f} % of RAIN_ALL"),
            ("RAIN_HEAD_IMPACT",  f"{n_head_imp_ev:>7,}",
             f"{n_head_imp_ev/max(n_head_ev,1)*100:.0f} % of RAIN_HEAD"),
            ("RAIN_STRONG",       f"{n_rain_strong_ev:>7,}",
             f"{n_rain_strong_ev/max(n_head_imp_ev,1)*100:.0f} % of HEAD_IMPACT"),
            ("RAIN_TAIL_DRY",     f"{n_tail_dry_ev:>7,}",
             f"{n_tail_dry_ev/max(n_tail_ev,1)*100:.0f} % of RAIN_TAIL"),
            ("RAIN_FULL",         f"{n_full_ev:>7,}",
             f"{n_full_ev/max(n_head_imp_ev,1)*100:.0f} % of HEAD_IMPACT"),
        ]
        _clr_map = {
            "IRRIGATION": CLR_IRRIG, "IRRIGATION_STRONG": CLR_IRRIG_STRONG,
            "RAIN_ALL": CLR_RAIN_ALL, "RAIN_HEAD": CLR_HEAD, "RAIN_TAIL": CLR_TAIL,
            "RAIN_HEAD_IMPACT": CLR_HEAD_IMP, "RAIN_STRONG": CLR_RAIN_STRONG,
            "RAIN_TAIL_DRY": CLR_TAIL_DRY, "RAIN_FULL": CLR_FULL,
            "Total unique": "#333333",
        }
        y_start = 0.97
        row_h   = 0.072
        for i, (name, count, pct) in enumerate(rows):
            y = y_start - i * row_h
            is_header = name.startswith("──")
            fc = _clr_map.get(name, None)
            fw = "bold" if (not is_header and fc is not None) else "normal"
            fi = "normal"
            fs = 10 if not is_header else 8.5
            tc = fc if fc else ("#555" if is_header else "#222")
            ax_tbl.text(0.03, y, name,  transform=ax_tbl.transAxes,
                        fontsize=fs, fontweight=fw, color=tc, va="top", family="monospace")
            ax_tbl.text(0.60, y, count, transform=ax_tbl.transAxes,
                        fontsize=fs, fontweight=fw, color=tc, va="top",
                        ha="right", family="monospace")
            ax_tbl.text(0.63, y, pct,   transform=ax_tbl.transAxes,
                        fontsize=fs, color="#444", va="top", family="monospace")

        fig_pie.savefig(chart_irrig_pie, dpi=150, bbox_inches="tight")
        plt.close(fig_pie)

        # ── Overview histogram chart: 3 rows × 6 columns (no pie row) ──────
        fig_ov, axes_ov = plt.subplots(3, 6, figsize=(28, 13),
                                        gridspec_kw={"hspace": 0.55, "wspace": 0.38})
        fig_ov.suptitle("Irrigation & Rain Event Statistics v2 — Metric Distributions",
                         fontsize=12, fontweight="bold")

        # Histogram rows — six key types
        _groups = [
            (m_rain_all,    "RAIN_ALL",         CLR_RAIN_ALL,     n_rain_all_ev),
            (m_head_imp,    "RAIN_HEAD_IMPACT", CLR_HEAD_IMP,     n_head_imp_ev),
            (m_rain_strong, "RAIN_STRONG",      CLR_RAIN_STRONG,  n_rain_strong_ev),
            (m_tail_dry,    "RAIN_TAIL_DRY",    CLR_TAIL_DRY,     n_tail_dry_ev),
            (m_full,        "RAIN_FULL",        CLR_FULL,         n_full_ev),
            (m_irrig_strong,"IRRIG_STRONG",     CLR_IRRIG_STRONG, n_irr_strong_ev),
        ]

        def _std_hist_row(row_i, col_name, x_label, bins_per_col):
            """Draw a standard single-series histogram row using per-column bins."""
            for col_i, (mask, lbl, clr, cnt) in enumerate(_groups):
                ax  = axes_ov[row_i, col_i]
                bins = bins_per_col[col_i]
                vals = ev.loc[mask, col_name].dropna() if col_name in ev.columns else pd.Series()
                if not vals.empty:
                    ax.hist(vals, bins=bins, color=clr, alpha=0.85, edgecolor="white")
                    med = vals.median()
                    ax.axvline(med, color="black", lw=1.3, linestyle="--",
                               label=f"med {med:.1f}")
                    ax.legend(fontsize=7, loc="upper right")
                ax.set_title(f"{lbl}\n(n = {cnt:,})", fontsize=8,
                             color=clr, fontweight="bold")
                ax.set_xlabel(x_label, fontsize=7)
                ax.set_ylabel("Events" if col_i == 0 else "", fontsize=7)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.3)

        # ── Row 1: Precipitation ─────────────────────────────────────────
        # v2 stores the rain sum in column "rain_pre24h-10" (per depth) or
        # via the onset-time total.  Use "rain_pre72h-10" if 24h not present.
        rain_col_v2 = ("rain_pre24h-10" if "rain_pre24h-10" in ev.columns
                       else "rain_post24h-10" if "rain_post24h-10" in ev.columns
                       else None)
        if rain_col_v2 is None:
            for _c in range(6):
                axes_ov[0, _c].set_visible(False)
        else:
            rain_sl = pd.concat([
                ev.loc[m_rain_all, rain_col_v2],
                ev.loc[m_short,    rain_col_v2],
                ev.loc[m_impact,   rain_col_v2],
            ]).dropna()
            r_hi = float(rain_sl.quantile(0.99)) + 1 if not rain_sl.empty else 35.0
            bins_rain_shared = np.linspace(0, r_hi, 27)
            bins_rain_irrig  = np.linspace(0, 10.0, 22)   # capped at 10 mm
            _std_hist_row(1, rain_col_v2,
                          "Precipitation pre-peak 24 h (mm)",
                          [bins_rain_irrig, bins_rain_shared,
                           bins_rain_shared, bins_rain_shared])

        # ── Row 2: ΔVWC — one grouped bar cluster per bin, 3 depths side by side
        DEPTH_COLS_DV = ["delta_vwc-10", "delta_vwc-30", "delta_vwc-45"]
        DEPTH_LBL_DV  = ["-10 cm", "-30 cm", "-45 cm"]
        DEPTH_CLR_DV  = ["#1b7837", "#762a83", "#e08214"]
        dv_avail = [c for c in DEPTH_COLS_DV if c in ev.columns]

        if dv_avail:
            # Shared x-bins from all depths + all event types
            dv_all  = pd.concat([ev[c] for c in dv_avail]).dropna()
            dv_lo   = max(0.0, float(dv_all.quantile(0.01)))
            dv_hi   = float(dv_all.quantile(0.99))
            bins_dv = np.linspace(dv_lo, dv_hi + (dv_hi - dv_lo) * 0.02, 27)
            bw      = bins_dv[1] - bins_dv[0]
            n_d     = len(dv_avail)
            bar_w   = bw / n_d * 0.88
            d_off   = np.linspace(-(n_d - 1) / 2, (n_d - 1) / 2, n_d) * (bw / n_d)
            centers = (bins_dv[:-1] + bins_dv[1:]) / 2

            for col_i, (mask, lbl, clr, cnt) in enumerate(_groups):
                ax = axes_ov[1, col_i]
                all_heights = []

                for d_i, (dcol, dlbl, dclr) in enumerate(zip(dv_avail,
                                                              DEPTH_LBL_DV,
                                                              DEPTH_CLR_DV)):
                    vals = ev.loc[mask, dcol].dropna()
                    if vals.empty:
                        continue
                    counts, _ = np.histogram(vals, bins=bins_dv)
                    ax.bar(centers + d_off[d_i], counts, width=bar_w,
                           color=dclr, alpha=0.85, label=dlbl, edgecolor="white")
                    all_heights.extend(counts.tolist())


                ax.set_title(f"{lbl}\n(n = {cnt:,})", fontsize=8,
                             color=clr, fontweight="bold")
                ax.set_xlabel("ΔVWC at onset (%-pts)", fontsize=7)
                ax.set_ylabel("Events" if col_i == 0 else "", fontsize=7)
                ax.legend(fontsize=6, loc="upper right")
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.3)
        else:
            for c in range(6):
                axes_ov[1, c].set_visible(False)

        # ── Row 3: Drying time — grouped bars per depth (same pattern as row 2)
        DRY_COLS = ["dry_h-10", "dry_h-30", "dry_h-45"]
        DRY_LBL  = ["-10 cm", "-30 cm", "-45 cm"]
        dr_avail = [c for c in DRY_COLS if c in ev.columns]

        if dr_avail:
            dr_all  = pd.concat([ev[c] for c in dr_avail]).dropna()
            dr_hi   = min(float(dr_all.quantile(0.97)), 7 * 24) if not dr_all.empty else 168
            bins_dr = np.linspace(0, dr_hi, 27)
            bw_dr   = bins_dr[1] - bins_dr[0]
            n_dr    = len(dr_avail)
            bar_w_dr  = bw_dr / n_dr * 0.88
            d_off_dr  = np.linspace(-(n_dr - 1) / 2, (n_dr - 1) / 2, n_dr) * (bw_dr / n_dr)
            centers_dr = (bins_dr[:-1] + bins_dr[1:]) / 2

            for col_i, (mask, lbl, clr, cnt) in enumerate(_groups):
                ax = axes_ov[2, col_i]
                for d_i, (dcol, dlbl, dclr) in enumerate(zip(dr_avail, DRY_LBL, DEPTH_CLR_DV)):
                    vals = ev.loc[mask, dcol].dropna()
                    if vals.empty:
                        continue
                    counts, _ = np.histogram(vals, bins=bins_dr)
                    ax.bar(centers_dr + d_off_dr[d_i], counts, width=bar_w_dr,
                           color=dclr, alpha=0.85, label=dlbl, edgecolor="white")
                ax.set_title(f"{lbl}\n(n = {cnt:,})", fontsize=8,
                             color=clr, fontweight="bold")
                ax.set_xlabel("Drying time (hours)", fontsize=7)
                ax.set_ylabel("Events" if col_i == 0 else "", fontsize=7)
                ax.legend(fontsize=6, loc="upper right")
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.3)
        else:
            for c in range(6):
                axes_ov[2, c].set_visible(False)

        fig_ov.savefig(chart_irrig_overview, dpi=150, bbox_inches="tight")
        plt.close(fig_ov)

        # ── Seasonal chart (2 rows) ───────────────────────────────────────
        fig_s, (ax_cnt, ax_drybox) = plt.subplots(2, 1, figsize=(14, 9),
                                                    gridspec_kw={"hspace": 0.4})
        fig_s.suptitle("Seasonal Patterns — Irrigation & Rain Events",
                        fontsize=12, fontweight="bold")

        # Top: six rain types (left axis) + irrigation (right axis)
        months = list(range(1, 13))
        x_pos  = np.arange(12)
        bw     = 0.13
        _season_groups = [
            (m_rain_all, CLR_RAIN_ALL, "Rain all"),
            (m_head,     CLR_HEAD,     "Rain head"),
            (m_tail,     CLR_TAIL,     "Rain tail"),
            (m_head_imp, CLR_HEAD_IMP, "Head impact"),
            (m_tail_dry, CLR_TAIL_DRY, "Tail dry"),
            (m_full,     CLR_FULL,     "Full"),
        ]
        cnt_irr = np.array([int((ev.loc[m_irrig, "month"] == m).sum()) for m in months])
        for g_i, (mask, clr, lbl) in enumerate(_season_groups):
            cnt = np.array([int((ev.loc[mask, "month"] == m).sum()) for m in months])
            offset = (g_i - (len(_season_groups) - 1) / 2) * bw
            ax_cnt.bar(x_pos + offset, cnt, width=bw, color=clr, alpha=0.85, label=lbl)

        ax_cnt2 = ax_cnt.twinx()
        ax_cnt2.bar(x_pos + (len(_season_groups) - 0.5) * bw / 2,
                    cnt_irr, width=bw, color=CLR_IRRIG, alpha=0.75, label="Irrigation")
        ax_cnt2.set_ylabel("Irrigation count (right axis)", color=CLR_IRRIG, fontsize=8)
        ax_cnt2.tick_params(axis="y", labelcolor=CLR_IRRIG, labelsize=7)
        ax_cnt2.set_ylim(0, max(cnt_irr.max() * 4, 1))

        ax_cnt.set_xticks(x_pos)
        ax_cnt.set_xticklabels(MONTH_ABBR, fontsize=9)
        ax_cnt.set_ylabel("Event count (rain types, left axis)", fontsize=8)
        ax_cnt.set_title(
            "Events per calendar month — rain hierarchy left axis, irrigation right axis",
            fontsize=10)
        ax_cnt.legend(fontsize=7.5, loc="upper left", ncol=3)
        ax_cnt.grid(axis="y", alpha=0.3)

        # Bottom: drying time boxplot — key types side by side
        if dry_col:
            box_types = [
                (m_rain_all, CLR_RAIN_ALL, "steelblue", "Rain all"),
                (m_head_imp, CLR_HEAD_IMP, "#155724",   "Head impact"),
                (m_tail_dry, CLR_TAIL_DRY, "#7a4800",   "Tail dry"),
                (m_full,     CLR_FULL,     "#001a3d",   "Full"),
            ]
            offset = 0.22
            for i, (mask, clr, med_clr, lbl) in enumerate(box_types):
                pos_off = (i - 1.5) * offset
                box_data = [ev.loc[mask & (ev["month"] == m), dry_col].dropna().values
                            for m in months]
                ax_drybox.boxplot(
                    box_data, positions=x_pos + pos_off,
                    widths=0.18, patch_artist=True,
                    boxprops=dict(facecolor=clr, alpha=0.6),
                    medianprops=dict(color=med_clr, lw=1.5),
                    whiskerprops=dict(color=clr), capprops=dict(color=clr),
                    flierprops=dict(marker=".", color=clr, markersize=3, alpha=0.4),
                    manage_ticks=False,
                )
            ax_drybox.legend(handles=[
                _Patch(facecolor=c, alpha=0.6, label=l)
                for _, c, _, l in box_types
            ], fontsize=9)
        ax_drybox.set_xticks(x_pos)
        ax_drybox.set_xticklabels(MONTH_ABBR, fontsize=9)
        ax_drybox.set_ylabel("Drying time at −10 cm (hours)")
        ax_drybox.set_title("Drying time by month (−10 cm depth)", fontsize=10)
        ax_drybox.grid(axis="y", alpha=0.3)

        fig_s.savefig(chart_irrig_seasonal, dpi=150, bbox_inches="tight")
        plt.close(fig_s)

        # ── Extended statistics chart ─────────────────────────────────────
        chart_irrig_ext = report_dir / "chart_irrigation_extended.png"
        if not _chart_fresh(chart_irrig_ext):
            ev["_hour"]  = pd.to_datetime(ev["event_time"]).dt.hour
            ev["_month"] = pd.to_datetime(ev["event_time"]).dt.month

            dyn_cols   = ["delta_pct_dyn-10", "delta_pct_dyn-30", "delta_pct_dyn-45"]
            dyn_avail  = [c for c in dyn_cols if c in ev.columns]
            if dyn_avail:
                ev["delta_pct_dyn_mean"] = ev[dyn_avail].mean(axis=1, skipna=True)

            # Groups for boxplots: only the meaningful rain subsets
            bp_groups = [
                (m_rain_all, "RAIN\nALL",  CLR_RAIN_ALL),
                (m_head,     "RAIN\nHEAD", CLR_HEAD),
                (m_tail,     "RAIN\nTAIL", CLR_TAIL),
                (m_head_imp, "HEAD\nIMP",  CLR_HEAD_IMP),
                (m_tail_dry, "TAIL\nDRY",  CLR_TAIL_DRY),
                (m_full,     "FULL",        CLR_FULL),
            ]

            fig_ext, axes_ext = plt.subplots(2, 3, figsize=(18, 10),
                                              gridspec_kw={"hspace": 0.50, "wspace": 0.38})
            fig_ext.suptitle("Rain Event Distribution Statistics by Type",
                              fontsize=12, fontweight="bold")

            def _bp(ax, col, title, ylabel, clamp_q=0.99):
                data   = [ev.loc[mask, col].dropna().values
                          for mask, *_ in bp_groups]
                labels = [lbl for _, lbl, _ in bp_groups]
                clrs   = [clr for _, _, clr in bp_groups]
                bp_    = ax.boxplot(data, patch_artist=True, widths=0.55,
                                    medianprops=dict(color="black", lw=2),
                                    flierprops=dict(marker=".", markersize=2,
                                                    alpha=0.3))
                for patch, clr in zip(bp_["boxes"], clrs):
                    patch.set_facecolor(clr); patch.set_alpha(0.6)
                for whisker, clr in zip(bp_["whiskers"],
                                        [c for c in clrs for _ in range(2)]):
                    whisker.set_color(clr)
                for cap, clr in zip(bp_["caps"],
                                    [c for c in clrs for _ in range(2)]):
                    cap.set_color(clr)
                # Clamp y-axis to ignore extreme outliers
                all_vals = ev[[col]].dropna()[col]
                if not all_vals.empty:
                    ax.set_ylim(bottom=max(0, all_vals.quantile(0.01)),
                                top=all_vals.quantile(clamp_q) * 1.1)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_title(title, fontsize=9, fontweight="bold")
                ax.set_ylabel(ylabel, fontsize=8)
                ax.grid(axis="y", alpha=0.3)

            # (0,0) Hour-of-day distribution
            ax_hr = axes_ext[0, 0]
            for mask, lbl, clr in bp_groups:
                hrs = ev.loc[mask, "_hour"].dropna().values
                if len(hrs):
                    ax_hr.hist(hrs, bins=24, range=(0, 24), color=clr,
                               alpha=0.5, density=True,
                               label=lbl.replace("\n", " "))
            ax_hr.set_xlabel("Hour of day (UTC)")
            ax_hr.set_ylabel("Density")
            ax_hr.set_xticks(range(0, 24, 3))
            ax_hr.set_title("Onset hour of day", fontsize=9, fontweight="bold")
            ax_hr.legend(fontsize=6.5, ncol=2)
            ax_hr.grid(alpha=0.3)

            # (0,1) Month-of-year distribution (normalised)
            ax_mo = axes_ext[0, 1]
            for mask, lbl, clr in bp_groups:
                mos = ev.loc[mask, "_month"].dropna().values
                if len(mos):
                    ax_mo.hist(mos, bins=12, range=(1, 13), color=clr,
                               alpha=0.5, density=True,
                               label=lbl.replace("\n", " "))
            ax_mo.set_xticks(range(1, 13))
            ax_mo.set_xticklabels(MONTH_ABBR, fontsize=7)
            ax_mo.set_ylabel("Density")
            ax_mo.set_title("Onset month", fontsize=9, fontweight="bold")
            ax_mo.legend(fontsize=6.5, ncol=2)
            ax_mo.grid(alpha=0.3)

            # (0,2) ΔVWC absolute at −10 cm
            if "delta_vwc-10" in ev.columns:
                _bp(axes_ext[0, 2], "delta_vwc-10",
                    "ΔVWC at −10 cm (absolute)", "ΔVWC (%-pts)")

            # (1,0) ΔVWC as % of dynamic range (mean across depths)
            if "delta_pct_dyn_mean" in ev.columns:
                _bp(axes_ext[1, 0], "delta_pct_dyn_mean",
                    "ΔVWC as % of dynamic range\n(mean −10/−30/−45 cm)",
                    "% of dyn. range", clamp_q=0.97)

            # (1,1) Drying time at −10 cm
            if dry_col:
                _bp(axes_ext[1, 1], dry_col,
                    "Drying time at −10 cm", "Hours", clamp_q=0.95)

            # (1,2) Scatter: ΔVWC vs drying time (−10 cm)
            ax_sc = axes_ext[1, 2]
            if "delta_vwc-10" in ev.columns and dry_col:
                for mask, lbl, clr in bp_groups:
                    sub_sc = ev.loc[mask, ["delta_vwc-10", dry_col]].dropna()
                    if not sub_sc.empty:
                        ax_sc.scatter(sub_sc["delta_vwc-10"], sub_sc[dry_col],
                                      color=clr, s=8, alpha=0.4,
                                      label=lbl.replace("\n", " "))
            ax_sc.set_xlabel("ΔVWC at −10 cm (%-pts)", fontsize=8)
            ax_sc.set_ylabel("Drying time at −10 cm (h)", fontsize=8)
            ax_sc.set_title("ΔVWC vs Drying time", fontsize=9, fontweight="bold")
            ax_sc.legend(fontsize=6.5, ncol=2)
            ax_sc.grid(alpha=0.25)

            fig_ext.savefig(chart_irrig_ext, dpi=150, bbox_inches="tight")
            plt.close(fig_ext)

        # ── Classification schematic ──────────────────────────────────────
        if not _chart_fresh(chart_irrig_schematic):
            _build_schematic(chart_irrig_schematic)

        # ── Irrigation minimum VWC change distribution ────────────────────
        if not _chart_fresh(chart_irrig_min_vwc):
            dv_cols  = ["delta_vwc-10",      "delta_vwc-30",      "delta_vwc-45"]
            dpc_cols = ["delta_pct_dyn-10",  "delta_pct_dyn-30",  "delta_pct_dyn-45"]
            ev_irr   = ev[m_irrig].copy()

            # Per-event minimum across all three depths
            dv_avail  = [c for c in dv_cols  if c in ev_irr.columns]
            dpc_avail = [c for c in dpc_cols if c in ev_irr.columns]

            if dv_avail:
                ev_irr["_min_abs"] = ev_irr[dv_avail].min(axis=1)
            if dpc_avail:
                ev_irr["_min_dyn"] = ev_irr[dpc_avail].min(axis=1)

            fig_mv, axes_mv = plt.subplots(1, 2, figsize=(14, 5),
                                            gridspec_kw={"wspace": 0.35})
            fig_mv.suptitle(
                "Minimum VWC Change Across All Three Depths During Irrigation Events\n"
                "(per event: min of −10 cm / −30 cm / −45 cm)",
                fontsize=11, fontweight="bold")

            if "_min_abs" in ev_irr.columns:
                vals = ev_irr["_min_abs"].dropna()
                x_hi = float(vals.quantile(0.98))   # right end from data
                x_lo = -10.0                         # fixed left end as requested
                bins_abs = np.linspace(x_lo, x_hi, 42)
                axes_mv[0].hist(vals.clip(lower=x_lo, upper=x_hi),
                                bins=bins_abs, color=CLR_IRRIG,
                                alpha=0.80, edgecolor="white")
                axes_mv[0].set_xlim(x_lo, x_hi)
                med = vals.median()
                axes_mv[0].axvline(med, color="black", lw=1.5, linestyle="--",
                                   label=f"Median: {med:.2f} %-pts")
                axes_mv[0].axvline(1.0, color="#888", lw=1.2, linestyle=":",
                                   label="1 %-pt threshold")
                axes_mv[0].set_xlabel("Minimum ΔVWC across depths (%-pts)", fontsize=9)
                axes_mv[0].set_ylabel("Number of irrigation events", fontsize=9)
                axes_mv[0].set_title(
                    f"Absolute change — n={len(vals):,}\n"
                    f"med={med:.2f}  p25={float(vals.quantile(.25)):.2f}  "
                    f"p75={float(vals.quantile(.75)):.2f} %-pts",
                    fontsize=9)
                axes_mv[0].legend(fontsize=8)
                axes_mv[0].grid(alpha=0.3)

            if "_min_dyn" in ev_irr.columns:
                vals = ev_irr["_min_dyn"].dropna()
                x_hi_dyn = float(vals.quantile(0.98))   # right end from data
                x_lo_dyn = -10.0                         # fixed left end
                bins_dyn = np.linspace(x_lo_dyn, x_hi_dyn, 42)
                axes_mv[1].hist(vals.clip(lower=x_lo_dyn, upper=x_hi_dyn),
                                bins=bins_dyn, color=CLR_IRRIG_STRONG,
                                alpha=0.80, edgecolor="white")
                axes_mv[1].set_xlim(x_lo_dyn, x_hi_dyn)
                med = vals.median()
                axes_mv[1].axvline(med, color="black", lw=1.5, linestyle="--",
                                   label=f"Median: {med:.1f} % dyn")
                axes_mv[1].axvline(5.0, color="#888", lw=1.2, linestyle=":",
                                   label="5 % dyn threshold")
                axes_mv[1].axvline(68.0, color=CLR_IRRIG_STRONG, lw=1.2,
                                   linestyle=":", label="68 % dyn (IRRIG_STRONG, ≈1σ)")
                axes_mv[1].set_xlabel("Minimum ΔVWC as % of sensor dynamic range",
                                      fontsize=9)
                axes_mv[1].set_ylabel("Number of irrigation events", fontsize=9)
                axes_mv[1].set_title(
                    f"Relative to dynamic range — n={len(vals):,}\n"
                    f"med={med:.1f}  p25={float(vals.quantile(.25)):.1f}  "
                    f"p75={float(vals.quantile(.75)):.1f} %",
                    fontsize=9)
                axes_mv[1].legend(fontsize=8)
                axes_mv[1].grid(alpha=0.3)

            fig_mv.savefig(chart_irrig_min_vwc, dpi=150, bbox_inches="tight")
            plt.close(fig_mv)

    # ── Section 19 in PDF story ───────────────────────────────────────────
    story.append(Paragraph(
        "19 &ndash; Irrigation &amp; Rain Event Statistics",
        styles["SubHead"]))

    if not irrig_csv.exists():
        story.append(Paragraph(
            "No v2 irrigation event data found. Run <i>src/_poc_irrigation_v2.py</i> "
            "first to generate <i>TreeTabularData/irrigation_events_v2_all.csv</i>.",
            styles["Body"]))
        story.append(PageBreak())
    else:
        _n_uniq = n_irr_ev + n_rain_all_ev   # unique events (subsets not counted twice)
        _p = lambda v, base: f"{v/max(base,1)*100:.0f}"
        story.append(Paragraph(
            f"Detected <b>{_n_uniq:,} unique water input events</b> across "
            f"<b>{n_trees_w} trees</b> "
            f"(IRRIGATION + RAIN_ALL; all other types are subsets and are not counted "
            f"again in the total). "
            f"<b>IRRIGATION</b> {n_irr_ev:,} ({_p(n_irr_ev,_n_uniq)} % of total) — "
            f"simultaneous VWC rise ≥ 20 % of dyn. range in all three depths within 4 h. "
            f"<b>IRRIGATION_STRONG</b> {n_irr_strong_ev:,} ({_p(n_irr_strong_ev,n_irr_ev)} % of IRRIGATION) — "
            f"same but ≥ 80 % of dyn. range in all three depths. "
            f"<b>RAIN_ALL</b> {n_rain_all_ev:,} ({_p(n_rain_all_ev,_n_uniq)} % of total) — 24 h rolling sum ≥ 4 mm. "
            f"<b>RAIN_HEAD</b> {n_head_ev:,} ({_p(n_head_ev,n_rain_all_ev)} % of RAIN_ALL) — ≥ 68 % of 36 h pre-peak rain within ±12 h. "
            f"<b>RAIN_TAIL</b> {n_tail_ev:,} ({_p(n_tail_ev,n_rain_all_ev)} % of RAIN_ALL) — ≥ 68 % of 36 h post-peak rain within ±12 h. "
            f"<b>RAIN_HEAD_IMPACT</b> {n_head_imp_ev:,} ({_p(n_head_imp_ev,n_head_ev)} % of HEAD) — ΔVWC ≥ 5 % dyn. AND ≥ 1 %-pt in ≥ 1 depth. "
            f"<b>RAIN_STRONG</b> {n_rain_strong_ev:,} ({_p(n_rain_strong_ev,n_head_imp_ev)} % of HEAD_IMPACT) — same criteria in all three depths. "
            f"<b>RAIN_TAIL_DRY</b> {n_tail_dry_ev:,} ({_p(n_tail_dry_ev,n_tail_ev)} % of TAIL) — measurable drying return. "
            f"<b>RAIN_FULL</b> {n_full_ev:,} ({_p(n_full_ev,n_head_imp_ev)} % of HEAD_IMPACT) — intersection of HEAD_IMPACT and TAIL_DRY.",
            styles["Body"]))
        story.append(Spacer(1, 3*mm))

        chart_events_per_city = report_dir / "chart_events_per_city.png"
        if chart_events_per_city.exists():
            story.append(PageBreak())
            story.append(_proportional_image(chart_events_per_city))
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(
                "<b>Event distribution by city.</b> "
                "Left panel: absolute count of RAIN_ALL events (blue, upper bar) and "
                "IRRIGATION events (red, lower bar) per city. "
                "IRRIGATION_STRONG events (dark red) are a subset of IRRIGATION and "
                "shown as an overlay on the same bar — they are not added on top. "
                "The label &#39;n=&#39; annotates the number of sensors in each city. "
                "Right panel: same counts divided by the sensor count in that city, "
                "making cities with different numbers of sensors directly comparable. "
                "Y-axis labels include mean elevation above sea level (m a.s.l.) to "
                "contextualise systematic differences: low-elevation cities often have "
                "sandier, faster-draining soils; high-elevation cities may have "
                "higher soil water storage capacity, altering both the irrigation "
                "strategy and the magnitude of the VWC response. "
                "Cities are sorted by RAIN_ALL event count (most events at top).",
                styles["Note"]))
            story.append(Spacer(1, 4*mm))
            story.append(PageBreak())

        chart_irrig_pie = report_dir / "chart_irrigation_pie.png"
        if chart_irrig_pie.exists():
            story.append(_proportional_image(chart_irrig_pie))
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(
                "Pie chart showing event-type proportions. Slice colours match the "
                "legend labels. The summary table lists all nine types with their "
                "counts and the percentage relative to their parent category "
                "(e.g. RAIN_HEAD as % of RAIN_ALL, RAIN_STRONG as % of HEAD_IMPACT). "
                "IRRIGATION and RAIN_ALL are the two root categories; all others are "
                "subsets and should not be added to the total.",
                styles["Note"]))
            story.append(Spacer(1, 4*mm))

        if chart_irrig_overview.exists():
            story.append(_proportional_image(chart_irrig_overview))
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(
                "<b>Row 0:</b> event-type pie chart and count summary (four v2 types). "
                "<b>Rows 1–3:</b> one panel per event type "
                "(IRRIGATION / RAIN_ALL / RAIN_SHORT / RAIN_IMPACT), each with its "
                "own independent y-axis. Same x-axis bins within each row. "
                "Dashed line = median. "
                "Row 1: precipitation in the 24 h before the VWC peak; IRRIGATION "
                "capped at 10 mm (near-zero by definition). "
                "Row 2: ΔVWC as grouped depth bars (−10 / −30 / −45 cm, "
                "green / purple / orange). "
                "Row 3: drying time per depth (same grouped style).",
                styles["Note"]))
            story.append(Spacer(1, 4*mm))

        if chart_irrig_seasonal.exists():
            story.append(_proportional_image(chart_irrig_seasonal))
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(
                "<b>Top:</b> grouped monthly bars — RAIN_ALL / HEAD / TAIL / HEAD_IMPACT "
                "/ TAIL_DRY / FULL share the left axis; IRRIGATION is on a separate right "
                "axis (red). "
                "<b>Bottom:</b> side-by-side drying-time boxplots for the key event types. "
                "Box = IQR, whiskers = 1.5 × IQR, dots = outliers.",
                styles["Note"]))
        story.append(Spacer(1, 4*mm))

        chart_irrig_ext = report_dir / "chart_irrigation_extended.png"
        if chart_irrig_ext.exists():
            story.append(_proportional_image(chart_irrig_ext))
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(
                "Extended statistics across all six rain event types (colour-coded). "
                "<b>Top-left:</b> normalised density of onset hour-of-day (UTC) — "
                "reveals diurnal patterns in convective vs. stratiform events. "
                "<b>Top-centre:</b> onset month distribution — seasonality. "
                "<b>Top-right:</b> ΔVWC at −10 cm (absolute %-pts). "
                "<b>Bottom-left:</b> ΔVWC as a percentage of the sensor dynamic range "
                "(mean of −10/−30/−45 cm) — normalises for sensor-to-sensor variability. "
                "<b>Bottom-centre:</b> drying time at −10 cm (hours). "
                "<b>Bottom-right:</b> scatter of ΔVWC vs drying time at −10 cm. "
                "All boxplots show IQR / 1.5 × IQR whiskers, y-axis clamped at the "
                "99th percentile.",
                styles["Note"]))
        story.append(Spacer(1, 4*mm))

        chart_irrig_schematic = report_dir / "chart_irrigation_schematic.png"
        if chart_irrig_schematic.exists():
            story.append(_proportional_image(chart_irrig_schematic))
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(
                "<b>Left:</b> classification hierarchy — each box shows the event type, "
                "its definition, and the threshold applied.  Arrows indicate subset "
                "relationships (child ⊂ parent).  RAIN_FULL is the intersection of "
                "RAIN_HEAD_IMPACT and RAIN_TAIL_DRY. "
                "<b>Right:</b> annotated schematic timeseries showing where each metric "
                "is measured on a stylised rain event.  The ±12 h inner window and "
                "±36 h outer window (blue shading) define the concentration fractions "
                "frac_pre and frac_post.  The green double-headed arrow shows the "
                "ΔVWC measured from pre-event baseline to peak.  The orange arrow "
                "shows the drying time from the VWC peak back to baseline. "
                "Dynamic range bounds (p05 / p95, grey dotted lines) contextualise "
                "the relative thresholds used for IRRIGATION_STRONG and RAIN_STRONG.",
                styles["Note"]))
        story.append(Spacer(1, 4*mm))

        chart_irrig_min_vwc = report_dir / "chart_irrigation_min_vwc.png"
        if chart_irrig_min_vwc.exists():
            story.append(_proportional_image(chart_irrig_min_vwc))
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(
                "Distribution of the <b>minimum</b> VWC change recorded across all "
                "three measurement depths (−10/−30/−45 cm) for each individual "
                "irrigation event. Using the minimum rather than the mean or maximum "
                "shows the worst-case depth response and helps verify that the event "
                "was truly detected at all depths. "
                "<b>Left:</b> absolute change in %-pts — the dotted line at 1 %-pt "
                "marks the RAIN_HEAD_IMPACT threshold. "
                "<b>Right:</b> change expressed as a percentage of the sensor's "
                "full dynamic range (p05–p95) — the dotted line at 5 % marks the "
                "RAIN_STRONG / HEAD_IMPACT relative threshold; the coloured line at "
                "68 % marks the IRRIGATION_STRONG threshold (≈ 1σ, roughly two-thirds "
                "of the full dynamic range). "
                "Events below the 68 % line are IRRIGATION only; those above are "
                "IRRIGATION_STRONG, meaning all three depths showed a dominant "
                "soil-moisture response.",
                styles["Note"]))
        story.append(PageBreak())

    doc.build(story)
    print(f"\n  Report saved to: {pdf_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    project_dir = repo_root / "GISData"
    output_dir = repo_root / "TreeTabularData"

    if not project_dir.is_dir():
        print(f"ERROR: GIS data directory not found: {project_dir}")
        print(f"       Expected: ../GISData/ relative to this script")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  URBAN TREE DATASET PROCESSOR")
    print("=" * 65)

    print(f"\n[1/4] Scanning for shapefiles in: {project_dir}")
    city_shapefiles = discover_city_shapefiles(project_dir)
    if not city_shapefiles:
        print("  No treeLocations.shp files found. Exiting.")
        sys.exit(1)
    print(f"  Found {len(city_shapefiles)} city dataset(s).")

    print("\n[2/4] Loading tree data...")
    gdf = load_all_trees(city_shapefiles)

    _csv_out = output_dir / "all_tree_locations.csv"
    gdf.drop(columns="geometry").to_csv(_csv_out, index=False)
    print(f"  Exported combined tree table → {_csv_out.relative_to(repo_root)}")

    _sensor_csv_out = script_dir / "sensor_locations.csv"
    _export_sensor_csv(gdf, _sensor_csv_out)

    print("\n[3/4] Exporting per-tree JSON files...")
    json_stats = export_tree_jsons(gdf, output_dir)

    print("\n[4/4] Generating PDF report...")
    generate_report(gdf, output_dir, json_stats)

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
