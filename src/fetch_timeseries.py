#!/usr/bin/env python3
"""
Climavi Timeseries Fetcher for TreeDataBase
=============================================
Fetches ALL available sensor, weather, and device telemetry from the
Climavi API and stores per-tree CSVs + summary plots.

Folder structure:
    TreeDataBase/
    ├── GISData/
    ├── src/
    │   ├── fetch_timeseries.py          <-- this script
    │   └── api_key.txt                  <-- your Climavi API key
    └── TreeTabularData/
        ├── trees/{devEUI}/
        │   ├── sensor_data.csv
        │   ├── latest_attributes.json
        │   └── timeseries_plot.png
        └── weather_stations/{name}.csv

Usage:  cd src && python fetch_timeseries.py
Requires:  pip install geopandas pandas requests tqdm matplotlib
"""

import hashlib, json, os, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GIS_DIR = REPO_ROOT / "GISData"
OUTPUT_DIR = REPO_ROOT / "TreeTabularData"
API_KEY_PATH = SCRIPT_DIR / "api_key.txt"

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_BASE = "https://iot.climavi.eu:443"
ENTITIES_URL = f"{API_BASE}/api/entitiesQuery/find"
TELEMETRY_URL = f"{API_BASE}/api/plugins/telemetry/DEVICE/{{entityId}}/values/timeseries"
SHAPEFILE_NAME = "treeLocations.shp"
DEFAULT_FROM_DATE = "2020-01-01"

# Maximum distance (metres, projected CRS) within which two tree locations are
# considered the same physical tree for serial-replacement matching.
COORD_MATCH_THRESHOLD = 5.0

ALL_TS_KEYS = [
    "-10|ENV__SOIL__VWC", "-30|ENV__SOIL__VWC", "-45|ENV__SOIL__VWC",
    "-10|ENV__SOIL__T", "-30|ENV__SOIL__T", "-45|ENV__SOIL__T",
    "-10|ENV__SOIL__CAPACITANCE__ABSOLUTE",
    "-30|ENV__SOIL__CAPACITANCE__ABSOLUTE",
    "-45|ENV__SOIL__CAPACITANCE__ABSOLUTE",
    "TOP|ENV__ATMO__T", "TOP|ENV__ATMO__T__DRY", "TOP|ENV__ATMO__T__WET",
    "DOC|ENV__SOIL__IRRIGATION",
    "DEVICE|DEV__ENERGY__VBAT", "DEVICE|DEV__ENERGY__VCAP",
    "DEVICE|DEV__RF__RSSI", "DEVICE|DEV__RF__RSRP",
    "DEVICE|DEV__RF__RSRQ", "DEVICE|DEV__RF__SINR",
    "MTB|ENV__ATMO__T", "MTB|ENV__ATMO__RH", "MTB|ENV__ATMO__P",
    "MTB|ENV__ATMO__DEWPOINT", "MTB|ENV__ATMO__RAIN__DELTA",
    "MTB|ENV__ATMO__WIND__SPEED", "MTB|ENV__ATMO__WIND__DIRECTION",
    "MTB|ENV__SOIL__ET", "MTB|ENV__SOIL__T", "MTB|ENV__SOIL__VWC",
]

LATEST_ATTRIBUTE_KEYS = [
    "latitude", "longitude",
    "ENV__ATMO__RAIN__24h", "ENV__ATMO__RAIN__3d", "ENV__ATMO__RAIN__7d",
    "ENV__ATMO__RAIN__14d", "ENV__ATMO__RAIN__30d",
    "-10|ENV__SOIL__VWC__fc", "-30|ENV__SOIL__VWC__fc",
    "-45|ENV__SOIL__VWC__fc", "REC|ENV__SOIL__IRRIGATION",
]

# ---------------------------------------------------------------------------
# Serial replacement helpers
# ---------------------------------------------------------------------------
def _content_hash(data: dict) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


def _get_last_coords(json_path: Path) -> tuple[float, float] | None:
    """Return (geometry_x, geometry_y) from the latest snapshot, or None."""
    try:
        with json_path.open(encoding="utf-8") as f:
            history = json.load(f)
        latest = history[-1] if isinstance(history, list) else history
        x, y = latest.get("geometry_x"), latest.get("geometry_y")
        if x is not None and y is not None:
            return float(x), float(y)
    except Exception:
        pass
    return None


def _apply_serial_replacement(trees_dir: Path, new_eui: str, old_eui: str) -> None:
    """Rename or merge old_eui folder into new_eui and record the change.

    Two cases:
    - new folder does not exist: simple rename, then update tree_data.json.
    - new folder already exists (created by a prior API fetch before the rename
      could run) but has no tree_data.json: write the updated tree_data.json
      into the new folder and remove the old folder. The new folder's
      sensor_data.csv is kept (it contains the full history under the new EUI).
    """
    old_dir = trees_dir / old_eui
    new_dir = trees_dir / new_eui

    json_src = old_dir / "tree_data.json"
    if not json_src.exists():
        print(f"  [SERIAL] Warning: tree_data.json missing in {old_dir}")
        if not new_dir.exists():
            old_dir.rename(new_dir)
            print(f"  [SERIAL] Renamed folder (no tree_data.json): {old_eui}/ -> {new_eui}/")
        else:
            shutil.rmtree(old_dir)
            print(f"  [SERIAL] Removed orphan folder (no tree_data.json): {old_eui}/")
        return

    with json_src.open(encoding="utf-8") as f:
        history = json.load(f)
    if isinstance(history, dict):
        history = [history]

    latest = dict(history[-1])
    serial_history = list(latest.get("serial_history") or [])
    serial_history.append({
        "devEUI": old_eui,
        "replaced_on": datetime.now(timezone.utc).date().isoformat(),
    })
    new_snapshot = {k: v for k, v in latest.items() if not k.startswith("_")}
    new_snapshot["devEUI"] = new_eui
    new_snapshot["serial_history"] = serial_history
    new_snapshot["_content_hash"] = _content_hash(new_snapshot)
    new_snapshot["_snapshot_ts"] = datetime.now(timezone.utc).isoformat()
    history.append(new_snapshot)

    if new_dir.exists():
        # Folder was already created by a prior API fetch — merge into it.
        with (new_dir / "tree_data.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False, default=str)
        shutil.rmtree(old_dir)
        print(f"  [SERIAL] Merged: tree_data.json written to {new_eui}/, {old_eui}/ removed")
    else:
        old_dir.rename(new_dir)
        with (new_dir / "tree_data.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False, default=str)
        print(f"  [SERIAL] Renamed {old_eui}/ -> {new_eui}/, tree_data.json updated")


def detect_and_apply_serial_replacements(
    dev_euis_info: dict[str, dict], trees_dir: Path
) -> None:
    """Detect and apply sensor hardware replacements using spatial matching.

    For each devEUI that appears in the shapefile but has no folder yet, checks
    whether any existing folder whose devEUI is no longer in the shapefile lies
    within COORD_MATCH_THRESHOLD metres (projected CRS). If so, that folder is
    the predecessor: it is renamed to the new devEUI and tree_data.json is
    updated with a serial_history entry. The check is idempotent — once the new
    folder exists it is skipped on subsequent runs.
    """
    if not trees_dir.exists():
        return

    current_euis = set(dev_euis_info.keys())  # already upper-cased

    # Folders on disk whose devEUI is no longer in the shapefile (orphans)
    orphan_coords: dict[str, tuple[float, float]] = {}
    for d in trees_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name.upper() in current_euis:
            continue
        coords = _get_last_coords(d / "tree_data.json")
        if coords is not None:
            orphan_coords[d.name] = coords

    if not orphan_coords:
        return

    for new_eui, info in dev_euis_info.items():
        nx, ny = info.get("x"), info.get("y")
        if nx is None or ny is None:
            continue

        # Find the nearest orphan within the distance threshold
        best_eui, best_dist = None, float("inf")
        for old_eui, (ox, oy) in orphan_coords.items():
            dist = ((nx - ox) ** 2 + (ny - oy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best_eui = dist, old_eui

        if best_eui is None or best_dist > COORD_MATCH_THRESHOLD:
            continue

        new_json = trees_dir / new_eui / "tree_data.json"
        if new_json.exists():
            # tree_data.json already present — check whether the old EUI was
            # already recorded in serial_history (migration ran but left a
            # stale folder behind, e.g. after a Ctrl-C between the two steps).
            try:
                with new_json.open(encoding="utf-8") as f:
                    _h = json.load(f)
                _latest = _h[-1] if isinstance(_h, list) else _h
                _recorded = [e.get("devEUI") for e in (_latest.get("serial_history") or [])]
                if best_eui in _recorded:
                    shutil.rmtree(trees_dir / best_eui)
                    print(f"  [SERIAL] Removed stale folder: {best_eui}/ (already in {new_eui}/ serial_history)")
                    orphan_coords.pop(best_eui)
                    continue
            except Exception:
                pass
            # tree_data.json exists but serial_history not yet updated
            # (e.g. created by urban_tree_report.py before the rename) —
            # fall through to _apply_serial_replacement which will merge.

        _apply_serial_replacement(trees_dir, new_eui, best_eui)
        orphan_coords.pop(best_eui)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def load_api_key():
    if API_KEY_PATH.exists():
        k = API_KEY_PATH.read_text().strip()
        if k: return k
    return os.environ.get("CLIMAVI_API_KEY", "").strip() or None

def get_auth_headers():
    k = load_api_key()
    if not k:
        print(f"ERROR: No API key. Create {API_KEY_PATH}"); sys.exit(1)
    return {"Content-Type": "application/json",
            "X-Authorization": f"ApiKey {k}"}

# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------
def fetch_all_devices(headers):
    devs, page = [], 0
    while True:
        r = requests.post(ENTITIES_URL, headers=headers, json={
            "entityFilter": {"type":"entityType","resolveMultiple":True,
                             "entityType":"DEVICE"},
            "entityFields": [{"type":"ENTITY_FIELD","key":"name"},
                             {"type":"ENTITY_FIELD","key":"label"}],
            "latestValues": [{"type":"ATTRIBUTE","key":"latitude"},
                             {"type":"ATTRIBUTE","key":"longitude"}],
            "pageLink": {"page":page,"pageSize":500,
                         "sortOrder":{"key":{"key":"name",
                                             "type":"ENTITY_FIELD"},
                                      "direction":"ASC"}}})
        if r.status_code != 200: break
        d = r.json()
        for x in d.get("data",[]):
            f=x.get("latest",{}).get("ENTITY_FIELD",{})
            devs.append({"entityId":x["entityId"]["id"],
                         "nameId":f.get("name",{}).get("value",""),
                         "label":f.get("label",{}).get("value","")})
        if not d.get("hasNext"): break
        page += 1
    return devs

def build_eui_map(devices):
    m = {}
    for d in devices:
        eid = d["entityId"]
        for raw in (d["nameId"].upper().strip(), d["label"].upper().strip()):
            if not raw: continue
            m[raw] = eid
            if "_" in raw: m[raw.split("_")[-1]] = eid
            for pfx in ("CLIMAVI_","CLIMAVI","AGVO_","AGVO"):
                if raw.startswith(pfx): m[raw[len(pfx):]] = eid
    return m

# ---------------------------------------------------------------------------
# Latest attributes
# ---------------------------------------------------------------------------
def fetch_latest_attributes(headers, eids):
    results = {}
    for i in range(0,len(eids),50):
        batch = eids[i:i+50]
        r = requests.post(ENTITIES_URL, headers=headers, json={
            "entityFilter":{"type":"entityList","entityType":"DEVICE",
                            "entityList":batch},
            "entityFields":[{"type":"ENTITY_FIELD","key":"name"},
                            {"type":"ENTITY_FIELD","key":"label"}],
            "latestValues":([{"type":"ATTRIBUTE","key":k}
                             for k in LATEST_ATTRIBUTE_KEYS]
                            +[{"type":"TIME_SERIES","key":k}
                              for k in ALL_TS_KEYS]),
            "pageLink":{"page":0,"pageSize":50,
                        "sortOrder":{"key":{"key":"label",
                                            "type":"ENTITY_FIELD"},
                                     "direction":"ASC"}}})
        if r.status_code != 200: continue
        for dev in r.json().get("data",[]):
            eid = dev["entityId"]["id"]; lat = {}
            for cat in ("ATTRIBUTE","TIME_SERIES"):
                for k,o in dev.get("latest",{}).get(cat,{}).items():
                    if o.get("ts",0)>0 and o.get("value","")!="":
                        lat[k]={"value":o["value"],"ts":o["ts"]}
            results[eid]=lat
    return results

# ---------------------------------------------------------------------------
# Timeseries
# ---------------------------------------------------------------------------
def fetch_timeseries(headers, eid, keys, from_date):
    url = TELEMETRY_URL.format(entityId=eid)
    p = {"keys":",".join(keys),
         "startTs":int(datetime.strptime(from_date,"%Y-%m-%d").timestamp()*1000),
         "endTs":int(time.time()*1000),
         "orderBy":"ASC","useStrictDataTypes":True,"limit":100000000}
    r = requests.get(url, headers=headers, params=p)
    if r.status_code == 200: return r.json()
    if r.status_code == 429:
        tqdm.write("  [RATE LIMIT] 30 s..."); time.sleep(30)
        r = requests.get(url, headers=headers, params=p)
        if r.status_code == 200: return r.json()
    return None

def ts_to_df(dump):
    if not dump: return pd.DataFrame()
    dump = {k:v for k,v in dump.items() if v}
    if not dump: return pd.DataFrame()
    all_ts = set()
    for v in dump.values():
        for e in v: all_ts.add(e["ts"])
    lu = {k:{e["ts"]:e["value"] for e in v} for k,v in dump.items()}
    recs = []
    for ts in sorted(all_ts):
        rec = {"timestamp":ts,
               "datetime":datetime.fromtimestamp(ts/1000,tz=timezone.utc
                          ).strftime("%Y-%m-%d %H:%M:%S")}
        for k,l in lu.items(): rec[k]=l.get(ts)
        recs.append(rec)
    return pd.DataFrame(recs)

def fetch_and_save(headers, eid, keys, csv_path):
    from_date = DEFAULT_FROM_DATE
    df_ex = pd.DataFrame()
    if csv_path.exists():
        try:
            df_ex = pd.read_csv(csv_path)
            df_ex["datetime"] = pd.to_datetime(df_ex["datetime"])
            from_date = df_ex["datetime"].max().date().strftime("%Y-%m-%d")
        except: df_ex = pd.DataFrame()
    dump = fetch_timeseries(headers, eid, keys, from_date)
    if dump is None: return -1
    df_new = ts_to_df(dump)
    if df_new.empty and df_ex.empty: return 0
    if not df_ex.empty and not df_new.empty:
        df = pd.concat([df_ex, df_new]).drop_duplicates(subset="timestamp",
                                                         keep="last")
    elif not df_new.empty: df = df_new
    else: return len(df_ex)
    df.sort_values("timestamp", inplace=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return len(df)

# ---------------------------------------------------------------------------
# Shapefile discovery
# ---------------------------------------------------------------------------
def discover_dev_euis(gis_dir) -> dict[str, dict]:
    """Return {EUI_upper: {"project": str, "x": float|None, "y": float|None}}."""
    euis = {}
    for entry in sorted(gis_dir.iterdir()):
        if not entry.is_dir(): continue
        cands = [entry / SHAPEFILE_NAME]
        for sub in entry.iterdir():
            if sub.is_dir(): cands.append(sub / SHAPEFILE_NAME)
        for shp in cands:
            if shp.exists():
                gdf = gpd.read_file(shp)
                for _, row in gdf.iterrows():
                    eui = str(row.get("devEUI", "")).strip()
                    proj = str(row.get("project", entry.name)).strip()
                    if eui and eui.lower() != "nan":
                        geom = row.geometry
                        if geom is not None and not geom.is_empty:
                            c = geom.centroid
                            x, y = c.x, c.y
                        else:
                            x, y = None, None
                        euis[eui.upper()] = {"project": proj, "x": x, "y": y}
    return euis

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
C = {"10":"#2E86AB","30":"#F18F01","45":"#C73E1D",
     "mtb":"#7B2D8E","rain":"#4ECDC4","irrig":"#44BBA4",
     "bat":"#355070","et":"#A23B72",
     "rh":"#44BBA4","p":"#6D597A","wind":"#F18F01",
     "dew":"#8DB580","rssi":"#E94F37","sinr":"#2E86AB",
     "rsrp":"#F18F01","rsrq":"#C73E1D"}

GAP_THRESHOLD = np.timedelta64(7, "D")
LEGEND_FS = 10          # legend font size
LEGEND_MS = 2.5         # legend marker/line scale
LEGEND_HANDLELENGTH = 3 # wider colour swatch in legend
Y_PAD_FRAC = 0.18       # 18 % headroom above data for the legend


def _load(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    dt = df["datetime"].values
    cols = {}
    for c in df.columns:
        if c in ("timestamp","datetime"): continue
        v = pd.to_numeric(df[c], errors="coerce").values
        if np.any(np.isfinite(v)): cols[c] = v
    return dt, cols


def _break_gaps(dt, vals):
    """Insert NaN where gaps > 7 days so matplotlib breaks the line."""
    if len(dt) < 2: return dt, vals
    gaps = np.where(np.diff(dt) > GAP_THRESHOLD)[0]
    if len(gaps) == 0: return dt, vals
    parts_dt, parts_v = [], []
    prev = 0
    for gi in gaps:
        parts_dt.append(dt[prev:gi+1])
        parts_v.append(vals[prev:gi+1])
        mid = dt[gi] + (dt[gi+1] - dt[gi]) / 2
        parts_dt.append(np.array([mid]))
        parts_v.append(np.array([np.nan]))
        prev = gi + 1
    parts_dt.append(dt[prev:])
    parts_v.append(vals[prev:])
    return np.concatenate(parts_dt), np.concatenate(parts_v)


def _plot(ax, dt, vals, **kw):
    """Line plot with gap breaking."""
    m = np.isfinite(vals)
    if m.sum() == 0: return
    dt_c, v_c = np.array(dt)[m], vals[m]
    dt_g, v_g = _break_gaps(dt_c, v_c)
    ax.plot(dt_g, v_g, **kw)


def _pad_yaxis(ax, frac=Y_PAD_FRAC, bottom_frac=0.02):
    """Add headroom at the top of the y-axis so the legend sits above data."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    if span <= 0: return
    ax.set_ylim(lo - span * bottom_frac, hi + span * frac)


def _legend(ax, loc="upper left", extra_handles=None, extra_labels=None):
    """Place a single-row legend with large font and colour swatches."""
    h, l = ax.get_legend_handles_labels()
    if extra_handles:
        h = h + extra_handles
        l = l + (extra_labels or [])
    if not h: return
    ax.legend(h, l, fontsize=LEGEND_FS, markerscale=LEGEND_MS,
              handlelength=LEGEND_HANDLELENGTH,
              ncol=len(h), loc=loc,
              framealpha=0.85, edgecolor="#cccccc")


def _legend_twin(ax, ax2, loc="upper left"):
    """Combined single-row legend from two axes."""
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h, l = h1 + h2, l1 + l2
    if not h: return
    ax.legend(h, l, fontsize=LEGEND_FS, markerscale=LEGEND_MS,
              handlelength=LEGEND_HANDLELENGTH,
              ncol=len(h), loc=loc,
              framealpha=0.85, edgecolor="#cccccc")


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------
def generate_sensor_plot(csv_path, dev_eui, output_path):
    dt, cols = _load(csv_path)
    if not cols: return

    fig, axes = plt.subplots(6, 1, figsize=(20, 30), sharex=True)
    fig.suptitle(f"All timeseries data of sensor {dev_eui}",
                 fontsize=22, fontweight="bold", y=0.998)

    # ── 1: Soil Water Content + Rain + Irrigation ─────────────────────────
    ax = axes[0]
    for d, c in [("10",C["10"]),("30",C["30"]),("45",C["45"])]:
        k = f"-{d}|ENV__SOIL__VWC"
        if k in cols:
            _plot(ax, dt, cols[k], color=c, lw=0.8, alpha=0.85,
                  label=f"Sensor VWC {d} cm [%]")
    if "MTB|ENV__SOIL__VWC" in cols:
        _plot(ax, dt, cols["MTB|ENV__SOIL__VWC"], color=C["mtb"],
              lw=0.8, ls="--", alpha=0.7, label="Meteoblue soil VWC [%]")
    if "DOC|ENV__SOIL__IRRIGATION" in cols:
        irr = cols["DOC|ENV__SOIL__IRRIGATION"]; m = np.isfinite(irr)
        if m.sum() > 0:
            vwc_vals = [cols[k] for k in cols
                        if "ENV__SOIL__VWC" in k and "MTB" not in k
                        and "fc" not in k]
            y_top = np.nanmax(vwc_vals) if vwc_vals else 40
            ax.scatter(np.array(dt)[m], np.full(m.sum(), y_top * 0.97),
                       marker="v", c=C["irrig"], s=80, zorder=5,
                       label=f"Irrigation ({m.sum()} events)")
    ax.set_ylabel("VWC [%]")
    _pad_yaxis(ax)
    # Rain on twin axis — bars from bottom
    ax2 = ax.twinx()
    if "MTB|ENV__ATMO__RAIN__DELTA" in cols:
        rain = cols["MTB|ENV__ATMO__RAIN__DELTA"]; m = np.isfinite(rain)
        if m.sum() > 0:
            ax2.bar(np.array(dt)[m], rain[m], width=0.04,
                    color=C["rain"], alpha=0.5, label="Rain MTB [mm]")
            ax2.set_ylabel("Rain [mm]", color=C["rain"])
    _legend_twin(ax, ax2, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title("Soil Water Content, Precipitation & Irrigation Events",
                 fontsize=14, fontweight="bold")

    # ── 2: Raw Capacitance ────────────────────────────────────────────────
    ax = axes[1]
    for d, c in [("10",C["10"]),("30",C["30"]),("45",C["45"])]:
        k = f"-{d}|ENV__SOIL__CAPACITANCE__ABSOLUTE"
        if k in cols:
            _plot(ax, dt, cols[k], color=c, lw=0.7, alpha=0.8,
                  label=f"Capacitance {d} cm [pF]")
    ax.set_ylabel("Capacitance [pF]")
    _pad_yaxis(ax)
    _legend(ax)
    ax.grid(alpha=0.3)
    ax.set_title("Raw Soil Capacitance (sensor hardware diagnostic)",
                 fontsize=14, fontweight="bold")

    # ── 3: All Temperatures ───────────────────────────────────────────────
    ax = axes[2]
    for k, lab, c in [
        ("TOP|ENV__ATMO__T__DRY", "Sensor dry bulb", "#C73E1D"),
        ("TOP|ENV__ATMO__T__WET", "Sensor wet bulb", "#2E86AB"),
        ("MTB|ENV__ATMO__T", "Meteoblue air", C["mtb"]),
        ("MTB|ENV__ATMO__DEWPOINT", "Meteoblue dewpoint", C["dew"]),
    ]:
        if k in cols:
            _plot(ax, dt, cols[k], color=c, lw=0.6, alpha=0.7,
                  label=f"{lab} [°C]")
    if "TOP|ENV__ATMO__T__DRY" not in cols and "TOP|ENV__ATMO__T" in cols:
        _plot(ax, dt, cols["TOP|ENV__ATMO__T"], color="#C73E1D", lw=0.5,
              alpha=0.5, label="Sensor housing T [°C]")
    ax2 = ax.twinx()
    for d, c in [("10",C["10"]),("30",C["30"]),("45",C["45"])]:
        k = f"-{d}|ENV__SOIL__T"
        if k in cols:
            _plot(ax2, dt, cols[k], color=c, lw=0.6, ls=":", alpha=0.6,
                  label=f"Soil T {d} cm [°C]")
    if "MTB|ENV__SOIL__T" in cols:
        _plot(ax2, dt, cols["MTB|ENV__SOIL__T"], color=C["mtb"],
              lw=0.6, ls=":", alpha=0.6, label="MTB soil T [°C]")
    ax2.set_ylabel("Soil Temperature [°C]", fontsize=9)
    _pad_yaxis(ax); _pad_yaxis(ax2)
    _legend_twin(ax, ax2, loc="upper left")
    ax.set_ylabel("Air Temperature [°C]")
    ax.grid(alpha=0.3)
    ax.set_title("Air & Soil Temperature (sensor measurements + Meteoblue)",
                 fontsize=14, fontweight="bold")

    # ── 4: Wind Speed & Air Pressure ──────────────────────────────────────
    ax = axes[3]
    if "MTB|ENV__ATMO__WIND__SPEED" in cols:
        _plot(ax, dt, cols["MTB|ENV__ATMO__WIND__SPEED"],
              color=C["wind"], lw=0.6, alpha=0.7,
              label="Wind speed MTB [m/s]")
    ax.set_ylabel("Wind speed [m/s]", color=C["wind"])
    ax.tick_params(axis="y", labelcolor=C["wind"])
    _pad_yaxis(ax)
    ax2 = ax.twinx()
    if "MTB|ENV__ATMO__P" in cols:
        _plot(ax2, dt, cols["MTB|ENV__ATMO__P"], color=C["p"],
              lw=0.6, alpha=0.7, label="Pressure MTB [hPa]")
    ax2.set_ylabel("Pressure [hPa]", color=C["p"])
    ax2.tick_params(axis="y", labelcolor=C["p"])
    _pad_yaxis(ax2)
    _legend_twin(ax, ax2, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title("Wind Speed & Air Pressure (Meteoblue)",
                 fontsize=14, fontweight="bold")

    # ── 5: ET + Battery + Humidity ────────────────────────────────────────
    ax = axes[4]
    if "MTB|ENV__SOIL__ET" in cols:
        _plot(ax, dt, cols["MTB|ENV__SOIL__ET"], color=C["et"],
              lw=0.7, alpha=0.8, label="Evapotranspiration MTB [mm]")
    ax.set_ylabel("ET [mm]", color=C["et"])
    ax.tick_params(axis="y", labelcolor=C["et"])
    _pad_yaxis(ax)
    ax2 = ax.twinx()
    if "DEVICE|DEV__ENERGY__VCAP" in cols:
        _plot(ax2, dt, cols["DEVICE|DEV__ENERGY__VCAP"], color=C["bat"],
              lw=0.8, alpha=0.8, label="Battery charge [%]")
    if "DEVICE|DEV__ENERGY__VBAT" in cols:
        _plot(ax2, dt, cols["DEVICE|DEV__ENERGY__VBAT"], color="#1B998B",
              lw=0.8, ls="--", alpha=0.8, label="Battery voltage [mV]")
    if "MTB|ENV__ATMO__RH" in cols:
        _plot(ax2, dt, cols["MTB|ENV__ATMO__RH"], color=C["rh"],
              lw=0.5, alpha=0.5, label="Humidity MTB [%]")
    ax2.set_ylabel("Battery / Humidity [%]")
    _pad_yaxis(ax2)
    _legend_twin(ax, ax2, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title("Evapotranspiration, Battery Status & Relative Humidity",
                 fontsize=14, fontweight="bold")

    # ── 6: RF Signal Quality ──────────────────────────────────────────────
    ax = axes[5]
    rf_keys = [("DEVICE|DEV__RF__RSSI","RSSI [dB]",C["rssi"]),
               ("DEVICE|DEV__RF__RSRP","RSRP [dBm]",C["rsrp"]),
               ("DEVICE|DEV__RF__RSRQ","RSRQ [dB]",C["rsrq"])]
    has_rf = False
    for k, lab, c in rf_keys:
        if k in cols:
            _plot(ax, dt, cols[k], color=c, lw=0.6, alpha=0.8, label=lab)
            has_rf = True
    ax.set_ylabel("Signal strength [dB / dBm]")
    _pad_yaxis(ax)
    ax2 = ax.twinx()
    if "DEVICE|DEV__RF__SINR" in cols:
        _plot(ax2, dt, cols["DEVICE|DEV__RF__SINR"], color=C["sinr"],
              lw=0.6, alpha=0.8, label="SINR [dB]")
        ax2.set_ylabel("SINR [dB]", color=C["sinr"])
        _pad_yaxis(ax2)
        has_rf = True
    if has_rf:
        _legend_twin(ax, ax2, loc="upper left")
    else:
        ax.text(0.5, 0.5, "No RF signal data available for this sensor",
                ha="center", va="center", fontsize=14, color="grey",
                transform=ax.transAxes)
    ax.grid(alpha=0.3)
    ax.set_title("Data Connectivity & RF Signal Quality",
                 fontsize=14, fontweight="bold")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not GIS_DIR.is_dir():
        print(f"ERROR: GISData not found: {GIS_DIR}"); sys.exit(1)

    print("="*65)
    print("  CLIMAVI TIMESERIES FETCHER")
    print("="*65)

    trees_dir = OUTPUT_DIR / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/6] Scanning shapefiles in: {GIS_DIR}")
    dev_euis_info = discover_dev_euis(GIS_DIR)
    dev_euis = {eui: info["project"] for eui, info in dev_euis_info.items()}
    print(f"  Found {len(dev_euis)} unique devEUIs.")
    if not dev_euis: return
    detect_and_apply_serial_replacements(dev_euis_info, trees_dir)

    print("\n[2/6] Authenticating...")
    headers = get_auth_headers()
    print("  API key loaded.")

    print("\n[3/6] Fetching device registry...")
    all_devs = fetch_all_devices(headers)
    print(f"  {len(all_devs)} devices.")
    eui_map = build_eui_map(all_devs)
    ws = {d["label"]:d["entityId"] for d in all_devs
          if d["label"].lower().startswith("wetter")}
    print(f"  Weather stations: {len(ws)}")

    resolved, unresolved = {}, []
    for eui in dev_euis:
        eid = eui_map.get(eui.upper())
        if eid: resolved[eui] = eid
        else: unresolved.append(eui)
    print(f"  Resolved: {len(resolved)} / {len(dev_euis)}")
    for u in unresolved[:10]:
        print(f"    [UNRESOLVED] {u} ({dev_euis[u]})")

    print(f"\n[4/6] Fetching timeseries ({len(ALL_TS_KEYS)} keys)...")
    stats = {"fetched":0,"empty":0,"errors":0}
    for eui,eid in tqdm(resolved.items(), desc="  Sensors"):
        csv = trees_dir / eui / "sensor_data.csv"
        try:
            n = fetch_and_save(headers, eid, ALL_TS_KEYS, csv)
            if n>0: stats["fetched"]+=1
            elif n==0: stats["empty"]+=1
            else: stats["errors"]+=1
        except Exception as e:
            stats["errors"]+=1; tqdm.write(f"  [ERR] {eui}: {e}")
    print(f"\n  {stats['fetched']} fetched, {stats['empty']} empty, "
          f"{stats['errors']} errors")

    print("\n  Fetching latest attributes...")
    la = fetch_latest_attributes(headers, list(resolved.values()))
    e2u = {v:k for k,v in resolved.items()}
    saved = 0
    for eid,attrs in la.items():
        eui = e2u.get(eid)
        if not eui or not attrs: continue
        p = trees_dir / eui / "latest_attributes.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        attrs["_fetched_at"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S")
        attrs["_entityId"] = eid
        with open(p,"w",encoding="utf-8") as f:
            json.dump(attrs, f, indent=2, ensure_ascii=False)
        saved += 1
    print(f"  Saved latest_attributes.json for {saved} trees.")

    print(f"\n[5/6] Weather stations ({len(ws)})...")
    wd = OUTPUT_DIR / "weather_stations"; wd.mkdir(parents=True, exist_ok=True)
    for name, wid in tqdm(ws.items(), desc="  Weather"):
        csv = wd / f"{name}.csv"
        try:
            n = fetch_and_save(headers, wid, ALL_TS_KEYS, csv)
            tqdm.write(f"  {name}: {n} rows")
        except Exception as e: tqdm.write(f"  [ERR] {name}: {e}")

    print(f"\n[6/6] Generating timeseries plots...")
    plotted = 0
    for eui in tqdm(resolved.keys(), desc="  Plotting"):
        csv = trees_dir / eui / "sensor_data.csv"
        png = trees_dir / eui / "timeseries_plot.png"
        if csv.exists():
            try: generate_sensor_plot(csv, eui, png); plotted += 1
            except Exception as e: tqdm.write(f"  [PLOT ERR] {eui}: {e}")
    print(f"  Generated {plotted} plots.")

    print("\n" + "="*65 + "\n  DONE\n" + "="*65)

if __name__ == "__main__":
    main()
