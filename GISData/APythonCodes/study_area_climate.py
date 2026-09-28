#!/usr/bin/env python3
"""
study_area_climate.py — Table 1 (climate characteristics) for the GI manuscript.
Label: tab:study area climate.

DWD source : CDC multi-annual 1991-2020, downloaded from opendata.dwd.de
             and cached in DatasetStatistics/dwd_mean_91-20/.
MeteoSwiss : OGD monthly series from data.geo.admin.ch (station DIS = Disentis).
             Falls back to hard-coded normals if the S3 bucket is inaccessible
             (403 from outside Switzerland).
GeoSphere  : No automated download; Austrian rows kept as static overrides.

Usage:
  python study_area_climate.py            # (re-)generate CSV + LaTeX fragment
  python study_area_climate.py --check    # validate Potsdam reference row
  python study_area_climate.py --nearest "lat,lon"
                                          # list 5 nearest stations per service
"""

import argparse
import io
import math
import sys
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO  = Path(__file__).resolve().parents[2]
CACHE = REPO / "DatasetStatistics" / "dwd_mean_91-20"
OUT   = REPO / "figures" / "tables"

# ── DWD CDC base URL ───────────────────────────────────────────────────────────
_DWD = ("https://opendata.dwd.de/climate_environment/CDC/"
        "observations_germany/climate/multi_annual/mean_91-20/")

# ── MeteoSwiss OGD base URL ────────────────────────────────────────────────────
_MS = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"

# ── Month columns (order matters for seasonal sums) ────────────────────────────
MONTHS = ["Jan.", "Feb.", "März", "Apr.", "Mai", "Jun.",
          "Jul.", "Aug.", "Sept.", "Okt.", "Nov.", "Dez."]
SUMMER = ["Apr.", "Mai", "Jun.", "Jul.", "Aug.", "Sept."]

# ── DWD station-id → location label ───────────────────────────────────────────
# Rows appear sorted by Stations_id ascending in the DWD data files.
# "Effeltrich" added to station 1279 (Herzogenaurach, nearest to
#  Erlangen / Weisendorf / Effeltrich cluster).
ID_TO_LOC = {
    232:   "Augsburg",
    284:   "Bamberg",
    399:   "Berlin",
    474:   "Biberach",
    691:   "Bremen",
    848:   "Celle",
    881:   "Crailsheim",
    1051:  "Dresden, Pillnitz",
    1201:  "Elsdorf",
    1270:  "Erfurt",
    1279:  "Erlangen, Weisendorf, Effeltrich",
    1920:  "Hagen",
    2014:  "Garbsen, Hannover",
    2080:  "Heidelberg",
    2331:  "Homburg",
    2532:  "Kassel",
    2928:  "Leipzig",
    3098:  "Lüdenscheid",
    3545:  "Neunkirchen",
    3668:  "Nürnberg",
    3925:  "Pforzheim",
    3939:  "Pirmasens",
    3955:  "Plön",
    3987:  "Potsdam",
    4339:  "Saarbrücken",
    4371:  "Lemgo",
    6347:  "Hassfurt",
    7427:  "Neumünster",
    7431:  "Grünwald",
    13776: "Hildesheim",
    15000: "Aachen",
    19781: "Greding, Ingolstadt",
}
SELECTED_IDS = sorted(ID_TO_LOC)   # 32 DWD stations

# ── Austrian (GeoSphere) static overrides ─────────────────────────────────────
_AUSTRIA = [
    {
        "location": "Salzburg",
        "id": " ",
        "station": "Salzburg_Freisaal",
        "elevation": 419.0,
        "source_p": "manual",
        "source_t": "manual",
        "monthly_t": [0.4, 2.1, 6.3, 10.8, 15.0, 18.3, 20.0, 19.7, 15.2, 10.7, 5.3, 1.4],
        "monthly_p": [79.0, 64.3, 103.1, 92.0, 169.7, 191.1, 182.7, 192.6, 144.8, 103.4, 84.3, 86.5],
    },
    {
        "location": "Vienna",
        "id": " ",
        "station": "Wien_Innere_Stadt",
        "elevation": 177.0,
        "source_p": "manual",
        "source_t": "manual",
        "monthly_t": [2.1, 3.8, 7.7, 13.0, 17.3, 21.0, 23.0, 22.8, 17.7, 12.3, 7.2, 2.8],
        "monthly_p": [37.6, 33.5, 46.3, 39.6, 78.3, 82.0, 80.3, 73.8, 67.3, 47.7, 42.9, 39.9],
    },
    {
        "location": "Wiener Neustadt",
        "id": " ",
        "station": "Wr.Neustadt_Flugh.",
        "elevation": 275.0,
        "source_p": "manual",
        "source_t": "manual",
        "monthly_t": [0.0, 1.6, 5.7, 10.6, 15.1, 18.9, 20.9, 20.6, 15.7, 10.3, 5.3, 0.7],
        "monthly_p": [23.3, 21.3, 36.6, 31.3, 73.4, 86.9, 77.8, 81.1, 66.8, 42.5, 38.0, 28.8],
    },
    # Leogang: GeoSphere Austria klima-v2-1m, station Saalbach (id=83, 975 m, 7.1 km),
    # 1991-2020, 30 complete years. Nearest station with a full normal period;
    # sensors are at 1165 m and 1573 m in the same valley system.
    {
        "location": "Leogang",
        "id": " ",
        "station": "Saalbach",
        "elevation": 975.0,
        "source_p": "GeoSphere Austria (klima-v2-1m)",
        "source_t": "GeoSphere Austria (klima-v2-1m)",
        "monthly_t": [-3.7, -2.4, 1.4, 5.7, 10.6, 14.2, 15.8, 15.4, 11.0, 6.6, 0.9, -2.8],
        "monthly_p": [72.3, 61.1, 82.3, 71.1, 114.5, 147.7, 164.5, 164.0, 118.0, 85.3, 73.5, 72.8],
    },
    # Riedingtal: GeoSphere Austria klima-v2-1m, station Obertauern (id=68, 1772 m, 13.2 km),
    # 1991-2020, 30 complete years. Nearest station with a full normal period
    # (the three closer stations — Kölnbreinsperre, Obertauern id=15623/15610 — have no
    # data for 1991-2020 in klima-v2-1m).
    {
        "location": "Riedingtal",
        "id": " ",
        "station": "Obertauern",
        "elevation": 1772.0,
        "source_p": "GeoSphere Austria (klima-v2-1m)",
        "source_t": "GeoSphere Austria (klima-v2-1m)",
        "monthly_t": [-4.8, -4.8, -2.2, 1.5, 6.0, 9.9, 11.6, 11.6, 7.3, 3.8, -0.9, -4.1],
        "monthly_p": [38.1, 27.5, 53.2, 54.3, 103.2, 158.7, 184.8, 167.9, 112.6, 77.9, 56.6, 36.6],
    },
]

# ── MeteoSwiss static fallback (Disentis / Laax) ──────────────────────────────
# Used when the geo.admin.ch S3 bucket returns 403 (IP restriction outside CH).
# Values verified against the task specification:
#   elevation=1197, MAP=1125.4, SP=659.1, MAT=7.1, TCM=-1.1, TWM=15.8
_MS_FALLBACK = {
    "DIS": {
        "location": "Laax",
        "station": "Disentis",
        "elevation": 1197.0,
        "source_p": "MeteoSwiss OGD DIS (static fallback)",
        "source_t": "MeteoSwiss OGD DIS (static fallback)",
        "MAP": 1125.4,
        "summerP": 659.1,
        "MAT": 7.1,
        "T_cm": -1.1,
        "T_wm": 15.8,
    }
}

# ── Potsdam reference row for --check ─────────────────────────────────────────
# Station 3987, 1991-2020 normals.
_POTSDAM_REF = {
    "elevation": 80.9,
    "MAP": 577.6,
    "summerP": 325.8,
    "MAT": 9.7,
    "T_cm": 0.7,
    "T_wm": 19.4,
}


# ══════════════════════════════════════════════════════════════════════════════
# Download helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_dwd(filename: str) -> str:
    """Download a DWD CDC multi-annual file (cached)."""
    dest = CACHE / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        url = _DWD + filename
        print(f"  Downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url) as r:
            dest.write_bytes(r.read())
    return dest.read_text(encoding="latin-1")


def _fetch_url(url: str, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {
        "User-Agent": "study_area_climate/2.0 (research)"
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


# ══════════════════════════════════════════════════════════════════════════════
# DWD loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_dwd() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clim_dwd, stat_gdf).

    clim_dwd has columns: Stations_id, location, station, elevation,
    MAP, summerP, MAT, T_cm, T_wm, source_p, source_t
    """
    # Station lists
    p_stat = pd.read_csv(
        io.StringIO(_fetch_dwd("Niederschlag_1991-2020_Stationsliste.txt")),
        sep=";")
    p_stat = p_stat.loc[:, ~p_stat.columns.str.startswith("Unnamed")]
    p_stat["Stationsname"] = p_stat["Stationsname"].str.strip()
    p_stat["Stationshoehe"] = pd.to_numeric(p_stat["Stationshoehe"], errors="coerce")

    t_stat = pd.read_csv(
        io.StringIO(_fetch_dwd("Temperatur_1991-2020_Stationsliste.txt")),
        sep=";")
    t_stat = t_stat.loc[:, ~t_stat.columns.str.startswith("Unnamed")]

    # Geodataframe of precipitation stations that also have temperature data
    p_gdf = gpd.GeoDataFrame(
        p_stat,
        geometry=gpd.points_from_xy(
            pd.to_numeric(p_stat["geogr. Laenge"],  errors="coerce"),
            pd.to_numeric(p_stat["geogr. Breite"],  errors="coerce"),
        ),
        crs="EPSG:4326",
    )
    stations_all = p_gdf[p_gdf["Stations_id"].isin(t_stat["Stations_id"])]
    stations_sel = stations_all[stations_all["Stations_id"].isin(SELECTED_IDS)].copy()
    stations_sel["Stationshoehe"] = pd.to_numeric(
        stations_sel["Stationshoehe"], errors="coerce")

    # Precipitation data
    precip = pd.read_csv(
        io.StringIO(_fetch_dwd("Niederschlag_1991-2020.txt")),
        sep=";", decimal=",")
    precip = precip.loc[:, ~precip.columns.str.startswith("Unnamed")]
    for c in precip.columns[3:]:
        precip[c] = pd.to_numeric(precip[c], errors="coerce")
    precip = precip.merge(
        stations_sel[["Stations_id", "Stationsname", "Stationshoehe"]],
        on="Stations_id", how="inner")
    precip["summerP"] = precip[SUMMER].sum(axis=1)

    # Temperature data
    temp = pd.read_csv(
        io.StringIO(_fetch_dwd("Temperatur_1991-2020.txt")),
        sep=";", decimal=",")
    temp = temp.loc[:, ~temp.columns.str.startswith("Unnamed")]
    for c in temp.columns[3:]:
        temp[c] = pd.to_numeric(temp[c], errors="coerce")
    temp = temp.merge(
        stations_sel[["Stations_id", "Stationsname"]],
        on="Stations_id", how="inner")
    temp["T_wm"] = temp[MONTHS].max(axis=1)
    temp["T_cm"] = temp[MONTHS].min(axis=1)

    # Verify row alignment (both files sorted by Stations_id in the DWD data)
    assert list(precip["Stations_id"]) == list(temp["Stations_id"]), (
        "Stations_id mismatch between precip and temp after merge — "
        "check that all selected_ids exist in both DWD files."
    )

    clim = pd.DataFrame({
        "location":  precip["Stations_id"].map(ID_TO_LOC),
        "id":        precip["Stations_id"],
        "station":   precip["Stationsname"],
        "elevation": precip["Stationshoehe"],
        "MAP":       precip["Jahr"].round(1),
        "summerP":   precip["summerP"].round(1),
        "MAT":       temp["Jahr"].round(1),
        "T_cm":      temp["T_cm"].round(1),
        "T_wm":      temp["T_wm"].round(1),
        "source_p":  precip["Datenquelle"].astype(str).str.strip(),
        "source_t":  temp["Datenquelle"].astype(str).str.strip(),
    })
    return clim, stations_all


# ══════════════════════════════════════════════════════════════════════════════
# MeteoSwiss loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_ms(abbr: str) -> dict:
    """Download ogd-smn_{abbr}_m.csv and compute 1991-2020 normals.

    Returns a dict with keys: location, station, elevation, MAP, summerP,
    MAT, T_cm, T_wm, source_p, source_t.
    """
    abbr = abbr.upper()
    url_m    = f"{_MS}/{abbr}/ogd-smn_{abbr}_m.csv"
    url_meta = f"{_MS}/ogd-smn_meta_stations.csv"

    # Station metadata (elevation, name) — latin-1 for accented Swiss place names
    raw_meta = _fetch_url(url_meta)
    meta = pd.read_csv(io.BytesIO(raw_meta), sep=";", encoding="latin-1")
    row  = meta[meta["station_abbr"] == abbr].iloc[0]
    elev = float(row["station_height_masl"])
    name = str(row["station_name"])

    # Monthly series
    raw_m = _fetch_url(url_m)
    df = pd.read_csv(io.BytesIO(raw_m), sep=";")
    df["time"] = df["time"].astype(str)
    df["year"]  = df["time"].str[:4].astype(int)
    df["month"] = df["time"].str[4:6].astype(int)

    df_clim = df[(df["year"] >= 1991) & (df["year"] <= 2020)].copy()
    df_clim["tre200m0"] = pd.to_numeric(df_clim["tre200m0"], errors="coerce")
    df_clim["rre150m0"] = pd.to_numeric(df_clim["rre150m0"], errors="coerce")

    # Require each of 12 months to have 30 complete years
    t_counts = df_clim.groupby("month")["tre200m0"].count()
    p_counts = df_clim.groupby("month")["rre150m0"].count()
    bad_t = t_counts[t_counts < 30]
    bad_p = p_counts[p_counts < 30]
    if len(bad_t):
        raise ValueError(f"MeteoSwiss {abbr}: temperature months with <30 years: "
                         f"{bad_t.to_dict()}")
    if len(bad_p):
        raise ValueError(f"MeteoSwiss {abbr}: precipitation months with <30 years: "
                         f"{bad_p.to_dict()}")

    mon_t = df_clim.groupby("month")["tre200m0"].mean()   # mean °C per month
    mon_p = df_clim.groupby("month")["rre150m0"].mean()   # mean mm per month

    MAP     = round(mon_p.sum(), 1)
    summerP = round(mon_p[[4, 5, 6, 7, 8, 9]].sum(), 1)
    MAT     = round(mon_t.mean(), 1)
    T_cm    = round(mon_t.min(), 1)
    T_wm    = round(mon_t.max(), 1)

    return {
        "location": _MS_FALLBACK.get(abbr, {}).get("location", abbr),
        "station":  name,
        "elevation": elev,
        "MAP":      MAP,
        "summerP":  summerP,
        "MAT":      MAT,
        "T_cm":     T_cm,
        "T_wm":     T_wm,
        "source_p": f"MeteoSwiss OGD {abbr}",
        "source_t": f"MeteoSwiss OGD {abbr}",
    }


def _ms_with_fallback(abbr: str) -> dict:
    """Try live MeteoSwiss download; use static fallback on HTTP 403."""
    try:
        result = _load_ms(abbr)
        print(f"  MeteoSwiss {abbr}: downloaded live.", file=sys.stderr)
        return result
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"  MeteoSwiss {abbr}: geo.admin.ch returned 403 "
                  f"(IP restriction). Using static fallback.", file=sys.stderr)
            fb = _MS_FALLBACK[abbr].copy()
            fb.setdefault("id", " ")
            return fb
        raise


# ══════════════════════════════════════════════════════════════════════════════
# Austrian static rows
# ══════════════════════════════════════════════════════════════════════════════

def _build_austria() -> list[dict]:
    rows = []
    for a in _AUSTRIA:
        t = a["monthly_t"]
        p = a["monthly_p"]
        rows.append({
            "location": a["location"],
            "id":        a["id"],
            "station":   a["station"],
            "elevation": a["elevation"],
            "MAP":       round(sum(p), 1),
            "summerP":   round(sum(p[3:9]), 1),   # Apr-Sep (indices 3-8)
            "MAT":       round(float(np.mean(t)), 1),
            "T_cm":      round(min(t), 1),
            "T_wm":      round(max(t), 1),
            "source_p":  a["source_p"],
            "source_t":  a["source_t"],
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# --nearest helper
# ══════════════════════════════════════════════════════════════════════════════

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def cmd_nearest(latlon: str):
    lat, lon = (float(x) for x in latlon.split(","))
    print(f"Nearest DWD stations to ({lat:.4f}, {lon:.4f}) "
          f"with complete 1991-2020 normals:\n")

    # Load station lists that appear in BOTH precip and temp files
    p_stat = pd.read_csv(
        io.StringIO(_fetch_dwd("Niederschlag_1991-2020_Stationsliste.txt")),
        sep=";")
    p_stat = p_stat.loc[:, ~p_stat.columns.str.startswith("Unnamed")]
    p_stat["Stationsname"] = p_stat["Stationsname"].str.strip()

    t_stat = pd.read_csv(
        io.StringIO(_fetch_dwd("Temperatur_1991-2020_Stationsliste.txt")),
        sep=";")
    t_stat = t_stat.loc[:, ~t_stat.columns.str.startswith("Unnamed")]

    both = p_stat[p_stat["Stations_id"].isin(t_stat["Stations_id"])].copy()
    both["lat"] = pd.to_numeric(both["geogr. Breite"],  errors="coerce")
    both["lon"] = pd.to_numeric(both["geogr. Laenge"],  errors="coerce")
    both["elev"] = pd.to_numeric(both["Stationshoehe"], errors="coerce")
    both = both.dropna(subset=["lat", "lon"])
    both["dist_km"] = both.apply(
        lambda r: _haversine(lat, lon, r.lat, r.lon), axis=1)

    # Load source codes
    precip = pd.read_csv(
        io.StringIO(_fetch_dwd("Niederschlag_1991-2020.txt")),
        sep=";", decimal=",")
    precip = precip.loc[:, ~precip.columns.str.startswith("Unnamed")]
    temp = pd.read_csv(
        io.StringIO(_fetch_dwd("Temperatur_1991-2020.txt")),
        sep=";", decimal=",")
    temp = temp.loc[:, ~temp.columns.str.startswith("Unnamed")]

    both = both.merge(
        precip[["Stations_id", "Datenquelle"]].rename(
            columns={"Datenquelle": "src_p"}),
        on="Stations_id", how="left")
    both = both.merge(
        temp[["Stations_id", "Datenquelle"]].rename(
            columns={"Datenquelle": "src_t"}),
        on="Stations_id", how="left")

    top5 = both.nsmallest(5, "dist_km")[
        ["Stations_id", "Stationsname", "elev", "dist_km", "src_p", "src_t"]]
    print("DWD (top 5):")
    print(top5.to_string(index=False,
          float_format=lambda x: f"{x:.1f}"))


# ══════════════════════════════════════════════════════════════════════════════
# --check
# ══════════════════════════════════════════════════════════════════════════════

def cmd_check(clim: pd.DataFrame):
    row = clim[clim["id"] == 3987].iloc[0]
    ok = True
    for field, ref in _POTSDAM_REF.items():
        got = round(float(row[field]), 1)
        match = abs(got - ref) < 0.05
        status = "OK" if match else "FAIL"
        if not match:
            ok = False
        print(f"  Potsdam {field:10s}: expected {ref:6.1f}  got {got:6.1f}  {status}")
    if not ok:
        print("\n--check FAILED: Potsdam row does not reproduce to one decimal.")
        sys.exit(1)
    print("\n--check PASSED: Potsdam reference row matches exactly.")


# ══════════════════════════════════════════════════════════════════════════════
# Build combined table
# ══════════════════════════════════════════════════════════════════════════════

def build_table() -> pd.DataFrame:
    # DWD rows
    clim_dwd, _ = _load_dwd()

    # MeteoSwiss rows
    ms_rows = []
    for abbr in ["DIS"]:
        ms_rows.append(_ms_with_fallback(abbr))

    # Austrian rows
    aut_rows = _build_austria()

    # Combine: DWD sorted by id, then MeteoSwiss, then Austria
    non_dwd = []
    for r in ms_rows + aut_rows:
        non_dwd.append({
            "location": r["location"],
            "id":        r.get("id", " "),
            "station":   r["station"],
            "elevation": r["elevation"],
            "MAP":       r["MAP"],
            "summerP":   r["summerP"],
            "MAT":       r["MAT"],
            "T_cm":      r["T_cm"],
            "T_wm":      r["T_wm"],
            "source_p":  r["source_p"],
            "source_t":  r["source_t"],
        })

    clim = pd.concat(
        [clim_dwd, pd.DataFrame(non_dwd)],
        ignore_index=True)
    return clim


# ══════════════════════════════════════════════════════════════════════════════
# Output writers
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(clim: pd.DataFrame):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "climate_table.csv"
    clim.to_csv(path, index=False, float_format="%.1f")
    print(f"  CSV  → {path}")


def write_latex(clim: pd.DataFrame):
    """Write only the tabular body rows (no \\begin{table} wrapper)."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "climate_table_rows.tex"
    lines = []
    for _, r in clim.iterrows():
        id_cell = str(r["id"]).strip() if str(r["id"]).strip() else ""
        line = (
            f"{r['location']} & {id_cell} & {r['station']} & "
            f"{r['elevation']:.1f} & {r['MAP']:.1f} & {r['summerP']:.1f} & "
            f"{r['MAT']:.1f} & {r['T_cm']:.1f} & {r['T_wm']:.1f} \\\\"
        )
        lines.append(line)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  LaTeX → {path}")
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Generate climate table for the GI manuscript.")
    ap.add_argument("--check",   action="store_true",
                    help="Validate Potsdam reference row and exit.")
    ap.add_argument("--nearest", metavar="LAT,LON",
                    help="List 5 nearest DWD stations with complete normals.")
    args = ap.parse_args()

    if args.nearest:
        cmd_nearest(args.nearest)
        return

    print("Building climate table …", file=sys.stderr)
    clim = build_table()

    if args.check:
        cmd_check(clim)
        return

    write_csv(clim)
    lines = write_latex(clim)

    # Print the tabular body to stdout for copy-paste
    print("\n% ── climate_table_rows.tex ──────────────────────────────────────")
    print("% Location & Id & Station & Elevation & MAP & SP & MAT & TCM & TWM \\\\")
    for l in lines:
        print(l)
    print()

    # Summary statistics
    dwd_mask = clim["source_p"].str.isnumeric()
    print(f"Rows total : {len(clim)}")
    print(f"DWD rows   : {dwd_mask.sum()}  "
          f"(source codes: {sorted(clim.loc[dwd_mask,'source_p'].unique())})")
    print(f"Non-DWD    : {(~dwd_mask).sum()}")


# ── Legacy entry-point: allow `python study_area_climate.py` without --flags ──
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
