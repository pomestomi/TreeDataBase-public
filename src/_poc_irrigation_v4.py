#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Irrigation event detection — v4 (synchronized with v3 BSC rain method).

Changes vs v1 (_poc_irrigation.py):
  1. Rain detection: OLD rolling-sum peaks replaced by BSC rain event windows
     from bsc_events_all.csv (v3 Huff-quartile method).
     Rain label now carries huff_q + bsc columns instead of RAIN_SHORT/RAIN_LONG.
  2. VWC jump detection: CHANGED from all-3-depths to majority rule (>= 2 of 3 depths).
     _is_sustained likewise requires persistence at >= 2 depths.
  3. Mixed events: VWC jump inside a BSC rain window -> EXCLUDED (not written to CSV).
     BSC rain event overlapping a VWC jump -> also EXCLUDED.
     Only clean, unambiguous events reach the output CSV.
  4. Fallback for trees without BSC events: treat as no-rain
     (all VWC jumps -> IRRIGATION, no RAIN rows for that tree).

Output: irrigation_events_poc.csv  (same path, same base columns, adds huff_q/bsc/bsc_total_mm).
"""
import sys, io, os, hashlib, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT      = Path(__file__).resolve().parent.parent
TREES_DIR      = REPO_ROOT / "TreeTabularData" / "trees"
ALL_EVENTS_CSV = REPO_ROOT / "TreeTabularData" / "irrigation_events_all.csv"
BSC_EVENTS_CSV = REPO_ROOT / "TreeTabularData" / "bsc_events_all.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_WORKERS   = max(1, (os.cpu_count() or 4) - 1)
FORCE_REPROCESS = True   # always reprocess — algorithm changed

VWC_COLS = ["-10|ENV__SOIL__VWC", "-30|ENV__SOIL__VWC", "-45|ENV__SOIL__VWC"]
RAIN_COL = "MTB|ENV__ATMO__RAIN__DELTA"
TEMP_COL = "MTB|ENV__ATMO__T"

# VWC jump detection — CHANGED: majority rule
JUMP_THR        = 1.5   # VWC % / 30-min step
JUMP_MIN_DEPTHS = 2     # ← NEW: require >= this many depths (was 3 = all)
PERSIST_H       = 2.0   # hours post-jump for persistence check
PERSIST_DELTA   = 0.5   # VWC % minimum (post_median - pre_median)

# BSC rain window matching
BSC_RAIN_BUFFER_H = 0   # extra hours to extend BSC [ev_start, ev_end] in each direction
                          # 0 = use exact event window

# Combined event merging
EVENT_MERGE_H  = 12     # merge VWC events closer than this

# Metric extraction
PRE_WIN_H   = 2
PEAK_WIN_H  = 24
DRY_MAX_H   = 7 * 24
DRYING_FRAC = 0.10
PRE_RAIN_H  = 24
VWC_MIN_READINGS = 3

# VWC dynamic range
VWC_SMOOTH_STEPS = 8
VWC_DYN_MIN      = 2.0
VWC_DYN_MAX      = 50.0

# Fallback rain threshold for trees without BSC data
RAIN_THR_FALLBACK = 2.0   # mm / ±12 h: if no BSC events, use this to flag RAIN
RAIN_CLASS_H      = 12

DEPTH_LABELS = {"-10": "-10 cm", "-30": "-30 cm", "-45": "-45 cm"}
DEPTH_COLORS = {"-10": "#1b7837", "-30": "#762a83", "-45": "#e08214"}
PLOT_WIN_D   = 5

_HASH_FILENAME = ".params_hash_v4"


def _config_hash() -> str:
    params = {
        "JUMP_THR": JUMP_THR,
        "JUMP_MIN_DEPTHS": JUMP_MIN_DEPTHS,
        "PERSIST_H": PERSIST_H,
        "PERSIST_DELTA": PERSIST_DELTA,
        "BSC_RAIN_BUFFER_H": BSC_RAIN_BUFFER_H,
        "EVENT_MERGE_H": EVENT_MERGE_H,
        "VWC_MIN_READINGS": VWC_MIN_READINGS,
        "RAIN_THR_FALLBACK": RAIN_THR_FALLBACK,
        "RAIN_CLASS_H": RAIN_CLASS_H,
        "PRE_WIN_H": PRE_WIN_H,
        "PEAK_WIN_H": PEAK_WIN_H,
        "DRY_MAX_H": DRY_MAX_H,
        "DRYING_FRAC": DRYING_FRAC,
    }
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.md5(blob).hexdigest()


_CURRENT_HASH = _config_hash()

# ---------------------------------------------------------------------------
# Load BSC rain events once at import time (shared across workers)
# ---------------------------------------------------------------------------
def _load_bsc_index() -> dict:
    """Return {eui: [(ev_start, ev_end, huff_q, bsc, total_mm, duration_h), ...]}."""
    if not BSC_EVENTS_CSV.exists():
        print(f"[WARN] BSC events file not found: {BSC_EVENTS_CSV}", flush=True)
        return {}
    bsc = pd.read_csv(BSC_EVENTS_CSV,
                      usecols=["eui", "ev_start", "ev_end",
                               "huff_q", "bsc", "total_mm", "duration_h"],
                      parse_dates=["ev_start", "ev_end"])
    idx = {}
    for eui_val, grp in bsc.groupby("eui"):
        idx[str(eui_val)] = list(grp[["ev_start", "ev_end",
                                       "huff_q", "bsc",
                                       "total_mm", "duration_h"]].itertuples(index=False, name=None))
    return idx


_BSC_INDEX: dict = _load_bsc_index()
print(f"BSC index loaded: {len(_BSC_INDEX)} trees with rain events", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _merge_times(times, gap_h):
    merged = []
    for t in sorted(times):
        if not merged or (t - merged[-1]) > pd.Timedelta(hours=gap_h):
            merged.append(t)
    return merged


def _vwc_dynamic_range(rs30: pd.DataFrame) -> dict:
    result = {}
    for col in VWC_COLS:
        depth = col.split("|")[0]
        smoothed = (rs30[col]
                    .rolling(VWC_SMOOTH_STEPS, center=True,
                             min_periods=VWC_SMOOTH_STEPS // 2)
                    .median())
        valid = smoothed[(smoothed > VWC_DYN_MIN) & (smoothed < VWC_DYN_MAX)]
        result[f"vwc_min{depth}"] = round(float(valid.min()), 2) if not valid.empty else None
        result[f"vwc_max{depth}"] = round(float(valid.max()), 2) if not valid.empty else None
    return result


def _load_tree_attrs(eui: str) -> dict:
    import json as _json
    from datetime import date
    json_path = TREES_DIR / eui / "tree_data.json"
    if not json_path.exists():
        return {}
    try:
        snapshots = _json.loads(json_path.read_text(encoding="utf-8"))
        latest = snapshots[-1] if isinstance(snapshots, list) else snapshots
    except Exception:
        return {}
    skip = {"serial_history"}
    attrs = {f"tree_{k}": v for k, v in latest.items()
             if not k.startswith("_") and k not in skip}
    for date_key in ("tree_germDate", "tree_plantDate"):
        raw = attrs.get(date_key)
        if raw and str(raw) not in ("None", "nan", ""):
            try:
                d = pd.Timestamp(raw).date()
                attrs["tree_age_years"] = round((date.today() - d).days / 365.25, 1)
                break
            except Exception:
                pass
    return attrs


def _in_bsc_window(t: pd.Timestamp, bsc_windows: list) -> tuple:
    """Return (True, huff_q, bsc, total_mm, duration_h) if t inside any BSC window."""
    buf = pd.Timedelta(hours=BSC_RAIN_BUFFER_H)
    for ev_start, ev_end, huff_q, bsc, total_mm, duration_h in bsc_windows:
        if (ev_start - buf) <= t <= (ev_end + buf):
            return True, huff_q, bsc, total_mm, duration_h
    return False, None, None, None, None


# ---------------------------------------------------------------------------
# Per-tree processing
# ---------------------------------------------------------------------------
_EMPTY = (pd.DataFrame(), 0, 0, 0, 0)


def process_tree(eui: str) -> tuple:
    out_dir   = TREES_DIR / eui
    csv_path  = out_dir / "sensor_data.csv"
    event_dir = out_dir / "Irrigation Events"
    poc_csv   = event_dir / "irrigation_events_poc.csv"

    if not csv_path.exists():
        return _EMPTY

    hash_file = event_dir / _HASH_FILENAME
    if not FORCE_REPROCESS and poc_csv.exists() and hash_file.exists():
        if poc_csv.stat().st_mtime >= csv_path.stat().st_mtime:
            if hash_file.read_text().strip() == _CURRENT_HASH:
                try:
                    cached = pd.read_csv(poc_csv, parse_dates=["event_time"])
                    n_r = int((cached["label"] == "RAIN").sum())
                    n_i = int((cached["label"] == "IRRIGATION").sum())
                    return cached, len(cached), 0, n_r, n_i
                except Exception:
                    pass

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    missing_vwc = [c for c in VWC_COLS if c not in df.columns]
    if missing_vwc or RAIN_COL not in df.columns:
        return _EMPTY

    vwc_raw = df[["datetime"] + VWC_COLS].set_index("datetime").sort_index()
    rs30    = vwc_raw.resample("30min").mean().ffill(limit=2)

    met_cols = ["datetime", RAIN_COL] + ([TEMP_COL] if TEMP_COL in df.columns else [])
    met_raw  = df[met_cols].set_index("datetime").sort_index()
    rs1h     = met_raw.resample("1h").mean().ffill(limit=1)
    if TEMP_COL not in rs1h.columns:
        rs1h[TEMP_COL] = np.nan

    dyn_range  = _vwc_dynamic_range(rs30)
    tree_attrs = _load_tree_attrs(eui)

    # ── BSC rain windows for this tree ────────────────────────────────────────
    bsc_windows = _BSC_INDEX.get(eui, [])   # list of (ev_start, ev_end, huff_q, bsc, mm, dur_h)
    has_bsc     = bool(bsc_windows)

    # ── Path 1: VWC majority-jump + persistence ───────────────────────────────
    delta = rs30.diff()

    # CHANGE 2: count depths with jump >= JUMP_THR; require >= JUMP_MIN_DEPTHS
    jump_counts = sum(
        (delta[c].fillna(0) >= JUMP_THR).astype(int)
        for c in VWC_COLS
    )
    jump_mask = jump_counts >= JUMP_MIN_DEPTHS

    raw_jump_ev = rs30.index[jump_mask].tolist()
    merged_jump = _merge_times(raw_jump_ev, EVENT_MERGE_H)

    def _is_sustained(ev_t):
        pre_end  = ev_t - pd.Timedelta(minutes=31)
        pre_sl   = rs30.loc[pre_end - pd.Timedelta(hours=PRE_WIN_H) : pre_end]
        post_sl  = rs30.loc[ev_t : ev_t + pd.Timedelta(hours=PERSIST_H)]
        if pre_sl.empty or post_sl.empty:
            return False
        n_ok = 0
        for col in VWC_COLS:
            pre_med  = pre_sl[col].median()
            post_med = post_sl[col].median()
            if not (pd.isna(pre_med) or pd.isna(post_med)):
                if post_med - pre_med >= PERSIST_DELTA:
                    n_ok += 1
        return n_ok >= JUMP_MIN_DEPTHS  # CHANGE 2

    vwc_cands = [t for t in merged_jump if _is_sustained(t)]

    # ── CHANGE 1: classify VWC candidates vs BSC rain windows ─────────────────
    # A VWC jump inside a BSC rain window -> EXCLUDED (mixed signal)
    vwc_events  = []   # clean IRRIGATION events
    mixed_times = set()
    for t in vwc_cands:
        inside, *_ = _in_bsc_window(t, bsc_windows)
        if inside:
            mixed_times.add(t)   # remember: this BSC window is contaminated
        else:
            vwc_events.append(t)

    # ── Path 2: BSC rain events that are NOT contaminated by a VWC jump ───────
    bsc_rain_events = []   # (ev_start, huff_q, bsc, total_mm, duration_h)
    for ev_start, ev_end, huff_q, bsc, total_mm, duration_h in bsc_windows:
        # Check if any VWC jump overlaps this BSC window
        overlap = any(
            (ev_start - pd.Timedelta(hours=BSC_RAIN_BUFFER_H)) <= t <=
            (ev_end   + pd.Timedelta(hours=BSC_RAIN_BUFFER_H))
            for t in vwc_cands
        )
        if not overlap:
            bsc_rain_events.append((ev_start, huff_q, bsc, total_mm, duration_h))

    # ── Fallback for trees without BSC data ───────────────────────────────────
    fallback_rain_events = []
    if not has_bsc:
        # Use simple ±12h rain sum > RAIN_THR_FALLBACK to catch obvious rain
        # (conservative: only re-label VWC jumps, don't add separate rain-only rows)
        clean_vwc = []
        for t in vwc_events:
            t_lo = t - pd.Timedelta(hours=RAIN_CLASS_H)
            t_hi = t + pd.Timedelta(hours=RAIN_CLASS_H)
            rain_24h = rs1h.loc[t_lo:t_hi, RAIN_COL].dropna().sum()
            if rain_24h >= RAIN_THR_FALLBACK:
                fallback_rain_events.append((t, rain_24h))
            else:
                clean_vwc.append(t)
        vwc_events = clean_vwc

    def _has_vwc_data(ev_t):
        peak_end = ev_t + pd.Timedelta(hours=PEAK_WIN_H)
        n_ok = sum(
            1 for col in VWC_COLS
            if len(rs30.loc[ev_t:peak_end, col].dropna()) >= VWC_MIN_READINGS
        )
        return n_ok >= JUMP_MIN_DEPTHS

    vwc_events      = [t for t in vwc_events      if _has_vwc_data(t)]
    bsc_rain_times  = [t for t, *_ in bsc_rain_events if _has_vwc_data(t)]
    bsc_rain_meta   = {t: rest for (t, *rest) in bsc_rain_events if _has_vwc_data(t)}

    all_event_times = sorted(set(vwc_events) | set(bsc_rain_times))
    if not all_event_times:
        return _EMPTY

    # ── Metric extraction (shared) ────────────────────────────────────────────
    records = []
    for ev_t in all_event_times:
        is_irrig = ev_t in vwc_events
        meta     = bsc_rain_meta.get(ev_t)   # (huff_q, bsc, total_mm, duration_h) or None

        pre_end   = ev_t - pd.Timedelta(minutes=31)
        pre_start = pre_end - pd.Timedelta(hours=PRE_WIN_H)
        peak_end  = ev_t + pd.Timedelta(hours=PEAK_WIN_H)
        dry_end   = ev_t + pd.Timedelta(hours=DRY_MAX_H)

        # Rain context (for both types)
        t_lo     = ev_t - pd.Timedelta(hours=RAIN_CLASS_H)
        t_hi     = ev_t + pd.Timedelta(hours=RAIN_CLASS_H)
        rain_24h = rs1h.loc[t_lo:t_hi, RAIN_COL].dropna().sum()
        pre_rain = rs1h.loc[ev_t - pd.Timedelta(hours=PRE_RAIN_H) : ev_t,
                            RAIN_COL].dropna().sum()

        if is_irrig:
            label  = "IRRIGATION"
            source = "VWC"
            huff_q_val  = None
            bsc_val     = None
            total_mm_v  = None
            duration_h_v = None
        else:
            label  = "RAIN"
            source = "BSC_RAIN"
            if meta is not None:
                huff_q_val, bsc_val, total_mm_v, duration_h_v = meta
            else:
                huff_q_val = bsc_val = total_mm_v = duration_h_v = None

        row: dict = {
            "eui":           eui,
            "event_time":    ev_t,
            "label":         label,
            "source":        source,
            "rain_24h_mm":   round(float(rain_24h), 2),
            "pre_rain_mm":   round(float(pre_rain), 2),
            "month":         ev_t.month,
            "doy":           ev_t.day_of_year,
            "huff_q":        huff_q_val,
            "bsc":           bsc_val,
            "bsc_total_mm":  total_mm_v,
            "bsc_duration_h": duration_h_v,
        }

        t_peak_per_depth: dict = {}
        for col in VWC_COLS:
            depth = col.split("|")[0]
            pre_sl  = rs30.loc[pre_start:pre_end, col].dropna()
            pre_vwc = float(pre_sl.median()) if not pre_sl.empty else np.nan
            peak_sl = rs30.loc[ev_t:peak_end,  col].dropna()
            if peak_sl.empty:
                peak_vwc, t_peak = np.nan, pd.NaT
            else:
                t_peak   = peak_sl.idxmax()
                peak_vwc = float(peak_sl.max())
            t_peak_per_depth[depth] = t_peak
            delta_vwc = (peak_vwc - pre_vwc
                         if not (np.isnan(peak_vwc) or np.isnan(pre_vwc)) else np.nan)
            t_to_peak = ((t_peak - ev_t).total_seconds() / 60
                         if not pd.isna(t_peak) else np.nan)
            dry_thr = (pre_vwc + DRYING_FRAC * delta_vwc
                       if not np.isnan(delta_vwc) else np.nan)
            dry_sl = (rs30.loc[t_peak:dry_end, col].dropna()
                      if not pd.isna(t_peak) else pd.Series(dtype=float))
            t_dry = np.nan
            if not np.isnan(dry_thr) and not dry_sl.empty:
                crossed = dry_sl[dry_sl <= dry_thr]
                if not crossed.empty:
                    t_dry = crossed.index[0]
            drying_h = ((t_dry - t_peak).total_seconds() / 3600
                        if not pd.isna(t_dry) and not pd.isna(t_peak) else np.nan)
            if not pd.isna(t_peak) and not pd.isna(t_dry):
                temp_sum = float(rs1h.loc[t_peak:t_dry, TEMP_COL].dropna().sum())
                add_rain = float(rs1h.loc[t_peak:t_dry, RAIN_COL].dropna().sum())
            else:
                temp_sum = add_rain = np.nan

            def _r(v):
                return round(float(v), 2) if not (isinstance(v, float) and np.isnan(v)) else None

            row.update({
                f"pre_vwc{depth}":    _r(pre_vwc),
                f"peak_vwc{depth}":   _r(peak_vwc),
                f"delta_vwc{depth}":  _r(delta_vwc),
                f"t_peak_min{depth}": _r(t_to_peak),
                f"dry_h{depth}":      _r(drying_h),
                f"temp_sum{depth}":   _r(temp_sum),
                f"add_rain{depth}":   _r(add_rain),
            })

        records.append(row)

    result_df = pd.DataFrame(records)
    for k, v in dyn_range.items():
        result_df[k] = v
    for k, v in tree_attrs.items():
        result_df[k] = v

    event_dir.mkdir(exist_ok=True)
    result_df.to_csv(poc_csv, index=False)
    (event_dir / _HASH_FILENAME).write_text(_CURRENT_HASH)

    n_r = int((result_df["label"] == "RAIN").sum())
    n_i = int((result_df["label"] == "IRRIGATION").sum())
    return result_df, len(result_df), 0, n_r, n_i


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    euis = sorted(p.name for p in TREES_DIR.iterdir() if p.is_dir())
    print(f"Processing {len(euis)} trees with {MAX_WORKERS} workers …", flush=True)

    all_dfs   = []
    n_irrig   = n_rain = 0

    if MAX_WORKERS == 1:
        results = [process_tree(eui) for eui in tqdm(euis)]
    else:
        with Pool(MAX_WORKERS) as pool:
            results = list(tqdm(pool.imap(process_tree, euis), total=len(euis)))

    for df, _, _, nr, ni in results:
        if not df.empty:
            all_dfs.append(df)
            n_rain   += nr
            n_irrig  += ni

    print(f"\nDone. IRRIGATION events: {n_irrig}  RAIN events: {n_rain}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(ALL_EVENTS_CSV, index=False)
        print(f"Combined CSV: {ALL_EVENTS_CSV}  ({len(combined)} rows)")


if __name__ == "__main__":
    main()
