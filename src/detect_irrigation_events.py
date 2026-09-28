#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POC Irrigation / Rain event detection v2 — cleaner event hierarchy.

EVENT HIERARCHY (detected in this order of clarity):

  IRRIGATION   — simultaneous rise >= IRRIG_RISE_THR (20%) of the full
                 dynamic range of ALL 3 VWC channels within IRRIG_WINDOW_H
                 (4 h).  The clearest signal; not dependent on rain data.

  RAIN_ALL     — every distinct rain peak where the 24 h rolling
                 precipitation sum >= 4 mm.  Onset = hour of maximum
                 hourly intensity within the 24 h window.

  RAIN_HEAD        — subset of RAIN_ALL: >= 68 % of the 36 h PRE-peak
                     precipitation fell within the inner 12 h window
                     (concentrated onset / short head).

  RAIN_TAIL        — subset of RAIN_ALL: >= 68 % of the 36 h POST-peak
                     precipitation fell within the inner 12 h window
                     (sharp stop / short tail).

  RAIN_HEAD_IMPACT — subset of RAIN_HEAD: at least one VWC depth rises
                     >= 2 % of its full dynamic range within PEAK_WIN_H
                     of onset (measurable soil moisture uptake).

  RAIN_TAIL_DRY    — subset of RAIN_TAIL: at least one VWC depth shows
                     a valid drying return within DRY_MAX_H of the peak
                     (drying dynamics can be measured).

  RAIN_FULL        — intersection of RAIN_HEAD_IMPACT and RAIN_TAIL_DRY:
                     concentrated on both sides, clear uptake, and
                     measurable drying — the gold-standard event class.

DYNAMIC RANGE per channel:
  5th–95th percentile of the 4 h smoothed VWC timeseries, clipped to
  [VWC_DYN_MIN, VWC_DYN_MAX] to exclude sensor-gap artefacts.
  Stored on every event row so responses can be normalised.

Run on N_TREES trees first to verify detection quality.
Outputs CSVs only — no PNG generation at this stage.

Usage:
    cd src
    python _poc_irrigation_v2.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
_sys.path.insert(0, str(Path(__file__).resolve().parent / "OldCodeAttempts"))
try:
    from _event_plot import draw_event_figure
    _HAS_EVENT_PLOT = True
except ImportError:
    _HAS_EVENT_PLOT = False
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TREES_DIR      = Path(__file__).resolve().parent.parent / "TreeTabularData" / "trees"
ALL_EVENTS_CSV = Path(__file__).resolve().parent.parent / "TreeTabularData" / \
                 "irrigation_events_v2_all.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_WORKERS         = max(1, (os.cpu_count() or 4) - 1)
N_EXAMPLES          = 20   # random examples to plot per event type
GENERATE_EVENT_PLOTS = False  # set True to save one PNG per event (very slow)
EXAMPLES_DIR   = Path(__file__).resolve().parent.parent / "DatasetStatistics" / "rain_examples"

VWC_COLS = ["-10|ENV__SOIL__VWC", "-30|ENV__SOIL__VWC", "-45|ENV__SOIL__VWC"]
RAIN_COL = "MTB|ENV__ATMO__RAIN__DELTA"
TEMP_COL = "MTB|ENV__ATMO__T"

# Dynamic range
VWC_SMOOTH_STEPS = 8    # rolling-median window (30 min steps) = 4 h
VWC_DYN_MIN      = 2.0  # exclude readings ≤ this (sensor gaps / air)
VWC_DYN_MAX      = 50.0 # exclude readings ≥ this (saturation artefacts)

# Irrigation detection
IRRIG_RISE_THR  = 0.20  # min rise as fraction of dynamic range (all 3 channels)
IRRIG_WINDOW_H  = 4     # forward-looking window to find the rise
IRRIG_MERGE_H   = 12    # merge detections closer than this

# Rain detection
RAIN_MIN_24H       = 4.0  # mm: min 24 h rolling sum to qualify
RAIN_PEAK_MERGE_H  = 8    # hours: merge nearby 24 h-sum peaks
RAIN_MERGE_H       = 8    # hours: merge onset times
# Onset refinement
ONSET_LOOKBACK_H = 4    # hours before detected onset to search for earlier VWC rise
ONSET_REFINE_DYN = 0.05 # min VWC drop (fraction of dynamic range) to trigger refinement

RAIN_WIN_H_INNER   = 12   # hours each side: inner window (24 h total around peak)
RAIN_WIN_H_OUTER   = 36   # hours each side: outer window for concentration ratio
RAIN_HEAD_FRAC     = 0.68 # min fraction of 36 h pre-peak rain within inner 12 h
RAIN_TAIL_FRAC     = 0.68 # min fraction of 36 h post-peak rain within inner 12 h
RAIN_IMPACT_DYN    = 0.05 # min DVWC as fraction of dynamic range (5 %)
RAIN_IMPACT_ABS    = 1.0  # min DVWC in absolute %-pts — BOTH criteria must be met
IRRIG_EXCL_H       = 36   # hours: suppress rain events within this window of irrigation

# Stronger subclasses
IRRIG_STRONG_DYN   = 0.68 # IRRIGATION_STRONG: rise >= 68 % of dyn. range in ALL 3 depths (≈1σ)
# RAIN_STRONG reuses RAIN_IMPACT_DYN and RAIN_IMPACT_ABS but requires ALL 3 depths

# Minimum data coverage for RAIN_IMPACT classification.
# Both the 24 h before and the 24 h after the event onset must contain at
# least 2 actual transmissions.  Some sensors in the dataset transmit as
# infrequently as every 8 hours (~3 readings per 24 h); requiring 2 ensures
# at least one reading in each half of the window while still rejecting
# events where the sensor was completely offline for an entire day.
# Coverage is evaluated on the raw count grid (no forward-fill).
COVERAGE_WIN_H       = 24   # hours: coverage check window on each side of onset
COVERAGE_MIN_SAMPLES = 2    # minimum actual transmissions in each 24 h window

# Metric extraction
PRE_WIN_H    = 2       # hours before onset for baseline VWC
PEAK_WIN_H   = 24      # hours after onset to search for VWC peak
DRY_MAX_H    = 7 * 24  # max hours to search for drying return
DRYING_FRAC  = 0.10    # "dried" when VWC <= pre + frac * (peak - pre)

# Plotting
PLOT_WIN_D   = 5       # +-days around onset in each PNG
DEPTH_LABELS = {"-10": "-10 cm", "-30": "-30 cm", "-45": "-45 cm"}
DEPTH_COLORS = {"-10": "#1b7837", "-30": "#762a83", "-45": "#e08214"}
EVENT_LINE_COLOR = {
    "IRRIGATION":  "#D63030",
    "RAIN_ALL":    "#6BAED6",
    "RAIN_SHORT":  "#2E86AB",
    "RAIN_IMPACT": "#003f7f",
}


# ---------------------------------------------------------------------------
# Helpers shared across all trees
# ---------------------------------------------------------------------------
def _merge_times(times, gap_h):
    """Keep only the first timestamp within each gap_h cluster."""
    merged = []
    for t in sorted(times):
        if not merged or (t - merged[-1]) > pd.Timedelta(hours=gap_h):
            merged.append(t)
    return merged


def _find_local_peaks(series, min_val, min_gap_h):
    """Return timestamps of local maxima >= min_val with minimum spacing."""
    vals, idx, peaks = series.values, series.index, []
    for i in range(1, len(vals) - 1):
        if vals[i] >= min_val and vals[i] >= vals[i-1] and vals[i] > vals[i+1]:
            if not peaks or (idx[i] - peaks[-1]) > pd.Timedelta(hours=min_gap_h):
                peaks.append(idx[i])
            elif vals[i] > series.loc[peaks[-1]]:
                peaks[-1] = idx[i]
    return peaks


def _dynamic_range(rs30):
    """
    Compute 5th / 95th percentile (= dynamic range) per VWC channel.
    Uses a 4 h rolling median to suppress short-term spikes.
    Returns dict: {depth_key: {"p05": float, "p95": float, "dyn": float}}
    """
    dr = {}
    for col in VWC_COLS:
        depth = col.split("|")[0]
        smoothed = (rs30[col]
                    .rolling(VWC_SMOOTH_STEPS, center=True,
                             min_periods=VWC_SMOOTH_STEPS // 2)
                    .median())
        valid = smoothed[(smoothed > VWC_DYN_MIN) & (smoothed < VWC_DYN_MAX)]
        p05 = float(valid.quantile(0.05)) if not valid.empty else np.nan
        p95 = float(valid.quantile(0.95)) if not valid.empty else np.nan
        dr[depth] = {
            "p05": round(p05, 2) if np.isfinite(p05) else None,
            "p95": round(p95, 2) if np.isfinite(p95) else None,
            "dyn": round(p95 - p05, 2) if (np.isfinite(p05) and np.isfinite(p95)) else None,
        }
    return dr


def _extract_event_metrics(ev_t, rs30, rs1h, dr, ev_end=None):
    """
    Extract per-depth VWC metrics and rain metrics for one event onset.

    ev_end: end of the causal event (used for post-event rain exclusion window).
    If None, falls back to ev_t (conservative: treats onset as event end).

    Returns a flat dict ready to append to a records list.
    """
    row = {}

    # Dynamic range (same for all events of this tree)
    for col in VWC_COLS:
        depth = col.split("|")[0]
        row[f"vwc_p05{depth}"]  = dr[depth]["p05"]
        row[f"vwc_p95{depth}"]  = dr[depth]["p95"]
        row[f"dyn_range{depth}"] = dr[depth]["dyn"]

    # Per-depth VWC metrics
    for col in VWC_COLS:
        depth   = col.split("|")[0]
        dyn     = dr[depth]["dyn"] or np.nan
        pre_end = ev_t - pd.Timedelta(minutes=31)
        pre_sl  = rs30.loc[pre_end - pd.Timedelta(hours=PRE_WIN_H) : pre_end, col].dropna()
        pre_vwc = float(pre_sl.median()) if not pre_sl.empty else np.nan

        peak_end = ev_t + pd.Timedelta(hours=PEAK_WIN_H)
        peak_sl  = rs30.loc[ev_t : peak_end, col].dropna()
        if peak_sl.empty:
            peak_vwc, t_peak = np.nan, pd.NaT
        else:
            t_peak   = peak_sl.idxmax()
            peak_vwc = float(peak_sl.max())

        delta_vwc = (peak_vwc - pre_vwc
                     if np.isfinite(peak_vwc) and np.isfinite(pre_vwc) else np.nan)
        delta_pct = (delta_vwc / dyn * 100
                     if np.isfinite(delta_vwc) and np.isfinite(dyn) and dyn > 0 else np.nan)
        t_to_peak = ((t_peak - ev_t).total_seconds() / 60
                     if not pd.isna(t_peak) else np.nan)

        # Drying time
        dry_thr = (pre_vwc + DRYING_FRAC * delta_vwc
                   if np.isfinite(delta_vwc) else np.nan)
        t_dry = np.nan
        if np.isfinite(dry_thr) and not pd.isna(t_peak):
            dry_sl = rs30.loc[t_peak : ev_t + pd.Timedelta(hours=DRY_MAX_H), col].dropna()
            crossed = dry_sl[dry_sl <= dry_thr]
            if not crossed.empty:
                t_dry = crossed.index[0]
        drying_h = ((t_dry - t_peak).total_seconds() / 3600
                    if not pd.isna(t_dry) and not pd.isna(t_peak) else np.nan)

        # Post-event rain accumulation over [ev_end, t_dry].
        # ev_end is the end of the causal event (passed in); falls back to ev_t.
        # Used downstream to exclude events where intervening rain > 5 mm
        # (those events violate the clean single-event drying assumption).
        _excl_start = ev_end if ev_end is not None else ev_t
        t_dry_ts = t_dry if not isinstance(t_dry, float) else pd.NaT
        if not pd.isna(t_dry_ts) and RAIN_COL in rs1h.columns and t_dry_ts > _excl_start:
            _rain_sl = rs1h.loc[_excl_start : t_dry_ts, RAIN_COL].dropna()
            rain_post_ev_mm = float(_rain_sl.sum()) if not _rain_sl.empty else 0.0
        else:
            rain_post_ev_mm = np.nan

        def _r2(v): return round(float(v), 2) if np.isfinite(v) else None
        def _r1(v): return round(float(v), 1) if np.isfinite(v) else None
        def _r0(v): return round(float(v), 0) if np.isfinite(v) else None

        # Temperature sum over the drying window (peak → dry endpoint)
        # same window as dry_h, so avg_temp = temp_sum / dry_h
        t_dry_ts = t_dry if not isinstance(t_dry, float) else pd.NaT
        t_sum_end = t_dry_ts if not pd.isna(t_dry_ts) else t_peak
        if not pd.isna(t_peak) and not pd.isna(t_sum_end) and TEMP_COL in rs1h.columns:
            temp_sl  = rs1h.loc[t_peak : t_sum_end, TEMP_COL].dropna()
            temp_sum = float(temp_sl.sum()) if not temp_sl.empty else np.nan
        else:
            temp_sum = np.nan

        row.update({
            f"pre_vwc{depth}"          : _r2(pre_vwc),
            f"peak_vwc{depth}"         : _r2(peak_vwc),
            f"delta_vwc{depth}"        : _r2(delta_vwc),
            f"delta_pct_dyn{depth}"    : _r1(delta_pct),
            f"t_peak_min{depth}"       : _r0(t_to_peak),
            f"dry_h{depth}"            : _r1(drying_h),
            f"temp_sum{depth}"         : _r1(temp_sum),
            f"rain_post_ev_mm{depth}"  : _r2(rain_post_ev_mm),
        })

    # Rain metrics around the VWC-10 peak (use shallowest channel as reference)
    ref_col     = "-10|ENV__SOIL__VWC"
    ref_depth   = "-10"
    t_peak_ref  = pd.NaT
    peak_sl_ref = rs30.loc[ev_t : ev_t + pd.Timedelta(hours=PEAK_WIN_H), ref_col].dropna()
    if not peak_sl_ref.empty:
        t_peak_ref = peak_sl_ref.idxmax()

    def _rain_sum(t_start, t_end):
        return float(rs1h.loc[t_start:t_end, RAIN_COL].dropna().sum()) \
               if RAIN_COL in rs1h.columns else np.nan

    if not pd.isna(t_peak_ref):
        row["rain_pre24h_mm"]  = round(_rain_sum(t_peak_ref - pd.Timedelta(hours=24),  t_peak_ref), 2)
        row["rain_post24h_mm"] = round(_rain_sum(t_peak_ref, t_peak_ref + pd.Timedelta(hours=24)), 2)
        row["rain_pre72h_mm"]  = round(_rain_sum(t_peak_ref - pd.Timedelta(hours=72),  t_peak_ref), 2)
        row["rain_post72h_mm"] = round(_rain_sum(t_peak_ref, t_peak_ref + pd.Timedelta(hours=72)), 2)
    else:
        row["rain_pre24h_mm"] = row["rain_post24h_mm"] = None
        row["rain_pre72h_mm"] = row["rain_post72h_mm"] = None

    return row


# ---------------------------------------------------------------------------
# Per-event plot — delegates to shared draw_event_figure in _event_plot.py
# ---------------------------------------------------------------------------
def _plot_event(ev_t, ev_type, row, rs30, rs1h, rain_roll24, dr,
                eui, event_num, out_dir):
    """Save a comprehensive annotated PNG for one event."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
           for k, v in row.items()}

    fig     = draw_event_figure(eui, ev_t, ev_type, row, rs30, rs1h, rain_roll24, dr)
    out_png = out_dir / f"event-{eui}-{event_num:03d}-{ev_type}.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-tree processing
# ---------------------------------------------------------------------------
def process_tree(eui: str) -> pd.DataFrame:
    out_dir  = TREES_DIR / eui
    csv_path = out_dir / "sensor_data.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    # ── Load ────────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    missing_vwc = [c for c in VWC_COLS if c not in df.columns]
    if missing_vwc or RAIN_COL not in df.columns:
        tqdm.write(f"  {eui}: missing columns {missing_vwc or [RAIN_COL]}, skipping")
        return pd.DataFrame()

    # ── Resample ────────────────────────────────────────────────────────────
    vwc_raw  = df[["datetime"] + VWC_COLS].set_index("datetime").sort_index()
    rs30     = vwc_raw.resample("30min").mean().ffill(limit=2)
    rs30_cnt = vwc_raw.resample("30min").count()   # actual transmissions, no fill

    met_cols = ["datetime", RAIN_COL] + ([TEMP_COL] if TEMP_COL in df.columns else [])
    met_raw  = df[met_cols].set_index("datetime").sort_index()
    rs1h     = met_raw.resample("1h").mean().ffill(limit=1)
    if TEMP_COL not in rs1h.columns:
        rs1h[TEMP_COL] = np.nan

    # ── Dynamic range ────────────────────────────────────────────────────────
    dr = _dynamic_range(rs30)

    # Validate that all channels have a usable dynamic range
    if any(dr[d.split("|")[0]]["dyn"] is None for d in VWC_COLS
           if d.split("|")[0] in dr):
        tqdm.write(f"  {eui}: insufficient dynamic range data, skipping")
        return pd.DataFrame()

    dyn = {col.split("|")[0]: (dr[col.split("|")[0]]["dyn"] or 0) for col in VWC_COLS}

    # ── 1. IRRIGATION detection ──────────────────────────────────────────────
    # For each timestamp, compute the forward 4 h max VWC minus current VWC.
    # If this rise >= 20 % of dynamic range simultaneously in ALL 3 channels,
    # flag the timestamp as an irrigation onset.
    irrig_flags = {}
    win_steps   = IRRIG_WINDOW_H * 2   # 30-min steps in the window

    for col in VWC_COLS:
        depth = col.split("|")[0]
        # Forward-looking rolling max: reverse series, roll, reverse back
        fwd_max   = rs30[col].iloc[::-1].rolling(win_steps, min_periods=1).max().iloc[::-1]
        rise_fwd  = fwd_max - rs30[col]
        threshold = IRRIG_RISE_THR * dyn[depth]
        irrig_flags[depth] = (rise_fwd >= threshold) & (rise_fwd > 0)

    sim_irrig   = irrig_flags["-10"] & irrig_flags["-30"] & irrig_flags["-45"]
    irrig_times = _merge_times(rs30.index[sim_irrig].tolist(), IRRIG_MERGE_H)

    # ── IRRIGATION_STRONG: rise >= 80 % dyn. range simultaneously in all 3 depths
    irrig_strong = []
    _pre_td = pd.Timedelta(minutes=31)
    _pre_win = pd.Timedelta(hours=PRE_WIN_H)
    _pk_win  = pd.Timedelta(hours=PEAK_WIN_H)
    for t in irrig_times:
        ok = True
        for col in VWC_COLS:
            depth = col.split("|")[0]
            dyn   = dr[depth]["dyn"]
            if not dyn or dyn <= 0: ok = False; break
            pre_sl  = rs30.loc[t - _pre_td - _pre_win : t - _pre_td, col].dropna()
            peak_sl = rs30.loc[t : t + _pk_win, col].dropna()
            if pre_sl.empty or peak_sl.empty: ok = False; break
            if float(peak_sl.max()) - float(pre_sl.median()) < IRRIG_STRONG_DYN * dyn:
                ok = False; break
        if ok:
            irrig_strong.append(t)

    # ── 2. RAIN_ALL detection ────────────────────────────────────────────────
    rain_series  = rs1h[RAIN_COL].fillna(0)
    rain_roll24  = rain_series.rolling(24, min_periods=1).sum()
    rain_peaks   = _find_local_peaks(rain_roll24, RAIN_MIN_24H, RAIN_PEAK_MERGE_H)

    rain_all = []
    for peak_t in rain_peaks:
        window = rs1h.loc[peak_t - pd.Timedelta(hours=23) : peak_t, RAIN_COL].dropna()
        onset  = window.idxmax() if not window.empty else peak_t
        rain_all.append(onset)
    rain_all = _merge_times(rain_all, RAIN_MERGE_H)

    # ── Onset refinement: shift onset to the VWC minimum within LOOKBACK window ─
    # If any VWC depth was significantly lower (> ONSET_REFINE_DYN * dyn_range)
    # within ONSET_LOOKBACK_H hours BEFORE the detected onset, the soil moisture
    # rise had already begun before the rain peak.  Move the onset to the VWC
    # minimum in that window — i.e., the driest point just before infiltration.
    lookback_td = pd.Timedelta(hours=ONSET_LOOKBACK_H)
    refined_all = []
    for t in rain_all:
        new_t = t   # default: no change
        for col in VWC_COLS:
            depth = col.split("|")[0]
            dyn   = dr[depth]["dyn"]
            if not dyn or dyn <= 0:
                continue
            # VWC at the detected onset
            at_t = rs30.loc[t : t + pd.Timedelta(minutes=1), col].dropna()
            if at_t.empty:
                continue
            v_onset = float(at_t.iloc[0])
            # VWC in the lookback window (excluding the onset step itself)
            window = rs30.loc[t - lookback_td : t - pd.Timedelta(minutes=29), col].dropna()
            if window.empty:
                continue
            v_min = float(window.min())
            # Only refine if the minimum is significantly below the onset value
            if v_onset - v_min > ONSET_REFINE_DYN * dyn:
                t_min = window.idxmin()
                if t_min < new_t:
                    new_t = t_min
        refined_all.append(new_t)
    rain_all = refined_all

    # ── Suppress rain events within +-IRRIG_EXCL_H of any irrigation ────────
    # Irrigation events dominate the moisture signal; rain events in that
    # window are unreliable and should not be labelled as rain detections.
    rain_all = [t for t in rain_all
                if not any(abs((t - t_i).total_seconds()) <= IRRIG_EXCL_H * 3600
                           for t_i in irrig_times)]

    # ── 3. Classify RAIN_HEAD, RAIN_TAIL and their VWC-based subsets ─────────
    inner_td = pd.Timedelta(hours=RAIN_WIN_H_INNER)
    outer_td = pd.Timedelta(hours=RAIN_WIN_H_OUTER)
    cov_td   = pd.Timedelta(hours=COVERAGE_WIN_H)

    rain_head         = []
    rain_tail         = []
    rain_head_impact  = []
    rain_tail_dry     = []
    gap_dropped_times = []   # HEAD/TAIL events with no usable VWC coverage

    for t in rain_all:
        # Concentration fractions (both sides independently)
        ip    = float(rain_series.loc[t - inner_td : t].sum())
        ipost = float(rain_series.loc[t : t + inner_td].sum())
        op    = float(rain_series.loc[t - outer_td : t].sum())
        opost = float(rain_series.loc[t : t + outer_td].sum())
        frac_pre  = ip    / op    if op    > 0 else 1.0
        frac_post = ipost / opost if opost > 0 else 1.0

        is_head = frac_pre  >= RAIN_HEAD_FRAC
        is_tail = frac_post >= RAIN_TAIL_FRAC

        if is_head: rain_head.append(t)
        if is_tail: rain_tail.append(t)

        if not (is_head or is_tail):
            continue   # no VWC checks needed

        has_impact = False
        has_drying = False
        all_gaps   = True

        pre_td  = pd.Timedelta(minutes=31)
        pre_win = pd.Timedelta(hours=PRE_WIN_H)
        pk_win  = pd.Timedelta(hours=PEAK_WIN_H)
        dry_end = t + pd.Timedelta(hours=DRY_MAX_H)

        for col in VWC_COLS:
            depth = col.split("|")[0]

            # Coverage gate
            cov_n_pre  = int(rs30_cnt.loc[t - cov_td : t, col].sum()) \
                         if col in rs30_cnt.columns else 0
            cov_n_post = int(rs30_cnt.loc[t : t + cov_td, col].sum()) \
                         if col in rs30_cnt.columns else 0
            if cov_n_pre < COVERAGE_MIN_SAMPLES or cov_n_post < COVERAGE_MIN_SAMPLES:
                continue
            all_gaps = False

            # Baseline and peak
            pre_end = t - pre_td
            pre_sl  = rs30.loc[pre_end - pre_win : pre_end, col].dropna()
            peak_sl = rs30.loc[t : t + pk_win, col].dropna()
            if pre_sl.empty or peak_sl.empty:
                continue

            pre_vwc  = float(pre_sl.median())
            t_peak   = peak_sl.idxmax()
            peak_vwc = float(peak_sl.max())
            delta    = peak_vwc - pre_vwc

            # RAIN_HEAD_IMPACT: rise >= 5 % of dynamic range AND >= 1 %-pt absolute
            dyn = dr[depth]["dyn"]
            if is_head and not has_impact and dyn and dyn > 0 \
                    and delta >= RAIN_IMPACT_DYN * dyn \
                    and delta >= RAIN_IMPACT_ABS:
                has_impact = True

            # RAIN_TAIL_DRY: same dual threshold, then checks drying return
            if is_tail and not has_drying and delta > 0 \
                    and dyn and dyn > 0 \
                    and delta >= RAIN_IMPACT_DYN * dyn \
                    and delta >= RAIN_IMPACT_ABS:
                dry_thr = pre_vwc + DRYING_FRAC * delta
                dry_sl  = rs30.loc[t_peak : dry_end, col].dropna()
                if not dry_sl.empty and not dry_sl[dry_sl <= dry_thr].empty:
                    has_drying = True

        if all_gaps and (is_head or is_tail):
            gap_dropped_times.append(t)
            continue

        if is_head and has_impact:
            rain_head_impact.append(t)
        if is_tail and has_drying:
            rain_tail_dry.append(t)

    # RAIN_FULL: intersection of RAIN_HEAD_IMPACT and RAIN_TAIL_DRY
    head_impact_set = set(rain_head_impact)
    rain_full = [t for t in rain_tail_dry if t in head_impact_set]

    # RAIN_STRONG: subset of RAIN_HEAD_IMPACT where ALL 3 depths meet both thresholds
    rain_strong = []
    for t in rain_head_impact:
        ok = True
        for col in VWC_COLS:
            depth = col.split("|")[0]
            dyn   = dr[depth]["dyn"]
            if not dyn or dyn <= 0: ok = False; break
            cov_pre  = int(rs30_cnt.loc[t - cov_td : t, col].sum()) \
                       if col in rs30_cnt.columns else 0
            cov_post = int(rs30_cnt.loc[t : t + cov_td, col].sum()) \
                       if col in rs30_cnt.columns else 0
            if cov_pre < COVERAGE_MIN_SAMPLES or cov_post < COVERAGE_MIN_SAMPLES:
                ok = False; break
            pre_sl  = rs30.loc[t - _pre_td - _pre_win : t - _pre_td, col].dropna()
            peak_sl = rs30.loc[t : t + _pk_win, col].dropna()
            if pre_sl.empty or peak_sl.empty: ok = False; break
            delta = float(peak_sl.max()) - float(pre_sl.median())
            if delta < RAIN_IMPACT_DYN * dyn or delta < RAIN_IMPACT_ABS:
                ok = False; break
        if ok:
            rain_strong.append(t)

    # ── Compile all events into records ─────────────────────────────────────
    all_events = (
        [(t, "IRRIGATION")          for t in irrig_times]      +
        [(t, "IRRIGATION_STRONG")   for t in irrig_strong]     +
        [(t, "RAIN_ALL")            for t in rain_all]          +
        [(t, "RAIN_HEAD")           for t in rain_head]         +
        [(t, "RAIN_TAIL")           for t in rain_tail]         +
        [(t, "RAIN_HEAD_IMPACT")    for t in rain_head_impact]  +
        [(t, "RAIN_STRONG")         for t in rain_strong]       +
        [(t, "RAIN_TAIL_DRY")       for t in rain_tail_dry]     +
        [(t, "RAIN_FULL")           for t in rain_full]         +
        [(t, "RAIN_GAP_DROP")       for t in gap_dropped_times]
    )

    if not all_events:
        return pd.DataFrame()

    records = []
    for ev_t, ev_type in all_events:
        row = {
            "eui"        : eui,
            "event_time" : ev_t,
            "event_type" : ev_type,
            "month"      : ev_t.month,
            "doy"        : ev_t.day_of_year,
        }
        # Pass event end for the post-event rain exclusion window.
        # Irrigation: detection uses a 4 h forward window; that window's end is
        # the earliest defensible proxy for when irrigation water delivery stops.
        # Rain events in this per-tree CSV have no external ev_end; use onset as
        # conservative fallback (BSC rain ev_end is handled at the model level).
        if ev_type in ("IRRIGATION", "IRRIGATION_STRONG"):
            _ev_end = ev_t + pd.Timedelta(hours=IRRIG_WINDOW_H)
        else:
            _ev_end = ev_t
        row.update(_extract_event_metrics(ev_t, rs30, rs1h, dr, ev_end=_ev_end))
        records.append(row)

    result_df = pd.DataFrame(records)

    # ── Save per-tree CSV ────────────────────────────────────────────────────
    event_dir = out_dir / "Irrigation Events"
    event_dir.mkdir(exist_ok=True)
    result_df.to_csv(event_dir / "irrigation_events_v2.csv", index=False)

    # ── Generate per-event PNGs (optional — disabled by default) ────────────
    if _HAS_EVENT_PLOT and GENERATE_EVENT_PLOTS:
        plot_dir = out_dir / "Irrigation Events v2"
        plot_dir.mkdir(exist_ok=True)
        for old in plot_dir.glob(f"event-{eui}-*.png"):
            old.unlink()
        for event_num, (ev_t, ev_type) in enumerate(all_events, start=1):
            if ev_type != "RAIN_GAP_DROP":
                row = result_df[result_df["event_time"] == ev_t].iloc[0].to_dict() \
                      if not result_df[result_df["event_time"] == ev_t].empty else {}
                try:
                    _plot_event(ev_t, ev_type, row, rs30, rs1h, rain_roll24,
                                dr, eui, event_num, plot_dir)
                except Exception as exc:
                    tqdm.write(f"    [PLOT ERR] {eui} #{event_num}: {exc}")

    counts = {et: sum(1 for _, t in all_events if t == et)
              for et in ["IRRIGATION", "IRRIGATION_STRONG",
                         "RAIN_ALL", "RAIN_HEAD", "RAIN_TAIL",
                         "RAIN_HEAD_IMPACT", "RAIN_STRONG",
                         "RAIN_TAIL_DRY", "RAIN_FULL", "RAIN_GAP_DROP"]}
    return result_df, counts


# ---------------------------------------------------------------------------
# Random example plot generation
# ---------------------------------------------------------------------------
_EXAMPLE_TYPES = [
    "RAIN_ALL", "RAIN_HEAD", "RAIN_TAIL",
    "RAIN_HEAD_IMPACT", "RAIN_STRONG",
    "RAIN_TAIL_DRY", "RAIN_FULL",
    "IRRIGATION", "IRRIGATION_STRONG",
]


def _generate_examples(combined_csv: Path, n: int = N_EXAMPLES, seed: int = 42):
    """Sample n random events per type and save annotated PNGs to EXAMPLES_DIR."""
    if not _HAS_EVENT_PLOT:
        print("  [examples] _event_plot not available — skipping example plots.")
        print("             (CSV was already saved; plots are optional.)")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not combined_csv.exists():
        print("  [examples] Combined CSV not found — skipping.")
        return
    ev = pd.read_csv(combined_csv, parse_dates=["event_time"],
                     dtype={"eui": str}, low_memory=False)
    rng = __import__("random").Random(seed)
    print(f"\n  Generating {n} examples per event type → {EXAMPLES_DIR}")
    for ev_type in _EXAMPLE_TYPES:
        subset = ev[ev["event_type"] == ev_type]
        if subset.empty:
            print(f"    {ev_type}: no events"); continue
        sample   = subset.sample(n=min(n, len(subset)), random_state=seed)
        type_dir = EXAMPLES_DIR / ev_type
        type_dir.mkdir(parents=True, exist_ok=True)
        ok = 0
        for i, (_, row) in enumerate(tqdm(sample.iterrows(),
                                          total=len(sample),
                                          desc=f"    {ev_type}")):
            eui   = str(row["eui"])
            onset = pd.Timestamp(row["event_time"])
            csv_p = TREES_DIR / eui / "sensor_data.csv"
            if not csv_p.exists(): continue
            try:
                df = pd.read_csv(csv_p, parse_dates=["datetime"])
                df = df.sort_values("datetime").reset_index(drop=True)
                vwc_avail = [c for c in VWC_COLS if c in df.columns]
                rs30 = (df[["datetime"] + vwc_avail]
                        .set_index("datetime").sort_index()
                        .resample("30min").mean().ffill(limit=2))
                met  = [c for c in ["datetime", RAIN_COL, TEMP_COL]
                        if c in df.columns]
                rs1h = (df[met].set_index("datetime").sort_index()
                        .resample("1h").mean().ffill(limit=1))
                if TEMP_COL not in rs1h.columns:
                    rs1h[TEMP_COL] = np.nan
                rain   = rs1h[RAIN_COL].fillna(0) if RAIN_COL in rs1h.columns \
                         else pd.Series(0.0, index=rs1h.index)
                roll24 = rain.rolling(24, min_periods=1).sum()
                # Compute dynamic range on-the-fly
                dr = {}
                for col in vwc_avail:
                    depth    = col.split("|")[0]
                    smoothed = rs30[col].rolling(
                        VWC_SMOOTH_STEPS, center=True,
                        min_periods=VWC_SMOOTH_STEPS // 2).median()
                    valid = smoothed[(smoothed > VWC_DYN_MIN) &
                                     (smoothed < VWC_DYN_MAX)]
                    p05  = float(valid.quantile(0.05)) if not valid.empty else None
                    p95  = float(valid.quantile(0.95)) if not valid.empty else None
                    dr[depth] = {
                        "p05": round(p05, 2) if p05 else None,
                        "p95": round(p95, 2) if p95 else None,
                        "dyn": round(p95 - p05, 2) if (p05 and p95) else None,
                    }
                row_dict = {
                    k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                    for k, v in row.items()
                }
                fig = draw_event_figure(
                    eui, onset, ev_type, row_dict, rs30, rs1h, roll24, dr)
                out = type_dir / f"example_{i+1:03d}.png"
                fig.savefig(out, dpi=120, bbox_inches="tight")
                plt.close(fig)
                ok += 1
            except Exception as e:
                tqdm.write(f"    [example ERR] {eui}: {e}")
        print(f"    {ok}/{len(sample)} saved")


# ---------------------------------------------------------------------------
# Module-level worker (must be at top level so multiprocessing can pickle it)
# ---------------------------------------------------------------------------
def _worker(eui: str) -> tuple:
    """Wrap process_tree so exceptions are returned as data, not raised."""
    try:
        result = process_tree(eui)
        if isinstance(result, tuple):
            df_ev, counts = result
            return eui, df_ev, counts, None
        return eui, pd.DataFrame(), {}, None
    except Exception as exc:
        return eui, pd.DataFrame(), {}, str(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("  IRRIGATION EVENT DETECTOR v2")
    print("=" * 65)

    print(f"\n[1/3] Scanning for sensor data in: {TREES_DIR}")
    eui_dirs = sorted(
        d for d in TREES_DIR.iterdir()
        if d.is_dir() and (d / "sensor_data.csv").exists()
    )
    print(f"  Found {len(eui_dirs)} trees with sensor_data.csv.")
    eui_names = [d.name for d in eui_dirs]

    print(f"\n[2/3] Detecting events ({len(eui_dirs)} trees, single-threaded) ...")
    all_frames = []
    skipped    = 0

    for eui in tqdm(eui_names, desc="  Trees"):
        _, df_ev, counts, err = _worker(eui)
        if err:
            print(f"  [ERR] {eui}: {err}", flush=True)
        elif not df_ev.empty:
            all_frames.append(df_ev)
            print(
                f"  {eui}: "
                f"irrig={counts.get('IRRIGATION',0)}"
                f"(strong={counts.get('IRRIGATION_STRONG',0)})  "
                f"all={counts.get('RAIN_ALL',0)}  "
                f"head_imp={counts.get('RAIN_HEAD_IMPACT',0)}"
                f"(strong={counts.get('RAIN_STRONG',0)})  "
                f"tail_dry={counts.get('RAIN_TAIL_DRY',0)}  "
                f"full={counts.get('RAIN_FULL',0)}  "
                f"gap={counts.get('RAIN_GAP_DROP',0)}",
                flush=True,
            )
        else:
            skipped += 1

    print(f"\n  {len(all_frames)} trees with events, "
          f"{skipped} skipped (missing columns or no events).")

    print(f"\n[3/3] Saving combined output ...")
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(ALL_EVENTS_CSV, index=False)
        print(f"  Combined CSV saved -> {ALL_EVENTS_CSV}")
        print(f"\n  Event counts across {len(all_frames)} trees:")
        for et in ["IRRIGATION", "RAIN_ALL", "RAIN_SHORT", "RAIN_IMPACT"]:
            n = (combined["event_type"] == et).sum()
            print(f"    {et:<15}: {n:>5}")
        print(f"\n  Total columns in output: {len(combined.columns)}")
        print(f"  Columns: {list(combined.columns)}")
    else:
        print("  No events detected across any tree.")

    _generate_examples(ALL_EVENTS_CSV)

    print("\n" + "=" * 65 + "\n  DONE\n" + "=" * 65)
