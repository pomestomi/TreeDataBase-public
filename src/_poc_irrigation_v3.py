#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_poc_irrigation_v3.py — BSC (Binary Shape Code) rain event classification
Reference: Terranova & Iaquinta (2011), Nat. Hazards Earth Syst. Sci., 11, 751-757

For each tree with rain sensor data:
  1. Detect rain events (>= 5 mm total, separated by >= 6 h dry gap)
  2. Compute SRP (Standardized Rainfall Profile) — normalized cumulative rain
  3. Assign BSC (4-digit binary code) by comparing SRP areas to USRP
  4. Assign Huff quartile from StAC (time of max intensity)
  5. Save 2 example PNGs (SRP + VWC response) in bsc_events/ subfolder
  6. Save per-tree bsc_events.csv

Global outputs:
  TreeTabularData/bsc_events_all.csv
  TreeTabularData/rf_dataset.csv          (bsc_events_all joined with treeLocations metadata)
  DatasetStatistics/_report_assets/chart_bsc_quartiles.png  (Fig. 4 equivalent)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
try:
    import geopandas as gpd
    _HAS_GEOPANDAS = True
except ImportError:
    _HAS_GEOPANDAS = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parent.parent
TREES_DIR    = REPO_ROOT / "TreeTabularData" / "trees"
OUT_ALL      = REPO_ROOT / "TreeTabularData" / "bsc_events_all.csv"
OUT_RF       = REPO_ROOT / "TreeTabularData" / "rf_dataset.csv"
GIS_DIR      = REPO_ROOT / "GISData"
REPORT_DIR   = REPO_ROOT / "TreeTabularData" / "_report_assets"

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
RAIN_COL     = "MTB|ENV__ATMO__RAIN__DELTA"
VWC_COLS     = ["-10|ENV__SOIL__VWC", "-30|ENV__SOIL__VWC", "-45|ENV__SOIL__VWC"]
DEPTH_COLORS = {"-10": "#1b7837", "-30": "#762a83", "-45": "#e08214"}
DEPTH_LABELS = {"-10": "-10 cm",  "-30": "-30 cm",  "-45": "-45 cm"}

# ---------------------------------------------------------------------------
# Detection parameters
# ---------------------------------------------------------------------------
MIN_EVENT_MM  = 5.0    # minimum total rainfall per event (lowered from 12.7 for VWC analysis)
LITERATURE_MM = 12.7   # original Wischmeier & Smith / Terranova threshold — shown as reference line
DRY_GAP_H     = 6      # consecutive dry hours to split events
MIN_STEPS     = 3      # minimum rainy hours per event
RAIN_ZERO     = 0.1    # mm threshold below which an hour counts as "dry"
N_EXAMPLES         = 2      # example plots per tree
N_EXAMPLES_GLOBAL  = 5      # examples per BSC code in BSCRainExamples/

# ---------------------------------------------------------------------------
# VWC / temperature metric parameters (matching v2 conventions)
# ---------------------------------------------------------------------------
TEMP_COL         = "MTB|ENV__ATMO__T"
PRE_WIN_H        = 2        # hours before onset for pre-event VWC baseline
POST_EVENT_SEARCH_H = 24    # hours after ev_end to search for VWC peak (cut off at next rain)
DRY_MAX_H        = 7 * 24   # max hours to search for drying return
DRYING_FRAC      = 0.10     # "dried" when VWC <= pre + frac*(peak-pre)
VWC_SMOOTH_STEPS = 8        # rolling-median window (30 min × 8 = 4 h)
VWC_DYN_MIN      = 2.0      # exclude below this (sensor gaps)
VWC_DYN_MAX      = 50.0     # exclude above this (saturation artefacts)

# ---------------------------------------------------------------------------
# USRP quartile areas (analytical): A*_k = (tau_k^2 - tau_{k-1}^2) / 2
# tau_k = k/4 for k = 1..4
# A*_1=0.03125  A*_2=0.09375  A*_3=0.15625  A*_4=0.21875
# ---------------------------------------------------------------------------
TAU_BOUNDS = [0.0, 0.25, 0.50, 0.75, 1.0]
USRP_AREAS = [(TAU_BOUNDS[k] ** 2 - TAU_BOUNDS[k - 1] ** 2) / 2
              for k in range(1, 5)]

# Colour palette for all 16 BSC codes
_BSC_PALETTE = {
    "0000": "#a6cee3", "0001": "#1f78b4", "0010": "#b2df8a", "0011": "#33a02c",
    "0100": "#fb9a99", "0101": "#e31a1c", "0110": "#fdbf6f", "0111": "#ff7f00",
    "1000": "#cab2d6", "1001": "#6a3d9a", "1010": "#ffff99", "1011": "#b15928",
    "1100": "#8dd3c7", "1101": "#ffffb3", "1110": "#bebada", "1111": "#fb8072",
}
_QUARTILE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]


# ---------------------------------------------------------------------------
# 1. Event detection
# ---------------------------------------------------------------------------
def detect_rain_events(rain_1h: pd.Series) -> list:
    """
    Return sorted list of (ev_start, ev_end, total_mm) for events meeting:
      - total_mm >= MIN_EVENT_MM
      - separated by >= DRY_GAP_H consecutive dry hours
      - >= MIN_STEPS rainy hours
    """
    rain = rain_1h.fillna(0.0)
    events = []

    # Assign a monotone group ID: increment when rain resumes after a dry gap
    group_id    = []
    current_gid = 0
    dry_counter = 0

    for v in rain.values:
        if v > RAIN_ZERO:
            if dry_counter >= DRY_GAP_H or not group_id:
                current_gid += 1
            dry_counter = 0
        else:
            dry_counter += 1
        group_id.append(current_gid)

    group_s  = pd.Series(group_id, index=rain.index)
    has_rain = rain > RAIN_ZERO

    for gid in group_s[has_rain].unique():
        if gid == 0:
            continue
        rainy = rain[(group_s == gid) & has_rain]
        if len(rainy) < MIN_STEPS:
            continue
        total = float(rainy.sum())
        if total < MIN_EVENT_MM:
            continue
        events.append((rainy.index[0], rainy.index[-1], total))

    return sorted(events)


# ---------------------------------------------------------------------------
# 2. SRP + BSC computation
# ---------------------------------------------------------------------------
def compute_bsc(ev_start, ev_end, rain_1h: pd.Series) -> dict | None:
    """
    Compute SRP and BSC for one rain event.

    BSC S_k = 1 if SRP area in quartile k exceeds USRP area, else 0.
    Returns a dict (with private _tau / _pi arrays for plotting) or None.
    """
    ev_rain = rain_1h.loc[ev_start:ev_end].fillna(0.0)
    total   = float(ev_rain.sum())
    if total <= 0 or len(ev_rain) < 2:
        return None

    dur_h = (ev_end - ev_start).total_seconds() / 3600
    if dur_h <= 0:
        dur_h = max(float(len(ev_rain) - 1), 1.0)

    # Dimensionless time (tau) and cumulative rainfall (pi)
    t0  = ev_start
    tau = np.array([(ts - t0).total_seconds() / 3600 / dur_h
                    for ts in ev_rain.index], dtype=float)
    pi  = ev_rain.cumsum().values / total

    # Prepend origin
    if tau[0] > 1e-9:
        tau = np.concatenate([[0.0], tau])
        pi  = np.concatenate([[0.0], pi])

    tau = np.clip(tau, 0.0, 1.0)

    # Trapezoidal integral of pi over each quartile (Eq. in paper)
    areas = []
    for k in range(4):
        lo, hi = TAU_BOUNDS[k], TAU_BOUNDS[k + 1]
        pi_lo  = float(np.interp(lo, tau, pi))
        pi_hi  = float(np.interp(hi, tau, pi))
        inner  = (tau > lo) & (tau < hi)
        t_q    = np.concatenate([[lo], tau[inner], [hi]])
        p_q    = np.concatenate([[pi_lo], pi[inner], [pi_hi]])
        _, idx = np.unique(t_q, return_index=True)
        areas.append(float(np.trapezoid(p_q[idx], t_q[idx])))

    bsc_bits = [1 if areas[k] > USRP_AREAS[k] else 0 for k in range(4)]
    bsc_str  = "".join(str(b) for b in bsc_bits)

    # StAC = dimensionless time of maximum hourly intensity
    max_idx  = int(ev_rain.values.argmax())
    stac     = float(np.clip(
        (ev_rain.index[max_idx] - t0).total_seconds() / 3600 / dur_h, 0.0, 1.0))
    huff_q   = int(min(4, int(stac * 4) + 1))

    return {
        "ev_start":          ev_start,
        "ev_end":            ev_end,
        "total_mm":          round(total, 2),
        "duration_h":        round(dur_h, 1),
        "max_intensity_mmh": round(float(ev_rain.max()), 2),
        "stac":              round(stac, 3),
        "huff_q":            huff_q,
        "bsc":               bsc_str,
        "bsc_s1": bsc_bits[0], "bsc_s2": bsc_bits[1],
        "bsc_s3": bsc_bits[2], "bsc_s4": bsc_bits[3],
        "area_1": round(areas[0], 5), "area_2": round(areas[1], 5),
        "area_3": round(areas[2], 5), "area_4": round(areas[3], 5),
        "_tau": tau,
        "_pi":  pi,
    }


# ---------------------------------------------------------------------------
# 2b. Dynamic range and VWC response metric helpers
# ---------------------------------------------------------------------------
def _dynamic_range(rs30: pd.DataFrame) -> dict:
    """
    5th / 95th percentile dynamic range per VWC channel from 4 h rolling median.
    Returns {depth: {"p05": float, "p95": float, "dyn": float}}.
    """
    dr = {}
    for col in VWC_COLS:
        depth    = col.split("|")[0]
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


def _extract_vwc_metrics(ev_start, ev_end, _stac, _duration_h,
                         rs30: pd.DataFrame, rs1h: pd.DataFrame, dr: dict) -> dict:
    """
    Per-depth VWC metrics and temperature sums for one event onset.

    Peak VWC search window end = max(ev_end + EVENT_EXT_FRAC * duration,
                                     t_stac_abs + STAC_MIN_H)

    Public keys (saved to CSV):
      month, doy
      vwc_p05{d}, vwc_p95{d}, dyn_range{d}
      pre_vwc{d}, peak_vwc{d}, delta_vwc{d}, delta_pct_dyn{d}, t_peak_min{d}, dry_h{d}
      temp_sum{d}   (deg-C·h from VWC peak until drying endpoint, same window as dry_h)
      rain_pre24h_mm, rain_post24h_mm, rain_pre72h_mm, rain_post72h_mm

    Private keys (plot annotations only, filtered before CSV export):
      _t_peak{d}, _t_dry{d}   (pd.Timestamp or pd.NaT)
    """
    row = {"month": int(ev_start.month), "doy": int(ev_start.day_of_year)}

    # Peak-search window: ev_start → ev_end + 24 h, truncated at the first
    # post-event rain hour so subsequent rainfall cannot bias the VWC peak.
    _post_end = ev_end + pd.Timedelta(hours=POST_EVENT_SEARCH_H)
    if RAIN_COL in rs1h.columns:
        _after = rs1h.loc[ev_end + pd.Timedelta(hours=1) : _post_end, RAIN_COL]
        _wet   = _after[_after > RAIN_ZERO]
        vwc_peak_end = _wet.index[0] if not _wet.empty else _post_end
    else:
        vwc_peak_end = _post_end

    for col in VWC_COLS:
        depth = col.split("|")[0]
        row[f"vwc_p05{depth}"]   = dr[depth]["p05"]
        row[f"vwc_p95{depth}"]   = dr[depth]["p95"]
        row[f"dyn_range{depth}"] = dr[depth]["dyn"]

    def _r2(v): return round(float(v), 2) if np.isfinite(v) else None
    def _r1(v): return round(float(v), 1) if np.isfinite(v) else None
    def _r0(v): return round(float(v), 0) if np.isfinite(v) else None

    for col in VWC_COLS:
        depth = col.split("|")[0]
        dyn   = dr[depth]["dyn"] or np.nan

        # Pre-event baseline — median of 2 h window ending 31 min before onset
        pre_end = ev_start - pd.Timedelta(minutes=31)
        pre_sl  = rs30.loc[pre_end - pd.Timedelta(hours=PRE_WIN_H) : pre_end, col].dropna()
        pre_vwc = float(pre_sl.median()) if not pre_sl.empty else np.nan

        peak_sl = rs30.loc[ev_start : vwc_peak_end, col].dropna()
        if peak_sl.empty:
            peak_vwc, t_peak = np.nan, pd.NaT
        else:
            t_peak   = peak_sl.idxmax()
            peak_vwc = float(peak_sl.max())

        delta_vwc = (peak_vwc - pre_vwc
                     if np.isfinite(peak_vwc) and np.isfinite(pre_vwc) else np.nan)
        delta_pct = (delta_vwc / dyn * 100
                     if np.isfinite(delta_vwc) and np.isfinite(dyn) and dyn > 0 else np.nan)
        t_to_peak = ((t_peak - ev_start).total_seconds() / 60
                     if not pd.isna(t_peak) else np.nan)

        # Drying time — first return to pre + DRYING_FRAC * delta
        dry_thr = (pre_vwc + DRYING_FRAC * delta_vwc
                   if np.isfinite(delta_vwc) else np.nan)
        t_dry = pd.NaT
        if np.isfinite(dry_thr) and not pd.isna(t_peak):
            dry_sl  = rs30.loc[t_peak : ev_start + pd.Timedelta(hours=DRY_MAX_H),
                               col].dropna()
            crossed = dry_sl[dry_sl <= dry_thr]
            if not crossed.empty:
                t_dry = crossed.index[0]
        drying_h = ((t_dry - t_peak).total_seconds() / 3600
                    if not pd.isna(t_dry) and not pd.isna(t_peak) else np.nan)

        # Temperature sum from VWC peak to drying endpoint — same window as dry_h,
        # so avg_temp = temp_sum / dry_h is a meaningful average temperature.
        # (Using ev_start would inflate temp_sum for events with a long rise phase.)
        t_sum_end = t_dry if not pd.isna(t_dry) else t_peak
        if not pd.isna(t_peak) and not pd.isna(t_sum_end) and TEMP_COL in rs1h.columns:
            temp_sl  = rs1h.loc[t_peak : t_sum_end, TEMP_COL].dropna()
            temp_sum = float(temp_sl.sum()) if not temp_sl.empty else np.nan
        else:
            temp_sum = np.nan

        row.update({
            f"pre_vwc{depth}"       : _r2(pre_vwc),
            f"peak_vwc{depth}"      : _r2(peak_vwc),
            f"delta_vwc{depth}"     : _r2(delta_vwc),
            f"delta_pct_dyn{depth}" : _r1(delta_pct),
            f"t_peak_min{depth}"    : _r0(t_to_peak),
            f"dry_h{depth}"         : _r1(drying_h),
            f"temp_sum{depth}"      : _r1(temp_sum),
            f"_t_peak{depth}"       : t_peak,
            f"_t_dry{depth}"        : t_dry,
        })

    # Rain sums relative to -10 cm VWC peak
    ref_col  = "-10|ENV__SOIL__VWC"
    t_pk_ref = pd.NaT
    if ref_col in rs30.columns:
        pk_sl = rs30.loc[ev_start : vwc_peak_end, ref_col].dropna()
        if not pk_sl.empty:
            t_pk_ref = pk_sl.idxmax()

    def _rain_sum(t_s, t_e):
        if RAIN_COL not in rs1h.columns:
            return None
        return round(float(rs1h.loc[t_s : t_e, RAIN_COL].dropna().sum()), 2)

    if not pd.isna(t_pk_ref):
        row["rain_pre24h_mm"]  = _rain_sum(t_pk_ref - pd.Timedelta(hours=24), t_pk_ref)
        row["rain_post24h_mm"] = _rain_sum(t_pk_ref, t_pk_ref + pd.Timedelta(hours=24))
        row["rain_pre72h_mm"]  = _rain_sum(t_pk_ref - pd.Timedelta(hours=72), t_pk_ref)
        row["rain_post72h_mm"] = _rain_sum(t_pk_ref, t_pk_ref + pd.Timedelta(hours=72))
    else:
        row["rain_pre24h_mm"] = row["rain_post24h_mm"] = None
        row["rain_pre72h_mm"] = row["rain_post72h_mm"] = None

    return row


# ---------------------------------------------------------------------------
# 3. Example plot: SRP panel + VWC response panel
# ---------------------------------------------------------------------------
def _plot_bsc_example(result: dict, eui: str,
                      rain_1h: pd.Series, vwc_30min, out_path: Path):
    tau      = result["_tau"]
    pi       = result["_pi"]
    bsc      = result["bsc"]
    stac     = result["stac"]
    huff_q   = result["huff_q"]
    ev_start = result["ev_start"]
    ev_end   = result["ev_end"]

    fig, (ax_srp, ax_vwc) = plt.subplots(
        2, 1, figsize=(9, 7.5),
        gridspec_kw={"height_ratios": [1, 1.7]},
    )
    fig.suptitle(
        f"{eui}  |  {ev_start.strftime('%Y-%m-%d')}  |  "
        f"BSC = {bsc}  |  Quartile {huff_q}  |  "
        f"{result['total_mm']} mm  |  {result['duration_h']} h",
        fontsize=10, fontweight="bold",
    )

    # --- SRP panel ---
    band_cols = ["#f5f5f5", "#ebebeb", "#f5f5f5", "#ebebeb"]
    for k in range(4):
        ax_srp.axvspan(TAU_BOUNDS[k], TAU_BOUNDS[k + 1],
                       color=band_cols[k], alpha=0.6, zorder=0)
        ax_srp.axvline(TAU_BOUNDS[k + 1], color="#cccccc", lw=0.7, zorder=1)

    usrp_t = np.array([0.0, 1.0])
    ax_srp.plot(usrp_t, usrp_t, color="#888888", lw=1.2, ls="--",
                label="USRP (uniform)", zorder=2)

    usrp_pi = np.interp(tau, [0.0, 1.0], [0.0, 1.0])
    ax_srp.fill_between(tau, pi, usrp_pi,
                        where=(pi >= usrp_pi),
                        color="#2166ac", alpha=0.25, label=r"$A^+$", zorder=3)
    ax_srp.fill_between(tau, pi, usrp_pi,
                        where=(pi < usrp_pi),
                        color="#d73027", alpha=0.25, label=r"$A^-$", zorder=3)

    ax_srp.plot(tau, pi, color=_BSC_PALETTE.get(bsc, "#333333"),
                lw=2.0, label=f"SRP  (BSC = {bsc})", zorder=4)

    # StAC marker
    pi_stac = float(np.interp(stac, tau, pi))
    ax_srp.axvline(stac, color="#d62728", lw=1.0, ls=":", alpha=0.8, zorder=5)
    ax_srp.annotate(
        f"StAC={stac:.2f}",
        xy=(stac, pi_stac),
        xytext=(min(stac + 0.04, 0.88), max(pi_stac - 0.14, 0.04)),
        fontsize=8, color="#d62728",
        arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.7),
    )

    # BSC bit labels in quartile centres
    for k, bit in enumerate(bsc):
        mid = (TAU_BOUNDS[k] + TAU_BOUNDS[k + 1]) / 2
        ax_srp.text(mid, 0.05, bit, ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color="#222222", zorder=6)
        # USRP vs SRP area comparison
        sign = ">" if result[f"bsc_s{k+1}"] else "<="
        ax_srp.text(mid, 0.93,
                    f"A={result[f'area_{k+1}']:.4f}\n{sign} {USRP_AREAS[k]:.4f}",
                    ha="center", va="top", fontsize=6.5, color="#444444", zorder=6)

    ax_srp.set_xlim(0, 1)
    ax_srp.set_ylim(0, 1.05)
    ax_srp.set_xlabel(r"$\tau$  (dimensionless time)", fontsize=9)
    ax_srp.set_ylabel(r"$\pi$  (dimensionless cum. rain)", fontsize=9)
    ax_srp.legend(fontsize=8, loc="upper left")
    ax_srp.set_title(f"Standardized Rainfall Profile  —  Huff Quartile {huff_q}",
                     fontsize=9)

    # --- VWC + rain panel ---
    win_start = ev_start - pd.Timedelta(hours=24)
    win_end   = ev_end   + pd.Timedelta(hours=48)

    ax_rain = ax_vwc.twinx()
    rain_win = rain_1h.loc[win_start:win_end].fillna(0)
    ax_rain.bar(rain_win.index, rain_win.values,
                width=1.0 / 24, color="#6baed6", alpha=0.35, align="edge",
                label="Rain (mm/h)", zorder=1)
    ax_rain.set_ylabel("Rain (mm/h)", fontsize=8, color="#6baed6")
    ax_rain.tick_params(axis="y", labelcolor="#6baed6", labelsize=7)
    ax_rain.set_ylim(bottom=0)

    if vwc_30min is not None:
        vwc_win = vwc_30min.loc[win_start:win_end]
        for col in VWC_COLS:
            if col in vwc_win.columns:
                depth = col.split("|")[0]
                ax_vwc.plot(vwc_win.index, vwc_win[col],
                            color=DEPTH_COLORS[depth], lw=1.5,
                            label=DEPTH_LABELS[depth], zorder=3)

    # Absolute time of peak intensity (StAC mapped back to real time)
    t_stac_abs = ev_start + pd.Timedelta(hours=stac * result["duration_h"])

    ax_vwc.axvspan(ev_start, ev_end, color="#ffffb3", alpha=0.45, zorder=0)
    ax_vwc.axvline(ev_start, color="#888888", lw=1.0, ls="--", alpha=0.7, zorder=2,
                   label="Event start")
    ax_vwc.axvline(t_stac_abs, color="#d62728", lw=1.3, ls=":", alpha=0.9, zorder=2,
                   label=f"StAC={stac:.2f} (peak intensity)")

    # VWC metric annotations: baseline line, peak dot, drying endpoint, metrics table
    _depth_keys = [c.split("|")[0] for c in VWC_COLS]
    if any(f"pre_vwc{d}" in result for d in _depth_keys) and vwc_30min is not None:
        table_rows = ["Depth   | ΔVWC  | Δdyn  | dry   | Tsum "]
        for col in VWC_COLS:
            depth = col.split("|")[0]
            c     = DEPTH_COLORS[depth]
            lbl   = DEPTH_LABELS[depth]
            pre   = result.get(f"pre_vwc{depth}")
            peak  = result.get(f"peak_vwc{depth}")
            delt  = result.get(f"delta_vwc{depth}")
            dpct  = result.get(f"delta_pct_dyn{depth}")
            dr_h  = result.get(f"dry_h{depth}")
            t_pk  = result.get(f"_t_peak{depth}")
            t_dr  = result.get(f"_t_dry{depth}")
            tsum  = result.get(f"temp_sum{depth}")

            if pre is not None:
                ax_vwc.axhline(pre, color=c, lw=0.8, ls=":", alpha=0.5, zorder=2)
            if peak is not None and t_pk is not None and not pd.isna(t_pk):
                ax_vwc.plot(t_pk, peak, "o", color=c, ms=5, mec="white", mew=0.8,
                            zorder=5)
            if t_dr is not None and not pd.isna(t_dr):
                ax_vwc.axvline(t_dr, color=c, lw=0.7, ls="-.", alpha=0.6, zorder=2)

            d_str = f"+{delt:.1f}%" if delt is not None else "n/a   "
            p_str = f"{dpct:.0f}%"  if dpct is not None else "n/a  "
            h_str = f"{dr_h:.0f}h"  if dr_h is not None else "n/a  "
            t_str = f"{tsum:.0f}°" if tsum is not None else "n/a  "
            table_rows.append(
                f"{lbl:<7s} | {d_str:<6s}| {p_str:<6s}| {h_str:<6s}| {t_str}"
            )

        ax_vwc.text(
            0.99, 0.02, "\n".join(table_rows),
            transform=ax_vwc.transAxes,
            va="bottom", ha="right", fontsize=6.5,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.85),
        )

    ax_vwc.set_zorder(2)
    ax_vwc.set_facecolor("none")
    ax_vwc.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_vwc.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax_vwc.xaxis.get_majorticklabels(),
             rotation=30, ha="right", fontsize=8)
    ax_vwc.set_ylabel("VWC (%)", fontsize=9)
    ax_vwc.set_xlabel("Date", fontsize=9)
    ax_vwc.legend(fontsize=8, loc="upper left", ncol=2)
    ax_vwc.set_title("Soil moisture response  (yellow = event window, red dotted = StAC)",
                     fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Per-tree processing
# ---------------------------------------------------------------------------
def _load_tree_timeseries(eui: str):
    """Return (rain_1h, vwc_30min) for a tree, or (None, None) on failure."""
    csv_path = TREES_DIR / eui / "sensor_data.csv"
    if not csv_path.exists():
        return None, None
    try:
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
    except Exception:
        return None, None
    df = df.sort_values("datetime").reset_index(drop=True)
    if RAIN_COL not in df.columns:
        return None, None
    df_idx  = df.set_index("datetime").sort_index()
    rain_1h = df_idx[RAIN_COL].resample("1h").sum().fillna(0.0)
    vwc_30min = None
    if all(c in df.columns for c in VWC_COLS):
        vwc_raw   = df[["datetime"] + VWC_COLS].set_index("datetime").sort_index()
        vwc_30min = vwc_raw.resample("30min").mean().ffill(limit=2)
    return rain_1h, vwc_30min


def _load_tree_full(eui: str):
    """
    Return (rain_1h, vwc_30min, rs1h, dr) ready for _extract_vwc_metrics,
    or (None, None, None, None) if any required data channel is missing.
    rs1h includes temperature for temp_sum computation.
    dr is the VWC dynamic range dict.
    """
    csv_path = TREES_DIR / eui / "sensor_data.csv"
    if not csv_path.exists():
        return None, None, None, None
    try:
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
    except Exception:
        return None, None, None, None
    df = df.sort_values("datetime").reset_index(drop=True)
    if RAIN_COL not in df.columns or not all(c in df.columns for c in VWC_COLS):
        return None, None, None, None
    df_idx  = df.set_index("datetime").sort_index()
    rain_1h = df_idx[RAIN_COL].resample("1h").sum().fillna(0.0)
    temp_1h = (df_idx[TEMP_COL].resample("1h").mean().ffill(limit=1)
               if TEMP_COL in df.columns
               else pd.Series(np.nan, index=rain_1h.index, name=TEMP_COL))
    rs1h      = pd.DataFrame({RAIN_COL: rain_1h, TEMP_COL: temp_1h})
    vwc_raw   = df[["datetime"] + VWC_COLS].set_index("datetime").sort_index()
    vwc_30min = vwc_raw.resample("30min").mean().ffill(limit=2)
    dr        = _dynamic_range(vwc_30min)
    return rain_1h, vwc_30min, rs1h, dr


def process_tree(eui: str) -> pd.DataFrame:
    csv_path = TREES_DIR / eui / "sensor_data.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
    except Exception:
        return pd.DataFrame()

    df = df.sort_values("datetime").reset_index(drop=True)
    if RAIN_COL not in df.columns:
        return pd.DataFrame()

    # Resample rain (sum) and temperature (mean) separately to 1 h
    df_idx  = df.set_index("datetime").sort_index()
    rain_1h = df_idx[RAIN_COL].resample("1h").sum().fillna(0.0)
    if TEMP_COL in df.columns:
        temp_1h = df_idx[TEMP_COL].resample("1h").mean().ffill(limit=1)
    else:
        temp_1h = pd.Series(np.nan, index=rain_1h.index, name=TEMP_COL)
    rs1h = pd.DataFrame({RAIN_COL: rain_1h, TEMP_COL: temp_1h})

    # Resample VWC to 30 min and compute dynamic range
    has_vwc   = all(c in df.columns for c in VWC_COLS)
    vwc_30min = None
    dr        = None
    if has_vwc:
        vwc_raw   = df[["datetime"] + VWC_COLS].set_index("datetime").sort_index()
        vwc_30min = vwc_raw.resample("30min").mean().ffill(limit=2)
        dr        = _dynamic_range(vwc_30min)

    events = detect_rain_events(rain_1h)
    if not events:
        return pd.DataFrame()

    results = []
    for ev_start, ev_end, _ in events:
        r = compute_bsc(ev_start, ev_end, rain_1h)
        if r is None:
            continue
        r["eui"] = eui
        if has_vwc and dr is not None:
            r.update(_extract_vwc_metrics(
                ev_start, r["ev_end"], r["stac"], r["duration_h"],
                vwc_30min, rs1h, dr,
            ))
        results.append(r)

    if not results:
        return pd.DataFrame()

    # Output subfolder
    bsc_dir = TREES_DIR / eui / "bsc_events"
    bsc_dir.mkdir(exist_ok=True)

    # 2 example plots — always regenerate to include latest VWC annotations
    sorted_r = sorted(results, key=lambda r: r["total_mm"], reverse=True)
    for n, r in enumerate(sorted_r[:N_EXAMPLES], 1):
        fname = (f"bsc_example_{n:02d}"
                 f"_{r['ev_start'].strftime('%Y%m%d')}"
                 f"_{r['bsc']}.png")
        try:
            _plot_bsc_example(r, eui, rain_1h, vwc_30min, bsc_dir / fname)
        except Exception as exc:
            tqdm.write(f"  {eui}: example {n} plot failed — {exc}")

    # Build output DataFrame (drop private annotation timestamps and plot arrays)
    rows = [{k: v for k, v in r.items() if not k.startswith("_")}
            for r in results]
    df_out = pd.DataFrame(rows)
    df_out.to_csv(bsc_dir / "bsc_events.csv", index=False)

    return df_out


# ---------------------------------------------------------------------------
# 5. Global distribution chart (Figure 4 equivalent)
# ---------------------------------------------------------------------------
def plot_bsc_quartile_distribution(df_all: pd.DataFrame, out_path: Path):
    """
    For each Huff quartile (I–IV), show the % of total events per BSC code.
    All observed codes are shown individually, sorted by overall frequency.
    """
    if df_all.empty or "bsc" not in df_all.columns:
        return

    total = len(df_all)
    bsc_counts = df_all["bsc"].value_counts()
    codes = bsc_counts.index.tolist()  # all codes, most frequent first

    # Quartile labels with total share
    q_sizes = [df_all["huff_q"].eq(q).sum() for q in range(1, 5)]
    q_labels = [f"Quartile {r} ({q_sizes[r-1]/total*100:.0f} %)"
                for r in range(1, 5)]

    # Build matrix: rows = BSC codes, cols = quartile, value = % of ALL events
    matrix = np.zeros((len(codes), 4))
    for i, code in enumerate(codes):
        for q in range(1, 5):
            mask = (df_all["bsc"] == code) & (df_all["huff_q"] == q)
            matrix[i, q - 1] = mask.sum() / total * 100

    # Stacked bar chart: one stack per quartile, segments = BSC codes
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(4)
    bottom = np.zeros(4)

    for i, code in enumerate(codes):
        color = _BSC_PALETTE.get(code, "#dddddd")
        vals  = matrix[i]
        bars  = ax.bar(x, vals, bottom=bottom, color=color,
                       label=code, edgecolor="white", linewidth=0.4, zorder=2)
        # Annotate segments >= 1.5 %
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if v >= 1.5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom[j] + v / 2,
                    f"{v:.1f}",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if v > 3 else "#222222",
                )
        bottom += vals

    # Total bar height label
    for j in range(4):
        total_h = bottom[j]
        ax.text(x[j], total_h + 0.4, f"{total_h:.1f} %",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, fontsize=10)
    ax.set_ylabel("Percentage of events [%]", fontsize=10)
    ax.set_title(
        f"BSC Distribution by Huff Quartile   "
        f"(n = {total} events, >= {MIN_EVENT_MM} mm, {DRY_GAP_H} h dry gap)",
        fontsize=11,
    )
    ax.legend(
        title="BSC", fontsize=8, title_fontsize=8,
        ncol=min(len(codes), 8),
        bbox_to_anchor=(0.5, -0.18), loc="upper center",
    )
    ax.set_ylim(0, bottom.max() * 1.12)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart -> {out_path}")


# ---------------------------------------------------------------------------
# 6. BSC scatter plots: total_mm vs duration_h / max_intensity_mmh
# ---------------------------------------------------------------------------
def plot_bsc_scatter(df_all: pd.DataFrame, out_path: Path):
    """
    Two scatter plots coloured by BSC code:
      Left:  event sum [mm]  vs. event duration [h]
      Right: event sum [mm]  vs. max hourly intensity [mm/h]
    A horizontal reference line marks LITERATURE_MM (12.7 mm).
    """
    if df_all.empty or "bsc" not in df_all.columns:
        return
    if "max_intensity_mmh" not in df_all.columns or "duration_h" not in df_all.columns:
        return

    n_total   = len(df_all)
    codes_ord = df_all["bsc"].value_counts().index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_dur, ax_int = axes

    for code in codes_ord:
        sub   = df_all[df_all["bsc"] == code]
        color = _BSC_PALETTE.get(code, "#cccccc")
        kw    = dict(color=color, s=8, alpha=0.65, linewidths=0,
                     label=f"{code} (n={len(sub)})", zorder=3)
        ax_dur.scatter(sub["total_mm"], sub["duration_h"],  **kw)
        ax_int.scatter(sub["total_mm"], sub["max_intensity_mmh"], **kw)

    for ax in axes:
        ax.axvline(LITERATURE_MM, color="#222222", lw=1.2, ls="--", zorder=4,
                   label=f"{LITERATURE_MM} mm (literature threshold)")
        ax.set_xlabel("Event total rainfall [mm]", fontsize=10)
        ax.grid(alpha=0.25, zorder=0)

    ax_dur.set_ylabel("Event duration [h]", fontsize=10)
    ax_int.set_ylabel("Max hourly intensity [mm/h]", fontsize=10)
    ax_dur.set_title("Event sum vs. Duration", fontsize=10)
    ax_int.set_title("Event sum vs. Max intensity", fontsize=10)

    handles, labels = ax_dur.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="BSC code", fontsize=7.5, title_fontsize=8,
        ncol=min(len(codes_ord) + 1, 9),
        bbox_to_anchor=(0.5, -0.02), loc="upper center",
    )
    fig.suptitle(
        f"BSC Rain Events — Rainfall Characteristics   "
        f"(n = {n_total}, >= {MIN_EVENT_MM} mm)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.10, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart -> {out_path}")


# ---------------------------------------------------------------------------
# 7. BSC code example gallery
# ---------------------------------------------------------------------------
def plot_bsc_code_examples(df_all: pd.DataFrame, out_dir: Path,
                           n_per_code: int = N_EXAMPLES_GLOBAL,
                           min_impact_pct: float = 20.0):
    """
    For each observed BSC code, save the n_per_code heaviest qualifying events as PNG.
    Output:  out_dir/{bsc}_{rank:02d}_{YYYYMMDD}_{eui}.png

    Qualifying events must:
      (1) Have data in all three VWC channels  (delta_vwc-10/-30/-45 all non-NaN)
      (2) Show >= min_impact_pct % of dynamic range in at least one channel

    VWC peak and drying timestamps are re-computed from raw timeseries so that
    full annotations (peak dots, drying lines, baselines) are drawn correctly.
    """
    if df_all.empty or "bsc" not in df_all.columns:
        return

    df_all = df_all.copy()
    df_all["bsc"]      = df_all["bsc"].astype(str).str.zfill(4)
    df_all["ev_start"] = pd.to_datetime(df_all["ev_start"])
    df_all["ev_end"]   = pd.to_datetime(df_all["ev_end"])

    # Filter 1: all three VWC channels have data
    _depths = ["-10", "-30", "-45"]
    vwc_complete = pd.Series(True, index=df_all.index)
    for _d in _depths:
        _col = f"delta_vwc{_d}"
        if _col in df_all.columns:
            vwc_complete &= df_all[_col].notna()
        else:
            vwc_complete[:] = False

    # Filter 2: at least one depth >= min_impact_pct % of dynamic range
    _pct_cols = [f"delta_pct_dyn{_d}" for _d in _depths
                 if f"delta_pct_dyn{_d}" in df_all.columns]
    if _pct_cols:
        has_impact = df_all[_pct_cols].max(axis=1) >= min_impact_pct
    else:
        has_impact = pd.Series(False, index=df_all.index)

    df_filt = df_all[vwc_complete & has_impact].reset_index(drop=True)
    print(f"\n  Example filter: {len(df_filt)}/{len(df_all)} events qualify "
          f"(100% VWC data + >={min_impact_pct:.0f}% dyn range)")

    out_dir.mkdir(parents=True, exist_ok=True)
    codes = sorted(df_filt["bsc"].unique())
    print(f"  Generating quality-filtered BSC examples ({n_per_code}/code) -> {out_dir}")

    for code in codes:
        subset = (
            df_filt[df_filt["bsc"] == code]
            .sort_values("total_mm", ascending=False)
            .head(n_per_code)
        )
        for rank, (_, row) in enumerate(subset.iterrows(), 1):
            eui      = str(row["eui"]).zfill(16)
            ev_start = row["ev_start"]
            ev_end   = row["ev_end"]

            rain_1h, vwc_30min, rs1h, dr = _load_tree_full(eui)
            if rain_1h is None:
                tqdm.write(f"  BSCExamples {code} #{rank}: cannot load {eui}")
                continue

            # Regenerate _tau/_pi arrays — not saved in CSV
            r = compute_bsc(ev_start, ev_end, rain_1h)
            if r is None:
                continue

            # Re-run metric extraction to recover actual _t_peak / _t_dry timestamps
            vwc_metrics = _extract_vwc_metrics(
                ev_start, ev_end, r["stac"], r["duration_h"],
                vwc_30min, rs1h, dr,
            )
            r.update(vwc_metrics)

            # Fill any remaining fields from the saved CSV row
            for col in row.index:
                if col not in r:
                    val = row[col]
                    r[col] = None if (isinstance(val, float) and np.isnan(val)) else val

            fname = (f"{code}_{rank:02d}"
                     f"_{ev_start.strftime('%Y%m%d')}"
                     f"_{eui}.png")
            try:
                _plot_bsc_example(r, eui, rain_1h, vwc_30min, out_dir / fname)
            except Exception as exc:
                tqdm.write(f"  BSCExamples {code} #{rank}: {exc}")

    print(f"  Examples saved -> {out_dir}")


# ---------------------------------------------------------------------------
# 8. RF dataset builder — joins bsc_events_all.csv with treeLocations metadata
# ---------------------------------------------------------------------------

# Shapefile DBF names are limited to 10 chars; geopandas appends _1/_2 for dupes.
_SHP_RENAME = {
    "sealedSurf": "sealedSurf_2m5",
    "sealedSu_1": "sealedSurf_5m",
    "sealedSu_2": "sealedSurf_7m5",
    "greenDeta2": "greenDeta_2m5",
    "greenDeta5": "greenDeta_5m",
    "greenDeta7": "greenDeta_7m5",
    "greenAtta2": "greenAtta_2m5",
    "greenAtta5": "greenAtta_5m",
    "greenAtta7": "greenAtta_7m5",
    "buildings2": "buildings_2m5",
    "buildings5": "buildings_5m",
    "buildings7": "buildings_7m5",
    "slope":      "slope_deg",
    "flotAcc":    "flowAccumulation",
    "amsl":       "elevation_m",
    "DTGW":       "depthToGroundwater_m",
    "SVF":        "skyViewFactor",
    "conductiv":  "conductivity",
    "saltCont":   "saltContent",
    "ammNSolubl": "ammNSoluble",
    "nitrNSolub": "nitrNSoluble",
    "phSoluble":  "pHSoluble",
    "crownDiam":  "crownDiam_m",
    "stemDiam":   "stemDiam_m",
    "assesDate":  "assessDate",
    "senInsDate": "sensorInstallDate",
    "senRmvDate": "sensorRemoveDate",
    "cutDwnDate": "cutDownDate",
    "germDate":   "germinationDate",
    "plantDate":  "plantingDate",
    "soilDate":   "soilSampleDate",
}


def build_rf_dataset(df_all: pd.DataFrame) -> None:
    """
    Join df_all (from bsc_events_all.csv) with treeLocations shapefile metadata
    and write TreeTabularData/rf_dataset.csv.
    Skipped silently if geopandas is not installed.
    """
    if not _HAS_GEOPANDAS:
        print("  [SKIP] geopandas not installed — rf_dataset.csv not written")
        return

    shp_files = sorted(GIS_DIR.glob("**/VectorLayers/treeLocations.shp"))
    if not shp_files:
        print(f"  [SKIP] No treeLocations.shp found under {GIS_DIR}")
        return

    frames = []
    for shp in shp_files:
        gdf = gpd.read_file(shp)
        if gdf.empty:
            continue
        df_shp = gdf.drop(columns=["geometry"]).copy()
        df_shp["city_folder"] = shp.parts[-3]
        frames.append(df_shp)

    if not frames:
        print("  [SKIP] All treeLocations.shp files are empty.")
        return

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        meta = pd.concat(frames, ignore_index=True)
    meta = meta.rename(columns=_SHP_RENAME)

    # Deduplicate: keep the most recently assessed entry per EUI
    if "assessDate" in meta.columns:
        meta["assessDate"] = pd.to_datetime(meta["assessDate"], errors="coerce")
        meta = meta.sort_values("assessDate", ascending=False, na_position="last")
    meta = meta.drop_duplicates(subset="devEUI", keep="first").reset_index(drop=True)
    print(f"  Tree metadata: {len(meta)} unique EUIs from {len(frames)} shapefile(s)")

    df_rf = df_all.merge(
        meta.rename(columns={"devEUI": "eui"}),
        on="eui",
        how="left",
        suffixes=("", "_meta"),
    )

    n_matched   = int(df_rf["project"].notna().sum()) if "project" in df_rf.columns else 0
    n_unmatched = len(df_all) - n_matched
    if n_unmatched > 0:
        missing = sorted(df_rf.loc[df_rf["project"].isna(), "eui"].unique())
        print(f"  [WARN] {n_unmatched} events ({len(missing)} EUIs) "
              f"have no shapefile entry")

    OUT_RF.parent.mkdir(parents=True, exist_ok=True)
    df_rf.to_csv(OUT_RF, index=False)
    print(f"  RF dataset: {len(df_rf)} rows × {len(df_rf.columns)} cols  ->  {OUT_RF}")


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  BSC Rain Event Classifier  (Terranova & Iaquinta 2011)")
    print(f"  Min event: {MIN_EVENT_MM} mm  |  Dry gap: {DRY_GAP_H} h")
    print("=" * 60)

    euids = sorted(
        d.name for d in TREES_DIR.iterdir()
        if d.is_dir() and (d / "sensor_data.csv").exists()
    )
    print(f"\n  Trees with sensor_data.csv: {len(euids)}")

    all_frames = []
    n_events = 0
    n_trees_ok = 0

    pbar = tqdm(euids, desc="Processing trees")
    for eui in pbar:
        pbar.set_postfix(eui=eui, refresh=False)
        try:
            df = process_tree(eui)
        except Exception as exc:
            tqdm.write(f"  {eui}: ERROR — {exc}")
            continue
        if not df.empty:
            all_frames.append(df)
            n_events  += len(df)
            n_trees_ok += 1

    if not all_frames:
        print("\n  No events found. Check RAIN_COL and MIN_EVENT_MM threshold.")
        return

    df_all = pd.concat(all_frames, ignore_index=True)
    df_all.to_csv(OUT_ALL, index=False)
    print(f"\n  Events: {n_events}  |  Trees with events: {n_trees_ok}")
    print(f"  Saved -> {OUT_ALL}")

    # BSC frequency summary
    bsc_freq = df_all["bsc"].value_counts(normalize=True).mul(100).round(1)
    print("\n  Top BSC frequencies:")
    for code, pct in bsc_freq.head(10).items():
        print(f"    {code}: {pct:.1f} %")

    # Huff quartile distribution
    q_freq = df_all["huff_q"].value_counts().sort_index()
    print("\n  Huff quartile counts:")
    for q, cnt in q_freq.items():
        print(f"    Q{q}: {cnt}  ({cnt/n_events*100:.1f} %)")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plot_bsc_quartile_distribution(df_all, REPORT_DIR / "chart_bsc_quartiles.png")
    plot_bsc_scatter(df_all, REPORT_DIR / "chart_bsc_scatter.png")

    BSC_EXAMPLES_DIR = Path(__file__).resolve().parent / "BSCRainExamples"
    plot_bsc_code_examples(df_all, BSC_EXAMPLES_DIR, n_per_code=N_EXAMPLES_GLOBAL)

    print("\n  Building RF dataset ...")
    build_rf_dataset(df_all)

    print("\n  Done.")


if __name__ == "__main__":
    main()
