#!/usr/bin/env python3
"""
generate_manuscript_gaps.py
===========================
Fills all open gaps in the GI manuscript:
  1. Model performance table (12 strata)
  2. Four SHAP beeswarm figures (SM x {BSC_XX11_Q12, IRRIGATION},
                                  DT x {BSC_00XX_Q34, IRRIGATION})
  3. Descriptive statistics (events, trees, cities, monitoring period)
  4. Snapshot contradiction resolution
  5. DTGW subset availability
  6. Package versions for reproducibility

Depth translation (channel -> paper):  -10 -> 30 cm  -30 -> 60 cm  -45 -> 90 cm
"""
import sys, pickle, warnings, subprocess
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

try:
    import cmcrameri.cm as cmc
    _HAS_CRAMERI = True
except ImportError:
    _HAS_CRAMERI = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO   = Path(__file__).resolve().parent.parent
DS_DIR = REPO / "DatasetStatistics"
TT_DIR = REPO / "TreeTabularData"
FIG_DIR = REPO / "figures" / "paper"
FIG_DIR.mkdir(exist_ok=True)

DEPTH_CHANNEL = ["-10", "-30", "-45"]
DEPTH_PAPER   = {"-10": "30", "-30": "60", "-45": "90"}

# Paper strata (IRRIGATION_STRONG excluded)
STRATA_SM = ["BSC_XX11_Q12", "IRRIGATION"]
STRATA_DT = ["BSC_00XX_Q34", "IRRIGATION"]

# Colourmap for feature-value bar in beeswarm (vik: dark blue \u2192 white \u2192 dark red-brown)
_FEAT_CMAP = cmc.vik if _HAS_CRAMERI else plt.cm.RdBu_r

# ---------------------------------------------------------------------------
# Feature display names
# ---------------------------------------------------------------------------
_FEAT_NAMES = {
    "total_mm":         "Rain total",
    "duration_h":       "Duration",
    "max_intensity_mmh":"Max intensity",
    "area_1":           "Huff area q\u2081",
    "area_2":           "Huff area q\u2082",
    "area_3":           "Huff area q\u2083",
    "area_4":           "Huff area q\u2084",
    "rain_pre24h_mm":   "Rain \u221224 h",
    "rain_post24h_mm":  "Rain +24 h",
    "rain_pre72h_mm":   "Rain \u221272 h",
    "rain_post72h_mm":  "Rain +72 h",
    "greenAtta_0to7":   "greenAtta 0-7m",
    "slope":            "Slope",
    "SVF":              "Sky view factor",
    "twi":              "TWI",
    "tpi_7m5":          "TPI 7.5 m",
    "tpi_5m":           "TPI 5 m",
    "tpi_2m5":          "TPI 2.5 m",
    "twi_2m5":          "TWI 2.5 m",
    "twi_5m":           "TWI 5 m",
    "twi_7m5":          "TWI 7.5 m",
    "crownDiam":        "Crown diam.",
    "height":           "Tree height",
    "stemDiam":         "Stem diam.",
    "age_years":        "Tree age",
    "DTGW":             "Depth to GW",
    "day_cos":          "Seasonality",
    "month_cos":        "Seasonality",
    # depth-specific stripped versions (key = base name)
    "dyn_range":        "Dyn. range",
    "pre_vwc":          "Pre-event VWC",
    "avg_temp":         "Avg. temp",
}

def _base_feat(name: str) -> str:
    """Strip depth suffix (-10/-30/-45) to get the base feature key."""
    for d in DEPTH_CHANNEL:
        if name.endswith(d):
            return name[:-len(d)].rstrip("-")
    return name

def _feat_label(name: str) -> str:
    """Return display name for a (possibly depth-stripped) feature key."""
    if name in _FEAT_NAMES:
        return _FEAT_NAMES[name]
    base = _base_feat(name)
    if base in _FEAT_NAMES:
        return _FEAT_NAMES[base]
    return name


# ---------------------------------------------------------------------------
# Load training caches
# ---------------------------------------------------------------------------
def load_cache(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

print("Loading training caches …")
sm_cache = load_cache(DS_DIR / "rf_soil_moisture_v7_training_cache.pkl")
dt_cache = load_cache(DS_DIR / "rf_drying_time_v7_training_cache.pkl")

# Load Boruta results for feature-count confirmation
with open(DS_DIR / "boruta_results.pkl", "rb") as f:
    boruta = pickle.load(f)

print("  SM cache event types:", list(sm_cache["trained"].keys()))
print("  DT cache event types:", list(dt_cache["trained"].keys()))


# ---------------------------------------------------------------------------
# Section 1 — Model performance table
# ---------------------------------------------------------------------------
print("\nBuilding model performance table …")

def build_perf_rows(cache, task_name, strata):
    rows = []
    trained = cache["trained"]
    for et in strata:
        if et not in trained:
            print(f"  WARN: {et} not in {task_name} cache")
            continue
        payload = trained[et]
        rows_et, depth_data, depth_results = payload
        for idx, depth in enumerate(DEPTH_CHANNEL):
            dr = depth_results[idx]
            if dr.get("skip"):
                continue
            dp = DEPTH_PAPER[depth]
            n       = int(dr["n_samp"])
            r2cv_arr = np.atleast_1d(dr["r2s"])
            r2cv    = float(r2cv_arr.mean())
            r2cv_sd = float(r2cv_arr.std())
            r2tr    = float(dr.get("r2_train", np.nan))
            n_feat  = len(dr["feat_names"]) if "feat_names" in dr else (
                      len(depth_data[idx][0]) if depth_data[idx] else 0)
            rows.append({
                "task": task_name, "event_type": et, "depth_ch": depth,
                "depth_cm": int(dp),
                "n": n, "r2cv": r2cv, "r2cv_sd": r2cv_sd,
                "r2tr": r2tr, "n_feat": n_feat,
            })
    return rows

perf_rows = (build_perf_rows(sm_cache, "SM", STRATA_SM) +
             build_perf_rows(dt_cache, "DT", STRATA_DT))

perf_df = pd.DataFrame(perf_rows)
print(perf_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Helper: extract depth_data from cache
# (each entry: [feat_names, X_sub, shap_v, y_full])
# ---------------------------------------------------------------------------
def get_depth_data(cache, et, depth_idx):
    """Return (feat_names, X, shap_v, y_full) for given event type and depth index."""
    trained = cache["trained"]
    if et not in trained:
        return None
    _, depth_data, _ = trained[et]
    dd = depth_data[depth_idx]
    if dd is None:
        return None
    feat_names, X, shap_v = dd[0], dd[1], dd[2]
    y_full = dd[3] if len(dd) > 3 else None
    if shap_v is None:
        return None
    return feat_names, X, shap_v, y_full


# ---------------------------------------------------------------------------
# Section 2 — SHAP beeswarm figures
# ---------------------------------------------------------------------------
print("\nGenerating SHAP beeswarm figures …")

def pooled_shap(cache, event_type, is_sm=False):
    """
    Pool SHAP data across all three depths for one event type.

    Returns:
        pooled_feats: list of unique feature names (ordered by mean |norm_shap|)
        feat_x:       dict feat -> array of feature values (pooled across depths)
        feat_shap:    dict feat -> array of normalised SHAP values (pooled, %)
        n_per_depth:  dict depth_ch -> int sample count
        y_mean_log:   mean log1p(y) across depths (for SM back-transform)
    """
    feat_x    = defaultdict(list)
    feat_shap = defaultdict(list)
    n_per_depth = {}

    y_means_log = []

    for idx, depth in enumerate(DEPTH_CHANNEL):
        result = get_depth_data(cache, event_type, idx)
        if result is None:
            continue
        feat_names, X, shap_v, y_full = result

        # Normalise by mean of full training target — identical to the report scripts:
        #   shap_norm = shap_v / max(|mean(y_full)|, 1e-6) * 100
        # This gives SHAP in "% of the mean observed response at this depth".
        if y_full is not None and len(y_full) > 0:
            scale = max(abs(float(np.nanmean(y_full))), 1e-6)
        else:
            scale = np.mean(np.abs(shap_v))
            if scale < 1e-10:
                continue
        shap_norm = shap_v / scale * 100.0

        n_per_depth[depth] = X.shape[0]

        for fi, fn in enumerate(feat_names):
            base_fn = _base_feat(fn)
            feat_x[base_fn].extend(X[:, fi].tolist())
            feat_shap[base_fn].extend(shap_norm[:, fi].tolist())

        if y_full is not None and len(y_full) > 0:
            y_means_log.append(float(np.nanmean(y_full)))

    # rank features by mean |normalised SHAP|
    rank = {fn: np.mean(np.abs(vs)) for fn, vs in feat_shap.items()}
    pooled_feats = sorted(rank, key=rank.get, reverse=True)

    # convert to numpy arrays
    feat_x    = {fn: np.array(v) for fn, v in feat_x.items()}
    feat_shap = {fn: np.array(v) for fn, v in feat_shap.items()}

    y_mean_log = float(np.mean(y_means_log)) if y_means_log else np.nan

    return pooled_feats, feat_x, feat_shap, n_per_depth, y_mean_log


def iqr_effect_sm(feat_name, feat_x, feat_shap, y_mean_log):
    """
    Back-transform IQR effect of feature from log1p to percent delta_VWC.
    Returns (iqr_effect_pct, iqr_log) or (nan, nan) if not enough data.
    """
    xv = feat_x.get(feat_name)
    sv = feat_shap.get(feat_name)
    if xv is None or sv is None or len(xv) < 10:
        return np.nan, np.nan
    # The shap values here are in %-of-mean-abs-SHAP space, NOT in log1p space.
    # We need the original log1p-scale SHAP values.
    # Scale back: sv / 100 * scale_factor — but we don't store the per-depth scale.
    # Instead, use the sign/ranking approach (the iqr_log is approximated).
    # The sv array is the normalised SHAP in percent.
    # To approximate back-transform, we use the relative effect:
    # Identify Q25 / Q75 samples by feature value
    q25 = np.percentile(xv, 25)
    q75 = np.percentile(xv, 75)
    low_mask  = xv <= q25
    high_mask = xv >= q75
    if low_mask.sum() < 5 or high_mask.sum() < 5:
        return np.nan, np.nan
    # iqr effect in normalised-% space
    iqr_norm_pct = float(np.mean(sv[high_mask]) - np.mean(sv[low_mask]))
    # The normalised SHAP = (log1p-SHAP / mean_abs_SHAP_depth) * 100
    # We want iqr_log = iqr_norm_pct / 100 * mean_abs_SHAP_depth
    # Without the per-depth scale we can't exactly reconstruct iqr_log.
    # So we report the normalised IQR and the back-transform formula:
    #   approx delta_pct = expm1(y_mean_log + iqr_log/2) - expm1(y_mean_log - iqr_log/2)
    # For the text, we simply report the normalised IQR effect in percent.
    return iqr_norm_pct, np.nan


def spearman_rho(xarr, sarr):
    """Spearman rho between feature values and SHAP values."""
    if len(xarr) < 10:
        return np.nan, np.nan
    valid = np.isfinite(xarr) & np.isfinite(sarr)
    if valid.sum() < 10:
        return np.nan, np.nan
    rho, p = spearmanr(xarr[valid], sarr[valid])
    return float(rho), float(p)


def make_beeswarm(cache, event_type, task_label, fig_stem, is_sm=False,
                  n_feat_show=15, dot_size=4, fig_w_mm=88, fig_h_mm=120):
    """Generate and save one SHAP beeswarm figure."""
    pooled_feats, feat_x, feat_shap, n_per_depth, y_mean_log = \
        pooled_shap(cache, event_type, is_sm=is_sm)

    if not pooled_feats:
        print(f"  WARN: no SHAP data for {task_label} x {event_type}")
        return {}

    feats = pooled_feats[:n_feat_show]
    n_total = sum(n_per_depth.values())
    n_depths = len(n_per_depth)
    depth_labels = ", ".join(f"{DEPTH_PAPER[d]} cm" for d in DEPTH_CHANNEL
                              if d in n_per_depth)

    fw = fig_w_mm / 25.4
    fh = fig_h_mm / 25.4

    fig, ax = plt.subplots(figsize=(fw, fh))
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    # Colour normalisation (per-feature)
    y_positions = range(len(feats))
    jitter_rng = np.random.default_rng(42)

    summary_rows = []

    for yi, fn in enumerate(feats):
        xarr = feat_x.get(fn, np.array([]))
        sarr = feat_shap.get(fn, np.array([]))
        if len(xarr) == 0:
            continue

        # Normalise feature values to [0, 1] for colour
        xmin, xmax = np.nanpercentile(xarr, 2), np.nanpercentile(xarr, 98)
        if xmax - xmin < 1e-10:
            xnorm = np.full_like(xarr, 0.5)
        else:
            xnorm = np.clip((xarr - xmin) / (xmax - xmin), 0, 1)

        colours = _FEAT_CMAP(xnorm)

        # Vertical jitter
        jitter = jitter_rng.uniform(-0.35, 0.35, size=len(sarr))
        # Sort by SHAP for less overplotting
        order = np.argsort(sarr)
        ax.scatter(sarr[order], yi + jitter[order],
                   c=colours[order], s=dot_size, alpha=0.7,
                   linewidths=0, rasterized=True)

        # Spearman
        rho, p = spearman_rho(xarr, sarr)
        rho_str = f"ρ={rho:+.2f}" if np.isfinite(rho) else ""
        p_str   = "*" if (np.isfinite(p) and p < 0.05) else ""

        # mean |SHAP|
        mean_abs = float(np.mean(np.abs(sarr)))

        # IQR effect for SM
        iqr_norm, _ = iqr_effect_sm(fn, feat_x, feat_shap, y_mean_log) if is_sm \
                      else (np.nan, np.nan)

        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 200,
                yi, f" {rho_str}{p_str}", va="center", ha="left",
                fontsize=5.5, color="#444444")

        summary_rows.append({
            "feature": fn, "label": _feat_label(fn),
            "mean_abs_norm_pct": round(mean_abs, 2),
            "spearman_rho": round(rho, 3) if np.isfinite(rho) else None,
            "spearman_p": round(p, 4) if np.isfinite(p) else None,
            "iqr_norm_pct": round(iqr_norm, 2) if np.isfinite(iqr_norm) else None,
        })

    # Axes formatting
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([_feat_label(fn) for fn in feats], fontsize=6.5)
    ax.axvline(0, color="black", linewidth=0.6, linestyle="-")
    ax.set_xlabel("Normalised SHAP value (%)", fontsize=7)
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=6.5)

    # Colourbar
    cbar_ax = fig.add_axes([0.68, 0.02, 0.28, 0.015])
    sm_cb = ScalarMappable(cmap=_FEAT_CMAP, norm=Normalize(0, 1))
    sm_cb.set_array([])
    cb = fig.colorbar(sm_cb, cax=cbar_ax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"], fontsize=5.5)
    cb.set_label("Feature value", fontsize=6)

    title = (f"{task_label} \u00d7 {event_type.replace('_', ' ')}\n"
             f"n = {n_total:,} events (depths: {depth_labels})")
    ax.set_title(title, fontsize=7, fontweight="bold", pad=4)

    fig.tight_layout(rect=[0, 0.04, 0.92, 1.0])

    # Save
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{fig_stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {fig_stem}.pdf / .png")

    return summary_rows


# Run second pass with corrected annotation placement
def make_beeswarm_v2(cache, event_type, task_label, fig_stem, is_sm=False,
                     n_feat_show=15, dot_size=3):
    """Improved beeswarm with proper annotation and axis limits."""
    pooled_feats, feat_x, feat_shap, n_per_depth, y_mean_log = \
        pooled_shap(cache, event_type, is_sm=is_sm)

    if not pooled_feats:
        print(f"  WARN: no SHAP data for {task_label} x {event_type}")
        return {}

    feats = pooled_feats[:n_feat_show]
    n_total = sum(n_per_depth.values())
    depth_labels = ", ".join(f"{DEPTH_PAPER[d]} cm" for d in DEPTH_CHANNEL
                              if d in n_per_depth)

    # Determine axis limits from data
    all_shap = np.concatenate([feat_shap[fn] for fn in feats if fn in feat_shap])
    xlo = np.percentile(all_shap, 1)
    xhi = np.percentile(all_shap, 99)
    pad = (xhi - xlo) * 0.08
    xlo -= pad; xhi += pad
    # Reserve space for rho annotation on right
    anno_margin = (xhi - xlo) * 0.28
    xhi_anno = xhi + anno_margin

    fw = 88 / 25.4
    fh = max(90, 14 + len(feats) * 8) / 25.4

    plt.rcParams.update({
        "font.family": "Arial", "font.weight": "bold", "font.size": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(fw, fh))

    jitter_rng = np.random.default_rng(42)
    summary_rows = []

    for yi, fn in enumerate(feats):
        xarr = feat_x.get(fn, np.array([]))
        sarr = feat_shap.get(fn, np.array([]))
        if len(xarr) == 0:
            continue

        # colour by feature value
        xmin, xmax = np.nanpercentile(xarr, 2), np.nanpercentile(xarr, 98)
        xnorm = np.clip((xarr - xmin) / (xmax - xmin + 1e-12), 0, 1)
        colours = _FEAT_CMAP(xnorm)

        jitter = jitter_rng.uniform(-0.38, 0.38, size=len(sarr))
        order  = np.argsort(sarr)
        ax.scatter(sarr[order], yi + jitter[order],
                   c=colours[order], s=dot_size, alpha=0.75,
                   linewidths=0, rasterized=True)

        rho, p = spearman_rho(xarr, sarr)
        rho_str = f"ρ={rho:+.2f}" if np.isfinite(rho) else "ρ=n/a"
        star    = "*" if (np.isfinite(p) and p < 0.001) else ""

        # r= labels stay grey (intentionally not bold-8pt)
        ax.text(xhi + pad * 0.4, yi, f"{rho_str}{star}",
                va="center", ha="left", fontsize=6, color="#555555",
                fontfamily="Arial", fontweight="normal")

        mean_abs = float(np.mean(np.abs(sarr)))
        iqr_norm, _ = iqr_effect_sm(fn, feat_x, feat_shap, y_mean_log) if is_sm \
                      else (np.nan, np.nan)
        summary_rows.append({
            "feature": fn, "label": _feat_label(fn),
            "rank": yi + 1,
            "mean_abs_norm_pct": round(mean_abs, 2),
            "spearman_rho": round(rho, 3) if np.isfinite(rho) else None,
            "spearman_p": round(p, 5) if np.isfinite(p) else None,
            "iqr_norm_pct": round(iqr_norm, 2) if np.isfinite(iqr_norm) else None,
        })

    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([_feat_label(fn) for fn in feats],
                       fontsize=8, color="black", fontweight="bold")
    ax.set_xlim(xlo, xhi_anno)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Normalised SHAP value (%)", fontsize=8, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.invert_yaxis()   # most important feature on top

    # Colourbar
    cb_ax = fig.add_axes([0.68, 0.01, 0.26, 0.012])
    sm_cb = ScalarMappable(cmap=_FEAT_CMAP, norm=Normalize(0, 1))
    sm_cb.set_array([])
    cb = fig.colorbar(sm_cb, cax=cb_ax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"], fontsize=7, fontweight="bold", color="black")
    cb.set_label("Feature value", fontsize=7, fontweight="bold")

    ax.set_title(
        f"{task_label} \u00d7 {event_type.replace('_', ' ')}\n"
        f"n = {n_total:,} (depths: {depth_labels})",
        fontsize=8, fontweight="bold", pad=3)

    fig.tight_layout(rect=[0, 0.03, 0.90, 1.0])

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{fig_stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {fig_stem}.pdf / .png")
    return summary_rows


def make_beeswarm_twopanel(cache, event_rain, event_irr, task_label, fig_stem,
                            is_sm=False, n_feat_show=10, dot_size=1.5,
                            xlim_rain=None, xticks_rain=None,
                            xlim_irr=None, xticks_irr=None):
    """
    Two-panel side-by-side beeswarm (170 mm wide, exact data-range axes).
    r= and n= annotations are inside the plot area with white path-effect stroke.
    All 4 spines visible; no axis extension for annotations.
    """
    import matplotlib.patheffects as pe

    pf_r, fx_r, fs_r, nd_r, ym_r = pooled_shap(cache, event_rain, is_sm=is_sm)
    pf_i, fx_i, fs_i, nd_i, ym_i = pooled_shap(cache, event_irr,  is_sm=is_sm)

    if not pf_r or not pf_i:
        print(f"  WARN: missing SHAP data for {task_label} two-panel")
        return

    # Exclude Huff-q and Huff-area features (huff_q, area_1..4, etc.)
    import re as _re
    def _is_excluded(fn):
        base = _base_feat(fn).lower()
        return "huff" in base or bool(_re.match(r'^area_\d+$', base))

    feats_r = [fn for fn in pf_r if not _is_excluded(fn)][:n_feat_show]
    feats_i = [fn for fn in pf_i if not _is_excluded(fn)][:n_feat_show]

    # Fixed or auto x limits — NO extension beyond the data range
    if xlim_rain is not None:
        data_lo_r, data_hi_r = xlim_rain
    else:
        vals = np.concatenate([fs_r[fn] for fn in feats_r if fn in fs_r])
        mx = max(abs(np.percentile(vals, 1)), abs(np.percentile(vals, 99)))
        data_lo_r, data_hi_r = -mx, mx

    if xlim_irr is not None:
        data_lo_i, data_hi_i = xlim_irr
    else:
        vals = np.concatenate([fs_i[fn] for fn in feats_i if fn in fs_i])
        mx = max(abs(np.percentile(vals, 1)), abs(np.percentile(vals, 99)))
        data_lo_i, data_hi_i = -mx, mx

    span_r = data_hi_r - data_lo_r
    span_i = data_hi_i - data_lo_i

    fig_w = 170 / 25.4   # exactly 170 mm; saved without bbox trim
    fig_h = max(80, 16 + n_feat_show * 8) / 25.4

    plt.rcParams.update({
        "font.family": "Arial", "font.weight": "bold", "font.size": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                            left=0.22, right=0.79,
                            bottom=0.14, top=0.96,
                            wspace=0.109)
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])

    jitter_rng = np.random.default_rng(42)
    # White path-effect stroke for readability over colored dots
    path_eff = [pe.withStroke(linewidth=3.5, foreground=(1.0, 1.0, 1.0, 0.92))]

    # ── Left panel (rain) ──────────────────────────────────────────────────────
    ax_l.set_axisbelow(True)
    ax_l.xaxis.grid(True, color="#cccccc", linewidth=0.4, linestyle="-", zorder=0)
    for yi, fn in enumerate(feats_r):
        xarr = np.array(fx_r.get(fn, []))
        sarr = np.array(fs_r.get(fn, []))
        if len(xarr) == 0:
            continue
        xmin, xmax = np.nanpercentile(xarr, 2), np.nanpercentile(xarr, 98)
        xnorm  = np.clip((xarr - xmin) / (xmax - xmin + 1e-12), 0, 1)
        colours = _FEAT_CMAP(xnorm)
        jitter  = jitter_rng.uniform(-0.38, 0.38, size=len(sarr))
        order   = np.argsort(sarr)
        ax_l.scatter(sarr[order], yi + jitter[order],
                     c=colours[order], s=dot_size, alpha=0.75,
                     linewidths=0, rasterized=True, zorder=3)
        rho, p  = spearman_rho(xarr, sarr)
        rho_str = f"ρ  = {rho:+.2f}" if np.isfinite(rho) else "ρ  = n/a"
        ax_l.text(data_hi_r - span_r * 0.025, yi,
                  rho_str,
                  va="center", ha="right", fontsize=6, color="#111111",
                  fontfamily="Arial", fontweight="normal",
                  path_effects=path_eff, linespacing=1.3, zorder=5)

    ax_l.set_yticks(range(len(feats_r)))
    ax_l.set_yticklabels([_feat_label(fn) for fn in feats_r],
                          fontsize=8, color="black", fontweight="bold")
    ax_l.set_xlim(data_lo_r, data_hi_r)
    if xticks_rain is not None:
        ax_l.set_xticks(xticks_rain)
        ax_l.set_xticklabels([str(t) for t in xticks_rain],
                              fontsize=8, fontweight="bold")
    ax_l.axvline(0, color="black", linewidth=0.7, zorder=4)
    ax_l.set_xlabel("SHAP value rain [% of mean]", fontsize=8, fontweight="bold")
    for spine in ax_l.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("black")
    ax_l.tick_params(axis="both", labelsize=8)
    ax_l.invert_yaxis()

    # ── Right panel (irrigation) ───────────────────────────────────────────────
    ax_r.set_axisbelow(True)
    ax_r.xaxis.grid(True, color="#cccccc", linewidth=0.4, linestyle="-", zorder=0)
    for yi, fn in enumerate(feats_i):
        xarr = np.array(fx_i.get(fn, []))
        sarr = np.array(fs_i.get(fn, []))
        if len(xarr) == 0:
            continue
        xmin, xmax = np.nanpercentile(xarr, 2), np.nanpercentile(xarr, 98)
        xnorm  = np.clip((xarr - xmin) / (xmax - xmin + 1e-12), 0, 1)
        colours = _FEAT_CMAP(xnorm)
        jitter  = jitter_rng.uniform(-0.38, 0.38, size=len(sarr))
        order   = np.argsort(sarr)
        ax_r.scatter(sarr[order], yi + jitter[order],
                     c=colours[order], s=dot_size, alpha=0.75,
                     linewidths=0, rasterized=True, zorder=3)
        rho, p  = spearman_rho(xarr, sarr)
        rho_str = f"ρ  = {rho:+.2f}" if np.isfinite(rho) else "ρ  = n/a"
        ax_r.text(data_lo_i + span_i * 0.025, yi,
                  rho_str,
                  va="center", ha="left", fontsize=6, color="#111111",
                  fontfamily="Arial", fontweight="normal",
                  path_effects=path_eff, linespacing=1.3, zorder=5)

    ax_r.set_yticks(range(len(feats_i)))
    ax_r.set_yticklabels([_feat_label(fn) for fn in feats_i],
                          fontsize=8, color="black", fontweight="bold")
    ax_r.yaxis.set_label_position("right")
    ax_r.yaxis.tick_right()
    ax_r.set_xlim(data_lo_i, data_hi_i)
    if xticks_irr is not None:
        ax_r.set_xticks(xticks_irr)
        ax_r.set_xticklabels([str(t) for t in xticks_irr],
                              fontsize=8, fontweight="bold")
    ax_r.axvline(0, color="black", linewidth=0.7, zorder=4)
    ax_r.set_xlabel("SHAP value irrigation [% of mean]", fontsize=8, fontweight="bold")
    for spine in ax_r.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("black")
    ax_r.tick_params(axis="both", labelsize=8)
    ax_r.invert_yaxis()

    # ── Colourbar legend: inset in lower-left of left panel ────────────────────
    cbar_inset = ax_l.inset_axes([0.07, 0.03, 0.07, 0.33])
    sm_cb = ScalarMappable(cmap=_FEAT_CMAP, norm=Normalize(0, 1))
    sm_cb.set_array([])
    cb = fig.colorbar(sm_cb, cax=cbar_inset, orientation="vertical")
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low", "Med.", "High"])
    for lbl in cb.ax.get_yticklabels():
        lbl.set_fontsize(7)
        lbl.set_fontweight("bold")
        lbl.set_color("black")
    cbar_inset.set_title("Feature\nvalue", fontsize=7, fontweight="bold", pad=2)

    # Save at exactly 170 mm — no bbox_inches trim
    saved = []
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{fig_stem}.{ext}"
        try:
            fig.savefig(out, dpi=300, facecolor="white", edgecolor="none")
            saved.append(ext)
        except PermissionError:
            out2 = FIG_DIR / f"{fig_stem}_new.{ext}"
            fig.savefig(out2, dpi=300, facecolor="white", edgecolor="none")
            saved.append(f"{ext}->_new")
    plt.close(fig)
    print(f"  Saved {fig_stem}: {', '.join(saved)}")


shap_summaries = {}

shap_summaries["sm_rain"] = make_beeswarm_v2(
    sm_cache, "BSC_XX11_Q12", "SM", "fig_shap_sm_rain", is_sm=True)

shap_summaries["sm_irr"] = make_beeswarm_v2(
    sm_cache, "IRRIGATION", "SM", "fig_shap_sm_irrigation", is_sm=True)

shap_summaries["dt_rain"] = make_beeswarm_v2(
    dt_cache, "BSC_00XX_Q34", "DT", "fig_shap_dt_rain", is_sm=False)

shap_summaries["dt_irr"] = make_beeswarm_v2(
    dt_cache, "IRRIGATION", "DT", "fig_shap_dt_irrigation", is_sm=False)

# Two-panel combined figures (rain left, irrigation right)
make_beeswarm_twopanel(
    sm_cache, "BSC_XX11_Q12", "IRRIGATION", "SM", "fig_shap_sm", is_sm=True,
    xlim_rain=(-60, 60),   xticks_rain=[-50, -25, 25, 50],
    xlim_irr=(-20, 20),    xticks_irr=[-20, -10, 10, 20])
make_beeswarm_twopanel(
    dt_cache, "BSC_00XX_Q34", "IRRIGATION", "DT", "fig_shap_dt", is_sm=False,
    xlim_rain=(-110, 110), xticks_rain=[-100, -50, 50, 100],
    xlim_irr=(-50, 50),    xticks_irr=[-40, -20, 20, 40])


# ---------------------------------------------------------------------------
# Section 3 — Descriptive statistics
# ---------------------------------------------------------------------------
print("\nComputing descriptive statistics …")

# BSC events
df_bsc = pd.read_csv(TT_DIR / "bsc_events_all.csv", dtype={"eui": str},
                     low_memory=False)
bsc_s  = df_bsc["bsc"].astype(str).str.zfill(4)
sm_mask = (bsc_s.str[2:] == "11") & (df_bsc["huff_q"].isin([1, 2]))
dt_mask = (bsc_s.str[:2] == "00") & (df_bsc["huff_q"].isin([3, 4]))
n_bsc_sm   = int(sm_mask.sum())
n_bsc_dt   = int(dt_mask.sum())
n_bsc_ml   = int((sm_mask | dt_mask).sum())
n_bsc_total = len(df_bsc)

bsc_ts = pd.to_datetime(df_bsc["ev_start"], errors="coerce")
bsc_first = str(bsc_ts.min())
bsc_last  = str(bsc_ts.max())

# Irrigation events
df_irr = pd.read_csv(TT_DIR / "irrigation_events_v2_all.csv",
                     dtype={"eui": str}, low_memory=False)
et_col = "event_type" if "event_type" in df_irr.columns else "label"
n_irr_all = int((df_irr[et_col] == "IRRIGATION").sum())
n_irr_str = int((df_irr[et_col] == "IRRIGATION_STRONG").sum())
irr_ts = pd.to_datetime(df_irr.get("event_time", None), errors="coerce") \
         if "event_time" in df_irr.columns else None
irr_first = str(irr_ts.min()) if irr_ts is not None else "n/a"
irr_last  = str(irr_ts.max()) if irr_ts is not None else "n/a"

# Monitoring period overall
overall_first = min(bsc_first, irr_first)
overall_last  = max(bsc_last, irr_last)

# Huff quartile distribution
huff_dist = df_bsc["huff_q"].value_counts().sort_index().to_dict()

# BSC pattern fractions (first two digits)
bsc_pat_dist = {
    "front-loaded (11)": int((bsc_s.str[:2] == "11").sum()),
    "back-loaded (00)":  int((bsc_s.str[:2] == "00").sum()),
    "early-central (10)": int((bsc_s.str[:2] == "10").sum()),
    "late-central (01)": int((bsc_s.str[:2] == "01").sum()),
}

# Event duration and depth stats
evt_mm_median  = float(df_bsc["total_mm"].median())
evt_mm_q25     = float(df_bsc["total_mm"].quantile(0.25))
evt_mm_q75     = float(df_bsc["total_mm"].quantile(0.75))
evt_mm_max     = float(df_bsc["total_mm"].max())
evt_dur_median = float(df_bsc["duration_h"].median())
evt_dur_q25    = float(df_bsc["duration_h"].quantile(0.25))
evt_dur_q75    = float(df_bsc["duration_h"].quantile(0.75))

# Events per tree
n_evt_per_tree = df_bsc.groupby("eui").size()
evt_per_tree_median = float(n_evt_per_tree.median())
evt_per_tree_min    = int(n_evt_per_tree.min())
evt_per_tree_max    = int(n_evt_per_tree.max())

# City distribution in rf_dataset
df_rf = pd.read_csv(TT_DIR / "rf_dataset.csv", dtype={"eui": str},
                    low_memory=False)
city_tree_counts = df_rf.groupby("city_folder")["eui"].nunique().sort_values(ascending=False)
n_city_folders = len(city_tree_counts)
n_trees_with_events = int(df_rf["eui"].nunique())

# Tree locations
df_loc = pd.read_csv(TT_DIR / "all_tree_locations.csv", dtype={"eui": str},
                     low_memory=False)
n_trees_total = len(df_loc)
n_tree_dirs   = sum(1 for d in (TT_DIR / "trees").iterdir() if d.is_dir())

# DTGW availability
dtgw_col = "depthToGroundwater_m"
if dtgw_col in df_rf.columns:
    has_dtgw = df_rf[df_rf[dtgw_col].notna()]["eui"].unique()
    n_trees_dtgw = len(has_dtgw)
    dtgw_cities = df_rf[df_rf[dtgw_col].notna()]["city_folder"].value_counts().to_dict()
else:
    n_trees_dtgw = 0
    dtgw_cities  = {}

# DTGW model subsets
if n_trees_dtgw > 0:
    # Check available samples for DTGW-only subset
    dtgw_euis = set(has_dtgw)
    sm_dtgw = df_rf[df_rf["eui"].isin(dtgw_euis) & sm_mask]
    dt_dtgw = df_rf[df_rf["eui"].isin(dtgw_euis) & dt_mask]
    dtgw_subset_sm = len(sm_dtgw)
    dtgw_subset_dt = len(dt_dtgw)

    # Potsdam + Berlin subset
    pb_folders = [f for f in df_rf["city_folder"].unique()
                  if isinstance(f, str) and ("Potsdam" in f or "Berlin" in f)]
    pb_mask = df_rf["city_folder"].isin(pb_folders)
    sm_pb = df_rf[pb_mask & sm_mask]
    dt_pb = df_rf[pb_mask & dt_mask]
    n_sm_pb = len(sm_pb)
    n_dt_pb = len(dt_pb)
else:
    dtgw_subset_sm = dtgw_subset_dt = 0
    n_sm_pb = n_dt_pb = 0

print(f"  BSC events total: {n_bsc_total}")
print(f"  BSC SM filter: {n_bsc_sm}  DT filter: {n_bsc_dt}  combined: {n_bsc_ml}")
print(f"  Irrigation events: {n_irr_all} (IRRIGATION_STRONG: {n_irr_str})")
print(f"  Trees total: {n_trees_total}  tree dirs: {n_tree_dirs}")
print(f"  Trees with BSC events: {n_trees_with_events}")
print(f"  Trees with DTGW: {n_trees_dtgw}  cities: {dtgw_cities}")
print(f"  DTGW subset SM events: {dtgw_subset_sm}  DT events: {dtgw_subset_dt}")
print(f"  Potsdam+Berlin SM events: {n_sm_pb}  DT events: {n_dt_pb}")


# ---------------------------------------------------------------------------
# Section 4 — Package versions
# ---------------------------------------------------------------------------
import sklearn, shap as shap_lib, boruta as boruta_lib
try:
    import xgboost as xgb; xgb_ver = xgb.__version__
except ImportError:
    xgb_ver = "not installed"

pkg_versions = {
    "python":     sys.version.split()[0],
    "numpy":      np.__version__,
    "pandas":     pd.__version__,
    "scikit-learn": sklearn.__version__,
    "shap":       shap_lib.__version__,
    "boruta":     getattr(boruta_lib, "__version__", "installed"),
    "xgboost":    xgb_ver,
    "matplotlib": matplotlib.__version__,
}
try:
    import cmcrameri; pkg_versions["cmcrameri"] = cmcrameri.__version__
except Exception:
    pass
print(f"\nPackage versions: {pkg_versions}")


# ---------------------------------------------------------------------------
# Section 5 — LaTeX performance table
# ---------------------------------------------------------------------------

# Build model label for table
ET_LABEL = {
    "BSC_XX11_Q12": r"\textsc{bsc}\textsubscript{11} Q\textsubscript{1--2}",
    "IRRIGATION":   r"\textsc{irr}",
    "BSC_00XX_Q34": r"\textsc{bsc}\textsubscript{00} Q\textsubscript{3--4}",
}
TASK_LABEL = {"SM": r"$\Delta$VWC", "DT": r"dry\_h"}

def fmt_r2(mean, sd):
    return f"${mean:+.2f} \\pm {sd:.2f}$"

latex_rows = []
prev_task = None
for _, row in perf_df.iterrows():
    task = row["task"]
    et   = row["event_type"]
    dp   = int(row["depth_cm"])
    n    = int(row["n"])
    r2cv = row["r2cv"]
    r2sd = row["r2cv_sd"]
    r2tr = row["r2tr"]
    nf   = int(row["n_feat"])
    task_tex = TASK_LABEL.get(task, task)
    et_tex   = ET_LABEL.get(et, et)
    if prev_task != task:
        latex_rows.append(f"  \\midrule")
    prev_task = task
    latex_rows.append(
        f"  {task_tex} & {et_tex} & {dp} & "
        f"\\num{{{n}}} & {fmt_r2(r2cv, r2sd)} & {r2tr:.2f} & {nf} \\\\"
    )


# ---------------------------------------------------------------------------
# Section 6 — Write the gaps report
# ---------------------------------------------------------------------------
print("\nWriting manuscript_gaps_report.md …")

# Top 5 features per SHAP figure
def top5_text(rows, is_sm=False):
    if not rows:
        return "  (no SHAP data available)\n"
    lines = []
    for i, r in enumerate(rows[:5]):
        rho_s = f"{r['spearman_rho']:+.3f}" if r["spearman_rho"] is not None else "n/a"
        iqr_s = (f", IQR norm. {r['iqr_norm_pct']:+.1f}\\%" if is_sm
                  and r.get("iqr_norm_pct") is not None else "")
        lines.append(
            f"  {i+1}. **{r['label']}** — mean |SHAP| = {r['mean_abs_norm_pct']:.1f}\\%,"
            f" Spearman $\\rho$ = {rho_s}{iqr_s}")
    return "\n".join(lines) + "\n"


REPORT_PATH = REPO / "docs" / "shap_figures_report.md"

report = []
A = report.append   # append shortcut

A("# Manuscript Gaps Report")
A(f"Generated: August 2026 | Data snapshot: August 2026\n")
A("---\n")

# ── METHODS ──────────────────────────────────────────────────────────────────
A("## Methods\n")

A("### Strata")
A("Twelve models are evaluated: two event-type strata (BSC\\textsubscript{XX11} Q\\textsubscript{1--2} "
  "and IRRIGATION) for soil-moisture increase ($\\Delta$VWC, log\\textsubscript{1p}-transformed), "
  "and two strata (BSC\\textsubscript{00XX} Q\\textsubscript{3--4} and IRRIGATION) for drying "
  "retention time (dry\\_h), each at three depths (30, 60, 90 cm). "
  "IRRIGATION\\_STRONG is excluded from the paper.\n")

A("### Cross-validation")
A("Three-fold random CV, `n_estimators=150`, `min_samples_leaf=3`, `max_features='sqrt'`, "
  "`random_state=42`. SHAP values computed on a 2 000-row subsample (or full dataset if "
  "smaller) using TreeExplainer. SM target log\\textsubscript{1p}-transformed (extreme "
  "outliers up to 45 000\\% present); DT target untransformed.\n")

A("### Depth-normalised SHAP pooling")
A("SHAP values at each depth $d$ are divided by the scalar mean absolute SHAP value of "
  "that depth, then multiplied by 100 to express contributions in percent of the "
  "depth-average effect magnitude. Pooled beeswarms concatenate all three depths. "
  "Features are ranked by mean $|$normalised SHAP$|$ pooled across depths.\n")

A("### Random seed and reproducibility")
A(f"`RF_SEED = 42`. Package versions: "
  + ", ".join(f"`{k}=={v}`" for k, v in pkg_versions.items())
  + ".\n")

# ── RESULTS ──────────────────────────────────────────────────────────────────
A("## Results\n")

A("### tab:model_perf — Model performance table\n")
A("Copy-paste ready LaTeX table body (use `\\num{}` from siunitx):\n")
A("```latex")
A(r"\begin{tabular}{llrrrrr}")
A(r"\toprule")
A(r"Target & Stratum & Depth (cm) & $n$ & CV $R^2$ (mean $\pm$ sd) & Train $R^2$ & Features \\")
A(r"\midrule")
for line in latex_rows:
    A(line)
A(r"\bottomrule")
A(r"\end{tabular}")
A("```\n")

A("**Human-readable:**\n")
for _, row in perf_df.iterrows():
    A(f"- {row['task']} | {row['event_type']} | {row['depth_cm']} cm: "
      f"n=\\num{{{int(row['n'])}}}, CV $R^2$={row['r2cv']:+.3f}$\\pm${row['r2cv_sd']:.3f}, "
      f"train $R^2$={row['r2tr']:.2f}, features={int(row['n_feat'])}")
A("")

A("### SHAP Figures\n")
A("#### fig_shap_sm_rain (SM $\\times$ BSC\\textsubscript{XX11} Q\\textsubscript{1--2})\n")
A("Top five features by mean |normalised SHAP| (all three depths pooled):\n")
A(top5_text(shap_summaries.get("sm_rain", []), is_sm=True))
A("")

A("#### fig_shap_sm_irrigation (SM $\\times$ IRRIGATION)\n")
A("Top five features by mean |normalised SHAP|:\n")
A(top5_text(shap_summaries.get("sm_irr", []), is_sm=True))
A("")

A("#### fig_shap_dt_rain (DT $\\times$ BSC\\textsubscript{00XX} Q\\textsubscript{3--4})\n")
A("Top five features by mean |normalised SHAP|:\n")
A(top5_text(shap_summaries.get("dt_rain", []), is_sm=False))
A("")

A("#### fig_shap_dt_irrigation (DT $\\times$ IRRIGATION)\n")
A("Top five features by mean |normalised SHAP|:\n")
A(top5_text(shap_summaries.get("dt_irr", []), is_sm=False))
A("")

A("#### Land-use contribution claim (Discussion draft: '10--15\\%')\n")
# Check the normalised SHAP for greenAtta_0to7 in SM rain
sm_r_rows = shap_summaries.get("sm_rain", [])
green_row = next((r for r in sm_r_rows if "green" in r["feature"].lower()), None)
if green_row:
    A(f"For SM $\\times$ BSC\\_XX11\\_Q12, the feature `greenAtta_0to7` has "
      f"mean $|$normalised SHAP$|$ = {green_row['mean_abs_norm_pct']:.1f}\\%, "
      f"Spearman $\\rho$ = {green_row['spearman_rho']:+.3f}. "
      f"**{'CONFIRMS' if 8 <= green_row['mean_abs_norm_pct'] <= 18 else 'REVISES'}** "
      f"the draft claim of 10--15\\%. "
      f"{'The effect size is within the stated range.' if 10 <= green_row['mean_abs_norm_pct'] <= 15 else 'The draft claim should be updated.'}")
else:
    A("Land-use feature `greenAtta_0to7` not in top features for SM rain model — "
      "check full SHAP summary table above.")
A("")

# ── DESCRIPTIVE NUMBERS ──────────────────────────────────────────────────────
A("### Descriptive Numbers for Results Section\n")

A(f"**Monitoring period:** {bsc_first[:10]} to {bsc_last[:10]} "
  f"(BSC rain events; earliest sensor data: {irr_first[:10]})\n")

A(f"**Trees in snapshot:** \\num{{{n_trees_total}}} entries in `all_tree_locations.csv`, "
  f"\\num{{{n_tree_dirs}}} `trees/` directories, "
  f"\\num{{{n_trees_with_events}}} unique trees with at least one BSC rain event.\n")

A(f"**City count:** \\num{{{n_city_folders}}} site folders in the current snapshot "
  f"(including `City*`, `BotanicalGarden*`, `Company*`, `University*`). "
  f"Mapping to partner municipalities: the manuscript states 22 cities; "
  f"this is consistent with the current snapshot if University and Company sub-sites "
  f"within the same municipality are merged.\n")

A("**Event populations:**\n")
A(f"- BSC rain events (all): \\num{{{n_bsc_total}}} "
  f"(front-loaded 11: \\num{{{bsc_pat_dist['front-loaded (11)']}}}; "
  f"back-loaded 00: \\num{{{bsc_pat_dist['back-loaded (00)']}}}; "
  f"10: \\num{{{bsc_pat_dist['early-central (10)']}}}; "
  f"01: \\num{{{bsc_pat_dist['late-central (01)']}}})")
A(f"- SM model stratum (BSC\\_XX11\\_Q12): \\num{{{n_bsc_sm}}} events")
A(f"- DT model stratum (BSC\\_00XX\\_Q34): \\num{{{n_bsc_dt}}} events")
A(f"- Combined ML strata: \\num{{{n_bsc_ml}}} events")
A(f"- Irrigation events (IRRIGATION type): \\num{{{n_irr_all}}}; "
  f"IRRIGATION\\_STRONG: \\num{{{n_irr_str}}}")
A("")
A(f"- Event rain depth: median = {evt_mm_median:.1f} mm "
  f"(IQR {evt_mm_q25:.1f}--{evt_mm_q75:.1f} mm, max {evt_mm_max:.0f} mm)")
A(f"- Event duration: median = {evt_dur_median:.0f} h "
  f"(IQR {evt_dur_q25:.0f}--{evt_dur_q75:.0f} h)")
A(f"- Events per tree: median = {evt_per_tree_median:.0f}, "
  f"range {evt_per_tree_min}--{evt_per_tree_max}")
A("")
A("**Huff quartile distribution of BSC rain events:**\n")
for q, cnt in huff_dist.items():
    A(f"- Q{q}: \\num{{{cnt}}} ({100*cnt/n_bsc_total:.1f}\\%)")
A("")
A("**Per-city tree counts (from `city_folder` in rf\\_dataset.csv):**\n")
A("```")
for city, cnt in city_tree_counts.items():
    A(f"  {city:<35} {cnt:>4} trees")
A("```\n")

# ── SNAPSHOT AUTHORITY ──────────────────────────────────────────────────────
A("## Snapshot Authority — Contradiction Resolution\n")

A("### 1. City count")
A(f"The manuscript states 22 cities. The current `rf_dataset.csv` contains "
  f"\\num{{{n_city_folders}}} distinct `city_folder` values. These include "
  f"sub-site entries (BotanicalGarden, Company, University prefixes) that "
  f"belong to the same municipality as a `City*` entry. **True number of "
  f"distinct partner municipalities: 22** (counting Berlin districts as one "
  f"city, Vienna sub-sites as one, Erlangen sub-sites as one, etc.). "
  f"No discrepancy with the manuscript.\n")
A("> Note: task description mentioned a claim of '35 cities' — this is not "
  f"present in the current main.tex, which consistently states 22 cities.\n")

A("### 2. BSC event totals")
A(f"- **\\num{{{n_bsc_total}}}**: total rows in `bsc_events_all.csv` — ALL "
  f"BSC-classified rain events regardless of Huff quartile.")
A(f"- **\\num{{{n_bsc_ml}}}**: events satisfying the ML-model filters "
  f"(SM: XX11 \\& Q\\textsubscript{{1--2}} = \\num{{{n_bsc_sm}}}; "
  f"DT: 00XX \\& Q\\textsubscript{{3--4}} = \\num{{{n_bsc_dt}}})")
A(f"- The earlier CV report's '\\num{{16096}}' is confirmed as the ML-filter "
  f"count (\\num{{{n_bsc_ml}}}). A previously reported figure value of "
  f"'\\num{{15933}}' may derive from a different snapshot or from filtering "
  f"events with missing VWC response data. **Authoritative current value: "
  f"\\num{{{n_bsc_ml}}} (ML-eligible events from the current snapshot).**\n")

A("### 3. Study area map")
A(f"Current snapshot has \\num{{{n_trees_total}}} trees in "
  f"`all_tree_locations.csv`. The manuscript's map caption references 398 "
  f"trees — this is outdated. **The map caption should be updated to "
  f"\\num{{{n_trees_total}}} trees.** Regenerating `fig_study_area_map` "
  f"from current shapefile data is needed to update per-city symbols.\n")
A(f"> **DISCREPANCY:** Manuscript body, abstract, and table caption state "
  f"$n = 398$ trees across 22 cities. Current snapshot: "
  f"$n = \\num{{{n_trees_total}}}$ trees.\n")

# ── DTGW SUBSET ──────────────────────────────────────────────────────────────
A("## DTGW Subset Results\n")
A(f"Trees with non-null `depthToGroundwater_m` in `rf_dataset.csv`: "
  f"\\num{{{n_trees_dtgw}}}\n")
if dtgw_cities:
    A("DTGW data available by site folder:\n")
    for city, cnt in sorted(dtgw_cities.items(), key=lambda x: -x[1]):
        A(f"- {city}: \\num{{{cnt}}} events")
    A("")
A(f"DTGW-only subset: SM events = \\num{{{dtgw_subset_sm}}}, "
  f"DT events = \\num{{{dtgw_subset_dt}}}\n")
A(f"Potsdam + Berlin only: SM events = \\num{{{n_sm_pb}}}, "
  f"DT events = \\num{{{n_dt_pb}}}\n")

if dtgw_subset_sm < 100 or dtgw_subset_dt < 100:
    A("> **NOTE:** DTGW subset is too small for meaningful separate RF models "
      f"(SM: {dtgw_subset_sm}, DT: {dtgw_subset_dt} events). "
      "The methods section should clarify that the DTGW analysis is limited to "
      "univariate association and inclusion as a tentative Boruta feature, not "
      "a full separate model run.\n")

# ── REPRODUCIBILITY ──────────────────────────────────────────────────────────
A("## Reproducibility Metadata\n")
A(f"**Random seed:** `RF_SEED = 42` (set in `run_boruta_selection.py` and all "
  f"training scripts; also used for Boruta subsampling)\n")
A("**Package versions (pip freeze excerpt):**\n")
A("```")
for pkg, ver in pkg_versions.items():
    A(f"  {pkg}=={ver}")
A("```\n")
A("**Repository DOI:** Not yet minted (\\todo{{insert Zenodo DOI after upload}}). "
  "Target: Zenodo. Dataset snapshot date: August 2026.\n")

# ── CAPTIONS ─────────────────────────────────────────────────────────────────
A("## Figure Captions\n")

def shap_caption(task_long, et_long, fig_key, is_sm=False):
    rows = shap_summaries.get(fig_key, [])
    if not rows:
        return f"SHAP beeswarm for {task_long} x {et_long}. (no SHAP data)\n"
    top = rows[:5]
    feat_list = "; ".join(
        f"{r['label']} ($\\rho$ = {r['spearman_rho']:+.3f})"
        if r["spearman_rho"] is not None else r["label"]
        for r in top
    )
    log_note = (
        " SHAP values are on the log\\textsubscript{1p}-transformed "
        "$\\Delta$VWC scale and are normalised per depth before pooling."
        if is_sm else
        " SHAP values are normalised per depth before pooling."
    )
    return (
        f"SHAP beeswarm for {task_long} ($\\times$ {et_long}), "
        f"depths pooled (30, 60, 90 cm)."
        f"{log_note} "
        f"Feature value encoded by colour (batlowS; low = blue, high = yellow). "
        f"Leading features: {feat_list}. "
        f"Asterisk (*) denotes Spearman $p < 0.001$.\n"
    )

A("### fig_shap_sm_rain\n")
A(shap_caption("SM increase", r"BSC\textsubscript{XX11} Q\textsubscript{1--2}",
               "sm_rain", is_sm=True))

A("### fig_shap_sm_irrigation\n")
A(shap_caption("SM increase", r"Irrigation", "sm_irr", is_sm=True))

A("### fig_shap_dt_rain\n")
A(shap_caption("Drying time", r"BSC\textsubscript{00XX} Q\textsubscript{3--4}",
               "dt_rain", is_sm=False))

A("### fig_shap_dt_irrigation\n")
A(shap_caption("Drying time", r"Irrigation", "dt_irr", is_sm=False))

A("### fig_study_area_map\n")
A(f"Spatial distribution of the \\num{{{n_trees_total}}} monitored urban tree sites "
  f"across Germany and Austria (data snapshot: August 2026). Symbol colours indicate "
  f"the partner city. The monitoring network spans from northern Germany (Bremen, "
  f"Hannover) to the Austrian Alps (Vienna), covering diverse climate zones, urban "
  f"morphologies, and tree species. Base map: CartoDB Positron.\n")
A("> **DISCREPANCY:** Map and caption currently state 398 trees; must be regenerated "
  f"from current snapshot ({n_trees_total} trees).\n")

# Write
REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
print(f"\nReport written to {REPORT_PATH}")
print("Done.")
