#!/usr/bin/env python3
"""
IQR Feature Effect Analysis — Planting Pit & Land Use Recommendations.

Uses SHAP values from the RF training caches (SM v4, DT v3) to compute:
  "Moving from Q25 to Q75 of feature X changes:
     soil moisture increase by  ± Y % VWC dynamic range
     soil drying time by        ± Z hours"

SHAP isolates each feature's effect independently of correlated features.
For SM the target is log1p(ΔVWC % dyn range) → results converted back via expm1.
For DT the target is raw dry_h (hours) → results directly interpretable.

Output:  rf_planting_recommendations.pdf
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PDF    = SCRIPT_DIR / "rf_planting_recommendations.pdf"

SM_CACHE = SCRIPT_DIR / "rf_soil_moisture_v4_training_cache.pkl"
DT_CACHE = SCRIPT_DIR / "rf_drying_time_v3_training_cache.pkl"

# ── Feature registry ────────────────────────────────────────────────────────────
# (category, unit_label, long_description, design_relevant)
FEAT_META = {
    "greenAtta_0to7":    ("Green Coverage",   "% area",   "Attached green within 7.5 m radius (aggregate)",        True),
    "greenAtta_2m5":     ("Green Coverage",   "% area",   "Attached green 0–2.5 m ring",                           True),
    "greenAtta_5m":      ("Green Coverage",   "% area",   "Attached green 2.5–5 m ring",                           True),
    "greenAtta_7m5":     ("Green Coverage",   "% area",   "Attached green 5–7.5 m ring",                           True),
    "greenDeta_0to7":    ("Green Coverage",   "% area",   "Detached green within 7.5 m radius (aggregate)",        True),
    "greenDeta_2m5":     ("Green Coverage",   "% area",   "Detached green 0–2.5 m ring",                           True),
    "greenDeta_5m":      ("Green Coverage",   "% area",   "Detached green 2.5–5 m ring",                           True),
    "greenDeta_7m5":     ("Green Coverage",   "% area",   "Detached green 5–7.5 m ring",                           True),
    "area_1":            ("Sealed Surface",   "% area",   "Sealed surface zone 1 (0–2 m, planting pit)",           False),
    "area_2":            ("Sealed Surface",   "% area",   "Sealed surface zone 2 (2–5 m)",                         False),
    "area_3":            ("Sealed Surface",   "% area",   "Sealed surface zone 3 (5–10 m)",                        False),
    "area_4":            ("Sealed Surface",   "% area",   "Sealed surface zone 4 (>10 m, surroundings)",           False),
    "stemDiam":          ("Tree Morphology",  "cm",       "Stem diameter at breast height (DBH)",                  True),
    "crownDiam":         ("Tree Morphology",  "m",        "Crown diameter",                                         True),
    "height":            ("Tree Morphology",  "m",        "Tree height",                                            True),
    "SVF":               ("Sky View Factor",  "0–1",      "Sky view factor (0 = closed canopy, 1 = open sky)",     True),
    "slope":             ("Topography",       "°",        "Terrain slope angle",                                    True),
    "twi":               ("Topography",       "–",        "Topographic Wetness Index (higher = more water acc.)",  True),
    "tpi_7m5":           ("Topography",       "m",        "Topographic Position Index (7.5 m radius)",             True),
    "DTGW":              ("Groundwater",      "m",        "Depth to groundwater table",                             True),
    "pre_vwc-10":        ("Soil State",       "% VWC",    "Pre-event soil moisture at −10 cm",                     False),
    "pre_vwc-30":        ("Soil State",       "% VWC",    "Pre-event soil moisture at −30 cm",                     False),
    "pre_vwc-45":        ("Soil State",       "% VWC",    "Pre-event soil moisture at −45 cm",                     False),
    "dyn_range-10":      ("Soil State",       "% VWC",    "Dynamic VWC range at −10 cm",                           False),
    "dyn_range-30":      ("Soil State",       "% VWC",    "Dynamic VWC range at −30 cm",                           False),
    "dyn_range-45":      ("Soil State",       "% VWC",    "Dynamic VWC range at −45 cm",                           False),
    "avg_temp-10":       ("Temperature",      "°C",       "Mean air temperature during event",                     False),
    "avg_temp-30":       ("Temperature",      "°C",       "Mean air temperature during event",                     False),
    "avg_temp-45":       ("Temperature",      "°C",       "Mean air temperature during event",                     False),
    "total_mm":          ("Precipitation",    "mm",       "Total event precipitation",                              False),
    "rain_pre24h_mm":    ("Precipitation",    "mm",       "Rainfall 24 h before event",                            False),
    "rain_pre72h_mm":    ("Precipitation",    "mm",       "Rainfall 72 h before event",                            False),
    "rain_post24h_mm":   ("Precipitation",    "mm",       "Rainfall 24 h after event",                             False),
    "rain_post72h_mm":   ("Precipitation",    "mm",       "Rainfall 72 h after event",                             False),
    "max_intensity_mmh": ("Precipitation",    "mm/h",     "Peak rainfall intensity",                               False),
    "duration_h":        ("Precipitation",    "h",        "Rain event duration",                                    False),
    "day_cos":           ("Seasonality",      "–",        "Day of year (cosine encoding; lower = summer)",         False),
}

# Features excluded from all PDF output (not practically influenceable)
EXCL_FEATS = {"area_1", "area_2", "area_3", "area_4"}

CAT_ORDER = ["Green Coverage","Sealed Surface","Tree Morphology","Sky View Factor",
             "Topography","Groundwater","Soil State","Temperature","Precipitation","Seasonality"]

CAT_COLORS = {
    "Green Coverage":  "#c8e6c9",
    "Sealed Surface":  "#fce4ec",
    "Tree Morphology": "#e3f2fd",
    "Sky View Factor": "#f3e5f5",
    "Topography":      "#fff3e0",
    "Groundwater":     "#e8eaf6",
    "Soil State":      "#f5f5f5",
    "Temperature":     "#f5f5f5",
    "Precipitation":   "#f5f5f5",
    "Seasonality":     "#f5f5f5",
}

SHORT = {
    "greenAtta_0to7": "gAtta 0–7m", "greenAtta_2m5": "gAtta 2.5m",
    "greenAtta_5m": "gAtta 5.0m",   "greenAtta_7m5": "gAtta 7.5m",
    "greenDeta_0to7": "gDeta 0–7m",
    "rain_pre24h_mm": "rain_pre24h", "rain_pre72h_mm": "rain_pre72h",
    "rain_post24h_mm": "rain_post24h", "rain_post72h_mm": "rain_post72h",
    "max_intensity_mmh": "max_intens", "tpi_7m5": "tpi_7.5m",
}
def sh(f): return SHORT.get(f, f)

C_H   = "#1a252f"; C_HT = "white"
C_SH  = "#2c3e50"
C_POS = "#1e8449";  C_NEG = "#c0392b"   # text colours for + / - effects

ET_SHORT = {
    "BSC_XX11_Q12": "BSC (rain)", "BSC_00XX_Q34": "BSC (dry)",
    "IRRIGATION": "Irrigation",   "IRRIGATION_STRONG": "Irr. Strong",
}
DEPTHS = ["−10 cm", "−30 cm", "−45 cm"]


# ── load caches ────────────────────────────────────────────────────────────────
def load_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── core IQR effect calculator ─────────────────────────────────────────────────
def iqr_effect_sm(dr, f_idx):
    """
    SM effect: SHAP delta Q75–Q25 in log1p space → converted to ΔVWC % dyn range.
    Returns (effect_pct, q25_feat, q75_feat, n_low, n_high, mean_sm_pct)
    mean_sm_pct = arithmetic mean of the observed ΔVWC % (full training set)
    """
    X      = dr["X"]           # (2000, n_feats) — SHAP subsample
    y_full = dr["y"]           # (n_full,)  — log1p targets, full dataset
    y_shap = dr["y_shap"]      # (2000,)    — log1p predictions (SHAP subset)
    sv     = dr["shap_v"]      # (2000, n_feats) — SHAP values in log1p space

    # Mean ΔVWC in original % space across the full training set
    mean_sm_pct = float(np.mean(np.expm1(y_full)))

    xf    = X[:, f_idx]
    q25   = float(np.percentile(xf, 25))
    q75   = float(np.percentile(xf, 75))
    if q25 == q75:
        return 0.0, q25, q75, 0, 0, mean_sm_pct

    lo = xf <= q25;  hi = xf >= q75
    n_lo, n_hi = lo.sum(), hi.sum()
    if n_lo < 5 or n_hi < 5:
        return 0.0, q25, q75, int(n_lo), int(n_hi), mean_sm_pct

    shap_lo = float(np.mean(sv[lo, f_idx]))
    shap_hi = float(np.mean(sv[hi, f_idx]))
    d_shap  = shap_hi - shap_lo          # delta in log1p space

    # Convert: at the mean log1p prediction level, what is the delta in % VWC?
    mu  = float(np.mean(y_shap))         # mean log1p prediction
    eff = float(np.expm1(mu + d_shap) - np.expm1(mu))   # % VWC dyn range
    return eff, q25, q75, int(n_lo), int(n_hi), mean_sm_pct


def iqr_effect_dt(dr, f_idx):
    """
    DT effect: SHAP delta Q75–Q25 directly in hours.
    Returns (effect_h, q25_feat, q75_feat, n_low, n_high, mean_dt_h)
    mean_dt_h = arithmetic mean drying time across the full training set
    """
    X     = dr["X"]
    y_full= dr["y"]        # (n_full,) — raw hours, full dataset
    sv    = dr["shap_v"]   # (n, n_feats) — SHAP values in hours

    # Mean drying time across the full training set
    mean_dt_h = float(np.mean(y_full))

    xf  = X[:, f_idx]
    q25 = float(np.percentile(xf, 25))
    q75 = float(np.percentile(xf, 75))
    if q25 == q75:
        return 0.0, q25, q75, 0, 0, mean_dt_h

    lo = xf <= q25;  hi = xf >= q75
    n_lo, n_hi = lo.sum(), hi.sum()
    if n_lo < 5 or n_hi < 5:
        return 0.0, q25, q75, int(n_lo), int(n_hi), mean_dt_h

    eff = float(np.mean(sv[hi, f_idx]) - np.mean(sv[lo, f_idx]))
    return eff, q25, q75, int(n_lo), int(n_hi), mean_dt_h


# ── collect all effects ────────────────────────────────────────────────────────
def collect_effects(sm_cache, dt_cache):
    """
    Returns a list of dicts, one per (feature, event_type, depth).
    """
    records = []

    # ── SM effects ─────────────────────────────────────────────────────────────
    for et, (_, _, depth_results) in sm_cache["trained"].items():
        for di, dr in enumerate(depth_results):
            if dr["skip"]:
                continue
            depth  = DEPTHS[di]
            fnames = dr["feat_names"]
            for fi, feat in enumerate(fnames):
                eff, q25, q75, n_lo, n_hi, mean_sm = iqr_effect_sm(dr, fi)
                rel_sm = (eff / mean_sm * 100.0) if (mean_sm and not np.isnan(eff)) else np.nan
                records.append({
                    "feature": feat, "event_type": et, "depth": depth,
                    "model": "SM",
                    "effect_sm_pct":   eff,
                    "rel_effect_sm":   rel_sm,    # % of mean SM per event
                    "mean_sm_pct":     mean_sm,   # mean ΔVWC % for this stratum
                    "effect_dt_h":     np.nan,
                    "rel_effect_dt":   np.nan,
                    "mean_dt_h":       np.nan,
                    "q25": q25, "q75": q75,
                    "n_lo": n_lo, "n_hi": n_hi,
                    "n_samples": dr["n_samp"],
                    "imp": float(dr["imp_m"][fi]),
                    "r2": float(np.mean(dr["r2s"])),
                })

    # ── DT effects ─────────────────────────────────────────────────────────────
    for et, (_, _, depth_results) in dt_cache["trained"].items():
        for di, dr in enumerate(depth_results):
            if dr["skip"]:
                continue
            depth  = DEPTHS[di]
            fnames = dr["feat_names"]
            for fi, feat in enumerate(fnames):
                eff, q25, q75, n_lo, n_hi, mean_dt = iqr_effect_dt(dr, fi)
                rel_dt = (eff / mean_dt * 100.0) if (mean_dt and not np.isnan(eff)) else np.nan
                records.append({
                    "feature": feat, "event_type": et, "depth": depth,
                    "model": "DT",
                    "effect_sm_pct":   np.nan,
                    "rel_effect_sm":   np.nan,
                    "mean_sm_pct":     np.nan,
                    "effect_dt_h":     eff,
                    "rel_effect_dt":   rel_dt,    # % of mean DT for this stratum
                    "mean_dt_h":       mean_dt,   # mean dry_h hours for this stratum
                    "q25": q25, "q75": q75,
                    "n_lo": n_lo, "n_hi": n_hi,
                    "n_samples": dr["n_samp"],
                    "imp": float(dr["imp_m"][fi]),
                    "r2": float(np.mean(dr["r2s"])),
                })

    return pd.DataFrame(records)


def weighted_avg(df_sub, col, weight_col="n_samples"):
    """Weighted mean, ignoring NaN."""
    mask = df_sub[col].notna()
    if mask.sum() == 0:
        return np.nan
    sub = df_sub[mask]
    return float(np.average(sub[col], weights=sub[weight_col]))


def summarise(df):
    """
    Aggregate per feature: weighted-average effects across depths × event types.
    Returns DataFrame with one row per feature.
    """
    rows = []
    for feat, grp in df.groupby("feature"):
        sm_grp = grp[grp.model == "SM"]
        dt_grp = grp[grp.model == "DT"]

        avg_sm      = weighted_avg(sm_grp, "effect_sm_pct")
        avg_dt      = weighted_avg(dt_grp, "effect_dt_h")
        avg_rel_sm  = weighted_avg(sm_grp, "rel_effect_sm")   # % of mean SM
        avg_rel_dt  = weighted_avg(dt_grp, "rel_effect_dt")   # % of mean DT
        avg_mean_sm = weighted_avg(sm_grp, "mean_sm_pct")     # mean ΔVWC % reference
        avg_mean_dt = weighted_avg(dt_grp, "mean_dt_h")       # mean hours reference
        avg_imp_sm  = weighted_avg(sm_grp, "imp") if not sm_grp.empty else np.nan
        avg_imp_dt  = weighted_avg(dt_grp, "imp") if not dt_grp.empty else np.nan

        # Typical Q25/Q75 across appearances (median)
        q_rows = grp.drop_duplicates(["event_type", "depth"])
        q25 = float(q_rows.q25.median())
        q75 = float(q_rows.q75.median())

        cat, unit, desc, design = FEAT_META.get(feat, ("Other", "–", feat, False))

        rows.append({
            "feature": feat, "short": sh(feat),
            "category": cat, "unit": unit,
            "description": desc, "design_relevant": design,
            "avg_effect_sm_pct":  avg_sm,
            "avg_rel_effect_sm":  avg_rel_sm,
            "avg_mean_sm_pct":    avg_mean_sm,
            "avg_effect_dt_h":    avg_dt,
            "avg_rel_effect_dt":  avg_rel_dt,
            "avg_mean_dt_h":      avg_mean_dt,
            "avg_imp_sm": avg_imp_sm,
            "avg_imp_dt": avg_imp_dt,
            "q25": q25, "q75": q75,
            "n_et_sm": sm_grp.event_type.nunique(),
            "n_et_dt": dt_grp.event_type.nunique(),
        })

    return pd.DataFrame(rows).sort_values(
        ["category", "feature"], key=lambda s: s.map(
            lambda v: (CAT_ORDER.index(FEAT_META.get(v, ("Other",))[0])
                       if v in FEAT_META else 99, v)
        ) if s.name == "feature" else s
    ).reset_index(drop=True)


# ── plotting helpers ───────────────────────────────────────────────────────────
def _fmt(v, digits=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.{digits}f}" if abs(v) < 1000 else f"{v:+.0f}"

def _fmtq(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) < 1:
        return f"{v:.3f}"
    if abs(v) < 10:
        return f"{v:.2f}"
    return f"{v:.1f}"

def _fmt_sm_comb(abs_val, rel_val, ref_val):
    """Combined SM cell: '−0.866% (−1%, µ=123%)'"""
    if np.isnan(abs_val):
        return "—"
    s = f"{abs_val:+.3f}%"
    parts = []
    if not np.isnan(rel_val):
        parts.append(f"{rel_val:+.0f}%")
    if not np.isnan(ref_val):
        parts.append(f"µ={ref_val:.0f}%")
    return s + (f" ({', '.join(parts)})" if parts else "")

def _fmt_dt_comb(abs_val, rel_val, ref_val):
    """Combined DT cell: '+1.84h (+2%, µ=66h)'"""
    if np.isnan(abs_val):
        return "—"
    s = f"{abs_val:+.2f}h"
    parts = []
    if not np.isnan(rel_val):
        parts.append(f"{rel_val:+.0f}%")
    if not np.isnan(ref_val):
        parts.append(f"µ={ref_val:.0f}h")
    return s + (f" ({', '.join(parts)})" if parts else "")


# ── page 1 – title & methodology ──────────────────────────────────────────────
def page_title(pdf):
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    ax.text(0.5, 0.93,
            "Feature IQR Effects — Practical Planting Pit & Land Use Analysis",
            ha="center", va="top", fontsize=16, fontweight="bold",
            color=C_H, transform=ax.transAxes)

    ax.text(0.5, 0.87,
            "RF models: Soil Moisture v4  (Boruta + greenAtta_0to7)  ·  Drying Time v3  (same feature set)\n"
            "Effect computed via SHAP values (TreeSHAP) from the 2,000-sample explanation subset",
            ha="center", va="top", fontsize=10, color=C_SH, transform=ax.transAxes)

    body = (
        "METHOD\n"
        "SHAP (SHapley Additive exPlanations) isolates the individual contribution of each\n"
        "feature to each prediction, independent of correlations with other features.\n\n"
        "For each feature the dataset is split into the bottom 25 % (Q25) and top 25 % (Q75)\n"
        "of observed values. The SHAP contributions are averaged within each group, giving\n"
        "the expected model response at low vs high feature levels.\n\n"
        "The IQR Effect = mean SHAP contribution at Q75 − mean SHAP contribution at Q25.\n\n"
        "UNITS\n"
        "• Soil Moisture (SM): effect expressed as Δ% of dynamic VWC range\n"
        "  (the full sensor range from dry to field capacity is 100 %)\n"
        "  Example: +2.5 % means the soil stores 2.5 % more of its capacity per rain event\n"
        "  when moving from Q25 to Q75 of that feature.\n\n"
        "• Drying Time (DT): effect expressed in hours\n"
        "  Example: +8 h means soil stays moist 8 h longer per event at Q75 vs Q25.\n\n"
        "SIGN CONVENTION\n"
        "• SM effect positive  (+) → more water stored per event at higher feature level\n"
        "• SM effect negative  (−) → less water stored\n"
        "• DT effect positive  (+) → soil stays moist longer (better for tree water availability)\n"
        "• DT effect negative  (−) → soil dries faster\n\n"
        "DESIGN RELEVANCE\n"
        "Features are grouped into: Green Coverage · Sealed Surface · Tree Morphology ·\n"
        "Sky View Factor · Topography · Groundwater · Environmental context\n\n"
        "Only design-relevant features (those a planner/municipality can influence) are\n"
        "highlighted on subsequent pages. Environmental context features (precipitation,\n"
        "temperature, soil state) are included as reference."
    )
    ax.text(0.08, 0.78, body,
            ha="left", va="top", fontsize=9.5, color="#2c3e50",
            transform=ax.transAxes, family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#ecf0f1", edgecolor="#bdc3c7"))

    # Color legend boxes
    yleg = 0.05
    for i, (cat, col) in enumerate(list(CAT_COLORS.items())[:8]):
        x = 0.02 + i * 0.125
        ax.add_patch(FancyBboxPatch(
            (x, yleg), 0.11, 0.04,
            boxstyle="round,pad=0.005",
            transform=ax.transAxes,
            facecolor=col, edgecolor="#aaa", linewidth=0.8
        ))
        ax.text(x + 0.055, yleg + 0.02, cat,
                ha="center", va="center", fontsize=6.5,
                color=C_H, transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── page 2 – design feature summary table ─────────────────────────────────────
def page_summary(pdf, summary):
    design = summary[summary.design_relevant].copy()

    # Sort within each category by abs SM effect (or DT if SM missing)
    def sort_key_fn(row):
        v = abs(row.avg_effect_sm_pct) if not np.isnan(row.avg_effect_sm_pct) else \
            abs(row.avg_effect_dt_h)   if not np.isnan(row.avg_effect_dt_h)   else 0
        cat_i = CAT_ORDER.index(row.category) if row.category in CAT_ORDER else 99
        return (cat_i, -v)

    design["_sk"] = [sort_key_fn(r) for _, r in design.iterrows()]
    design = design.sort_values("_sk").drop("_sk", axis=1).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    fig.suptitle(
        "Design-Relevant Features — Weighted Average IQR Effects\n"
        "(averaged across all event types and sensor depths, weighted by n_samples)",
        fontsize=12, fontweight="bold", y=0.99, color=C_H
    )

    # 10 columns; widths sum to 1.0 — explicit to prevent overflow
    col_labels = [
        "Category", "Feature", "Unit",
        "Q25", "Q75", "IQR",
        "SM Δ%VWC\n(rel%, µ)",
        "DT Δh\n(rel%, µ)",
        "SM imp.", "DT imp.",
    ]
    col_widths = [0.09, 0.18, 0.05, 0.06, 0.06, 0.06, 0.23, 0.19, 0.04, 0.04]

    rows, row_bg = [], []
    for _, r in design.iterrows():
        iqr_span = r.q75 - r.q25
        rows.append([
            r.category[:14],
            r.description[:26],
            r.unit,
            _fmtq(r.q25),
            _fmtq(r.q75),
            _fmtq(iqr_span),
            _fmt_sm_comb(r.avg_effect_sm_pct, r.avg_rel_effect_sm, r.avg_mean_sm_pct),
            _fmt_dt_comb(r.avg_effect_dt_h,   r.avg_rel_effect_dt, r.avg_mean_dt_h),
            f"{r.avg_imp_sm:.4f}" if not np.isnan(r.avg_imp_sm) else "—",
            f"{r.avg_imp_dt:.4f}" if not np.isnan(r.avg_imp_dt) else "—",
        ])
        row_bg.append(CAT_COLORS.get(r.category, "#f5f5f5"))

    t = ax.table(cellText=rows, colLabels=col_labels,
                 colWidths=col_widths, bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)

    for j in range(len(col_labels)):
        c = t[0, j]
        c.set_facecolor(C_H); c.get_text().set_color(C_HT)
        c.get_text().set_fontweight("bold"); c.get_text().set_fontsize(7.5)

    for i, (bg, row_data) in enumerate(zip(row_bg, rows)):
        for j in range(len(col_labels)):
            t[i+1, j].set_facecolor(bg)
        for j_col in [6, 7]:
            val_str = row_data[j_col]
            cell = t[i+1, j_col]
            if val_str.startswith("+"):
                cell.get_text().set_color(C_POS)
                cell.get_text().set_fontweight("bold")
            elif val_str.startswith("-"):
                cell.get_text().set_color(C_NEG)
                cell.get_text().set_fontweight("bold")

    t[0, 1].get_text().set_ha("left")
    for i in range(len(rows)):
        t[i+1, 1].get_text().set_ha("left")
        t[i+1, 0].get_text().set_ha("left")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── pages 3-5 – per-event-type detail tables ──────────────────────────────────
def page_detail(pdf, df, model_type, sm_or_dt="SM"):
    """One page per event type with per-depth breakdown."""
    effect_col = "effect_sm_pct" if sm_or_dt == "SM" else "effect_dt_h"
    effect_unit = "% VWC dyn range" if sm_or_dt == "SM" else "hours"
    sub_df = df[df.model == sm_or_dt]
    event_types = sorted(sub_df.event_type.unique())

    for et in event_types:
        et_df  = sub_df[sub_df.event_type == et]
        depths = sorted(et_df.depth.unique(), key=lambda d: int(d.replace(" cm","").replace("−","")))

        n_depths = len(depths)
        fig = plt.figure(figsize=(14, 3.5 * n_depths + 1.5))
        fig.patch.set_facecolor("white")
        et_lbl = ET_SHORT.get(et, et)
        fig.suptitle(
            f"{sm_or_dt} Feature IQR Effects — {et_lbl}\n"
            f"({effect_unit})  ·  SHAP-based, Q25→Q75  ·  green text = beneficial  red text = detrimental",
            fontsize=11, fontweight="bold", y=0.99, color=C_H
        )

        gs = gridspec.GridSpec(n_depths, 1, figure=fig,
                               hspace=0.45, top=0.93, bottom=0.03,
                               left=0.01, right=0.99)

        for di, dep in enumerate(depths):
            ax = fig.add_subplot(gs[di])
            ax.axis("off")
            dep_df = et_df[(et_df.depth == dep) & ~et_df.feature.isin(EXCL_FEATS)].sort_values("imp", ascending=False)
            if dep_df.empty:
                ax.set_title(f"{dep} — no data", fontsize=9, color=C_SH)
                continue

            n_samp = int(dep_df.n_samples.iloc[0])
            r2     = float(dep_df.r2.iloc[0])
            ax.set_title(
                f"{dep}   n={n_samp:,}   CV R²={r2:.4f}",
                fontsize=9, color=C_SH, pad=4
            )

            rel_col  = "rel_effect_sm"  if sm_or_dt == "SM" else "rel_effect_dt"
            mean_col = "mean_sm_pct"    if sm_or_dt == "SM" else "mean_dt_h"

            # 8 columns with explicit widths — prevents text overflow
            col_labels = [
                "Category", "Feature", "Unit",
                "Q25", "Q75", "IQR",
                f"IQR Effect ({effect_unit})\n(rel%, µ val.)",
                "Feat. Imp.",
            ]
            col_widths = [0.09, 0.19, 0.05, 0.07, 0.07, 0.07, 0.37, 0.09]

            rows, row_bg = [], []
            for _, row in dep_df.iterrows():
                cat, unit, desc, _design = FEAT_META.get(
                    row.feature, ("Other", "–", row.feature, False)
                )
                iqr_span = row.q75 - row.q25
                rel_val  = row[rel_col]
                ref_val  = row[mean_col]
                if sm_or_dt == "SM":
                    comb = _fmt_sm_comb(row[effect_col], rel_val, ref_val)
                else:
                    comb = _fmt_dt_comb(row[effect_col], rel_val, ref_val)
                rows.append([
                    cat[:14], desc[:26], unit,
                    _fmtq(row.q25), _fmtq(row.q75), _fmtq(iqr_span),
                    comb,
                    f"{row.imp:.5f}",
                ])
                row_bg.append(CAT_COLORS.get(cat, "#f5f5f5"))

            n_rows = len(rows)
            fs = 7.5 if n_rows <= 14 else (6.5 if n_rows <= 20 else 5.5)

            t = ax.table(cellText=rows, colLabels=col_labels,
                         colWidths=col_widths, bbox=[0, 0, 1, 1])
            t.auto_set_font_size(False)
            t.set_fontsize(fs)

            for j in range(len(col_labels)):
                c = t[0, j]
                c.set_facecolor(C_H); c.get_text().set_color(C_HT)
                c.get_text().set_fontweight("bold")

            for i, bg in enumerate(row_bg):
                for j in range(len(col_labels)):
                    t[i+1, j].set_facecolor(bg)
                val_str = rows[i][6]
                cell    = t[i+1, 6]
                if val_str.startswith("+"):
                    cell.get_text().set_color(C_POS)
                    cell.get_text().set_fontweight("bold")
                elif val_str.startswith("-"):
                    cell.get_text().set_color(C_NEG)
                    cell.get_text().set_fontweight("bold")
                t[i+1, 1].get_text().set_ha("left")
                t[i+1, 0].get_text().set_ha("left")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ── page – recommendations ────────────────────────────────────────────────────
def page_recommendations(pdf, summary):
    """
    Auto-generated from all design-relevant features, grouped by category.
    Sealed surface (area_*) is excluded via design_relevant=False in FEAT_META.
    """
    design = summary[summary.design_relevant].copy()

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    fig.suptitle(
        "Practical Recommendations — Green Coverage, Tree & Site Features",
        fontsize=13, fontweight="bold", y=0.99, color=C_H
    )

    def sign_desc(val, rel_val, unit):
        if np.isnan(val): return "—"
        s = "+" if val > 0 else ""
        arrow = "↑ benef." if val > 0 else "↓ detrim."
        rel_s = f"({s}{rel_val:.0f}% of µ)" if not np.isnan(rel_val) else ""
        return f"{s}{val:.2f} {unit} {rel_s}  {arrow}"

    # Categories shown in display order (Sealed Surface excluded via design_relevant)
    SHOW_CATS = [
        "Green Coverage", "Tree Morphology", "Sky View Factor",
        "Topography", "Groundwater",
    ]

    lines = []
    for idx, cat in enumerate(SHOW_CATS, start=1):
        cat_rows = design[design.category == cat].copy()
        if cat_rows.empty:
            continue
        color = CAT_COLORS.get(cat, "#f5f5f5")
        lines.append(("section", f"{idx}. {cat}", color))

        # Sort by abs SM effect, DT as fallback
        cat_rows["_srt"] = (
            cat_rows.avg_effect_sm_pct.abs()
            .combine_first(cat_rows.avg_effect_dt_h.abs())
            .fillna(0)
        )
        cat_rows = cat_rows.sort_values("_srt", ascending=False)

        for _, r in cat_rows.iterrows():
            _, unit, desc, _ = FEAT_META.get(r.feature, ("Other", "–", r.feature, False))
            sm_txt = sign_desc(r.avg_effect_sm_pct, r.avg_rel_effect_sm, "% VWC") \
                     if not np.isnan(r.avg_effect_sm_pct) else "—"
            dt_txt = sign_desc(r.avg_effect_dt_h,   r.avg_rel_effect_dt, "h") \
                     if not np.isnan(r.avg_effect_dt_h)   else "—"
            q_txt = f"Q25={_fmtq(r.q25)}–{_fmtq(r.q75)} {unit}"
            lines.append(("feat", r.feature, sm_txt, dt_txt, q_txt, desc, color))

    # 5 columns with explicit widths — prevents text overflow
    col_widths_rec = [0.12, 0.16, 0.27, 0.24, 0.21]
    col_labels_hdr = [
        "Feature", "Range",
        "SM effect (% VWC, rel.%)",
        "DT effect (h, rel.%)",
        "Description",
    ]

    tbl_rows, tbl_bg = [], []
    for item in lines:
        if item[0] == "section":
            tbl_rows.append([item[1], "", "", "", ""])
            tbl_bg.append(item[2])
        else:
            _, feat, sm_txt, dt_txt, q_txt, desc, color = item
            tbl_rows.append([sh(feat), q_txt, sm_txt, dt_txt, desc[:30]])
            tbl_bg.append(color)

    t = ax.table(cellText=tbl_rows, colLabels=col_labels_hdr,
                 colWidths=col_widths_rec, bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)

    for j in range(5):
        c = t[0, j]
        c.set_facecolor(C_H); c.get_text().set_color(C_HT)
        c.get_text().set_fontweight("bold")

    for i, (bg, row_data) in enumerate(zip(tbl_bg, tbl_rows)):
        for j in range(5):
            t[i+1, j].set_facecolor(bg)
        is_section = (row_data[1] == "" and row_data[2] == "")
        if is_section:
            t[i+1, 0].get_text().set_fontweight("bold")
            t[i+1, 0].get_text().set_fontsize(8)
        else:
            for j_col, txt in [(2, row_data[2]), (3, row_data[3])]:
                if "benef." in txt:
                    t[i+1, j_col].get_text().set_color(C_POS)
                    t[i+1, j_col].get_text().set_fontweight("bold")
                elif "detrim." in txt:
                    t[i+1, j_col].get_text().set_color(C_NEG)
                    t[i+1, j_col].get_text().set_fontweight("bold")

    for i in range(len(tbl_rows)):
        t[i+1, 0].get_text().set_ha("left")
        t[i+1, 4].get_text().set_ha("left")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading training caches …")
    sm_cache = load_cache(SM_CACHE)
    dt_cache = load_cache(DT_CACHE)

    print("Computing IQR effects (SHAP-based) …")
    df = collect_effects(sm_cache, dt_cache)

    summary = summarise(df)
    print(f"  {len(summary)} unique features analysed")
    print(f"  {summary.design_relevant.sum()} design-relevant features")

    print(f"Generating {OUT_PDF.name} …")
    with PdfPages(OUT_PDF) as pdf:
        page_title(pdf)
        print("  p.1  title & methodology")

        page_summary(pdf, summary)
        print("  p.2  design-feature summary table")

        print("  p.3–5  SM detail per event type:")
        page_detail(pdf, df, "SM model", sm_or_dt="SM")

        print("  p.6–8  DT detail per event type:")
        page_detail(pdf, df, "DT model", sm_or_dt="DT")

        page_recommendations(pdf, summary)
        print("  p.9  planting pit recommendations")

        pdf.infodict().update({
            "Title":   "Planting Pit IQR Feature Effect Analysis",
            "Author":  "TreeDataBase — RF v4 SM / v3 DT",
            "Subject": "SHAP-based Q25→Q75 effects on soil moisture and drying time",
        })

    print(f"\n✓  Saved → {OUT_PDF}")

    # Print key findings to console
    print("\n=== KEY FINDINGS (design-relevant features, weighted avg) ===")
    print(f"  {'Feature':14s}  {'IQR range':22s}  "
          f"{'SM Δ%VWC':>10s}  {'SM rel%':>8s}  {'Ref SM':>7s}  "
          f"{'DT Δh':>9s}  {'DT rel%':>8s}  {'Ref DT':>7s}")
    design = summary[summary.design_relevant].copy()
    for _, r in design.iterrows():
        sm_s  = f"{r.avg_effect_sm_pct:+.3f}" if not np.isnan(r.avg_effect_sm_pct) else "    —"
        smr_s = f"{r.avg_rel_effect_sm:+.1f}%" if not np.isnan(r.avg_rel_effect_sm) else "    —"
        smm_s = f"{r.avg_mean_sm_pct:.2f}"     if not np.isnan(r.avg_mean_sm_pct)   else "  —"
        dt_s  = f"{r.avg_effect_dt_h:+.2f}"   if not np.isnan(r.avg_effect_dt_h)   else "    —"
        dtr_s = f"{r.avg_rel_effect_dt:+.1f}%" if not np.isnan(r.avg_rel_effect_dt) else "    —"
        dtm_s = f"{r.avg_mean_dt_h:.1f}"       if not np.isnan(r.avg_mean_dt_h)     else "  —"
        print(f"  {sh(r.feature):14s}  "
              f"Q25={_fmtq(r.q25):6s}–Q75={_fmtq(r.q75):6s} {r.unit:8s}  "
              f"{sm_s:>10s}  {smr_s:>8s}  {smm_s:>7s}  "
              f"{dt_s:>9s}  {dtr_s:>8s}  {dtm_s:>7s}")


if __name__ == "__main__":
    main()
