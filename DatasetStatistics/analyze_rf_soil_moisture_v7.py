#!/usr/bin/env python3
"""
Soil Moisture RF — v7  STANDALONE  (post-event rain removed; v2 CSV fix).

Changes vs. v4 (production baseline):
  CHANGE 1: rain_post24h_mm and rain_post72h_mm removed from predictor set.
            These features are not defensible predictors of the soil moisture
            RESPONSE to an event — they describe what happens after the peak.
            The exclusion rule (CHANGE 2) does not apply to the SM model because
            drying time is not the target; all events with a valid delta_pct_dyn
            remain in the training set.

  FIX: irrigation events now loaded from irrigation_events_v2_all.csv (same CSV
       used by run_boruta_selection.py). The v1 CSV was missing delta_pct_dyn-10/30/45,
       causing IRRIGATION SM to produce 0 samples.
       v7 restores IRRIGATION SM training (n≈3,051 per depth).

  STANDALONE: all dependencies inlined — does not import any other analysis script.

Rain events filtered to front-loaded events (BSC_XX11_Q12 + Huff Q1/Q2).
Irrigation types: IRRIGATION, IRRIGATION_STRONG.
Target variable:  delta_pct_dyn-{depth}  (VWC rise as % of sensor dynamic range)
Target transform: log1p applied before training (right-skewed distribution)

Reads:  TreeTabularData/rf_dataset.csv
        TreeTabularData/irrigation_events_v2_all.csv
        TreeTabularData/trees/{eui}/tree_data.json   (irrigation only)
        DatasetStatistics/boruta_results.pkl          (Boruta feature selection)
Writes: DatasetStatistics/rf_soil_moisture_v7_training_cache.pkl
        DatasetStatistics/rf_soil_moisture_v7_report.pdf
        DatasetStatistics/rf_soil_moisture_v7_results.csv
"""
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

from pathlib import Path
import json
import pickle
import re
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import spearmanr

try:
    import shap as _shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
RF_CSV     = REPO_ROOT / "TreeTabularData" / "rf_dataset.csv"
EVENTS_CSV = REPO_ROOT / "TreeTabularData" / "irrigation_events_v2_all.csv"
TREES_DIR  = REPO_ROOT / "TreeTabularData" / "trees"
OUT_PDF    = SCRIPT_DIR / "rf_soil_moisture_v7_report.pdf"
OUT_CSV    = SCRIPT_DIR / "rf_soil_moisture_v7_results.csv"
OUT_CACHE  = SCRIPT_DIR / "rf_soil_moisture_v7_training_cache.pkl"

_BORUTA_PKL = SCRIPT_DIR / "boruta_results.pkl"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EVENT_TYPES = ["BSC_XX11_Q12", "IRRIGATION", "IRRIGATION_STRONG"]

DEPTHS     = ["-10", "-30", "-45"]
DEPTH_LBLS = {"-10": "−10 cm", "-30": "−30 cm", "-45": "−45 cm"}
DEPTH_CLRS = {"-10": "#1b7837", "-30": "#762a83", "-45": "#e08214"}

TYPE_COLORS = {
    "BSC_XX11_Q12":    "#2d6a2d",
    "IRRIGATION":      "#D63030",
    "IRRIGATION_STRONG": "#8B0000",
}
TYPE_LABELS = {
    "BSC_XX11_Q12":    "BSC XX11 + Huff Q1/Q2  (front-loaded rain)",
    "IRRIGATION":      "Irrigation",
    "IRRIGATION_STRONG": "Irrigation strong",
}

N_TOP_FEATURES   = 20
TARGET_UNIT      = "log1p(ΔVWC % dyn. range)"
RF_ESTIMATORS    = 150
RF_SEED          = 42
CV_FOLDS         = 3
PERM_REPEATS     = 10
SHAP_MAX_SAMPLES = 2000
MIN_SAMPLES      = 30

# Tree attribute columns to load from tree_data.json (irrigation events only)
TREE_ATTRS = [
    "greenAtta2","greenAtta5","greenAtta7",
    "greenDeta2","greenDeta5","greenDeta7",
    "buildings2","buildings5","buildings7",
    "sealedSu2","sealedSu5","sealedSu7",
    "greenAttCD","greenDetCD","buildingCD","sealedSuCD",
    "amsl","slope","flowAcc","SVF","DTGW",
    "twi","twi2m5","twi5m","twi7m5",
    "tpi2m5","tpi5m","tpi7m5","tpiCD",
    "crownDiam","height","stemDiam","age_years",
    "pH","saltCont","conductiv","nSoluble","ammNSolubl",
    "nitrNSolub","mgSoluble","phSoluble","kSoluble",
    "part1Perc","part2Perc","part3Perc","part4Perc",
    "Kf","tpv","dbd","coarFrag",
    "pF1.5","pF1.8","pF2.0","pF2.5","pF3.0","pF4.2",
]

# Minimal ALWAYS_EXCLUDE: hard leakage prevention only.
# Boruta filtering (below) handles redundant / uninformative features.
ALWAYS_EXCLUDE = {
    "eui", "ev_start", "ev_end", "bsc",
    "event_time", "event_type", "label",
    "month", "doy", "season",
    # SM outcomes — never predictors
    "peak_vwc-10", "peak_vwc-30", "peak_vwc-45",
    "t_peak_min-10", "t_peak_min-30", "t_peak_min-45",
    "delta_pct_dyn-10", "delta_pct_dyn-30", "delta_pct_dyn-45",
    "delta_vwc-10", "delta_vwc-30", "delta_vwc-45",
    # dry_h is the drying-time target — not a predictor for SM
    "dry_h-10", "dry_h-30", "dry_h-45",
    # temp_sum encodes the same time window as the SM target
    "temp_sum-10", "temp_sum-30", "temp_sum-45",
    # CHANGE 1: post-event rainfall removed from predictor set
    "rain_post24h_mm", "rain_post72h_mm",
}

# ---------------------------------------------------------------------------
# Column rename maps — unify BSC-CSV and irrigation-JSON naming conventions
# ---------------------------------------------------------------------------
_IRR_COL_RENAME = {
    "tree_amsl": "elevation", "tree_slope": "slope",
    "tree_flowAcc": "flowAcc", "tree_SVF": "SVF",
    "tree_DTGW": "DTGW",
    "tree_twi": "twi", "tree_twi2m5": "twi_2m5",
    "tree_twi5m": "twi_5m", "tree_twi7m5": "twi_7m5",
    "tree_tpi2m5": "tpi_2m5", "tree_tpi5m": "tpi_5m",
    "tree_tpi7m5": "tpi_7m5", "tree_tpiCD": "tpiCD",
    "tree_crownDiam": "crownDiam", "tree_height": "height",
    "tree_stemDiam": "stemDiam", "tree_age_years": "age_years",
    "tree_greenAtta2": "greenAtta_2m5", "tree_greenAtta5": "greenAtta_5m",
    "tree_greenAtta7": "greenAtta_7m5", "tree_greenAtta_0to5": "greenAtta_0to5",
    "tree_greenAtta_0to7": "greenAtta_0to7", "tree_greenAttCD": "greenAttCD",
    "tree_greenDeta2": "greenDeta_2m5", "tree_greenDeta5": "greenDeta_5m",
    "tree_greenDeta7": "greenDeta_7m5", "tree_greenDeta_0to5": "greenDeta_0to5",
    "tree_greenDeta_0to7": "greenDeta_0to7", "tree_greenDetCD": "greenDetCD",
    "tree_green2": "green_2m5", "tree_green5": "green_5m",
    "tree_green7": "green_7m5", "tree_green0to5": "green_0to5",
    "tree_green0to7": "green_0to7",
    "tree_buildings2": "buildings_2m5", "tree_buildings5": "buildings_5m",
    "tree_buildings7": "buildings_7m5", "tree_buildings_0to5": "buildings_0to5",
    "tree_buildings_0to7": "buildings_0to7", "tree_buildingCD": "buildingCD",
    "tree_sealedSu2": "sealedSu_2m5", "tree_sealedSu5": "sealedSu_5m",
    "tree_sealedSu7": "sealedSu_7m5", "tree_sealedSu_0to5": "sealedSu_0to5",
    "tree_sealedSu_0to7": "sealedSu_0to7", "tree_sealedSuCD": "sealedSuCD",
    "tree_pH": "pH", "tree_saltCont": "saltContent",
    "tree_conductiv": "conductivity", "tree_nSoluble": "nSoluble",
    "tree_ammNSolubl": "ammNSoluble", "tree_nitrNSolub": "nitrNSoluble",
    "tree_mgSoluble": "mgSoluble", "tree_phSoluble": "pHSoluble",
    "tree_kSoluble": "kSoluble",
    "tree_part1Perc": "part1Perc", "tree_part2Perc": "part2Perc",
    "tree_part3Perc": "part3Perc", "tree_part4Perc": "part4Perc",
    "tree_Kf": "Kf", "tree_tpv": "tpv",
    "tree_dbd": "dbd", "tree_coarFrag": "coarFrag",
    "tree_pF1.5": "pF1.5", "tree_pF1.8": "pF1.8",
    "tree_pF2.0": "pF2.0", "tree_pF2.5": "pF2.5",
    "tree_pF3.0": "pF3.0", "tree_pF4.2": "pF4.2",
}

_BSC_COL_RENAME = {
    "elevation_m": "elevation",
    "slope_deg": "slope",
    "flowAccumulation": "flowAcc",
    "skyViewFactor": "SVF",
    "crownDiam_m": "crownDiam",
    "tpi2m5": "tpi_2m5", "tpi5m": "tpi_5m", "tpi7m5": "tpi_7m5",
    "twi2m5": "twi_2m5", "twi5m": "twi_5m", "twi7m5": "twi_7m5",
    "sealedSurf_2m5": "sealedSu_2m5", "sealedSurf_5m": "sealedSu_5m",
    "sealedSurf_7m5": "sealedSu_7m5",
    "sealedSu2": "sealedSu_2m5", "sealedSu5": "sealedSu_5m",
    "sealedSu7": "sealedSu_7m5",
}


def _unify_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Rename columns according to rename_map, dropping source if target already exists."""
    to_rename = {old: new for old, new in rename_map.items() if old in df.columns}
    safe = {}
    for old, new in to_rename.items():
        if new in df.columns and new != old:
            df = df.drop(columns=[old])
        else:
            safe[old] = new
    return df.rename(columns=safe)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tree_attrs(euis) -> pd.DataFrame:
    """Load numeric tree attributes from tree_data.json for a list of EUIs."""
    rows = []
    for eui in euis:
        jp = TREES_DIR / str(eui) / "tree_data.json"
        if not jp.exists():
            continue
        try:
            data = json.loads(jp.read_text(encoding="utf-8", errors="replace"))
            latest = data[-1] if isinstance(data, list) and data else {}
            row = {"eui": str(eui)}
            for k in TREE_ATTRS:
                v = latest.get(k)
                try:
                    row[f"tree_{k}"] = float(v) if v is not None else np.nan
                except (TypeError, ValueError):
                    row[f"tree_{k}"] = np.nan
            rows.append(row)
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("eui")


def _add_aggregate_landuse(df: pd.DataFrame) -> pd.DataFrame:
    """Add 0→5 m and 0→7.5 m aggregate land-use columns (sum of concentric rings)."""
    # Rain-event pattern: category_2m5 / category_5m / category_7m5
    for cat_src, cat_dst in [
        ('greenAtta', 'greenAtta'), ('greenDeta', 'greenDeta'),
        ('buildings', 'buildings'), ('sealedSurf', 'sealedSu'),
    ]:
        c2, c5, c7 = f"{cat_src}_2m5", f"{cat_src}_5m", f"{cat_src}_7m5"
        if c2 in df.columns and c5 in df.columns:
            df[f"{cat_dst}_0to5"] = df[[c2, c5]].sum(axis=1, min_count=1)
        if c2 in df.columns and c5 in df.columns and c7 in df.columns:
            df[f"{cat_dst}_0to7"] = df[[c2, c5, c7]].sum(axis=1, min_count=1)
    # Irrigation-event pattern: tree_category2 / tree_category5 / tree_category7
    for cat in ['greenAtta', 'greenDeta', 'buildings', 'sealedSu']:
        c2, c5, c7 = f"tree_{cat}2", f"tree_{cat}5", f"tree_{cat}7"
        if c2 in df.columns and c5 in df.columns:
            df[f"tree_{cat}_0to5"] = df[[c2, c5]].sum(axis=1, min_count=1)
        if c2 in df.columns and c5 in df.columns and c7 in df.columns:
            df[f"tree_{cat}_0to7"] = df[[c2, c5, c7]].sum(axis=1, min_count=1)
    # Combined green (attached + detached) — rain-event naming
    for r_atta, r_deta, dst in [
        ('greenAtta_2m5',  'greenDeta_2m5',  'green_2m5'),
        ('greenAtta_5m',   'greenDeta_5m',   'green_5m'),
        ('greenAtta_7m5',  'greenDeta_7m5',  'green_7m5'),
        ('greenAtta_0to5', 'greenDeta_0to5', 'green_0to5'),
        ('greenAtta_0to7', 'greenDeta_0to7', 'green_0to7'),
    ]:
        if r_atta in df.columns and r_deta in df.columns:
            df[dst] = df[[r_atta, r_deta]].sum(axis=1, min_count=1)
    # Combined green (attached + detached) — irrigation-event naming (tree_ prefix)
    for r_atta, r_deta, dst in [
        ('tree_greenAtta2',     'tree_greenDeta2',     'tree_green2'),
        ('tree_greenAtta5',     'tree_greenDeta5',     'tree_green5'),
        ('tree_greenAtta7',     'tree_greenDeta7',     'tree_green7'),
        ('tree_greenAtta_0to5', 'tree_greenDeta_0to5', 'tree_green0to5'),
        ('tree_greenAtta_0to7', 'tree_greenDeta_0to7', 'tree_green0to7'),
    ]:
        if r_atta in df.columns and r_deta in df.columns:
            df[dst] = df[[r_atta, r_deta]].sum(axis=1, min_count=1)
    return df


def load_event_groups() -> dict:
    """
    Returns {event_type: DataFrame} for each analysed group.
    Rain events come from rf_dataset.csv; irrigation from irrigation_events_v2_all.csv.
    """
    groups = {}

    # -- BSC rain events: front-loaded XX11 + Q1/Q2 -------------------------
    if not RF_CSV.exists():
        print(f"  [WARN] {RF_CSV.name} not found — rain models skipped")
    else:
        df = pd.read_csv(RF_CSV, dtype={"eui": str}, low_memory=False)
        bsc_s = df["bsc"].astype(str).str.zfill(4)
        mask  = (bsc_s.str[2:] == "11") & (df["huff_q"].isin([1, 2]))
        sub   = df[mask].copy()
        sub["day_cos"] = np.cos(2 * np.pi * sub["doy"].fillna(172) / 365)
        sub = _add_aggregate_landuse(sub)
        for d in DEPTHS:
            ts, dh, at = f'temp_sum{d}', f'dry_h{d}', f'avg_temp{d}'
            if ts in sub.columns and dh in sub.columns:
                sub[at] = sub[ts] / sub[dh].replace(0, np.nan)
        print(f"  BSC_XX11_Q12: {len(sub):,} / {len(df):,} rain events pass filter")
        if not sub.empty:
            sub = _unify_columns(sub, _BSC_COL_RENAME)
            groups["BSC_XX11_Q12"] = sub

    # -- Irrigation events (v2 CSV: has delta_pct_dyn-10/30/45) -----------
    if not EVENTS_CSV.exists():
        print(f"  [WARN] {EVENTS_CSV.name} not found — irrigation models skipped")
    else:
        df_irr = pd.read_csv(EVENTS_CSV, dtype={"eui": str}, low_memory=False)
        if "event_type" not in df_irr.columns and "label" in df_irr.columns:
            df_irr["event_type"] = df_irr["label"]
        elif "label" not in df_irr.columns and "event_type" in df_irr.columns:
            df_irr["label"] = df_irr["event_type"]
        if "month" not in df_irr.columns and "event_time" in df_irr.columns:
            ts = pd.to_datetime(df_irr["event_time"], errors="coerce")
            df_irr["month"] = ts.dt.month
            df_irr["doy"]   = ts.dt.dayofyear
        if "doy" not in df_irr.columns and "event_time" in df_irr.columns:
            df_irr["doy"] = pd.to_datetime(df_irr["event_time"], errors="coerce").dt.dayofyear
        df_irr["day_cos"] = np.cos(2 * np.pi * df_irr["doy"].fillna(172) / 365)
        attrs = load_tree_attrs(df_irr["eui"].unique())
        if not attrs.empty:
            new_cols = [c for c in attrs.columns if c not in df_irr.columns]
            if new_cols:
                df_irr = df_irr.join(attrs[new_cols], on="eui")
            print(f"  Joined tree attrs for {len(attrs):,} sensors")
        df_irr = _add_aggregate_landuse(df_irr)
        df_irr = _unify_columns(df_irr, _IRR_COL_RENAME)
        for et in ["IRRIGATION", "IRRIGATION_STRONG"]:
            sub = df_irr[df_irr["event_type"] == et].copy()
            for d in DEPTHS:
                ts, dh, at = f'temp_sum{d}', f'dry_h{d}', f'avg_temp{d}'
                if ts in sub.columns and dh in sub.columns:
                    sub[at] = sub[ts] / sub[dh].replace(0, np.nan)
            print(f"  {et}: {len(sub):,} events")
            if not sub.empty:
                groups[et] = sub

    return groups


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def prepare_features(ev_df: pd.DataFrame, target_col: str) -> tuple:
    depth = target_col.replace("delta_pct_dyn", "")
    exclude = ALWAYS_EXCLUDE | {
        # Raw VWC increments are direct transformations of the target
        "delta_vwc-10","delta_vwc-30","delta_vwc-45",
        # Exclude dry_h and pre_vwc / sensor stats for other depths
        "dry_h-10","dry_h-30","dry_h-45",
        "pre_vwc-10","pre_vwc-30","pre_vwc-45",
        "vwc_p05-10","vwc_p95-10","dyn_range-10",
        "vwc_p05-30","vwc_p95-30","dyn_range-30",
        "vwc_p05-45","vwc_p95-45","dyn_range-45",
    }
    # Re-include depth-specific stats for the target depth only
    # vwc_p05/p95 excluded entirely — they are redundant with dyn_range (= p95 - p05)
    for keep in [f"dyn_range{depth}", f"pre_vwc{depth}", f"dry_h{depth}"]:
        exclude.discard(keep)

    feature_cols = [
        c for c in ev_df.columns
        if c not in exclude and c != target_col
        and pd.api.types.is_numeric_dtype(ev_df[c])
    ]
    sub = ev_df[[target_col] + feature_cols].dropna(subset=[target_col])
    sub = sub.dropna(axis=1, how="all")
    feature_cols = [c for c in feature_cols if c in sub.columns]
    sub = sub.copy()
    sub[feature_cols] = sub[feature_cols].fillna(sub[feature_cols].median())
    return sub[feature_cols].values, sub[target_col].values, feature_cols, len(sub)


# ---------------------------------------------------------------------------
# Boruta feature selection (loaded from boruta_results.pkl)
# ---------------------------------------------------------------------------
_BORUTA_SELECTED = {}
_boruta_current_et = None

if _BORUTA_PKL.exists():
    with open(_BORUTA_PKL, "rb") as _f:
        _boruta_all = pickle.load(_f)
    for _et, _depths in _boruta_all.get("sm", {}).items():
        _BORUTA_SELECTED[_et] = {
            _d: set(_info["confirmed"]) | set(_info["tentative"])
            for _d, _info in _depths.items()
        }
    for _feat_set in (s for d in _BORUTA_SELECTED.values() for s in d.values()):
        if "month_cos" in _feat_set:
            _feat_set.discard("month_cos")
            _feat_set.add("day_cos")
        # CHANGE 1: force-remove from Boruta selection if present
        _feat_set.discard("rain_post24h_mm")
        _feat_set.discard("rain_post72h_mm")
    print(f"  [v7] Boruta results loaded for {len(_BORUTA_SELECTED)} event type(s): "
          f"{list(_BORUTA_SELECTED)}")
else:
    print("  [v7 WARN] boruta_results.pkl not found — using full feature set")

_orig_prepare_features = prepare_features


def prepare_features(ev_df, target_col):
    """v7 wrapper: base feature prep → log1p transform → Boruta filter."""
    X, y, feat_names, n = _orig_prepare_features(ev_df, target_col)
    # Keep only positive target values; apply log1p transform
    pos = y > 0
    X   = X[pos]
    y   = np.log1p(y[pos])
    n   = len(y)
    depth   = target_col.replace("delta_pct_dyn", "")
    allowed = _BORUTA_SELECTED.get(_boruta_current_et, {}).get(depth)
    if allowed:
        mask = [i for i, f in enumerate(feat_names) if f in allowed]
        if len(mask) >= 2:
            X          = X[:, mask]
            feat_names = [feat_names[i] for i in mask]
    return X, y, feat_names, n


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_and_evaluate(X, y, n_samples):
    n_folds = max(2, min(CV_FOLDS, n_samples // 10))
    rf = RandomForestRegressor(
        n_estimators=RF_ESTIMATORS, max_features="sqrt",
        min_samples_leaf=3, random_state=RF_SEED, n_jobs=-1,
    )
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=RF_SEED)

    print(f"      [1/4] cross-validation  ({n_folds} folds × {RF_ESTIMATORS} trees) …",
          end="", flush=True)
    t0 = time.perf_counter()
    r2s = cross_val_score(rf, X, y, cv=cv, scoring="r2")
    print(f"  CV R² = {r2s.mean():.3f} ± {r2s.std():.3f}  ({time.perf_counter()-t0:.1f} s)")

    print(f"      [2/4] fitting final RF  ({RF_ESTIMATORS} trees) …",
          end="", flush=True)
    t0 = time.perf_counter()
    rf.fit(X, y)
    r2_train = rf.score(X, y)
    y_pred   = rf.predict(X)
    print(f"  done  ({time.perf_counter()-t0:.1f} s)  train R² = {r2_train:.3f}")

    print(f"      [3/4] permutation importance  ({PERM_REPEATS} repeats) …",
          end="", flush=True)
    t0 = time.perf_counter()
    perm = permutation_importance(rf, X, y, n_repeats=PERM_REPEATS,
                                  random_state=RF_SEED, n_jobs=-1)
    print(f"  done  ({time.perf_counter()-t0:.1f} s)")

    shap_v = None
    X_shap = X
    y_shap = y
    if HAS_SHAP and n_samples >= 50:
        if SHAP_MAX_SAMPLES and n_samples > SHAP_MAX_SAMPLES:
            _rng  = np.random.default_rng(RF_SEED)
            _idx  = np.sort(_rng.choice(n_samples, SHAP_MAX_SAMPLES, replace=False))
            X_shap = X[_idx]
            y_shap = y[_idx]
            n_shap = SHAP_MAX_SAMPLES
            _note  = f"  (subsample {SHAP_MAX_SAMPLES:,} of {n_samples:,})"
        else:
            n_shap = n_samples
            _note  = ""
        chunk_size = 500
        n_chunks = (n_shap + chunk_size - 1) // chunk_size
        print(f"      [4/4] SHAP values  ({n_shap:,} samples{_note}, {n_chunks} batches) …",
              flush=True)
        t0 = time.perf_counter()
        try:
            explainer = _shap.TreeExplainer(rf)
            chunks = []
            for i, start in enumerate(range(0, n_shap, chunk_size), 1):
                end = min(start + chunk_size, n_shap)
                chunks.append(explainer.shap_values(X_shap[start:end]))
                elapsed = time.perf_counter() - t0
                print(f"\r             batch {i}/{n_chunks}  "
                      f"({end:,}/{n_shap:,} samples)  {elapsed:.0f} s elapsed",
                      end="", flush=True)
            print()
            sv = np.concatenate(chunks, axis=0)
            shap_v = sv[:, :, 0] if sv.ndim == 3 else sv
            print(f"      [4/4] done  ({time.perf_counter()-t0:.1f} s total)")
        except Exception as _e:
            print(f"\n      [4/4] failed — {_e}")
    else:
        print("      [4/4] SHAP skipped")

    return rf, r2s, perm.importances_mean, perm.importances_std, shap_v, X_shap, r2_train, y_pred, y_shap


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _importance_bar(ax, feat_names, imp_mean, imp_std, n_top, title,
                    depth_color, r2_mean, r2_std, n_samples,
                    y_std=None, target_unit=""):
    order    = np.argsort(imp_mean)[::-1][:n_top]
    top_mean = imp_mean[order]
    top_std  = imp_std[order]
    colors   = [depth_color if m > 0.0005 else "#cccccc" for m in top_mean]
    y_pos    = np.arange(len(order))
    ax.barh(y_pos, top_mean[::-1], xerr=top_std[::-1],
            color=colors[::-1], alpha=0.85, edgecolor="white",
            error_kw=dict(ecolor="gray", lw=0.7, capsize=2))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [feat_names[i].replace("tree_", "") for i in order[::-1]], fontsize=7.5)
    ax.set_xlabel(
        "Permutation importance (mean ΔR²)\n"
        "Positive = informative,  Negative = adds noise / redundant",
        fontsize=7.5,
    )
    ax.axvline(0, color="gray", lw=0.7, ls="--")
    ax.set_title(
        f"{title}\nCV R² = {r2_mean:.3f} ± {r2_std:.3f}   n = {n_samples:,}",
        fontsize=9, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    if y_std is not None and target_unit:
        _est = y_std * (0.04 ** 0.5)
        ax.text(
            0.98, 0.02,
            f"Physical scale (rough):\n"
            f"ΔR²=0.04 → ±{_est:.1f} {target_unit}\n"
            f"(σ_y × √importance)\n"
            f"Use SHAP for exact units.",
            transform=ax.transAxes, fontsize=6, va="bottom", ha="right",
            color="#444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fafafa",
                      alpha=0.85, edgecolor="#ccc"),
        )


def _col_xfrac(ax, pts: float) -> float:
    """Convert a desired point-offset from the y-axis spine to an axes x-fraction.

    Using this helper makes annotation column spacing consistent in absolute
    points regardless of the axes physical width.
    """
    fig = ax.figure
    ax_width_pts = ax.get_position().width * fig.get_size_inches()[0] * 72.0
    return -pts / ax_width_pts


def _fmt_pval(val: float, feat_name: str = '') -> str:
    """Format a p10/p90 feature-value percentile with a best-effort unit suffix."""
    if not np.isfinite(val):
        return 'n/a'
    n = feat_name.lower().replace('tree_', '')
    if 'temp' in n:
        unit = '°C'
    elif '_mm' in n or n.startswith('rain_pre') or n.startswith('rain_post'):
        unit = 'mm'
    elif n.startswith('max_int'):
        unit = 'mm/h'
    elif 'duration_h' in n or (n.startswith('dry_h') and 'perc' not in n):
        unit = 'h'
    elif n.startswith('stac'):
        unit = 'mm'
    elif 'perc' in n:
        unit = '%'
    elif 'diam' in n or n == 'height':
        unit = 'm'
    elif 'elevation' in n or n == 'amsl':
        unit = 'm'
    elif 'dtgw' in n or 'depthto' in n:
        unit = 'm'
    elif 'latitude' in n:
        unit = '°N'
    elif 'longitude' in n:
        unit = '°E'
    elif n == 'doy':
        unit = 'd'
    else:
        unit = ''
    s = f'{val:.3g}'
    return f'{s} {unit}' if unit else s


def _shap_beeswarm(ax, feat_names, X, shap_vals, n_top, title, target_unit="", x_lim=None, allow_area=False, y_mean=None):
    """Signed SHAP beeswarm: each dot = one sample, coloured by feature value."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:n_top]
    if not allow_area:
        order = [i for i in order if not re.match(r'^area_?\d+$', feat_names[i])]
    rng      = np.random.RandomState(0)
    for y_i, feat_idx in enumerate(order[::-1]):
        sv  = shap_vals[:, feat_idx]
        fv  = X[:, feat_idx]
        lo = np.nanpercentile(fv, 10)
        hi = np.nanpercentile(fv, 90)
        fv_n = np.clip((fv - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        colors = plt.cm.RdBu_r(fv_n)
        jitter = rng.uniform(-0.28, 0.28, len(sv))
        ax.scatter(sv, y_i + jitter, c=colors, s=5, alpha=0.55,
                   linewidths=0, rasterized=True)
        mean_sv = float(np.mean(sv))
        corr_val = float(spearmanr(fv, sv)[0]) if np.std(fv) > 1e-9 else 0.0
        _x_ms = _col_xfrac(ax, 30)
        _x_rh = _col_xfrac(ax, 78)
        _x_lo = _col_xfrac(ax, 130)
        _x_hi = _col_xfrac(ax, 182)
        ax.text(_x_ms, y_i, f"{mean_sv:+.3g}",
                transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=7,
                color="#cc0000" if mean_sv >= 0 else "#0044cc",
                clip_on=False)
        ax.text(_x_rh, y_i, f"{corr_val:+.2f}",
                transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=7,
                color="#cc0000" if corr_val >= 0 else "#0044cc",
                clip_on=False)
        ax.text(_x_lo, y_i, _fmt_pval(lo, feat_names[feat_idx]),
                transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=6.5, color="#555",
                clip_on=False)
        ax.text(_x_hi, y_i, _fmt_pval(hi, feat_names[feat_idx]),
                transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=6.5, color="#555",
                clip_on=False)
    if x_lim is None:
        all_sv = np.concatenate([shap_vals[:, i] for i in order])
        finite = all_sv[np.isfinite(all_sv)]
        lo = float(np.nanmin(finite)) if len(finite) else -1.0
        hi = min(float(np.nanmax(finite)) if len(finite) else 1.0, 300.0)
        pad = max(0.02 * (hi - lo), 1e-6)
        left, right = lo - pad, hi + pad
    elif isinstance(x_lim, (tuple, list)):
        left, right = float(x_lim[0]), float(x_lim[1])
    else:
        left, right = float(-x_lim), float(x_lim)
    ax.set_xlim(left, right)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [feat_names[i].replace("tree_", "") for i in order[::-1]], fontsize=7)
    ax.tick_params(axis='y', pad=200)
    _tu = f" [{target_unit}]" if target_unit else ""
    _interp = (f"   │   +X {target_unit} = feature contributes +X {target_unit} to VWC response"
               if target_unit else "")
    ax.set_xlabel(
        f"SHAP value{_tu}  (+: greater VWC rise  ·  −: smaller VWC rise){_interp}\n"
        "Dot colour: feature value  (blue = low,  red = high)\n"
        "Left margin — μ SHAP: mean signed SHAP  ·  ρ(fv,sv): Spearman ρ(feature value, SHAP)\n"
        "  ρ>0: high value → greater rise  ·  ρ<0: → smaller  ·  |ρ|→1: consistent direction\n"
        "lo(fv) / hi(fv): 10th / 90th percentile of feature value (= color scale boundaries)",
        fontsize=7.5,
    )
    _mean_sfx = (f"  ·  mean = {y_mean:.2f} {target_unit}" if y_mean is not None else "")
    ax.set_title(f"SHAP beeswarm — {title}{_mean_sfx}", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.text(_col_xfrac(ax, 30), 1.01, "μ SHAP", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.5, fontweight="bold", clip_on=False)
    ax.text(_col_xfrac(ax, 78), 1.01, "ρ(fv,sv)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.5, fontweight="bold", clip_on=False)
    ax.text(_col_xfrac(ax, 130), 1.01, "lo(fv)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.5, fontweight="bold", clip_on=False)
    ax.text(_col_xfrac(ax, 182), 1.01, "hi(fv)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.5, fontweight="bold", clip_on=False)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap="RdBu_r", norm=Normalize(0, 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.35, aspect=10, pad=0.01)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"], fontsize=6.5)
    cb.set_label("Feature\nvalue", fontsize=6.5)


# ---------------------------------------------------------------------------
# Combined SHAP beeswarm page (all depths pooled) + interpretation text
# ---------------------------------------------------------------------------
def build_shap_combined_page(ev_type: str, depth_data: list, x_lim=None) -> plt.Figure:
    """
    Pools SHAP values from every depth model for one event type and draws a
    single beeswarm ranked by mean |SHAP| across depths.  A text panel below
    explains how to convert SHAP values to physical units (absolute VWC %).

    depth_data: list of (feat_names, X, shap_v, y) — one entry per depth;
                entries with shap_v = None are skipped.
    """
    valid = [(fn, X, sv, y) for fn, X, sv, y in depth_data if sv is not None]
    if not valid:
        return None

    # Pool SHAP values — normalized to % of each depth's mean before pooling
    # so that depths with different absolute response magnitudes are comparable.
    feat_shap: dict = {}   # feature_name -> list[1-D array]
    feat_fval: dict = {}   # feature_name -> list[1-D array]
    all_y = np.concatenate([y for _, _, _, y in valid])
    depth_means_list = [float(np.nanmean(y)) for _, _, _, y in valid]

    def _merge_depth(name):
        """Strip depth suffix (-10/-30/-45) so each depth contributes to the same row."""
        for d in DEPTHS:
            if name.endswith(d):
                return name[:-len(d)]
        return name

    for (feat_names, X, shap_v, _y), _dm in zip(valid, depth_means_list):
        _scale = max(abs(_dm), 1e-6)
        shap_v_norm = shap_v / _scale * 100   # → % of this depth's mean
        for fi, name in enumerate(feat_names):
            if re.match(r'^area_?\d+$', name):
                continue
            feat_shap.setdefault(_merge_depth(name), []).append(shap_v_norm[:, fi])
            feat_fval.setdefault(_merge_depth(name), []).append(X[:, fi])

    pooled_sv = {n: np.concatenate(arrs) for n, arrs in feat_shap.items()}
    pooled_fv = {n: np.concatenate(arrs) for n, arrs in feat_fval.items()}

    mean_abs  = {n: np.abs(sv).mean() for n, sv in pooled_sv.items()}
    top_names = sorted(mean_abs, key=mean_abs.get, reverse=True)[:N_TOP_FEATURES]
    if not top_names:
        return None

    # Summary statistics for the interpretation text
    top1     = top_names[0]
    sv_top1  = pooled_sv[top1]
    p25      = float(np.percentile(sv_top1, 25))
    p75      = float(np.percentile(sv_top1, 75))
    y_mean   = float(all_y.mean())
    y_std    = float(all_y.std())
    n_pooled = len(all_y)
    dyn_ref  = 18.0   # reference dynamic range (% VWC) used in worked examples

    clr = TYPE_COLORS.get(ev_type, "#555")
    fig_h = max(11, len(top_names) * 0.42 + 6.5)
    fig = plt.figure(figsize=(14, fig_h))
    gs  = mgridspec.GridSpec(2, 1, height_ratios=[3, 1.15], hspace=0.44)

    ax_bee = fig.add_subplot(gs[0])
    ax_txt = fig.add_subplot(gs[1])
    ax_txt.axis("off")

    # ---- beeswarm ----
    all_sv_flat = np.concatenate(list(pooled_sv.values()))
    rng = np.random.RandomState(0)
    for y_i, name in enumerate(reversed(top_names)):
        sv = pooled_sv[name]
        fv = pooled_fv[name]
        lo = np.nanpercentile(fv, 10)
        hi = np.nanpercentile(fv, 90)
        fv_n   = np.clip((fv - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        colors = plt.cm.RdBu_r(fv_n)
        jitter = rng.uniform(-0.28, 0.28, len(sv))
        ax_bee.scatter(sv, y_i + jitter, c=colors, s=6, alpha=0.45,
                       linewidths=0, rasterized=True)
        mean_sv = float(np.mean(sv))
        corr_val = float(spearmanr(fv, sv)[0]) if np.std(fv) > 1e-9 else 0.0
        _x_ms = _col_xfrac(ax_bee, 30)
        _x_rh = _col_xfrac(ax_bee, 78)
        _x_lo = _col_xfrac(ax_bee, 130)
        _x_hi = _col_xfrac(ax_bee, 182)
        ax_bee.text(_x_ms, y_i, f"{mean_sv:+.3g}",
                    transform=ax_bee.get_yaxis_transform(),
                    ha="center", va="center", fontsize=7,
                    color="#cc0000" if mean_sv >= 0 else "#0044cc",
                    clip_on=False)
        ax_bee.text(_x_rh, y_i, f"{corr_val:+.2f}",
                    transform=ax_bee.get_yaxis_transform(),
                    ha="center", va="center", fontsize=7,
                    color="#cc0000" if corr_val >= 0 else "#0044cc",
                    clip_on=False)
        ax_bee.text(_x_lo, y_i, _fmt_pval(lo, name),
                    transform=ax_bee.get_yaxis_transform(),
                    ha="center", va="center", fontsize=6.5, color="#555",
                    clip_on=False)
        ax_bee.text(_x_hi, y_i, _fmt_pval(hi, name),
                    transform=ax_bee.get_yaxis_transform(),
                    ha="center", va="center", fontsize=6.5, color="#555",
                    clip_on=False)

    if x_lim is None:
        all_sv = np.concatenate([pooled_sv[n] for n in top_names])
        finite = all_sv[np.isfinite(all_sv)]
        lo = float(np.nanmin(finite)) if len(finite) else -1.0
        hi = min(float(np.nanmax(finite)) if len(finite) else 1.0, 300.0)
        pad = max(0.02 * (hi - lo), 1e-6)
        left, right = lo - pad, hi + pad
    elif isinstance(x_lim, (tuple, list)):
        left, right = float(x_lim[0]), float(x_lim[1])
    else:
        left, right = float(-x_lim), float(x_lim)
    ax_bee.set_xlim(left, right)
    ax_bee.axvline(0, color="gray", lw=0.8, ls="--")
    ax_bee.set_yticks(range(len(top_names)))
    ax_bee.set_yticklabels(
        [n.replace("tree_", "") for n in reversed(top_names)], fontsize=7)
    ax_bee.tick_params(axis='y', pad=200)
    ax_bee.set_xlabel(
        f"SHAP value [% of depth mean]   +  → higher VWC response,   −  → lower VWC response\n"
        f"Dot colour: feature value  (blue = low,  red = high)   │   "
        f"Dots pooled from {len(valid)} depth model(s); each depth's SHAP ÷ depth mean × 100 before pooling\n"
        f"SHAP = +10 means feature pushes predicted VWC response 10 % above that depth's mean response",
        fontsize=8,
    )
    _dm_str = "  /  ".join(f"{m:.1f}" for m in depth_means_list)
    ax_bee.set_title(
        f"SHAP Beeswarm — All Depths Combined  (normalized to % of depth mean)\n"
        f"{TYPE_LABELS.get(ev_type, ev_type)}   "
        f"(n = {n_pooled:,} event×depth samples  |  depth means: {_dm_str} % dyn.range)",
        fontsize=10, fontweight="bold", color=clr,
    )
    ax_bee.grid(axis="x", alpha=0.25)
    ax_bee.text(_col_xfrac(ax_bee, 30), 1.01, "μ SHAP", transform=ax_bee.transAxes,
                ha="center", va="bottom", fontsize=6.5, fontweight="bold", clip_on=False)
    ax_bee.text(_col_xfrac(ax_bee, 78), 1.01, "ρ(fv,sv)", transform=ax_bee.transAxes,
                ha="center", va="bottom", fontsize=6.5, fontweight="bold", clip_on=False)
    ax_bee.text(_col_xfrac(ax_bee, 130), 1.01, "lo(fv)", transform=ax_bee.transAxes,
                ha="center", va="bottom", fontsize=6.5, fontweight="bold", clip_on=False)
    ax_bee.text(_col_xfrac(ax_bee, 182), 1.01, "hi(fv)", transform=ax_bee.transAxes,
                ha="center", va="bottom", fontsize=6.5, fontweight="bold", clip_on=False)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap="RdBu_r", norm=Normalize(0, 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax_bee, shrink=0.30, aspect=14, pad=0.01)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"], fontsize=7)
    cb.set_label("Feature\nvalue", fontsize=7)

    # ---- interpretation text ----
    _dm_rows = "\n".join(
        f"  Depth {i+1}: mean = {m:.1f} % dyn.range  →  "
        f"SHAP +10 % of mean = +{m * 0.10:.1f} % dyn.range  ≈  "
        f"+{m * 0.10 * dyn_ref / 100:.3f} % abs. VWC  (at dyn.range = {dyn_ref:.0f} %)"
        for i, m in enumerate(depth_means_list)
    )
    lines = [
        "HOW TO READ THIS CHART  (Combined depth view — SHAP normalized to % of depth mean)",
        "═" * 90,
        "SHAP value (x-axis) — unit: % of depth-specific mean VWC response",
        "  Each depth's SHAP values were divided by that depth's training mean before pooling.",
        "  This makes all depths directly comparable regardless of their absolute response magnitude.",
        "  • +10 %  →  this feature pushes the predicted VWC response 10 % above that depth's mean.",
        "  • −10 %  →  this feature pulls the predicted VWC response 10 % below that depth's mean.",
        "",
        f"Per-depth means used for normalization ({len(valid)} depth(s)):",
        _dm_rows,
        "",
        f"Example — top feature  \"{top1.replace('tree_', '')}\"  "
        f"(normalized SHAP IQR: {p25:+.1f} to {p75:+.1f} % of depth mean):",
        f"  Low feature value  → SHAP ≈ {p25:+.1f}%   High feature value → SHAP ≈ {p75:+.1f}%",
        f"  Difference: ~{p75 - p25:.1f} % of each depth's mean  →  in physical units per depth:",
        *[f"    Depth {i+1} (mean {m:.1f} % dyn.range):  Δ ≈ {(p75 - p25) * m / 100:.1f} % dyn.range  "
          f"≈ {(p75 - p25) * m / 100 * dyn_ref / 100:.3f} % abs. VWC"
          for i, m in enumerate(depth_means_list)],
        "",
        f"Total samples: {n_pooled:,} pooled across {len(valid)} depth(s)  │  "
        f"μ SHAP and ρ in the left margin are computed on the normalized values.",
        "",
        "Permutation importance (bar charts on preceding pages) — unit: mean ΔR²  (NOT in % dyn.range!)",
        f"  Physical scale estimate per depth:  contribution ≈ σ_y × √importance",
        f"  (use SHAP values, not permutation importance, for direct attribution in physical units)",
    ]
    ax_txt.text(
        0.01, 0.97, "\n".join(lines),
        transform=ax_txt.transAxes, fontsize=7.8, va="top", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f4f8",
                  edgecolor="#aaa", alpha=0.92),
    )
    return fig


_LAND_USE_PFXS = (
    'green', 'buildings', 'sealedSu', 'sealedSurf',
    'tree_green', 'tree_buildings', 'tree_sealedSu',
)

_TOPO_PFXS = (
    'tpi', 'twi', 'slope', 'amsl', 'elevation', 'flowAcc', 'flotAcc', 'flowAccum',
    'SVF', 'skyViewFactor', 'DTGW', 'depthToGroundwater',
    'latitude', 'longitude',
    'tree_tpi', 'tree_twi', 'tree_slope', 'tree_amsl',
    'tree_flowAcc', 'tree_flotAcc', 'tree_SVF', 'tree_DTGW',
)

_TREE_PFXS = (
    'crownDiam', 'height', 'stemDiam', 'vitality',
    'tree_crownDiam', 'tree_height', 'tree_stemDiam', 'tree_vitality',
)

_RAIN_PFXS = (
    'total_mm', 'rain_pre', 'rain_post', 'max_intensity', 'stac', 'duration_h',
    'doy', 'month', 'huff_q', 'tpv', 'area',
)


# ---------------------------------------------------------------------------
# Pooled beeswarm helper (shared by landuse and topo pages)
# ---------------------------------------------------------------------------
def _draw_pooled_beeswarm(ax, pooled_sv: dict, pooled_fv: dict, theme: str,
                          target_unit: str = "", x_lim=None, x_right_cap=None,
                          depth_means=None):
    """Beeswarm from pre-pooled {name: 1-D array} dicts, sorted by mean |SHAP|."""
    top_names = sorted(pooled_sv, key=lambda n: np.abs(pooled_sv[n]).mean(), reverse=True)
    if not top_names:
        ax.text(0.5, 0.5, "No SHAP data", ha="center", va="center", transform=ax.transAxes)
        return
    rng = np.random.RandomState(0)
    for y_i, name in enumerate(reversed(top_names)):
        sv = pooled_sv[name]
        fv = pooled_fv[name]
        lo_f = np.nanpercentile(fv, 10)
        hi_f = np.nanpercentile(fv, 90)
        fv_n   = np.clip((fv - lo_f) / (hi_f - lo_f + 1e-12), 0.0, 1.0)
        colors = plt.cm.RdBu_r(fv_n)
        jitter = rng.uniform(-0.28, 0.28, len(sv))
        ax.scatter(sv, y_i + jitter, c=colors, s=5, alpha=0.55,
                   linewidths=0, rasterized=True)
        mean_sv = float(np.mean(sv))
        corr_val = float(spearmanr(fv, sv)[0]) if np.std(fv) > 1e-9 else 0.0
        _x_ms = _col_xfrac(ax, 30)
        _x_rh = _col_xfrac(ax, 78)
        _x_lo = _col_xfrac(ax, 130)
        _x_hi = _col_xfrac(ax, 182)
        ax.text(_x_ms, y_i, f"{mean_sv:+.3g}",
                transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=7,
                color="#cc0000" if mean_sv >= 0 else "#0044cc",
                clip_on=False)
        ax.text(_x_rh, y_i, f"{corr_val:+.2f}",
                transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=7,
                color="#cc0000" if corr_val >= 0 else "#0044cc",
                clip_on=False)
        ax.text(_x_lo, y_i, _fmt_pval(lo_f, name),
                transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=6.5, color="#555",
                clip_on=False)
        ax.text(_x_hi, y_i, _fmt_pval(hi_f, name),
                transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=6.5, color="#555",
                clip_on=False)
    if x_lim is None:
        all_sv = np.concatenate([pooled_sv[n] for n in top_names])
        finite = all_sv[np.isfinite(all_sv)]
        lo_x = float(np.nanmin(finite)) if len(finite) else -1.0
        hi_x = float(np.nanmax(finite)) if len(finite) else  1.0
        if x_right_cap is not None:
            hi_x = min(hi_x, x_right_cap)
        pad = max(0.02 * (hi_x - lo_x), 1e-6)
        left, right = lo_x - pad, hi_x + pad
    elif isinstance(x_lim, (tuple, list)):
        left, right = float(x_lim[0]), float(x_lim[1])
    else:
        left, right = float(-x_lim), float(x_lim)
    ax.set_xlim(left, right)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels([n.replace("tree_", "") for n in reversed(top_names)], fontsize=7)
    ax.tick_params(axis='y', pad=200)
    _normalized = depth_means is not None and len(depth_means) > 0
    _tu = " [% of depth mean]" if _normalized else (f" [{target_unit}]" if target_unit else "")
    _norm_sfx = "  (normalized to % of depth mean)" if _normalized else ""
    _depth_info = ""
    if _normalized:
        _dm_str = "  /  ".join(
            f"Depth {i+1}: {m:.1f} {target_unit}" for i, m in enumerate(depth_means)
        )
        _depth_info = (
            f"\nNormalization: raw SHAP ÷ depth mean → values in % of mean, comparable across depths."
            f"  Depth means: {_dm_str}"
        )
    ax.set_xlabel(
        f"SHAP value{_tu}  (all depths pooled{_norm_sfx})  ·  +: greater VWC rise  ·  −: smaller VWC rise\n"
        "Dot colour: feature value  (blue = low,  red = high)\n"
        "Left margin — μ SHAP: mean signed SHAP  ·  ρ(fv,sv): Spearman ρ(feature value, SHAP)\n"
        "  ρ>0: high value → greater rise  ·  ρ<0: → smaller  ·  |ρ|→1: consistent direction\n"
        f"lo(fv) / hi(fv): 10th / 90th percentile of feature value (= color scale boundaries){_depth_info}",
        fontsize=7.5,
    )
    ax.set_title(f"SHAP beeswarm (all depths pooled{_norm_sfx}) — {theme}", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.text(_col_xfrac(ax, 30), 1.01, "μ SHAP", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.5, fontweight="bold", clip_on=False)
    ax.text(_col_xfrac(ax, 78), 1.01, "ρ(fv,sv)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.5, fontweight="bold", clip_on=False)
    ax.text(_col_xfrac(ax, 130), 1.01, "lo(fv)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.5, fontweight="bold", clip_on=False)
    ax.text(_col_xfrac(ax, 182), 1.01, "hi(fv)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.5, fontweight="bold", clip_on=False)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap="RdBu_r", norm=Normalize(0, 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.35, aspect=10, pad=0.01)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"], fontsize=6.5)
    cb.set_label("Feature\nvalue", fontsize=6.5)


# ---------------------------------------------------------------------------
# Land-use focused page
# ---------------------------------------------------------------------------
def build_landuse_page(ev_type: str, depth_results: list, n_events: int,
                       x_lim_shap=None) -> plt.Figure:
    """
    Dedicated page showing only land-use features (green area / sealed surface)
    in permutation importance and SHAP beeswarm for each depth, plus a pooled
    all-depths beeswarm at the bottom.
    Returns None if no land-use features are present in any depth model.
    """
    clr   = TYPE_COLORS.get(ev_type, "#555")
    lbl   = TYPE_LABELS.get(ev_type, ev_type)
    ncols = 2 if HAS_SHAP else 1
    n_rows = len(DEPTHS) + (1 if HAS_SHAP else 0)

    fig = plt.figure(figsize=(20, 6 * n_rows + 2))
    fig.suptitle(
        f"Land Use Feature Analysis — {lbl}\n"
        f"Target: soil moisture response [{TARGET_UNIT}] per depth   ({n_events:,} events)\n"
        "Features: green (combined) / greenAttached / greenDetached / buildings / sealedSurface "
        "(2.5 / 5 / 7.5 m rings, 0→5 m + 0→7.5 m aggregates, crown-diameter)",
        fontsize=12, fontweight="bold", color=clr, y=1.002,
    )
    gs = mgridspec.GridSpec(n_rows, ncols, figure=fig,
                            hspace=0.55, wspace=0.75)

    any_plotted = False
    pool_sv: dict = {}
    pool_fv: dict = {}
    depth_means_list: list = []

    for row_i, dr in enumerate(depth_results):
        if dr['skip']:
            ax = fig.add_subplot(gs[row_i, 0])
            ax.text(0.5, 0.5, dr['msg'],
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Depth {dr['dlbl']}", fontsize=9)
            continue

        lu_mask = np.array([any(fn.startswith(p) for p in _LAND_USE_PFXS)
                            for fn in dr['feat_names']])
        if not lu_mask.any():
            ax = fig.add_subplot(gs[row_i, 0])
            ax.text(0.5, 0.5, "No land-use features in model",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Depth {dr['dlbl']}", fontsize=9)
            continue

        lu_feat = [fn for fn, m in zip(dr['feat_names'], lu_mask) if m]
        lu_imp  = dr['imp_m'][lu_mask]
        lu_std  = dr['imp_s'][lu_mask]

        ax0 = fig.add_subplot(gs[row_i, 0])
        _importance_bar(ax0, lu_feat, lu_imp, lu_std, len(lu_feat),
                        f"Depth {dr['dlbl']} — land use permutation importance",
                        dr['dclr'], dr['r2s'].mean(), dr['r2s'].std(), dr['n_samp'],
                        y_std=None, target_unit=TARGET_UNIT)

        if HAS_SHAP and dr['shap_v'] is not None and ncols == 2:
            lu_shap = dr['shap_v'][:, lu_mask]
            lu_X    = dr['X'][:, lu_mask]
            ax1 = fig.add_subplot(gs[row_i, 1])
            _shap_beeswarm(ax1, lu_feat, lu_X, lu_shap, len(lu_feat),
                           f"Depth {dr['dlbl']}", target_unit=TARGET_UNIT,
                           x_lim=x_lim_shap, y_mean=float(np.nanmean(dr['y'])))
            _dm = float(np.nanmean(dr['y']))
            _scale = max(abs(_dm), 1e-6)
            depth_means_list.append(_dm)
            for fi, fn in enumerate(lu_feat):
                pool_sv.setdefault(fn, []).append(lu_shap[:, fi] / _scale * 100)
                pool_fv.setdefault(fn, []).append(lu_X[:, fi])
        any_plotted = True

    if HAS_SHAP and pool_sv:
        pooled_sv = {n: np.concatenate(arrs) for n, arrs in pool_sv.items()}
        pooled_fv = {n: np.concatenate(arrs) for n, arrs in pool_fv.items()}
        ax_pool = fig.add_subplot(gs[len(DEPTHS), :])
        _draw_pooled_beeswarm(ax_pool, pooled_sv, pooled_fv,
                              "land use", target_unit=TARGET_UNIT,
                              depth_means=depth_means_list)

    if not any_plotted:
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Terrain / topographic feature page
# ---------------------------------------------------------------------------
def build_topo_page(ev_type: str, depth_results: list, n_events: int,
                    x_lim_shap=None) -> plt.Figure:
    """
    Dedicated page showing only topographic/terrain features (TPI, TWI, slope,
    amsl, SVF, DTGW, flow accumulation) per depth, plus a pooled all-depths
    beeswarm at the bottom.
    Returns None if no terrain features are present in any depth model.
    """
    clr   = TYPE_COLORS.get(ev_type, "#555")
    lbl   = TYPE_LABELS.get(ev_type, ev_type)
    ncols = 2 if HAS_SHAP else 1
    n_rows = len(DEPTHS) + (1 if HAS_SHAP else 0)

    fig = plt.figure(figsize=(20, 6 * n_rows + 2))
    fig.suptitle(
        f"Terrain Feature Analysis — {lbl}\n"
        f"Target: VWC rise [{TARGET_UNIT}] per depth   ({n_events:,} events)\n"
        "Features: TWI · TPI (2.5/5/7.5 m + crown-diam) · slope · amsl · "
        "flow accumulation · SVF · depth to groundwater",
        fontsize=12, fontweight="bold", color=clr, y=1.002,
    )
    gs = mgridspec.GridSpec(n_rows, ncols, figure=fig,
                            hspace=0.55, wspace=0.75)

    any_plotted = False
    pool_sv: dict = {}
    pool_fv: dict = {}
    depth_means_list: list = []

    for row_i, dr in enumerate(depth_results):
        if dr['skip']:
            ax = fig.add_subplot(gs[row_i, 0])
            ax.text(0.5, 0.5, dr['msg'],
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Depth {dr['dlbl']}", fontsize=9)
            continue

        tp_mask = np.array([any(fn.startswith(p) for p in _TOPO_PFXS)
                            for fn in dr['feat_names']])
        if not tp_mask.any():
            ax = fig.add_subplot(gs[row_i, 0])
            ax.text(0.5, 0.5, "No terrain features in model",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Depth {dr['dlbl']}", fontsize=9)
            continue

        tp_feat = [fn for fn, m in zip(dr['feat_names'], tp_mask) if m]
        tp_imp  = dr['imp_m'][tp_mask]
        tp_std  = dr['imp_s'][tp_mask]

        ax0 = fig.add_subplot(gs[row_i, 0])
        _importance_bar(ax0, tp_feat, tp_imp, tp_std, len(tp_feat),
                        f"Depth {dr['dlbl']} — terrain permutation importance",
                        dr['dclr'], dr['r2s'].mean(), dr['r2s'].std(), dr['n_samp'],
                        y_std=None, target_unit=TARGET_UNIT)

        if HAS_SHAP and dr['shap_v'] is not None and ncols == 2:
            tp_shap = dr['shap_v'][:, tp_mask]
            tp_X    = dr['X'][:, tp_mask]
            ax1 = fig.add_subplot(gs[row_i, 1])
            _shap_beeswarm(ax1, tp_feat, tp_X, tp_shap, len(tp_feat),
                           f"Depth {dr['dlbl']}", target_unit=TARGET_UNIT,
                           x_lim=x_lim_shap, y_mean=float(np.nanmean(dr['y'])))
            _dm = float(np.nanmean(dr['y']))
            _scale = max(abs(_dm), 1e-6)
            depth_means_list.append(_dm)
            for fi, fn in enumerate(tp_feat):
                pool_sv.setdefault(fn, []).append(tp_shap[:, fi] / _scale * 100)
                pool_fv.setdefault(fn, []).append(tp_X[:, fi])
        any_plotted = True

    if HAS_SHAP and pool_sv:
        pooled_sv = {n: np.concatenate(arrs) for n, arrs in pool_sv.items()}
        pooled_fv = {n: np.concatenate(arrs) for n, arrs in pool_fv.items()}
        ax_pool = fig.add_subplot(gs[len(DEPTHS), :])
        _draw_pooled_beeswarm(ax_pool, pooled_sv, pooled_fv,
                              "terrain", target_unit=TARGET_UNIT,
                              depth_means=depth_means_list)

    if not any_plotted:
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Tree morphology feature page
# ---------------------------------------------------------------------------
def build_tree_page(ev_type: str, depth_results: list, n_events: int,
                    x_lim_shap=None) -> plt.Figure:
    """
    Dedicated page showing only tree morphology features (crown diameter, height,
    stem diameter, vitality) per depth, plus a pooled all-depths beeswarm at
    the bottom.
    Returns None if no tree morphology features are present in any depth model.
    """
    clr   = TYPE_COLORS.get(ev_type, "#555")
    lbl   = TYPE_LABELS.get(ev_type, ev_type)
    ncols = 2 if HAS_SHAP else 1
    n_rows = len(DEPTHS) + (1 if HAS_SHAP else 0)

    fig = plt.figure(figsize=(20, 6 * n_rows + 2))
    fig.suptitle(
        f"Tree Morphology Feature Analysis — {lbl}\n"
        f"Target: VWC rise [{TARGET_UNIT}] per depth   ({n_events:,} events)\n"
        "Features: crown diameter · height · stem diameter · vitality",
        fontsize=12, fontweight="bold", color=clr, y=1.002,
    )
    gs = mgridspec.GridSpec(n_rows, ncols, figure=fig,
                            hspace=0.55, wspace=0.75)

    any_plotted = False
    pool_sv: dict = {}
    pool_fv: dict = {}
    depth_means_list: list = []

    for row_i, dr in enumerate(depth_results):
        if dr['skip']:
            ax = fig.add_subplot(gs[row_i, 0])
            ax.text(0.5, 0.5, dr['msg'],
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Depth {dr['dlbl']}", fontsize=9)
            continue

        tree_mask = np.array([any(fn.startswith(p) for p in _TREE_PFXS)
                              for fn in dr['feat_names']])
        if not tree_mask.any():
            ax = fig.add_subplot(gs[row_i, 0])
            ax.text(0.5, 0.5, "No tree morphology features in model",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Depth {dr['dlbl']}", fontsize=9)
            continue

        tree_feat = [fn for fn, m in zip(dr['feat_names'], tree_mask) if m]
        tree_imp  = dr['imp_m'][tree_mask]
        tree_std  = dr['imp_s'][tree_mask]

        ax0 = fig.add_subplot(gs[row_i, 0])
        _importance_bar(ax0, tree_feat, tree_imp, tree_std, len(tree_feat),
                        f"Depth {dr['dlbl']} — tree morphology permutation importance",
                        dr['dclr'], dr['r2s'].mean(), dr['r2s'].std(), dr['n_samp'],
                        y_std=None, target_unit=TARGET_UNIT)

        if HAS_SHAP and dr['shap_v'] is not None and ncols == 2:
            tree_shap = dr['shap_v'][:, tree_mask]
            tree_X    = dr['X'][:, tree_mask]
            ax1 = fig.add_subplot(gs[row_i, 1])
            _shap_beeswarm(ax1, tree_feat, tree_X, tree_shap, len(tree_feat),
                           f"Depth {dr['dlbl']}", target_unit=TARGET_UNIT,
                           x_lim=x_lim_shap, y_mean=float(np.nanmean(dr['y'])))
            _dm = float(np.nanmean(dr['y']))
            _scale = max(abs(_dm), 1e-6)
            depth_means_list.append(_dm)
            for fi, fn in enumerate(tree_feat):
                pool_sv.setdefault(fn, []).append(tree_shap[:, fi] / _scale * 100)
                pool_fv.setdefault(fn, []).append(tree_X[:, fi])
        any_plotted = True

    if HAS_SHAP and pool_sv:
        pooled_sv = {n: np.concatenate(arrs) for n, arrs in pool_sv.items()}
        pooled_fv = {n: np.concatenate(arrs) for n, arrs in pool_fv.items()}
        ax_pool = fig.add_subplot(gs[len(DEPTHS), :])
        _draw_pooled_beeswarm(ax_pool, pooled_sv, pooled_fv,
                              "tree morphology", target_unit=TARGET_UNIT,
                              depth_means=depth_means_list)

    if not any_plotted:
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Rain event feature page
# ---------------------------------------------------------------------------
def build_rain_page(ev_type: str, depth_results: list, n_events: int,
                    x_lim_shap=None) -> plt.Figure:
    """
    Dedicated page showing only rain event features (total rain, pre/post event
    rain, peak intensity, StAcc, duration, Huff quartile, rain distribution
    percentages, day-of-year) per depth, plus a pooled all-depths beeswarm at
    the bottom.
    Returns None if no rain event features are present in any depth model.
    """
    clr   = TYPE_COLORS.get(ev_type, "#555")
    lbl   = TYPE_LABELS.get(ev_type, ev_type)
    ncols = 2 if HAS_SHAP else 1
    n_rows = len(DEPTHS) + (1 if HAS_SHAP else 0)

    fig = plt.figure(figsize=(20, 6 * n_rows + 2))
    fig.suptitle(
        f"Rain Event Feature Analysis — {lbl}\n"
        f"Target: VWC rise [{TARGET_UNIT}] per depth   ({n_events:,} events)\n"
        "Features: total rain · pre/post event rain · peak intensity · StAcc · "
        "duration · Huff quartile · rain distribution percentages · day-of-year",
        fontsize=12, fontweight="bold", color=clr, y=1.002,
    )
    gs = mgridspec.GridSpec(n_rows, ncols, figure=fig,
                            hspace=0.55, wspace=0.75)

    any_plotted = False
    pool_sv: dict = {}
    pool_fv: dict = {}
    depth_means_list: list = []

    for row_i, dr in enumerate(depth_results):
        if dr['skip']:
            ax = fig.add_subplot(gs[row_i, 0])
            ax.text(0.5, 0.5, dr['msg'],
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Depth {dr['dlbl']}", fontsize=9)
            continue

        rain_mask = np.array([any(fn.startswith(p) for p in _RAIN_PFXS)
                              for fn in dr['feat_names']])
        if not rain_mask.any():
            ax = fig.add_subplot(gs[row_i, 0])
            ax.text(0.5, 0.5, "No rain event features in model",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Depth {dr['dlbl']}", fontsize=9)
            continue

        rain_feat = [fn for fn, m in zip(dr['feat_names'], rain_mask) if m]
        rain_imp  = dr['imp_m'][rain_mask]
        rain_std  = dr['imp_s'][rain_mask]

        ax0 = fig.add_subplot(gs[row_i, 0])
        _importance_bar(ax0, rain_feat, rain_imp, rain_std, len(rain_feat),
                        f"Depth {dr['dlbl']} — rain event permutation importance",
                        dr['dclr'], dr['r2s'].mean(), dr['r2s'].std(), dr['n_samp'],
                        y_std=None, target_unit=TARGET_UNIT)

        if HAS_SHAP and dr['shap_v'] is not None and ncols == 2:
            rain_shap = dr['shap_v'][:, rain_mask]
            rain_X    = dr['X'][:, rain_mask]
            ax1 = fig.add_subplot(gs[row_i, 1])
            _shap_beeswarm(ax1, rain_feat, rain_X, rain_shap, len(rain_feat),
                           f"Depth {dr['dlbl']}", target_unit=TARGET_UNIT,
                           x_lim=x_lim_shap, allow_area=True,
                           y_mean=float(np.nanmean(dr['y'])))
            _dm = float(np.nanmean(dr['y']))
            _scale = max(abs(_dm), 1e-6)
            depth_means_list.append(_dm)
            for fi, fn in enumerate(rain_feat):
                pool_sv.setdefault(fn, []).append(rain_shap[:, fi] / _scale * 100)
                pool_fv.setdefault(fn, []).append(rain_X[:, fi])
        any_plotted = True

    if HAS_SHAP and pool_sv:
        pooled_sv = {n: np.concatenate(arrs) for n, arrs in pool_sv.items()}
        pooled_fv = {n: np.concatenate(arrs) for n, arrs in pool_fv.items()}
        ax_pool = fig.add_subplot(gs[len(DEPTHS), :])
        _draw_pooled_beeswarm(ax_pool, pooled_sv, pooled_fv,
                              "rain event", target_unit=TARGET_UNIT,
                              depth_means=depth_means_list)

    if not any_plotted:
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# Per-event-type page
# ---------------------------------------------------------------------------
def train_event_group(ev_type: str, ev_sub: pd.DataFrame):
    """Train one model per depth.  Returns (results_rows, depth_data, depth_results)."""
    results_rows  = []
    depth_data    = []   # (feat_names, X, shap_v, y) — for build_shap_combined_page
    depth_results = []   # one dict per DEPTH (skip or full model data)

    for depth in DEPTHS:
        target = f"delta_pct_dyn{depth}"
        dlbl   = DEPTH_LBLS[depth]
        dclr   = DEPTH_CLRS[depth]

        if target not in ev_sub.columns:
            depth_results.append({'skip': True,
                                  'msg': f"Column {target} not found", 'dlbl': dlbl})
            continue

        X, y, feat_names, n_samp = prepare_features(ev_sub, target)
        _min_n = 10 if ev_type in {"IRRIGATION", "IRRIGATION_STRONG"} else MIN_SAMPLES
        if n_samp < _min_n or len(feat_names) < 3:
            depth_results.append({'skip': True,
                                  'msg': f"Insufficient data  (n={n_samp})", 'dlbl': dlbl})
            continue

        print(f"    {ev_type} | {dlbl}: n={n_samp:,}  features={len(feat_names)}")
        print(f"      y: min={y.min():.2f}  max={y.max():.2f}  "
              f"mean={y.mean():.2f}  std={y.std():.2f}")
        rf, r2s, imp_m, imp_s, shap_v, X_shap, r2_train, y_pred, y_shap = train_and_evaluate(X, y, n_samp)
        depth_data.append((feat_names, X_shap, shap_v, y))
        top3 = np.argsort(imp_m)[::-1][:3]
        print("      Top-3: " + ", ".join(
            f"{feat_names[i]} ({imp_m[i]:.4f})" for i in top3))

        for rank, idx in enumerate(np.argsort(imp_m)[::-1][:N_TOP_FEATURES], 1):
            results_rows.append({
                "event_type": ev_type, "depth": dlbl, "rank": rank,
                "feature": feat_names[idx],
                "importance": round(imp_m[idx], 6),
                "importance_std": round(imp_s[idx], 6),
                "cv_r2_mean": round(float(r2s.mean()), 4),
                "cv_r2_std":  round(float(r2s.std()), 4),
                "n_samples": n_samp,
            })

        depth_results.append({
            'skip': False, 'dlbl': dlbl, 'dclr': dclr,
            'feat_names': feat_names, 'X': X_shap, 'y': y, 'shap_v': shap_v,
            'imp_m': imp_m, 'imp_s': imp_s, 'r2s': r2s, 'n_samp': n_samp,
            'r2_train': r2_train, 'y_pred': y_pred, 'y_shap': y_shap,
        })

    return results_rows, depth_data, depth_results


_orig_train_event_group = train_event_group


def train_event_group(ev_type, ev_sub):
    """v7 wrapper: sets current event-type for Boruta lookup."""
    global _boruta_current_et
    _boruta_current_et = ev_type
    return _orig_train_event_group(ev_type, ev_sub)


def build_event_page(ev_type: str, depth_results: list, n_events: int,
                     x_lim_shap=None) -> plt.Figure:
    clr   = TYPE_COLORS.get(ev_type, "#555")
    lbl   = TYPE_LABELS.get(ev_type, ev_type)
    ncols = 2 if HAS_SHAP else 1

    fig = plt.figure(figsize=(20, 7 * ncols * len(DEPTHS) // 2 + 4))
    fig.suptitle(
        f"Random Forest Feature Importance — {lbl}\n"
        f"Target: ΔVWC as % of dynamic range at each depth   ({n_events:,} events)",
        fontsize=13, fontweight="bold", color=clr, y=1.002,
    )
    gs = mgridspec.GridSpec(len(DEPTHS), ncols, figure=fig,
                            hspace=0.55, wspace=0.75)

    for row_i, dr in enumerate(depth_results):
        if dr['skip']:
            ax = fig.add_subplot(gs[row_i, 0])
            ax.text(0.5, 0.5, dr['msg'],
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Depth {dr['dlbl']}", fontsize=9)
            continue

        ax0 = fig.add_subplot(gs[row_i, 0])
        _importance_bar(ax0, dr['feat_names'], dr['imp_m'], dr['imp_s'], N_TOP_FEATURES,
                        f"Depth {dr['dlbl']} — permutation importance",
                        dr['dclr'], dr['r2s'].mean(), dr['r2s'].std(), dr['n_samp'],
                        y_std=dr['y'].std(), target_unit=TARGET_UNIT)

        if HAS_SHAP and dr['shap_v'] is not None and ncols == 2:
            ax1 = fig.add_subplot(gs[row_i, 1])
            _shap_beeswarm(ax1, dr['feat_names'], dr['X'], dr['shap_v'], N_TOP_FEATURES,
                           f"Depth {dr['dlbl']}", target_unit=TARGET_UNIT,
                           x_lim=x_lim_shap, y_mean=float(np.nanmean(dr['y'])))

    return fig


# ---------------------------------------------------------------------------
# Summary page
# ---------------------------------------------------------------------------
def build_summary_page(results_df: pd.DataFrame) -> plt.Figure:
    et_with_data = [et for et in EVENT_TYPES
                    if not results_df[results_df["event_type"] == et].empty]
    ncols = min(len(et_with_data), 3)
    nrows = (len(et_with_data) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 7 * nrows),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.45},
                             squeeze=False)
    fig.suptitle(
        "Feature Importance Summary — mean permutation importance across depths",
        fontsize=12, fontweight="bold",
    )
    for ax, et in zip(axes.flat, et_with_data):
        sub = results_df[results_df["event_type"] == et]
        feat_imp = (sub.groupby("feature")["importance"]
                    .mean().sort_values(ascending=False).head(N_TOP_FEATURES))
        clr = TYPE_COLORS.get(et, "#555")
        y   = np.arange(len(feat_imp))
        ax.barh(y, feat_imp.values[::-1], color=clr, alpha=0.80, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f.replace("tree_", "") for f in feat_imp.index[::-1]], fontsize=7.5)
        ax.axvline(0, color="gray", lw=0.7, ls="--")
        ax.set_xlabel("Mean permutation importance (avg across depths)", fontsize=8)
        avg_r2 = sub["cv_r2_mean"].mean()
        ax.set_title(f"{TYPE_LABELS.get(et, et)}\navg R² = {avg_r2:.3f}",
                     fontsize=9, fontweight="bold", color=clr)
        ax.grid(axis="x", alpha=0.3)
    for ax in list(axes.flat)[len(et_with_data):]:
        ax.set_visible(False)
    return fig


# ---------------------------------------------------------------------------
# Scatter page
# ---------------------------------------------------------------------------
def build_scatter_page(ev_type: str, ev_sub: pd.DataFrame,
                       top_features: list) -> plt.Figure:
    target = "delta_pct_dyn-10"
    if target not in ev_sub.columns:
        return None
    ev_plot = ev_sub.copy()
    clr  = TYPE_COLORS.get(ev_type, "#555")
    top6 = [f for f in top_features[:6] if f in ev_plot.columns]
    if not top6:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             gridspec_kw={"hspace": 0.50, "wspace": 0.38})
    fig.suptitle(
        f"Top feature scatter plots — {TYPE_LABELS.get(ev_type, ev_type)}\n"
        f"Target: ΔVWC at −10 cm (% dyn. range)   (n = {len(ev_plot):,})",
        fontsize=11, fontweight="bold", color=clr,
    )
    for ax, feat in zip(axes.flat, top6 + [""] * (6 - len(top6))):
        if not feat:
            ax.set_visible(False); continue
        sub = ev_plot[[feat, target]].dropna()
        ax.scatter(sub[feat], sub[target], alpha=0.35, s=10, color=clr, edgecolors="none",
                   rasterized=True)
        if len(sub) >= 5 and sub[feat].std() > 0:
            from scipy import stats as _stats
            sl, ic, r, p, _ = _stats.linregress(sub[feat], sub[target])
            xl = np.array([sub[feat].min(), sub[feat].max()])
            ax.plot(xl, sl * xl + ic, "k--", lw=1.3, alpha=0.7)
            ax.set_title(f"{feat.replace('tree_','')}\nr={r:.2f}  p={p:.3g}",
                         fontsize=8, fontweight="bold")
        else:
            ax.set_title(feat.replace("tree_", ""), fontsize=8)
        ax.set_xlabel(feat.replace("tree_", ""), fontsize=7.5)
        ax.set_ylabel("ΔVWC at −10 cm (% dyn. range)", fontsize=7.5)
        ax.grid(alpha=0.25)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  SOIL MOISTURE RF  —  v7  (post-event rain removed + v2 CSV fix)")
    print("  CHANGE 1: rain_post24h_mm, rain_post72h_mm removed from features")
    print("  Target: log1p(delta_pct_dyn)  — log-transformed VWC rise")
    print("  STANDALONE: all dependencies inlined")
    print("  Boruta feature filtering: ON" if _BORUTA_SELECTED else
          "  Boruta feature filtering: OFF (pkl missing)")
    print("=" * 65)

    print("\n[1/4] Loading data ...")
    groups = load_event_groups()
    if not groups:
        print("  ERROR: no event groups loaded — exiting.")
        return

    for et, df in groups.items():
        resp_counts = {d: int(df[f"delta_pct_dyn{d}"].notna().sum())
                       for d in DEPTHS if f"delta_pct_dyn{d}" in df.columns}
        cnt_str = "  ".join(f"{DEPTH_LBLS[d]}: {n}" for d, n in resp_counts.items())
        print(f"  {et}: {len(df):,} events  |  valid delta_pct_dyn → {cnt_str}")

    print("\n[2/4] Training models ...")
    all_results = []

    with PdfPages(OUT_PDF) as pdf:
        # Cover page
        fig_c, ax_c = plt.subplots(figsize=(14, 8))
        ax_c.axis("off")
        txt = (
            "Random Forest Feature Importance Analysis\n"
            "Soil Moisture Response Drivers\n\n"
            f"Event groups: {', '.join(EVENT_TYPES)}\n\n"
            "BSC_XX11_Q12 selection:\n"
            "  • BSC bits 3-4 = '11'  (cumulative curve HIGH in Q3 & Q4\n"
            "    → most rain already fell before the 3rd quartile begins)\n"
            "  • Huff quartile 1 or 2  (peak intensity in 1st half of event)\n"
            "  → Front-loaded impulse: VWC peak is clearly attributable to\n"
            "    the initial rainfall burst, not to a sustained late input\n\n"
            f"Depths: −10 cm / −30 cm / −45 cm\n"
            f"Target:  delta_pct_dyn-{{depth}} — VWC rise as % of sensor dyn. range\n"
            f"Model:   RandomForestRegressor ({RF_ESTIMATORS} trees, {CV_FOLDS}-fold CV)\n"
            f"Importance: permutation importance ({PERM_REPEATS} repeats, signed)\n"
            f"SHAP: {'signed beeswarm  (shap available)' if HAS_SHAP else 'not available  (pip install shap)'}\n\n"
            "Leakage prevention:\n"
            "  Excluded: peak_vwc, t_peak_min, delta_vwc at all depths\n"
            "  Retained: pre_vwc, dyn_range, dry_h for the target depth only\n"
            "  Excluded: vwc_p05/p95 (redundant with dyn_range = p95 − p05)\n"
        )
        ax_c.text(0.06, 0.94, txt, transform=ax_c.transAxes, fontsize=10.5,
                  va="top", family="monospace",
                  bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f4f8",
                            edgecolor="#aaa", alpha=0.9))
        ax_c.set_title("Random Forest Feature Importance — Soil Moisture Response",
                       fontsize=16, fontweight="bold", pad=20)
        pdf.savefig(fig_c, bbox_inches="tight", dpi=150); plt.close(fig_c)

        type_top = {}
        # ── Pass 1: train all models, collect SHAP values ──
        _trained  = {}
        _shap_all = []
        for et in EVENT_TYPES:
            if et not in groups:
                print(f"\n  ── {et}: no data, skipped ──")
                continue
            print(f"\n  ── {et} (training) ──")
            rows, depth_data, depth_results = train_event_group(et, groups[et])
            _trained[et] = (rows, depth_data, depth_results)
            all_results.extend(rows)
            for _, _, _sv, _ in depth_data:
                if _sv is not None:
                    _shap_all.append(_sv.ravel())

        # Save training cache so replot_rf_soil_moisture.py can regenerate the PDF
        _cache = {"trained": _trained, "all_results": all_results}
        with open(OUT_CACHE, "wb") as _f:
            pickle.dump(_cache, _f)
        print(f"  Training cache saved → {OUT_CACHE}")

        # ── Pass 2: plot all event pages — each subplot uses its own 95 % x-limit ──
        for et in EVENT_TYPES:
            if et not in _trained:
                continue
            rows, depth_data, depth_results = _trained[et]
            fig = build_event_page(et, depth_results, len(groups[et]),
                                   x_lim_shap=None)
            pdf.savefig(fig, bbox_inches="tight", dpi=150); plt.close(fig)
            if HAS_SHAP and depth_data:
                fig_comb = build_shap_combined_page(et, depth_data, x_lim=None)
                if fig_comb is not None:
                    pdf.savefig(fig_comb, bbox_inches="tight", dpi=150)
                    plt.close(fig_comb)
            fig_lu = build_landuse_page(et, depth_results, len(groups[et]),
                                        x_lim_shap=None)
            if fig_lu is not None:
                pdf.savefig(fig_lu, bbox_inches="tight", dpi=150)
                plt.close(fig_lu)
            fig_tp = build_topo_page(et, depth_results, len(groups[et]),
                                     x_lim_shap=None)
            if fig_tp is not None:
                pdf.savefig(fig_tp, bbox_inches="tight", dpi=150)
                plt.close(fig_tp)
            fig_tr = build_tree_page(et, depth_results, len(groups[et]),
                                     x_lim_shap=None)
            if fig_tr is not None:
                pdf.savefig(fig_tr, bbox_inches="tight", dpi=150); plt.close(fig_tr)
            fig_rn = build_rain_page(et, depth_results, len(groups[et]),
                                     x_lim_shap=None)
            if fig_rn is not None:
                pdf.savefig(fig_rn, bbox_inches="tight", dpi=150); plt.close(fig_rn)
            if rows:
                df_r = pd.DataFrame(rows)
                type_top[et] = (df_r.groupby("feature")["importance"]
                                .mean().sort_values(ascending=False).index.tolist())

        results_df = pd.DataFrame(all_results)

        print("\n[3/4] Summary and scatter pages ...")
        if not results_df.empty:
            pdf.savefig(build_summary_page(results_df), bbox_inches="tight", dpi=150)
            plt.close("all")
            for et in EVENT_TYPES:
                if et not in groups or et not in type_top:
                    continue
                fig_sc = build_scatter_page(et, groups[et], type_top[et])
                if fig_sc is not None:
                    pdf.savefig(fig_sc, bbox_inches="tight", dpi=150); plt.close(fig_sc)

    print("\n[4/4] Saving results CSV ...")
    if not results_df.empty:
        results_df.to_csv(OUT_CSV, index=False)
        print(f"  Saved → {OUT_CSV}")
    else:
        print("  No results to save.")

    print(f"\nReport → {OUT_PDF}")
    if not HAS_SHAP:
        print("  (pip install shap  to enable beeswarm plots)")
    print("\n" + "=" * 65 + "\n  DONE\n" + "=" * 65)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
