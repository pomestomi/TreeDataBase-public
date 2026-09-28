#!/usr/bin/env python3
"""
Generate the manuscript figures (Copernicus GI).

Usage
-----
  python make_paper_figures.py                          # all figures
  python make_paper_figures.py --only fig_corr_heatmap
  python make_paper_figures.py --data-date 2026-08-13
  python make_paper_figures.py --seed 123

Output: figures/paper/{name}.pdf + {name}.png (300 dpi)

Figures generated
-----------------
  fig_event_scatter        log-log total_mm vs duration_h, coloured by Huff Q
  fig_event_shapes         BSC 16-pattern bar + Huff Q frequency
  fig_corr_heatmap         Spearman rho lower-triangle heatmap (SM subset)
  fig_univariate_increase  univariate scatter: SM features vs delta_pct_dyn-30
  fig_univariate_drying    univariate scatter: DT features vs dry_h-30
  fig_feature_extraction   example VWC time-series event with annotations
  fig_landuse_scheme       land-use polygon overlay for CityErlangen
  fig_site_examples        SKIPPED -- requires field photographs
"""

import argparse
import datetime
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
})
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
FIG_DIR     = SCRIPT_DIR / "paper"
DATA_DIR    = REPO_ROOT / "TreeTabularData"
RF_CSV      = DATA_DIR / "rf_dataset.csv"
TREES_DIR   = DATA_DIR / "trees"
GIS_DIR     = REPO_ROOT / "GISData"
ERLANGEN_VL = GIS_DIR / "CityErlangen" / "VectorLayers"

FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Layout constants (Copernicus GI column widths)
# ---------------------------------------------------------------------------
SINGLE_COL_IN = 3.5    # 88 mm
DOUBLE_COL_IN = 7.28   # 185 mm
DPI           = 300

# ---------------------------------------------------------------------------
# Crameri colour maps
# ---------------------------------------------------------------------------
try:
    import cmcrameri.cm as cmc
    _HAS_CRAMERI = True
except ImportError:
    _HAS_CRAMERI = False
    print("WARNING: cmcrameri not installed -- falling back to matplotlib defaults",
          file=sys.stderr)


def _cmap_batlow():
    return cmc.batlow if _HAS_CRAMERI else plt.cm.viridis


def _cmap_batlowS():
    return cmc.batlowS if _HAS_CRAMERI else plt.cm.tab10


def _cmap_vik():
    return cmc.vik if _HAS_CRAMERI else plt.cm.RdBu_r


# Fixed depth colours (kept consistent across all figures)
DEPTH_CLRS = {"-10": "#1b7837", "-30": "#762a83", "-45": "#e08214"}


def _hq_colours():
    """Four colours for Huff quartiles, sampled from batlowS."""
    cm  = _cmap_batlowS()
    pts = np.linspace(0.05, 0.95, 4)
    return {i + 1: cm(pts[i]) for i in range(4)}


def _lu_colours():
    """Land-use category colours from batlowS."""
    cm  = _cmap_batlowS()
    pts = np.linspace(0.05, 0.95, 4)
    return {
        "green_att": cm(pts[0]),
        "green_det": cm(pts[1]),
        "building":  cm(pts[2]),
        "sealed":    cm(pts[3]),
    }


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "pdf.fonttype":   42,   # embed fonts in PDF (required by most journals)
    "ps.fonttype":    42,
    "font.family":    "Arial",
    "font.weight":    "bold",
    "font.size":       8,
    "axes.titlesize":  8,
    "axes.labelsize":  8,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":     150,
})

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _stamp(ax, text, *, loc="lower right", fontsize=6, alpha=0.5):
    kw = dict(transform=ax.transAxes, fontsize=fontsize,
              color="gray", alpha=alpha, va="bottom")
    if "right" in loc:
        ax.text(0.99, 0.01, text, ha="right", **kw)
    else:
        ax.text(0.01, 0.01, text, ha="left", **kw)


def _save(name, fig):
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    with PdfPages(pdf) as pp:
        pp.savefig(fig, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=DPI)
    print(f"  -> {pdf.name}  +  {png.name}")
    plt.close(fig)


def _load_rf() -> pd.DataFrame:
    if not RF_CSV.exists():
        sys.exit(f"ERROR: rf_dataset.csv not found at {RF_CSV}")
    df = pd.read_csv(RF_CSV, low_memory=False)
    df["bsc_str"] = df["bsc"].astype(str).str.zfill(4)
    return df


def _load_sensor_ts(eui):
    p = TREES_DIR / str(eui) / "sensor_data.csv"
    if not p.exists():
        return None
    ts = pd.read_csv(p, parse_dates=["datetime"], low_memory=False)
    return ts.sort_values("datetime").set_index("datetime")


def _subset_sm(df: pd.DataFrame, depth="-30") -> pd.DataFrame:
    """BSC XX11 + Huff Q1/Q2 -- SM model events."""
    tgt  = f"delta_pct_dyn{depth}"
    mask = (
        (df["bsc_str"].str[2:] == "11") &
        (df["huff_q"].isin([1, 2])) &
        df[tgt].notna() &
        (df[tgt] > 0)
    )
    return df[mask].copy()


def _subset_dt(df: pd.DataFrame, depth="-30") -> pd.DataFrame:
    """BSC 00XX + Huff Q3/Q4 -- DT model events."""
    tgt  = f"dry_h{depth}"
    mask = (
        (df["bsc_str"].str[:2] == "00") &
        (df["huff_q"].isin([3, 4])) &
        df[tgt].notna() &
        (df[tgt] > 0)
    )
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Figure: fig_event_scatter
# ---------------------------------------------------------------------------

def fig_event_scatter(df: pd.DataFrame, data_date: str, seed: int):
    """Log-log scatter: total_mm vs duration_h, coloured by Huff Q."""
    sub = df.dropna(subset=["total_mm", "duration_h", "huff_q"]).copy()
    sub = sub[(sub["total_mm"] > 0) & (sub["duration_h"] > 0)]
    print(f"  n events = {len(sub)}")
    HQ_CLRS = _hq_colours()
    for q in [1, 2, 3, 4]:
        n = (sub["huff_q"] == q).sum()
        print(f"    Huff Q{q}: {n} ({100*n/len(sub):.1f}%)")

    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, SINGLE_COL_IN * 0.9))
    for q in [1, 2, 3, 4]:
        s = sub[sub["huff_q"] == q]
        ax.scatter(s["total_mm"], s["duration_h"],
                   c=[HQ_CLRS[q]], s=5, alpha=0.5, linewidths=0,
                   label=f"Q{q} (n={len(s)})", zorder=2 + q)

    for thresh, ls, lbl in [(5, "--", "5 mm"), (12.7, ":", "12.7 mm")]:
        ax.axvline(thresh, color="gray", lw=0.7, ls=ls, zorder=1, label=lbl)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total rainfall (mm)")
    ax.set_ylabel("Duration (h)")
    ax.legend(fontsize=6, markerscale=2, ncol=2, loc="upper left")
    _stamp(ax, f"data: {data_date}")
    fig.tight_layout()
    _save("fig_event_scatter", fig)


# ---------------------------------------------------------------------------
# Figure: fig_event_shapes
# ---------------------------------------------------------------------------

def fig_event_shapes(df: pd.DataFrame, data_date: str, seed: int):
    """BSC 16-pattern bar chart + Huff Q frequency."""
    sub = df.dropna(subset=["bsc_str", "huff_q"]).copy()
    print(f"  n events = {len(sub)}")

    all_patterns = [f"{i:04b}" for i in range(16)]
    counts = sub["bsc_str"].value_counts().reindex(all_patterns, fill_value=0)

    colours = []
    for p in all_patterns:
        if p[2:] == "11":
            colours.append("#762a83")   # SM model
        elif p[:2] == "00":
            colours.append("#1b7837")   # DT model
        else:
            colours.append("#aaaaaa")

    HQ_CLRS = _hq_colours()
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COL_IN, 3.5),
                              gridspec_kw={"height_ratios": [2.5, 1]})

    ax = axes[0]
    ax.bar(all_patterns, counts.values, color=colours, width=0.7, zorder=2)
    ax.set_xlabel("BSC pattern (b1 b2 b3 b4)")
    ax.set_ylabel("Event count")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    ax.legend(handles=[
        mpatches.Patch(color="#762a83", label="XX11 → SM model"),
        mpatches.Patch(color="#1b7837", label="00XX → DT model"),
        mpatches.Patch(color="#aaaaaa", label="other"),
    ], fontsize=6, loc="upper right")
    _stamp(ax, f"data: {data_date}")

    ax2 = axes[1]
    qcounts = sub["huff_q"].value_counts().sort_index()
    bars    = ax2.bar([str(q) for q in qcounts.index], qcounts.values,
                      color=[HQ_CLRS[q] for q in qcounts.index],
                      width=0.6, zorder=2)
    ax2.set_xlabel("Huff quartile")
    ax2.set_ylabel("Count")
    ax2.grid(axis="y", lw=0.4, alpha=0.4)
    for bar, val in zip(bars, qcounts.values):
        pct = 100 * val / qcounts.sum()
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{pct:.0f}%", ha="center", va="bottom", fontsize=6)

    fig.tight_layout()
    _save("fig_event_shapes", fig)


# ---------------------------------------------------------------------------
# Figure: fig_corr_heatmap
# ---------------------------------------------------------------------------

def fig_corr_heatmap(df: pd.DataFrame, data_date: str, seed: int):
    """Spearman rho lower-triangle heatmap – all trees, static predictors only."""
    import matplotlib.font_manager as fm

    # ── Load full tree attribute table (all 554 trees) ────────────────────────
    atl_path = DATA_DIR / "all_tree_locations.csv"
    if not atl_path.exists():
        print("  [SKIP] all_tree_locations.csv not found")
        return
    sub = pd.read_csv(atl_path, low_memory=False)

    # ── Join median dyn_range-10 from rf_dataset (510 trees with rain events) ─
    dyn = (df.groupby("eui", as_index=False)["dyn_range-10"]
             .median()
             .rename(columns={"eui": "devEUI"}))
    sub = sub.merge(dyn, on="devEUI", how="left")

    # ── Aggregate ring-sum columns ────────────────────────────────────────────
    sub["greenAtta_0to7"] = sub[["greenAtta2", "greenAtta5", "greenAtta7"]].sum(axis=1, min_count=1)
    sub["greenDeta_0to7"] = sub[["greenDeta2", "greenDeta5", "greenDeta7"]].sum(axis=1, min_count=1)
    sub["buildings_0to7"] = sub[["buildings2", "buildings5", "buildings7"]].sum(axis=1, min_count=1)
    sub["sealedSu_0to7"]  = sub[["sealedSu2",  "sealedSu5",  "sealedSu7" ]].sum(axis=1, min_count=1)

    # ── Tree age ──────────────────────────────────────────────────────────────
    ref = pd.Timestamp("2024-06-01")
    for dcol in ["plantDate", "germDate"]:
        if dcol in sub.columns:
            d = pd.to_datetime(sub[dcol], errors="coerce")
            a = (ref - d).dt.days / 365.25
            if "age_years" not in sub.columns:
                sub["age_years"] = a
            else:
                sub["age_years"] = sub["age_years"].fillna(a)

    n_trees = len(sub)

    # ── Feature list ──────────────────────────────────────────────────────────
    # Labels follow the manuscript predictor table / SHAP figure naming.
    FEAT_MAP = [
        ("greenAtta 2.5 m",   "greenAtta2"),
        ("greenAtta 5 m",     "greenAtta5"),
        ("greenAtta 7.5 m",   "greenAtta7"),
        ("greenAtta 0-7.5 m", "greenAtta_0to7"),
        ("greenDeta 2.5 m",   "greenDeta2"),
        ("greenDeta 5 m",     "greenDeta5"),
        ("greenDeta 7.5 m",   "greenDeta7"),
        ("greenDeta 0-7.5 m", "greenDeta_0to7"),
        ("Buildings 2.5 m",   "buildings2"),
        ("Buildings 5 m",     "buildings5"),
        ("Buildings 7.5 m",   "buildings7"),
        ("Buildings 0-7.5 m", "buildings_0to7"),
        ("Sealed 2.5 m",      "sealedSu2"),
        ("Sealed 5 m",        "sealedSu5"),
        ("Sealed 7.5 m",      "sealedSu7"),
        ("Sealed 0-7.5 m",    "sealedSu_0to7"),
        ("TWI",               "twi"),
        ("TPI 2.5 m",         "tpi2m5"),
        ("TPI 5 m",           "tpi5m"),
        ("TPI 7.5 m",         "tpi7m5"),
        ("TPI crown diam.",   "tpiCD"),
        ("Slope",             "slope"),
        ("Elevation",         "elevation"),
        ("Flow acc.",         "flotAcc"),
        ("Sky view factor",   "SVF"),
        ("Depth to GW",       "DTGW"),
        ("Crown diam.",       "crownDiam"),
        ("Tree height",       "height"),
        ("Stem diam.",        "stemDiam"),
        ("Tree age",          "age_years"),
        ("Dyn. range",        "dyn_range-10"),
    ]
    FEAT_MAP = [(lbl, col) for lbl, col in FEAT_MAP if col in sub.columns]
    labels   = [lbl for lbl, _ in FEAT_MAP]
    cols     = [col for _, col in FEAT_MAP]
    n_feat   = len(cols)
    print(f"  n features = {n_feat},  n trees = {n_trees}")

    # ── Pairwise Spearman ─────────────────────────────────────────────────────
    rho_mat = np.full((n_feat, n_feat), np.nan)
    np.fill_diagonal(rho_mat, 1.0)
    for i in range(n_feat):
        for j in range(i):
            pair = sub[[cols[i], cols[j]]].dropna()
            if len(pair) >= 10:
                r, _ = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                rho_mat[i, j] = r
                rho_mat[j, i] = r
    tri = rho_mat[np.tril_indices(n_feat, k=-1)]
    print(f"  max |rho| in lower triangle = {np.nanmax(np.abs(tri[~np.isnan(tri)])):.3f}")

    # ── Font: Arial 8 pt bold ─────────────────────────────────────────────────
    fp = fm.FontProperties(family="Arial", size=8, weight="bold")

    # ── Plot ──────────────────────────────────────────────────────────────────
    W_IN     = 130 / 25.4
    mask     = np.triu(np.ones_like(rho_mat, dtype=bool), k=1)
    rho_plot = np.where(mask, np.nan, rho_mat)

    # Colorbar flush to top; bottom clear of the diagonal including tick-label
    # text, which extends ~0.05 axes-fraction to the left of the bar.
    cbar_x      = 0.94
    cbar_w      = 0.025
    label_reach = 0.05                              # width of "-1.00" at 8 pt
    diag_y      = 1.0 - (cbar_x - label_reach)     # diagonal y at label left edge
    cbar_bottom = diag_y + 2.0 / n_feat            # two-cell margin above diagonal
    cbar_top    = 1.0 - 0.5 / n_feat               # half-cell from axes top edge
    cbar_h      = cbar_top - cbar_bottom

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    fig, ax = plt.subplots(figsize=(W_IN, W_IN))
    im   = ax.imshow(rho_plot, cmap=_cmap_vik(), norm=norm, aspect="equal")

    cax  = ax.inset_axes([cbar_x, cbar_bottom, cbar_w, cbar_h])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.set_label("Spearman ρ", labelpad=6)
    cbar.ax.yaxis.label.set_font_properties(fp)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_font_properties(fp)

    ax.set_xticks(range(n_feat))
    ax.set_yticks(range(n_feat))
    ax.set_xticklabels(labels, rotation=90, fontproperties=fp)
    ax.set_yticklabels(labels, fontproperties=fp)

    # Centred title inside the empty upper triangle
    ax.text(0.50, 0.99, f"n = {n_trees} trees",
            transform=ax.transAxes, ha="center", va="top",
            fontproperties=fp, color="black")

    fig.tight_layout()
    _save("fig_corr_heatmap", fig)


# ---------------------------------------------------------------------------
# Shared: 2x3 univariate grid
# ---------------------------------------------------------------------------

def _univariate_panel(sub: pd.DataFrame, tgt: str, feat_list: list,
                      xlabels: list, fig_name: str, data_date: str,
                      ylabel: str = None, log_y_feats: set = None):
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(DOUBLE_COL_IN, DOUBLE_COL_IN * 0.62))
    if log_y_feats is None:
        log_y_feats = set()

    print(f"\n  {'Feature':<28} {'n':>5}  {'slope':>9}  {'R²':>6}  {'ρ':>7}  {'p':>8}")
    print(f"  {'-'*67}")

    for ax, feat, xlabel in zip(axes.flat, feat_list, xlabels):
        tmp       = sub[[feat, tgt]].copy()
        tmp[feat] = pd.to_numeric(tmp[feat], errors="coerce")
        tmp[tgt]  = pd.to_numeric(tmp[tgt],  errors="coerce")
        valid     = tmp.dropna()
        use_logy  = feat in log_y_feats
        if use_logy:
            valid = valid[valid[tgt] > 0]
        x, y      = valid[feat].values, valid[tgt].values

        rho, pval = stats.spearmanr(x, y)
        r2, slope, intercept = 0.0, 0.0, 0.0
        if len(x) > 3:
            # fit in log(y) space when log y-axis is requested
            y_fit = np.log(y) if use_logy else y
            slope, intercept, r_lin, _, _ = stats.linregress(x, y_fit)
            r2    = r_lin ** 2
            xline = np.linspace(x.min(), x.max(), 100)
            yline = np.exp(slope * xline + intercept) if use_logy else slope * xline + intercept
            ax.plot(xline, yline, color="crimson", lw=1.2, zorder=3)

        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        print(f"  {xlabel:<28} {len(x):>5}  {slope:>+9.4f}  {r2:>6.3f}  {rho:>+7.3f}  {pval:>8.4f} {sig}")

        ax.scatter(x, y, s=4, alpha=0.45, linewidths=0,
                   c=x, cmap=_cmap_batlow(), zorder=2)
        if use_logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=7, fontweight="bold")
        ax.set_ylabel(ylabel or tgt, fontsize=7, fontweight="bold")
        ax.set_title(
            f"slope={slope:+.3f}   $R^2$={r2:.3f}   ρ={rho:+.2f}{sig}   n={len(x)}",
            fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.grid(lw=0.3, alpha=0.35)

    for ax in list(axes.flat)[len(feat_list):]:
        ax.set_visible(False)

    fig.tight_layout()
    _save(fig_name, fig)


# ---------------------------------------------------------------------------
# Figure: fig_univariate_increase
# ---------------------------------------------------------------------------

def fig_univariate_increase(df: pd.DataFrame, data_date: str, seed: int):
    """Univariate scatter: top-6 site characteristics vs delta_pct_dyn-30."""
    sub  = _subset_sm(df, depth="-30")
    tgt  = "delta_pct_dyn-30"
    feats  = ["pF3.0",   "pF4.2",   "pH",
               "kSoluble", "stemDiam_m", "depthToGroundwater_m"]
    labels = ["pF 3.0 (vol.%)", "pF 4.2 (vol.%)", "Soil pH",
               "K soluble (mg/l)", "Stem diam. (m)", "Depth to GW (m)"]
    feats  = [f for f in feats if f in sub.columns]
    labels = labels[:len(feats)]
    _univariate_panel(sub, tgt, feats, labels,
                      "fig_univariate_increase", data_date,
                      ylabel="ΔVWC_norm (%, 60 cm)",
                      log_y_feats={"stemDiam_m"})


# ---------------------------------------------------------------------------
# Figure: fig_univariate_drying
# ---------------------------------------------------------------------------

def fig_univariate_drying(df: pd.DataFrame, data_date: str, seed: int):
    """Univariate scatter: top-6 site characteristics vs dry_h-30."""
    sub  = _subset_dt(df, depth="-30")
    tgt  = "dry_h-30"
    feats  = ["part5Perc", "part1Perc",    "part7Perc",
               "pH",        "elevation",    "greenAtta_2m5"]
    labels = ["Particle fr. 5 (%)", "Particle fr. 1 (%)", "Particle fr. 7 (%)",
               "Soil pH",            "Elevation (m)",       "Green att. 2.5 m (m²)"]
    feats  = [f for f in feats if f in sub.columns]
    labels = labels[:len(feats)]
    _univariate_panel(sub, tgt, feats, labels,
                      "fig_univariate_drying", data_date,
                      ylabel="t_dry (h, 60 cm)")


# ---------------------------------------------------------------------------
# Figure: fig_feature_extraction
# ---------------------------------------------------------------------------

def fig_feature_extraction(df: pd.DataFrame, data_date: str, seed: int):
    """
    VWC time-series for the SM event with the largest delta_pct_dyn-30,
    with rain bars, pre/peak annotations, and event-span shading.
    """
    sub = _subset_sm(df, depth="-30").dropna(subset=["ev_start", "ev_end", "eui"])
    if sub.empty:
        print("  SKIPPING -- no valid SM events")
        return

    row      = sub.loc[sub["delta_pct_dyn-30"].idxmax()]
    eui      = row["eui"]
    ev_start = pd.Timestamp(row["ev_start"])
    ev_end   = pd.Timestamp(row["ev_end"])
    print(f"  eui={eui}  event {ev_start} -> {ev_end}")
    print(f"  delta_pct_dyn-30 = {row['delta_pct_dyn-30']:.2f}%")

    ts = _load_sensor_ts(eui)
    if ts is None:
        print(f"  SKIPPING -- sensor_data.csv not found for eui={eui}")
        return

    t0     = ev_start - pd.Timedelta("24h")
    t1     = ev_end   + pd.Timedelta("72h")
    ts_win = ts.loc[t0:t1]
    if ts_win.empty:
        print("  SKIPPING -- no sensor data in window")
        return

    rain_col = "MTB|ENV__ATMO__RAIN__DELTA"
    depths   = ["-10", "-30", "-45"]
    vwc_cols = [f"{d}|ENV__SOIL__VWC" for d in depths]

    fig, (ax_rain, ax_vwc) = plt.subplots(
        2, 1, figsize=(DOUBLE_COL_IN, 2.9), sharex=True,
        gridspec_kw={"height_ratios": [0.6, 2]}
    )

    if rain_col in ts_win.columns:
        width_days = pd.Timedelta("15min") / pd.Timedelta("1D")
        ax_rain.bar(ts_win.index,
                    ts_win[rain_col].clip(lower=0),
                    width=width_days,
                    color="#2166ac", alpha=0.7, linewidth=0)
    ax_rain.set_ylabel("Rain\n(mm/15 min)", fontsize=6)
    ax_rain.set_ylim(bottom=0)
    ax_rain.tick_params(bottom=False)
    ax_rain.axvspan(ev_start, ev_end, alpha=0.12, color="#2166ac", zorder=0)

    for d, col in zip(depths, vwc_cols):
        if col in ts_win.columns:
            ax_vwc.plot(ts_win.index, ts_win[col],
                        color=DEPTH_CLRS[d], lw=1.0,
                        label=f"{d.lstrip('-')} cm")

    col30 = "-30|ENV__SOIL__VWC"
    if col30 in ts_win.columns and not ts_win.loc[ev_start:ev_end].empty:
        pre_val = float(row.get("pre_vwc-30", np.nan))
        slice30 = ts_win.loc[ev_start:ev_end, col30].dropna()
        if not slice30.empty and not np.isnan(pre_val):
            peak_ts  = slice30.idxmax()
            peak_val = float(slice30.max())
            ax_vwc.axhline(pre_val, color="gray", lw=0.7, ls="--", zorder=1)
            ax_vwc.annotate("pre-VWC",
                            xy=(ev_start, pre_val),
                            xytext=(ev_start + pd.Timedelta("2h"), pre_val + 0.8),
                            fontsize=6, color="gray",
                            arrowprops=dict(arrowstyle="->", color="gray", lw=0.7))
            ax_vwc.annotate("peak",
                            xy=(peak_ts, peak_val),
                            xytext=(peak_ts + pd.Timedelta("3h"), peak_val + 1.0),
                            fontsize=6,
                            arrowprops=dict(arrowstyle="->", lw=0.7))

    ax_vwc.axvspan(ev_start, ev_end, alpha=0.12, color="#2166ac", zorder=0)
    ax_vwc.set_ylabel("VWC (%)")
    ax_vwc.set_xlabel("Time")
    ax_vwc.legend(fontsize=6, loc="upper right")
    _stamp(ax_vwc, f"eui={eui}  |  data: {data_date}")
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    _save("fig_feature_extraction", fig)


# ---------------------------------------------------------------------------
# Figure: fig_landuse_scheme
# ---------------------------------------------------------------------------

def fig_landuse_scheme(df: pd.DataFrame, data_date: str, seed: int):
    """
    Land-use polygon overlay for CityErlangen: greenAtta rings, buildings.
    """
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("ERROR: geopandas is required for fig_landuse_scheme")

    shp_map = {
        "green_att_2m5": ERLANGEN_VL / "greenAtta2m5.shp",
        "green_att_5m0": ERLANGEN_VL / "greenAtta5m0.shp",
        "green_att_7m5": ERLANGEN_VL / "greenAtta7m5.shp",
        "buildings_2m5": ERLANGEN_VL / "buildings2m5.shp",
        "tree_locs":     ERLANGEN_VL / "treeLocations.shp",
    }
    missing = [k for k, p in shp_map.items() if not p.exists()]
    if missing:
        sys.exit(f"ERROR fig_landuse_scheme: missing shapefiles: {missing}")

    LU_CLRS = _lu_colours()
    layers  = {k: gpd.read_file(p).to_crs(epsg=3857)
               for k, p in shp_map.items()}

    trees = layers["tree_locs"]
    tree  = trees.iloc[0]
    geom  = tree.geometry
    # Handle both Point and MultiPoint geometry types
    if hasattr(geom, "x"):
        cx, cy = geom.x, geom.y
    else:
        pt = list(geom.geoms)[0]
        cx, cy = pt.x, pt.y
    buf   = 200  # metres in Web Mercator

    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN * 1.5, SINGLE_COL_IN * 1.5))

    layer_style = {
        "green_att_2m5": dict(color=LU_CLRS["green_att"], alpha=0.35, linewidth=0.4, edgecolor="white"),
        "green_att_5m0": dict(color=LU_CLRS["green_att"], alpha=0.30, linewidth=0.4, edgecolor="white"),
        "green_att_7m5": dict(color=LU_CLRS["green_att"], alpha=0.25, linewidth=0.4, edgecolor="white"),
        "buildings_2m5": dict(color=LU_CLRS["building"],  alpha=0.60, linewidth=0.4, edgecolor="white"),
    }
    for k, gdf in layers.items():
        if k == "tree_locs":
            continue
        clipped = gdf.cx[cx - buf:cx + buf, cy - buf:cy + buf]
        if not clipped.empty:
            clipped.plot(ax=ax, **layer_style[k])

    # Tree points on top
    trees.cx[cx - buf:cx + buf, cy - buf:cy + buf].plot(
        ax=ax, color="black", markersize=7, zorder=10)

    try:
        import contextily as ctx
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.45)
    except Exception as e:
        print(f"  contextily basemap skipped: {e}")

    ax.set_xlim(cx - buf, cx + buf)
    ax.set_ylim(cy - buf, cy + buf)
    ax.set_axis_off()
    ax.legend(handles=[
        mpatches.Patch(color=LU_CLRS["green_att"], alpha=0.5,
                       label="Green (attenuating)"),
        mpatches.Patch(color=LU_CLRS["building"],  alpha=0.7,
                       label="Buildings"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="black", markersize=5, label="Tree"),
    ], fontsize=6, loc="lower right", framealpha=0.85)
    _stamp(ax, f"CityErlangen  |  data: {data_date}")
    fig.tight_layout()
    _save("fig_landuse_scheme", fig)


# ---------------------------------------------------------------------------
# Figure: fig_site_examples  (no data -- placeholder)
# ---------------------------------------------------------------------------

def fig_site_examples(df: pd.DataFrame, data_date: str, seed: int):
    print("  SKIPPED -- fig_site_examples requires field photographs. "
          "Assemble manually in LaTeX.")


# ---------------------------------------------------------------------------
# Kapellensteg sensor data helpers
# ---------------------------------------------------------------------------

_KAPP_EUI     = "8C1F64098000005C"
_VWC_COLS     = ["-10|ENV__SOIL__VWC", "-30|ENV__SOIL__VWC", "-45|ENV__SOIL__VWC"]
_TEMP_COLS    = ["-10|ENV__SOIL__T",   "-30|ENV__SOIL__T",   "-45|ENV__SOIL__T"]
_RAIN_COL     = "MTB|ENV__ATMO__RAIN__DELTA"
_DEPTHS       = ["-10", "-30", "-45"]
_DEPTH_LABELS = ["10 cm", "30 cm", "45 cm"]
# Site-level dynamic range (from irrigation_events_poc.csv vwc_min/max)
_DYN_RANGE    = {"-10": 33.51, "-30": 31.62, "-45": 33.48}


def _load_kapp_ts() -> pd.DataFrame:
    p = TREES_DIR / _KAPP_EUI / "sensor_data.csv"
    if not p.exists():
        sys.exit(f"ERROR: sensor_data.csv not found at {p}")
    ts = pd.read_csv(p, parse_dates=["datetime"], low_memory=False)
    return ts.sort_values("datetime").set_index("datetime")


def _kapp_resample(ts: pd.DataFrame, t0: str, t1: str) -> pd.DataFrame:
    """Slice window and resample to regular 15-min grid."""
    cols = [c for c in _VWC_COLS + _TEMP_COLS + [_RAIN_COL] if c in ts.columns]
    win  = ts.loc[t0:t1, cols].copy()
    # Drop exact duplicate timestamps
    win  = win[~win.index.duplicated(keep="last")]
    interp_cols = [c for c in _VWC_COLS + _TEMP_COLS if c in win.columns]
    # Resample to 15-min: mean for VWC/temp, sum for rain
    agg = {c: "mean" for c in interp_cols}
    if _RAIN_COL in win.columns:
        agg[_RAIN_COL] = "sum"
    out = win.resample("15min").agg(agg)
    # Interpolate VWC/temp gaps; fill rain NaN with 0
    out[interp_cols] = out[interp_cols].interpolate("time")
    if _RAIN_COL in out.columns:
        out[_RAIN_COL] = out[_RAIN_COL].fillna(0)
    return out


def _event_metrics_raw(ts_win: pd.DataFrame, onset: pd.Timestamp,
                        pre_h: int = 3, look_ahead_h: int = 168):
    """
    Compute pre_vwc, peak_vwc, delta_vwc, delta_pct_dyn, dry_h, temp_sum
    for each depth from raw resampled data around a known onset timestamp.
    """
    metrics = {}
    for d, col, tcol in zip(_DEPTHS, _VWC_COLS,
                             [c if c in ts_win.columns else None for c in _TEMP_COLS]):
        if col not in ts_win.columns:
            continue
        pre_win  = ts_win.loc[onset - pd.Timedelta(hours=pre_h):onset, col].dropna()
        pre_vwc  = float(pre_win.median()) if not pre_win.empty else np.nan

        post     = ts_win.loc[onset:onset + pd.Timedelta(hours=look_ahead_h), col].dropna()
        if post.empty or np.isnan(pre_vwc):
            metrics[d] = dict(pre=np.nan, peak=np.nan, delta=np.nan,
                              pct_dyn=np.nan, dry_h=np.nan, temp_sum=np.nan,
                              peak_ts=onset)
            continue

        peak_ts  = post.idxmax()
        peak_vwc = float(post.max())
        delta    = peak_vwc - pre_vwc
        pct_dyn  = 100 * delta / _DYN_RANGE[d] if delta > 0 else np.nan

        # Drying: first time after peak where VWC <= pre_vwc
        after_peak = post.loc[peak_ts:]
        dry_ts     = after_peak[after_peak <= pre_vwc + 0.5]
        dry_h      = float((dry_ts.index[0] - onset).total_seconds() / 3600) \
                     if not dry_ts.empty else np.nan

        # Temperature sum over drying period
        temp_sum = np.nan
        if tcol and tcol in ts_win.columns and not np.isnan(dry_h):
            dry_end = onset + pd.Timedelta(hours=dry_h)
            t_ser   = ts_win.loc[onset:dry_end, tcol].dropna()
            # Integrate: mean T × elapsed h (trapz over hourly grid)
            if len(t_ser) > 1:
                hrs = (t_ser.index - t_ser.index[0]).total_seconds() / 3600
                temp_sum = float(np.trapz(t_ser.values, hrs))

        metrics[d] = dict(pre=pre_vwc, peak=peak_vwc, delta=delta,
                          pct_dyn=pct_dyn, dry_h=dry_h, temp_sum=temp_sum,
                          peak_ts=peak_ts)
    return metrics


def _kapp_4panel(ts_win, events, fig_name, title, data_date,
                 t_start: str, t_end: str):
    """
    Draw the 4-panel figure (VWC-10, VWC-30, VWC-45, Rain).
    events: list of dicts with keys onset, metrics, label, color, linestyle
    """
    depth_colors = [DEPTH_CLRS[d] for d in _DEPTHS]
    rain_color   = "#2166ac"

    fig, axes = plt.subplots(4, 1, figsize=(DOUBLE_COL_IN, 5.5), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2, 2, 0.9]})
    fig.suptitle(title, fontsize=7.5, y=0.995)

    for ax_idx, (d, col, lbl, clr) in enumerate(
            zip(_DEPTHS, _VWC_COLS, _DEPTH_LABELS, depth_colors)):
        ax = axes[ax_idx]
        if col not in ts_win.columns:
            ax.set_visible(False)
            continue

        ser = ts_win[col].dropna()
        ax.plot(ser.index, ser.values, color=clr, lw=0.9, label=f"VWC {lbl}")
        ax.set_ylabel(f"VWC {lbl} (%)", fontsize=7)
        ax.grid(lw=0.3, alpha=0.35, axis="y")

        for ev in events:
            onset   = ev["onset"]
            m       = ev["metrics"].get(d, {})
            pre     = m.get("pre", np.nan)
            peak    = m.get("peak", np.nan)
            pct     = m.get("pct_dyn", np.nan)
            dry_h   = m.get("dry_h", np.nan)
            t_sum   = m.get("temp_sum", np.nan)
            peak_ts = m.get("peak_ts", onset)

            ev_color = ev.get("color", "red")
            ax.axvline(onset, color=ev_color, lw=0.9, ls="--", zorder=4)

            if not np.isnan(pre):
                ax.axhline(pre, color=clr, lw=0.6, ls=":", alpha=0.7, zorder=3)

            # Arrow pre → peak with label
            if not (np.isnan(pre) or np.isnan(peak)) and peak > pre + 1:
                pct_lbl = f"d = {pct:.1f}%" if not np.isnan(pct) else f"Δ = {peak-pre:.1f}%"
                ax.annotate(
                    "", xy=(peak_ts, peak), xytext=(peak_ts, pre),
                    arrowprops=dict(arrowstyle="<->", color=clr, lw=1.2),
                    zorder=5,
                )
                mid_y = (pre + peak) / 2
                ax.text(peak_ts + pd.Timedelta(hours=3), mid_y,
                        pct_lbl, color=clr, fontsize=6.5, fontweight="bold",
                        va="center", zorder=5)

            # Drying annotation (bottom-right corner of panel)
            if not np.isnan(dry_h):
                dry_str = f"Drying: {dry_h:.0f} h"
            else:
                dry_str = "Drying: not within window"
            t_str = (f"Temp sum: {t_sum:.0f} °C·h\n" if not np.isnan(t_sum) else
                     "Temp sum: n/a\n")
            ax.text(0.99, 0.04, t_str + dry_str,
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=5.5, color="gray", alpha=0.85)

    # Rain panel
    ax_rain = axes[3]
    if _RAIN_COL in ts_win.columns:
        rain = ts_win[_RAIN_COL].fillna(0).clip(lower=0)
        width = pd.Timedelta("15min") / pd.Timedelta("1D")
        ax_rain.bar(rain.index, rain.values, width=width,
                    color=rain_color, alpha=0.75, linewidth=0)
        # Cumulative line on secondary axis
        ax2 = ax_rain.twinx()
        cumrain = rain.cumsum()
        ax2.plot(cumrain.index, cumrain.values,
                 color=rain_color, lw=0.8, ls="--", alpha=0.6)
        ax2.set_ylabel("Cum. (mm)", fontsize=5.5, color=rain_color, alpha=0.7)
        ax2.tick_params(axis="y", labelsize=5, colors=rain_color)
    ax_rain.set_ylabel("Rain\n(mm/15 min)", fontsize=6)
    ax_rain.set_ylim(bottom=0)
    for ev in events:
        ax_rain.axvline(ev["onset"], color=ev.get("color", "red"),
                        lw=0.9, ls="--", zorder=4)

    # Onset markers on all panels
    for ev in events:
        for ax in axes[:3]:
            ax.axvspan(ev["onset"], ev.get("end", ev["onset"] + pd.Timedelta("1h")),
                       alpha=0.07, color=ev.get("color", "red"), zorder=0)

    # X-axis formatting
    import matplotlib.dates as mdates
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
    axes[3].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    axes[3].tick_params(axis="x", labelsize=6)

    _stamp(axes[3], f"Kapellensteg ({_KAPP_EUI})  |  data: {data_date}")
    fig.tight_layout()
    _save(fig_name, fig)


# ---------------------------------------------------------------------------
# Figure: fig_kapellensteg_rain  (28 May 2025) — combined VWC panel
# ---------------------------------------------------------------------------

def fig_kapellensteg_rain(df: pd.DataFrame, data_date: str, seed: int):
    """
    2-panel explanatory figure: combined VWC (30/60/90 cm display) + rain.
    Rain event 28 May 2025. Crameri hawaii colours.
    - Amplitude arrows: rotated 90° label; 30cm to left, 60cm lower.
    - Drying arrows: horizontal label above; dry_end = peak_ts + dry_h (from-peak convention).
    - 90th pct / pre-event VWC lines: labels outside right y-axis, collision-resolved.
    - Exact x window, no padding.
    """
    import matplotlib.dates as mdates
    import cmcrameri.cm as cmc
    from matplotlib.transforms import blended_transform_factory

    T0, T1 = "2025-05-27 12:00", "2025-06-02 00:00"

    disp_lbl  = {"-10": "30 cm",  "-30": "60 cm",  "-45": "90 cm"}
    disp_var  = {"-10": "30cm",   "-30": "60cm",   "-45": "90cm"}
    kapp_clrs = {
        "-10": cmc.hawaii(0.05),   # dark magenta — shallowest (30 cm)
        "-30": cmc.hawaii(0.38),   # dark amber   — medium   (60 cm)
        "-45": cmc.hawaii(0.72),   # bright green — deepest  (90 cm)
    }

    ts  = _load_kapp_ts()
    win = _kapp_resample(ts, T0, T1)

    # p05 / p95 from 4 h rolling-median of full timeseries — exact same computation
    # as _dynamic_range() in _poc_irrigation_v3.py, so dyn_range = p95 - p05
    # and ΔVWCnorm = (peak - pre) / dyn_range * 100 is directly readable from
    # the span between the two horizontal reference lines.
    rs30 = ts[_VWC_COLS].resample("30min").mean()
    pct05_map, pct95_map = {}, {}
    for col, dep in zip(_VWC_COLS, _DEPTHS):
        if col not in rs30.columns:
            continue
        smoothed = rs30[col].rolling(8, center=True, min_periods=4).median()
        valid    = smoothed[(smoothed > 2.0) & (smoothed < 50.0)]
        if valid.empty:
            continue
        pct05_map[dep] = float(valid.quantile(0.05))
        pct95_map[dep] = float(valid.quantile(0.95))

    onset = pd.Timestamp("2025-05-28 02:00")
    m     = _event_metrics_raw(win, onset, pre_h=3, look_ahead_h=168)
    # Override pct_dyn with rf_dataset values; dry_h (from peak) also from rf_dataset
    for d, pct, dry in zip(_DEPTHS, [120.6, 104.3, 67.1], [99.0, 91.0, 73.0]):
        if d in m:
            m[d]["pct_dyn"] = pct
            m[d]["dry_h"]   = dry   # hours from PEAK (rf_dataset convention)

    print("  Rain event 28 May 2025 (28.0 mm, Q4, BSC 0000):")
    for dep in _DEPTHS:
        if dep in m:
            print(f"    {dep}cm: pre={m[dep]['pre']:.1f}%  peak={m[dep]['peak']:.1f}%  "
                  f"d_pct_dyn={m[dep]['pct_dyn']:.1f}%  dry_h={m[dep]['dry_h']} h from peak")

    fig, (ax_vwc, ax_rain) = plt.subplots(
        2, 1, figsize=(170 / 25.4, 3.3), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    drying_y_offset = {"-10": -9.0, "-30": -6.0, "-45": -3.0}
    min_pre = min(m[d]["pre"] for d in _DEPTHS
                  if d in m and not np.isnan(m[d].get("pre", np.nan)))
    max_vwc = max(win[c].max() for c in _VWC_COLS if c in win.columns)

    # Separate any p95 pair closer than 0.5 %
    vwc_95_map = dict(pct95_map)
    for _ in range(10):
        moved = False
        for da, db in zip(sorted(vwc_95_map, key=vwc_95_map.get),
                          sorted(vwc_95_map, key=vwc_95_map.get)[1:]):
            gap = vwc_95_map[db] - vwc_95_map[da]
            if gap < 0.5:
                shift = (0.5 - gap) / 2
                vwc_95_map[da] -= shift
                vwc_95_map[db] += shift
                moved = True
        if not moved:
            break

    # Accumulate right-side label positions (y_true, text, color)
    right_labels_raw = []

    for col, dep in zip(_VWC_COLS, _DEPTHS):
        if col not in win.columns:
            continue
        clr = kapp_clrs[dep]
        lbl = disp_lbl[dep]
        var = disp_var[dep]
        ser = win[col].dropna()

        ax_vwc.plot(ser.index, ser.values, color=clr, lw=1.1, label=f"VWC {lbl}")

        md      = m.get(dep, {})
        pre     = md.get("pre", np.nan)
        peak    = md.get("peak", np.nan)
        pct     = md.get("pct_dyn", np.nan)
        dry_h   = md.get("dry_h", np.nan)   # hours from peak
        peak_ts = md.get("peak_ts", onset)

        # p95 line (4h-smoothed full series) — defines upper end of dyn_range
        vwc_95 = vwc_95_map.get(dep, np.nan)
        if not np.isnan(vwc_95):
            ax_vwc.axhline(vwc_95, color=clr, lw=0.8, ls="--", alpha=0.55, zorder=1)
            right_labels_raw.append((vwc_95, "p95", clr))

        # p05 label only — no line (would overlap with drying arrows)
        vwc_05 = pct05_map.get(dep, np.nan)
        if not np.isnan(vwc_05):
            right_labels_raw.append((vwc_05, "p05", clr))

        # Pre-event VWC dotted line — anchors the eye to baseline before onset
        if not np.isnan(pre):
            ax_vwc.axhline(pre, color=clr, lw=0.7, ls=":", alpha=0.55)

        # Thin vertical line at VWC peak — marks the response maximum
        ax_vwc.axvline(peak_ts, color=clr, lw=0.5, ls="-", alpha=0.35, zorder=2)
        ax_rain.axvline(peak_ts, color=clr, lw=0.5, ls="-", alpha=0.35, zorder=2)

        # Amplitude arrow — rotated 90° label; per-depth side/position
        if not (np.isnan(pre) or np.isnan(peak)) and peak > pre + 0.5:
            ax_vwc.annotate(
                "", xy=(peak_ts, peak), xytext=(peak_ts, pre),
                arrowprops=dict(arrowstyle="<->", color=clr, lw=1.2),
                zorder=5,
            )
            amp_lbl = (f"$\\mathbf{{\\Delta VWC}}_{{\\mathbf{{norm}}}}$ = {pct:.0f}%"
                       if not np.isnan(pct) else
                       f"$\\mathbf{{\\Delta VWC}}_{{\\mathbf{{norm}}}}$ = {peak-pre:.1f}%")
            amp_y = 24.5 + {"-10": 5.0, "-30": 0.0, "-45": -1.5}.get(dep, 0.0)
            if dep == "-10":   # 30 cm — label to the LEFT of the arrow
                amp_x = peak_ts - pd.Timedelta(hours=5)
            else:              # 60 cm / 90 cm — label just right of arrow
                amp_x = peak_ts + pd.Timedelta(hours=3)
            ax_vwc.text(
                amp_x, amp_y, amp_lbl,
                color=clr, fontsize=8, fontweight="bold",
                va="center", ha="center", rotation=90, zorder=6,
            )

        # Drying end: dry_h is from PEAK (rf_dataset convention)
        # dry_end = peak_ts + dry_h  (NOT onset + dry_h)
        if not np.isnan(dry_h):
            dry_end = peak_ts + pd.Timedelta(hours=dry_h)
            ax_vwc.axvline(dry_end, color=clr, lw=0.8, ls="--", alpha=0.7, zorder=3)
            ax_rain.axvline(dry_end, color=clr, lw=0.8, ls="--", alpha=0.7, zorder=3)

            arrow_y = min_pre + drying_y_offset[dep]
            ax_vwc.annotate(
                "", xy=(dry_end, arrow_y), xytext=(peak_ts, arrow_y),
                arrowprops=dict(arrowstyle="<->", color=clr, lw=1.0,
                                shrinkA=0, shrinkB=0),
                zorder=5,
            )
            mid_t = pd.Timestamp("2025-05-30 12:00")
            ax_vwc.text(
                mid_t, arrow_y,
                f"$\\mathbf{{t}}_{{\\mathbf{{dry}}}}$ = {dry_h:.0f} h",
                color=clr, fontsize=8, ha="center", va="bottom",
                fontweight="bold", zorder=6,
            )

    # Single black 'pre-event' label at mean pre across depths
    _pre_vals = [m[d]["pre"] for d in _DEPTHS if d in m and not np.isnan(m[d].get("pre", np.nan))]
    if _pre_vals:
        right_labels_raw.append((float(np.mean(_pre_vals)), "pre-event", "black"))

    # Onset vertical line
    ax_vwc.axvline(onset, color="#d73027", lw=1.0, ls="--", zorder=4,
                   label="Event onset")
    ax_rain.axvline(onset, color="#d73027", lw=1.0, ls="--", zorder=4)

    # --- Collision-resolved right-axis labels ---
    def _resolve(labels, min_gap=2.0):
        sl = sorted(labels, key=lambda x: x[0])
        ys = [l[0] for l in sl]
        for _ in range(30):
            moved = False
            for i in range(1, len(ys)):
                g = ys[i] - ys[i - 1]
                if g < min_gap:
                    shift = (min_gap - g) / 2
                    ys[i - 1] -= shift
                    ys[i]     += shift
                    moved = True
            if not moved:
                break
        return [(ys[i], sl[i][1], sl[i][2]) for i in range(len(sl))]

    trans_r = blended_transform_factory(ax_vwc.transAxes, ax_vwc.transData)
    for y_adj, txt, clr in _resolve(right_labels_raw):
        if txt == "p05":   # no horizontal plot line → draw a small tick on the axis
            ax_vwc.plot([1.0, 1.012], [y_adj, y_adj],
                        transform=trans_r, color=clr, lw=0.9, alpha=0.75,
                        clip_on=False, solid_capstyle="round")
        ax_vwc.text(1.015, y_adj, txt, transform=trans_r,
                    fontsize=8, fontweight="bold", va="center", ha="left",
                    color=clr, alpha=0.9, clip_on=False)

    # Y-limits
    y_bottom = min_pre + min(drying_y_offset.values()) - 1.8
    ax_vwc.set_ylim(bottom=y_bottom, top=43)
    ax_vwc.set_ylabel("VWC (%)", fontsize=8, fontweight="bold")
    ax_vwc.grid(lw=0.3, alpha=0.3)
    leg = ax_vwc.legend(fontsize=8, loc="upper right", framealpha=0.75, ncol=2)
    for _t in leg.get_texts():
        _t.set_fontweight('bold')

    # --- Rain panel ---
    if _RAIN_COL in win.columns:
        # Native sensor interval is 1 h — aggregate 15-min resampled data back to hourly
        rain_1h = win[_RAIN_COL].resample("1h").sum().fillna(0).clip(lower=0)
        width = pd.Timedelta("50min") / pd.Timedelta("1D")
        ax_rain.bar(rain_1h.index, rain_1h.values, width=width,
                    color="#2166ac", alpha=0.75, linewidth=0)
        ax2 = ax_rain.twinx()
        ax2.plot(rain_1h.cumsum().index, rain_1h.cumsum().values,
                 color="#2166ac", lw=0.8, ls="--", alpha=0.6)
        ax2.set_ylabel("Cum. (mm)", fontsize=8, fontweight="bold", color="#2166ac", alpha=0.7)
        ax2.tick_params(axis="y", labelsize=8, colors="#2166ac")
    ax_rain.set_ylabel("Rain (mm/h)", fontsize=8, fontweight="bold")
    ax_rain.set_ylim(bottom=0)
    ax_rain.grid(lw=0.3, alpha=0.3, axis="x")

    # X-axis: exact window, daily ticks, minor at noon
    ax_rain.set_xlim(pd.Timestamp(T0), pd.Timestamp(T1))
    ax_rain.xaxis.set_major_locator(mdates.DayLocator())
    ax_rain.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax_rain.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))
    ax_rain.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    for _ax in fig.get_axes():
        for _lbl in _ax.get_xticklabels() + _ax.get_yticklabels():
            _lbl.set_fontweight('bold')
            _lbl.set_fontfamily('Arial')
    leg = ax_vwc.get_legend()
    if leg:
        for _t in leg.get_texts():
            _t.set_fontfamily('Arial')
    fig.subplots_adjust(right=0.80)   # reserve right margin for pct labels
    _save("fig_kapellensteg_rain", fig)


# ---------------------------------------------------------------------------
# Figure: fig_kapellensteg_irrigation  (8 – 13 Jul 2026)
# ---------------------------------------------------------------------------

def fig_kapellensteg_irrigation(df: pd.DataFrame, data_date: str, seed: int):
    """
    2-panel VWC + rain figure for Kapellensteg irrigation event, 8–13 Jul 2026.
    Same style as fig_kapellensteg_rain: combined VWC panel, Crameri hawaii colours,
    amplitude arrows, drying arrows, p05/p95 from full-timeseries rolling median.
    """
    import matplotlib.dates as mdates
    import cmcrameri.cm as cmc
    from matplotlib.transforms import blended_transform_factory

    T0, T1 = "2026-07-07 00:00", "2026-07-14 00:00"
    disp_lbl  = {"-10": "30 cm",  "-30": "60 cm",  "-45": "90 cm"}
    disp_var  = {"-10": "30cm",   "-30": "60cm",   "-45": "90cm"}
    kapp_clrs = {
        "-10": cmc.hawaii(0.05),
        "-30": cmc.hawaii(0.38),
        "-45": cmc.hawaii(0.72),
    }

    ts  = _load_kapp_ts()
    win = _kapp_resample(ts, T0, T1)

    # p05/p95 from 4h rolling-median of full timeseries — matches rf_dataset _dynamic_range()
    rs30 = ts[_VWC_COLS].resample("30min").mean()
    pct05_map, pct95_map = {}, {}
    for col, dep in zip(_VWC_COLS, _DEPTHS):
        if col not in rs30.columns:
            continue
        smoothed = rs30[col].rolling(8, center=True, min_periods=4).median()
        valid    = smoothed[(smoothed > 2.0) & (smoothed < 50.0)]
        if valid.empty:
            continue
        pct05_map[dep] = float(valid.quantile(0.05))
        pct95_map[dep] = float(valid.quantile(0.95))

    onset = pd.Timestamp("2026-07-08 04:45")
    m     = _event_metrics_raw(win, onset, pre_h=3, look_ahead_h=168)

    print("  Irrigation event 8 Jul 2026:")
    for d in _DEPTHS:
        if d in m:
            print(f"    {disp_lbl[d]}: pre={m[d]['pre']:.1f}%  peak={m[d]['peak']:.1f}%  "
                  f"d_pct_dyn={m[d]['pct_dyn']:.1f}%  dry_h={m[d]['dry_h']:.1f} h from peak")

    fig, ax_vwc = plt.subplots(
        1, 1, figsize=(170 / 25.4, 2.6),
    )

    drying_y_offset = {"-10": -9.0, "-30": -6.0, "-45": -3.0}
    min_pre = min(m[d]["pre"] for d in _DEPTHS if d in m and not np.isnan(m[d].get("pre", np.nan)))
    max_vwc = max(win[c].max() for c in _VWC_COLS if c in win.columns)

    # Separate p95 pairs < 0.5%
    vwc_95_map = dict(pct95_map)
    for _ in range(10):
        moved = False
        for da, db in zip(sorted(vwc_95_map, key=vwc_95_map.get),
                          sorted(vwc_95_map, key=vwc_95_map.get)[1:]):
            gap = vwc_95_map[db] - vwc_95_map[da]
            if gap < 0.5:
                shift = (0.5 - gap) / 2
                vwc_95_map[da] -= shift
                vwc_95_map[db] += shift
                moved = True
        if not moved:
            break

    right_labels_raw = []

    for col, dep in zip(_VWC_COLS, _DEPTHS):
        if col not in win.columns:
            continue
        clr = kapp_clrs[dep]; lbl = disp_lbl[dep]; var = disp_var[dep]
        ser = win[col].dropna()
        ax_vwc.plot(ser.index, ser.values, color=clr, lw=1.1, label=f"VWC {lbl}")

        md      = m.get(dep, {})
        pre     = md.get("pre", np.nan)
        peak    = md.get("peak", np.nan)
        pct     = md.get("pct_dyn", np.nan)
        dry_h   = md.get("dry_h", np.nan)
        peak_ts = md.get("peak_ts", onset)

        # p95 dashed line
        vwc_95 = vwc_95_map.get(dep, np.nan)
        if not np.isnan(vwc_95):
            ax_vwc.axhline(vwc_95, color=clr, lw=0.8, ls="--", alpha=0.55, zorder=1)
            right_labels_raw.append((vwc_95, "p95", clr))

        # p05 label only — no line (would overlap with drying arrows)
        vwc_05 = pct05_map.get(dep, np.nan)
        if not np.isnan(vwc_05):
            right_labels_raw.append((vwc_05, "p05", clr))

        # Pre-event VWC dotted line
        if not np.isnan(pre):
            ax_vwc.axhline(pre, color=clr, lw=0.7, ls=":", alpha=0.55)

        # Thin vertical line at VWC peak
        ax_vwc.axvline(peak_ts, color=clr, lw=0.5, ls="-", alpha=0.35, zorder=2)

        # Amplitude arrow — all peaks nearly simultaneous; stagger labels horizontally
        if not (np.isnan(pre) or np.isnan(peak)) and peak > pre + 0.5:
            ax_vwc.annotate("", xy=(peak_ts, peak), xytext=(peak_ts, pre),
                            arrowprops=dict(arrowstyle="<->", color=clr, lw=1.2), zorder=5)
            amp_lbl = (f"$\\mathbf{{\\Delta VWC}}_{{\\mathbf{{norm}}}}$ = {pct:.0f}%"
                       if not np.isnan(pct) else
                       f"$\\mathbf{{\\Delta VWC}}_{{\\mathbf{{norm}}}}$ = {peak-pre:.1f}%")
            amp_y = 28.0 + {"-10": 5.0, "-30": 0.0, "-45": -3.0}.get(dep, 0.0)
            if dep == "-10":
                amp_x = peak_ts - pd.Timedelta(hours=6)
            elif dep == "-30":
                amp_x = peak_ts + pd.Timedelta(hours=5)
            else:
                amp_x = peak_ts + pd.Timedelta(hours=10)
            ax_vwc.text(amp_x, amp_y, amp_lbl, color=clr, fontsize=8,
                        fontweight="bold",
                        va="center", ha="center", rotation=90, zorder=6)

        # Drying arrows: peak → dry_end (dry_h from peak, _event_metrics_raw convention)
        if not np.isnan(dry_h):
            dry_end = peak_ts + pd.Timedelta(hours=dry_h)
            ax_vwc.axvline(dry_end, color=clr, lw=0.8, ls="--", alpha=0.7, zorder=3)
            arrow_y = min_pre + drying_y_offset[dep]
            ax_vwc.annotate("", xy=(dry_end, arrow_y), xytext=(peak_ts, arrow_y),
                            arrowprops=dict(arrowstyle="<->", color=clr, lw=1.0,
                                           shrinkA=0, shrinkB=0), zorder=5)
            mid_t = pd.Timestamp("2026-07-10 12:00")
            ax_vwc.text(mid_t, arrow_y,
                        f"$\\mathbf{{t}}_{{\\mathbf{{dry}}}}$ = {dry_h:.0f} h",
                        color=clr, fontsize=8, ha="center", va="bottom",
                        fontweight="bold", zorder=6)

    # Single black 'pre-event' label at mean pre across depths
    _pre_vals = [m[d]["pre"] for d in _DEPTHS if d in m and not np.isnan(m[d].get("pre", np.nan))]
    if _pre_vals:
        right_labels_raw.append((float(np.mean(_pre_vals)), "pre-event", "black"))

    ax_vwc.axvline(onset, color="#d73027", lw=1.0, ls="--", zorder=4, label="Event onset")

    # Collision-resolved right-axis labels with p05 tick marks
    def _resolve(labels, min_gap=2.0):
        sl = sorted(labels, key=lambda x: x[0])
        ys = [l[0] for l in sl]
        for _ in range(30):
            moved = False
            for i in range(1, len(ys)):
                g = ys[i] - ys[i - 1]
                if g < min_gap:
                    shift = (min_gap - g) / 2
                    ys[i - 1] -= shift
                    ys[i]     += shift
                    moved = True
            if not moved:
                break
        return [(ys[i], sl[i][1], sl[i][2]) for i in range(len(sl))]

    trans_r = blended_transform_factory(ax_vwc.transAxes, ax_vwc.transData)
    for y_adj, txt, clr in _resolve(right_labels_raw):
        if txt == "p05":
            ax_vwc.plot([1.0, 1.012], [y_adj, y_adj],
                        transform=trans_r, color=clr, lw=0.9, alpha=0.75,
                        clip_on=False, solid_capstyle="round")
        ax_vwc.text(1.015, y_adj, txt, transform=trans_r,
                    fontsize=8, fontweight="bold", va="center", ha="left",
                    color=clr, alpha=0.9, clip_on=False)

    y_bottom = min_pre + min(drying_y_offset.values()) - 1.8
    y_top    = max_vwc + 2.5
    ax_vwc.set_ylim(bottom=y_bottom, top=y_top)
    ax_vwc.set_ylabel("VWC (%)", fontsize=8, fontweight="bold")
    ax_vwc.grid(lw=0.3, alpha=0.3)
    leg = ax_vwc.legend(fontsize=8, loc="upper right", framealpha=0.75, ncol=2)
    for _t in leg.get_texts():
        _t.set_fontweight('bold')

    # X-axis (single panel — no shared rain axis)
    ax_vwc.set_xlim(pd.Timestamp(T0), pd.Timestamp(T1))
    ax_vwc.xaxis.set_major_locator(mdates.DayLocator())
    ax_vwc.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax_vwc.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))
    ax_vwc.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    for _ax in fig.get_axes():
        for _lbl in _ax.get_xticklabels() + _ax.get_yticklabels():
            _lbl.set_fontweight('bold')
            _lbl.set_fontfamily('Arial')
    leg = ax_vwc.get_legend()
    if leg:
        for _t in leg.get_texts():
            _t.set_fontfamily('Arial')
    fig.subplots_adjust(right=0.80)
    _save("fig_kapellensteg_irrigation", fig)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------
FIGURES = {
    "fig_event_scatter":            fig_event_scatter,
    "fig_event_shapes":             fig_event_shapes,
    "fig_corr_heatmap":             fig_corr_heatmap,
    "fig_univariate_increase":      fig_univariate_increase,
    "fig_univariate_drying":        fig_univariate_drying,
    "fig_feature_extraction":       fig_feature_extraction,
    "fig_landuse_scheme":           fig_landuse_scheme,
    "fig_site_examples":            fig_site_examples,
    "fig_kapellensteg_rain":        fig_kapellensteg_rain,
    "fig_kapellensteg_irrigation":  fig_kapellensteg_irrigation,
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate manuscript figures.")
    parser.add_argument("--only", metavar="FIGNAME",
                        help=f"Generate only this figure. "
                             f"Choices: {', '.join(FIGURES)}")
    parser.add_argument("--data-date",
                        default=str(datetime.date.today()),
                        help="Data snapshot date stamped in figure corner (YYYY-MM-DD).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (printed and set at startup).")
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f"Random seed : {args.seed}")
    print(f"Data date   : {args.data_date}")
    print(f"Output dir  : {FIG_DIR}")
    print()

    if args.only:
        if args.only not in FIGURES:
            sys.exit(
                f"ERROR: unknown figure '{args.only}'.\n"
                f"Available: {', '.join(FIGURES)}"
            )
        targets = {args.only: FIGURES[args.only]}
    else:
        targets = FIGURES

    df = _load_rf()

    for name, fn in targets.items():
        print(f"[{name}]")
        fn(df, args.data_date, args.seed)
        print()


if __name__ == "__main__":
    main()
