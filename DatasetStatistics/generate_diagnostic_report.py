#!/usr/bin/env python3
"""
Diagnostic report generator for RF soil-moisture and drying-time models.

Requires caches re-generated after the r2_train / y_pred / y_shap additions
to analyze_rf_soil_moisture.py / analyze_rf_drying_time.py.

Usage:
    python generate_diagnostic_report.py
    python generate_diagnostic_report.py --sm    # soil moisture only
    python generate_diagnostic_report.py --dt    # drying time only
    python generate_diagnostic_report.py --out my_report.pdf
"""
import argparse
import pickle
import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

SM_CACHE    = SCRIPT_DIR / "rf_soil_moisture_v4_training_cache.pkl"
DT_CACHE    = SCRIPT_DIR / "rf_drying_time_v3_training_cache.pkl"
BORUTA_PKL  = SCRIPT_DIR / "boruta_results.pkl"
OUT_PDF     = SCRIPT_DIR / "rf_diagnostic_report.pdf"

N_TOP_FEAT_SCATTER = 6   # feature-vs-target scatter: show top N features
CLIP_PCT = 5             # percentile used for axis clipping (5th–95th = 90% coverage)

DEPTH_LBLS   = {"-10": "−10 cm", "-30": "−30 cm", "-45": "−45 cm"}
DEPTH_COLORS = {"-10": "#e74c3c", "-30": "#3498db", "-45": "#27ae60"}  # red / blue / green

SM_TYPE_LABELS = {
    "BSC_XX11_Q12":    "Rain (front-loaded)",
    "IRRIGATION":      "Irrigation",
    "IRRIGATION_STRONG": "Irrigation strong",
}
DT_TYPE_LABELS = {
    "BSC_00XX_Q34":    "Rain (back-loaded)",
    "IRRIGATION":      "Irrigation",
    "IRRIGATION_STRONG": "Irrigation strong",
}

# ---------------------------------------------------------------------------
# Load Boruta baseline: {target_key: {et: {depth_key: (r2_mean, r2_std, confirmed, tentative)}}}
# ---------------------------------------------------------------------------
_BORUTA = {}
if BORUTA_PKL.exists():
    with open(BORUTA_PKL, "rb") as _f:
        _boruta_raw = pickle.load(_f)
    for _tkey, _et_data in _boruta_raw.items():
        _BORUTA[_tkey] = {}
        for _et, _depths in _et_data.items():
            _BORUTA[_tkey][_et] = {}
            for _depth, _info in _depths.items():
                _r2 = _info.get("r2_selected", (np.nan, np.nan))
                _BORUTA[_tkey][_et][_depth] = {
                    "r2_mean":   float(_r2[0]),
                    "r2_std":    float(_r2[1]),
                    "confirmed": list(_info.get("confirmed", [])),
                    "tentative": list(_info.get("tentative", [])),
                }


def _boruta_lookup(target_key, et, dlbl):
    """Return Boruta dict for (target_key, et, depth_label) or None."""
    d_key = dlbl.replace(" cm", "").strip().replace("−", "-")
    return _BORUTA.get(target_key, {}).get(et, {}).get(d_key)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r2_ols(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    _, _, r, _, _ = stats.linregress(x[mask], y[mask])
    return r ** 2


def _ols_line(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])
    return slope, intercept


def _axis_lim(arr, pct=CLIP_PCT, pad=0.05):
    """(lo, hi) covering the central (100-2*pct)% of data, with small padding."""
    a = arr[np.isfinite(arr)]
    if len(a) == 0:
        return 0, 1
    lo = float(np.percentile(a, pct))
    hi = float(np.percentile(a, 100 - pct))
    span = hi - lo
    if span == 0:
        span = max(abs(hi), 1.0)
    return lo - pad * span, hi + pad * span


def _caption(fig, text, y=0.01):
    """Add a small italic caption at the bottom of a figure."""
    fig.text(0.5, y, text, ha="center", va="bottom",
             fontsize=7, style="italic", color="#444",
             wrap=True, transform=fig.transFigure)


def load_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Page – R² summary table
# ---------------------------------------------------------------------------

def _fmt_feat_names(confirmed, tentative, per_line=4):
    """Return feature names wrapped with newlines. Tentative features marked (t)."""
    items = sorted(confirmed) + [f"{f}(t)" for f in sorted(tentative)]
    if not items:
        return "—"
    # remap pkl-era name to current name in display
    items = [("day_cos" if f == "month_cos" else f) for f in items]
    chunks = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return "\n".join(chunks)


def _build_r2_table(trained_dict, type_labels, title, target_key):
    """Coloured R² summary table with Boruta comparison, feature counts, and feature names."""
    rows = []
    feat_strings = []   # parallel list for row-height calculation
    for et, (_, _, depth_results) in trained_dict.items():
        et_lbl = type_labels.get(et, et)
        for dr in depth_results:
            b = _boruta_lookup(target_key, et, dr["dlbl"])
            b_r2  = f"{b['r2_mean']:+.3f}" if b else "—"
            b_std = f"±{b['r2_std']:.3f}"  if b else ""
            b_nf  = str(len(b["confirmed"]) + len(b["tentative"])) if b else "—"
            feat_str = _fmt_feat_names(b["confirmed"], b["tentative"]) if b else "—"
            feat_strings.append(feat_str)
            if dr["skip"]:
                rows.append({
                    "Event type": et_lbl, "Depth": dr["dlbl"],
                    "n": "—", "CV R²": "—", "±": "",
                    "Train R²": "—", "Δ": "—",
                    "n feats": "—",
                    "Boruta CV R²": b_r2, "±  ": b_std, "B feats": b_nf,
                    "Boruta features": feat_str,
                })
            else:
                cv_mean = float(dr["r2s"].mean())
                cv_std  = float(dr["r2s"].std())
                r2_tr   = dr.get("r2_train")
                delta   = (r2_tr - cv_mean) if r2_tr is not None else None
                n_feats = len(dr.get("feat_names", []))
                rows.append({
                    "Event type": et_lbl, "Depth": dr["dlbl"],
                    "n": dr["n_samp"],
                    "CV R²": f"{cv_mean:+.3f}",
                    "±": f"{cv_std:.3f}",
                    "Train R²": f"{r2_tr:.3f}" if r2_tr is not None else "—",
                    "Δ": f"{delta:+.3f}" if delta is not None else "—",
                    "n feats": n_feats,
                    "Boruta CV R²": b_r2, "±  ": b_std, "B feats": b_nf,
                    "Boruta features": feat_str,
                })

    df = pd.DataFrame(rows)
    col_labels = list(df.columns)
    cell_text  = df.values.tolist()

    # Scale figure height to accommodate multi-line feature-name cells
    INCHES_PER_LINE = 0.26
    HDR_INCHES      = 0.32
    MARGIN_INCHES   = 2.5   # title + caption + padding
    AXES_FRAC       = 0.88  # fraction of figure height used by axes (after tight_layout)

    lines_per_row = [max(1, s.count("\n") + 1) for s in feat_strings]
    total_data_in = sum(lines_per_row) * INCHES_PER_LINE
    fig_h = max(3.5, total_data_in + HDR_INCHES + MARGIN_INCHES)
    axes_h_in = fig_h * AXES_FRAC

    fig, ax = plt.subplots(figsize=(26, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(range(len(col_labels)))

    # Adjust per-row heights so multi-line cells display fully
    feat_col_idx = col_labels.index("Boruta features")
    hdr_h  = HDR_INCHES / axes_h_in
    for c in range(len(col_labels)):
        tbl[0, c].set_height(hdr_h)
    for r_i, n_lines in enumerate(lines_per_row, start=1):
        row_h = n_lines * INCHES_PER_LINE / axes_h_in
        for c in range(len(col_labels)):
            tbl[r_i, c].set_height(row_h)

    cv_col_idx = col_labels.index("CV R²")
    b_col_idx  = col_labels.index("Boruta CV R²")
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
            continue
        if r % 2 == 0:
            cell.set_facecolor("#f0f4f8")
        cell.set_linewidth(0.4)
        if c == feat_col_idx:
            cell.get_text().set_ha("left")
            cell.PAD = 0.02
        for col_idx in (cv_col_idx, b_col_idx):
            if c == col_idx:
                txt = cell.get_text().get_text()
                try:
                    v = float(txt)
                    if v >= 0.5:
                        cell.set_facecolor("#c8e6c9")   # green
                    elif v >= 0.2:
                        cell.set_facecolor("#fff9c4")   # yellow
                    elif v < 0:
                        cell.set_facecolor("#ffcdd2")   # red
                except ValueError:
                    pass

    log_note = ("  SM target is log1p-transformed (extreme outliers up to 45,000 %) — "
                "CV R² values are on the log1p scale and directly comparable with Boruta."
                if target_key == "sm" else "")
    _caption(fig,
        "CV R² = mean cross-validation R² (3-fold).  "
        "Train R² = in-sample fit.  "
        "Δ = Train R² − CV R² (large Δ = overfitting).  "
        "n feats = features used after Boruta filtering.  "
        "Boruta CV R² = baseline CV R² from Boruta run (confirmed-only subset, ≤3000 rows).  "
        "B feats = confirmed + tentative features in Boruta.  "
        "Boruta features = confirmed features (tentative marked with (t)).  "
        "Colour: green ≥ 0.5 · yellow ≥ 0.2 · red < 0." + log_note)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# Page – feature detail table (confirmed / tentative / used per depth)
# ---------------------------------------------------------------------------

def _features_detail_page(trained_dict, type_labels, title, target_key):
    """One figure listing confirmed, tentative, and actually-used features per (et, depth)."""
    figs = []
    for et, (_, _, depth_results) in trained_dict.items():
        active = [dr for dr in depth_results if not dr["skip"]]
        if not active:
            continue

        et_lbl = type_labels.get(et, et)
        nrows  = len(active)
        fig, axes = plt.subplots(nrows, 1, figsize=(18, 5.5 * nrows + 1.5),
                                 squeeze=False)
        fig.suptitle(
            f"Feature Detail  ·  {et_lbl}  ({title})",
            fontsize=11, fontweight="bold",
        )

        for row_i, dr in enumerate(active):
            ax = axes[row_i, 0]
            ax.axis("off")

            b = _boruta_lookup(target_key, et, dr["dlbl"])
            confirmed  = sorted(b["confirmed"]) if b else []
            tentative  = sorted(b["tentative"]) if b else []
            used       = sorted(dr.get("feat_names", []))

            # Wrap long lists for display
            def _wrap(lst, width=90):
                return "\n".join(textwrap.wrap(", ".join(lst) if lst else "—", width))

            b_r2_str = (f"{b['r2_mean']:+.4f} ± {b['r2_std']:.4f}" if b
                        else "not available")
            cv_mean  = float(dr["r2s"].mean())
            cv_std   = float(dr["r2s"].std())

            # Features Boruta confirmed/tentative but not in training (excluded by hard rules)
            boruta_selected = set(confirmed) | set(tentative)
            used_set        = set(used)
            excluded_despite_boruta = sorted(boruta_selected - used_set)
            extra_in_training       = sorted(used_set - boruta_selected)

            text_lines = [
                f"Depth: {dr['dlbl']}    n = {dr['n_samp']:,}",
                f"Training CV R²: {cv_mean:+.4f} ± {cv_std:.4f}    "
                f"Train R²: {dr.get('r2_train', float('nan')):.4f}",
                f"Boruta CV R² (confirmed subset): {b_r2_str}",
                "",
                f"Boruta confirmed ({len(confirmed)}):  {_wrap(confirmed)}",
                f"Boruta tentative ({len(tentative)}):  {_wrap(tentative)}",
                f"Used in training ({len(used)}):       {_wrap(used)}",
            ]
            if excluded_despite_boruta:
                text_lines.append(
                    f"[!] Boruta-confirmed but excluded from training "
                    f"({len(excluded_despite_boruta)}): "
                    f"{_wrap(excluded_despite_boruta)}"
                )
            if extra_in_training:
                text_lines.append(
                    f"[+] Used in training but not in Boruta selection "
                    f"({len(extra_in_training)}): {_wrap(extra_in_training)}"
                )
            ax.text(0.02, 0.97, "\n".join(text_lines),
                    transform=ax.transAxes,
                    fontsize=8, va="top", ha="left", family="monospace",
                    bbox=dict(facecolor="#f8f9fa", edgecolor="#dee2e6",
                              boxstyle="round,pad=0.5"))
            ax.set_title(f"{dr['dlbl']}", fontsize=9, loc="left", pad=4)

        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        figs.append(fig)
    return figs


# ---------------------------------------------------------------------------
# Page – predicted vs actual + residual
# ---------------------------------------------------------------------------

def _pred_vs_actual_page(trained_dict, type_labels, target_label):
    """One figure per event type: rows=depth, cols=[pred-actual, residual]."""
    figs = []
    for et, (_, _, depth_results) in trained_dict.items():
        active = [dr for dr in depth_results if not dr["skip"]]
        if not active:
            continue
        if not any(dr.get("y_pred") is not None for dr in active):
            continue

        nrows = len(active)
        fig, axes = plt.subplots(nrows, 2, figsize=(12, 4.5 * nrows + 1.5),
                                 squeeze=False)
        fig.suptitle(
            f"Predicted vs Actual  ·  {type_labels.get(et, et)}\n"
            f"Target: {target_label}",
            fontsize=11, fontweight="bold",
        )

        for row_i, dr in enumerate(active):
            y      = dr["y"]
            y_pred = dr.get("y_pred")
            if y_pred is None:
                for ax in axes[row_i]:
                    ax.text(0.5, 0.5, "No y_pred in cache\n(re-run training)",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=9, color="#888")
                    ax.set_title(dr["dlbl"], fontsize=9)
                continue

            residuals = y - y_pred
            r2_cv = float(dr["r2s"].mean())
            r2_tr = dr.get("r2_train", np.nan)
            n     = dr["n_samp"]
            clr   = dr.get("dclr", "#555")

            # shared axis limits (90 % coverage of combined actual + predicted)
            combined = np.concatenate([y, y_pred])
            lo, hi = _axis_lim(combined)

            # ── Predicted vs Actual ──────────────────────────────────────────
            ax = axes[row_i, 0]
            ax.scatter(y_pred, y, s=7, alpha=0.35, color=clr, linewidths=0,
                       rasterized=True)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="1:1 line", zorder=3)
            ols = _ols_line(y_pred, y)
            if ols:
                xs = np.linspace(lo, hi, 200)
                ax.plot(xs, ols[0] * xs + ols[1], color="#e74c3c",
                        lw=1.0, alpha=0.85, label="OLS fit", zorder=4)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel(f"Predicted  [{target_label}]", fontsize=8)
            ax.set_ylabel(f"Actual  [{target_label}]", fontsize=8)
            ax.set_title(
                f"{dr['dlbl']}   n={n:,}\n"
                f"Train R²={r2_tr:.3f}   CV R²={r2_cv:.3f}",
                fontsize=8.5,
            )
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
            ax.tick_params(labelsize=7)
            # Interpretation note
            ax.text(0.98, 0.03,
                    "Points on 1:1 line = perfect prediction\n"
                    "Scatter width ∝ prediction error\n"
                    "OLS slope < 1 = regression to mean",
                    transform=ax.transAxes, fontsize=5.5, color="#555",
                    ha="right", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))

            # ── Residual plot ────────────────────────────────────────────────
            ax2 = axes[row_i, 1]
            res_lo, res_hi = _axis_lim(residuals)
            pred_lo, pred_hi = _axis_lim(y_pred)
            ax2.scatter(y_pred, residuals, s=7, alpha=0.35, color=clr,
                        linewidths=0, rasterized=True)
            ax2.axhline(0, color="k", lw=1.0, linestyle="--", zorder=3)
            ax2.set_xlim(pred_lo, pred_hi)
            ax2.set_ylim(res_lo, res_hi)
            ax2.set_xlabel(f"Predicted  [{target_label}]", fontsize=8)
            ax2.set_ylabel("Residual  (actual − predicted)", fontsize=8)
            ax2.set_title(f"{dr['dlbl']}   residuals", fontsize=8.5)
            ax2.tick_params(labelsize=7)
            ax2.text(0.98, 0.03,
                     "Ideal: random band around 0\n"
                     "Funnel shape → heteroscedasticity\n"
                     "Curve → unmodelled nonlinearity",
                     transform=ax2.transAxes, fontsize=5.5, color="#555",
                     ha="right", va="bottom",
                     bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))

        _caption(fig,
            "Axes clipped to central 90 % of data (5th–95th percentile) to suppress outlier distortion.  "
            "Dashed line = 1:1 reference (perfect prediction).  Red line = OLS trend.")
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        figs.append(fig)
    return figs


# ---------------------------------------------------------------------------
# Page – individual feature vs target scatter (top N features)
# ---------------------------------------------------------------------------

def _feature_scatter_pages(trained_dict, type_labels, target_label):
    """One page per event type: all depths overlaid with distinct colors per panel."""
    figs = []
    for et, (_, _, depth_results) in trained_dict.items():
        active = [dr for dr in depth_results if not dr["skip"]]
        if not active:
            continue

        # Collect per-depth data keyed by depth string ("-10", "-30", "-45")
        depth_data = {}
        for dr in active:
            depth_key = next((dk for dk, lbl in DEPTH_LBLS.items() if lbl == dr["dlbl"]), None)
            if depth_key is None:
                continue
            y_s = dr.get("y_shap")
            if y_s is None:
                y_s = dr.get("y")
            if y_s is None or len(y_s) != len(dr["X"]):
                continue
            depth_data[depth_key] = {
                "X": dr["X"],
                "y": y_s.astype(float),
                "feat_names": dr["feat_names"],
                "imp_m": dr["imp_m"],
            }

        if not depth_data:
            continue

        # Build a union of top features ranked by average importance across depths
        imp_accum = {}
        for dd in depth_data.values():
            for fname, imp in zip(dd["feat_names"], dd["imp_m"]):
                imp_accum[fname] = imp_accum.get(fname, 0.0) + imp
        top_features = sorted(imp_accum, key=imp_accum.get, reverse=True)[:N_TOP_FEAT_SCATTER]

        ncols = 3
        nrows = (len(top_features) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.5 * ncols, 4.5 * nrows + 1.8),
                                 squeeze=False)
        fig.suptitle(
            f"Feature vs Target  ·  {type_labels.get(et, et)}\n"
            f"Target: {target_label}   "
            f"(top {N_TOP_FEAT_SCATTER} features by avg. permutation importance across depths)",
            fontsize=10, fontweight="bold",
        )

        # Depth legend handles
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=DEPTH_COLORS.get(dk, "#888"),
                       markersize=6, label=DEPTH_LBLS.get(dk, dk))
            for dk in sorted(depth_data.keys())
        ]

        for k, fname in enumerate(top_features):
            ax = axes[k // ncols, k % ncols]
            short = fname.replace("tree_", "")

            all_x, all_y = [], []
            for dk, dd in sorted(depth_data.items()):
                if fname not in dd["feat_names"]:
                    continue
                fi  = dd["feat_names"].index(fname)
                xv  = dd["X"][:, fi].astype(float)
                yv  = dd["y"]
                msk = np.isfinite(xv) & np.isfinite(yv)
                if msk.sum() < 5:
                    continue
                clr = DEPTH_COLORS.get(dk, "#888")
                ax.scatter(xv[msk], yv[msk], s=6, alpha=0.30,
                           color=clr, linewidths=0, rasterized=True,
                           label=DEPTH_LBLS.get(dk, dk))
                # per-depth OLS line
                ols = _ols_line(xv[msk], yv[msk])
                if ols:
                    xl = _axis_lim(xv[msk])
                    xs = np.linspace(xl[0], xl[1], 200)
                    ax.plot(xs, ols[0] * xs + ols[1],
                            color=clr, lw=1.4, alpha=0.90)
                all_x.append(xv[msk])
                all_y.append(yv[msk])

            if not all_x:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
                ax.set_title(short, fontsize=7.5)
                continue

            all_x_cat = np.concatenate(all_x)
            all_y_cat = np.concatenate(all_y)
            ax.set_xlim(_axis_lim(all_x_cat))
            ax.set_ylim(_axis_lim(all_y_cat))
            ax.set_xlabel(short, fontsize=7)
            ax.set_ylabel(target_label, fontsize=7)
            ax.tick_params(labelsize=6)

            # Overall OLS across all depths for slope direction
            ols_all = _ols_line(all_x_cat, all_y_cat)
            r2_all  = _r2_ols(all_x_cat, all_y_cat)
            direction = ""
            if ols_all and abs(ols_all[0]) > 1e-9:
                direction = "  ↑" if ols_all[0] > 0 else "  ↓"
            avg_imp = imp_accum.get(fname, 0.0) / max(len(depth_data), 1)
            ax.set_title(f"{short}\nR²={r2_all:.3f}  imp={avg_imp:.4f}{direction}",
                         fontsize=7.5)

        # Hide unused panels and add shared legend
        for k in range(len(top_features), nrows * ncols):
            axes[k // ncols, k % ncols].axis("off")
        fig.legend(handles=legend_handles, loc="lower right",
                   bbox_to_anchor=(0.98, 0.02), fontsize=8,
                   title="Depth", title_fontsize=8, framealpha=0.8)

        _caption(fig,
            "Each panel overlays all measurement depths (colours in legend).  "
            "OLS line per depth shows trend direction.  "
            "R²/imp averaged across depths.  "
            "Axes clipped to central 90 % of values.")
        fig.tight_layout(rect=[0, 0.06, 1, 0.93])
        figs.append(fig)
    return figs


# ---------------------------------------------------------------------------
# Page – feature correlation heatmap
# ---------------------------------------------------------------------------

def _corr_heatmap_pages(trained_dict, type_labels):
    figs = []
    for et, (_, _, depth_results) in trained_dict.items():
        for dr in depth_results:
            if dr["skip"]:
                continue
            X_s        = dr["X"]
            feat_names = dr["feat_names"]
            if len(feat_names) < 2:
                continue

            df_feat = pd.DataFrame(X_s, columns=feat_names)
            corr    = df_feat.corr()

            # shorten labels: strip "tree_" prefix
            short_names = [n.replace("tree_", "") for n in feat_names]
            n = len(feat_names)
            fig_h = max(6, n * 0.42 + 1.5)
            fig, ax = plt.subplots(figsize=(fig_h + 1.5, fig_h))

            im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1,
                           aspect="auto")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            fs = max(4, 8 - n // 8)
            ax.set_xticklabels(short_names, rotation=90, fontsize=fs)
            ax.set_yticklabels(short_names, fontsize=fs)
            cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label("Pearson r", fontsize=8)
            cb.ax.tick_params(labelsize=7)
            ax.set_title(
                f"Feature Correlation  ·  {type_labels.get(et, et)} | {dr['dlbl']}",
                fontsize=9, fontweight="bold",
            )

            # annotate cells if n ≤ 20
            if n <= 20:
                for i in range(n):
                    for j in range(n):
                        v = corr.values[i, j]
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=5,
                                color="white" if abs(v) > 0.6 else "black")

            _caption(fig,
                "Pearson correlation between every pair of input features.  "
                "Red = positive, blue = negative.  "
                "Features in a strongly correlated cluster (|r| > 0.7) carry "
                "redundant information — the model may be unstable if many such clusters exist.  "
                "Diagonal is always 1 (self-correlation).")
            fig.tight_layout(rect=[0, 0.04, 1, 1])
            figs.append(fig)
    return figs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sm",  action="store_true", help="Soil moisture only")
    parser.add_argument("--dt",  action="store_true", help="Drying time only")
    parser.add_argument("--out", default=str(OUT_PDF), help="Output PDF path")
    args = parser.parse_args()

    use_sm = not args.dt
    use_dt = not args.sm

    caches = []
    if use_sm and SM_CACHE.exists():
        caches.append((SM_CACHE, "Soil Moisture", "log1p(ΔVWC % dyn. range)", SM_TYPE_LABELS, "sm"))
    elif use_sm:
        print(f"WARNING: soil moisture cache not found: {SM_CACHE}")
    if use_dt and DT_CACHE.exists():
        caches.append((DT_CACHE, "Drying Time", "dry_h (h)", DT_TYPE_LABELS, "dt"))
    elif use_dt:
        print(f"WARNING: drying time cache not found: {DT_CACHE}")

    if not caches:
        print("No cache files found. Run the training scripts first.")
        return

    out_path = Path(args.out)
    print(f"Writing diagnostic report → {out_path}")

    with PdfPages(out_path) as pdf:
        for cache_path, model_name, target_label, type_labels, target_key in caches:
            print(f"\n  Loading {cache_path.name} …")
            cache   = load_cache(cache_path)
            trained = cache["trained"]

            print("    R² overview table …")
            fig = _build_r2_table(
                trained, type_labels,
                f"{model_name} — R² Overview  (Training vs Cross-Validation vs Boruta)",
                target_key,
            )
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            print("    Feature detail pages …")
            for fig in _features_detail_page(trained, type_labels, model_name, target_key):
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            print("    Predicted vs actual / residual pages …")
            for fig in _pred_vs_actual_page(trained, type_labels, target_label):
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            print("    Feature vs target scatter pages …")
            for fig in _feature_scatter_pages(trained, type_labels, target_label):
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            print("    Feature correlation heatmaps …")
            for fig in _corr_heatmap_pages(trained, type_labels):
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"\nDone → {out_path}")


if __name__ == "__main__":
    main()
