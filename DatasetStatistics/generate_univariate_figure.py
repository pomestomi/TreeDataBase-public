"""
Generate fig_univariate_sm_6panel.pdf/.png
Six panels, ONE target (SM increase, delta_pct_dyn at 30 cm), BSC_XX11_Q12 stratum.
Features: crown diameter, greenAtta_0to7, TWI, TPI (7.5m), SVF, DTGW.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, linregress
import warnings
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TT = REPO / "TreeTabularData"
FIG_DIR = REPO / "figures" / "paper"
FIG_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df_rf = pd.read_csv(TT / "rf_dataset.csv", dtype={"eui": str}, low_memory=False)

# Coerce problem columns to numeric
for c in ["crownDiam_m", "twi", "tpi7m5", "skyViewFactor", "depthToGroundwater_m"]:
    df_rf[c] = pd.to_numeric(
        df_rf[c].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )

# Aggregate greenAtta_0to7
for c in ["greenAtta_2m5", "greenAtta_5m", "greenAtta_7m5"]:
    df_rf[c] = pd.to_numeric(df_rf[c], errors="coerce")
df_rf["greenAtta_0to7"] = df_rf[["greenAtta_2m5", "greenAtta_5m", "greenAtta_7m5"]].sum(
    axis=1, min_count=1
)

# Filter to SM stratum
bsc_s = df_rf["bsc"].astype(str).str.zfill(4)
sm_mask = (bsc_s.str[:2] == "11") & (df_rf["huff_q"].isin([1, 2]))
df_sm = df_rf[sm_mask].copy()
df_sm["delta_pct_dyn"] = pd.to_numeric(df_sm["delta_pct_dyn-10"], errors="coerce")
df_sm = df_sm[df_sm["delta_pct_dyn"] > 0].copy()
df_sm["target_log"] = np.log1p(df_sm["delta_pct_dyn"])

TARGET_LABEL = r"$\Delta$VWC (log$_{1+x}$ %, 30 cm)"

# Feature spec: (col, label, unit)
FEATURES = [
    ("crownDiam_m",          "Crown diameter",           "m"),
    ("greenAtta_0to7",       "Green cover 0–7 m",        "%"),
    ("twi",                  "TWI",                      "–"),
    ("tpi7m5",               "TPI (7.5 m)",              "–"),
    ("skyViewFactor",        "Sky-view factor",          "–"),
    ("depthToGroundwater_m", "Depth to groundwater",     "m"),
]

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

fig = plt.figure(figsize=(7.5, 5.0))
gs = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.36,
              left=0.07, right=0.97, top=0.93, bottom=0.10)

PANEL_LETTERS = list("abcdef")

for pi, (col, label, unit) in enumerate(FEATURES):
    ax = fig.add_subplot(gs[pi // 3, pi % 3])
    sub = df_sm[[col, "target_log"]].dropna()
    n = len(sub)
    x = sub[col].astype(float).values
    y = sub["target_log"].astype(float).values

    # Stats
    rho, p_rho = spearmanr(x, y)
    slp, intercept, r_lin, _, _ = linregress(x, y)
    r2 = r_lin ** 2
    sig_str = "***" if p_rho < 0.001 else ("**" if p_rho < 0.01 else ("*" if p_rho < 0.05 else ""))

    # Hexbin (coloured density)
    hb = ax.hexbin(x, y, gridsize=30, mincnt=1, linewidths=0.0,
                   cmap="YlOrRd", alpha=0.85)

    # Regression line (trimmed to data range)
    x_fit = np.linspace(np.percentile(x, 2), np.percentile(x, 98), 200)
    y_fit = intercept + slp * x_fit
    ax.plot(x_fit, y_fit, color="#2b4c7e", linewidth=1.2, zorder=5)

    # Axis labels
    xlabel = f"{label} ({unit})" if unit not in {"–", ""} else label
    ax.set_xlabel(xlabel)
    if pi % 3 == 0:
        ax.set_ylabel(TARGET_LABEL)

    # Panel letter + stats
    rho_str = f"$\\rho$ = {rho:+.3f}{sig_str}"
    r2_str  = f"$R^2$ = {r2:.3f}"
    n_str   = f"$n$ = {n:,}"
    ax.set_title(f"({PANEL_LETTERS[pi]}) {label}", loc="left", fontsize=8, pad=3)
    ax.text(0.97, 0.97, f"{rho_str}\n{r2_str}\n{n_str}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

fig.suptitle(
    r"Univariate associations — $\Delta$VWC rain response (BSC$_\mathrm{XX11}$ Q$_{1\text{–}2}$, 30 cm)",
    fontsize=8.5, y=0.99,
)

out_stem = FIG_DIR / "fig_univariate_sm_6panel"
fig.savefig(str(out_stem) + ".pdf", dpi=300, bbox_inches="tight")
fig.savefig(str(out_stem) + ".png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved {out_stem}.pdf / .png")

# ── Print stats table for report ─────────────────────────────────────────────
print("\n6-panel univariate stats (SM BSC_XX11_Q12, delta_pct_dyn at 30 cm):")
print(f"{'Feature':<28} {'n':>6} {'rho':>8} {'p':>10} {'R2':>8} {'slope':>10}")
for col, label, unit in FEATURES:
    sub = df_sm[[col, "target_log"]].dropna()
    n = len(sub)
    if n < 10:
        print(f"  {label:<26} {'n/a':>6}")
        continue
    x = sub[col].astype(float).values
    y = sub["target_log"].astype(float).values
    rho, p_rho = spearmanr(x, y)
    slp, intercept, r_lin, _, _ = linregress(x, y)
    r2 = r_lin ** 2
    sig = "***" if p_rho < 0.001 else ("**" if p_rho < 0.01 else ("*" if p_rho < 0.05 else "  "))
    print(f"  {label:<26} {n:>6} {rho:>+7.3f}{sig}  {p_rho:>10.2e}  {r2:>6.4f}  {slp:>+10.4f}")
