"""
Reproduce chart_bsc_quartiles with:
  - cmc.roma staggered palette (same as fig_pie_overview)
  - Arial 8 pt bold throughout
  - Luminance-aware label colour (white on dark, black on light)
  - No title
  - SM stratum (XX11, Q1+Q2): black dashed border
  - DT stratum (00XX, Q3+Q4): white dashed border
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm
import cmcrameri.cm as cmc
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO    = Path(__file__).resolve().parents[1]
TT      = REPO / "TreeTabularData"
FIG_DIR = REPO / "figures" / "paper"
FIG_DIR.mkdir(exist_ok=True)

# ── Font: Arial 8 pt bold ─────────────────────────────────────────────────────
# Prefer Arial; fall back gracefully to Helvetica / sans-serif
_FONT_PREF = ["Arial", "Helvetica", "DejaVu Sans"]
_available = {f.name for f in fm.fontManager.ttflist}
_font_name  = next((f for f in _FONT_PREF if f in _available), "sans-serif")

plt.rcParams.update({
    "font.family":     _font_name,
    "font.size":       8,
    "font.weight":     "bold",
    "axes.labelsize":  8,
    "axes.titlesize":  8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

# ── cmc.vik interleaved palette (same colormap as fig_corr_heatmap) ──────────
# 16 colors: 8 from the blue end + 8 from the warm end, interleaved so that
# adjacent BSC codes alternate cold/warm for maximum contrast.
_ALL_CODES = [f"{i:04b}" for i in range(16)]
_n_half    = len(_ALL_CODES) // 2                     # 8
_blues     = np.linspace(0.03, 0.35, _n_half)         # dark→medium blue
_warms     = np.linspace(0.65, 0.97, _n_half)         # medium→dark warm
_VIK_POS   = np.array([v for pair in zip(_blues, _warms) for v in pair])
_BSC_PALETTE = {code: cmc.vik(_VIK_POS[k]) for k, code in enumerate(_ALL_CODES)}

# Hatch: solid → dots → diagonal lines, cycling every 3 codes
def _hatch_for(k):
    r = k % 3
    if r == 1: return '..'
    if r == 2: return '//'
    return ''

_BSC_HATCH = {code: _hatch_for(k) for k, code in enumerate(_ALL_CODES)}

def _luminance(rgba):
    def lin(c): return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgba[:3]
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)

MIN_EVENT_MM = 5.0
DRY_GAP_H    = 6

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(TT / "bsc_events_all.csv", dtype={"eui": str}, low_memory=False)
df["bsc"] = df["bsc"].astype(str).str.zfill(4)

total  = len(df)
codes  = df["bsc"].value_counts().index.tolist()

q_sizes = [df["huff_q"].eq(q).sum() for q in range(1, 5)]
q_labels = [f"Quartile {r} ({q_sizes[r-1]/total*100:.1f} %)"
            for r in range(1, 5)]

matrix = np.zeros((len(codes), 4))
for i, code in enumerate(codes):
    for q in range(1, 5):
        mask = (df["bsc"] == code) & (df["huff_q"] == q)
        matrix[i, q - 1] = mask.sum() / total * 100

# ── Per-cell threshold + stack ordering ──────────────────────────────────────
OTHER_GREY = '#AAAAAA'
CELL_THR   = 1.0   # percent; per code×quartile cell

# Stack order: SM (XX11) → DT-only (00XX, not XX11) → rest ≥1% → Other (grey top)
def _is_sm(c):      return c[2:] == '11'
def _is_dt_only(c): return c[:2] == '00' and c[2:] != '11'

code_vals  = {c: matrix[i] for i, c in enumerate(codes)}
sm_order   = [c for c in codes if _is_sm(c)]
dt_order   = [c for c in codes if _is_dt_only(c)]
rest_order = [c for c in codes if not _is_sm(c) and not _is_dt_only(c)]
draw_order = sm_order + dt_order + rest_order

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(170 / 25.4, 3.6))
x      = np.arange(4)
bottom = np.zeros(4)

highlight_sm = []
highlight_dt = []
grey_per_q   = np.zeros(4)   # all below-threshold contributions → drawn at top

for code in draw_order:
    vals  = code_vals[code]
    rgba  = _BSC_PALETTE.get(code, (0.8, 0.8, 0.8, 1.0))
    hatch = _BSC_HATCH.get(code, '')

    indiv = np.where(vals >= CELL_THR, vals, 0.0)
    small = vals - indiv
    grey_per_q += small                     # accumulate; drawn at top after loop

    if indiv.any():
        bars = ax.bar(x, indiv, bottom=bottom, color=rgba,
                      label=code, edgecolor="white", linewidth=0.4,
                      hatch=hatch, zorder=2)

        for j, (bar, v) in enumerate(zip(bars, indiv)):
            if v >= 1.5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom[j] + v / 2,
                    f"{v:.1f} %",
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color="black",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                )
            if v > 0:
                seg = (bar.get_x(), bottom[j], bar.get_width(), v)
                if code[2:] == '11':        # XX11 → SM green, all quartiles
                    highlight_sm.append(seg)
                elif code[:2] == '00':      # 00XX (not XX11) → DT pink, all quartiles
                    highlight_dt.append(seg)

    bottom += indiv   # advance by coloured portion only; grey goes on top

# Draw all "Other" (grey) as a single block at the very top
if grey_per_q.any():
    bars_g = ax.bar(x, grey_per_q, bottom=bottom,
                    color=OTHER_GREY, edgecolor="white", linewidth=0.4, zorder=2)
    for j, (bar, v) in enumerate(zip(bars_g, grey_per_q)):
        if v >= 1.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom[j] + v / 2,
                f"{v:.1f} %",
                ha="center", va="center", fontsize=8,
                fontweight="bold", color="black",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )
    bottom += grey_per_q

# ── Highlight rectangles: merge adjacent same-stratum segments per column ─────
SM_COL = "#39FF14"   # neon green
DT_COL = "#FF10F0"   # neon pink

def _merge_highlight(segs):
    """One spanning box per x-column, covering all adjacent segments."""
    from collections import defaultdict
    cols = defaultdict(list)
    for (bx, by, bw, bh) in segs:
        cols[round(bx, 4)].append((by, by + bh, bw, bx))
    merged = []
    for _, entries in sorted(cols.items()):
        bx_val  = entries[0][3]
        bw_val  = entries[0][2]
        min_by  = min(e[0] for e in entries)
        max_top = max(e[1] for e in entries)
        merged.append((bx_val, min_by, bw_val, max_top - min_by))
    return merged

for (bx, by, bw, bh) in _merge_highlight(highlight_sm):
    ax.add_patch(mpatches.FancyBboxPatch(
        (bx, by), bw, bh,
        boxstyle="square,pad=0", linewidth=1.5,
        edgecolor=SM_COL, facecolor="none", linestyle="--", zorder=10,
    ))

for (bx, by, bw, bh) in _merge_highlight(highlight_dt):
    ax.add_patch(mpatches.FancyBboxPatch(
        (bx, by), bw, bh,
        boxstyle="square,pad=0", linewidth=1.8,
        edgecolor=DT_COL, facecolor="none", linestyle="--", zorder=10,
    ))

# ── Axes decoration ───────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(q_labels, fontsize=8, fontweight="bold")
ax.set_ylabel("Percentage of events [%]", fontsize=8, fontweight="bold")
ax.set_ylim(0, bottom.max() * 1.08)
ax.grid(axis="y", alpha=0.3, zorder=0)

# ── Single organized legend ───────────────────────────────────────────────────
# Layout: left column = SM stratum / DT stratum / Other (<1 %)
#         remaining columns = BSC codes in semantic order (SM → DT → rest)
# Achieved by building the handle list in row-major order so matplotlib's
# row-by-row ncol fill puts the three meta-entries in column 0.
import math
prop = {"family": _font_name, "size": 8, "weight": "bold"}

patch_sm = mpatches.Patch(edgecolor=SM_COL, facecolor="none", linewidth=1.5,
                           linestyle="--")
patch_dt = mpatches.Patch(edgecolor=DT_COL, facecolor="none", linewidth=1.8,
                           linestyle="--")
patch_ot = mpatches.Patch(facecolor=OTHER_GREY, edgecolor="white", linewidth=0.4)

handles_raw, labels_raw = ax.get_legend_handles_labels()
handle_map = {l: h for h, l in zip(handles_raw, labels_raw)}

# Column-major filling: items 0,1,2 land in column 0 → put SM/DT/Other first,
# then all BSC codes sorted numerically (0000→1111, matches colour assignment).
_sorted_visible = sorted(handle_map)
bsc_h = [handle_map[c] for c in _sorted_visible]
bsc_l = list(_sorted_visible)

n_rows    = 3                              # SM / DT / Other
n_bsc_col = math.ceil(len(bsc_h) / n_rows)
ncol      = 1 + n_bsc_col

# Pad so BSC block fills n_rows × n_bsc_col
_blank = mpatches.Patch(facecolor="none", edgecolor="none", linewidth=0)
while len(bsc_h) < n_rows * n_bsc_col:
    bsc_h.append(_blank)
    bsc_l.append("")

all_handles = [patch_sm, patch_dt, patch_ot] + bsc_h
all_labels  = ["SM stratum (XX11)", "DT stratum (00XX)", "Other (<1 %)"] + bsc_l

fig.legend(
    all_handles, all_labels,
    ncol=ncol,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    prop=prop,
    framealpha=0.9,
    handlelength=1.6,
    columnspacing=0.8,
    borderpad=0.4,
)

plt.tight_layout(rect=[0, 0.15, 1, 1])

out = FIG_DIR / "fig_event_shapes"
fig.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
fig.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}.pdf / .png  (font: {_font_name})")
