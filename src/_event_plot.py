"""
Shared comprehensive event-plot function.
Imported by _poc_irrigation_v2.py and plot_rain_examples.py.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
VWC_COLS   = ["-10|ENV__SOIL__VWC", "-30|ENV__SOIL__VWC", "-45|ENV__SOIL__VWC"]
VWC_LABELS = ["-10 cm", "-30 cm", "-45 cm"]
VWC_COLORS = ["#1b7837", "#762a83", "#e08214"]
RAIN_COL   = "MTB|ENV__ATMO__RAIN__DELTA"
TEMP_COL   = "MTB|ENV__ATMO__T"

TYPE_COLORS = {
    "IRRIGATION":       "#D63030",
    "RAIN_ALL":         "#9DC3E6",
    "RAIN_HEAD":        "#2E86AB",
    "RAIN_TAIL":        "#9B59B6",
    "RAIN_HEAD_IMPACT": "#1b7837",
    "RAIN_TAIL_DRY":    "#F18F01",
    "RAIN_FULL":        "#003f7f",
    "RAIN_GAP_DROP":    "#888888",
}

INNER_WIN_H = 12   # +-12 h concentration window
MID_WIN_H   = 24   # +-24 h reference window
OUTER_WIN_H = 36   # +-36 h outer concentration window
PLOT_WIN_H  = 72   # total display window each side


def _safe(v):
    """Return float or None — NaN-safe."""
    try:
        f = float(v)
        return None if (f != f) else f   # NaN check
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main drawing function
# ---------------------------------------------------------------------------
def draw_event_figure(
    eui: str,
    onset: pd.Timestamp,
    ev_type: str,
    row: dict,
    rs30: pd.DataFrame,
    rs1h: pd.DataFrame,
    rain_roll24: pd.Series,
    dr: dict,
) -> plt.Figure:
    """
    Build a comprehensive 3-panel event figure with all detected features.

    Panels
    ------
    0 (header bar)  — event type, EUI, onset, classification label
    1 (VWC panel)   — timeseries + peak arrow + drying arrow + time-to-peak
                      arrow + all metric annotations + window shading
    2 (rain panel)  — hourly bars + 24h sum + window shading + fraction text
    """
    clr   = TYPE_COLORS.get(ev_type, "black")
    t0    = onset - pd.Timedelta(hours=PLOT_WIN_H)
    t1    = onset + pd.Timedelta(hours=PLOT_WIN_H)

    fig   = plt.figure(figsize=(18, 10), layout="constrained")
    gs    = mgridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[0.06, 2.6, 1.3],
    )
    ax_hdr  = fig.add_subplot(gs[0])
    ax_vwc  = fig.add_subplot(gs[1])
    ax_rain = fig.add_subplot(gs[2], sharex=ax_vwc)

    # ── Header bar ───────────────────────────────────────────────────────────
    ax_hdr.axis("off")
    ax_hdr.set_facecolor(clr)
    fig.patch.set_facecolor("#f4f4f4")
    ax_hdr.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, transform=ax_hdr.transAxes,
        boxstyle="square,pad=0", facecolor=clr, edgecolor="none", zorder=0))
    ax_hdr.text(
        0.5, 0.5,
        f"  {ev_type}   |   {eui}   |   onset: {onset.strftime('%Y-%m-%d %H:%M')}  ",
        transform=ax_hdr.transAxes, fontsize=10, fontweight="bold",
        ha="center", va="center", color="white",
    )

    # ── Shared window shading (apply to both VWC and rain panels) ─────────────
    win_specs = [
        (OUTER_WIN_H, "#2E86AB", 0.06, f"±{OUTER_WIN_H} h outer"),
        (MID_WIN_H,   "#2E86AB", 0.10, f"±{MID_WIN_H} h"),
        (INNER_WIN_H, "#2E86AB", 0.16, f"±{INNER_WIN_H} h inner"),
    ]
    for ax in (ax_vwc, ax_rain):
        for h, c, a, _ in win_specs:
            ax.axvspan(onset - pd.Timedelta(hours=h),
                       onset + pd.Timedelta(hours=h),
                       color=c, alpha=a, zorder=0)

    # ── VWC panel ─────────────────────────────────────────────────────────────
    vwc_avail = [c for c in VWC_COLS if c in rs30.columns]

    y_lo_global = np.inf
    y_hi_global = -np.inf

    peak_markers   = {}   # depth → t_peak
    baseline_vals  = {}   # depth → pre_vwc

    for col, lbl, dc in zip(vwc_avail, VWC_LABELS, VWC_COLORS):
        depth = col.split("|")[0]
        sl = rs30.loc[t0:t1, col].dropna()
        if sl.empty:
            continue
        ax_vwc.plot(sl.index, sl.values, color=dc, lw=1.4, label=lbl, zorder=3)
        y_lo_global = min(y_lo_global, float(sl.min()))
        y_hi_global = max(y_hi_global, float(sl.max()))

        # Pre-event baseline
        pre_v = _safe(row.get(f"pre_vwc{depth}"))
        if pre_v is not None:
            ax_vwc.axhline(pre_v, color=dc, lw=0.9, linestyle=":",
                           alpha=0.65, zorder=2)
            baseline_vals[depth] = pre_v

        # Peak time + marker
        t_pk_min = _safe(row.get(f"t_peak_min{depth}"))
        peak_v   = _safe(row.get(f"peak_vwc{depth}"))
        if t_pk_min is not None and peak_v is not None:
            t_peak = onset + pd.Timedelta(minutes=t_pk_min)
            if t0 <= t_peak <= t1:
                ax_vwc.axvline(t_peak, color=dc, lw=0.8,
                               linestyle=":", alpha=0.55, zorder=2)
                ax_vwc.scatter([t_peak], [peak_v], color=dc, s=50,
                               zorder=5, marker="^")
                peak_markers[depth] = t_peak

    if y_lo_global == np.inf:
        y_lo_global, y_hi_global = 0, 10
    y_span = max(y_hi_global - y_lo_global, 2.0)

    # Vertical ↕ arrows for delta VWC
    arrow_kw = dict(arrowstyle="<->", lw=1.5, mutation_scale=12)
    for col, lbl, dc in zip(vwc_avail, VWC_LABELS, VWC_COLORS):
        depth = col.split("|")[0]
        dv    = _safe(row.get(f"delta_vwc{depth}"))
        dyn   = dr.get(depth, {}).get("dyn") if isinstance(dr, dict) else None
        pre_v = baseline_vals.get(depth)
        t_pk  = peak_markers.get(depth)
        if None in (dv, pre_v, t_pk) or abs(dv) < 0.1:
            continue
        peak_v = pre_v + dv
        ax_vwc.annotate(
            "", xy=(t_pk, peak_v), xytext=(t_pk, pre_v),
            arrowprops=dict(color=dc, **arrow_kw), zorder=6,
        )
        dyn_str = f" ({abs(dv)/dyn*100:.0f}%dyn)" if dyn and dyn > 0 else ""
        ax_vwc.text(
            t_pk + pd.Timedelta(hours=1.5),
            (pre_v + peak_v) / 2,
            f"{dv:+.2f}%{dyn_str}",
            color=dc, fontsize=7.5, fontweight="bold", va="center", zorder=7,
        )

    # Horizontal ↔ arrows: time-to-peak and drying time.
    # Each depth gets its own row so arrows never overlap.
    # Row layout (from just below data, downward):
    #   rows 0-2  →  time-to-peak  (-10, -30, -45)
    #   rows 3-5  →  drying time   (-10, -30, -45)
    row_step     = y_span * 0.13   # vertical spacing between rows
    arrow_top    = y_lo_global - y_span * 0.12  # first row baseline

    for row_idx, (col, dc) in enumerate(zip(vwc_avail, VWC_COLORS)):
        depth    = col.split("|")[0]
        t_pk     = peak_markers.get(depth)
        t_pk_min = _safe(row.get(f"t_peak_min{depth}"))
        dry_h    = _safe(row.get(f"dry_h{depth}"))

        # ── Time-to-peak arrow (rows 0-2) ────────────────────────────────
        y_tp = arrow_top - row_idx * row_step
        if t_pk is not None and t_pk_min is not None and t_pk_min > 30:
            ax_vwc.annotate(
                "", xy=(t_pk, y_tp), xytext=(onset, y_tp),
                arrowprops=dict(arrowstyle="<->", color=dc, lw=1.1,
                                mutation_scale=10),
                zorder=6,
            )
            ax_vwc.text(
                onset + (t_pk - onset) / 2, y_tp - row_step * 0.35,
                f"t_peak={t_pk_min:.0f}min",
                color=dc, fontsize=6.5, ha="center", va="top", zorder=7,
            )

        # ── Drying time arrow (rows 3-5, below the time-to-peak block) ───
        y_dry = arrow_top - (len(vwc_avail) + 0.4 + row_idx) * row_step
        if t_pk is not None and dry_h is not None:
            t_dry = t_pk + pd.Timedelta(hours=dry_h)
            if t_dry <= t1:
                ax_vwc.annotate(
                    "", xy=(t_dry, y_dry), xytext=(t_pk, y_dry),
                    arrowprops=dict(arrowstyle="<->", color=dc, lw=1.1,
                                    mutation_scale=10),
                    zorder=6,
                )
                ax_vwc.text(
                    t_pk + (t_dry - t_pk) / 2, y_dry - row_step * 0.35,
                    f"dry={dry_h:.0f}h",
                    color=dc, fontsize=6.5, ha="center", va="top", zorder=7,
                )
            else:
                ax_vwc.annotate(
                    "", xy=(t1, y_dry), xytext=(t_pk, y_dry),
                    arrowprops=dict(arrowstyle="-|>", color=dc, lw=1.0,
                                    mutation_scale=10),
                    zorder=6,
                )
                ax_vwc.text(
                    t1, y_dry - row_step * 0.35,
                    f"dry>{dry_h:.0f}h",
                    color=dc, fontsize=6.5, ha="right", va="top", zorder=7,
                )

    # Compute the lowest arrow row for ylim
    n_depths     = len(vwc_avail)
    arrow_y_dry  = arrow_top - (n_depths + 0.4 + (n_depths - 1)) * row_step

    # Window labels at top of VWC panel
    for h, c, a, lbl in win_specs:
        ax_vwc.text(
            onset + pd.Timedelta(hours=h * 0.92),
            y_hi_global + y_span * 0.02,
            lbl, color=c, fontsize=6.5, ha="right", va="bottom", alpha=0.85,
        )

    ax_vwc.axvline(onset, color=clr, lw=2.2, linestyle="--",
                   label="Onset", zorder=4)
    ax_vwc.set_ylim(arrow_y_dry - y_span * 0.10, y_hi_global + y_span * 0.18)
    ax_vwc.set_ylabel("VWC (%)", fontsize=9)
    ax_vwc.legend(fontsize=7.5, loc="upper left", ncol=6)
    ax_vwc.grid(alpha=0.25, zorder=1)
    plt.setp(ax_vwc.get_xticklabels(), visible=False)

    # Metric / feature text box
    lines = []
    # Concentration fractions (from row — these are stored in rain_concentration.csv,
    # not directly in v2 CSV, so use rain pre/post amounts to estimate)
    for tag, pre_k, post_k, outer_k_pre, outer_k_post in [
        ("±12h/±36h",
         "rain_pre24h-10", "rain_post24h-10",
         "rain_pre72h-10", "rain_post72h-10"),
    ]:
        ip   = _safe(row.get(pre_k))
        ipost = _safe(row.get(post_k))
        op   = _safe(row.get(outer_k_pre))
        opost = _safe(row.get(outer_k_post))
        if ip is not None and op is not None and op > 0:
            lines.append(f"frac_pre  {tag}: {ip/op:.2f}  ({ip:.1f}/{op:.1f}mm)")
        if ipost is not None and opost is not None and opost > 0:
            lines.append(f"frac_post {tag}: {ipost/opost:.2f}  ({ipost:.1f}/{opost:.1f}mm)")
    for depth in ("-10", "-30", "-45"):
        dv  = _safe(row.get(f"delta_vwc{depth}"))
        dyn = dr.get(depth, {}).get("dyn") if isinstance(dr, dict) else None
        dry = _safe(row.get(f"dry_h{depth}"))
        if dv is not None:
            dyn_s = f" ({abs(dv)/dyn*100:.0f}%dyn)" if dyn and dyn > 0 else ""
            dry_s = f"  dry={dry:.0f}h" if dry is not None else ""
            lines.append(f"Δ{depth}cm: {dv:+.2f}%{dyn_s}{dry_s}")

    if lines:
        ax_vwc.text(
            0.995, 0.99, "\n".join(lines),
            transform=ax_vwc.transAxes, fontsize=7.5,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.92),
        )

    # ── Rain panel ────────────────────────────────────────────────────────────
    rain_sl = rs1h[RAIN_COL].fillna(0).loc[t0:t1] \
              if RAIN_COL in rs1h.columns else pd.Series(dtype=float)
    roll_sl = rain_roll24.loc[t0:t1]

    if not rain_sl.empty:
        ax_rain.bar(rain_sl.index, rain_sl.values,
                    width=pd.Timedelta(hours=1), color="steelblue",
                    alpha=0.55, label="Hourly rain (mm/h)", zorder=3)
    ax_rain.plot(roll_sl.index, roll_sl.values, color="navy", lw=1.2,
                 linestyle="--", alpha=0.8, label="24 h sum (mm)", zorder=3)
    ax_rain.axvline(onset, color=clr, lw=2.2, linestyle="--", zorder=4)

    # Temperature twin axis
    if TEMP_COL in rs1h.columns:
        temp_sl = rs1h[TEMP_COL].loc[t0:t1].dropna()
        if not temp_sl.empty:
            ax_t = ax_rain.twinx()
            ax_t.plot(temp_sl.index, temp_sl.values, color="tomato",
                      lw=0.9, alpha=0.7, label="Temp (°C)")
            ax_t.set_ylabel("Temp (°C)", color="tomato", fontsize=8)
            ax_t.tick_params(axis="y", labelcolor="tomato", labelsize=7)
            ax_t.legend(fontsize=7, loc="upper right")

    # Annotate rain amounts from row
    rain_ann = []
    for k, lbl in [("rain_pre24h-10",  "pre ±12h"),
                   ("rain_post24h-10", "post ±12h"),
                   ("rain_pre72h-10",  "pre ±36h"),
                   ("rain_post72h-10", "post ±36h")]:
        v = _safe(row.get(k))
        if v is not None:
            rain_ann.append(f"{lbl}: {v:.1f}mm")
    if rain_ann:
        ax_rain.text(
            0.995, 0.97, "\n".join(rain_ann),
            transform=ax_rain.transAxes, fontsize=7.5,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.92),
        )

    # Window labels
    for h, c, a, lbl in win_specs:
        ax_rain.text(
            onset + pd.Timedelta(hours=h * 0.92),
            ax_rain.get_ylim()[1] * 0.95 if ax_rain.get_ylim()[1] > 0
            else 0.1,
            lbl, color=c, fontsize=6.5, ha="right", va="top", alpha=0.85,
        )

    ax_rain.set_ylabel("Rain (mm)", fontsize=9)
    ax_rain.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax_rain.grid(alpha=0.25, zorder=1)
    ax_rain.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax_rain.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax_rain.tick_params(axis="x", labelsize=7)

    # Enforce fixed ±72 h window AFTER all annotations — prevents arrows or
    # text from expanding the x-axis beyond the intended range.
    ax_vwc.set_xlim(t0, t1)
    ax_rain.set_xlim(t0, t1)

    return fig
