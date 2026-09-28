"""
Generates fig_sensor_deployment.pdf (and .png) for the manuscript.

The figure shows two filled time series on the same axes:
  - Red area:  sensors installed (senInsDate recorded, not yet removed)
  - Blue area: sensors actively transmitting VWC data on that day
    (>= 2 depths with VWC > 2 %, qualifying active threshold)

The red region visible above the blue represents installed but
non-transmitting (offline) sensors.

Output
------
figures/paper/fig_sensor_deployment.pdf
figures/paper/fig_sensor_deployment.png
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import geopandas as gpd
from pathlib import Path
from datetime import date

# ── Match font style used across all manuscript figures ────────────────────────
import matplotlib.font_manager as _fm
_FONT = "Arial"

plt.rcParams.update({
    "font.family":      _FONT,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "font.size":         8,
    "font.weight":      "bold",
    "axes.titlesize":    8,
    "axes.titleweight": "bold",
    "axes.labelsize":    8,
    "axes.labelweight": "bold",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
})

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
GIS_DIR    = REPO_ROOT / "GISData"
TREES_DIR  = REPO_ROOT / "TreeTabularData" / "trees"
OUT_DIR    = SCRIPT_DIR / "paper"
OUT_DIR.mkdir(exist_ok=True)

# ── Constants (must match urban_tree_report.py) ────────────────────────────────
VWC_KEYS              = ["-10|ENV__SOIL__VWC", "-30|ENV__SOIL__VWC", "-45|ENV__SOIL__VWC"]
VWC_ACTIVE_THRESHOLD  = 2.0   # % — depth must exceed this to count as active
VWC_ACTIVE_MIN_DEPTHS = 2     # minimum depths above threshold for the day to count
PLOT_START            = pd.Timestamp("2023-04-01")

# Publication colours
C_TRANSMITTING = "#2E86AB"   # steel blue
C_INSTALLED    = "#D63030"   # muted red


# ── Step 1: load installation intervals from all shapefiles ───────────────────

def load_sensor_intervals() -> list[tuple[pd.Timestamp, pd.Timestamp | None]]:
    """
    Scan every treeLocations.shp and return a list of (ins_date, rmv_date or None)
    for every sensor that has a senInsDate.
    """
    intervals = []
    shp_paths = sorted(GIS_DIR.glob("*/VectorLayers/treeLocations.shp"))
    if not shp_paths:
        raise FileNotFoundError(
            f"No treeLocations.shp found under {GIS_DIR}. "
            "Run compute_landuse.py or add_sensor_dates.py first.")

    for shp in shp_paths:
        try:
            gdf = gpd.read_file(shp)
        except Exception as e:
            print(f"  Warning: could not read {shp}: {e}")
            continue

        for _, row in gdf.iterrows():
            ins_raw = row.get("senInsDate")
            if not pd.notna(ins_raw):
                continue
            ins_d = pd.Timestamp(ins_raw).normalize()

            rmv_candidates = []
            for col in ("senRmvDate", "cutDwnDate"):
                raw = row.get(col)
                if pd.notna(raw):
                    rmv_candidates.append(pd.Timestamp(raw).normalize())
            rmv_d = min(rmv_candidates) if rmv_candidates else None
            intervals.append((ins_d, rmv_d))

    print(f"  Loaded {len(intervals)} sensor installation intervals from "
          f"{len(shp_paths)} shapefile(s).")
    return intervals


# ── Step 2: build daily transmitting-sensor count from sensor_data.csv ────────

def build_daily_transmitting() -> pd.Series:
    """
    For each tree folder, find every calendar day on which at least
    VWC_ACTIVE_MIN_DEPTHS depth channels reported VWC > VWC_ACTIVE_THRESHOLD.
    Aggregate into a daily count across all sensors.
    """
    dirs = [d for d in TREES_DIR.iterdir() if d.is_dir()]
    print(f"  Scanning {len(dirs)} tree folders for VWC activity…", flush=True)

    all_daily: dict[str, set] = {}

    for i, d in enumerate(dirs, 1):
        if i % 100 == 0:
            print(f"    {i}/{len(dirs)}…", flush=True)
        csv = d / "sensor_data.csv"
        if not csv.exists():
            continue
        try:
            df = pd.read_csv(
                csv,
                parse_dates=["datetime"],
                usecols=lambda c: c in (["datetime"] + VWC_KEYS),
                low_memory=False,
            )
        except Exception:
            continue

        vwc_cols = [c for c in df.columns if c in VWC_KEYS]
        if not vwc_cols:
            continue

        df = df.dropna(subset=["datetime"])
        df["_has_vwc"] = df[vwc_cols].notna().any(axis=1)
        df_vwc = df[df["_has_vwc"]].copy()
        if df_vwc.empty:
            continue

        df_vwc["_date"] = df_vwc["datetime"].dt.date
        n_active = df_vwc[vwc_cols].gt(VWC_ACTIVE_THRESHOLD).sum(axis=1)
        min_d = min(VWC_ACTIVE_MIN_DEPTHS, len(vwc_cols))
        active_dates = set(df_vwc.loc[n_active >= min_d, "_date"].unique())
        all_daily[d.name] = active_dates

    if not all_daily:
        return pd.Series(dtype=int)

    all_dates: set = set()
    for ds in all_daily.values():
        all_dates |= ds
    date_range = pd.date_range(min(all_dates), max(all_dates), freq="D")

    counts = [
        sum(1 for ds in all_daily.values() if d.date() in ds)
        for d in date_range
    ]
    print(f"  Built daily counts over {len(date_range)} days.")
    return pd.Series(counts, index=date_range)


# ── Step 3: build daily installed count ───────────────────────────────────────

def build_daily_installed(intervals: list, date_range: pd.DatetimeIndex) -> pd.Series:
    counts = [
        sum(
            1 for ins_d, rmv_d in intervals
            if ins_d <= d and (rmv_d is None or d < rmv_d)
        )
        for d in date_range
    ]
    return pd.Series(counts, index=date_range)


# ── Step 4: plot ───────────────────────────────────────────────────────────────

def plot(daily_transmitting: pd.Series, daily_installed: pd.Series):
    # Clip to PLOT_START
    tr = daily_transmitting.loc[daily_transmitting.index >= PLOT_START]
    ins = daily_installed.loc[daily_installed.index >= PLOT_START]

    fig, ax = plt.subplots(figsize=(170 / 25.4, 70 / 25.4))

    # Installed (red, below)
    ax.fill_between(ins.index, ins.values,
                    color=C_INSTALLED, alpha=0.30, lw=0)
    ax.plot(ins.index, ins.values,
            color=C_INSTALLED, lw=0.9, alpha=0.75,
            label="Installed (not transmitting)")

    # Transmitting (blue, on top)
    ax.fill_between(tr.index, tr.values,
                    color=C_TRANSMITTING, alpha=0.55, lw=0)
    ax.plot(tr.index, tr.values,
            color=C_TRANSMITTING, lw=0.9, alpha=0.90,
            label="Transmitting VWC data")

    ax.set_xlim(left=PLOT_START, right=daily_transmitting.index.max())
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Number of sensors")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center",
             fontsize=8, fontweight="bold", fontfamily=_FONT)

    _fp = _fm.FontProperties(family=_FONT, size=8, weight="bold")
    ax.legend(loc="upper left", framealpha=0.85, prop=_fp)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.4)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Building fig_sensor_deployment…")

    print("\n[1/3] Loading installation intervals from shapefiles…")
    intervals = load_sensor_intervals()

    print("\n[2/3] Building daily transmitting-sensor counts…")
    daily_transmitting = build_daily_transmitting()

    if daily_transmitting.empty:
        raise RuntimeError("No VWC data found. Check TREES_DIR path.")

    print("\n[3/3] Building daily installed counts and plotting…")
    daily_installed = build_daily_installed(intervals, daily_transmitting.index)

    fig = plot(daily_transmitting, daily_installed)

    pdf_path = OUT_DIR / "fig_sensor_deployment.pdf"
    png_path = OUT_DIR / "fig_sensor_deployment.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved:")
    print(f"  {pdf_path.relative_to(REPO_ROOT)}")
    print(f"  {png_path.relative_to(REPO_ROOT)}")

    # Quick stats for the caption
    peak_tr  = int(daily_transmitting.max())
    peak_ins = int(daily_installed.max())
    t_start  = daily_transmitting.index.min().strftime("%B %Y")
    t_end    = daily_transmitting.index.max().strftime("%B %Y")
    print(f"\n  Peak transmitting: {peak_tr} sensors")
    print(f"  Peak installed:    {peak_ins} sensors")
    print(f"  Period:            {t_start} – {t_end}")


if __name__ == "__main__":
    main()
