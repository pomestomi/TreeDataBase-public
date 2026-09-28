"""
Publication figure: fig_pie_overview  (v2, 2026-08)

Three-panel donut pie — two-column journal width (183 mm).

  (a) IRRIGATION events per city
  (b) BSC rain events per city  [two-shade: XX11 lighter / 00XX darker]
  (c) Cumulative sensor-active days per city

ONE shared city legend below all three panels.

Palette  : tab20 + tab20b (matching SHAP plots in generate_figures.py).
           Colours assigned by city position in sorted(city_label) across all
           48 city labels. Erlangen uses tab20 green (#2ca02c, idx 4) instead
           of its natural light-grey (idx 15) to distinguish it from "Other".
           Cities ranked by total BSC events; top 8 get individual colours.
           All other cities are grouped as "Other" (grey) in every panel.
BSC masks: SM model — last two chars "11", Huff Q1/Q2  (front-loaded rain)
           DT model — first two chars "00", Huff Q3/Q4  (back-loaded rain)

Outputs  :
  TreeTabularData/_report_assets/fig_pie_overview.pdf   ← replaces old version
  TreeTabularData/_report_assets/fig_pie_overview.png
  TreeTabularData/_report_assets/fig_pie_overview_table.csv
  TreeTabularData/_report_assets/fig_pie_overview_table.tex
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import defaultdict
from pathlib import Path

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

REPO      = Path(__file__).resolve().parent.parent
GIS_DIR   = REPO / "GISData"
OUT_DIR   = REPO / "TreeTabularData" / "_report_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IRR_V2    = REPO / "TreeTabularData" / "irrigation_events_v2_all.csv"
IRR_V1    = REPO / "TreeTabularData" / "irrigation_events_all.csv"
BSC_CSV   = REPO / "TreeTabularData" / "rf_dataset.csv"
TREES_DIR = REPO / "TreeTabularData" / "trees"

# ── Palette: tab20 + tab20b (matching generate_figures.py SHAP plots) ─────────
# Colours from plt.cm.tab20.colors + plt.cm.tab20b.colors, indexed by position
# of each city in sorted(city_label) across all 48 cities.
# Exception: Erlangen's natural index-15 colour (#c7c7c7, light grey) is
# replaced with tab20 index-4 green (#2ca02c) to distinguish it from "Other".
_SHAP_CITY_COLORS = {
    "Potsdam":   "#1f77b4",  # tab20 idx 40 (= 0 mod 40) — blue
    "Erlangen":  "#2ca02c",  # tab20 idx  4 (exception: natural idx-15 = #c7c7c7)
    "Hagen":     "#393b79",  # tab20b idx 20 — dark blue-purple
    "Bamberg":   "#aec7e8",  # tab20 idx  1 — light blue
    "Berlin":    "#ff7f0e",  # tab20 idx  2 (Berlin-Charlottenburg) — orange
    "Garbsen":   "#17becf",  # tab20 idx 18 — cyan
    "Nürnberg":  "#d6616b",  # tab20b idx 34 — salmon red
    "Salzburg":  "#ffbb78",  # tab20 idx  3 (Salzburg Uni) — peach
}
_OTHER_COLOR = "#AAAAAA"

# Number of cities assigned individual colours (= top N by BSC total)
N_NAMED = 8

# Within-panel minimum slice size to avoid near-invisible labeled slivers
# (named cities below this % of panel total go into "Other" for that panel)
IRR_MIN_PCT  = 0.8    # ~1% — suppresses Salzburg's 7 IRR events
SDAY_MIN_PCT = 0.5    # effectively keeps all top-8 cities in sensor-days panel

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.weight":     "bold",
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.titleweight": "bold",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# ── Project → canonical city ──────────────────────────────────────────────────
_PROJECT_TO_CITY = {
    "Botanical Garden Bremen":          "Bremen",
    "Botanical Garden Vienna":          "Vienna",
    "City Aachen":                      "Aachen",
    "City Bamberg":                     "Bamberg",
    "City Berlin Charlottenburg":       "Berlin",
    "City Berlin Frh XBerg":            "Berlin",
    "City Berlin Neukölln":             "Berlin",
    "City Biberach":                    "Biberach",
    "City Celle":                       "Celle",
    "City Crailsheim":                  "Crailsheim",
    "City Erlangen":                    "Erlangen",
    "City Garbsen":                     "Garbsen",
    "City Grünwald":                    "Grünwald",
    "City Hagen":                       "Hagen",
    "City Hanover":                     "Hannover",
    "City Hassfurt":                    "Hassfurt",
    "City Hildesheim":                  "Hildesheim",
    "City Ingolstadt":                  "Ingolstadt",
    "City Kassel":                      "Kassel",
    "City Leipzig":                     "Leipzig",
    "City Lemgo":                       "Lemgo",
    "City Luedenscheid":                "Lüdenscheid",
    "City Neunkirchen":                 "Neunkirchen",
    "City Nuernberg":                   "Nürnberg",
    "City Pforzheim":                   "Pforzheim",
    "City Pirmasens":                   "Pirmasens",
    "City Potsdam":                     "Potsdam",
    "City Stein":                       "Nürnberg",
    "City Vienna":                      "Vienna",
    "City Weisendorf":                  "Weisendorf",
    "Company Bremen":                   "Bremen",
    "Company Effeltrich":               "Effeltrich",
    "Company Erfurt":                   "Erfurt",
    "Company Erlangen":                 "Erlangen",
    "Company Heidelberg":               "Heidelberg",
    "Company Homburg":                  "Homburg",
    "Company Laax":                     "Laax",
    "Company Nörvenich":                "Aachen",
    "Company Pillnitz":                 "Pillnitz",
    "Company Plön":                     "Plön",
    "Company Saarbrücken":              "Saarbrücken",
    "Company Vienna":                   "Vienna",
    "Schlösserland Sachsen":            "Dresden",
    "University Erlangen Green Office": "Erlangen",
    "University Salzburg":              "Salzburg",
}


def _proj_to_city(proj):
    if pd.isna(proj): return None
    p = str(proj).strip()
    c = _PROJECT_TO_CITY.get(p)
    if c is None:
        for pfx in ("City ", "Company ", "Botanical Garden ", "University "):
            if p.startswith(pfx):
                c = p[len(pfx):]
                break
    return c or p


def _shade(hex_c, f):
    """f < 1 tints toward white; f > 1 shades toward black."""
    h = hex_c.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    if f <= 1.0:
        t = 1.0 - f
        r2, g2, b2 = int(r+(255-r)*t), int(g+(255-g)*t), int(b+(255-b)*t)
    else:
        t = f - 1.0
        r2, g2, b2 = int(r*(1-t)), int(g*(1-t)), int(b*(1-t))
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, r2)), max(0, min(255, g2)), max(0, min(255, b2)))


def _lum(hex_c):
    h = hex_c.lstrip("#")
    def lin(c): return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
    r, g, b = int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255
    return 0.2126*lin(r) + 0.7152*lin(g) + 0.0722*lin(b)


# ── Data loaders ──────────────────────────────────────────────────────────────
def load_eui_city():
    gdfs = []
    for shp in sorted(GIS_DIR.glob("*/VectorLayers/treeLocations.shp")):
        try: gdfs.append(gpd.read_file(shp))
        except Exception: pass
    gdf = pd.concat(gdfs, ignore_index=True)
    eui_col = "devEUI" if "devEUI" in gdf.columns else "EUI"
    out = {}
    for _, row in gdf.iterrows():
        eui  = str(row.get(eui_col, "")).strip().upper()
        city = _proj_to_city(row.get("project", ""))
        if eui and city:
            out[eui] = city
    return out


def load_irr_by_city(eui_city):
    p = IRR_V2 if IRR_V2.exists() else IRR_V1
    df = pd.read_csv(p, dtype={"eui": str}, low_memory=False)
    if "event_type" in df.columns:
        df = df[df["event_type"] == "IRRIGATION"]
    elif "label" in df.columns:
        df = df[df["label"] == "IRRIGATION"]
    df["city"] = df["eui"].str.strip().str.upper().map(eui_city)
    df = df[df["city"].notna()]
    return df.groupby("city").size()


def load_bsc_split(eui_city):
    """Returns (bsc_df, n_xx11_raw, n_00xx_raw)."""
    bsc = pd.read_csv(BSC_CSV, usecols=["eui", "bsc", "huff_q"],
                      dtype={"eui": str}, low_memory=False)
    bsc_s = bsc["bsc"].astype(str).str.zfill(4)
    m11 = (bsc_s.str[2:] == "11") & bsc["huff_q"].isin([1, 2])
    m00 = (bsc_s.str[:2] == "00") & bsc["huff_q"].isin([3, 4])
    sub = bsc[m11 | m00].copy()
    sub["city"] = sub["eui"].str.strip().str.upper().map(eui_city)
    sub = sub[sub["city"].notna()].copy()
    # Mutually exclusive masks (different huff_q ranges)
    sub["type"] = np.where(m00.loc[sub.index].values, "00XX", "XX11")
    return sub[["eui", "city", "type"]], int(m11.sum()), int(m00.sum())


def load_sensor_days(eui_city):
    dirs = [d for d in TREES_DIR.iterdir() if d.is_dir()]
    result = defaultdict(int)
    print(f"  Scanning {len(dirs)} tree folders for sensor-data days …", flush=True)
    for i, d in enumerate(dirs, 1):
        if i % 100 == 0:
            print(f"    {i}/{len(dirs)} …", flush=True)
        csv = d / "sensor_data.csv"
        if not csv.exists():
            continue
        try:
            df = pd.read_csv(csv, low_memory=False)
        except Exception:
            continue
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        if df.empty:
            continue
        vwc = [c for c in df.columns if "VWC" in c.upper()]
        if not vwc:
            continue
        active = df[df[vwc].gt(2.0).any(axis=1)]["datetime"].dt.date.nunique()
        city = eui_city.get(d.name.upper())
        if city:
            result[city] += active
    return pd.Series(result).sort_values(ascending=False)


# ── Slice builders ─────────────────────────────────────────────────────────────
def simple_slices(counts, city_order, city_color, min_pct=0.0):
    """
    Return (sizes, colors) for a single-colour donut.
    Named cities (city_order) → individual coloured slices, sorted by count desc.
    Named cities below min_pct of panel total → merged into Other.
    All non-named cities → Other (appended last).
    """
    total = float(counts.sum()) if len(counts) else 1.0
    named_data = []
    other_n    = 0

    for city in city_order:
        n   = int(counts.get(city, 0))
        pct = 100.0 * n / total if total > 0 else 0.0
        if n > 0 and pct >= min_pct:
            named_data.append((n, city_color[city]))
        else:
            other_n += n

    named_data.sort(key=lambda x: -x[0])   # size descending
    # Non-named cities
    other_n += int(counts[~counts.index.isin(set(city_order))].sum())

    sizes  = [d[0] for d in named_data]
    colors = [d[1] for d in named_data]
    if other_n > 0:
        sizes.append(other_n)
        colors.append(_OTHER_COLOR)
    return sizes, colors


def split_slices(bsc_df, city_order, city_color):
    """
    BSC split: top-N named cities ordered by total BSC count (city_order order).
    Within each city: XX11 (lighter shade) then 00XX (darker shade).
    Non-named cities → two grey Other slices at end.
    """
    c11 = bsc_df[bsc_df["type"] == "XX11"].groupby("city").size().to_dict()
    c00 = bsc_df[bsc_df["type"] == "00XX"].groupby("city").size().to_dict()

    sizes, colors = [], []
    for city in city_order:
        base = city_color[city]
        n1   = c11.get(city, 0)
        n2   = c00.get(city, 0)
        if n1 > 0:
            sizes.append(n1)
            colors.append(_shade(base, 0.72))   # lighter = XX11
        if n2 > 0:
            sizes.append(n2)
            colors.append(_shade(base, 1.30))   # darker  = 00XX

    named_set = set(city_order)
    o11 = sum(c11.get(c, 0) for c in c11 if c not in named_set)
    o00 = sum(c00.get(c, 0) for c in c00 if c not in named_set)
    if o11 > 0:
        sizes.append(o11)
        colors.append(_shade(_OTHER_COLOR, 0.80))
    if o00 > 0:
        sizes.append(o00)
        colors.append(_shade(_OTHER_COLOR, 1.18))
    return sizes, colors


# ── Donut drawing ──────────────────────────────────────────────────────────────
def draw_donut(ax, sizes, colors, n_total, title, pct_min=5.0):
    """Ring donut with luminance-aware % labels and N in the centre hole."""
    wedges, _, autotexts = ax.pie(
        sizes, labels=None,
        autopct=lambda p: f"{p:.0f}%" if p >= pct_min else "",
        colors=colors,
        startangle=90,       # 12 o'clock
        counterclock=False,  # clockwise
        pctdistance=0.78,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white", "width": 0.55},
    )
    for wedge, at in zip(wedges, autotexts):
        if at.get_text():
            fc  = wedge.get_facecolor()
            hx  = "#{:02x}{:02x}{:02x}".format(
                int(fc[0]*255), int(fc[1]*255), int(fc[2]*255))
            at.set_color("white" if _lum(hx) < 0.40 else "#111111")
            at.set_fontsize(8)
            at.set_fontweight("bold")
    ax.text(0, 0, f"N = {n_total:,}",
            ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_title(title, fontsize=8, fontweight="bold", pad=6)


# ── CSV + LaTeX table ──────────────────────────────────────────────────────────
def export_table(irr_by_city, c11_by_city, c00_by_city, sensor_days):
    all_cities = (set(irr_by_city.index) | set(c11_by_city) |
                  set(c00_by_city) | set(sensor_days.index))
    rows = []
    for c in all_cities:
        irr  = int(irr_by_city.get(c, 0))
        xx11 = int(c11_by_city.get(c, 0))
        d00  = int(c00_by_city.get(c, 0))
        bsc  = xx11 + d00
        sday = int(sensor_days.get(c, 0))
        rows.append({"city": c, "irr": irr, "bsc_xx11": xx11,
                     "bsc_00xx": d00, "bsc_total": bsc, "sensor_days": sday})

    df = (pd.DataFrame(rows)
            .sort_values("bsc_total", ascending=False)
            .reset_index(drop=True))

    n_irr  = int(irr_by_city.sum())
    n_xx11 = sum(c11_by_city.values())
    n_00xx = sum(c00_by_city.values())
    n_bsc  = n_xx11 + n_00xx
    n_sday = int(sensor_days.sum())

    df["irr_pct"] = (df["irr"] / n_irr * 100).round(1)
    df["bsc_pct"] = (df["bsc_total"] / n_bsc * 100).round(1)

    total_row = pd.DataFrame([{
        "city": "Total",
        "irr": n_irr, "bsc_xx11": n_xx11, "bsc_00xx": n_00xx,
        "bsc_total": n_bsc, "sensor_days": n_sday,
        "irr_pct": 100.0, "bsc_pct": 100.0,
    }])
    df_out = pd.concat([df, total_row], ignore_index=True)

    csv_path = OUT_DIR / "fig_pie_overview_table.csv"
    df_out.to_csv(csv_path, index=False, float_format="%.1f")
    print(f"  Saved → {csv_path.name}")

    # LaTeX tabular
    _esc = str.maketrans({"ü": r'\"u', "ä": r'\"a', "ö": r'\"o'})
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (r"City & Irr.\ events & BSC XX11 & BSC 00XX & BSC total"
         r" & Sensor-days & Irr.\,\% \\"),
        r"\midrule",
    ]
    for _, row in df_out.iterrows():
        city = row["city"].translate(_esc)
        if city == "Total":
            lines.append(r"\midrule")
        lines.append(
            f"{city} & {int(row['irr']):,} & {int(row['bsc_xx11']):,} & "
            f"{int(row['bsc_00xx']):,} & {int(row['bsc_total']):,} & "
            f"{int(row['sensor_days']):,} & {row['irr_pct']:.1f}\\%\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]

    tex_path = OUT_DIR / "fig_pie_overview_table.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved → {tex_path.name}")
    return df_out


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("[1] Building EUI → city map …")
    eui_city = load_eui_city()
    print(f"  {len(eui_city):,} EUI entries mapped")

    print("[2] Irrigation events …")
    irr_by_city = load_irr_by_city(eui_city)
    n_irr = int(irr_by_city.sum())
    print(f"  Total IRRIGATION events (mapped): {n_irr:,}")

    print("[3] BSC rain events (correct strata masks) …")
    bsc_df, n_xx11_raw, n_00xx_raw = load_bsc_split(eui_city)
    n_bsc_mapped = len(bsc_df)
    print(f"  SM XX11_Q12 (dataset total): {n_xx11_raw:,}   "
          f"DT 00XX_Q34 (dataset total): {n_00xx_raw:,}   "
          f"Mapped to city: {n_bsc_mapped:,}")
    c11_by_city = bsc_df[bsc_df["type"] == "XX11"].groupby("city").size().to_dict()
    c00_by_city = bsc_df[bsc_df["type"] == "00XX"].groupby("city").size().to_dict()

    print("[4] Sensor-data days …")
    sensor_days = load_sensor_days(eui_city)
    n_sdays = int(sensor_days.sum())
    print(f"  Total sensor-data days: {n_sdays:,}")

    # ── Top-N city selection (BSC-ranked, matches Okabe-Ito palette) ──────────
    bsc_total_by_city = pd.Series(
        {c: c11_by_city.get(c, 0) + c00_by_city.get(c, 0)
         for c in set(c11_by_city) | set(c00_by_city)}
    ).sort_values(ascending=False)

    city_order = bsc_total_by_city.index.tolist()[:N_NAMED]
    city_color = {c: _SHAP_CITY_COLORS.get(c, _OTHER_COLOR) for c in city_order}
    bsc_thresh_value = int(bsc_total_by_city.iloc[N_NAMED - 1])

    print(f"\n  Top-{N_NAMED} cities by BSC events (BSC rank threshold ≥ {bsc_thresh_value}):")
    for i, c in enumerate(city_order):
        tot  = int(bsc_total_by_city[c])
        col  = city_color[c]
        irr  = int(irr_by_city.get(c, 0))
        sday = int(sensor_days.get(c, 0))
        print(f"    [{i+1}] {c:<15} BSC={tot:>5,}  IRR={irr:>4,}  days={sday:>6,}  {col}")

    # ── Slice data ────────────────────────────────────────────────────────────
    irr_sizes, irr_colors = simple_slices(
        irr_by_city, city_order, city_color, min_pct=IRR_MIN_PCT)
    sd_sizes, sd_colors   = simple_slices(
        sensor_days, city_order, city_color, min_pct=SDAY_MIN_PCT)
    bsc_by_city = bsc_df.groupby("city").size()
    bsc_sizes, bsc_colors = simple_slices(
        bsc_by_city, city_order, city_color, min_pct=0.0)

    # ── Figure ────────────────────────────────────────────────────────────────
    FW = 170 / 25.4   # 170 mm
    FH = 75  / 25.4   # height sized to panels; legend captured by bbox_inches=tight

    fig = plt.figure(figsize=(FW, FH))
    gs  = gridspec.GridSpec(
        1, 3, figure=fig,
        left=0.01, right=0.99,
        top=0.91, bottom=0.01,
        wspace=0.04,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    draw_donut(ax_a, irr_sizes, irr_colors, n_irr,
               "(a) Irrigation events per city",
               pct_min=5.0)
    draw_donut(ax_b, bsc_sizes, bsc_colors, n_bsc_mapped,
               "(b) BSC rain events per city",
               pct_min=6.0)
    draw_donut(ax_c, sd_sizes, sd_colors, n_sdays,
               "(c) Sensor-active days per city",
               pct_min=5.0)

    # ── Shared city legend — placed below panels via fig.legend ──────────────
    patches = [
        mpatches.Patch(facecolor=city_color[c], label=c,
                       linewidth=0.3, edgecolor="white")
        for c in city_order
    ]
    patches.append(mpatches.Patch(facecolor=_OTHER_COLOR, label="Other",
                                  linewidth=0.3, edgecolor="white"))

    from matplotlib.font_manager import FontProperties
    fp = FontProperties(family="Arial", weight="bold", size=8)
    fig.legend(
        handles=patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        bbox_transform=fig.transFigure,
        ncol=len(patches),
        prop=fp,
        framealpha=0.0,
        handlelength=1.0, handleheight=0.65,
        handletextpad=0.35, columnspacing=0.75,
        labelspacing=0.20, borderpad=0.0,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_pdf = OUT_DIR / "fig_pie_overview.pdf"
    out_png = OUT_DIR / "fig_pie_overview.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved:\n  {out_pdf}\n  {out_png}")

    # ── Table ─────────────────────────────────────────────────────────────────
    print("[5] Exporting supplementary table …")
    export_table(irr_by_city, c11_by_city, c00_by_city, sensor_days)

    # ── Caption / methods numbers ──────────────────────────────────────────────
    print(f"\nCaption / methods numbers:")
    print(f"  (a) Irrigation events:    {n_irr:,}")
    print(f"  (b) BSC total (mapped):   {n_bsc_mapped:,}  "
          f"[SM XX11_Q12: {n_xx11_raw:,}  /  DT 00XX_Q34: {n_00xx_raw:,}]")
    print(f"  (c) Sensor-active days:   {n_sdays:,}")
    print(f"\n  Named cities (top-{N_NAMED} by BSC): {', '.join(city_order)}")
    print(f"  BSC events in named cities: "
          f"{int(bsc_total_by_city[city_order].sum()):,} / {int(bsc_total_by_city.sum()):,} "
          f"({100*bsc_total_by_city[city_order].sum()/bsc_total_by_city.sum():.1f}%)")

    # ── Front / back-loaded split text ────────────────────────────────────────
    n_xx11_mapped = int(bsc_df[bsc_df["type"] == "XX11"].shape[0])
    n_00xx_mapped = int(bsc_df[bsc_df["type"] == "00XX"].shape[0])
    n_bsc_m = n_xx11_mapped + n_00xx_mapped
    pct_xx11 = 100.0 * n_xx11_mapped / n_bsc_m
    pct_00xx = 100.0 * n_00xx_mapped / n_bsc_m

    print(f"\n── BSC front/back-loaded split (for in-text description) ──────────")
    print(f"  SM model stratum (XX11, front-loaded, Huff Q1/2): "
          f"{n_xx11_mapped:,}  ({pct_xx11:.1f}%)")
    print(f"  DT model stratum (00XX, back-loaded,  Huff Q3/4): "
          f"{n_00xx_mapped:,}  ({pct_00xx:.1f}%)")
    print(f"  Total (mapped to city):  {n_bsc_m:,}")

    # Per city
    print(f"\n  Per city (top 8):")
    for c in city_order:
        n1 = int(bsc_df[(bsc_df["city"] == c) & (bsc_df["type"] == "XX11")].shape[0])
        n2 = int(bsc_df[(bsc_df["city"] == c) & (bsc_df["type"] == "00XX")].shape[0])
        tot = n1 + n2
        print(f"    {c:<12}  XX11={n1:>4,} ({100*n1/tot:.0f}%)  "
              f"00XX={n2:>4,} ({100*n2/tot:.0f}%)")


if __name__ == "__main__":
    main()
