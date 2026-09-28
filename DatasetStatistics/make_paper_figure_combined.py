"""
Manuscript pie-chart overview figure — 3x2 layout:

  Row 0  (a) Trees per City      |  (b) Species by Genus
  Row 1  (c) Irrigation per City |  (d) BSC Rain Events per City
  Row 2  (e) Sensor-Data Days    |  Legend (bottom-right corner)

Design notes
------------
* Colours: Crameri "roma" (diverging blue-red), sampled with staggered/
  interleaved positions so adjacent list entries get maximally different hues.
* autopct labels shown only for slices >= 5%.  Labels switch to white on dark
  wedges (relative luminance < 0.40).
* Genera legend: top-N genera by count (where N = number of named cities) so
  both legend sections have exactly the same number of entries.
* Figure: 200 mm x 210 mm; bbox_inches='tight' captures any legend overflow.
* Font: Arial bold, 8 pt.
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
from matplotlib.font_manager import FontProperties
import cmcrameri.cm as cmc
from collections import defaultdict
from pathlib import Path

REPO      = Path(__file__).resolve().parent.parent
GIS_DIR   = REPO / "GISData"
OUT_DIR   = REPO / "TreeTabularData" / "_report_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IRR_CSV   = REPO / "TreeTabularData" / "irrigation_events_all.csv"
BSC_CSV   = REPO / "TreeTabularData" / "rf_dataset.csv"
TREES_DIR = REPO / "TreeTabularData" / "trees"

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

_OTHER_COLOR = "#BDBDBD"

# ── Crameri colour helpers ─────────────────────────────────────────────────────
def _staggered_positions(n, start=0.05, end=0.95):
    """
    Sample n positions from [start, end] so sequential entries alternate
    between opposite ends of the colormap — maximising perceived contrast
    between the first (largest) categories.
    E.g.: [0.05, 0.95, 0.10, 0.90, 0.15, 0.85, ...]
    """
    base = np.linspace(start, end, n)
    mid  = n // 2 + n % 2
    first  = list(base[:mid])
    second = list(reversed(base[mid:]))
    out = []
    for a, b in zip(first, second):
        out.append(a)
        out.append(b)
    if len(first) > len(second):
        out.append(first[-1])
    return np.array(out[:n])


def _crameri_hex(cmap, positions):
    rgba = cmap(positions)
    return ["#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
            for r, g, b, a in rgba]


def city_palette(n):
    return _crameri_hex(cmc.roma, _staggered_positions(n))


def genus_palette(n):
    return _crameri_hex(cmc.roma, _staggered_positions(n))


def _luminance(rgba_tuple):
    """Relative luminance (IEC 61966-2-1)."""
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgba_tuple[:3]
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


# ── Project → city mapping ─────────────────────────────────────────────────────
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
_SKIP_GENERA = {"", "nan", "none", "unknown", "1", "0", "-"}


def _proj_to_city(proj):
    if pd.isna(proj): return None
    p = str(proj).strip()
    city = _PROJECT_TO_CITY.get(p)
    if city is None:
        for pfx in ("City ","Company ","Botanical Garden ","University "):
            if p.startswith(pfx): city = p[len(pfx):]; break
    return city or p


def _normalise_genus(g):
    g = str(g).strip().lower()
    return "acer" if g == "acre" else g


# ── Data loading ───────────────────────────────────────────────────────────────
def load_gdf():
    gdfs = []
    for shp in sorted(GIS_DIR.glob("*/VectorLayers/treeLocations.shp")):
        try: gdfs.append(gpd.read_file(shp))
        except Exception: pass
    full = pd.concat(gdfs, ignore_index=True)
    full["_city"] = full["project"].map(_proj_to_city)
    eui_col = "devEUI" if "devEUI" in full.columns else "EUI"
    full["_eui"] = full[eui_col].astype(str).str.strip().str.upper()
    return full


def build_eui_city(gdf):
    return {r["_eui"]: r["_city"] for _, r in gdf.iterrows() if r["_city"]}


def compute_sensor_days(gdf, eui_city):
    dirs   = [d for d in TREES_DIR.iterdir() if d.is_dir()]
    result = defaultdict(int)
    print(f"  Scanning {len(dirs)} tree folders …", flush=True)
    for i, d in enumerate(dirs, 1):
        if i % 100 == 0: print(f"    {i}/{len(dirs)} …", flush=True)
        csv = d / "sensor_data.csv"
        if not csv.exists(): continue
        try: df = pd.read_csv(csv, low_memory=False)
        except Exception: continue
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        if df.empty: continue
        vwc_cols = [c for c in df.columns if "VWC" in c.upper()]
        if not vwc_cols: continue
        df["_date"] = df["datetime"].dt.date
        active = df[df[vwc_cols].gt(2.0).any(axis=1)]["_date"].nunique()
        city = eui_city.get(d.name.upper())
        if city: result[city] += active
    return pd.Series(result).sort_values(ascending=False)


def assign_city_colors(sensor_days, trees_by_city, irr_by_city, bsc_by_city,
                       trees_min=5, irr_min=50, bsc_min=100, days_min=2000):
    named = set()
    for c,n in trees_by_city.items():
        if n >= trees_min: named.add(c)
    for c,n in irr_by_city.items():
        if n >= irr_min: named.add(c)
    for c,n in bsc_by_city.items():
        if n >= bsc_min: named.add(c)
    for c,n in sensor_days.items():
        if n >= days_min: named.add(c)
    order  = [c for c in sensor_days.index if c in named]
    order += sorted(named - set(order))
    colors = city_palette(len(order))
    return dict(zip(order, colors)), order


# ── Pie drawing ────────────────────────────────────────────────────────────────
def _draw_pie(ax, sizes, colors, title, pct_min=5.0, startangle=140):
    wedges, _, autotexts = ax.pie(
        sizes, labels=None,
        autopct=lambda p: f"{p:.0f}%" if p >= pct_min else "",
        colors=colors, startangle=startangle, pctdistance=0.75,
        wedgeprops={"linewidth": 0.4, "edgecolor": "white"},
    )
    for wedge, at in zip(wedges, autotexts):
        if at.get_text():
            lum = _luminance(wedge.get_facecolor())
            at.set_color("white" if lum < 0.40 else "black")
            at.set_fontsize(8)
            at.set_fontweight("bold")
    ax.set_title(title, fontsize=8, fontweight="bold", pad=4)


def _city_slices(counts, city_color, threshold):
    counts = counts.sort_values(ascending=False)
    named  = counts[counts > threshold]
    other  = int(counts[counts <= threshold].sum())
    sizes  = list(named.values.astype(int))
    colors = [city_color.get(c, _OTHER_COLOR) for c in named.index]
    if other > 0:
        sizes.append(other); colors.append(_OTHER_COLOR)
    return sizes, colors


# ── Species data (genus-level, top-N to match city count) ─────────────────────
def species_data(gdf, n_top):
    """Return genus-level pie slices, keeping only the top n_top genera."""
    genera = gdf["genus"].map(_normalise_genus) if "genus" in gdf.columns \
             else pd.Series([], dtype=str)
    genera = genera[~genera.isin(_SKIP_GENERA)]
    genus_totals = genera.value_counts()          # sorted descending
    named  = genus_totals.head(n_top)
    other  = int(genus_totals.iloc[n_top:].sum())
    glist  = list(named.index)
    cols   = genus_palette(len(glist))
    sizes  = list(named.values.astype(int))
    if other > 0:
        glist.append("other"); sizes.append(other); cols.append(_OTHER_COLOR)
    return glist, sizes, cols, named


# ── Legend in bottom-right (2,1) cell ─────────────────────────────────────────
def draw_legend_cell(ax, city_color, city_order, genus_totals, genus_colors_map):
    ax.axis("off")
    fp = FontProperties(family="Arial", weight="bold", size=8)

    city_patches = [mpatches.Patch(facecolor=city_color[c], label=c,
                                   linewidth=0.3, edgecolor="white")
                    for c in city_order if c in city_color]
    city_patches.append(mpatches.Patch(facecolor=_OTHER_COLOR, label="Other",
                                       linewidth=0.3, edgecolor="white"))

    genus_patches = []
    for g in genus_totals.index:
        n_g = int(genus_totals[g])
        col = genus_colors_map.get(g, _OTHER_COLOR)
        genus_patches.append(mpatches.Patch(facecolor=col, linewidth=0.3,
                                            edgecolor="white",
                                            label=f"{g.capitalize()} ({n_g})"))
    genus_patches.append(mpatches.Patch(facecolor=_OTHER_COLOR,
                                        linewidth=0.3, edgecolor="white",
                                        label="Other"))

    common_kw = dict(
        framealpha=0.0,
        prop=fp,
        borderpad=0.25,
        labelspacing=0.22,
        handlelength=1.1,
        handleheight=0.75,
        handletextpad=0.40,
        columnspacing=0.55,
        borderaxespad=0.0,
        ncol=2,
    )

    # City legend — flush to the left edge of the cell
    leg_city = ax.legend(
        handles=city_patches,
        loc="upper left",
        bbox_to_anchor=(0.00, 0.98),
        bbox_transform=ax.transAxes,
        title="Cities",
        **common_kw,
    )
    leg_city.get_title().set_fontweight("bold")
    leg_city.get_title().set_fontsize(8)
    ax.add_artist(leg_city)

    # Vertical separator line — drawn between the two legend blocks.
    # x=0.43 sits in the visual gap; clip_on=False lets it extend to cell edges.
    ax.plot([0.43, 0.43], [0.01, 0.99],
            transform=ax.transAxes,
            color="#777777", linewidth=0.7, linestyle="-",
            solid_capstyle="butt", clip_on=False, zorder=0)

    # Genus legend — shifted left (0.44 instead of 0.48)
    leg_genus = ax.legend(
        handles=genus_patches,
        loc="upper left",
        bbox_to_anchor=(0.44, 0.98),
        bbox_transform=ax.transAxes,
        title="Tree genera",
        **common_kw,
    )
    leg_genus.get_title().set_fontweight("bold")
    leg_genus.get_title().set_fontsize(8)
    # leg_genus becomes the axes' active legend — no add_artist needed


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # 1. Shapefiles
    print("[1] Loading shapefile data …")
    gdf      = load_gdf()
    eui_city = build_eui_city(gdf)
    n_trees  = len(gdf)
    print(f"  Total trees: {n_trees}")
    trees_by_city = gdf["_city"].fillna("Unknown").value_counts()

    # 2. Irrigation events
    print("[2] Loading irrigation events …")
    irr = pd.read_csv(IRR_CSV, usecols=["eui","label","event_time"],
                      dtype={"eui": str}, low_memory=False)
    irr["_city"] = irr["eui"].str.strip().str.upper().map(eui_city)
    irr_ev       = irr[(irr["_city"].notna()) & (irr["label"] == "IRRIGATION")]
    irr_by_city  = irr_ev.groupby("_city").size().sort_values(ascending=False)
    n_irr        = int(irr_by_city.sum())
    print(f"  Total irrigation events: {n_irr}")

    # 3. BSC events (combined)
    print("[3] Loading BSC events …")
    bsc   = pd.read_csv(BSC_CSV, usecols=["eui","bsc","huff_q"],
                        dtype={"eui": str}, low_memory=False)
    bsc_s = bsc["bsc"].astype(str).str.zfill(4)
    mask  = ((bsc_s.str[:2] == "11") & bsc["huff_q"].isin([1,2])) | \
            ((bsc_s.str[:2] == "00") & bsc["huff_q"].isin([3,4]))
    bsc   = bsc[mask].copy()
    bsc["_city"]  = bsc["eui"].str.strip().str.upper().map(eui_city)
    bsc_by_city   = bsc[bsc["_city"].notna()].groupby("_city").size().sort_values(ascending=False)
    n_bsc         = int(bsc_by_city.sum())
    print(f"  Total BSC events: {n_bsc}")

    # 4. Sensor-days
    print("[4] Computing sensor-data days …")
    sensor_days = compute_sensor_days(gdf, eui_city)
    n_sdays     = int(sensor_days.sum())
    print(f"  Total sensor-data days: {n_sdays}")

    # 5. City colours
    print("[5] Assigning city colours …")
    city_color, city_order = assign_city_colors(
        sensor_days, trees_by_city, irr_by_city, bsc_by_city)
    n_cities = len(city_order)
    print(f"  Named cities: {n_cities}")

    # 6. Species — top-N genera where N matches named city count
    genus_list, gsizes, gcolors, genus_totals = species_data(gdf, n_top=n_cities)
    genus_colors_map = {g: c for g,c in zip(genus_list, gcolors) if g != "other"}
    print(f"  Named genera shown: {len(genus_totals)}")

    # ── Figure ────────────────────────────────────────────────────────────────
    print("[6] Building figure …")
    FW = 200 / 25.4   # 200 mm wide
    FH = 210 / 25.4   # 210 mm tall

    fig = plt.figure(figsize=(FW, FH))
    gs  = gridspec.GridSpec(
        3, 2, figure=fig,
        left=0.03, right=0.99,
        top=0.97,  bottom=0.02,
        hspace=0.22, wspace=0.12,
    )

    ax_trees = fig.add_subplot(gs[0, 0])
    ax_sp    = fig.add_subplot(gs[0, 1])
    ax_irr   = fig.add_subplot(gs[1, 0])
    ax_bsc   = fig.add_subplot(gs[1, 1])
    ax_sdays = fig.add_subplot(gs[2, 0])
    ax_leg   = fig.add_subplot(gs[2, 1])

    # ── Pies ──────────────────────────────────────────────────────────────────
    # (a) Trees per city
    tc = trees_by_city.sort_values(ascending=False)
    ts = list(tc[tc>=5].values.astype(int)) + \
         ([int(tc[tc<5].sum())] if (tc<5).any() else [])
    tc_c = [city_color.get(c, _OTHER_COLOR) for c in tc[tc>=5].index] + \
           ([_OTHER_COLOR] if (tc<5).any() else [])
    _draw_pie(ax_trees, ts, tc_c,
              f"(a) Trees per City\n(n = {n_trees:,})", pct_min=5.0)

    # (b) Species by genus
    _draw_pie(ax_sp, gsizes, gcolors,
              f"(b) Species by Genus\n(n = {n_trees:,})", pct_min=5.0)

    # (c) Irrigation per city
    i_sz, i_col = _city_slices(irr_by_city, city_color, 50)
    _draw_pie(ax_irr, i_sz, i_col,
              f"(c) Irrigation Events per City\n(n = {n_irr:,})", pct_min=5.0)

    # (d) BSC rain per city
    b_sz, b_col = _city_slices(bsc_by_city, city_color, 100)
    _draw_pie(ax_bsc, b_sz, b_col,
              f"(d) BSC Rain Events per City\n(n = {n_bsc:,})", pct_min=5.0)

    # (e) Cumulative sensor-data days
    sd_sz, sd_col = _city_slices(sensor_days, city_color, 2000)
    _draw_pie(ax_sdays, sd_sz, sd_col,
              f"(e) Cumulative Sensor-Data Days\n(n = {n_sdays:,})", pct_min=5.0)

    # ── Legend (bottom-right cell) ────────────────────────────────────────────
    draw_legend_cell(ax_leg, city_color, city_order, genus_totals, genus_colors_map)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_pdf = OUT_DIR / "fig_pie_overview.pdf"
    out_png = OUT_DIR / "fig_pie_overview.png"
    fig.savefig(out_pdf, dpi=180, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    print(f"\nn values for caption:")
    print(f"  Trees:              {n_trees}")
    print(f"  Irrigation events:  {n_irr}")
    print(f"  BSC rain events:    {n_bsc}")
    print(f"  Sensor-data days:   {n_sdays:,}")
    print(f"\nLegend entries: {n_cities} named cities, {len(genus_totals)} named genera")


if __name__ == "__main__":
    main()
