"""
Generates seven clean pie charts for the manuscript:

  1. chart_project_pie_paper.png    — trees per project, <=4 sensors → "Other",
                                      legend shows total count only (no online/offline)
  2. chart_species_pie_paper.png    — species distribution, legend is genus-level only,
                                      genera with <=5 trees → "Other"
  3. chart_irrigation_by_city.png   — absolute IRRIGATION event counts per city
  4. chart_bsc_by_city.png          — absolute BSC rain events (XX11_Q12 + 00XX_Q34)
                                      per city
  5. chart_sensor_days_by_city.png  — sum of valid sensor-data days per project/city
  6. chart_bsc_split_by_city.png    — BSC events per city, XX11 (lighter) vs 00XX
                                      (darker) split within each city wedge
  7. chart_seasonality_bsc.png      — BSC rain events by calendar month (Jan–Dec)
     chart_seasonality_irr.png      — IRRIGATION events by calendar month

All outputs go to TreeTabularData/_report_assets/.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from pathlib import Path

REPO      = Path(__file__).resolve().parent.parent
GIS_DIR   = REPO / "GISData"
OUT_DIR   = REPO / "TreeTabularData" / "_report_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IRR_CSV = REPO / "TreeTabularData" / "irrigation_events_all.csv"
BSC_CSV = REPO / "TreeTabularData" / "rf_dataset.csv"

# ── Font consistent with all manuscript figures ────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
    "font.size":        9,
    "legend.fontsize":  7.5,
})

# ── Colour palettes (same as urban_tree_report.py) ────────────────────────────
COLORS_PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
    "#44BBA4", "#E94F37", "#393E41", "#D4A373", "#6D597A",
    "#B56576", "#355070", "#EAAC8B", "#E56B6F", "#8DB580",
    "#F7B267", "#F79D65", "#F4845F", "#F27059", "#F25C54",
    "#7B2D8E", "#1B998B", "#5C677D", "#FF6B6B", "#4ECDC4",
    "#A5668B", "#0F4C5C", "#FB6107", "#36C9C6", "#C1666B",
]

GENUS_BASE_COLORS = {
    "tilia":        "#2E86AB", "acer":         "#A23B72",
    "quercus":      "#F18F01", "castanea":     "#C73E1D",
    "fagus":        "#355070", "fraxinus":     "#44BBA4",
    "prunus":       "#E94F37", "ostrya":       "#6D597A",
    "platanus":     "#D4A373", "carpinus":     "#B56576",
    "ulmus":        "#8DB580", "populus":      "#F7B267",
    "alnus":        "#3B1F2B", "magnolia":     "#E56B6F",
    "liquidambar":  "#EAAC8B", "corylus":      "#F4845F",
    "malus":        "#F27059", "ginkgo":       "#F25C54",
    "parrotia":     "#393E41", "sophora":      "#7B2D8E",
    "zelkova":      "#1B998B", "celtis":       "#5C677D",
    "sorbus":       "#FF6B6B", "chitalpa":     "#4ECDC4",
}

_OTHER_COLOR = "#BDBDBD"

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

VARIETY_IGNORE = {"", "nan", "none", "1", "0", "-", "n/a", "unknown"}


def _shade_color(hex_color, factor):
    """Tint/shade a hex colour.
    factor in [0,1]: interpolate from white (0) to the base colour (1).
    factor in [1,2]: interpolate from the base colour (1) toward black (2).
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    if factor <= 1.0:
        t = 1.0 - factor          # 0 = original, 1 = white
        r2 = int(r + (255 - r) * t)
        g2 = int(g + (255 - g) * t)
        b2 = int(b + (255 - b) * t)
    else:
        t = factor - 1.0          # 0 = original, 1 = black
        r2 = int(r * (1.0 - t))
        g2 = int(g * (1.0 - t))
        b2 = int(b * (1.0 - t))
    r2 = max(0, min(255, r2))
    g2 = max(0, min(255, g2))
    b2 = max(0, min(255, b2))
    return f"#{r2:02x}{g2:02x}{b2:02x}"


# ── Load GIS data ─────────────────────────────────────────────────────────────
def load_gdf():
    shps = sorted(GIS_DIR.glob("*/VectorLayers/treeLocations.shp"))
    gdfs = []
    for shp in shps:
        try:
            gdf = gpd.read_file(shp)
            gdfs.append(gdf)
        except Exception as e:
            print(f"  [warn] {shp.parent.parent.name}: {e}")
    if not gdfs:
        raise RuntimeError("No treeLocations.shp found")
    full = pd.concat([g for g in gdfs], ignore_index=True)
    print(f"  Loaded {len(full):,} rows from {len(shps)} shapefiles")
    return full


def build_eui_city_map(gdf):
    """Return {eui_upper: city_name} using project → city lookup."""
    mapping = {}
    for _, row in gdf.iterrows():
        eui = str(row.get("devEUI") or row.get("EUI") or "").strip().upper()
        if not eui:
            continue
        proj = str(row.get("project", "")).strip()
        city = _PROJECT_TO_CITY.get(proj)
        if city is None:
            for pfx in ("City ", "Company ", "Botanical Garden ",
                        "University ", "Castle ", "Schloss "):
                if proj.startswith(pfx):
                    city = proj[len(pfx):]
                    break
        mapping[eui] = city or proj or "Unknown"
    return mapping


# ── Pie helper ────────────────────────────────────────────────────────────────
def _make_pie(labels, sizes, colors, title, legend_labels, out_path,
              figsize=(9, 6), pct_threshold=2.5, legend_ncol=1,
              legend_anchor=(1.02, 0.5), legend_loc="center left",
              other_color=_OTHER_COLOR):
    """Generic pie chart with an external legend."""
    fig, ax = plt.subplots(figsize=figsize)
    wedges, _, autotexts = ax.pie(
        sizes, labels=None,
        autopct=lambda p: f"{p:.1f}%" if p >= pct_threshold else "",
        colors=colors, startangle=140, pctdistance=0.82,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(7.5)

    ax.legend(wedges, legend_labels,
              loc=legend_loc, bbox_to_anchor=legend_anchor,
              fontsize=7.5, framealpha=0.9, ncol=legend_ncol)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1: Trees per project  (group ≤ 4 sensors → "Other")
# ─────────────────────────────────────────────────────────────────────────────
def chart_project_pie(gdf):
    proj_counts = gdf["project"].fillna("Unknown").value_counts().sort_values(ascending=False)

    OTHER_MAX = 4   # projects with ≤ this many trees → "Other"
    named  = proj_counts[proj_counts > OTHER_MAX]
    other_n = int(proj_counts[proj_counts <= OTHER_MAX].sum())
    n_other_proj = int((proj_counts <= OTHER_MAX).sum())

    labels = list(named.index)
    sizes  = list(named.values)
    if other_n > 0:
        labels.append(f"Other ({n_other_proj} projects)")
        sizes.append(other_n)

    n = len(labels)
    base_colors = (COLORS_PALETTE * (n // len(COLORS_PALETTE) + 1))[:n]
    if other_n > 0:
        base_colors[-1] = _OTHER_COLOR

    legend_labels = [f"{lb}  ({cnt})" for lb, cnt in zip(labels, sizes)]

    _make_pie(labels, sizes, base_colors,
              "Trees per Project",
              legend_labels,
              OUT_DIR / "chart_project_pie_paper.png",
              figsize=(10, 6.5), pct_threshold=2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2: Species distribution (genus-level legend, ≤5 → "Other")
# ─────────────────────────────────────────────────────────────────────────────
def _build_species_label(genus, species, variety):
    genus   = str(genus).strip().lower()
    species = str(species).strip().lower()
    variety = str(variety).strip().lower()
    if genus in ("", "nan"):
        genus = "unknown"
    if genus == "acre":
        genus = "acer"
    parts = [genus.capitalize()]
    if species not in ("", "nan"):
        parts.append(species)
    if variety not in VARIETY_IGNORE:
        parts.append(f"'{variety}'")
    return genus, " ".join(parts)


def chart_species_pie(gdf):
    OTHER_MIN = 6   # genera with < this many trees → "Other"

    records = []
    for _, row in gdf.iterrows():
        genus, label = _build_species_label(
            row.get("genus", ""), row.get("species", ""), row.get("variety", ""))
        records.append({"genus": genus, "label": label})

    df = pd.DataFrame(records)
    genus_totals = df["genus"].value_counts()

    named_genera = genus_totals[genus_totals >= OTHER_MIN].index
    other_genera = genus_totals[genus_totals < OTHER_MIN].index
    other_total  = int(genus_totals[other_genera].sum())

    # Species counts restricted to named genera
    df_named = df[df["genus"].isin(named_genera)]
    species_counts = df_named["label"].value_counts()
    genus_of = df_named.drop_duplicates("label").set_index("label")["genus"]

    # Order: largest genus first, within genus largest species first
    ordered_labels, ordered_sizes, ordered_genus = [], [], []
    for g in named_genera:
        members = sorted(
            [l for l in species_counts.index if genus_of.get(l) == g],
            key=lambda l: species_counts[l], reverse=True)
        for m in members:
            ordered_labels.append(m)
            ordered_sizes.append(int(species_counts[m]))
            ordered_genus.append(g)

    # Build per-species color (shades within genus)
    genus_member_n = defaultdict(int)
    for g in ordered_genus:
        genus_member_n[g] += 1
    genus_idx = defaultdict(int)
    color_list = []
    for g in ordered_genus:
        base  = GENUS_BASE_COLORS.get(g, COLORS_PALETTE[hash(g) % len(COLORS_PALETTE)])
        n_m   = genus_member_n[g]
        i     = genus_idx[g]
        genus_idx[g] += 1
        color_list.append(base if n_m == 1 else
                          _shade_color(base, 0.65 + 0.70 * (i / max(n_m - 1, 1))))

    # "Other" slice
    if other_total > 0:
        ordered_labels.append("Other")
        ordered_sizes.append(other_total)
        ordered_genus.append("other")
        color_list.append(_OTHER_COLOR)

    # ── Legend: one entry per genus (not per species) ─────────────────────────
    # Show a colour swatch for each genus using its mid-shade, + count + n species
    genus_label_indices = {}   # genus → list of indices in ordered_labels
    for i, g in enumerate(ordered_genus):
        genus_label_indices.setdefault(g, []).append(i)

    legend_handles = []
    legend_texts   = []
    seen_genera = []  # preserve genus order
    for g in ordered_genus:
        if g not in seen_genera:
            seen_genera.append(g)

    for g in seen_genera:
        idxs   = genus_label_indices[g]
        total  = sum(ordered_sizes[i] for i in idxs)
        n_spec = len(idxs)
        if g == "other":
            label_text = f"Other  ({total})"
            swatch_col = _OTHER_COLOR
        else:
            # Median shade for the swatch
            swatch_col = color_list[idxs[len(idxs)//2]]
            n_spec_str = f", {n_spec} species" if n_spec > 1 else ""
            label_text = f"{g.capitalize()}  ({total}{n_spec_str})"
        patch = mpatches.Patch(facecolor=swatch_col, edgecolor="white", linewidth=0.5)
        legend_handles.append(patch)
        legend_texts.append(label_text)

    fig, ax = plt.subplots(figsize=(10, 7))
    wedges, _, autotexts = ax.pie(
        ordered_sizes, labels=None,
        autopct=lambda p: f"{p:.1f}%" if p >= 3.0 else "",
        colors=color_list, startangle=140, pctdistance=0.83,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(7.5)

    ax.legend(legend_handles, legend_texts,
              loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7.5, framealpha=0.9,
              ncol=2 if len(legend_texts) > 18 else 1)
    ax.set_title("Species Distribution (by genus)",
                 fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    out = OUT_DIR / "chart_species_pie_paper.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared: build city event counts and apply "Other" threshold
# ─────────────────────────────────────────────────────────────────────────────
def _city_pie(city_counts: pd.Series, title: str, out_name: str,
              other_min_n: int = 0):
    """
    city_counts: Series indexed by city name, values = event count.
    other_min_n: cities with <= this many events are grouped as "Other".
                 0 means auto (top 12 named, rest "Other").
    """
    city_counts = city_counts.sort_values(ascending=False)

    if other_min_n > 0:
        named  = city_counts[city_counts > other_min_n]
        other_n = int(city_counts[city_counts <= other_min_n].sum())
        n_other = int((city_counts <= other_min_n).sum())
    else:
        # Keep top 12, rest "Other"
        named  = city_counts.iloc[:12]
        tail   = city_counts.iloc[12:]
        other_n = int(tail.sum())
        n_other = len(tail)

    labels = list(named.index)
    sizes  = list(named.values.astype(int))

    if other_n > 0:
        labels.append(f"Other ({n_other} cities)")
        sizes.append(other_n)

    n = len(labels)
    colors = (COLORS_PALETTE * (n // len(COLORS_PALETTE) + 1))[:n]
    if other_n > 0:
        colors[-1] = _OTHER_COLOR

    total = sum(sizes)
    legend_labels = [f"{lb}  ({cnt:,})" for lb, cnt in zip(labels, sizes)]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    wedges, _, autotexts = ax.pie(
        sizes, labels=None,
        autopct=lambda p: f"{p:.1f}%" if p >= 2.0 else "",
        colors=colors, startangle=140, pctdistance=0.82,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(7.5)

    summary = mpatches.Patch(facecolor="none", edgecolor="none",
                             label=f"Total: {total:,} events")
    ax.legend(wedges + [summary],
              legend_labels + [f"Total: {total:,} events"],
              loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7.5, framealpha=0.9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3: Irrigation events per city
# ─────────────────────────────────────────────────────────────────────────────
def chart_irrigation_by_city(eui_city):
    irr = pd.read_csv(IRR_CSV, usecols=["eui", "label"], dtype={"eui": str},
                      low_memory=False)
    irr = irr[irr["label"] == "IRRIGATION"].copy()
    irr["city"] = irr["eui"].str.strip().str.upper().map(eui_city)
    irr = irr[irr["city"].notna()]
    city_counts = irr.groupby("city").size().sort_values(ascending=False)
    print(f"\n  Irrigation events by city (top 15):")
    for city, n in city_counts.head(15).items():
        print(f"    {city:<25} {n:>5}")

    # Keep cities with > 50 events individually
    _city_pie(city_counts, "Irrigation Events per City",
              "chart_irrigation_by_city.png", other_min_n=50)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4: BSC rain events (XX11_Q12 + 00XX_Q34) per city
# ─────────────────────────────────────────────────────────────────────────────
def chart_bsc_by_city(eui_city):
    bsc = pd.read_csv(BSC_CSV, usecols=["eui", "bsc", "huff_q"],
                      dtype={"eui": str}, low_memory=False)
    bsc_s = bsc["bsc"].astype(str).str.zfill(4)
    mask_xx11 = (bsc_s.str[:2] == "11") & bsc["huff_q"].isin([1, 2])
    mask_00xx = (bsc_s.str[:2] == "00") & bsc["huff_q"].isin([3, 4])
    bsc = bsc[mask_xx11 | mask_00xx].copy()
    bsc["city"] = bsc["eui"].str.strip().str.upper().map(eui_city)
    bsc = bsc[bsc["city"].notna()]
    city_counts = bsc.groupby("city").size().sort_values(ascending=False)
    print(f"\n  BSC rain events by city (top 15):")
    for city, n in city_counts.head(15).items():
        print(f"    {city:<25} {n:>5}")

    # Keep cities with > 100 events individually
    _city_pie(city_counts, "BSC Rain Events per City (XX11-Q1/2 + 00XX-Q3/4)",
              "chart_bsc_by_city.png", other_min_n=100)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5: Valid sensor-data days per project/city
# ─────────────────────────────────────────────────────────────────────────────
def chart_sensor_days_by_city(gdf, other_min_sensor_days=500):
    """
    For each tree folder, count unique calendar days that have at least one
    VWC reading > 2 % (same active threshold as fig_sensor_deployment.py).
    Sum across all sensors in the same city and plot as pie.
    """
    TREES_DIR = REPO / "TreeTabularData" / "trees"
    VWC_THRESHOLD = 2.0

    eui_city_map = build_eui_city_map(gdf)

    dirs = [d for d in TREES_DIR.iterdir() if d.is_dir()]
    print(f"  Scanning {len(dirs)} tree folders for valid data days …", flush=True)

    city_days: dict[str, int] = defaultdict(int)
    n_found = 0
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

        vwc_cols = [c for c in df.columns if "VWC" in c.upper()]
        if not vwc_cols:
            continue

        df["_date"] = df["datetime"].dt.date
        active = df[df[vwc_cols].gt(VWC_THRESHOLD).any(axis=1)]["_date"].unique()
        n_valid_days = len(active)
        if n_valid_days == 0:
            continue

        eui = d.name.upper()
        city = eui_city_map.get(eui)
        if city:
            city_days[city] += n_valid_days
            n_found += 1

    print(f"  Summed valid days for {n_found} sensors across {len(city_days)} cities")

    city_ser = pd.Series(city_days).sort_values(ascending=False)
    print("\n  Sensor-days by city (top 15):")
    for city, d in city_ser.head(15).items():
        print(f"    {city:<25} {d:>8,}")

    named  = city_ser[city_ser > other_min_sensor_days]
    other_n = int(city_ser[city_ser <= other_min_sensor_days].sum())
    n_other = int((city_ser <= other_min_sensor_days).sum())

    labels = list(named.index)
    sizes  = list(named.values.astype(int))
    if other_n > 0:
        labels.append(f"Other ({n_other} cities)")
        sizes.append(other_n)

    n = len(labels)
    colors = (COLORS_PALETTE * (n // len(COLORS_PALETTE) + 1))[:n]
    if other_n > 0:
        colors[-1] = _OTHER_COLOR

    total = sum(sizes)
    legend_labels = [f"{lb}  ({cnt:,} sensor-days)" for lb, cnt in zip(labels, sizes)]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    wedges, _, autotexts = ax.pie(
        sizes, labels=None,
        autopct=lambda p: f"{p:.1f}%" if p >= 2.5 else "",
        colors=colors, startangle=140, pctdistance=0.82,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(7.5)

    summary = mpatches.Patch(facecolor="none", edgecolor="none",
                             label=f"Total: {total:,} sensor-days")
    ax.legend(wedges + [summary],
              legend_labels + [f"Total: {total:,} sensor-days"],
              loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7.5, framealpha=0.9)
    ax.set_title("Cumulative Valid Sensor-Data Days per City",
                 fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    out = OUT_DIR / "chart_sensor_days_by_city.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 6: BSC events per city split into XX11 (lighter) vs 00XX (darker)
# ─────────────────────────────────────────────────────────────────────────────
def chart_bsc_split_by_city(eui_city, other_min_n=100):
    bsc = pd.read_csv(BSC_CSV, usecols=["eui", "bsc", "huff_q"],
                      dtype={"eui": str}, low_memory=False)
    bsc_s = bsc["bsc"].astype(str).str.zfill(4)
    bsc = bsc.copy()
    bsc["city"]  = bsc["eui"].str.strip().str.upper().map(eui_city)
    bsc["type"]  = None
    mask_xx11 = (bsc_s.str[:2] == "11") & bsc["huff_q"].isin([1, 2])
    mask_00xx = (bsc_s.str[:2] == "00") & bsc["huff_q"].isin([3, 4])
    bsc.loc[mask_xx11, "type"] = "XX11"
    bsc.loc[mask_00xx, "type"] = "00XX"
    bsc = bsc[bsc["city"].notna() & bsc["type"].notna()]

    city_total = bsc.groupby("city").size().sort_values(ascending=False)
    named_cities = city_total[city_total > other_min_n].index.tolist()
    other_xx11 = int(bsc[~bsc["city"].isin(named_cities) & (bsc["type"] == "XX11")].shape[0])
    other_00xx = int(bsc[~bsc["city"].isin(named_cities) & (bsc["type"] == "00XX")].shape[0])

    # Build interleaved slices: [CityA-XX11, CityA-00XX, CityB-XX11, ...]
    labels, sizes, colors = [], [], []
    city_base_color = {}
    for i, city in enumerate(named_cities):
        base = COLORS_PALETTE[i % len(COLORS_PALETTE)]
        city_base_color[city] = base
        n_xx11 = int(bsc[(bsc["city"] == city) & (bsc["type"] == "XX11")].shape[0])
        n_00xx = int(bsc[(bsc["city"] == city) & (bsc["type"] == "00XX")].shape[0])
        if n_xx11 > 0:
            labels.append(f"{city} XX11")
            sizes.append(n_xx11)
            colors.append(_shade_color(base, 0.70))   # lighter
        if n_00xx > 0:
            labels.append(f"{city} 00XX")
            sizes.append(n_00xx)
            colors.append(_shade_color(base, 1.25))   # darker

    # "Other" as two gray slices
    if other_xx11 > 0:
        labels.append("Other XX11")
        sizes.append(other_xx11)
        colors.append(_shade_color(_OTHER_COLOR, 0.75))
    if other_00xx > 0:
        labels.append("Other 00XX")
        sizes.append(other_00xx)
        colors.append(_shade_color(_OTHER_COLOR, 1.20))

    total = sum(sizes)

    fig, ax = plt.subplots(figsize=(11, 7))
    wedges, _, autotexts = ax.pie(
        sizes, labels=None,
        autopct=lambda p: f"{p:.0f}%" if p >= 3.5 else "",
        colors=colors, startangle=140, pctdistance=0.84,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(7)

    # Legend: one entry per city showing both XX11 and 00XX swatches + counts
    legend_handles, legend_texts = [], []

    # Header line
    legend_handles.append(mpatches.Patch(facecolor="none", edgecolor="none"))
    legend_texts.append("   lighter = XX11_Q1/2   darker = 00XX_Q3/4")

    for city in named_cities:
        base = city_base_color[city]
        n_xx11 = int(bsc[(bsc["city"] == city) & (bsc["type"] == "XX11")].shape[0])
        n_00xx = int(bsc[(bsc["city"] == city) & (bsc["type"] == "00XX")].shape[0])
        total_c = n_xx11 + n_00xx

        # Two-tone swatch via a custom patch with a line down the middle
        # (easiest: use two separate small legend entries per city is messy;
        #  instead use the lighter shade as swatch and show counts in label)
        swatch = mpatches.Patch(
            facecolor=_shade_color(base, 0.70),
            edgecolor=_shade_color(base, 1.25), linewidth=2)
        legend_handles.append(swatch)
        legend_texts.append(
            f"{city}  (XX11: {n_xx11:,} | 00XX: {n_00xx:,})")

    # Other entry
    n_other_total = other_xx11 + other_00xx
    if n_other_total > 0:
        other_cities_n = int((city_total <= other_min_n).sum())
        legend_handles.append(
            mpatches.Patch(facecolor=_OTHER_COLOR, edgecolor="#999999", linewidth=1))
        legend_texts.append(
            f"Other ({other_cities_n} cities, "
            f"XX11: {other_xx11:,} | 00XX: {other_00xx:,})")

    # Summary
    legend_handles.append(mpatches.Patch(facecolor="none", edgecolor="none"))
    legend_texts.append(f"Total: {total:,} events")

    ax.legend(legend_handles, legend_texts,
              loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=7.5, framealpha=0.9)
    ax.set_title("BSC Rain Events per City\n(XX11-Q1/2 vs 00XX-Q3/4)",
                 fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    out = OUT_DIR / "chart_bsc_split_by_city.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 7: Seasonality (monthly event counts)
# ─────────────────────────────────────────────────────────────────────────────
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# Season-based colour: winter blue, spring green, summer orange, autumn brown
_MONTH_COLORS = [
    "#4A90D9", "#5BA0E8", "#A8D5A2",  # Jan Feb Mar
    "#6EC96F", "#B8E07A", "#FFD166",  # Apr May Jun
    "#FF9F43", "#FF6B35", "#C73E1D",  # Jul Aug Sep
    "#8B4513", "#6B4C3B", "#3A7BD5",  # Oct Nov Dec
]


def _seasonality_pie(month_counts: pd.Series, title: str, out_name: str):
    sizes  = [int(month_counts.get(m, 0)) for m in range(1, 13)]
    total  = sum(sizes)
    labels = _MONTH_NAMES[:]
    colors = _MONTH_COLORS[:]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _, autotexts = ax.pie(
        sizes, labels=None,
        autopct=lambda p: f"{p:.1f}%" if p >= 2.0 else "",
        colors=colors, startangle=90,  # start at 12 o'clock = Jan
        pctdistance=0.78,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(7.5)

    legend_labels = [f"{m}  ({n:,})" for m, n in zip(_MONTH_NAMES, sizes)]
    summary = mpatches.Patch(facecolor="none", edgecolor="none",
                             label=f"Total: {total:,} events")
    ax.legend(wedges + [summary],
              legend_labels + [f"Total: {total:,} events"],
              loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, framealpha=0.9, ncol=1)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


def chart_seasonality():
    # BSC rain events — use `month` column directly
    bsc = pd.read_csv(BSC_CSV, usecols=["bsc", "huff_q", "month"],
                      low_memory=False)
    bsc_s = bsc["bsc"].astype(str).str.zfill(4)
    mask_xx11 = (bsc_s.str[:2] == "11") & bsc["huff_q"].isin([1, 2])
    mask_00xx = (bsc_s.str[:2] == "00") & bsc["huff_q"].isin([3, 4])
    bsc_filt = bsc[mask_xx11 | mask_00xx]
    bsc_month = pd.to_numeric(bsc_filt["month"], errors="coerce").dropna()
    _seasonality_pie(bsc_month.value_counts(),
                     "BSC Rain Events — Monthly Distribution",
                     "chart_seasonality_bsc.png")

    # Irrigation events — derive month from event_time
    irr = pd.read_csv(IRR_CSV, usecols=["label", "event_time"],
                      dtype={"label": str}, low_memory=False)
    irr = irr[irr["label"] == "IRRIGATION"]
    irr["month"] = pd.to_datetime(irr["event_time"], errors="coerce").dt.month
    _seasonality_pie(irr["month"].dropna().astype(int).value_counts(),
                     "Irrigation Events — Monthly Distribution",
                     "chart_seasonality_irr.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n[1/8] Loading shapefile data …")
    gdf = load_gdf()
    eui_city = build_eui_city_map(gdf)
    print(f"  EUI→city map: {len(eui_city):,} entries")

    print("\n[2/8] Chart 1: project pie …")
    chart_project_pie(gdf)

    print("\n[3/8] Chart 2: species pie …")
    chart_species_pie(gdf)

    print("\n[4/8] Chart 3: irrigation events by city …")
    chart_irrigation_by_city(eui_city)

    print("\n[5/8] Chart 4: BSC rain events by city (combined) …")
    chart_bsc_by_city(eui_city)

    print("\n[6/8] Chart 5: valid sensor-data days by city …")
    chart_sensor_days_by_city(gdf)

    print("\n[7/8] Chart 6: BSC rain events split XX11 vs 00XX …")
    chart_bsc_split_by_city(eui_city)

    print("\n[8/8] Chart 7: seasonality (BSC + irrigation) …")
    chart_seasonality()

    print("\nDone.")


if __name__ == "__main__":
    main()
