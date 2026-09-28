#!/usr/bin/env python3
"""
Generate all figures for the manuscript.

Run from the figures directory:
    python generate_figures.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Arc, Wedge
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import seaborn as sns
import contextily as ctx
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent
GIS_DIR     = REPO_ROOT / "GISData"
FIG_DIR     = SCRIPT_DIR / "paper"
FIG_DIR.mkdir(exist_ok=True)

# Folders excluded from analysis (no trees — only annuals/perennials)
EXCLUDED_DIRS = {"CityKarben", "CityWeißenburg", "CityOffenbach"}

CITY_DIRS = sorted([
    d for d in GIS_DIR.iterdir()
    if d.is_dir()
    and d.name not in EXCLUDED_DIRS
    and (d / "VectorLayers" / "treeLocations.shp").exists()
])

# Display names for all project folders (folder name → label)
CITY_LABELS = {
    # Municipal / city deployments
    "CityErlangen":               "Erlangen",
    "CityBamberg":                "Bamberg",
    "CityBerlinFrhXBerg":         "Berlin-Friedrichshain",
    "CityBerlinNeukölln":         "Berlin-Neukölln",
    "CityBerlinCharlottenburg":   "Berlin-Charlottenburg",
    "CityPotsdam":                "Potsdam",
    "CityHanover":                "Hannover",
    "CityGarbsen":                "Garbsen",
    "CityHildesheim":             "Hildesheim",
    "CityHagen":                  "Hagen",
    "CityLeipzig":                "Leipzig",
    "CityIngolstadt":             "Ingolstadt",
    "CityHassfurt":               "Hassfurt",
    "CityBiberach":               "Biberach",
    "CityErfurt":                 "Erfurt",
    "CityPirmasens":              "Pirmasens",
    "CityBremen":                 "Bremen",
    "CityAachen":                 "Aachen",
    "CityKassel":                 "Kassel",
    "CityLemgo":                  "Lemgo",
    "CityLüdenscheid":            "Lüdenscheid",
    "CityNürnberg":               "Nürnberg",
    "CityPforzheim":              "Pforzheim",
    "CityStein":                  "Stein",
    "CityVienna":                 "Vienna",
    "CityWeisendorf":             "Weisendorf",
    "CityCelle":                  "Celle",
    "CityCrailsheim":             "Crailsheim",
    "CityGrünwald":               "Grünwald",
    "CityNeunkirchen":            "Neunkirchen",
    "CityUniversitySalzburg":     "Salzburg (Uni)",
    # Company / private deployments
    "CompanyBremen":              "Bremen (Firma)",
    "CompanyEffeltrich":          "Effeltrich (Firma)",
    "CompanyErfurt":              "Erfurt (Firma)",
    "CompanyErlangen":            "Erlangen (Firma)",
    "CompanyHeidelberg":          "Heidelberg (Firma)",
    "CompanyHomburg":             "Homburg (Firma)",
    "CompanyLaax":                "Laax (Firma)",
    "CompanyNörvenich":           "Nörvenich (Firma)",
    "CompanyPappenheim":          "Pappenheim (Firma)",
    "CompanyPillnitz":            "Pillnitz (Firma)",
    "CompanyPlön":                "Plön (Firma)",
    "CompanySaarbrücken":         "Saarbrücken (Firma)",
    "CompanyVienna":              "Wien (Firma)",
    # Botanical gardens
    "BotanicalGardenBremen":      "Botanischer Garten Bremen",
    "BotanicalGardenVienna":      "Botanischer Garten Wien",
    # University / research
    "UniversityErlangen":         "Erlangen (Uni)",
    "UniversitySalzburg":         "Salzburg (Uni)",
    # Other
    "CastleAdminSaxony":          "Sachsen (Schloss)",
}

# ---------------------------------------------------------------------------
# Load all shapefiles
# ---------------------------------------------------------------------------
print("Loading shapefiles...")
frames = []
for city_dir in CITY_DIRS:
    shp = city_dir / "VectorLayers" / "treeLocations.shp"
    try:
        gdf = gpd.read_file(shp)
        gdf["city_dir"] = city_dir.name
        gdf["city_label"] = CITY_LABELS.get(city_dir.name, city_dir.name.replace("City", ""))
        frames.append(gdf)
    except Exception as exc:
        print(f"  Could not load {city_dir.name}: {exc}")

all_trees = pd.concat(frames, ignore_index=True)
print(f"  Loaded {len(all_trees)} trees from {len(frames)} cities.")

# Export combined attribute table (geometry dropped for plain CSV readability)
_csv_out = SCRIPT_DIR / "all_tree_locations.csv"
all_trees.drop(columns="geometry").to_csv(_csv_out, index=False)
print(f"  Exported combined tree table → {_csv_out.name}")

# Convert to WGS84 for mapping
all_trees_wgs = all_trees.to_crs("EPSG:4326")
all_trees_wgs["lon"] = all_trees_wgs.geometry.centroid.x
all_trees_wgs["lat"] = all_trees_wgs.geometry.centroid.y

# Compute age if germDate and soilDate or plantDate available
def parse_year(s):
    try:
        if pd.isna(s) or str(s).strip() in ("", "NULL", "None", "null"):
            return np.nan
        s = str(s).strip()
        if len(s) >= 4 and s[:4].isdigit():
            return int(s[:4])
    except Exception:
        pass
    return np.nan

if "germDate" in all_trees.columns:
    all_trees["germ_year"] = all_trees["germDate"].apply(parse_year)
    all_trees["tree_age"] = 2025 - all_trees["germ_year"]
    all_trees.loc[all_trees["tree_age"] < 0, "tree_age"] = np.nan
    all_trees.loc[all_trees["tree_age"] > 300, "tree_age"] = np.nan

# ---------------------------------------------------------------------------
# Color palette for cities (cycling)
# ---------------------------------------------------------------------------
city_list_sorted = sorted(all_trees["city_label"].dropna().unique())
palette_base = plt.cm.tab20.colors + plt.cm.tab20b.colors
city_color = {c: palette_base[i % len(palette_base)] for i, c in enumerate(city_list_sorted)}

# ============================================================================
# FIGURE 1: Map of monitoring sites
# ============================================================================
print("Generating Figure 1: Site map...")

fig, ax = plt.subplots(figsize=(10, 12))

# Reproject to Web Mercator for contextily
all_trees_wm = all_trees_wgs.to_crs("EPSG:3857")

# City centroids for labels
city_centroids = (all_trees_wm.groupby("city_label")
                  .apply(lambda g: g.geometry.unary_union.centroid)
                  .reset_index())
city_centroids.columns = ["city_label", "geometry"]
city_centroids = gpd.GeoDataFrame(city_centroids, crs="EPSG:3857")
city_centroids["n_trees"] = (all_trees_wm.groupby("city_label")
                              .size().reindex(city_centroids["city_label"]).values)

# Plot trees as scatter (small dots per city)
for city, grp in all_trees_wm.groupby("city_label"):
    xs = [g.centroid.x for g in grp.geometry]
    ys = [g.centroid.y for g in grp.geometry]
    ax.scatter(xs, ys, s=20, color=city_color.get(city, "gray"),
               alpha=0.7, linewidths=0, zorder=4)

# Annotate city names
for _, row in city_centroids.iterrows():
    ax.annotate(
        f"{row.city_label}\n(n={row.n_trees})",
        xy=(row.geometry.x, row.geometry.y),
        fontsize=6.5, ha="center", va="bottom",
        xytext=(0, 6), textcoords="offset points",
        color="black",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none"),
        zorder=6,
    )

try:
    ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron,
                    attribution_size=5, zoom="auto")
except Exception:
    ax.set_facecolor("#e8f4f8")

ax.set_axis_off()
ax.set_title(
    f"Urban Tree Soil Moisture Monitoring Network\n"
    f"{len(all_trees)} trees across {len(frames)} cities (Germany and Austria)",
    fontsize=11, fontweight="bold", pad=8
)

# Legend (one dot per city, sorted)
legend_elements = [
    mpatches.Patch(facecolor=city_color.get(c, "gray"), label=c)
    for c in city_list_sorted if c in all_trees["city_label"].values
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=5.5,
          ncol=2, framealpha=0.85, title="Partner city", title_fontsize=6)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig_map_sites.pdf", dpi=200, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_map_sites.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  → fig_map_sites saved.")

# ============================================================================
# FIGURE 2: Dataset overview – species, age, morphology
# ============================================================================
print("Generating Figure 2: Dataset overview...")

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# ── 2a: Top-20 genera ───────────────────────────────────────────────────────
ax_sp = fig.add_subplot(gs[0, :2])
if "genus" in all_trees.columns:
    genus_counts = (all_trees["genus"].dropna()
                    .str.strip().str.capitalize()
                    .value_counts().head(20))
    colors_sp = [plt.cm.Set2(i % 8) for i in range(len(genus_counts))]
    bars = ax_sp.barh(genus_counts.index[::-1], genus_counts.values[::-1],
                      color=colors_sp[::-1], edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, genus_counts.values[::-1]):
        ax_sp.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                   str(val), va="center", ha="left", fontsize=7.5)
    ax_sp.set_xlabel("Number of trees", fontsize=9)
    ax_sp.set_title("(a) Tree genera (top 20)", fontweight="bold", fontsize=10)
    ax_sp.spines[["top","right"]].set_visible(False)
    ax_sp.tick_params(labelsize=8)
else:
    ax_sp.text(0.5, 0.5, "genus field not available", ha="center", transform=ax_sp.transAxes)

# ── 2b: Trees per city ───────────────────────────────────────────────────────
ax_city = fig.add_subplot(gs[0, 2])
trees_per_city = (all_trees.groupby("city_label").size()
                  .sort_values(ascending=True).tail(20))
bar_colors = [city_color.get(c, "gray") for c in trees_per_city.index]
ax_city.barh(range(len(trees_per_city)), trees_per_city.values,
             color=bar_colors, edgecolor="white", linewidth=0.4)
ax_city.set_yticks(range(len(trees_per_city)))
ax_city.set_yticklabels(trees_per_city.index, fontsize=7.5)
ax_city.set_xlabel("Number of trees", fontsize=9)
ax_city.set_title("(b) Trees per city", fontweight="bold", fontsize=10)
ax_city.spines[["top","right"]].set_visible(False)

# ── 2c: Tree age ────────────────────────────────────────────────────────────
ax_age = fig.add_subplot(gs[1, 0])
if "tree_age" in all_trees.columns:
    age_data = all_trees["tree_age"].dropna()
    age_data = age_data[(age_data > 0) & (age_data < 250)]
    ax_age.hist(age_data, bins=30, color="#4e8098", edgecolor="white", linewidth=0.5)
    ax_age.axvline(age_data.median(), color="tomato", linestyle="--",
                   linewidth=1.5, label=f"Median: {age_data.median():.0f} a")
    ax_age.set_xlabel("Tree age (years)", fontsize=9)
    ax_age.set_ylabel("Count", fontsize=9)
    ax_age.set_title(f"(c) Tree age distribution\n(n={len(age_data)})", fontweight="bold", fontsize=10)
    ax_age.legend(fontsize=8)
    ax_age.spines[["top","right"]].set_visible(False)
else:
    ax_age.text(0.5, 0.5, "No age data", ha="center", transform=ax_age.transAxes)

# ── 2d: Stem diameter ───────────────────────────────────────────────────────
ax_stem = fig.add_subplot(gs[1, 1])
if "stemDiam" in all_trees.columns:
    stem_data = pd.to_numeric(all_trees["stemDiam"], errors="coerce").dropna()
    stem_data = stem_data[(stem_data > 0) & (stem_data < 3)]  # metres, plausible range
    # convert to cm if all values look like metres (< 5)
    if stem_data.median() < 5:
        stem_data_cm = stem_data * 100
    else:
        stem_data_cm = stem_data
    ax_stem.hist(stem_data_cm, bins=30, color="#74a57f", edgecolor="white", linewidth=0.5)
    ax_stem.axvline(stem_data_cm.median(), color="tomato", linestyle="--",
                    linewidth=1.5, label=f"Median: {stem_data_cm.median():.1f} cm")
    ax_stem.set_xlabel("Stem diameter (cm)", fontsize=9)
    ax_stem.set_ylabel("Count", fontsize=9)
    ax_stem.set_title(f"(d) Stem diameter distribution\n(n={len(stem_data_cm)})", fontweight="bold", fontsize=10)
    ax_stem.legend(fontsize=8)
    ax_stem.spines[["top","right"]].set_visible(False)
else:
    ax_stem.text(0.5, 0.5, "stemDiam field not available", ha="center", transform=ax_stem.transAxes)

# ── 2e: Crown diameter ──────────────────────────────────────────────────────
ax_crown = fig.add_subplot(gs[1, 2])
if "crownDiam" in all_trees.columns:
    crown_data = pd.to_numeric(all_trees["crownDiam"], errors="coerce").dropna()
    crown_data = crown_data[(crown_data > 0) & (crown_data < 30)]
    ax_crown.hist(crown_data, bins=25, color="#c2956c", edgecolor="white", linewidth=0.5)
    ax_crown.axvline(crown_data.median(), color="tomato", linestyle="--",
                     linewidth=1.5, label=f"Median: {crown_data.median():.1f} m")
    ax_crown.set_xlabel("Crown diameter (m)", fontsize=9)
    ax_crown.set_ylabel("Count", fontsize=9)
    ax_crown.set_title(f"(e) Crown diameter distribution\n(n={len(crown_data)})", fontweight="bold", fontsize=10)
    ax_crown.legend(fontsize=8)
    ax_crown.spines[["top","right"]].set_visible(False)
else:
    ax_crown.text(0.5, 0.5, "crownDiam field not available", ha="center", transform=ax_crown.transAxes)

fig.suptitle("Dataset Overview: Monitored Urban Trees", fontsize=13, fontweight="bold", y=1.01)
fig.savefig(FIG_DIR / "fig_dataset_overview.pdf", dpi=200, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_dataset_overview.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  → fig_dataset_overview saved.")

# ============================================================================
# FIGURE 3: Land use classification concept
# ============================================================================
print("Generating Figure 3: Land use classification concept...")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

def draw_land_use_diagram(ax, title):
    ax.set_xlim(-9, 9)
    ax.set_ylim(-9, 9)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.axis("off")

    # Background: sealed surface (gray)
    bg = plt.Rectangle((-9, -9), 18, 18, color="#b0b0b0", zorder=0)
    ax.add_patch(bg)

    # Annulus fills (hatched or colored)
    ring_colors = ["#e8c87a", "#c8d8a0", "#a0c8b0"]
    ring_radii = [7.5, 5.0, 2.5]
    ring_labels = ["5.0–7.5 m annulus", "2.5–5.0 m annulus", "0–2.5 m"]

    for r, c in zip(ring_radii, ring_colors):
        circ = plt.Circle((0, 0), r, color=c, alpha=0.25, zorder=1)
        ax.add_patch(circ)

    # Tree pit (green, attached)
    green_attached = plt.Rectangle((-2.5, -1.5), 3.5, 3.0,
                                   color="#3a7d44", alpha=0.7, zorder=2)
    ax.add_patch(green_attached)
    ax.text(-0.75, 0, "Green\n(attached)", color="white", fontsize=7,
            ha="center", va="center", zorder=5, fontweight="bold")

    # Green detached patch
    green_det = plt.Polygon([[-6, -3], [-3, -3], [-3, 3], [-6, 3]],
                             color="#6aaa74", alpha=0.7, zorder=2)
    ax.add_patch(green_det)
    ax.text(-4.5, 0, "Green\n(detached)", color="white", fontsize=6.5,
            ha="center", va="center", zorder=5, fontweight="bold")

    # Building footprint
    building = plt.Rectangle((3, 1.5), 3.5, 3.5, color="#8b8b8b", alpha=0.85, zorder=3)
    ax.add_patch(building)
    ax.text(4.75, 3.25, "Building", color="white", fontsize=7,
            ha="center", va="center", zorder=5, fontweight="bold")

    # Circles for radii
    for r, lbl in zip([2.5, 5.0, 7.5], ["2.5 m", "5 m", "7.5 m"]):
        circ = plt.Circle((0, 0), r, fill=False, edgecolor="black",
                           linewidth=1.3, linestyle="--", zorder=6)
        ax.add_patch(circ)
        ax.text(r * 0.72, r * 0.72, lbl, fontsize=7.5, color="black",
                ha="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, ec="none"))

    # Tree trunk symbol
    trunk = plt.Circle((0, 0), 0.35, color="#5a3a1a", zorder=8)
    ax.add_patch(trunk)
    crown_circ = plt.Circle((0, 0), 1.2, color="#2e7d32", alpha=0.7, zorder=7)
    ax.add_patch(crown_circ)
    ax.text(0, -1.8, "Tree", fontsize=7.5, ha="center", color="#2e7d32",
            fontweight="bold", zorder=9)

ax_diag = axes[0]
draw_land_use_diagram(ax_diag, "(a) Land use categories at three radii")

# Legend for (a)
legend_patches = [
    mpatches.Patch(color="#3a7d44", alpha=0.7, label="Green space (attached)"),
    mpatches.Patch(color="#6aaa74", alpha=0.7, label="Green space (detached)"),
    mpatches.Patch(color="#8b8b8b", alpha=0.85, label="Building footprint"),
    mpatches.Patch(color="#b0b0b0", label="Sealed surface (derived)"),
    Line2D([0], [0], color="black", linestyle="--", label="Analysis radii"),
]
ax_diag.legend(handles=legend_patches, loc="lower right", fontsize=7.5,
               framealpha=0.9)

# ── Right panel: bar chart of theoretical annulus areas ──────────────────
ax_bar = axes[1]
radii_labels = ["0–2.5 m\n(r = 2.5 m)", "2.5–5 m\n(annulus)", "5–7.5 m\n(annulus)"]
areas = [np.pi * 2.5**2, np.pi * (5.0**2 - 2.5**2), np.pi * (7.5**2 - 5.0**2)]
bar_col = ["#e8c87a", "#c8d8a0", "#a0c8b0"]
bars = ax_bar.bar(radii_labels, areas, color=bar_col, edgecolor="gray",
                  linewidth=0.8, width=0.55)
for bar, area in zip(bars, areas):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{area:.1f} m²", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_bar.set_ylabel("Theoretical annulus area (m²)", fontsize=9)
ax_bar.set_title("(b) Theoretical annulus areas", fontsize=10, fontweight="bold")
ax_bar.spines[["top","right"]].set_visible(False)
ax_bar.tick_params(labelsize=8.5)

# Add text explaining the approach
ax_bar.text(0.5, 0.95,
    "Each ring is an independent annulus.\n"
    "Sealed surface = annulus area\n"
    "− green attached − green detached\n"
    "− building footprint",
    transform=ax_bar.transAxes, ha="center", va="top", fontsize=8,
    bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="goldenrod", alpha=0.9))

fig.suptitle("Land Use Classification Around Urban Trees", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_land_use_concept.pdf", dpi=200, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_land_use_concept.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  → fig_land_use_concept saved.")

# ============================================================================
# FIGURE 4: Land use distributions from data
# ============================================================================
print("Generating Figure 4: Land use distributions...")

lu_cols = {
    "greenAtta2": ("greenAtta2", "#3a7d44", "Green att. 0–2.5 m"),
    "greenAtta5": ("greenAtta5", "#4a8d54", "Green att. 2.5–5 m"),
    "greenAtta7": ("greenAtta7", "#5a9d64", "Green att. 5–7.5 m"),
    "sealedSurf": ("sealedSurf", "#888888", "Sealed 0–2.5 m"),
    "sealedSu_1": ("sealedSu_1", "#aaaaaa", "Sealed 2.5–5 m"),
    "sealedSu_2": ("sealedSu_2", "#cccccc", "Sealed 5–7.5 m"),
}

present_lu = {k: v for k, v in lu_cols.items() if v[0] in all_trees.columns}

if len(present_lu) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Sealed surface fractions
    ax_seal = axes[0]
    seal_cols = [(col, lbl, clr) for k, (col, clr, lbl) in lu_cols.items()
                 if "Sealed" in lbl and col in all_trees.columns]
    theo_areas = [19.63, 58.90, 98.17]
    if seal_cols:
        box_data = []
        box_labels = []
        for (col, lbl, clr), theo in zip(seal_cols, theo_areas):
            vals = pd.to_numeric(all_trees[col], errors="coerce").dropna()
            vals = vals[(vals >= 0) & (vals <= theo + 5)]
            frac = vals / theo * 100
            box_data.append(frac)
            box_labels.append(lbl.replace("Sealed ", ""))
        bp = ax_seal.boxplot(box_data, labels=box_labels, patch_artist=True,
                             medianprops=dict(color="black", linewidth=2),
                             flierprops=dict(marker=".", markersize=2, alpha=0.3))
        for patch, col_def in zip(bp["boxes"], seal_cols):
            patch.set_facecolor(col_def[2])
            patch.set_alpha(0.7)
        ax_seal.set_ylabel("Sealed surface fraction (%)", fontsize=9)
        ax_seal.set_title("(a) Sealed surface fraction per ring", fontweight="bold", fontsize=10)
        ax_seal.axhline(50, color="tomato", linestyle=":", linewidth=1, alpha=0.7, label="50%")
        ax_seal.legend(fontsize=8)
        ax_seal.spines[["top","right"]].set_visible(False)

    # Green space (attached) fractions
    ax_green = axes[1]
    green_cols = [(col, lbl, clr) for k, (col, clr, lbl) in lu_cols.items()
                  if "Green att" in lbl and col in all_trees.columns]
    if green_cols:
        box_data_g = []
        box_labels_g = []
        for (col, lbl, clr), theo in zip(green_cols, theo_areas):
            vals = pd.to_numeric(all_trees[col], errors="coerce").dropna()
            vals = vals[(vals >= 0) & (vals <= theo + 5)]
            frac = vals / theo * 100
            box_data_g.append(frac)
            box_labels_g.append(lbl.replace("Green att. ", ""))
        bp2 = ax_green.boxplot(box_data_g, labels=box_labels_g, patch_artist=True,
                               medianprops=dict(color="black", linewidth=2),
                               flierprops=dict(marker=".", markersize=2, alpha=0.3))
        for patch in bp2["boxes"]:
            patch.set_facecolor("#3a7d44")
            patch.set_alpha(0.5)
        ax_green.set_ylabel("Green space fraction (%)", fontsize=9)
        ax_green.set_title("(b) Attached green space fraction per ring", fontweight="bold", fontsize=10)
        ax_green.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_land_use_distributions.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_land_use_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  → fig_land_use_distributions saved.")
else:
    print("  (skipped: insufficient land use columns)")

# ============================================================================
# FIGURE 5: Topographic indices distributions
# ============================================================================
print("Generating Figure 5: Topographic indices distributions...")

topo_cols = {
    "twi":    ("TWI (local, 200 m window)", "#2e86ab"),
    "tpi2m5": ("TPI (r = 2.5 m)",          "#a23b72"),
    "tpi5m":  ("TPI (r = 5.0 m)",           "#c45c13"),
    "tpi7m5": ("TPI (r = 7.5 m)",           "#e0a100"),
    "slope":  ("Slope (°)",                  "#557a7a"),
}

present_topo = {k: v for k, v in topo_cols.items() if k in all_trees.columns}
if present_topo:
    ncols = 3
    nrows = (len(present_topo) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    axes_flat = axes.flatten() if nrows > 1 else axes

    for ax_idx, (col, (label, color)) in enumerate(present_topo.items()):
        ax = axes_flat[ax_idx]
        vals = pd.to_numeric(all_trees[col], errors="coerce").dropna()
        # Remove extreme outliers
        q1, q99 = vals.quantile(0.01), vals.quantile(0.99)
        vals = vals[(vals >= q1) & (vals <= q99)]
        ax.hist(vals, bins=35, color=color, edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.axvline(vals.median(), color="black", linestyle="--", linewidth=1.5,
                   label=f"Median: {vals.median():.2f}")
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(f"{label}\n(n={len(vals)})", fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)

    for ax_idx in range(len(present_topo), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle("Topographic Index Distributions Across All Sites",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_topographic_distributions.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_topographic_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  → fig_topographic_distributions saved.")
else:
    print("  (skipped: no topographic columns found)")

# ============================================================================
# FIGURE 6: TWI and TPI methodology schematic
# ============================================================================
print("Generating Figure 6: TWI/TPI methodology schematic...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Topographic Index Computation Methods", fontsize=12, fontweight="bold", y=1.01)

# ── 6a: D8 flow direction ─────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 4.5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("(a) D8 flow accumulation\n(TWI basis)", fontweight="bold", fontsize=10)

# 5x5 DEM grid with example elevations
dem_demo = np.array([
    [52, 51, 50, 48, 45],
    [53, 52, 49, 46, 43],
    [54, 53, 51, 47, 42],
    [55, 54, 52, 50, 46],
    [56, 55, 54, 53, 51],
])
dem_demo = dem_demo[::-1, :]  # flip for display

cmap = plt.cm.terrain
norm = mcolors.Normalize(vmin=40, vmax=60)

for row in range(5):
    for col in range(5):
        elev = dem_demo[row, col]
        fc = cmap(norm(elev))
        rect = mpatches.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                   facecolor=fc, edgecolor="white", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(col, row, str(elev), ha="center", va="center",
                fontsize=7, color="white" if elev < 50 else "black",
                fontweight="bold")

# Draw a flow arrow from (3,3) -> (2,2) -> (1,1)
flow_path = [(3.5-0.5, 3.5-0.5), (2.5-0.5, 2.5-0.5), (1.5-0.5, 1.5-0.5)]
for i in range(len(flow_path) - 1):
    x0, y0 = flow_path[i][0], flow_path[i][1]
    x1, y1 = flow_path[i+1][0], flow_path[i+1][1]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="blue",
                                lw=2, mutation_scale=12))

ax.text(0.02, 0.02,
    "TWI = ln(A / tan β)\n"
    "A = upslope area (m²)\n"
    "β = local slope gradient",
    transform=ax.transAxes, fontsize=8,
    bbox=dict(boxstyle="round,pad=0.4", fc="lightblue", alpha=0.85))

# ── 6b: TPI neighbourhood ─────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_xlim(-4.5, 4.5)
ax2.set_ylim(-4.5, 4.5)
ax2.set_aspect("equal")
ax2.axis("off")
ax2.set_title("(b) Topographic Position Index (TPI)\nCircular neighbourhood (centre excluded)",
              fontweight="bold", fontsize=10)

for r_m, (r_col, r_label) in zip([2.5, 5.0, 7.5],
                                   [("#a23b72", "r=2.5 m"), ("#c45c13", "r=5.0 m"),
                                    ("#e0a100", "r=7.5 m")]):
    circ = plt.Circle((0, 0), r_m * 0.55, fill=False, edgecolor=r_col,
                       linewidth=2, linestyle="--", zorder=4)
    ax2.add_patch(circ)
    ax2.text(r_m * 0.55 * 0.72, r_m * 0.55 * 0.72, r_label,
             color=r_col, fontsize=8, ha="center", fontweight="bold")

ax2.add_patch(plt.Circle((0, 0), 0.22, color="#333", zorder=5))
ax2.text(0, 0, "z₀", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

ax2.text(0.02, 0.02,
    "TPI = z₀ − mean(z_neighbours)\n"
    "  > 0 : ridge / elevated\n"
    "  < 0 : valley / depression\n"
    "  ≈ 0 : flat terrain",
    transform=ax2.transAxes, fontsize=8,
    bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9))

# fill neighbourhood cells schematically
for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
    x = np.cos(angle) * 2.0
    y = np.sin(angle) * 2.0
    ax2.scatter(x, y, s=50, color="#c45c13", zorder=3, alpha=0.7)

# ── 6c: Example TWI vs sealing ────────────────────────────────────────────
ax3 = axes[2]
if "twi" in all_trees.columns and "sealedSurf" in all_trees.columns:
    twi_vals = pd.to_numeric(all_trees["twi"], errors="coerce")
    seal_vals = pd.to_numeric(all_trees["sealedSurf"], errors="coerce")
    mask = twi_vals.notna() & seal_vals.notna()
    twi_plot = twi_vals[mask].clip(lower=0, upper=20)
    seal_plot = seal_vals[mask] / 19.63 * 100  # fraction of 2.5 m ring
    seal_plot = seal_plot.clip(0, 100)
    ax3.scatter(twi_plot, seal_plot, s=8, alpha=0.3, color="#555", linewidths=0)
    ax3.set_xlabel("TWI (local)", fontsize=9)
    ax3.set_ylabel("Sealed surface fraction\nat 0–2.5 m (%)", fontsize=9)
    ax3.set_title("(c) TWI vs. sealed surface fraction\n(all sites)", fontweight="bold", fontsize=10)
    ax3.spines[["top","right"]].set_visible(False)
else:
    ax3.text(0.5, 0.5, "(data not available)", ha="center", transform=ax3.transAxes)
    ax3.set_title("(c) TWI vs. sealing (example)", fontweight="bold", fontsize=10)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig_twi_tpi_method.pdf", dpi=200, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_twi_tpi_method.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  → fig_twi_tpi_method saved.")

# ============================================================================
# FIGURE 7: SVF methodology schematic
# ============================================================================
print("Generating Figure 7: SVF methodology schematic...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Sky View Factor (SVF) Computation Methodology", fontsize=12, fontweight="bold")

def draw_svf_cross_section(ax, title, show_tree_correction=False):
    ax.set_xlim(-100, 100)
    ax.set_ylim(-5, 65)
    ax.set_title(title, fontweight="bold", fontsize=9)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Ground/DEM
    ground_y = 0
    ax.fill_between([-100, 100], [ground_y]*2, [-5]*2, color="#c4a882", alpha=0.8)
    ax.plot([-100, 100], [ground_y, ground_y], color="brown", linewidth=1.5)
    ax.text(-95, -3, "DEM (ground)", fontsize=7.5, color="brown")

    # Buildings profile
    building_profiles = [(-80, -40, 25), (10, 50, 35), (60, 90, 20)]
    for x0, x1, h in building_profiles:
        ax.fill_between([x0, x1], [h, h], [0, 0], color="#808080", alpha=0.85, zorder=3)

    # DSM label
    ax.text(-95, 28, "DSM", fontsize=7.5, color="#555",
            bbox=dict(boxstyle="round,pad=0.2", fc="lightgray", alpha=0.7))

    if show_tree_correction:
        # Original DSM (low at tree location)
        ax.fill_between([-20, 20], [0, 0], [0, 0], color="#4caf50", alpha=0)
        # Tree canopy replacing DSM
        for x_c, h_c in [(-10, 18), (0, 22), (10, 18)]:
            canopy = plt.Circle((x_c, h_c), 7, color="#2e7d32", alpha=0.7, zorder=5)
            ax.add_patch(canopy)
        ax.text(-12, 5, "DSM_mod:\ntree height\nreplaces DSM", fontsize=7,
                color="#2e7d32", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"))

    # Tree point
    ax.scatter([0], [0], s=60, color="darkgreen", zorder=6)
    ax.text(2, -3.5, "Tree site", fontsize=7.5, color="darkgreen")

    # SVF rays
    tree_x, tree_y = 0, 1
    n_rays = 8
    for i in range(n_rays):
        angle = np.pi / 2 + i * np.pi / (n_rays - 1)
        ray_len = 90
        dx = np.cos(angle) * ray_len
        dy = np.sin(angle) * ray_len
        # Check if ray is blocked by a building
        blocked = False
        target_x = tree_x + dx
        target_y = tree_y + dy
        for bx0, bx1, bh in building_profiles:
            if bx0 < target_x < bx1 and bh > 5:
                blocked = True
                t = (bx0 - tree_x) / (dx + 1e-12) if dx > 0 else (bx1 - tree_x) / (dx + 1e-12)
                target_x = tree_x + t * dx * 0.9
                target_y = tree_y + t * dy * 0.9
                break
        lc = "#ff6b35" if blocked else "steelblue"
        ax.annotate("", xy=(target_x, target_y), xytext=(tree_x, tree_y),
                    arrowprops=dict(arrowstyle="-|>", color=lc, lw=0.8,
                                    mutation_scale=6, alpha=0.6))

    # Sky hemisphere annotation
    sky_arc = Arc((tree_x, tree_y), 30, 30, angle=0,
                  theta1=0, theta2=180, color="steelblue", linewidth=1.2,
                  linestyle=":", zorder=7)
    ax.add_patch(sky_arc)

# Panel a: standard SVF (no correction)
draw_svf_cross_section(axes[0], "(a) SVF from raw DSM", show_tree_correction=False)
axes[0].text(0.5, 0.95,
    "SVF = fraction of sky hemisphere\nnot obstructed by DSM surface",
    transform=axes[0].transAxes, ha="center", va="top", fontsize=7.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.85))

# Panel b: modified DSM
draw_svf_cross_section(axes[1], "(b) SVF from modified DSM (SVF_mod)", show_tree_correction=True)
axes[1].text(0.5, 0.95,
    "DSM_mod replaces DSM with DEM+tree_height\nwhere tree canopy exceeds DSM",
    transform=axes[1].transAxes, ha="center", va="top", fontsize=7.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.85))

# Panel c: SVF vs SVF_mod scatter (conceptual)
ax3 = axes[2]
np.random.seed(42)
svf_raw  = np.random.uniform(0.3, 1.0, 120)
noise    = np.random.normal(0, 0.04, 120)
svf_mod  = np.clip(svf_raw + noise + np.where(svf_raw > 0.7, -0.06, 0.0), 0.1, 1.0)

ax3.scatter(svf_raw, svf_mod, s=18, alpha=0.5, color="orange", edgecolors="gray",
            linewidths=0.3)
ax3.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1.2,
         label="1:1 line", alpha=0.7)
ax3.set_xlabel("SVF (raw DSM)", fontsize=9)
ax3.set_ylabel("SVF_mod (modified DSM)", fontsize=9)
ax3.set_title("(c) SVF vs. SVF_mod\n(example comparison)", fontweight="bold", fontsize=9)
ax3.legend(fontsize=8)
ax3.set_xlim(0.1, 1.0)
ax3.set_ylim(0.1, 1.0)
ax3.set_aspect("equal")
ax3.spines[["top","right"]].set_visible(False)
ax3.text(0.02, 0.97,
    "Higher SVF = more open sky\n"
    "Lower SVF = obstructed sky\n"
    "(buildings/canopy)",
    transform=ax3.transAxes, ha="left", va="top", fontsize=8,
    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

plt.tight_layout()
fig.savefig(FIG_DIR / "fig_svf_method.pdf", dpi=200, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_svf_method.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  → fig_svf_method saved.")

# ============================================================================
# FIGURE 8: Event detection methodology
# ============================================================================
print("Generating Figure 8: Event detection methodology...")

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38)

# ── 8a: Synthetic time series showing rain event ─────────────────────────
ax_ts = fig.add_subplot(gs[0:2, 0])
np.random.seed(7)

# Construct realistic synthetic VWC + rain series
time = pd.date_range("2024-06-01", periods=24*4*7, freq="15min")
base10  = 22.0 + np.random.normal(0, 0.3, len(time))
base30  = 28.0 + np.random.normal(0, 0.2, len(time))
base45  = 33.0 + np.random.normal(0, 0.15, len(time))

# Apply a rain event at t = 48 h into the series
ev_idx = int(48 * 4)
response = np.zeros(len(time))
for i in range(ev_idx, len(time)):
    t_h = (i - ev_idx) / 4.0
    if t_h < 60:
        response[i] = 5.5 * np.exp(-t_h / 28) * (1 - np.exp(-t_h / 3))

vwc10 = (base10 + response * 1.0).cumsum() * 0 + base10 + response * 1.0
vwc30 = (base30 + response * 0.75).cumsum() * 0 + base30 + response * 0.75
vwc45 = (base45 + response * 0.5).cumsum() * 0 + base45 + response * 0.5

# Rain
rain = np.zeros(len(time))
rain[ev_idx:ev_idx+8] = [0.0, 0.5, 2.1, 3.8, 2.5, 1.2, 0.5, 0.2]

t0 = time[max(0, ev_idx - int(12*4))]
t1 = time[min(len(time)-1, ev_idx + int(7*24*4))]

colors_depth = {"10 cm": "#1b7837", "30 cm": "#762a83", "45 cm": "#e08214"}
depth_data   = {"10 cm": vwc10, "30 cm": vwc30, "45 cm": vwc45}

# VWC subplot (top)
ax_ts_rain = fig.add_subplot(gs[0, 0])
for depth, vwc in depth_data.items():
    mask_t = (time >= t0) & (time <= t1)
    ax_ts_rain.plot(time[mask_t], vwc[mask_t], linewidth=1.3,
                    color=colors_depth[depth], label=f"VWC {depth}")

event_t = time[ev_idx]
ax_ts_rain.axvline(event_t, color="steelblue", linestyle="--", linewidth=2, label="Event onset")

pre_vwc = vwc10[ev_idx - 8] if ev_idx > 8 else vwc10[0]
peak_val = vwc10[ev_idx:ev_idx+int(6*4)].max()
ax_ts_rain.annotate("", xy=(event_t + pd.Timedelta(hours=6), peak_val),
                    xytext=(event_t + pd.Timedelta(hours=6), pre_vwc),
                    arrowprops=dict(arrowstyle="<->", color="#1b7837", lw=1.8, mutation_scale=12))
ax_ts_rain.text(event_t + pd.Timedelta(hours=7.5), (pre_vwc + peak_val)/2,
               "Δ VWC\n(amplitude)", fontsize=7.5, color="#1b7837", va="center")

ax_ts_rain.set_ylabel("VWC (%)", fontsize=8)
ax_ts_rain.set_title("(a) Example precipitation event: soil moisture response",
                     fontweight="bold", fontsize=9)
ax_ts_rain.legend(fontsize=7, ncol=2, loc="upper right")
ax_ts_rain.spines[["top","right"]].set_visible(False)
ax_ts_rain.tick_params(axis="x", labelbottom=False)

# Rain subplot (bottom)
ax_ts_r = fig.add_subplot(gs[1, 0], sharex=ax_ts_rain)
mask_t = (time >= t0) & (time <= t1)
ax_ts_r.bar(time[mask_t], rain[mask_t], width=pd.Timedelta(minutes=15),
            color="steelblue", alpha=0.6, label="Precipitation (mm/15min)")
ax_ts_r.axvline(event_t, color="steelblue", linestyle="--", linewidth=2)
ax_ts_r.set_ylabel("Rain (mm)", fontsize=8)
ax_ts_r.spines[["top","right"]].set_visible(False)
ax_ts_r.tick_params(axis="x", rotation=25, labelsize=7.5)
ax_ts_r.legend(fontsize=7, loc="upper right")

# Add drying time annotation
dry_idx = ev_idx + int(40*4)
if dry_idx < len(time):
    ax_ts_rain.annotate("", xy=(time[dry_idx], pre_vwc + 0.5),
                        xytext=(time[ev_idx + int(6*4)], pre_vwc + 0.5),
                        arrowprops=dict(arrowstyle="<->", color="saddlebrown",
                                        lw=1.5, mutation_scale=10))
    ax_ts_rain.text(time[ev_idx + int(23*4)], pre_vwc - 0.8, "Retention time (dry_h)",
                   fontsize=7.5, color="saddlebrown", ha="center")

# ── 8b: Algorithm flowchart ──────────────────────────────────────────────
ax_flow = fig.add_subplot(gs[:, 1])
ax_flow.set_xlim(0, 10)
ax_flow.set_ylim(0, 24)
ax_flow.axis("off")
ax_flow.set_title("(b) Two-path event detection algorithm", fontweight="bold", fontsize=10)

def draw_box(ax, x, y, w, h, text, fc="#d0e8ff", ec="#3a7abf", fontsize=8.5, bold=False):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.15",
                          facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    fw = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fw, zorder=4, wrap=True,
            multialignment="center")

def draw_arrow(ax, x0, y0, x1, y1, label="", color="gray"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                                mutation_scale=10))
    if label:
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        ax.text(mx + 0.15, my, label, fontsize=7.5, color=color, va="center")

# Start
draw_box(ax_flow, 5, 23, 7, 1.0, "Sensor time-series\n(VWC at 10/30/45 cm + rain)", fc="#e8f4e8", ec="#4caf50", bold=True)

# Two paths
draw_arrow(ax_flow, 3.0, 22.5, 2.5, 21.2, color="#c45c13")
draw_arrow(ax_flow, 7.0, 22.5, 7.5, 21.2, color="#2e86ab")

# Path 1: VWC jump
draw_box(ax_flow, 2.5, 20.5, 4.0, 1.2,
         "Path 1 — VWC jump\nΔVWC ≥ 1.5 %-pts at\nALL 3 depths (30-min step)",
         fc="#ffeedd", ec="#c45c13", fontsize=7.5)
draw_arrow(ax_flow, 2.5, 19.9, 2.5, 18.9, color="#c45c13")
draw_box(ax_flow, 2.5, 18.3, 4.0, 1.0,
         "Persistence check\npost-median − pre-median\n≥ 0.5% over 2 h",
         fc="#fff3e0", ec="#c45c13", fontsize=7.5)

# Path 2: rain peak
draw_box(ax_flow, 7.5, 20.5, 4.0, 1.2,
         "Path 2 — Rain peak\n24-h rolling sum ≥ 4 mm\nlocal maximum",
         fc="#ddeeff", ec="#2e86ab", fontsize=7.5)
draw_arrow(ax_flow, 7.5, 19.9, 7.5, 18.9, color="#2e86ab")
draw_box(ax_flow, 7.5, 18.3, 4.0, 1.0,
         "Onset refinement\nscan back 6 h for\nearliest VWC rise ≥ 1%",
         fc="#e0f0ff", ec="#2e86ab", fontsize=7.5)

# Merge
draw_arrow(ax_flow, 2.5, 17.8, 5.0, 16.5, color="#555")
draw_arrow(ax_flow, 7.5, 17.8, 5.0, 16.5, color="#555")
draw_box(ax_flow, 5.0, 15.9, 6.0, 1.0,
         "Event merger\n(VWC+RAIN within 12 h → single event)",
         fc="#f3e5f5", ec="#7b1fa2", fontsize=7.5, bold=True)
draw_arrow(ax_flow, 5.0, 15.4, 5.0, 14.4, color="#7b1fa2")

# Classification
draw_box(ax_flow, 5.0, 13.8, 6.0, 1.0,
         "Classification\nRain ≥ 2 mm in ±12 h → RAIN\notherwise → IRRIGATION",
         fc="#e8f5e9", ec="#388e3c", fontsize=7.5)
draw_arrow(ax_flow, 5.0, 13.3, 5.0, 12.3, color="#388e3c")

# Feature extraction
draw_box(ax_flow, 5.0, 11.7, 7.5, 1.0,
         "Feature extraction per event × depth\n"
         "pre_vwc · delta_vwc · t_peak_min · dry_h · temp_sum · add_rain",
         fc="#fff8e1", ec="#f57f17", fontsize=7.5, bold=True)
draw_arrow(ax_flow, 5.0, 11.2, 5.0, 10.2, color="#f57f17")

# Output
draw_box(ax_flow, 5.0, 9.6, 7.5, 1.0,
         "Output: per-tree CSV + annotated plots\n"
         "Combined: irrigation_events_all.csv",
         fc="#e8eaf6", ec="#3f51b5", fontsize=7.5, bold=True)

# Config box
ax_flow.text(0.01, 0.02,
    "Key parameters:\n"
    "JUMP_THR = 1.5 %/30 min\n"
    "PERSIST_H = 2 h\n"
    "RAIN_EVENT_MIN = 4 mm\n"
    "EVENT_MERGE_H = 12 h\n"
    "DRY_MAX_H = 168 h (7 d)\n"
    "DRYING_FRAC = 10%",
    transform=ax_flow.transAxes, fontsize=7.5,
    bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="gray", alpha=0.95),
    va="bottom")

# ── 8c: VWC metrics per depth ─────────────────────────────────────────────
ax_met = fig.add_subplot(gs[2, 0])
metrics = ["delta_vwc", "t_peak_min / 60", "dry_h"]
metric_labels = ["Amplitude ΔVWC (%)", "Time to peak (h)", "Retention time (h)"]
depths = ["−10 cm", "−30 cm", "−45 cm"]
example_vals = np.array([
    [5.2, 3.8, 2.1],   # amplitude
    [1.5, 2.2, 3.0],   # time to peak
    [32, 45, 58],       # drying hours
])
x = np.arange(3)
width = 0.25
depth_colors_bar = ["#1b7837", "#762a83", "#e08214"]
for i, (depth, dc) in enumerate(zip(depths, depth_colors_bar)):
    vals_norm = example_vals[:, i] / example_vals.max(axis=1)
    ax_met.bar(x + i * width, vals_norm * 100, width, label=depth,
               color=dc, alpha=0.8, edgecolor="white")
ax_met.set_xticks(x + width)
ax_met.set_xticklabels(metric_labels, fontsize=8)
ax_met.set_ylabel("Relative magnitude (%)", fontsize=8)
ax_met.set_title("(c) Extracted event features by depth (schematic)", fontweight="bold", fontsize=9)
ax_met.legend(fontsize=7.5)
ax_met.spines[["top","right"]].set_visible(False)

fig.suptitle("Precipitation and Irrigation Event Detection Framework", fontsize=13,
             fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_event_detection.pdf", dpi=200, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_event_detection.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  → fig_event_detection saved.")

# ============================================================================
# FIGURE 9: Data flow / system architecture
# ============================================================================
print("Generating Figure 9: Data pipeline architecture...")

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_title("TreeDataBase: Sensor-to-Analysis Data Pipeline",
             fontsize=12, fontweight="bold", pad=10)

components = [
    # (x, y, w, h, label, fc, ec)
    (1.2,  5.5, 2.0, 1.0, "Climavi Sensor\n(in-soil, 3 depths)\nNB-IoT/LoRaWAN", "#e8f5e9", "#388e3c"),
    (1.2,  3.5, 2.0, 1.0, "QGIS Tree Database\n(treeLocations.shp)\n53 attributes/tree", "#e3f2fd", "#1976d2"),
    (1.2,  1.5, 2.0, 1.0, "DEM/DSM\n(LiDAR, open data)\n1 m resolution", "#fff3e0", "#f57c00"),
    (4.5,  5.5, 2.2, 1.0, "Climavi Cloud\n(ThingsBoard API)\nMeteoblue weather", "#f3e5f5", "#7b1fa2"),
    (4.5,  3.5, 2.2, 1.0, "Manual digitisation\n(QGIS polygons)\nLand use at 3 radii", "#e3f2fd", "#1565c0"),
    (4.5,  1.5, 2.2, 1.0, "GIS processing\n(QGIS Python scripts)\nTWI · TPI · SVF", "#fff8e1", "#f57f17"),
    (8.0,  4.5, 2.2, 1.0, "fetch_timeseries.py\n→ sensor_data.csv\n→ JSON snapshots", "#f3e5f5", "#9c27b0"),
    (8.0,  2.5, 2.2, 1.0, "compute_landuse.py\n→ land use fields\n  in treeLocations.shp", "#e8f5e9", "#388e3c"),
    (11.0, 4.5, 2.2, 1.0, "_poc_irrigation.py\nEvent detection\nFeature extraction", "#fff3e0", "#e64a19"),
    (11.0, 2.5, 2.2, 1.0, "urban_tree_report.py\nDataset statistics\nPDF + JSON export", "#e3f2fd", "#0288d1"),
    (11.0, 0.8, 2.2, 0.8, "irrigation_events_all.csv\ntree_data.json per tree\ndataset_report.pdf", "#e8eaf6", "#3f51b5"),
]

for x, y, w, h, text, fc, ec in components:
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.12",
                          facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=7.2, zorder=3,
            multialignment="center")

# Arrows
arrows = [
    (2.2, 5.5, 3.4, 5.5),
    (2.2, 3.5, 3.4, 3.5),
    (2.2, 1.5, 3.4, 1.5),
    (5.6, 5.5, 6.9, 5.0),
    (5.6, 3.5, 6.9, 3.0),
    (5.6, 1.5, 6.9, 2.0),
    (9.1, 4.5, 9.9, 4.5),
    (9.1, 2.5, 9.9, 2.5),
    (9.1, 4.2, 9.9, 2.8),
    (12.1, 4.0, 12.1, 3.1),
    (11.0, 2.0, 11.0, 1.2),
]
for x0, y0, x1, y1 in arrows:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="gray",
                                lw=1.2, mutation_scale=10))

# Section labels
for lbl, x, y in [("Input data", 1.2, 6.7), ("Data management", 4.5, 6.7),
                   ("Processing", 8.0, 6.7), ("Analysis & Output", 11.0, 6.7)]:
    ax.text(x, y, lbl, ha="center", va="center", fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="darkgray", alpha=0.9))
    ax.axvline(x - 1.5 if x < 3 else (x - 1.7 if x < 6 else (x - 1.6 if x < 10 else x - 1.7)),
               color="lightgray", linestyle=":", linewidth=0.8, ymin=0, ymax=0.93)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig_data_pipeline.pdf", dpi=200, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_data_pipeline.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  → fig_data_pipeline saved.")

# ============================================================================
# PRINT SUMMARY STATISTICS for the paper
# ============================================================================
print("\n" + "="*60)
print("DATASET SUMMARY STATISTICS")
print("="*60)
print(f"Total trees loaded: {len(all_trees)}")
print(f"Number of cities:   {len(frames)}")
if "genus" in all_trees.columns:
    n_genera = all_trees["genus"].dropna().str.strip().str.capitalize().nunique()
    print(f"Unique genera:      {n_genera}")
if "species" in all_trees.columns:
    n_species = all_trees["species"].dropna().nunique()
    print(f"Unique species:     {n_species}")
if "tree_age" in all_trees.columns:
    age = all_trees["tree_age"].dropna()
    age = age[(age > 0) & (age < 300)]
    if len(age):
        print(f"Tree age range:     {age.min():.0f}–{age.max():.0f} years (median {age.median():.0f})")
if "stemDiam" in all_trees.columns:
    sd = pd.to_numeric(all_trees["stemDiam"], errors="coerce").dropna()
    sd = sd[(sd > 0) & (sd < 5)]
    if sd.median() < 5:
        sd_cm = sd * 100
    else:
        sd_cm = sd
    if len(sd_cm):
        print(f"Stem diameter:      {sd_cm.min():.1f}–{sd_cm.max():.1f} cm (median {sd_cm.median():.1f} cm)")
if "height" in all_trees.columns:
    ht = pd.to_numeric(all_trees["height"], errors="coerce").dropna()
    ht = ht[(ht > 0) & (ht < 50)]
    if len(ht):
        print(f"Tree height:        {ht.min():.1f}–{ht.max():.1f} m (median {ht.median():.1f} m)")
if "twi" in all_trees.columns:
    twi_vals = pd.to_numeric(all_trees["twi"], errors="coerce").dropna()
    print(f"TWI range:          {twi_vals.min():.2f}–{twi_vals.max():.2f} (median {twi_vals.median():.2f})")
print("="*60)
print(f"\nAll figures saved to: {FIG_DIR}")
print("Done.")
