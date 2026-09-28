#!/usr/bin/env python3
"""
Generate a publication-quality study area map for the manuscript.

Shows all monitoring cities with their sensor counts as sized bubbles on a
clean vector map with country borders.  Automatically updates when new
treeLocations.shp files are added to GISData/.

First run downloads Natural Earth 1:50m country borders (~440 KB, one-time).

Run from the repo root:
    python figures/generate_study_area_map.py

Output:
    figures/paper/fig_study_area_map.pdf
    figures/paper/fig_study_area_map.png
"""
import io
import urllib.request
import warnings
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import geopandas as gpd

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "Arial", "font.weight": "bold", "font.size": 8,
    "axes.labelsize": 8, "axes.labelweight": "bold",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8,
})
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT  = SCRIPT_DIR.parent
GIS_DIR    = REPO_ROOT / "GISData"
FIG_DIR    = SCRIPT_DIR / "paper"
DATA_DIR   = SCRIPT_DIR / "data"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

NE_SHP  = DATA_DIR / "ne_50m_admin_0_countries.shp"
NE_URL  = ("https://naciscdn.org/naturalearth/50m/cultural/"
           "ne_50m_admin_0_countries.zip")

NE_STATES_SHP = DATA_DIR / "ne_10m_admin_1_states_provinces.shp"
NE_STATES_URL = ("https://naciscdn.org/naturalearth/10m/cultural/"
                 "ne_10m_admin_1_states_provinces.zip")

# ---------------------------------------------------------------------------
# City directories excluded from the study
# ---------------------------------------------------------------------------
EXCLUDED_DIRS = {"CityKarben", "CityWeißenburg", "CityOffenbach"}

# ---------------------------------------------------------------------------
# Map city_dir → normalised geographic city name
# Multiple deployment types (City/Company/University/Botanical) at the same
# location are merged into one bubble so each geographic city appears once.
# ---------------------------------------------------------------------------
GEO_CITY = {
    # ── Berlin districts → single Berlin bubble ──────────────────────────────
    "CityBerlinFrhXBerg":        "Berlin",
    "CityBerlinNeukölln":        "Berlin",
    "CityBerlinCharlottenburg":  "Berlin",
    # ── Erlangen area (City + Company + University) ───────────────────────────
    "CityErlangen":              "Erlangen",
    "CompanyErlangen":           "Erlangen",
    "UniversityErlangen":        "Erlangen",
    # ── Bremen (City + Company + Botanical Garden) ────────────────────────────
    "CityBremen":                "Bremen",
    "CompanyBremen":             "Bremen",
    "BotanicalGardenBremen":     "Bremen",
    # ── Vienna (City + Company + Botanical Garden) ────────────────────────────
    "CityVienna":                "Vienna",
    "CompanyVienna":             "Vienna",
    "BotanicalGardenVienna":     "Vienna",
    # ── Erfurt (City + Company) ───────────────────────────────────────────────
    "CityErfurt":                "Erfurt",
    "CompanyErfurt":             "Erfurt",
    # ── Salzburg (University, two dir names) ─────────────────────────────────
    "UniversitySalzburg":        "Salzburg",
    "CityUniversitySalzburg":    "Salzburg",
    # ── Single-location entries ───────────────────────────────────────────────
    "CityAachen":                "Aachen",
    "CityBamberg":               "Bamberg",
    "CityBiberach":              "Biberach",
    "CityCelle":                 "Celle",
    "CityCrailsheim":            "Crailsheim",
    "CityGarbsen":               "Garbsen",
    "CityGrünwald":              "Grünwald",
    "CityHagen":                 "Hagen",
    "CityHanover":               "Hannover",
    "CityHassfurt":              "Hassfurt",
    "CityHildesheim":            "Hildesheim",
    "CityIngolstadt":            "Ingolstadt",
    "CityKassel":                "Kassel",
    "CityLeipzig":               "Leipzig",
    "CityLemgo":                 "Lemgo",
    "CityLüdenscheid":           "Lüdenscheid",
    "CityNeunkirchen":           "Neunkirchen",
    "CityNürnberg":              "Nürnberg",
    "CityPforzheim":             "Pforzheim",
    "CityPirmasens":             "Pirmasens",
    "CityPotsdam":               "Potsdam",
    "CityStein":                 "Nürnberg",    # Nürnberg-Stein → merged into Nürnberg
    "CityWeisendorf":            "Weisendorf",
    "CompanyEffeltrich":         "Effeltrich",
    "CompanyHeidelberg":         "Heidelberg",
    "CompanyHomburg":            "Homburg",
    "CompanyNörvenich":          "Aachen",      # Nörvenich → merged into Aachen
    # CompanyPappenheim: sensors split by coordinates → see COORD_CITY below
    "CompanyPillnitz":           "Pillnitz",
    "CompanyPlön":               "Plön",
    "CompanySaarbrücken":        "Saarbrücken",
    "CastleAdminSaxony":         "Dresden",
    "CompanyLaax":               "Laax",
}

# Per-sensor city override, keyed by f"{lon:.2f},{lat:.2f}".
# Used for directories whose sensors span multiple geographic cities.
COORD_CITY = {
    "11.40,48.78": "Ingolstadt",       # CompanyPappenheim sensor inside Ingolstadt city
    "11.36,49.04": "Greding",          # CompanyPappenheim sensor near Greding
    "10.84,48.38": "Augsburg",         # CompanyPappenheim sensor in Augsburg
    "16.26,47.83": "Wiener Neustadt",  # CompanyVienna outlier sensor ~40 km south of Vienna
    "13.87,51.01": "Pillnitz",         # CastleAdminSaxony sensors clustered in Pillnitz
    # UniversitySalzburg sensors outside Salzburg city
    "12.74,47.41": "Leogang",
    "12.74,47.40": "Leogang",
    "13.42,47.19": "Riedingtal",
    "13.40,47.19": "Riedingtal",
}

# Manual sensor-count overrides (applied after shapefile aggregation).
CITY_N_OVERRIDE = {
    "Salzburg": 3,   # includes 0101002443000667 (Ursprung) which lacks GPS coords
}


# ---------------------------------------------------------------------------
# Natural Earth download
# ---------------------------------------------------------------------------
def get_country_borders() -> gpd.GeoDataFrame:
    """Return Natural Earth 1:50m Admin-0 country polygons, downloading once."""
    if not NE_SHP.exists():
        print(f"  Downloading Natural Earth 50m country borders ...")
        with urllib.request.urlopen(NE_URL, timeout=30) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
            zf.extractall(DATA_DIR)
        print("  Download complete.")
    world = gpd.read_file(NE_SHP)[["NAME", "SOVEREIGNT", "geometry"]]
    world.columns = ["name", "sovereign", "geometry"]
    return world


def get_state_borders() -> gpd.GeoDataFrame:
    """Return Natural Earth 1:10m Admin-1 state/province borders for DE, AT, CH."""
    if not NE_STATES_SHP.exists():
        print("  Downloading Natural Earth 10m state/province borders (~30 MB) ...")
        with urllib.request.urlopen(NE_STATES_URL, timeout=60) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
            zf.extractall(DATA_DIR)
        print("  Download complete.")
    states = gpd.read_file(NE_STATES_SHP)
    dach = states[states["admin"].isin(["Germany", "Austria", "Switzerland"])].copy()
    if dach.empty:
        print(f"  Warning: DACH filter returned 0 rows. Sample admin values: "
              f"{states['admin'].dropna().unique()[:10].tolist()}")
    if dach.crs and dach.crs.to_epsg() != 4326:
        dach = dach.to_crs(epsg=4326)
    return dach[["name", "admin", "geometry"]]


# ---------------------------------------------------------------------------
# Load and aggregate tree locations
# ---------------------------------------------------------------------------
def load_city_stats() -> gpd.GeoDataFrame:
    """
    Reads all treeLocations.shp files, maps each to a geographic city via
    GEO_CITY, and returns a GeoDataFrame with per-city sensor counts and
    representative coordinates (mean centroid in WGS84).
    """
    records = []
    for shp in sorted(GIS_DIR.glob("*/VectorLayers/treeLocations.shp")):
        city_dir = shp.parent.parent.name
        if city_dir in EXCLUDED_DIRS:
            continue
        geo_city = GEO_CITY.get(city_dir)
        if geo_city is None:
            for pfx in ("City", "Company", "BotanicalGarden", "University", "Castle"):
                if city_dir.startswith(pfx):
                    geo_city = city_dir[len(pfx):]
                    break
            else:
                geo_city = city_dir
        try:
            gdf = gpd.read_file(shp).to_crs("EPSG:4326")
            for geom in gdf.geometry:
                c = geom.centroid
                coord_key = f"{c.x:.2f},{c.y:.2f}"
                city = COORD_CITY.get(coord_key, geo_city)
                records.append({"city": city, "lon": c.x, "lat": c.y})
        except Exception as exc:
            print(f"  Warning: could not read {shp}: {exc}")

    import pandas as pd
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No tree shapefiles found under {GIS_DIR}")

    stats = (df.groupby("city")
               .agg(n=("lon", "count"), lon=("lon", "mean"), lat=("lat", "mean"))
               .reset_index())
    for city, n in CITY_N_OVERRIDE.items():
        stats.loc[stats["city"] == city, "n"] = n
    return stats


# ---------------------------------------------------------------------------
# Bubble repulsion — push overlapping bubbles apart in display space
# ---------------------------------------------------------------------------
def separate_bubbles(lons, lats, sizes, ax, fig, gap_pt=0.0, n_iter=1800):
    """
    Pack bubbles as close to geographic positions as possible without overlap.
    Phase 1 (repulsion): push overlapping pairs apart.
    Phase 2 (gravity):   pull each bubble toward its geographic origin if the
                         move does not create a new overlap.
    Works in display coordinates (canvas pixels at fig.dpi).
    Returns jittered (lons, lats) arrays.
    """
    trans     = ax.transData
    inv       = trans.inverted()
    px_per_pt = fig.dpi / 72.0

    pts      = np.array([trans.transform((lo, la))
                         for lo, la in zip(lons, lats)], dtype=float)
    orig_pts = pts.copy()
    radii_px = np.sqrt(np.asarray(sizes, float) / np.pi) * px_per_pt
    gap_px   = gap_pt * px_per_pt
    rng      = np.random.default_rng(0)

    for _ in range(n_iter):
        moved = False

        # ── Repulsion: push overlapping pairs apart ───────────────────────────
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d    = pts[j] - pts[i]
                dist = float(np.hypot(d[0], d[1]))
                need = radii_px[i] + radii_px[j] + gap_px
                if dist < need:
                    push = (need - dist) * 0.55
                    if dist < 1e-6:
                        ang  = rng.uniform(0, 2 * np.pi)
                        d    = np.array([np.cos(ang), np.sin(ang)])
                        dist = 1.0
                    else:
                        d = d / dist
                    pts[i] -= d * push
                    pts[j] += d * push
                    moved = True

        # ── Gravity: pull each bubble toward its geographic origin ────────────
        for i in range(len(pts)):
            disp     = orig_pts[i] - pts[i]
            disp_len = float(np.hypot(disp[0], disp[1]))
            if disp_len < 0.3:
                continue
            step = disp * (min(0.22 * disp_len, 2.5) / disp_len)
            test = pts[i] + step
            ok   = True
            for j in range(len(pts)):
                if j == i:
                    continue
                dist = float(np.hypot(pts[j][0] - test[0], pts[j][1] - test[1]))
                if dist < radii_px[i] + radii_px[j] + gap_px - 0.1:
                    ok = False
                    break
            if ok:
                pts[i] = test
                moved   = True

        if not moved:
            break

    result = np.array([inv.transform(p) for p in pts])
    return result[:, 0], result[:, 1]


# ---------------------------------------------------------------------------
# Map figure
# ---------------------------------------------------------------------------
def make_map(city_stats, world: gpd.GeoDataFrame,
             states: gpd.GeoDataFrame = None) -> plt.Figure:
    """
    Build a publication map with side-panel city labels and separated bubbles.
    Figure is 9.5 in × 4.7 in (≈ 241 mm × 119 mm).
    """
    # Extent: Germany + neighbours; extended east/south to show full Austria
    LON_MIN, LON_MAX = 5.2,  17.5
    LAT_MIN, LAT_MAX = 46.0, 55.3

    lat_c = (LAT_MIN + LAT_MAX) / 2.0
    asp   = 1.0 / np.cos(np.radians(lat_c))   # ≈ 1.59 at ~51°N

    # ── Figure and axes layout ────────────────────────────────────────────────
    # Side margins hold city labels; map takes the centre portion.
    AX_L, AX_B, AX_W, AX_H = 0.265, 0.04, 0.490, 0.93
    fig_w = 170 / 25.4  # manuscript width limit
    # Compute height so the map fills the axes exactly (no vertical letterboxing):
    # map display h/w = (lat_range * asp) / lon_range; axes_w = AX_W * fig_w
    fig_h = (fig_w * AX_W) * ((LAT_MAX - LAT_MIN) * asp / (LON_MAX - LON_MIN)) / AX_H
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax  = fig.add_axes([AX_L, AX_B, AX_W, AX_H])

    # ── Background ────────────────────────────────────────────────────────────
    ax.set_facecolor("#EBF2FB")

    # ── Country fills ─────────────────────────────────────────────────────────
    show = ["Germany", "France", "Switzerland", "Austria",
            "Poland", "Czechia", "Netherlands",
            "Belgium", "Luxembourg", "Denmark", "Croatia", "Hungary",
            "Liechtenstein", "Italy", "Slovenia", "Slovakia"]
    subset = world[world["name"].isin(show)]
    # Germany, Austria, Switzerland share the same fill (all contain sensors
    # or border Germany); all other neighbours use a lighter fill.
    dach   = subset[subset["name"].isin(["Germany", "Austria", "Switzerland"])]
    others = subset[~subset["name"].isin(["Germany", "Austria", "Switzerland"])]

    # Neighbours: fill + thin border (50m, not critical for resolution)
    others.plot(ax=ax, facecolor="#EDECEA", edgecolor="#BBBBBB",
                linewidth=0.45, zorder=1)
    # DACH fill only — all borders come from the 10m states layer below
    dach.plot  (ax=ax, facecolor="#F5F4F0", edgecolor="none", zorder=2)

    # ── State / canton borders (all from 10m → single resolution source) ──────
    if states is not None and not states.empty:
        try:
            from shapely.ops import unary_union as _unary_union
            # Interior state borders — light grey, thin
            states.boundary.plot(ax=ax, color="#AAAAAA", linewidth=0.30, zorder=2.5)
            # DACH outer border (external boundary of dissolved union) — dark grey
            _outer = gpd.GeoSeries([_unary_union(states.geometry)], crs=states.crs)
            _outer.boundary.plot(ax=ax, color="#666666", linewidth=0.85, zorder=3)
            # Inter-DACH country borders (DE-AT, DE-CH, AT-CH) — same dark grey.
            # These are interior to the dissolved union so they need explicit drawing.
            _by_country = states.dissolve(by="admin")
            _cgeoms = {c: _by_country.loc[c, "geometry"]
                       for c in ["Germany", "Austria", "Switzerland"]
                       if c in _by_country.index}
            _clist = list(_cgeoms.values())
            _inter = [_clist[i].intersection(_clist[j])
                      for i in range(len(_clist))
                      for j in range(i + 1, len(_clist))]
            _inter = [g for g in _inter if not g.is_empty]
            if _inter:
                gpd.GeoSeries(_inter, crs=states.crs).plot(
                    ax=ax, color="#666666", linewidth=0.85, zorder=3)
        except Exception as _exc:
            print(f"  Warning: could not plot state borders: {_exc}")
            dach.plot(ax=ax, facecolor="none", edgecolor="#666666",
                      linewidth=0.85, zorder=2)

    # ── Country name labels ───────────────────────────────────────────────────
    _country_labels = {
        "Germany":       (10.5, 51.5),
        "Poland":        (16.2, 53.0),
        "Czechia":       (15.4, 49.8),
        "Austria":       (14.3, 47.8),
        "Switzerland":   ( 7.5, 46.7),
        "France":        ( 6.8, 48.2),
        "Nether-\nlands": ( 6.05, 52.4),
        "Italy":         (11.5, 46.3),
    }
    _country_stroke = [pe.withStroke(linewidth=3.5, foreground="white")]
    for cname, (clon, clat) in _country_labels.items():
        if LON_MIN < clon < LON_MAX and LAT_MIN < clat < LAT_MAX:
            ax.text(clon, clat, cname, fontsize=6.0, color="#666666",
                    ha="center", va="center", style="italic", zorder=6,
                    path_effects=_country_stroke)

    # ── Axes limits, aspect, ticks, grid ─────────────────────────────────────
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect(asp)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.tick_params(labelsize=8, direction="out", length=3, width=0.6)
    ax.yaxis.set_zorder(5)   # render tick labels above the white cover rectangle (zorder=4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.grid(True, lw=0.3, color="#CCCCCC", linestyle=":", zorder=0)

    # ── Bubble sizing ─────────────────────────────────────────────────────────
    n_vals  = city_stats["n"].values.astype(float)
    n_max   = float(n_vals.max())
    MIN_S, MAX_S = 18, 220
    sizes = MIN_S + np.sqrt(n_vals / n_max) * (MAX_S - MIN_S)

    # ── Establish display transform before repulsion ──────────────────────────
    fig.canvas.draw()

    # Small white rectangles behind each y-axis tick label, covering annotation
    # lines (zorder=3) only where tick text sits. yaxis at zorder=5 renders on top.
    from matplotlib.patches import Rectangle as _Rect
    _rect_h = 0.034   # height per tick in axes fraction (≈ 1.7× 6 pt font at this scale)
    for _ty in ax.yaxis.get_majorticklocs():
        _, _y_ax = ax.transAxes.inverted().transform(ax.transData.transform((0, _ty)))
        ax.add_patch(_Rect(
            (-0.110, _y_ax - _rect_h / 2), 0.100, _rect_h,
            transform=ax.transAxes,
            facecolor="white", edgecolor="none",
            zorder=4, clip_on=False,
        ))

    lons_jit, lats_jit = separate_bubbles(
        city_stats["lon"].values.copy(),
        city_stats["lat"].values.copy(),
        sizes, ax, fig, gap_pt=0.0,
    )

    # ── Plot bubbles at jittered positions ────────────────────────────────────
    ax.scatter(lons_jit, lats_jit,
               s=sizes, c="#8B1A1A", alpha=0.90,
               edgecolors="white", linewidths=0.7, zorder=5,
               clip_on=False)

    for idx in range(len(city_stats)):
        n_s = int(city_stats.iloc[idx]["n"])
        fs  = 6.0
        ax.text(lons_jit[idx], lats_jit[idx], str(n_s),
                ha="center", va="center_baseline",
                fontsize=fs, fontweight="bold",
                color="white", zorder=6, clip_on=False)

    # ── Side-panel labels ─────────────────────────────────────────────────────
    # Cities with original lon < LON_SPLIT go to the LEFT panel, rest to the RIGHT.
    LON_SPLIT = 10.4

    left_mask  = city_stats["lon"].values < LON_SPLIT
    right_mask = ~left_mask

    left_idx  = np.where(left_mask)[0]
    right_idx = np.where(right_mask)[0]

    # Sort by jittered latitude (descending = highest city at top of panel)
    # Using jittered positions avoids line crossings after bubble repulsion.
    left_idx  = left_idx [np.argsort(-lats_jit[left_idx ])]
    right_idx = right_idx[np.argsort(-lats_jit[right_idx])]

    # Custom override for right panel: move Vienna and Wiener Neustadt to appear
    # between Greding and Ingolstadt (user preference for line routing).
    _rnames  = city_stats["city"].values
    _move_r  = {"Vienna", "Wiener Neustadt"}
    _keep_r  = [i for i in right_idx if _rnames[i] not in _move_r]
    _movs_r  = sorted([i for i in right_idx if _rnames[i] in _move_r],
                      key=lambda i: -lats_jit[i])
    _ins     = next((j + 1 for j, i in enumerate(_keep_r) if _rnames[i] == "Greding"),
                    len(_keep_r))
    right_idx = np.array(_keep_r[:_ins] + _movs_r + _keep_r[_ins:])

    # ── Custom panel ordering ─────────────────────────────────────────────────
    def _swap_by_name(arr, a, b):
        lst = list(arr)
        pa = next((j for j, i in enumerate(lst) if _rnames[i] == a), None)
        pb = next((j for j, i in enumerate(lst) if _rnames[i] == b), None)
        if pa is not None and pb is not None:
            lst[pa], lst[pb] = lst[pb], lst[pa]
        return np.array(lst)

    def _move_before(arr, city, anchor):
        lst = list(arr)
        pm = next((j for j, i in enumerate(lst) if _rnames[i] == city), None)
        if pm is not None:
            item = lst.pop(pm)
            pt = next((j for j, i in enumerate(lst) if _rnames[i] == anchor), None)
            if pt is not None:
                lst.insert(pt, item)
        return np.array(lst)

    right_idx = _swap_by_name(right_idx, "Berlin", "Potsdam")
    left_idx = _swap_by_name(left_idx, "Neunkirchen", "Heidelberg")
    left_idx = _swap_by_name(left_idx, "Crailsheim", "Pforzheim")

    right_idx = _swap_by_name(right_idx, "Dresden", "Erfurt")
    right_idx = _swap_by_name(right_idx, "Pillnitz", "Hassfurt")
    right_idx = _swap_by_name(right_idx, "Leipzig", "Erfurt")
    right_idx = _move_before(right_idx, "Ingolstadt", "Vienna")
    right_idx = _move_before(right_idx, "Augsburg", "Vienna")

    def _panel_ys(n, y_top=0.95, y_bot=0.05):
        """Evenly-spaced figure-fraction y-positions for n labels."""
        if n == 1:
            return [0.5]
        return [y_top - i * (y_top - y_bot) / (n - 1) for i in range(n)]

    left_ys  = _panel_ys(len(left_idx))
    right_ys = _panel_ys(len(right_idx))

    # x-coordinates of label endpoints in AXES fraction (can be < 0 or > 1).
    # Left labels: right-aligned, with enough clearance from y-axis ticks.
    # Right labels: left-aligned, just outside the axes box.
    # Convert from figure fraction: x_axes = (x_fig - AX_L) / AX_W
    X_LEFT_AX  = (AX_L - 0.080 - AX_L) / AX_W   # = -0.080 / AX_W
    X_RIGHT_AX = 0.012 / AX_W + 1.0              # = 1 + 0.012/AX_W

    # y conversion from figure fraction to axes fraction: y_axes=(y_fig-AX_B)/AX_H
    def _fig_to_ax_y(y_fig):
        return (y_fig - AX_B) / AX_H

    _stroke = [pe.withStroke(linewidth=2.0, foreground="white")]
    _all_anns = []

    def _add_label(city_name, lon_j, lat_j, n, x_ax, y_ax, ha):
        bubble_r = float(np.sqrt(
            (MIN_S + np.sqrt(n / n_max) * (MAX_S - MIN_S)) / np.pi))
        # relpos places the arrow connection at the correct text edge:
        # right-aligned labels → connect from right edge; left-aligned → left edge.
        relpos = (1.0, 0.5) if ha == "right" else (0.0, 0.5)
        ann = ax.annotate(
            city_name,
            xy=(lon_j, lat_j),
            xycoords="data",
            xytext=(x_ax, y_ax),
            textcoords="axes fraction",
            ha=ha, va="center",
            fontsize=8,
            color="#111111",
            path_effects=_stroke,
            arrowprops=dict(
                arrowstyle="-",
                color="#BBBBBB",
                lw=0.4,
                linestyle=(0, (5, 3)),
                shrinkA=0,
                shrinkB=bubble_r + 1.5,
                relpos=relpos,
            ),
            annotation_clip=False,
            zorder=3,
        )
        _all_anns.append(ann)

    for rank, idx in enumerate(left_idx):
        row = city_stats.iloc[idx]
        _add_label(row["city"], lons_jit[idx], lats_jit[idx],
                   int(row["n"]), X_LEFT_AX, _fig_to_ax_y(left_ys[rank]), "right")

    for rank, idx in enumerate(right_idx):
        row = city_stats.iloc[idx]
        _add_label(row["city"], lons_jit[idx], lats_jit[idx],
                   int(row["n"]), X_RIGHT_AX, _fig_to_ax_y(right_ys[rank]), "left")

    # ── Legend: bubble size reference (inside map, upper-left) ───────────────
    ref_ns = [n for n in [1, 10, 50, 100] if n <= int(n_max)]
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#8B1A1A",
               markeredgecolor="white", markeredgewidth=0.4,
               markersize=0.60 * 2.0 * np.sqrt(
                   (MIN_S + np.sqrt(n / n_max) * (MAX_S - MIN_S)) / np.pi),
               label=f"n = {n}")
        for n in ref_ns
    ]
    leg = ax.legend(
        handles=legend_handles,
        title="Sensors\nper city",
        title_fontsize=8,
        fontsize=8,
        loc="upper left",
        framealpha=0.92,
        edgecolor="#BBBBBB",
        fancybox=False,
        borderpad=0.4,
        handletextpad=0.4,
        labelspacing=0.6,
    )
    leg._legend_box.align = "left"
    leg.set_zorder(10)   # float above annotation lines (zorder=3) and bubbles (zorder=5)

    return fig, _all_anns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  STUDY AREA MAP")
    print("=" * 60)

    print("\nLoading city statistics from shapefiles ...")
    city_stats = load_city_stats()
    city_stats = city_stats.sort_values("n", ascending=False).reset_index(drop=True)
    print(f"  {len(city_stats)} geographic cities, "
          f"{city_stats['n'].sum()} total sensor trees")
    for _, row in city_stats.iterrows():
        print(f"    {row['city']:25s}  n={int(row['n']):3d}  "
              f"({row['lon']:.2f}°E, {row['lat']:.2f}°N)")

    print("\nLoading Natural Earth country borders ...")
    world = get_country_borders()

    print("\nLoading Natural Earth state/canton borders ...")
    states = get_state_borders()

    print("\nGenerating map ...")
    fig, _ = make_map(city_stats, world, states=states)

    out_base = FIG_DIR / "fig_study_area_map"
    fig.savefig(str(out_base) + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.0)
    fig.savefig(str(out_base) + ".png", dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    # Trim any residual white border from the PNG
    try:
        from PIL import Image, ImageChops
        _img  = Image.open(str(out_base) + ".png").convert("RGB")
        _crop = ImageChops.difference(_img, Image.new("RGB", _img.size, (255, 255, 255))).getbbox()
        if _crop:
            _img.crop(_crop).save(str(out_base) + ".png")
    except ImportError:
        pass
    print(f"\n  -> {out_base}.pdf")
    print(f"  -> {out_base}.png")
    print("\n" + "=" * 60 + "\n  DONE\n" + "=" * 60)


if __name__ == "__main__":
    main()
