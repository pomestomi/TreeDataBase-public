from pathlib import Path
import yaml
import os
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from cmcrameri import cm

plt.rcParams.update({
    "font.family":      "Arial",
    "font.weight":      "bold",
    "font.size":        7,
    "axes.labelsize":   7,
    "axes.labelweight": "bold",
    "axes.titlesize":   7,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
})

# paths
# project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_dir = str(Path(__file__).resolve().parents[2])
config_path = os.path.join(project_dir, "GISData", "APythonCodes", "config.yaml")
sensor_loc_path = os.path.join(project_dir, "DatasetStatistics", "sensor_locations.csv")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cities = cfg["cities"]
# labels = cfg["labels"]

sensor_loc = pd.read_csv(sensor_loc_path)
cols_except_eui = sensor_loc.columns.difference(["eui"])
sensor_loc = (sensor_loc.dropna(subset=cols_except_eui, how="all").drop_duplicates())

all_trees = []
# --------------------------------------------------
for city, state in cities.items():

    city_dir = os.path.join(project_dir, "GISData", city)
    tree_path = os.path.join(city_dir, "VectorLayers", "treeLocations.shp")

    if not os.path.exists(tree_path):
        print(f"Missing: {city}")
        continue

    trees = gpd.read_file(tree_path)

    trees["folder"] = city
    trees["state"] = state

    trees = trees.dropna(subset=["geometry"])

    # convert to regular DataFrame and remove geometry
    all_trees.append(trees.drop(columns="geometry"))

df = pd.concat(all_trees, ignore_index=True).drop_duplicates()
df = df.dropna(subset=["devEUI"])

exclude_eui = cfg.get("exclude_eui", [])
if exclude_eui:
    df = df[~df["devEUI"].isin(exclude_eui)]

print("Before:", len(df))
df = df.merge(sensor_loc[['eui', 'country', 'city']],
              left_on = "devEUI", right_on = "eui",
              how = "left", indicator=True)

print(df["_merge"].value_counts())

print("After:", len(df))
# df.loc[df["folder"] == "CityLemgo", "city"] = "Lemgo"

# Format numeric columns
cols_to_convert = [
    "height", "crownDiam", "stemDiam",
    "DTGW",
    "greenAttCD", "greenDetCD", "buildingCD", "sealedSuCD",
    "greenAtta2", "greenDeta2", "buildings2", "sealedSu2",
    "greenAtta5", "greenDeta5", "buildings5", "sealedSu5",
    "greenAtta7", "greenDeta7", "buildings7", "sealedSu7",
    "twi", "tpi2m5", "tpi5m", "tpi7m5",
    "slope", "amsl"]

for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# format date columns
for col in ["germDate", "plantDate"]:
    df[col] = (
        df[col]
        .replace([None, "None", ""], np.nan)   # unify missing values
        .astype("string")                      # preserves missing values as <NA>
        .str.extract(r"^(\d{4})", expand=False)  # first 4 digits at start
        .astype("float64"))

# summarize LU rings
df["greenAtta"] = df[["greenAtta2","greenAtta5", "greenAtta7"]].sum(axis=1, min_count=1)
df["greenDeta"] = df[["greenDeta2","greenDeta5", "greenDeta7"]].sum(axis=1, min_count=1)
df["buildings"] = df[["buildings2","buildings5", "buildings7"]].sum(axis=1, min_count=1)
df["sealedSu"] = df[["sealedSu2","sealedSu5", "sealedSu7"]].sum(axis=1, min_count=1)


# calculate tree age
df["startYear"] = df["germDate"].fillna(df["plantDate"])
df["age"] = (2026 - df["startYear"])

pd.set_option("display.max_rows", None)
df["genus"] = df["genus"].replace({None: np.nan})
df["genus"].value_counts(dropna=False).sort_index(key=lambda x: x.str.lower())

#df["genus"] = df["genus"].replace("acre", "acer")
#df["genus"] = df["genus"].replace("almus", "alnus")
#df["genus"] = df["genus"].replace("amelanchieer", "amelanchier")
#df["genus"] = df["genus"].replace("capinus", "carpinus")
df["genus"] = df["genus"].replace("carpinius", "carpinus")
#df["genus"] = df["genus"].replace("Aesculus", "aesculus")
#df["genus"] = df["genus"].replace("Fagus", "fagus")
#df["genus"] = df["genus"].replace("Fraxinus", "fraxinus")
#df["genus"] = df["genus"].replace("pterocarya\n", "pterocarya")
df["genus"] = df["genus"].replace("pauwlonia tomentosa", "paulownia")
df["genus"] = df["genus"].replace("pauwlonia", "paulownia")
df["genus"] = df["genus"].replace("parottia", "parrotia")

def _add_bp_legend(ax):
    """Compact dark-brown schematic boxplot key in the upper-left corner of ax."""
    import matplotlib.patches as mpatches

    ta  = ax.transAxes
    fs  = 7
    DARK = '#2D1209'
    FILL = '#7B3D1A'
    bx, bw = 0.195, 0.016

    # hardcoded axes-fraction positions
    q1    = 0.700
    q3    = 0.820
    med   = 0.715
    mn    = 0.787
    lw_y  = 0.640
    uw_y  = 0.940
    out_y = 0.952
    gm_y  = 0.855

    # between Hagen (i=7) and Other (i=18)
    dx = 0.39
    bx = 0.195 + dx

    tL   = 0.142 + dx
    tipL = bx - bw - 0.006
    tipR = bx + bw + 0.006
    tR   = 0.262 + dx

    # background box — height 0.420 so outlier label fits
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.088 + dx, 0.562), 0.415, 0.420,
        boxstyle='round,pad=0.008', linewidth=0.5,
        edgecolor='lightgray', facecolor='white', alpha=1.0,
        transform=ta, clip_on=False, zorder=8))

    # IQR box — lw=1.0 matches seaborn boxplot patch linewidth
    ax.add_patch(mpatches.FancyBboxPatch(
        (bx - bw, q1), 2 * bw, q3 - q1,
        boxstyle='square,pad=0', linewidth=1.0,
        edgecolor=DARK, facecolor=FILL,
        transform=ta, clip_on=False, zorder=10))

    # whisker stems — lw=1.0 matches seaborn
    for y0, y1 in [(lw_y, q1), (q3, uw_y)]:
        ax.plot([bx, bx], [y0, y1], color=DARK, lw=1.0,
                transform=ta, clip_on=False, zorder=10)
    # whisker caps — lw=1.0 matches seaborn
    for y in (lw_y, uw_y):
        ax.plot([bx - bw*0.6, bx + bw*0.6], [y, y],
                color=DARK, lw=1.0, transform=ta, clip_on=False, zorder=10)

    # median — lw=1.0 matches seaborn
    ax.plot([bx-bw, bx+bw], [med, med], color=DARK, lw=1.0,
            transform=ta, clip_on=False, zorder=11)
    # per-city mean (white, half box width with round caps) — matches plot_num
    ax.plot([bx-bw*0.5, bx+bw*0.5], [mn, mn], color='white', lw=1.0,
            solid_capstyle='round', transform=ta, clip_on=False, zorder=12)
    # overall mean dashed — extends all the way to the text label (no solid leader)
    ax.plot([0.096 + dx, tR - 0.003], [gm_y, gm_y], color='gray', lw=1.3,
            linestyle='--', alpha=0.6, transform=ta, clip_on=False, zorder=9)
    # outlier dot
    ax.plot([bx], [out_y], 'o', color=DARK, ms=2.0, alpha=0.7,
            transform=ta, clip_on=False, zorder=11)

    lkw = dict(color=DARK, lw=0.45, transform=ta, clip_on=False, zorder=9)
    tkw = dict(transform=ta, fontsize=fs, color=DARK,
               clip_on=False, zorder=15, va='center')

    for label, y in [('uw', uw_y), ('Q3', q3), ('Q1', q1), ('lw', lw_y)]:
        ax.plot([tL + 0.003, tipL], [y, y], **lkw)
        ax.text(tL, y, label, ha='right', **tkw)

    # outlier, mean, median get solid leader lines; overall mean uses dashed line only
    for label, y, x0 in [
        ('outlier', out_y, bx + 0.006),
        ('mean',    mn,    tipR),
        ('median',  med,   tipR),
    ]:
        ax.plot([x0, tR - 0.003], [y, y], **lkw)
        ax.text(tR, y, label, ha='left', **tkw)
    ax.text(tR, gm_y, 'overall mean', ha='left', **tkw)


# Plot function
def plot_num(df, num_col, group_col="city", ylabel=None, threshold=1, ax=None, ylim=None, show_counts=True, stagger=True):

    # order = sorted(df[group_col].dropna().unique(), key=str.lower)
    df = df.copy()

    # collapse rare groups
    counts = df[group_col].value_counts(dropna=False)
    rare = counts[counts < threshold].index

    df[group_col] = df[group_col].replace(rare, "Other")

    # define order (alphabetical, Other last)
    order = sorted(df[group_col].dropna().unique(), key=str.lower)
    if "Other" in order:
        order.remove("Other")
        order.append("Other")

    df[group_col] = pd.Categorical(df[group_col], categories=order, ordered=True)

    n = len(order)

    # create an even number of colours
    m = n if n % 2 == 0 else n + 1

    palette_all = cm.vik(np.linspace(0.1, 0.9, m)).tolist()
    mid = m // 2
    indices = []
    for i in range(mid):
        indices.extend([i, mid + i])
    indices = indices[:n]
    palette = [palette_all[i] for i in indices]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Boxplot
    sns.boxplot(data=df, x=group_col, y=num_col, order=order, ax=ax, showfliers=False, width=0.6, palette=palette)
    # Individual observations
    # sns.stripplot(data=df, x=group_col, y=num_col, order=order, ax=ax, color="black", alpha=0.5, size=3, jitter=True)

    mask = pd.Series(False, index=df.index)

    valid = df[num_col].notna()

    outlier_mask = (
        df.loc[valid]
        .groupby(group_col)[num_col]
        .transform(
            lambda x: (
                    (x < x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))) |
                    (x > x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25)))
            )
        )
        .fillna(False)
        .astype(bool)
    )

    mask.loc[outlier_mask.index] = outlier_mask
    outliers = df[mask]

    sns.stripplot(data=outliers, x=group_col, y=num_col, order=order, ax=ax, color="black", alpha=0.5, size=3, jitter=True)

    # h-line overall mean
    overall_mean = df[num_col].mean()
    ax.axhline(overall_mean, color="gray", linestyle="--", linewidth=1.3, alpha=0.6, zorder=0)
    # box-lines group mean
    means = df.groupby(group_col)[num_col].mean()
    for i, cat in enumerate(order):
        if cat in means.index:
            ax.plot([i - 0.15, i + 0.15], [means[cat], means[cat]],
                color="white", linewidth=1, solid_capstyle="round", zorder=5)

    # write number of NA values
    n_counts = df.groupby(group_col).size()
    na_counts = df.groupby(group_col)[num_col].apply(lambda x: x.isna().sum())

    _pot_i = next((j for j, c in enumerate(order) if c == "Potsdam"), None)

    for i, cat in enumerate(order):
        if cat in n_counts.index:
            n_val = n_counts[cat]
            na_val = na_counts.get(cat, 0)
            y_off = 0.018 if (stagger and i % 2 == 1) else 0.0
            x_nudge = 0.0
            if stagger and _pot_i is not None:
                if i == _pot_i - 1:
                    x_nudge = -0.10
                elif i == _pot_i + 1:
                    x_nudge = 0.10

            if show_counts:
                ax.text(
                    i + x_nudge,
                    1.07 + y_off,
                    f"{int(n_val)}",
                    transform=ax.get_xaxis_transform(),  # x in data, y in axes coords
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=6,
                    fontweight="normal",
                    clip_on=False)

            ax.text(
                i + x_nudge,
                1.01 + y_off,
                f"{int(n_val - na_val)}",
                transform=ax.get_xaxis_transform(),  # x in data, y in axes coords
                ha="center",
                va="bottom",
                color="gray",
                fontsize=6,
                fontweight="normal",
                clip_on=False)

    ax.tick_params(axis="x", rotation=90)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel if ylabel is not None else num_col)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Vertical grid lines
    ax.grid(axis="x", color="lightgray", linestyle="-", linewidth=0.8, alpha=0.4)
    ax.set_axisbelow(True)  # draw grid behind the boxes

    return ax

# summarize number of trees per City
df["city"].value_counts()
df["city"].nunique()

df["project"].value_counts()

# Genus distribution: summary stats, too many different genus for figure
pd.set_option("display.max_rows", None)
df["genus"].value_counts()
df["genus"].nunique()
df["genus"].isna().sum()

# summarize numeric tree characteristics
df[["age", "height", "crownDiam", "stemDiam"]].describe()

# Plot numeric tree characteristics
for t in [1,5]:
    if t < 5:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
    if t >= 5:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 12))

    plot_num(df, "age", ax=axes[0], ylabel="Age [years]", threshold=t, ylim=(0,100), show_counts=True)
    plot_num(df, "height", ax=axes[1], ylabel="Tree height [m]", threshold=t, show_counts=False)
    plot_num(df, "crownDiam", ax=axes[2], ylabel="Crown diameter [m]", threshold=t, show_counts=False)
    plot_num(df, "stemDiam", ax=axes[3], ylabel="Stem diameter [m]", threshold=t, show_counts=False)

    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)
    axes[2].tick_params(labelbottom=False)
    axes[3].tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.savefig(os.path.join(project_dir, f"figures/paper/descr_stats/boxplots_trees_num_thr{t}.pdf"), bbox_inches="tight")
    plt.show()

# summarize LU
df[["greenAtta", "greenDeta", "buildings", "sealedSu"]].describe()

# Plot LU
for t in [1,5]:
    if t < 5:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
    if t >= 5:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 12))

    plot_num(df, "greenAtta", ax=axes[0], ylabel="Green attached [m$^{2}$]", threshold=t, show_counts=True)
    plot_num(df, "greenDeta", ax=axes[1], ylabel="Green detached [m$^{2}$]", threshold=t, show_counts=False)
    plot_num(df, "buildings", ax=axes[2], ylabel="Buildings [m$^{2}$]", threshold=t, show_counts=False)
    plot_num(df, "sealedSu", ax=axes[3], ylabel="Sealed surface [m$^{2}$]", threshold=t, show_counts=False)

    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)
    axes[2].tick_params(labelbottom=False)
    axes[3].tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.savefig(os.path.join(project_dir, f"figures/paper/descr_stats/boxplots_LU_sum7m5_thr{t}.pdf"), bbox_inches="tight")
    plt.show()

# summarize twi tpi
df[["twi", "tpi2m5", "tpi5m", "tpi7m5"]].describe()

# Plot twi tpi
for t in [1,5]:
    if t < 5:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
    if t >= 5:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 12))

    plot_num(df, "twi", ax=axes[0], ylabel="TWI", threshold=t, show_counts=True)
    plot_num(df, "tpi2m5", ax=axes[1], ylabel="TPI 2.5 m", threshold=t, show_counts=False)
    plot_num(df, "tpi5m", ax=axes[2], ylabel="TPI 5.0 m", threshold=t, show_counts=False)
    plot_num(df, "tpi7m5", ax=axes[3], ylabel="TPI 7.5 m", threshold=t, show_counts=False)

    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)
    axes[2].tick_params(labelbottom=False)
    axes[3].tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.savefig(os.path.join(project_dir, f"figures/paper/descr_stats/boxplots_twi_tpi_thr{t}.pdf"), bbox_inches="tight")
    plt.show()

# summarize slope, elevation
df[["slope", "amsl"]].describe()

# Plot slope, elevation
for t in [1,5]:
    if t < 5:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    if t >= 5:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 7))

    plot_num(df, "slope", ax=axes[0], ylabel="Slope [?]", threshold=t, show_counts=True)
    plot_num(df, "elevation", ax=axes[1], ylabel="Elevation [m amsl]", threshold=t, show_counts=False)
#    plot_num(df, "amsl", ax=axes[1], ylabel="Elevation [m amsl]", threshold=t, show_counts=False)

    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.savefig(os.path.join(project_dir, f"figures/paper/descr_stats/boxplots_slope_elevation_thr{t}.pdf"), bbox_inches="tight")
    plt.show()

# summarize SVF, DTGW
df[["SVF", "DTGW"]].describe()
pd.set_option("display.max_columns", None)
df.loc[df["SVF"].isna()]


# Plot SVF, DTGW
for t in [1,5]:
    if t < 5:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    if t >= 5:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 7))

    plot_num(df, "SVF", ax=axes[0], ylabel="SVF [-]", threshold=t, show_counts=True)
    plot_num(df, "DTGW", ax=axes[1], ylabel="DTGW [m]", threshold=t, show_counts=False)

    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.savefig(os.path.join(project_dir, f"figures/paper/descr_stats/boxplots_SVF_DTGW_thr{t}.pdf"), bbox_inches="tight")
    plt.show()

# Plot selected features in 1 (or 2?) figures
# crown diameter, green attached, TWI, TPI 7.5, SVF, DTGW

for t in [4, 5]:
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(170 / 25.4, 170 / 25.4))

    plot_num(df, "crownDiam", ax=axes[0,0], ylabel="Crown diameter [m]", threshold=t, show_counts=True, stagger=(t == 4))
    plot_num(df, "greenAtta", ax=axes[0,1], ylabel="Green attached [m$^{2}$]", threshold=t, show_counts=True, stagger=(t == 4))
    plot_num(df, "twi", ax=axes[1,0], ylabel="TWI", threshold=t, show_counts=False, stagger=(t == 4))
    plot_num(df, "tpi7m5", ax=axes[1,1], ylabel="TPI", threshold=t, show_counts=False, stagger=(t == 4))
    plot_num(df, "SVF", ax=axes[2,0], ylabel="SVF", threshold=t, show_counts=False, stagger=(t == 4))
    plot_num(df, "DTGW", ax=axes[2,1], ylabel="DTGW [m]", threshold=t, show_counts=False, stagger=(t == 4))
    _add_bp_legend(axes[2,1])

    axes[0,0].tick_params(labelbottom=False)
    axes[0,1].tick_params(labelbottom=False)
    axes[1,0].tick_params(labelbottom=False)
    axes[1,1].tick_params(labelbottom=False)
    axes[2,0].tick_params(axis="x", rotation=90)
    axes[2,1].tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.savefig(os.path.join(project_dir, f"figures/paper/descr_stats/boxplots_selected_thr{t}.pdf"), bbox_inches="tight")
    plt.show()