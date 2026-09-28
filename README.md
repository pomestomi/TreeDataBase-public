# Urban Tree Soil Moisture Dataset (TreeDataBase)

A georeferenced dataset of urban trees across Central European cities, each equipped
with an IoT soil-moisture sensor from Agvolution's [climavi](https://climavi.eu) system.
The dataset combines continuous sensor telemetry at three depths with hourly gridded
weather data, tree-level attributes (species, morphology, vitality), hand-digitised
land-use context, topographic indices and soil laboratory analyses, plus the derived
rain- and irrigation-event tables and the Random-Forest / SHAP analyses built on them.

**Snapshot in this repository:** 555 tree sites in 45 deployment projects (Germany,
Austria, Switzerland); sensor data from June 2022; 25 819 classified rain events
(February 2024 – July 2026) at 510 trees; 3 355 detected irrigation events.

A data paper describing the dataset and the analyses is under review. Until it is
published, please cite the repository via `CITATION.cff` (see [Citation](#citation)).

---

## Start here: the QGIS project

**`GISData/AllTreesCadastre.qgz`** is the central entry point to the dataset and the best
place to begin. Opening it in [QGIS](https://qgis.org) (3.28 or newer) gives you a map
dashboard of every monitored tree across all deployments, with the hand-digitised land-use
rings, the tree attributes in the layer tables, and satellite and aerial imagery as
background. From there you can click any tree, read its attributes, note its `devEUI`, and
follow that identifier into the time series and event tables described below.

All layer paths in the project are relative, so it works straight from a clone with no
configuration: clone the repository, open the `.qgz`, and the layers load from
`GISData/*/VectorLayers/`. A handful of background layers stream from public web-map
services and need an internet connection; everything else is local.

Recommended path into the data:

1. Open `GISData/AllTreesCadastre.qgz` to get an overview and pick trees of interest.
2. Read this README's [dataset](#static-tree-attributes) and
   [event](#derived-event-tables) sections for the meaning of the fields.
3. Work with `TreeTabularData/rf_dataset.csv` (events joined with site attributes) or with
   the per-tree files under `TreeTabularData/trees/{devEUI}/`.
4. Skim `TreeTabularData/dataset_report.pdf` for the automated overview of the whole network.

---

## Contents

- [Start here: the QGIS project](#start-here-the-qgis-project)
- [What is in this repository](#what-is-in-this-repository)
- [What is not in this repository](#what-is-not-in-this-repository)
- [Data collection methodology](#data-collection-methodology)
- [Time-series parameters](#time-series-parameters)
- [Static tree attributes](#static-tree-attributes)
- [Derived event tables](#derived-event-tables)
- [Scripts and reproduction](#scripts-and-reproduction)
- [Contributing and collaboration](#contributing-and-collaboration)
- [Deployment projects](#deployment-projects)
- [Accessing live sensor data](#accessing-live-sensor-data)
- [Citation](#citation) · [License](#license) · [Contact](#contact)

---

## What is in this repository

```
TreeDataBase/
├── README.md · LICENSE (code, MIT) · LICENSE-DATA.md (data, CC BY 4.0)
├── CITATION.cff · CONTRIBUTING.md · requirements.txt
│
├── TreeTabularData/                    # the dataset
│   ├── trees/{devEUI}/
│   │   ├── sensor_data.csv             # full time series, 28 parameters (30-min resolution)
│   │   │                               #   (Potsdam and Hagen: on request, see RAW_DATA_ON_REQUEST.txt)
│   │   ├── tree_data.json              # versioned tree attribute snapshots
│   │   ├── latest_attributes.json      # rain sums, forecasts, irrigation recommendation
│   │   ├── bsc_events/bsc_events.csv   # rain events classified for this tree
│   │   └── Irrigation Events/          # irrigation events detected for this tree
│   ├── all_tree_locations.csv          # one row per tree site, ~80 attributes
│   ├── bsc_events_all.csv              # all classified rain events (25 819)
│   ├── irrigation_events_all.csv       # irrigation + rain events, detector v4
│   ├── irrigation_events_v2_all.csv    # event table with per-depth response metrics
│   ├── rf_dataset.csv                  # rain events joined with site attributes (model input)
│   ├── dataset_report.pdf              # automated dataset report
│   └── _report_assets/                 # charts used in the report
│
├── GISData/
│   ├── AllTreesCadastre.qgz            # ► START HERE: QGIS dashboard of the whole network
│   ├── {Project}/VectorLayers/         # treeLocations.shp + hand-digitised land-use rings
│   │                                   #   greenAtta{2m5,5m0,7m5}, greenDeta*, buildings*
│   ├── AEmptyLayers/                   # QGIS layer templates for new deployments
│   └── APythonCodes/                   # terrain, SVF, DEM download, groundwater scripts
│
├── src/                                # data pipeline
│   ├── fetch_timeseries.py             # climavi API → sensor_data.csv (needs an API key)
│   ├── compute_landuse.py              # polygon layers → land-use fields
│   ├── add_sensor_dates.py             # installation / removal dates
│   ├── _poc_irrigation_v3.py           # BSC rain-event classifier
│   ├── _poc_irrigation_v4.py           # irrigation detector (majority rule)
│   └── detect_irrigation_events.py     # event detector with per-depth response metrics
│
├── DatasetStatistics/                  # analyses
│   ├── build_rf_dataset.py             # join events with site attributes → rf_dataset.csv
│   ├── analyze_rf_soil_moisture_v7.py  # RF + SHAP: soil-moisture increase (production models)
│   ├── analyze_rf_drying_time_v7.py    # RF + SHAP: drying time (production models)
│   ├── feature_selection.txt           # candidate-feature configuration
│   ├── rf_*_v7_training_cache.pkl      # trained-model caches (regenerate figures without retraining)
│   ├── rf_*_v7_report.pdf · *_results.csv
│   ├── validate_irrigation_detection.py# detector validation against documented irrigation
│   ├── generate_manuscript_gaps.py     # SHAP beeswarm figures
│   ├── urban_tree_report.py            # dataset_report.pdf
│   └── dwd_mean_91-20/                 # DWD climate normals used for the study-area table
│
├── figures/
│   ├── make_paper_figures.py · generate_study_area_map.py · generate_figures.py
│   ├── paper/                          # all manuscript figures (PDF + PNG)
│   ├── appendix/                       # rain-event threshold analysis figure
│   └── tables/                         # climate table
└── README.md · LICENSE · CITATION.cff · CONTRIBUTING.md
```

## What is not in this repository

- **Raw sensor time series of the Potsdam and Hagen deployments**, and **the photographs
  of the Potsdam trees** (three tree-appraisal campaigns and the soil-profile pictures,
  with their appraisal documentation). Both are made available upon special request to the
  authors or curators of this dataset. See
  `TreeTabularData/trees/README_RESTRICTED_RAW_DATA.txt`, the note in each affected tree
  folder, and `GISData/CityPotsdam/PICTURES_ON_REQUEST.txt`.
  The tree locations, land-use layers, tree attributes and all events derived from these
  measurements (classified rain events, irrigation events, the aggregated event tables)
  are included.
- **Municipal tree cadastres** and other data received from partner cities and
  institutions (cadastre extracts, consultant reports, soil-survey documents). These were
  used during data collection but are not ours to redistribute.
- **DEM / DSM tiles** (several GB). They are open federal-state data and the scripts in
  `GISData/APythonCodes/` download them again; only the derived terrain indices are kept.
- **Private weather-station records** used for cross-checks.
- **Regenerable plots** (per-sensor summary plots, per-event plots, example plots). The
  scripts recreate them from the CSV files.
- The manuscript sources.

Photographs of the monitored trees in multiple seasons, and material from the list above
where the license allows, are available on request. See
[Contributing and collaboration](#contributing-and-collaboration).

---

## Data collection methodology

### Sensor hardware

Each monitored tree is equipped with a **climavi Soil Underground** sensor unit
(Agvolution GmbH). The probe is installed vertically in the root zone and measures at
three fixed depths (**10, 30 and 45 cm** below the surface), each with a capacitive
moisture sensor and a temperature sensor. The housing also carries a dry-bulb and a
wet-bulb thermometer near the soil surface.

Sensors transmit via **LoRaWAN** or **NB-IoT / LTE-M** to the climavi cloud platform
(ThingsBoard-based), typically every 30 minutes. Each sensor is identified by its
**devEUI**, which is the primary key linking every file in this repository.

### Weather data

Every sensor location is enriched server-side with hourly gridded weather from
**Meteoblue**: air temperature, relative humidity, dew point, air pressure,
precipitation, wind speed and direction, evapotranspiration, and modelled soil
temperature and moisture. Precipitation is available from February 2024 onward.

### Land-use classification

Land use around each tree was digitised manually in QGIS at three radii from the trunk
centre (**2.5 m, 5.0 m, 7.5 m**), each as an annulus (0–2.5, 2.5–5.0, 5.0–7.5 m;
theoretical areas 19.63, 58.90 and 98.17 m²). Three categories are digitised as polygon
layers per ring; sealed surface is derived by subtraction.

| Category | Description | Layer |
|---|---|---|
| Attached green | Permeable vegetated area hydraulically connected to the root zone | `greenAtta*.shp` |
| Detached green | Permeable vegetated area within the ring but disconnected (e.g. by a curb) | `greenDeta*.shp` |
| Buildings | Building footprints | `buildings*.shp` |
| Sealed surface | Impervious surfaces, computed as annulus area minus the other three | (derived) |

Each polygon carries the `devEUI` of its tree. `src/compute_landuse.py` computes the
areas and writes them to `treeLocations.shp`; if digitised areas slightly exceed the
annulus, all categories are scaled proportionally. A second set of four fields uses a
tree-specific radius of half the crown diameter (`*CD` fields, capped at 7.5 m).

### Terrain indices

From 1 m federal-state DEM/DSM data: topographic wetness index (`twi`, 200 m window),
slope, flow accumulation, topographic position index at 2.5 / 5 / 7.5 m radius
(`tpi2m5`, `tpi5m`, `tpi7m5`) and at crown radius (`tpiCD`), sky-view factor (`SVF`),
and depth to groundwater (`DTGW`) where public groundwater maps exist. Scripts in
`GISData/APythonCodes/`.

### Static tree attributes and soil samples

Tree-level metadata is maintained in QGIS (`treeLocations.shp`, one per project):
species, morphology (height, crown diameter, stem diameter), planting date, vitality,
land use, terrain, and, where soil samples have been analysed, pH, conductivity,
soluble nutrients, particle-size fractions and water-retention values (pF 1.5–4.2).

---

## Time-series parameters

`sensor_data.csv` holds up to 28 columns at 30-minute resolution. Not every parameter
exists for every device (modular hardware, changing firmware).

**Soil sensors** (depth −10, −30, −45 cm): `-{depth}|ENV__SOIL__VWC` (% volumetric water
content), `-{depth}|ENV__SOIL__T` (°C), `-{depth}|ENV__SOIL__CAPACITANCE__ABSOLUTE` (raw).

**On-device atmosphere:** `TOP|ENV__ATMO__T`, `TOP|ENV__ATMO__T__DRY`, `TOP|ENV__ATMO__T__WET` (°C).

**Meteoblue (prefix `MTB|`):** `ENV__ATMO__T` (°C), `ENV__ATMO__RH` (%), `ENV__ATMO__P` (hPa),
`ENV__ATMO__DEWPOINT` (°C), `ENV__ATMO__RAIN__DELTA` (mm per interval), `ENV__ATMO__WIND__SPEED`
(m/s), `ENV__ATMO__WIND__DIRECTION` (°), `ENV__SOIL__ET` (mm), `ENV__SOIL__T` (°C), `ENV__SOIL__VWC` (%).

**Device telemetry:** `DOC|ENV__SOIL__IRRIGATION` (L, documented irrigation),
`DEVICE|DEV__ENERGY__VCAP` (%), `DEVICE|DEV__ENERGY__VBAT` (mV), `DEVICE|DEV__RF__RSSI`,
`DEVICE|DEV__RF__SINR`, `DEVICE|DEV__RF__RSRP`, `DEVICE|DEV__RF__RSRQ`.

`latest_attributes.json` is a snapshot (not a series): rain sums over 24 h – 30 d, a 7-day
VWC forecast per depth, and the platform's irrigation recommendation.

---

## Static tree attributes

`all_tree_locations.csv` (and `tree_data.json` per tree) carry roughly 80 fields:

| Group | Fields |
|---|---|
| Identification | `fid`, `devEUI`, `treeName`, `project`, `latitude`, `longitude`, `amsl`, `_source_city_folder` |
| Tree | `genus`, `species`, `variety`, `height` (m), `crownDiam` (m), `stemDiam` (m), `vitality` |
| Dates | `germDate`, `plantDate`, `cutDwnDate`, `assesDate`, `soilDate`, `senInsDate`, `senRmvDate`, `Note` |
| Land use (m²) | `greenAtta2/5/7`, `greenDeta2/5/7`, `buildings2/5/7`, `sealedSu2/5/7`, crown-radius variants `greenAttCD`, `greenDetCD`, `buildingCD`, `sealedSuCD` |
| Terrain | `twi`, `tpi2m5`, `tpi5m`, `tpi7m5`, `tpiCD`, `slope`, `flowAcc` / `flotAcc`, `elevation`, `SVF`, `DTGW` |
| Soil chemistry | `pH`, `saltCont`, `conductiv`, `nSoluble`, `ammNSolubl`, `nitrNSolub`, `mgSoluble`, `phSoluble`, `kSoluble`, `naSoluble`, `clSoluble` |
| Soil physics | `part1Perc` … `part7Perc` (particle-size fractions, %), `Kf`, `tpv`, `dbd`, `coarFrag`, `pF1.5` … `pF4.2` |

`DTGW` is a mixed column: numeric depth in metres where a groundwater map allowed a value,
otherwise a class string (e.g. `"> 10 m (interpoliert)"`).

---

## Derived event tables

**Rain events: `bsc_events_all.csv` / `rf_dataset.csv`.** Events are detected on the
hourly Meteoblue precipitation series (≥ 5 mm total, ≥ 3 rainy hours, separated by ≥ 6 dry
hours) and classified with the **Binary Shape Code** of Terranova & Iaquinta (2011): the
cumulative rainfall profile is compared with a uniform profile in four time quartiles,
giving a 4-bit code (`1111` = front-loaded … `0000` = back-loaded), plus the Huff quartile
of the intensity peak. For each event and depth the tables record the pre-event VWC, the
peak and its timing, the rise as % of the sensor's dynamic range (`delta_pct_dyn`), and the
drying time back to baseline + 10 % (`dry_h`). `rf_dataset.csv` joins these events with all
site attributes. The 5 mm event-detection threshold is examined in
`figures/appendix/fig_rain_threshold.pdf`.

**Irrigation events: `irrigation_events_all.csv`, `irrigation_events_v2_all.csv`.** A
simultaneous VWC jump in at least two of the three depths, confirmed by a 2-hour persistence
check and outside rain windows. Validation against 1 548 field-documented irrigation
records at 121 trees: 1 336 assessable, recall 0.76, precision ≥ 0.31 (lower bound, since
many detections are undocumented irrigation); `DatasetStatistics/validate_irrigation_detection.py`
reproduces the validation.

**Models.** Twelve Random-Forest models (two targets × two event strata × three depths)
with Boruta feature selection and SHAP attribution; results, caches and reports in
`DatasetStatistics/`, figures in `figures/paper/`.

---

## Scripts and reproduction

```bash
pip install -r requirements.txt

# 1  land use from the QGIS polygon layers        → treeLocations.shp fields
python src/compute_landuse.py
# 2  time series from the climavi API (needs src/api_key.txt) → TreeTabularData/trees/*/sensor_data.csv
python src/fetch_timeseries.py
# 3  installation / removal dates
python src/add_sensor_dates.py
# 4  events
python src/_poc_irrigation_v3.py                  # BSC rain events   → bsc_events_all.csv
python src/_poc_irrigation_v4.py                  # irrigation events → irrigation_events_all.csv
# 5  model dataset and analyses
python DatasetStatistics/build_rf_dataset.py      # → rf_dataset.csv
python DatasetStatistics/analyze_rf_soil_moisture_v7.py
python DatasetStatistics/analyze_rf_drying_time_v7.py
python DatasetStatistics/validate_irrigation_detection.py
# 6  report and figures
python DatasetStatistics/urban_tree_report.py     # → dataset_report.pdf
python DatasetStatistics/generate_manuscript_gaps.py   # SHAP figures → figures/paper
python figures/generate_study_area_map.py         # study-area map (downloads Natural Earth borders once)
python figures/make_paper_figures.py              # remaining manuscript figures
```

The time-series fetcher requires a climavi API key in `src/api_key.txt` (ignored by git);
everything downstream of step 2 runs on the CSV files shipped in this repository. The
terrain scripts (`GISData/APythonCodes/`) additionally need GDAL/rasterio and re-download
the DEM/DSM tiles. Model training uses fixed seeds (`RF_SEED = 42`); the SHAP figures can be
regenerated from the shipped caches without retraining.

---

## Contributing and collaboration

This dataset is a living one, and we would like it to grow together with the community.

- **Pictures of the trees.** Photographs of the monitored trees, in multiple seasons, for
  Potsdam and for the other deployments, are available on request.
- **Data points and measurement campaigns.** Support with additional data points or with
  measurement campaigns at the monitored sites is encouraged and very welcome. We share our
  field protocols and QGIS templates so that new data fit the existing structure.
- **Soil samples waiting for analysis.** Many soil samples from monitored trees are on
  standby and have not been analysed yet. Support on the financial or infrastructural side
  (funding for laboratory analyses or access to a laboratory) would help a lot and would
  directly enlarge the soil-attribute coverage of the dataset.
- **Everybody can help collect data.** Even a measurement of stem diameter repeated over
  the years is a genuine contribution: it turns a static attribute into a growth record.
  Such measurements are thankfully included in the dataset and credited.

See `CONTRIBUTING.md` for how to submit data or code, or simply write to the contact below.

---

## Deployment projects

Germany: Aachen · Bamberg · Berlin Friedrichshain-Kreuzberg · Berlin Neukölln · Biberach ·
Bremen (Botanical Garden, company site) · Celle · Crailsheim · Effeltrich · Erfurt ·
Erlangen (city, company, university) · Garbsen · Grünwald · Hagen · Hanover · Hassfurt ·
Heidelberg (city, company) · Hildesheim · Homburg · Ingolstadt · Kassel · Leipzig · Lemgo ·
Lüdenscheid · Neunkirchen · Nörvenich · Nuremberg · Pappenheim · Pforzheim · Pillnitz ·
Pirmasens · Plön · Potsdam · Saarbrücken · Schlösserland Sachsen · Stein · Weisendorf
Austria: Vienna (city, Botanical Garden, company) · Salzburg (university)
Switzerland: Laax

Deployments range from 1 to 134 sensors per project.

---

## Accessing live sensor data

The time series can also be browsed interactively on the climavi platform at
[https://app.climavi.eu](https://app.climavi.eu). The `devEUI` of each tree links the
static attributes in this dataset to the live and historical data. REST API documentation:
[https://app.climavi.eu/app/user/me/api-docs](https://app.climavi.eu/app/user/me/api-docs).

---

## Citation

Until the data paper is published, please cite the repository:

> Maier, T. et al. (2026). *Urban Tree Soil Moisture Dataset (TreeDataBase)*, version 1.0.0.
> GitHub repository. (Machine-readable metadata in `CITATION.cff`.)

The paper reference will be added here on publication.

## License

- **Code** (`src/`, `GISData/APythonCodes/`, `DatasetStatistics/`, `figures/*.py`): MIT, see `LICENSE`.
- **Data, figures and photographs**: Creative Commons Attribution 4.0 (CC BY 4.0), see `LICENSE-DATA.md`.

## Contact

**Thomas Maier** - thomas.maier@fau.de, t.maier@agvolution.com
Friedrich-Alexander-Universität Erlangen-Nürnberg, Agvolution GmbH - [https://agvolution.com](https://agvolution.com) · climavi platform - [https://climavi.eu](https://climavi.eu)
