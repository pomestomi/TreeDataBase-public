# Data License — Creative Commons Attribution 4.0 International (CC BY 4.0)

The **data** in this repository is released for public use under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

You are free to **share** (copy and redistribute in any medium or format) and to
**adapt** (remix, transform and build upon) the material for any purpose,
including commercially, provided you give **appropriate credit** (see Citation in
the README), link to the license, and indicate if changes were made.

Full legal text: https://creativecommons.org/licenses/by/4.0/legalcode

## Scope

This license applies to:

- `TreeTabularData/` — sensor time series (`trees/*/sensor_data.csv`), tree attribute
  snapshots (`tree_data.json`, `latest_attributes.json`), the derived event tables
  (`bsc_events_all.csv`, `irrigation_events*.csv`, `rf_dataset.csv`),
  `all_tree_locations.csv` and the report assets
- `GISData/*/VectorLayers/` — sensor tree locations and the hand-digitised land-use
  polygon layers
- `figures/` — all figures and tables
- `DatasetStatistics/` — model results, caches and reports

Source code is licensed separately under the MIT License (see `LICENSE`).

## Third-party data

- Meteoblue weather variables (`MTB|…` columns) are provided through the Climavi platform;
  redistribution here is limited to the values at the sensor locations as stored in
  `sensor_data.csv`.
- DWD climate normals in `DatasetStatistics/dwd_mean_91-20/` are © Deutscher Wetterdienst,
  GeoNutzungsverordnung / DL-DE-BY-2.0.
- Terrain products (TWI, TPI, slope, SVF, flow accumulation) were derived from open
  federal-state DEM/DSM data (GovData Datenlizenz Deutschland 2.0, data.gv.at CC BY 4.0);
  the raw tiles are not redistributed and can be re-downloaded with the scripts in
  `GISData/APythonCodes/`.
- Municipal tree cadastres and partner reports used during data collection are **not**
  part of this repository.
- The raw sensor time series of the Potsdam and Hagen deployments and the photographs of
  the Potsdam trees are **not** part of this repository; they are available on request and
  are covered by this license once shared.
