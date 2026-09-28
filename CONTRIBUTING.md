# Contributing and collaboration

This dataset grows with every season, and it grows faster with help. We welcome
contributions of every size — from a single stem-diameter measurement to a full
measurement campaign.

## Ways to help

**Measure a tree.** Even one stem-diameter measurement per year on a monitored tree
is valuable: it turns a static attribute into a growth record. If you live or work
near one of the monitored sites, contact us and we will tell you which trees are
close by and how to measure them consistently (height above ground, tape vs. caliper).
Every contributed measurement is gratefully included in `treeLocations.shp` /
`all_tree_locations.csv` and credited.

**Add data points or a measurement campaign.** Additional attributes, repeated
vitality assessments, root-zone observations, infiltration tests, or a new city
deployment — all of these extend the dataset. We can share our field protocols and
the QGIS templates in `GISData/AEmptyLayers/` so that new data slot straight into the
existing structure.

**Help us analyse the soil samples that are waiting.** A large number of soil samples
from monitored trees have been collected and are stored, but not yet analysed. The
laboratory work (pH, conductivity, soluble nutrients, particle-size distribution,
water-retention curves) is the bottleneck. Financial support for laboratory analyses,
or access to laboratory infrastructure, would directly enlarge the soil-attribute
coverage of the dataset. Please get in touch if you can help on either side.

**Photographs of the trees.** Pictures of the monitored trees in multiple seasons — for
Potsdam, including three tree-appraisal campaigns and soil-profile photographs, and for
the other deployments — are available on request, as are the raw sensor time series of
the Potsdam and Hagen deployments.

## How to contribute data

1. Open an issue describing what you measured or collected (city, tree name or
   `devEUI`, date, method), or e-mail the maintainer.
2. For tabular contributions, a CSV with the columns `devEUI, date, attribute, value,
   unit, method, contributor` is ideal.
3. For new spatial data, follow the layer structure in `GISData/AEmptyLayers/` and the
   land-use digitising rules described in the README.

## How to contribute code

- Open an issue first for anything beyond a small fix.
- Keep scripts runnable from a fresh clone: paths relative to the repository root,
  no hard-coded local directories, dependencies listed in `requirements.txt`.
- Do not commit API keys (`src/api_key.txt` is ignored), DEM/DSM tiles, or anything
  received from a partner city under a non-public license.

## Contact

Thomas Maier — thomas.maier@fau.de
