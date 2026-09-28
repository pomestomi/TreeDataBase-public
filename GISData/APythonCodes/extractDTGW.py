# Extract Depth to Groundwater
#-----------------------------
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import pandas as pd
import numpy as np

# --------------------------------------------------
# define paths
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# project_dir = "D:/TreeDataBase"
# tree_loc = "CompanyErfurt"

for tree_loc in [
                #"CityPotsdam", "CityBerlinFrhXBerg", "CityBerlinNeukölln",
                # "CityAachen", "CityHagen", "CityLüdenscheid", "CompanyNörvenich",
                # "CompanyErfurt",
                # "CityLeipzig", "CastleAdminSaxony", "CompanyPillnitz",
                # "CityPirmasens",
                 "BotanicalGardenBremen", "CompanyBremen",
                 "BotanicalGardenVienna", "CityVienna", "CompanyVienna", "UniversitySalzburg",
                 "CityBamberg", "CityIngolstadt", "CompanyPappenheim"]:

    print(f"Processing GW {tree_loc}")

    city_dir = os.path.join(project_dir, "GISData", tree_loc)
    tree_path = os.path.join(city_dir, "VectorLayers", "treeLocations.shp")
    tree_svf_path = os.path.join(city_dir, "SVF", "trees_svf.shp")
    output_dir = os.path.join(city_dir, "GW")
    output_tree_path = os.path.join(output_dir, "trees_gw.shp")

    os.makedirs(output_dir, exist_ok=True)

    if tree_loc in ["CityPotsdam", "CityBerlinFrhXBerg", "CityBerlinNeukölln"]:
        method = "shapefile"
        bundesland = "brandenburg"
        gw_path = os.path.join(output_dir, "grundwasserisohypsen_2015", "shp", "GWD_2015F.shp")
        crs_dom = "EPSG:25833"  # Brandenburg/Berlin

    if tree_loc in ["CityAachen", "CityHagen", "CityLemgo", "CityLüdenscheid", "CompanyNörvenich"]:
        method = "shapefile"
        bundesland = "nrw"
        gw_path = os.path.join(output_dir, "grundwasserisohypsen_2006_2015", "erg_gw_gleichen_1m_geglaettet.shp")
        crs_dom = "EPSG:25832"

    if tree_loc == "CompanyErfurt":
        method = "shapefile"
        bundesland = "thueringen"
        gw_path = os.path.join(output_dir, "GW_FLURABSTAND_1zu50000", "gw_flurabstand_1zu50000.shp")
        crs_dom = "EPSG:25832"

    if tree_loc in ["CityLeipzig", "CastleAdminSaxony", "CompanyPillnitz"]:
        method = "wfs"
        bundesland = "sachsen"
        crs_dom = "EPSG:25833"

    if tree_loc == "CityPirmasens":
        method = "wfs"
        bundesland = "rlp"
        crs_dom = "EPSG:25832"

    if tree_loc in ["BotanicalGardenBremen", "CompanyBremen"]:
        method = "wfs"
        bundesland = "bremen"
        crs_dom = "EPSG:31467"

    if tree_loc in ["BotanicalGardenVienna", "CityVienna", "CompanyVienna", "UniversitySalzburg"]:
        method = "shapefile"
        bundesland = "vienna_salzburg"
        gw_path = os.path.join(output_dir, "k6_4_ggw_flur", "k6_4_ggw_flur.shp")
        crs_dom = "EPSG:31287"

    if tree_loc in ["CityBamberg", "CityIngolstadt", "CompanyPappenheim"]:
        method = "shapefile"
        bundesland = "bavaria"
        gw_path = os.path.join(project_dir, "GISData", "CityBamberg", "GW", "hk100_gwl_gesamt.shp")
        crs_dom = "EPSG:25832"

    # load data and reproject — prefer SVF output (has DEM column); fall back to
    # treeLocations.shp and derive DEM from amsl when SVF has not yet been run
    if os.path.exists(tree_svf_path):
        trees = gpd.read_file(tree_svf_path)
    else:
        print(f"  trees_svf.shp not found, falling back to treeLocations.shp")
        trees = gpd.read_file(tree_path)
        if "DEM" not in trees.columns and "amsl" in trees.columns:
            trees["DEM"] = pd.to_numeric(trees["amsl"], errors="coerce")

    trees = trees.to_crs(crs_dom)
    # remove DTGW column if already exists from earlier run
    trees = trees.drop(columns=["DTGW"], errors="ignore")

    if method == "shapefile":
        if not os.path.exists(gw_path):
            print(f"  GW shapefile not found, skipping: {gw_path}")
            continue
        # read GW data
        gw = gpd.read_file(gw_path)
        gw = gw.to_crs(crs_dom)

        if bundesland == "brandenburg":
            trees = gpd.sjoin_nearest(trees, gw[["ZLEVEL", "geometry"]], how="left", distance_col='distance_to_line')
            trees = trees.drop(columns=["index_right"], errors="ignore")
            trees["DTGW"] = trees.DEM - trees.ZLEVEL

        if bundesland == "nrw":
            trees = gpd.sjoin_nearest(trees, gw[["isol_201", "geometry"]], how="left", distance_col='distance_to_line')
            trees = trees.drop(columns=["index_right"], errors="ignore")
            trees["DTGW"] = trees.DEM - trees.isol_201

        if bundesland == "thueringen":
            trees = gpd.sjoin(trees, gw, how="left", predicate="within")
            trees = trees.drop(columns=["index_right"], errors="ignore")
            trees = trees.rename(columns= {"FLURABSTAN": "DTGW"})

        if bundesland == "vienna_salzburg":
            trees = gpd.sjoin(trees, gw[["flurabst", "flurab", "geometry"]], how="left", predicate="within")
            trees = trees.drop(columns=["index_right"], errors="ignore")
            trees = trees.rename(columns={"flurab": "DTGW"})

        if bundesland == "bavaria":
            trees = gpd.sjoin_nearest(trees, gw[["HOEHE", "geometry"]], how="left", distance_col='distance_to_line')
            trees = trees.drop(columns=["index_right"], errors="ignore")
            trees["DTGW"] = trees.DEM - trees.HOEHE
            if tree_loc == 'CompanyPappenheim':
                trees.loc[trees["treeName"] == "Greding Kindergarten", "DTGW"] = np.nan
            if tree_loc == 'CityBamberg':
                trees.loc[trees["devEUI"].isin(["0201002512000229",
                                                "0201002512000205",
                                                "0201002512000215",
                                                "0201002512000223",
                                                "0201002512000245",
                                                "0201002512000367",
                                                "0201002512000368",
                                                "8C1F640980000031",
                                                "0201002512000222",
                                                "0201002512000371",
                                                "0201002512000128",
                                                "0201002512000211",
                                                "0201002512000363"]), "DTGW"] = np.nan

    if method == "wfs":
        if bundesland == "sachsen":
            # from owslib.wfs import WebFeatureService
            # url = "https://luis.sachsen.de/arcgis/services/wasser/grundwasserdynamik_2016/MapServer/WFSServer"
            # wfs = WebFeatureService(url=url, version="2.0.0")
            # for layer in wfs.contents:
            #     print(layer)

            url = "WFS:https://luis.sachsen.de/arcgis/services/wasser/grundwasserdynamik_2016/MapServer/WFSServer"
            gw = gpd.read_file(url, layer="grundwasserdynamik_2016:Grundwasserdynamik_2016_1m")
            gw = gw.to_crs(crs_dom)
            trees = gpd.sjoin_nearest(trees, gw[["Contour", "geometry"]], how="left", distance_col='distance_to_line')
            trees = trees.drop(columns=["index_right"], errors="ignore")
            trees["DTGW"] = trees.DEM - trees.Contour

        if bundesland == "rlp":
            # import xml.etree.ElementTree as ET
            # feed_url = "https://www.geoportal.rlp.de/mapbender/php/mod_inspireDownloadFeed.php?id=3b3710e0-7a5f-4f2c-b911-e669c50728e6&type=DATASET&generateFrom=wfs&wfsid=616&featuretypeid=4242"
            # r = requests.get(feed_url)
            # xml = r.text
            # root = ET.fromstring(xml)
            # for elem in root.iter():
            #     if "href" in elem.attrib:
            #         print(elem.attrib["href"])

            base_url = "https://mapserver.lgb-rlp.de/cgi-bin/mc_gwo"

            params = {
                "SERVICE": "WFS",
                "REQUEST": "GetFeature",
                "VERSION": "2.0.0",
                "typeNames": "ms:GwGleichen",
                "srsName": "urn:ogc:def:crs:EPSG::25832",
                "outputFormat": "application/gml+xml; version=3.2"
            }

            r = requests.get(base_url, params=params)
            tile_path = os.path.join(output_dir, "gw_tile.gml")
            with open(tile_path, "wb") as f:
                f.write(r.content)

            gw = gpd.read_file(tile_path)
            gw = gw.to_crs(crs_dom)
            trees = gpd.sjoin_nearest(trees, gw[["ISOLINIE", "geometry"]], how="left", distance_col='distance_to_line')
            trees = trees.drop(columns=["index_right"], errors="ignore")
            trees["DTGW"] = trees.DEM - trees.ISOLINIE

        if bundesland == "bremen":
            url = (
                "https://gdfbmapserver.marum.de/cgi-bin/mapserv"
                "?Map=/root/Public/SUBVIntern/Internet_SUBV_wms_Mapbender_Baugrund.map"
                "&SERVICE=WFS"
                "&VERSION=1.1.0"
                "&REQUEST=GetFeature"
                "&TYPENAME=GrundwassergleichenHerbst2011"
                "&SRSNAME=EPSG:25832"
            )

            r = requests.get(url)
            tile_path = os.path.join(output_dir, "gw_tile.gml")
            with open(tile_path, "wb") as f:
                f.write(r.content)

            gw = gpd.read_file(tile_path)

            #gw = gpd.read_file(url)
            gw = gw.set_crs(crs_dom, allow_override=True)
            gw["GWmNN"] = (gw["GWmNN"].str.replace(",", ".", regex=False).astype(float))
            trees = gpd.sjoin_nearest(trees, gw[["GWmNN", "geometry"]], how="left", distance_col='distance_to_line')
            trees = trees.drop(columns=["index_right"], errors="ignore")
            trees["DTGW"] = trees.DEM - trees.GWmNN

# add DTGW to original tree shapefile
    trees_orig = gpd.read_file(tree_path)
    trees_orig["DTGW"] = trees["DTGW"]
    trees_orig.to_file(tree_path)

    # Plot DTGW
    # df = trees["DTGW"].dropna()
    df = pd.to_numeric(trees["DTGW"], errors="coerce").dropna()
    if len(df) > 0:
        plt.figure(figsize=(6,6))
        plt.boxplot(df)
        plt.ylabel("DTGW [m]")
        plt.xlabel(f"{tree_loc}")
        plt.xticks([])
        plt.text(1.45, max(df), f"n = {len(df)}", ha="right", va="top")
        plt.savefig(os.path.join(output_dir, "fig_DTGW_boxplot.pdf"), bbox_inches="tight")
        plt.show()

