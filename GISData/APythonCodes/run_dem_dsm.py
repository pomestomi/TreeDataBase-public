import yaml
import subprocess
import os


# paths
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.join(project_dir, "GISData", "APythonCodes", "config.yaml")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cities = cfg["cities"]
bundeslaender = cfg["bundeslaender"]

# OPTIONS
QUIET = True          # True: suppress subscript output
ONLY_MISSING = False    # True: skip existing outputs, False: run all (if not filtered out)
# CITY_FILTER = None     # None = all cities
# examples:
CITY_FILTER = ["CityCelle"]
# CITY_FILTER = None     # None = all cities

# CITY_FILTER = ["CityBamberg", "UniversityErlangen", "BotanicalGardenBremen", "CompanyBremen", "CityHildesheim"]
MODEL_FILTER = None    # None = DEM + DSM
# examples:
# MODEL_FILTER = ["DEM"]
# MODEL_FILTER = ["DSM"]

# --------------------------------------------------
# Loop through cities and models (all active in config.yaml

for city, bundesland in cities.items():

    # city filter
    if CITY_FILTER is not None:
        if city not in CITY_FILTER:
            continue

    bl_cfg = bundeslaender[bundesland]

    for model in ["DEM", "DSM"]:
        model_cfg = bl_cfg.get(model, {})
        script_name = model_cfg.get("script") or bl_cfg.get("script")
        if script_name is None:
            raise ValueError(f"No script defined for {bundesland} / {model}")

        # model filter
        if MODEL_FILTER is not None:
            if model not in MODEL_FILTER:
                continue

        # skip if bw DSM not available
        if bundesland == "bw" and model == "DSM":

            tmp_dir = os.path.join(project_dir, "GISData", city, model, "tiles")
            os.makedirs(tmp_dir, exist_ok=True)
            zip_files = [
                f for f in os.listdir(tmp_dir)
                if f.lower().endswith(".zip")]

            if len(zip_files) == 0:
                print("-----------------------------------")
                print(f"Processing: {city} | {model}")
                print(f"Download {model} tiles manually from https://opengeodata.lgl-bw.de/#/(sidenav:product/dom1)")
                print(f"Place ZIP files into:\n{tmp_dir}")

                if city == "CityBiberach":
                    print("Required tiles: 555-5328, 557-5328")
                if city == "CityPforzheim":
                    print("Required tiles: 473-5416, 475-5416, 475-5414, 477-5414, 477-5416, 479-5416")
                if city == "CompanyHeidelberg":
                    print("Required tile: 473-5472")
                continue

        # skip existing outputs
        output_path = os.path.join(project_dir, "GISData", city, model, f"{model}_tree_000.tif")
        if ONLY_MISSING and os.path.exists(output_path):
            print(f"Skipping existing: {city} | {model}")
            continue

        print("-----------------------------------")
        print(f"Processing: {city} | {model}")

        # launch downloader script
        cmd = [
            "python",
            os.path.join(project_dir, "GISData", "APythonCodes", f"{script_name}.py"),
            "--city", city,
            "--model", model]

        # quiet mode
        if QUIET:
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        else:
            result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"Failed: {city} {model}")
        else:
            print(f"Done: {city} {model}")
