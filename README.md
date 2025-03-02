
# Sprawl Index

This repo contains the code for calculating sprawl indices for [this paper](link_to_paper). To use it, download the required data, set up OpenRouteService locally, and create a Python environment.

## Quick Start

### 0. Clone the repo
```bash
git clone https://github.com/vraikonen/sprawl-index
cd sprawl-index
```
### 1. Get the Data
- **UrbanAtlas**: Download West Balkan FUAs from [here](https://land.copernicus.eu/en/products/urban-atlas/urban-atlas-change-2012-2018).
- **WorldPop 2020 Constrained**: Get data for each West Balkan country from [here](https://hub.worldpop.org/geodata/listing?id=29).
- **CORINE Land Cover 2018 (100m raster)**: Download [here](https://land.copernicus.eu/en/products/corine-land-cover/clc2018).

### 2. Organize the Data
Place downloaded files in:
- `data/raw/urban_atlas/` (zipped Urban Atlas file)
- `data/raw/population/` (WorldPop raster files)
- `data/raw/corine/` (CORINE dataset + metadata)

__NOTE__: Adjust paths in `config.yaml` if necessary, but definitely check the file to see if it match your current paths.

### 3. Set Up OpenRouteService (ORS)
- Download West Balkan road network from [OSM](https://download.geofabrik.de/).
- Merge country `.osm.pbf` files using `osmium`:
  ```bash
  sudo apt-get install osmium-tool
  osmium cat serbia-latest.osm.pbf \
             croatia-latest.osm.pbf \
             montenegro-latest.osm.pbf \
             kosovo-latest.osm.pbf \
             macedonia-latest.osm.pbf \
             bosnia-herzegovina-latest.osm.pbf \
             albania-latest.osm.pbf \
             -o wb.osm.pbf
  ```
- Follow [this guide](https://github.com/GIScience/openrouteservice) to set up the ORS Docker container and.
- Before compose-up: update `docker-compose.yaml` - change the OSM data path to `wb.osm.pbf` and mount the files directory.
- Ensure ORS is running on port 8080.

### 4. Set Up Python Environment

- Install [pyenv](https://github.com/pyenv/pyenv).
- Install [poetry](https://python-poetry.org/docs/).
```bash
pyenv install 3.12.8  
pyenv local 3.12.8  
poetry install --no-root
poetry shell
python main.py
```

### 5. Explore Results
Check the `results/` directory for `.gpkg` files (indices) and `plots` directory for plots.

__NOTE__: To use this code for other regions:
- Update `data/processed/city_centers.csv` with city centers for your FUAs.
- I suggest to use `process_zip_and_geopackage()` in `modules/utils.py` to generate a list of FUAs and manually add city center coordinates.
