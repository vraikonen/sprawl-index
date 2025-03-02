import yaml
import logging
import os

def load_config():
    config_file = 'config.yaml'
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Cannot get config file. Error: {e}")
        raise
    
    

import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS

os.makedirs('data/processed', exist_ok=True)
    
import os
import rasterio
from rasterio.merge import merge

import os
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.warp import reproject

def merge_and_reproject_rasters(folder_path, dst_crs='EPSG:3035'):
    """
    Merges all .tif raster files in the specified folder, saves the output, and reprojects it.
    
    Parameters:
    folder_path (str): Path to the folder containing raster (.tif) files.
    output_path (str): Path to save the merged raster.
    dst_crs (str): Target coordinate reference system (default is 'EPSG:3035').
    """
    output_path = "data/processed/test_wb_pop_merged3035.tif" # hardcoded 
    raster_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    
    if not raster_files:
        raise ValueError("No .tif files found in the specified folder.")
    
    # Open raster datasets
    raster_datasets = [rasterio.open(os.path.join(folder_path, file)) for file in raster_files]
    
    try:
        # Merge rasters
        merged, out_transform = merge(raster_datasets)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write merged raster to file
        with rasterio.open(output_path, 'w', driver='GTiff', 
                           width=merged.shape[2], height=merged.shape[1], count=1, 
                           dtype=merged.dtype, crs=raster_datasets[0].crs, transform=out_transform) as dst:
            dst.write(merged)
        
        print(f"Merged raster saved at {output_path}")

    finally:
        # Close input datasets to prevent memory leaks
        for dataset in raster_datasets:
            dataset.close()
    
    # Reproject to target CRS
    reprojected_path = output_path.replace('.tif', '_reprojected.tif')
    with rasterio.open(output_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(reprojected_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    
    # Delete the initial raster
    os.remove(output_path)
    print(f"Reprojected raster saved at {reprojected_path}")
    return reprojected_path
# Example usage
# merge_and_reproject_rasters("data/raw/wb_pop", "../data/processed/test_wb_pop_merged3035.tif")



import zipfile
import os
import geopandas as gpd
import pandas as pd
import fiona
import shutil

import zipfile
import os
import geopandas as gpd
import pandas as pd
import fiona
import shutil

def process_zip_and_geopackage(zip_file_path):
    fua_path = 'data/processed/fua_boundary.gpkg'
    name_path = 'data/processed/fua_names.csv'
    
    # Create a temporary extraction directory
    extract_to = 'temp_fuas'
    os.makedirs(extract_to, exist_ok=True)
    
    # Step 1: Extract the initial ZIP file and its contents (recursively)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        subfolders = [name for name in zip_ref.namelist() if name.endswith('/')]
        for subfolder in subfolders:
            subfolder_path = os.path.join(extract_to, subfolder)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref_nested:
                zip_ref_nested.extractall(subfolder_path)

    # Step 2: Extract all ZIP files in the extracted directory
    for root, _, files in os.walk(extract_to):
        for file in files:
            file_path = os.path.join(root, file)
            if zipfile.is_zipfile(file_path):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(root)
                os.remove(file_path)  # Delete the ZIP file after extraction

    # Step 3: Extract "Boundary" layers from all GeoPackage files and save to a new GeoPackage
    boundary_layers = []
    for root, _, files in os.walk(extract_to):
        for file in files:
            if file.endswith('.gpkg'):
                file_path = os.path.join(root, file)
                layers = fiona.listlayers(file_path)
                for layer_name in layers:
                    if "Boundary" in layer_name:
                        gpkg = gpd.read_file(file_path, layer=layer_name)
                        boundary_layers.append(gpkg)
    
    # Concatenate all boundary layers into a single GeoDataFrame
    if boundary_layers:
        merged_gdf = gpd.GeoDataFrame(pd.concat(boundary_layers, ignore_index=True))
        merged_gdf.to_file(fua_path, driver='GPKG')
    
    # Step 4: Get Fua names
    fua_names = []
    for index, row in merged_gdf.iterrows():
        fua_name = row['fua_name']
        fua_names.append(fua_name)
    df = pd.DataFrame({'fua_name': fua_names})
    df.to_csv(name_path, index=False)
    # Step 5: Cleanup - Remove the temporary extracted directory
    shutil.rmtree(extract_to)
    
    return fua_path, name_path

# Example usage:
zip_file_path = 'data/raw/urban_atlas/50f2e53c420a78af65c92f337e1402452c6500ae.zip'
fua_path = 'data/processed/all_fua_boundary_test.gpkg'
name_path = 'data/processed/fua_names.csv'

# Run the process
# process_zip_and_geopackage(zip_file_path, fua_path, name_path)



import rasterio
from rasterio.crs import CRS
import numpy as np

def assign_projection(input_raster_path, proj4_str= '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs'):
    """
    Reprojects a raster to a new CRS using the provided PROJ4 string and writes it in chunks to avoid memory issues.

    Parameters:
        input_raster_path (str): Path to the input raster file.
        output_raster_path (str): Path to save the reprojected raster file.
        proj4_str (str): PROJ4 string for the desired CRS.

    Returns:
        None
    """
    output_raster_path = 'data/processed/corine3035.tif'
    with rasterio.open(input_raster_path) as src:
        # Copy the metadata from the source raster
        new_profile = src.profile.copy()
        
        # Update the CRS of the raster to the new CRS defined by the PROJ4 string
        new_profile['crs'] = CRS.from_proj4(proj4_str)

        # Open the output raster with the new profile
        with rasterio.open(output_raster_path, 'w', **new_profile) as dest:
            # Get the block size to read and write in chunks
            block_size = 1024  # Adjust this size if necessary for your system
            for i in range(0, src.height, block_size):
                # Read a chunk of the raster
                window = rasterio.windows.Window(0, i, src.width, min(block_size, src.height - i))
                data = src.read(1, window=window)  # Read one band at a time (change if there are multiple bands)
                
                # Write the chunk to the destination raster
                dest.write(data, 1, window=window)

    print(f"Raster reprojected and saved to {output_raster_path}")
    return output_raster_path
# Example usage
input_raster_path = 'data/raw/corine/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif'
output_raster_path = 'data/processed/corine3035test.tif'
proj4_3035 = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs'


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def merge_and_plot(
    shape_path,
    dispersal_path,
    network_path,
    countries_path,
    output_gpkg="results/sprawl.gpkg",
    plots_dir="plots"
):
    """Merges multiple GeoPackages, saves the final result, removes intermediate files, and saves plots."""
    
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)

    # Load GeoDataFrames
    shape_gdf = gpd.read_file(shape_path)[["fua_name", "shape_index", "geometry"]]
    dispersal_gdf = gpd.read_file(dispersal_path)[["fua_name", "dispersal_index"]]
    network_gdf = gpd.read_file(network_path)[["fua_name", "network_index"]]

    # Merge them
    merged_gdf = shape_gdf.merge(dispersal_gdf, on="fua_name").merge(network_gdf, on="fua_name")

    # Save final merged GPKG
    merged_gdf.to_file(output_gpkg, driver="GPKG")
    merged_gdf_csv = merged_gdf.copy()
    merged_gdf_csv["geometry"] = merged_gdf_csv["geometry"].apply(lambda geom: geom.wkt if geom else None)
    merged_gdf_csv.to_csv("results/sprawl.csv", index=False)
    
    # Remove intermediate GPKG files
    for file in [shape_path, dispersal_path, network_path]:
        os.remove(file)

    # Load countries shapefile for background
    countries_gdf = gpd.read_file(countries_path)

    # Plot maps for each index
    for column, title in zip(["shape_index", "dispersal_index", "network_index"],
                              ["Shape Index", "Dispersal Index", "Network Index"]):
        fig, ax = plt.subplots(figsize=(10, 8))
        countries_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5)
        merged_gdf.plot(column=column, cmap="magma_r", legend=True, ax=ax)
        plt.title(title)
        plt.savefig(f"{plots_dir}/{column}_map.png")
        plt.close()

    # Plot distributions
    for column in ["shape_index", "dispersal_index", "network_index"]:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=merged_gdf, x=column, kde=True, bins=10, color="skyblue")
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.savefig(f"{plots_dir}/{column}_distribution.png")
        plt.close()

    print(f"Final GPKG saved to: {output_gpkg}")
    print(f"Plots saved to: {plots_dir}/")

# Run the function
# merge_and_plot()
