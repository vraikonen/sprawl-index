import os
from modules.utils import merge_and_reproject_rasters, process_zip_and_geopackage, assign_projection, load_config, merge_and_plot
from modules.index_computation import calculate_shape_index, calculate_dispersal_index, calculate_network_index

os.makedirs('results', exist_ok=True)

config = load_config() 

raw_population = config['data']['path_to_population']
raw_corine = config['data']['path_to_corine']
raw_urban_atlas = config['data']['path_to_urban_atlas']
city_centers = config['data']['city_centers']
countries = config['data']['countries']

# Preprocessing
path_to_proccessed_population = merge_and_reproject_rasters(raw_population)
path_to_fua_boundary, path_to_fua_names = process_zip_and_geopackage(raw_urban_atlas)
path_to_corine = assign_projection(raw_corine)

# Index calculation
path_to_shape_index_gpkg = calculate_shape_index(path_to_corine, path_to_fua_boundary, city_centers)
path_to_dispersal_index_gpkg = calculate_dispersal_index(path_to_proccessed_population, path_to_fua_boundary, city_centers)
path_to_network_index_gpkg = calculate_network_index(path_to_proccessed_population, path_to_fua_boundary, city_centers)

# Analysis
merge_and_plot(path_to_shape_index_gpkg, path_to_dispersal_index_gpkg, path_to_network_index_gpkg, countries)