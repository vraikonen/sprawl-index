
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
from shapely.geometry import mapping, Point
import os

def calculate_shape_index(raster_path, gpkg_file_path, city_centers_csv):
    """
    Function to calculate the shape index for FUAs based on a given raster, FUA boundaries, and city centers.
    
    Parameters:
        raster_path (str): Path to the raster file.
        gpkg_file_path (str): Path to the GeoPackage file containing FUA boundaries.
        city_centers_csv (str): Path to the CSV file containing city center coordinates.
        output_gpkg_path (str): Path to output GeoPackage for saving the shape index results.
        temp_dir (str): Directory for storing intermediate temporary files.
    """
    output_gpkg_path = 'results/shape_index.gpkg'
    temp_dir='../data/temp_var/'
    # Load the raster, FUA boundaries, and city centers
    raster = rasterio.open(raster_path)
    fuas = gpd.read_file(gpkg_file_path, layer="fua_boundary")
    
    city_centers_df = pd.read_csv(city_centers_csv)
    geometry = [Point(xy) for xy in zip(city_centers_df['long'], city_centers_df['lat'])]
    city_centers_gdf = gpd.GeoDataFrame(city_centers_df, geometry=geometry, crs='EPSG:4326')
    city_centers_gdf = city_centers_gdf.to_crs('EPSG:3035')
    
    # Create temporary directory
    os.makedirs(temp_dir, exist_ok=True)

    # Iterate over each FUA boundary and calculate the shape index
    for idx, fua_row in fuas.iterrows():
        fua_name = fua_row['fua_name']
        
        # Find the corresponding city center
        city_center_row = city_centers_gdf[city_centers_gdf['fua_name'] == fua_name]
        city_center_point = city_center_row.geometry.iloc[0]
        city_center_lon = city_center_point.x
        city_center_lat = city_center_point.y
        
        # Extract FUA boundary geometry
        geometry = fua_row['geometry']
        clipping_geometry = [geometry.__geo_interface__]
        
        # Clip the raster using the FUA boundary geometry
        clipped_raster, clipped_transform = rasterio.mask.mask(raster, clipping_geometry, crop=True)
        
        # Update metadata and save the clipped raster
        clipped_meta = raster.meta.copy()
        clipped_meta.update({
            'height': clipped_raster.shape[1],
            'width': clipped_raster.shape[2],
            'transform': clipped_transform
        })
        
        output_path = os.path.join(temp_dir, f"clipped_{fua_name}.tif")
        with rasterio.open(output_path, 'w', **clipped_meta) as dst:
            dst.write(clipped_raster)
        
        # Reclassify the raster
        with rasterio.open(output_path) as src:
            data = src.read(1)
            reclass_dict = {1: 1, 2: 1, 3: 1}
            reclassified_data = np.where(np.isin(data, list(reclass_dict.keys())), 1, 0)
        
            reclass_output_path = os.path.join(temp_dir, f"reclass_{fua_name}.tif")
            with rasterio.open(reclass_output_path, 'w', **src.meta) as dst:
                dst.write(reclassified_data, 1)
        
        # Calculate distances and shape index
        with rasterio.open(reclass_output_path) as src:
            data = src.read(1)
            transform = src.transform
            
            point_x, point_y = city_center_lon, city_center_lat
            row, col = ~transform * (point_x, point_y)
            
            # Calculate the distance to each pixel
            distances = np.zeros_like(data, dtype=np.float32)
            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    if data[r, c] == 1:
                        pixel_x, pixel_y = transform * (c, r)
                        distance = ((pixel_x - point_x)**2 + (pixel_y - point_y)**2)**0.5
                        distances[r, c] = distance
            
            # Calculate pixel area and weighted sum of distances
            pixel_width, pixel_height = src.res
            pixel_area = pixel_width * pixel_height
            weighted_sum = np.sum(distances * pixel_area)
            
            # Calculate built-up area
            built_up_area = (data[data == 1]).sum() * pixel_area
            
            # Calculate the shape index
            shape_index = (weighted_sum / built_up_area) / (0.377 * np.sqrt(built_up_area))
            fuas.loc[idx, 'shape_index'] = shape_index
        
        # Remove intermediate files
        os.remove(output_path)
        os.remove(reclass_output_path)
    
    # Remove temporary directory
    os.rmdir(temp_dir)

    # Save the result to a GeoPackage
    fuas.to_file(output_gpkg_path, driver='GPKG')
    
    return output_gpkg_path





import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
from shapely.geometry import mapping, Point
import os

def calculate_dispersal_index(raster_path, gpkg_file_path, city_centers_csv):
    """
    Function to calculate the dispersal index and shape index based on a given population raster,
    FUA boundaries, and city centers.
    
    Parameters:
        raster_path (str): Path to the population raster file.
        gpkg_file_path (str): Path to the GeoPackage file containing FUA boundaries.
        city_centers_csv (str): Path to the CSV file containing city center coordinates.
        output_gpkg_path (str): Path to output GeoPackage for saving the dispersal index results.
        temp_dir (str): Directory for storing intermediate temporary files.
    """
    output_gpkg_path = 'results/dispersal_index.gpkg'
    temp_dir='data/temp_var/'
    # Load the raster, FUA boundaries, and city centers
    raster = rasterio.open(raster_path)
    fuas = gpd.read_file(gpkg_file_path, layer="fua_boundary")
    
    city_centers_df = pd.read_csv(city_centers_csv)
    geometry = [Point(xy) for xy in zip(city_centers_df['long'], city_centers_df['lat'])]
    city_centers_gdf = gpd.GeoDataFrame(city_centers_df, geometry=geometry, crs='EPSG:4326')
    city_centers_gdf = city_centers_gdf.to_crs('EPSG:3035')
    
    # Create temporary directory
    os.makedirs(temp_dir, exist_ok=True)

    # Iterate over each FUA boundary and calculate the dispersal and shape index
    for idx, fua_row in fuas.iterrows():
        fua_name = fua_row['fua_name']
        
        # Find the corresponding city center
        city_center_row = city_centers_gdf[city_centers_gdf['fua_name'] == fua_name]
        city_center_point = city_center_row.geometry.iloc[0]
        city_center_lon = city_center_point.x
        city_center_lat = city_center_point.y
        
        # Extract FUA boundary geometry
        geometry = fua_row['geometry']
        clipping_geometry = [geometry.__geo_interface__]
        
        # Clip the raster using the FUA boundary geometry
        clipped_raster, clipped_transform = rasterio.mask.mask(raster, clipping_geometry, crop=True)
        
        # Update metadata and save the clipped raster
        clipped_meta = raster.meta.copy()
        clipped_meta.update({
            'height': clipped_raster.shape[1],
            'width': clipped_raster.shape[2],
            'transform': clipped_transform
        })
        
        output_path = os.path.join(temp_dir, f"clipped_{fua_name}.tif")
        with rasterio.open(output_path, 'w', **clipped_meta) as dst:
            dst.write(clipped_raster)
        
        # Process the clipped raster to calculate dispersal index
        with rasterio.open(output_path) as src:
            data = src.read(1)
            transform = src.transform
            
            # Get the pixel coordinates of the city center
            point_x, point_y = city_center_lon, city_center_lat
            row, col = ~transform * (point_x, point_y)
            
            # Calculate distances for pixels with population greater than 10
            distances = np.zeros_like(data, dtype=np.float32)
            distances_array = []
            pixel_values_array = []
            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    if data[r, c] > 10:
                        pixel_x, pixel_y = transform * (c, r)
                        distance = ((pixel_x - point_x)**2 + (pixel_y - point_y)**2)**0.5
                        distances[r, c] = distance
                        
                        # Append the distance and pixel value
                        distances_array.append(distance)
                        pixel_values_array.append(data[r, c])

            # Convert the lists to NumPy arrays for further calculations
            distances_array = np.array(distances_array)
            pixel_values_array = np.array(pixel_values_array)
            
            # Calculate the weighted sum of distances and total population
            weighted_pop_sum = np.sum(pixel_values_array * distances_array)
            total_population = (data[data > 10]).sum()
            
            # Calculate pixel area and weighted sum of distances
            pixel_width, pixel_height = src.res
            pixel_area = pixel_width * pixel_height
            num_entities = np.count_nonzero(data > 10)
            weighted_sum = np.sum(distances_array * pixel_area)
            
            # Calculate built-up area (total area of pixels with population > 10)
            built_up_area = num_entities * pixel_area
            
            # Calculate the dispersal index
            dispersal_index = (weighted_pop_sum / total_population) / (weighted_sum / built_up_area)
            
            # Calculate the shape index based on the population dataset
            shape_index = (weighted_sum / built_up_area) / (0.377 * np.sqrt(built_up_area))
            
            # Set the result values for the current FUA
            fuas.loc[idx, 'dispersal_index'] = dispersal_index
            fuas.loc[idx, 'shape_index_pop_data'] = shape_index
        
        # Remove intermediate file
        os.remove(output_path)
    
    # Remove temporary directory
    os.rmdir(temp_dir)

    # Save the result to a GeoPackage
    fuas.to_file(output_gpkg_path, driver='GPKG')
    return output_gpkg_path
    
    
    
    
import os
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point, Polygon
from rasterio.mask import mask
from shapely.ops import transform
import openrouteservice
from rasterstats import zonal_stats
# check this out https://github.com/GIScience/openrouteservice
def calculate_network_index(raster_path, gpkg_file_path, city_centers_csv):
    """
    Function to calculate the Network Indirectness Index for each FUA boundary using population raster data.
    
    Parameters:
        raster_path (str): Path to the population raster file.
        gpkg_file_path (str): Path to the GeoPackage file containing FUA boundaries.
        city_centers_csv (str): Path to the CSV file containing city center coordinates.
        temp_dir (str): Directory for storing intermediate files (default is '../data/temp_var/').
        output_gpkg_path (str): Path to output GeoPackage where the results will be saved.
    """
    output_gpkg_path='results/network_index.gpkg'
    temp_dir='data/temp_var'
    # Load the FUA boundaries and city center data
    fuas = gpd.read_file(gpkg_file_path, layer="fua_boundary")

    city_centers_df = pd.read_csv(city_centers_csv)
    geometry = [Point(xy) for xy in zip(city_centers_df['long'], city_centers_df['lat'])]
    city_centers_gdf = gpd.GeoDataFrame(city_centers_df, geometry=geometry, crs='EPSG:4326')
    city_centers_gdf = city_centers_gdf.to_crs('EPSG:3035')
    city_centers_gdf = city_centers_gdf.rename(columns={'geometry': 'center3035'})
    
    # Merge the city center data with FUA boundaries
    fuas = gpd.GeoDataFrame(pd.merge(fuas, city_centers_gdf, on='fua_name'))

    # Create temporary directory
    os.makedirs(temp_dir, exist_ok=True)

    # Iterate over each FUA boundary
    for idx, fua_row in fuas.iterrows():
        # Get the bounding box of the FUA
        bounding_box = fua_row['geometry'].bounds
        cell_size = 1000  # Grid cell size (1km x 1km)
        
        # Generate grid cells based on bounding box
        grid_cells = []
        for x in range(int(bounding_box[0]), int(bounding_box[2]), cell_size):
            for y in range(int(bounding_box[1]), int(bounding_box[3]), cell_size):
                polygon = Polygon([
                    (x, y),
                    (x + cell_size, y),
                    (x + cell_size, y + cell_size),
                    (x, y + cell_size),
                    (x, y)
                ])
                grid_cells.append(polygon)

        # Create grid GeoDataFrame with EPSG:3035
        grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs='EPSG:3035')

        # Clip grid based on the FUA boundary
        polygon_geometry = fua_row['geometry']
        polygon_gdf = gpd.GeoDataFrame(geometry=[polygon_geometry], crs=fuas.crs)
        intersection_gdf = gpd.overlay(grid_gdf, polygon_gdf, how='intersection')

        # Create centroids for the grid cells
        intersection_gdf['centroid'] = intersection_gdf['geometry'].centroid

        # Clip the population raster based on the FUA boundary
        with rasterio.open(raster_path) as src:
            clipping_geometry = [fua_row['geometry'].__geo_interface__]
            clipped_raster, clipped_transform = mask(src, clipping_geometry, crop=True)
            clipped_raster[(clipped_raster >= -99999) & (clipped_raster < 2)] = 0  # Set specific values to 0
            
            clipped_meta = src.meta.copy()
            clipped_meta.update({
                'height': clipped_raster.shape[1],
                'width': clipped_raster.shape[2],
                'transform': clipped_transform
            })

            output_path = os.path.join(temp_dir, 'clip.tif')
            with rasterio.open(output_path, 'w', **clipped_meta) as dst:
                dst.write(clipped_raster)

        # Define a function to check for zeros in raster data
        def check_zeros(x):
            return (x == 0).astype(int).sum()

        # Calculate zonal statistics using the intersection grid
        with rasterio.open(output_path) as src:
            affine_transform = src.transform
            stats = zonal_stats(
                vectors=intersection_gdf['geometry'],
                raster=src.read(1),
                affine=affine_transform,
                stats=['count', 'sum'],
                add_stats={'zeros_count': check_zeros},
                categorical=True
            )

        # Extract pixel counts and sums
        pixel_count_per_polygon = [stat['count'] for stat in stats]
        pixel_sum_per_polygon = [stat['sum'] if 'sum' in stat else 0 for stat in stats]
        pixel_count_zeros_per_polygon = [stat['zeros_count'] for stat in stats]

        # Add statistics to the GeoDataFrame
        intersection_gdf['pixel_count'] = pixel_count_per_polygon
        intersection_gdf['pixel_sum'] = pixel_sum_per_polygon
        intersection_gdf['pixel_count_zeros'] = pixel_count_zeros_per_polygon
        intersection_gdf['valid_pixels_count'] = intersection_gdf['pixel_count'] - intersection_gdf['pixel_count_zeros']
        
        # Drop invalid polygons
        intersection_gdf = intersection_gdf.drop(
            intersection_gdf[(intersection_gdf['valid_pixels_count'] == 0) | (intersection_gdf['pixel_count'] == 0)].index)

        # Convert centroids to EPSG 4326
        intersection_gdf['centroid4326'] = intersection_gdf['centroid'].to_crs(epsg=4326)

        # Get the city center coordinates
        city_center_point = fua_row.center3035
        city_center_lon = city_center_point.x
        city_center_lat = city_center_point.y
        city_center_coords = (fua_row.long, fua_row.lat)

        # Initialize OpenRouteService client
        client = openrouteservice.Client(base_url='http://localhost:8080/ors')

        # Initialize variables for weighted distances
        weighted_road_distance_sum = 0
        weighted_aerial_distance_sum = 0

        # Iterate over each row in the intersection GeoDataFrame
        for index, row in intersection_gdf.iterrows():
            centroid_coords = (row['centroid4326'].x, row['centroid4326'].y)
            try:
                # Calculate road distance
                route = client.directions(
                    coordinates=[centroid_coords, city_center_coords],
                    profile='driving-car',
                    format='geojson',
                    radiuses=[1000, 500]
                )
                road_distance = route['features'][0]['properties']['segments'][0]['distance']
            except Exception as e:
                print(f"Warning: Unable to find a route for cell center {centroid_coords} {city_center_coords}. Skipping.")
                continue
            
            # Calculate aerial (beeline) distance
            aerial_distance = row['centroid4326'].distance(Point(city_center_coords))
            point_3035x = row['centroid'].x
            point_3035y = row['centroid'].y
            aerial_dist = ((point_3035x - city_center_lon)**2 + (point_3035y - city_center_lat)**2)**0.5

            # Update weighted distance sums
            weighted_road_distance_sum += road_distance * row['pixel_sum']
            weighted_aerial_distance_sum += aerial_dist * row['pixel_sum']

        # Calculate the Network Indirectness Index
        network_indirectness_index = weighted_road_distance_sum / weighted_aerial_distance_sum

        # Store the result in the FUA DataFrame
        fuas.loc[idx, 'network_index'] = network_indirectness_index

        # Remove the intermediate raster
        os.remove(output_path)

        print(f"Network Indirectness Index for {fua_row.fua_name}: {network_indirectness_index}")
    
    # Remove temporary directory
    os.rmdir(temp_dir)

    # Save the results to a GeoPackage
    fuas['center3035'] = fuas['center3035'].to_wkt()
    fuas.to_file(output_gpkg_path, driver='GPKG')

    return output_gpkg_path