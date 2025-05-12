import os
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm 
import pandas as pd 

import numpy as np

def generate_points_in_polygon(polygon, spacing=-1.0):    
    if spacing <= 0:
        return [polygon.centroid]
    
    # Get bounding box
    minx, miny, maxx, maxy = polygon.bounds

    # Generate grid points within bounds
    x_vals = np.arange(minx, maxx, spacing)
    y_vals = np.arange(miny, maxy, spacing)
    
    points = [Point(x, y) for x in x_vals for y in y_vals if polygon.contains(Point(x, y))]
    points.append(polygon.centroid)

    # Remove duplicate centroid (if it's also in the grid)
    unique_points = list({(p.x, p.y): p for p in points}.values())
    
    return unique_points

def create_grid(filename : str, spacing : float = 0.1, n_points : int = 10000):
    
    # Load lake polygon shapefile
    print(f"Loading `{filename}` shapefile...")
    filepath = os.path.join("./geodata", f"{filename}", f"{filename}.shp")
    gdf : gpd.GeoDataFrame = gpd.read_file(filepath, engine="pyogrio")

    # Reproject to an equal-area CRS
    # gdf_proj = gdf.to_crs("EPSG:6933")
    gdf_proj = gdf

    # Compute area in square meters
    gdf_proj["area_m2"] = gdf_proj.geometry.area

    # Sort by real-world area
    gdf_sorted = gdf_proj.sort_values(by="area_m2", ascending=False)
    gdf_sorted = gdf_sorted.head(n_points)
    
    # Store all points as GeoDataFrame rows
    all_points = []

    for _, row in tqdm(gdf_sorted.iterrows(), desc="Generating points in polygons", total=len(gdf)):
        polygon = row.geometry
        if polygon.is_empty or not polygon.is_valid:
            continue
        
        points = generate_points_in_polygon(polygon, spacing)
        
        all_points.extend([[point.x, point.y, polygon.area] for point in points])

        if len(all_points) >= n_points: break

    all_points.sort(key=lambda x: x[2],reverse=True)  # Sort by area

    # convert to DataFrame
    df = pd.DataFrame(all_points, columns=["lat [deg]","lon [deg]", "area [m^2]"])
    df.to_csv(f"./grids/{filename}.csv", index=False)


def main(grids_polygon_names : list, n_points : int = 10000):
    """ Creates coverage grids used in the last AIST report experiments. """
    

    for filename in grids_polygon_names:
        
        # Create directory if it doesn't exist
        if not os.path.exists("./grids"):
            os.makedirs("./grids")

        # Check if the grid already exists
        if not os.path.exists(f"./grids/{filename}.csv"):
            print(f"Creating grid for `{filename}`...")
            
            # Create grid with specified spacing
            create_grid(filename, -1, n_points)
    
    print('Done!')

if __name__ == "__main__":
    """ 

    COMMANDS FOR REDUCING SHAPEFILES' SIZE

    ogr2ogr \
        -simplify 1 \
        ./HydroLAKES_polys_v10_simple/HydroLAKES_polys_v10_simple.shp \
        ./HydroLAKES_polys_v10/HydroLAKES_polys_v10.shp 

    ogr2ogr \
        -simplify 1 \
        ./HydroRIVERS_v10_simple/HydroRIVERS_v10_simple.shp \
        ./HydroRIVERS_v10/HydroRIVERS_v10.shp 

    """

    # List of grids to create
    grids_polygon_names = [
        # "HydroLAKES_polys_v10",
        "HydroLAKES_polys_v10_simple",
        # "HydroRIVERS_v10",
        "HydroRIVERS_v10_simple",
        # "ne_110m_rivers_lake_centerlines",
    ]
    
    # Create grids
    main(grids_polygon_names)
