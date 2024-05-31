import matplotlib.pyplot as plt
import pandas as pd


import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import datasets
from geopandas import GeoDataFrame

df = pd.read_csv('./lake_event_points.csv')

geometry = [Point(xy) for xy in zip(df['lon [deg]'], df['lat [deg]'])]
gdf = GeoDataFrame(df, geometry=geometry)   
# fig,ax = plt.subplots()

#this is a simple map that goes with geopandas
world = gpd.read_file(datasets.get_path('naturalearth_lowres'))
gdf.plot(ax=world.plot(figsize=(20, 10)), marker='o', color='cyan', markersize=8)
# plt.grid(True)
# plt.show()
print(len(df))
plt.savefig('./lake_event_points.png')