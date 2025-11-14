import geopandas as gpd

# Correct path to the Spain shapefile
fires = gpd.read_file(
    "/mnt/e/Code/3dchess/scenarios/wildfire_scenario/data/Spain/DL_FIRE_J1V-C2_681373/fire_nrt_J1V-C2_681373.shp"
)

# Print some info
print(fires.head())
print(fires.columns)

# Optional: export to CSV
output_path = "/mnt/e/Code/3dchess/scenarios/wildfire_scenario/data/Spain/firms_fires_spain.csv"
fires.to_csv(output_path, index=False)
print(f"✅ Saved to {output_path}")
