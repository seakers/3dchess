import pandas as pd
import os
import numpy as np

# -----------------------------------------------
# Paths
# -----------------------------------------------
BASE = os.path.join("E:\\", "Code", "3dchess", "scenarios", "wildfire_scenario", "resources")

J1 = os.path.join(BASE, "fire_nrt_J1V-C2_685922.csv")
J2 = os.path.join(BASE, "fire_nrt_J2V-C2_685923.csv")

OUTPUT = os.path.join(BASE, "wildfire_events.csv")
OUTPUT_NEARBY = os.path.join(BASE, "wildfire_events_nearby.csv")

# -----------------------------------------------
# Load CSVs
# -----------------------------------------------
print("Loading J1...")
df1 = pd.read_csv(J1)

print("Loading J2...")
df2 = pd.read_csv(J2)

# -----------------------------------------------
# Merge
# -----------------------------------------------
df = pd.concat([df1, df2], ignore_index=True)
print(f"Merged dataset shape: {df.shape}")

# Keep only rows with FRP (some Landsat/MODIS rows may not have it)
df = df.dropna(subset=["frp"])

# -----------------------------------------------
# Normalize severity
# severity = (FRP - min) / (max - min)
# -----------------------------------------------
frp_min = df["frp"].min()
frp_max = df["frp"].max()

df["severity"] = (df["frp"] - frp_min) / (frp_max - frp_min)

# -----------------------------------------------
# Apply threshold to choose only the strongest events
# e.g., severity >= 0.7
# -----------------------------------------------
threshold = 0.3
filtered = df[df["severity"] >= threshold].copy()

print(f"Selected {len(filtered)} high-severity events out of {len(df)}")

# -----------------------------------------------
# Build wildfire_events.csv in the needed format
# Format:
# lat [deg], lon [deg], start time [s], duration [s], severity, measurements
# -----------------------------------------------

# Convert time into datetime
def convert_to_dt(row):
    time = str(int(row["acq_time"])).zfill(4)
    hh = time[:2]
    mm = time[2:]
    return pd.to_datetime(f"{row['acq_date']} {hh}:{mm}", format="%Y-%m-%d %H:%M")

filtered["datetime"] = filtered.apply(convert_to_dt, axis=1)

# Use earliest timestamp as t0
t0 = filtered["datetime"].min()
filtered["start time [s]"] = (filtered["datetime"] - t0).dt.total_seconds()

# Constant duration (6,000s = 100 min)
filtered["duration [s]"] = 6000

# Fixed measurement types
filtered["measurements"] = "[visible,thermal,sar]"

# Final selection of columns
final = filtered.rename(columns={
    "latitude": "lat [deg]",
    "longitude": "lon [deg]"
})[[
    "lat [deg]",
    "lon [deg]",
    "start time [s]",
    "duration [s]",
    "severity",
    "measurements"
]]

# -----------------------------------------------
# Save
# -----------------------------------------------
final.to_csv(OUTPUT, index=False)
print(f"Saved wildfire events to:\n  {OUTPUT}")

# -----------------------------------------------
# Generate nearby points for each event
# Create points within a small radius around each event
# -----------------------------------------------
print("\nGenerating nearby points for each event...")

# Parameters for nearby points generation
radius_deg = 0.05  # ~5.5 km at equator
points_per_event = 8  # Number of nearby points to generate per event

nearby_points = []

for idx, row in final.iterrows():
    center_lat = row["lat [deg]"]
    center_lon = row["lon [deg]"]
    
    # Generate nearby points in a circular pattern
    for i in range(points_per_event):
        # Generate angle and distance
        angle = 2 * np.pi * i / points_per_event
        # Use a random distance within the radius for more natural distribution
        distance = radius_deg * np.random.uniform(0.3, 1.0)
        
        # Calculate offset (approximate, works well for small distances)
        # More accurate would use haversine, but for small distances this is fine
        lat_offset = distance * np.cos(angle)  # distance is already in degrees
        lon_offset = distance * np.sin(angle) / np.cos(np.radians(center_lat))  # adjust for latitude
        
        nearby_lat = center_lat + lat_offset
        nearby_lon = center_lon + lon_offset
        
        # Create a new row with the same event properties but new coordinates
        nearby_points.append({
            "lat [deg]": nearby_lat,
            "lon [deg]": nearby_lon,
            "start time [s]": row["start time [s]"],
            "duration [s]": row["duration [s]"],
            "severity": row["severity"],
            "measurements": row["measurements"]
        })

nearby_df = pd.DataFrame(nearby_points)
nearby_df.to_csv(OUTPUT_NEARBY, index=False)
print(f"Generated {len(nearby_df)} nearby points for {len(final)} events")
print(f"Saved nearby points to:\n  {OUTPUT_NEARBY}")
