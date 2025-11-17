import pandas as pd
import numpy as np
import os
import uuid

# ---------------------------------------------------------
# Paths - use relative paths based on script location
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, "resources")
J1 = os.path.join(BASE, "fire_nrt_J1V-C2_685922.csv") 
J2 = os.path.join(BASE, "fire_nrt_J2V-C2_685923.csv") 
OUTPUT = os.path.join(BASE, "wildfire_events.csv") 
OUTPUT_NEARBY = os.path.join(BASE, "wildfire_events_nearby.csv")
POINTS_OUT = os.path.join(SCRIPT_DIR, "data", "World", "wildfire_points_world.csv")
EVENTS_OUT = os.path.join(SCRIPT_DIR, "data", "World", "wildfire_events_world.csv")
# ---------------------------------------------------------
# Load CSV files
# ---------------------------------------------------------
df1 = pd.read_csv(J1)
# Check if J2 exists, if not just use df1
if os.path.exists(J2):
    df2 = pd.read_csv(J2)
    df = pd.concat([df1, df2], ignore_index=True)
else:
    print(f"Warning: {J2} not found, using only {J1}")
    df = df1.copy()

df = df.dropna(subset=["frp"])

# ---------------------------------------------------------
# Compute severity (normalized FRP)
# ---------------------------------------------------------
frp_min = df["frp"].min()
frp_max = df["frp"].max()
df["severity"] = (df["frp"] - frp_min) / (frp_max - frp_min)

# Choose strong events (threshold)
threshold = 0.3
strong = df[df["severity"] >= threshold].copy()

# ---------------------------------------------------------
# Convert acq_time + acq_date to datetime
# ---------------------------------------------------------
def to_dt(row):
    time = str(int(row["acq_time"])).zfill(4)
    hh = time[:2]
    mm = time[2:]
    # Try ISO format first (YYYY-MM-DD), fallback to MM/DD/YYYY
    date_str = f"{row['acq_date']} {hh}:{mm}"
    try:
        return pd.to_datetime(date_str, format="%Y-%m-%d %H:%M")
    except ValueError:
        # Fallback to MM/DD/YYYY format
        return pd.to_datetime(date_str, format="%m/%d/%Y %H:%M")

strong["datetime"] = strong.apply(to_dt, axis=1)

# Time origin
t0 = strong["datetime"].min()
strong["start time [s]"] = (strong["datetime"] - t0).dt.total_seconds()
strong["duration [s]"] = 3600

# ---------------------------------------------------------
# Compute decorrelation time
# ---------------------------------------------------------
a = 7200
eps = 0.05
strong["frp_norm"] = (strong["frp"] - frp_min) / (frp_max - frp_min)
strong["decorrelation time [s]"] = a / (strong["frp_norm"] + eps)

# ---------------------------------------------------------
# Build wildfire_points.csv
# We generate nearby points around strong events
# ---------------------------------------------------------
radius_deg = 0.05
points_per_event = 8

points = []

for _, row in strong.iterrows():
    lat0 = row["latitude"]
    lon0 = row["longitude"]

    for i in range(points_per_event):
        ang = 2 * np.pi * i / points_per_event
        dist = radius_deg * np.random.uniform(0.3, 1.0)

        lat = lat0 + dist * np.cos(ang)
        lon = lon0 + dist * np.sin(ang) / np.cos(np.radians(lat0))

        points.append((lat, lon))

# Convert to DataFrame + gp_index
points_df = pd.DataFrame(points, columns=["lat [deg]", "lon [deg]"])
points_df["gp_index"] = np.arange(len(points_df))

# Save points grid - ensure directory exists
os.makedirs(os.path.dirname(POINTS_OUT), exist_ok=True)
points_df.to_csv(POINTS_OUT, index=False)
print("Saved wildfire_points.csv with", len(points_df), "points")

# ---------------------------------------------------------
# Assign each event to closest grid point
# ---------------------------------------------------------
def closest_point(lat, lon, pdf):
    dlat = pdf["lat [deg]"] - lat
    dlon = (pdf["lon [deg]"] - lon) * np.cos(np.radians(lat))
    d = dlat**2 + dlon**2
    return pdf.iloc[d.idxmin()]["gp_index"]

strong["gp_index"] = strong.apply(
    lambda r: closest_point(r["latitude"], r["longitude"], points_df), axis=1
)

# ---------------------------------------------------------
# Prepare wildfire_events.csv
# ---------------------------------------------------------
strong["event type"] = "wildfire"
strong["id"] = [str(uuid.uuid4()) for _ in range(len(strong))]

events_df = strong.rename(columns={
    "latitude": "lat [deg]",
    "longitude": "lon [deg]"
})[[
    "gp_index",
    "lat [deg]",
    "lon [deg]",
    "start time [s]",
    "duration [s]",
    "severity",
    "event type",
    "decorrelation time [s]",
    "id"
]]

# Save events - ensure directory exists
os.makedirs(os.path.dirname(EVENTS_OUT), exist_ok=True)
events_df.to_csv(EVENTS_OUT, index=False)
print("Saved wildfire_events.csv with", len(events_df), "events")
