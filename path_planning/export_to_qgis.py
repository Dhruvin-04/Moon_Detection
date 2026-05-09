import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import math

R = 1737400

# Correct WKT for SimpleCylindrical Moon
from pyproj import CRS
MOON_CRS = CRS.from_proj4(
    "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 "
    "+x_0=0 +y_0=0 +a=1737400 +b=1737400 "
    "+units=m +no_defs"
)

def latlon_to_meters(lat, lon):
    x_m = lon * (math.pi/180) * R
    y_m = lat * (math.pi/180) * R
    return x_m, y_m

# ── 1. Science Stops → Point layer ──────────────────────────
stops_df = pd.read_csv("science_stops_final.csv")

stop_geoms = []
for _, row in stops_df.iterrows():
    x_m, y_m = latlon_to_meters(row['latitude'], row['longitude'])
    stop_geoms.append(Point(x_m, y_m))

stops_gdf = gpd.GeoDataFrame(stops_df, geometry=stop_geoms)
stops_gdf = stops_gdf.set_crs(MOON_CRS, allow_override=True)
stops_gdf.to_file("science_stops.gpkg", driver="GPKG")
print("✅ science_stops.gpkg saved")

# ── 2. Rover Path → Line layer ───────────────────────────────
path_df = pd.read_csv("rover_path_final.csv")

path_points = []
for _, row in path_df.iterrows():
    x_m, y_m = latlon_to_meters(row['latitude'], row['longitude'])
    path_points.append((x_m, y_m))

path_line = LineString(path_points)
path_gdf  = gpd.GeoDataFrame(
    {"id": [1], "total_km": [8.29], "waypoints": [len(path_df)]},
    geometry=[path_line]
)
path_gdf = path_gdf.set_crs(MOON_CRS, allow_override=True)
path_gdf.to_file("rover_path.gpkg", driver="GPKG")
print("✅ rover_path.gpkg saved")

# ── 3. Crater Detections → Point layer ──────────────────────
crater_df = pd.read_csv("craters_latlong.csv")

crater_geoms = []
for _, row in crater_df.iterrows():
    x_m, y_m = latlon_to_meters(row['latitude'], row['longitude'])
    crater_geoms.append(Point(x_m, y_m))

crater_gdf = gpd.GeoDataFrame(crater_df, geometry=crater_geoms)
crater_gdf = crater_gdf.set_crs(MOON_CRS, allow_override=True)
crater_gdf.to_file("crater_detections.gpkg", driver="GPKG")
print("✅ crater_detections.gpkg saved")

# ── 4. Landing Point → Point layer ──────────────────────────
land_x, land_y = latlon_to_meters(-85.28, 31.20)
land_gdf = gpd.GeoDataFrame(
    {"id": [1], "name": ["Landing Point"],
     "latitude": [-85.28], "longitude": [31.20]},
    geometry=[Point(land_x, land_y)]
)
land_gdf = land_gdf.set_crs(MOON_CRS, allow_override=True)
land_gdf.to_file("landing_point.gpkg", driver="GPKG")
print("✅ landing_point.gpkg saved")

print("\nAll files saved. Load in QGIS:")
print("  Layer → Add Layer → Add Vector Layer")
print("  Select each .gpkg file")