import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd

# ── 1. Load slope and roughness ──────────────────────────────
with rasterio.open("slope_map.tif") as f:
    slope_raw = f.read(1).astype(float)
    meta = f.meta.copy()
    transform = f.transform
    nodata = f.nodata

with rasterio.open("roughness_map.tif") as f:
    roughness_raw = f.read(1).astype(float)

# ── 2. Mask nodata ───────────────────────────────────────────
if nodata is not None:
    slope_raw[slope_raw == nodata] = np.nan
    roughness_raw[roughness_raw == nodata] = np.nan

# ── 3. Normalize to 0-1 ──────────────────────────────────────
def normalize(arr):
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    return (arr - mn) / (mx - mn)

slope_n     = normalize(slope_raw)
roughness_n = normalize(roughness_raw)

# ── 4. Load crater CSV and rasterize ────────────────────────
df = pd.read_csv("craters_latlong.csv")  # columns: latitude, longitude

# LOLA is in meters (SimpleCylindrical Moon)
# Convert lat/lon to meters
R = 1737400
import math
df['x_m'] = df['longitude'] * (math.pi/180) * R
df['y_m'] = df['latitude']  * (math.pi/180) * R

# Buffer each crater point by ~500m
shapes = []
for _, row in df.iterrows():
    pt = Point(row['x_m'], row['y_m'])
    shapes.append((pt.buffer(500).__geo_interface__, 1))

crater_mask = rasterize(
    shapes,
    out_shape=(meta['height'], meta['width']),
    transform=transform,
    fill=0,
    dtype='float32'
)

# ── 5. Build cost map ────────────────────────────────────────
w1, w2, w3 = 0.4, 0.3, 0.3   # slope, roughness, crater weights

cost_map = (w1 * slope_n +
            w2 * roughness_n +
            w3 * crater_mask)

# Normalize final cost map
cost_map = normalize(cost_map)

# Set nodata pixels to highest cost (impassable)
cost_map[np.isnan(slope_n)] = 1.0

# ── 6. Save cost map ─────────────────────────────────────────
meta.update({"dtype": "float32", "count": 1, "nodata": None})
with rasterio.open("cost_map.tif", "w", **meta) as dst:
    dst.write(cost_map.astype("float32"), 1)

print("Cost map saved successfully")
print(f"  Shape:   {cost_map.shape}")
print(f"  Min:     {cost_map.min():.3f}")
print(f"  Max:     {cost_map.max():.3f}")
print(f"  Mean:    {cost_map.mean():.3f}")