import numpy as np
import rasterio
import pandas as pd
import heapq
import math
import itertools
from scipy.ndimage import binary_dilation, gaussian_filter

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
LANDING_LAT  = -85.28
LANDING_LON  =  31.20
MIN_STOPS_PER_QUADRANT = 10
SUPPRESS_RADIUS_PX     = 5
SAFE_COST_THRESHOLD    = 0.80
R_MOON                 = 1737400

# ════════════════════════════════════════════════════════════
# 1. LOAD MAPS
# ════════════════════════════════════════════════════════════
print("Loading maps...")
with rasterio.open("cost_map.tif") as f:
    cost      = f.read(1).astype(float)
    transform = f.transform

with rasterio.open("slope_map.tif") as f:
    slope = f.read(1).astype(float)

with rasterio.open("roughness_map.tif") as f:
    roughness = f.read(1).astype(float)

rows, cols    = cost.shape
pixel_size_m  = abs(transform.a)
print(f"  Map size: {rows} × {cols} pixels")
print(f"  Pixel size: {pixel_size_m:.1f} m")

# ════════════════════════════════════════════════════════════
# 2. COORDINATE HELPERS
# ════════════════════════════════════════════════════════════
def latlon_to_pixel(lat, lon):
    x_m = lon * (math.pi/180) * R_MOON
    y_m = lat * (math.pi/180) * R_MOON
    c   = (x_m - transform.c) / transform.a
    r   = (y_m - transform.f) / transform.e
    return int(r), int(c)

def pixel_to_latlon(row, col):
    x_m = transform.c + col * transform.a
    y_m = transform.f + row * transform.e
    lon = x_m / ((math.pi/180) * R_MOON)
    lat = y_m / ((math.pi/180) * R_MOON)
    return round(lat, 6), round(lon, 6)

def pixel_distance_m(r1, c1, r2, c2):
    dx = (c2-c1) * pixel_size_m
    dy = (r2-r1) * pixel_size_m
    return math.sqrt(dx*dx + dy*dy)

# ════════════════════════════════════════════════════════════
# 3. SCIENCE SCORE MAP
# ════════════════════════════════════════════════════════════
print("Building science score map...")

def normalize(arr):
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx == mn: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

slope_n     = normalize(slope)
roughness_n = normalize(roughness)
roughness_s = gaussian_filter(roughness_n, sigma=2)

crater_zone  = (cost > 0.50).astype(float)
crater_edges = binary_dilation(
    crater_zone > 0, iterations=4).astype(float)
crater_edges[crater_zone > 0] = 0

science_score = (
    0.35 * roughness_s   +
    0.35 * crater_edges  +
    0.30 * (1 - slope_n)
)
science_score[cost    > SAFE_COST_THRESHOLD] = 0
science_score[slope_n > 0.75]               = 0

# ════════════════════════════════════════════════════════════
# 4. CIRCULAR SCAN
# ════════════════════════════════════════════════════════════
print(f"\nStarting circular scan...")
landing_row, landing_col = latlon_to_pixel(LANDING_LAT, LANDING_LON)
print(f"  Landing pixel: ({landing_row}, {landing_col})")

max_radius_px = int(math.sqrt(rows**2 + cols**2) / 2)
RING_STEP     = 2

def get_quadrant(r, c, ref_r, ref_c):
    if r <= ref_r and c >= ref_c: return "NE"
    if r <= ref_r and c <  ref_c: return "NW"
    if r >  ref_r and c >= ref_c: return "SE"
    return "SW"

quadrant_stops   = {"NE":[],"NW":[],"SE":[],"SW":[]}
quadrant_done    = {"NE":False,"NW":False,"SE":False,"SW":False}
suppressed       = np.zeros((rows,cols), dtype=bool)
winning_quadrant = None
winning_stops    = None
scan_radius_m    = None

for radius_px in range(5, max_radius_px, RING_STEP):
    r_min = max(0, landing_row - radius_px)
    r_max = min(rows, landing_row + radius_px + 1)
    c_min = max(0, landing_col - radius_px)
    c_max = min(cols, landing_col + radius_px + 1)

    rr, cc = np.meshgrid(
        np.arange(r_min, r_max),
        np.arange(c_min, c_max),
        indexing='ij'
    )
    dist      = np.sqrt((rr-landing_row)**2 + (cc-landing_col)**2)
    ring_mask = (dist >= radius_px-RING_STEP) & (dist < radius_px)
    ring_r, ring_c = np.where(ring_mask)

    for i in range(len(ring_r)):
        pr = ring_r[i] + r_min
        pc = ring_c[i] + c_min
        if not (0 <= pr < rows and 0 <= pc < cols): continue
        if suppressed[pr, pc]:                       continue
        if cost[pr, pc] > SAFE_COST_THRESHOLD:       continue
        sc = science_score[pr, pc]
        if sc <= 0:                                  continue

        q = get_quadrant(pr, pc, landing_row, landing_col)
        if quadrant_done[q]:                         continue

        too_close = any(
            pixel_distance_m(pr, pc, s['row'], s['col'])
            < SUPPRESS_RADIUS_PX * pixel_size_m
            for s in quadrant_stops[q]
        )
        if too_close: continue

        slope_deg   = float(slope[pr, pc])
        rough_val   = float(roughness[pr, pc])
        near_crater = bool(crater_edges[pr, pc] > 0)
        dist_land   = pixel_distance_m(
            pr, pc, landing_row, landing_col)

        just = []
        if near_crater:
            just.append(
                "Near crater rim — high mineralogical interest")
        if roughness_n[pr,pc] > 0.5:
            just.append(
                f"Elevated roughness ({rough_val:.1f}m) — "
                f"varied surface composition")
        if slope_n[pr,pc] < 0.3:
            just.append(
                f"Gentle slope ({slope_deg:.1f}°) — "
                f"safe for rover instruments")
        else:
            just.append(
                f"Moderate slope ({slope_deg:.1f}°) — "
                f"within rover traversal limits")
        just.append(
            f"Distance from landing: {dist_land/1000:.2f} km")
        just.append(
            f"Science score: {sc:.4f} "
            f"(slope={slope_n[pr,pc]:.2f}, "
            f"roughness={roughness_n[pr,pc]:.2f}, "
            f"crater_edge={float(crater_edges[pr,pc]):.2f})")

        lat, lon = pixel_to_latlon(pr, pc)
        quadrant_stops[q].append({
            "stop_id":       f"{q}_{len(quadrant_stops[q])+1:02d}",
            "quadrant":      q,
            "row":           pr,
            "col":           pc,
            "latitude":      lat,
            "longitude":     lon,
            "science_score": round(sc,4),
            "slope_deg":     round(slope_deg,2),
            "roughness_m":   round(rough_val,2),
            "near_crater":   near_crater,
            "dist_from_landing_km": round(dist_land/1000,3),
            "justification": " | ".join(just)
        })

        suppressed[
            max(0,pr-SUPPRESS_RADIUS_PX):pr+SUPPRESS_RADIUS_PX,
            max(0,pc-SUPPRESS_RADIUS_PX):pc+SUPPRESS_RADIUS_PX
        ] = True

    for q, stops in quadrant_stops.items():
        if len(stops) >= MIN_STOPS_PER_QUADRANT \
                and not quadrant_done[q]:
            quadrant_done[q] = True
            if winning_quadrant is None:
                winning_quadrant = q
                winning_stops    = stops[:MIN_STOPS_PER_QUADRANT]
                scan_radius_m    = radius_px * pixel_size_m
                print(f"\n  🏆 Quadrant {q} reached 10 stops first!")
                print(f"     Scan radius: {scan_radius_m/1000:.2f} km")

    if winning_quadrant is not None:
        break

# ════════════════════════════════════════════════════════════
# 5. A* BETWEEN TWO POINTS
# ════════════════════════════════════════════════════════════
def astar(cost_map, start, end):
    """Returns path as list of (row,col) tuples, or None"""
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g = np.full(cost_map.shape, np.inf)
    g[start] = 0
    f = np.full(cost_map.shape, np.inf)
    f[start] = math.sqrt(
        (start[0]-end[0])**2 + (start[1]-end[1])**2)

    nbrs = [(-1,-1),(-1,0),(-1,1),
            ( 0,-1),       ( 0,1),
            ( 1,-1),( 1,0),( 1,1)]

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == end:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            return path[::-1]

        for dr, dc in nbrs:
            nr, nc = cur[0]+dr, cur[1]+dc
            if not (0 <= nr < cost_map.shape[0] and
                    0 <= nc < cost_map.shape[1]): continue
            if cost_map[nr,nc] >= SAFE_COST_THRESHOLD: continue
            mv  = 1.414 if (dr and dc) else 1.0
            tg  = g[cur] + mv*(1+cost_map[nr,nc])
            if tg < g[nr,nc]:
                came_from[(nr,nc)] = cur
                g[nr,nc] = tg
                f[nr,nc] = tg + math.sqrt(
                    (nr-end[0])**2+(nc-end[1])**2)
                heapq.heappush(open_set,(f[nr,nc],(nr,nc)))
    return None

# ════════════════════════════════════════════════════════════
# 6. FIX START AND END, OPTIMIZE MIDDLE STOPS
# ════════════════════════════════════════════════════════════
print("\nOptimizing stop order...")

# Sort by distance — farthest stop is fixed as end
winning_stops_sorted = sorted(
    winning_stops,
    key=lambda s: s['dist_from_landing_km']
)

fixed_end   = winning_stops_sorted[-1]   # farthest = end
middle_stops = winning_stops_sorted[:-1]  # remaining 9

# ── Nearest neighbor through middle stops ───────────────────
def nearest_neighbor_open(stops, start_r, start_c, end_stop):
    """
    Nearest neighbor with forward bias:
    Penalizes stops that require moving back toward landing
    """
    remaining = stops.copy()
    ordered   = []
    cur_r, cur_c   = start_r, start_c
    land_r, land_c = start_r, start_c

    while remaining:
        def score(s):
            # Euclidean distance to candidate
            dist = math.sqrt(
                (s['row'] - cur_r)**2 +
                (s['col'] - cur_c)**2
            )
            # Distance from landing (higher = further = better)
            progress = math.sqrt(
                (s['row'] - land_r)**2 +
                (s['col'] - land_c)**2
            )
            # Penalize stops closer to landing than current pos
            cur_progress = math.sqrt(
                (cur_r - land_r)**2 +
                (cur_c - land_c)**2
            )
            backward_penalty = max(
                0, cur_progress - progress) * 2.0

            return dist + backward_penalty

        nearest = min(remaining, key=score)
        ordered.append(nearest)
        cur_r, cur_c = nearest['row'], nearest['col']
        remaining.remove(nearest)

    ordered.append(end_stop)
    return ordered

winning_stops_ordered = nearest_neighbor_open(
    middle_stops,
    landing_row, landing_col,
    fixed_end
)

# Reassign IDs
for i, stop in enumerate(winning_stops_ordered):
    stop['stop_id'] = f"{winning_quadrant}_{i+1:02d}"

print(f"  Fixed end: {fixed_end['stop_id']} "
      f"at {fixed_end['dist_from_landing_km']:.2f} km")
print(f"\nOptimized stop order:")
for s in winning_stops_ordered:
    print(f"  {s['stop_id']}  "
          f"dist={s['dist_from_landing_km']:.2f}km  "
          f"({s['latitude']}, {s['longitude']})")

# ════════════════════════════════════════════════════════════
# 7. BUILD PATH: LANDING → STOP1 → ... → STOP10
# ════════════════════════════════════════════════════════════
print(f"\nPlanning A* path in optimized order...")

waypoints = [(landing_row, landing_col)] + \
            [(s['row'], s['col']) 
             for s in winning_stops_ordered]

full_path    = []
total_dist_m = 0

for k in range(len(waypoints)-1):
    seg = astar(cost, waypoints[k], waypoints[k+1])
    if seg:
        total_dist_m += len(seg) * pixel_size_m
        full_path.extend(seg[:-1])
    else:
        print(f"  ⚠️  No path between "
              f"waypoint {k} and {k+1}")

full_path.append(waypoints[-1])

print(f"  ✅ Total waypoints: {len(full_path)}")
print(f"  ✅ Total distance:  {total_dist_m/1000:.2f} km")

# ════════════════════════════════════════════════════════════
# 9. SAVE RESULTS
# ════════════════════════════════════════════════════════════
stops_df = pd.DataFrame(winning_stops_ordered)
stops_df.to_csv("science_stops_final.csv", index=False)
print(f"\n✅ Science stops saved")

path_records = []
for step,(r,c) in enumerate(full_path):
    lat, lon = pixel_to_latlon(r, c)
    path_records.append({
        "step": step, "row": r, "col": c,
        "latitude": lat, "longitude": lon,
        "cost": round(float(cost[r,c]),4)
    })
pd.DataFrame(path_records).to_csv(
    "rover_path_final.csv", index=False)
print(f"✅ Rover path saved")

# Print report
print(f"\n{'='*60}")
print(f"SCIENCE STOP REPORT — Quadrant {winning_quadrant}")
print(f"Algorithm: Circular Scan + TSP (NN + 2-opt) + A*")
print(f"Scan radius: {scan_radius_m/1000:.2f} km")
print(f"Path distance: {total_dist_m/1000:.2f} km")
print(f"{'='*60}")
for s in winning_stops_ordered:
    print(f"\n{s['stop_id']}  "
          f"({s['latitude']}, {s['longitude']})")
    print(f"  Score:{s['science_score']}  "
          f"Slope:{s['slope_deg']}°  "
          f"Rough:{s['roughness_m']}m  "
          f"Near crater:{s['near_crater']}")
    for j in s['justification'].split(' | '):
        print(f"  → {j}")