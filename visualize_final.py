import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import math

# ── Load data ────────────────────────────────────────────────
with rasterio.open("cost_map.tif") as f:
    cost      = f.read(1).astype(float)
    transform = f.transform

with rasterio.open("hillshade_map.tif") as f:
    hill = f.read(1).astype(float)

with rasterio.open("slope_map.tif") as f:
    slope = f.read(1).astype(float)

path_df  = pd.read_csv("rover_path_final.csv")
stops_df = pd.read_csv("science_stops_final.csv")
crater_df= pd.read_csv("craters_latlong.csv")  # original OHRC detections

R = 1737400

def latlon_to_pixel(lat, lon):
    x_m = lon * (math.pi/180) * R
    y_m = lat * (math.pi/180) * R
    c   = (x_m - transform.c) / transform.a
    r   = (y_m - transform.f) / transform.e
    return float(r), float(c)

# Convert all to pixels
stops_df['pr'], stops_df['pc'] = zip(*[
    latlon_to_pixel(r.latitude, r.longitude)
    for _, r in stops_df.iterrows()
])
crater_df['pr'], crater_df['pc'] = zip(*[
    latlon_to_pixel(r.latitude, r.longitude)
    for _, r in crater_df.iterrows()
])

LANDING_LAT, LANDING_LON = -85.60, 26.80
land_r, land_c = latlon_to_pixel(LANDING_LAT, LANDING_LON)

# ── Figure setup ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 11), facecolor='#0d1117')
gs  = fig.add_gridspec(2, 3, 
                        width_ratios=[2.5, 1.2, 1.3],
                        height_ratios=[1, 1],
                        hspace=0.35, wspace=0.3)

ax_main  = fig.add_subplot(gs[:, 0])   # main map (full height)
ax_cost  = fig.add_subplot(gs[0, 1])   # cost map
ax_slope = fig.add_subplot(gs[1, 1])   # slope map
ax_table = fig.add_subplot(gs[:, 2])   # science stop table

# ── MAIN MAP (Hillshade + everything) ───────────────────────
hill_norm = (hill - np.nanpercentile(hill, 2)) / \
            (np.nanpercentile(hill, 98) - np.nanpercentile(hill, 2))
hill_norm = np.clip(hill_norm, 0, 1)

ax_main.imshow(hill_norm, cmap='gray', origin='upper',
               vmin=0, vmax=1)

# Overlay cost map semi-transparently
cmap_cost = LinearSegmentedColormap.from_list(
    'cost', ['#00ff0000', '#ffff0066', '#ff000099'])
ax_main.imshow(cost, cmap=cmap_cost, origin='upper',
               vmin=0, vmax=1, alpha=0.5)

# OHRC crater detections
ax_main.scatter(crater_df['pc'], crater_df['pr'],
                c='cyan', s=6, alpha=0.6,
                label='OHRC Crater Detections', zorder=3)

# Circular scan boundary
pixel_size_m = abs(transform.a)
scan_radius_px = 2250 / pixel_size_m
circle = plt.Circle((land_c, land_r), scan_radius_px,
                     color='yellow', fill=False,
                     linewidth=1.5, linestyle='--',
                     alpha=0.7, zorder=4,
                     label='Scan Boundary (2.25 km)')
ax_main.add_patch(circle)

# Quadrant dividers
ax_main.axhline(y=land_r, color='white', linewidth=0.8,
                linestyle=':', alpha=0.5)
ax_main.axvline(x=land_c, color='white', linewidth=0.8,
                linestyle=':', alpha=0.5)
for q, dx, dy in [('NE',15,-15),('NW',-25,-15),
                   ('SE',15,15), ('SW',-25,15)]:
    ax_main.text(land_c+dx, land_r+dy, q,
                 color='white', fontsize=8, alpha=0.6,
                 fontweight='bold')

# Rover path
ax_main.plot(path_df['col'], path_df['row'],
             color='#00ff88', linewidth=2,
             zorder=5, label='Rover Path')

# Landing point
ax_main.scatter([land_c], [land_r],
                c='white', s=120, marker='^',
                zorder=7, label='Landing Point')
ax_main.text(land_c+3, land_r-5, 'LAND',
             color='white', fontsize=7,
             path_effects=[pe.withStroke(
                 linewidth=2, foreground='black')])

# Science stops
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(stops_df)))
for i, (_, s) in enumerate(stops_df.iterrows()):
    ax_main.scatter([s['pc']], [s['pr']],
                    c=[colors[i]], s=80, marker='D',
                    edgecolors='white', linewidths=0.8,
                    zorder=6)
    ax_main.text(s['pc']+2, s['pr']-3,
                 f"S{i+1}",
                 color='white', fontsize=6,
                 path_effects=[pe.withStroke(
                     linewidth=1.5, foreground='black')])

ax_main.set_title('Chandrayaan-2 South Pole Rover Navigation\n'
                  'LOLA DEM + OHRC Crater Detection | NE Quadrant',
                  color='white', fontsize=11, pad=8)
ax_main.set_xlabel('Column (pixels)', color='white', fontsize=8)
ax_main.set_ylabel('Row (pixels)',    color='white', fontsize=8)
ax_main.tick_params(colors='white', labelsize=7)
for sp in ax_main.spines.values():
    sp.set_edgecolor('#444')
ax_main.legend(loc='lower right', fontsize=6.5,
               facecolor='#1a1a2e', labelcolor='white',
               framealpha=0.8)

# ── COST MAP panel ───────────────────────────────────────────
cmap2 = LinearSegmentedColormap.from_list(
    'rg', ['#00cc44', '#ffdd00', '#cc0000'])
im = ax_cost.imshow(cost, cmap=cmap2, origin='upper',
                    vmin=0, vmax=1)
ax_cost.scatter(stops_df['pc'], stops_df['pr'],
                c='white', s=20, marker='D', zorder=3)
ax_cost.scatter([land_c], [land_r],
                c='white', s=40, marker='^', zorder=4)
plt.colorbar(im, ax=ax_cost, fraction=0.046,
             label='Cost')
ax_cost.set_title('Cost Map', color='white', fontsize=9)
ax_cost.tick_params(colors='white', labelsize=6)
for sp in ax_cost.spines.values():
    sp.set_edgecolor('#444')

# ── SLOPE MAP panel ──────────────────────────────────────────
ax_slope.imshow(slope, cmap='hot', origin='upper')
ax_slope.scatter(stops_df['pc'], stops_df['pr'],
                 c='cyan', s=20, marker='D', zorder=3)
ax_slope.scatter([land_c], [land_r],
                 c='cyan', s=40, marker='^', zorder=4)
ax_slope.set_title('Slope Map', color='white', fontsize=9)
ax_slope.tick_params(colors='white', labelsize=6)
for sp in ax_slope.spines.values():
    sp.set_edgecolor('#444')

# ── SCIENCE STOP TABLE ───────────────────────────────────────
ax_table.set_facecolor('#0d1117')
ax_table.axis('off')

ax_table.text(0.5, 0.98,
              'Science Stop Report',
              transform=ax_table.transAxes,
              color='white', fontsize=10,
              fontweight='bold', ha='center', va='top')
ax_table.text(0.5, 0.93,
              'Quadrant NE | Scan radius: 2.25 km',
              transform=ax_table.transAxes,
              color='#aaaaaa', fontsize=7.5,
              ha='center', va='top')

y = 0.88
for i, (_, s) in enumerate(stops_df.iterrows()):
    c = colors[i]
    ax_table.add_patch(mpatches.FancyBboxPatch(
        (0.02, y-0.065), 0.96, 0.068,
        boxstyle="round,pad=0.01",
        facecolor=(*c[:3], 0.15),
        edgecolor=(*c[:3], 0.6),
        transform=ax_table.transAxes,
        zorder=1
    ))
    ax_table.text(0.06, y-0.01,
                  f"S{i+1}  {s['latitude']:.4f}°, "
                  f"{s['longitude']:.4f}°",
                  transform=ax_table.transAxes,
                  color='white', fontsize=7,
                  fontweight='bold', va='top')
    ax_table.text(0.06, y-0.033,
                  f"Score:{s['science_score']:.3f}  "
                  f"Slope:{s['slope_deg']:.1f}°  "
                  f"Rough:{s['roughness_m']:.0f}m  "
                  f"Dist:{s['dist_from_landing_km']:.2f}km",
                  transform=ax_table.transAxes,
                  color='#cccccc', fontsize=6.2, va='top')
    # Short justification
    just = s['justification'].split(' | ')[0]
    ax_table.text(0.06, y-0.053,
                  f"→ {just[:55]}",
                  transform=ax_table.transAxes,
                  color='#88cc88', fontsize=5.8, va='top',
                  style='italic')
    y -= 0.082

# Summary stats
ax_table.text(0.5, 0.03,
              f"Total path: 8.29 km  |  "
              f"Waypoints: 70  |  "
              f"Algorithm: A*",
              transform=ax_table.transAxes,
              color='#888888', fontsize=6.5,
              ha='center', va='bottom')

fig.suptitle(
    'Chandrayaan-2 Lunar South Pole Rover Mission\n'
    'LOLA 118m DEM + OHRC YOLO Crater Detection | '
    'Circular Scan → Quadrant Selection → A* Navigation',
    color='white', fontsize=12, y=0.98
)

plt.savefig("rover_mission_final.png",
            dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.show()
print("✅ Saved rover_mission_final.png")