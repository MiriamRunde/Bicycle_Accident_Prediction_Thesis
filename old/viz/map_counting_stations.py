import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx

# Load your data
df = pd.read_csv('features_with_counts.csv')

# Filter for valid counter locations
df_filtered = df[df['SiteID'].str.contains('SG', na=False)].dropna(subset=['Latitude', 'Longitude'])
df_filtered = df_filtered[~df_filtered['SiteID'].isin(['SG015', 'SG020', 'SG031'])]
counter_locations = df_filtered[['SiteID', 'Latitude', 'Longitude']].drop_duplicates()

# Create GeoDataFrame
geometry = [Point(xy) for xy in zip(counter_locations['Longitude'], counter_locations['Latitude'])]
gdf = gpd.GeoDataFrame(counter_locations, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)

# Plot with basemap
fig, ax = plt.subplots(figsize=(8, 8))
gdf.plot(ax=ax, color='violet', markersize=50, alpha=0.8, edgecolor='black', label='Counters')
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.set_title('London Counter Locations with Basemap', fontweight="bold", fontsize=22)
ax.set_axis_off()
plt.tight_layout()
plt.show()
