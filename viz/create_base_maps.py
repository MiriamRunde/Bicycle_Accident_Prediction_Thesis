import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

# Load CSV (data)
data_df = pd.read_csv("/Users/miriam/Documents/GitHub/thesis_clean/full_daily_grid_level_with_accidents.csv")  # Replace with your CSV filename

# First: drop duplicates for accidents per grid/day
unique_accidents_df = data_df[['grid', 'Date', 'accident']].drop_duplicates()

# Aggregate accident counts (unique accidents per grid)
accidents_agg = unique_accidents_df.groupby('grid')['accident'].sum().reset_index()

# Aggregate bike volume as before (no need to deduplicate bikes)
bike_agg = data_df.groupby('grid').agg({
    'Predicted Daily Count': ['sum', 'mean']
}).reset_index()
bike_agg.columns = ['grid', 'total_bikes', 'average_daily_bikes']

# Merge accidents and bike data
agg_df = accidents_agg.merge(bike_agg, on='grid')

print(agg_df)

# Add risk metrics to agg_df
agg_df['annual_relative_risk'] = agg_df['accident'] / agg_df['total_bikes']

# Daily relative risk per row (accident / predicted bike count)
data_df['daily_relative_risk'] = (data_df['accident'] / data_df['Predicted Daily Count'])*100
data_df['daily_relative_risk'] = data_df['daily_relative_risk'].replace([float('inf'), -float('inf')], 0).fillna(0)

# Average daily relative risk per grid
daily_risk_agg = data_df.groupby('grid')['daily_relative_risk'].mean().reset_index()
daily_risk_agg.columns = ['grid', 'mean_daily_relative_risk']

# Merge with agg_df
agg_df = agg_df.merge(daily_risk_agg, on='grid')

# Add a 'week' column to group by week
data_df['week'] = pd.to_datetime(data_df['Date']).dt.isocalendar().week

# Weekly relative risk per row (accident / predicted bike count)
data_df['weekly_relative_risk'] = (data_df['accident'] / data_df['Predicted Daily Count'])*100
data_df['weekly_relative_risk'] = data_df['weekly_relative_risk'].replace([float('inf'), -float('inf')], 0).fillna(0)

# Average weekly relative risk per grid
weekly_risk_agg = data_df.groupby('grid')['weekly_relative_risk'].mean().reset_index()
weekly_risk_agg.columns = ['grid', 'mean_weekly_relative_risk']

# Merge with agg_df
agg_df = agg_df.merge(weekly_risk_agg, on='grid')

print(agg_df)

# Load GeoJSON (grid geometries)
grid_gdf = gpd.read_file("shapefiles/grid_cells_enriched.geojson")  # Replace with your GeoJSON filename

# Merge geometries with data
merged_gdf = grid_gdf.merge(agg_df, on='grid')
merged_gdf = merged_gdf.to_crs(epsg=3857)  # Project for basemap
# Merge risks into spatial dataframe
merged_gdf = grid_gdf.merge(agg_df, on='grid', how='left')
merged_gdf = merged_gdf.to_crs(epsg=3857)
merged_gdf.fillna(0, inplace=True)  # Fill NaNs if needed
merged_gdf['total_bikes_scaled'] = merged_gdf['total_bikes'] / 1e6  # Millions


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Map 1: Total Accidents (Yearly Sum)
merged_gdf.plot(
    ax=axes[0, 0], 
    column='accident', 
    cmap='GnBu', 
    legend=True, 
    edgecolor='black', 
    linewidth=0.3, 
    alpha=0.7
)
ctx.add_basemap(axes[0, 0], source=ctx.providers.CartoDB.Positron)
axes[0, 0].set_title("Total Accidents (Yearly Sum)", fontsize=16).set_weight('bold')
axes[0, 0].axis('off')

# Map 2: Total Bike Volume (Yearly Sum)
merged_gdf.plot(
    ax=axes[0, 1], 
    column='total_bikes_scaled', 
    cmap='PuBu', 
    legend=True, 
    edgecolor='black', 
    linewidth=0.3, 
    alpha=0.7,
    legend_kwds={'format': '%.2f'}  # Two decimals
)
ctx.add_basemap(axes[0, 1], source=ctx.providers.CartoDB.Positron)
axes[0, 1].set_title("Total Bike Volume (Yearly Sum, Millions)", fontsize=16).set_weight('bold')
axes[0, 1].axis('off')

# Map 3: Average Daily Bike Volume
merged_gdf.plot(
    ax=axes[1, 0], 
    column='average_daily_bikes', 
    cmap='BuGn', 
    legend=True, 
    edgecolor='black', 
    linewidth=0.3, 
    alpha=0.7
)
ctx.add_basemap(axes[1, 0], source=ctx.providers.CartoDB.Positron)
axes[1, 0].set_title("Average Daily Bike Volume", fontsize=16).set_weight('bold')
axes[1, 0].axis('off')

# Map 4: Annual Relative Accident Risk
merged_gdf.plot(
    ax=axes[1, 1], 
    column='mean_daily_relative_risk', 
    cmap='BuPu', 
    legend=True, 
    edgecolor='black', 
    linewidth=0.3, 
    alpha=0.7,
    legend_kwds={
        'shrink': 0.6,
        'format': '%.2f%%'  # Format legend values as percentages with 2 decimals
    }
)
ctx.add_basemap(axes[1, 1], source=ctx.providers.CartoDB.Positron)
axes[1, 1].set_title("Mean Daily Relative Accident Risk (%)", fontsize=16).set_weight('bold')
axes[1, 1].axis('off')

for ax in axes.flat:
    ax.tick_params(axis='both', which='major', labelsize=40)

plt.tight_layout()
plt.show()



import numpy as np
from scipy.stats import pearsonr

# Log-transform (avoid log(0) by adding 1)
merged_gdf['log_total_bikes'] = np.log1p(merged_gdf['total_bikes'])
merged_gdf['log_accidents'] = np.log1p(merged_gdf['accident'])

# Pearson correlation (raw)
corr_raw, _ = pearsonr(merged_gdf['total_bikes'], merged_gdf['accident'])
print(f"Pearson correlation (raw): {corr_raw:.2f}")

# Pearson correlation (log-log)
corr_log, _ = pearsonr(merged_gdf['log_total_bikes'], merged_gdf['log_accidents'])
print(f"Pearson correlation (log-log): {corr_log:.2f}")



plt.figure(figsize=(8, 6))

plt.scatter(merged_gdf['total_bikes'], merged_gdf['accident'], alpha=0.7, edgecolor='k')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Total Bike Volume (log scale)')
plt.ylabel('Total Accidents (log scale)')
plt.title('Log-Log Relationship between Bike Volume and Accidents')

plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()


import statsmodels.api as sm

# Prepare log-log data
X = merged_gdf['log_total_bikes']
y = merged_gdf['log_accidents']

# Add constant for intercept
X = sm.add_constant(X)

# Fit log-log regression
model = sm.OLS(y, X).fit()

# Print results
print(model.summary())
