import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx



# Load CSV (data)
data_df = pd.read_csv("/Users/miriam/Documents/GitHub/thesis_clean/full_daily_grid_level_with_accidents.csv")  # Replace with your CSV filename


data_df['Date'] = pd.to_datetime(data_df['Date'])
data_df['day_of_week'] = data_df['Date'].dt.dayofweek
data_df['day_of_month'] = data_df['Date'].dt.day
data_df['month'] = data_df['Date'].dt.month
# Season as categorical
season_map = {12: 'winter', 1: 'winter', 2: 'winter',
              3: 'spring', 4: 'spring', 5: 'spring',
              6: 'summer', 7: 'summer', 8: 'summer',
              9: 'fall', 10: 'fall', 11: 'fall'}
data_df['season'] = data_df['Date'].dt.month.map(season_map)
data_df = pd.concat([data_df, pd.get_dummies(data_df['season'], prefix='season')], axis=1)
# UK key holidays
uk_key_holidays = pd.to_datetime([
    '2022-01-01', '2022-12-25', '2022-12-26',
    '2023-01-01', '2023-12-25', '2023-12-26'
])
data_df['is_holiday'] = (data_df['Date'].isin(uk_key_holidays)).astype(int)
data_df['weekend'] = (data_df['day_of_week'] >= 5).astype(int)


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

# Log-transform (avoid log(0) by adding 1)
merged_gdf['log_total_bikes'] = np.log1p(merged_gdf['total_bikes'])
merged_gdf['log_accidents'] = np.log1p(merged_gdf['accident'])

# Pearson correlation (raw)
corr_raw, _ = pearsonr(merged_gdf['total_bikes'], merged_gdf['accident'])
print(f"Pearson correlation (raw): {corr_raw:.2f}")

# Pearson correlation (log-log)
corr_log, _ = pearsonr(merged_gdf['log_total_bikes'], merged_gdf['log_accidents'])
print(f"Pearson correlation (log-log): {corr_log:.2f}")



land_use_250m = [
    'landuse_residential_pct_250m', 'landuse_commercial_pct_250m'
]
feature_groups = {
    'spatial_temporal': ['weekend'
    ],
    'meteorological': ['tavg'],
    'proximity_traffic': ['distance_to_center_m', 'max_speed'],
    'cycle_infra_250m': ['bike_lane_density_250m'
    ],
    'cycle_infra_500m': [
        'bike_lane_length_500m', 'bike_lane_density_500m',
        'cycleway_track_length_500m', 'cycleway_track_density_500m',
        'cycleway_lane_length_500m', 'cycleway_lane_density_500m',
        'cycleway_opposite_length_500m', 'cycleway_opposite_density_500m',
        'cycleway_opposite_lane_length_500m', 'cycleway_opposite_lane_density_500m',
        'cycleway_opposite_track_length_500m', 'cycleway_opposite_track_density_500m',
        'cycleway_shared_lane_length_500m', 'cycleway_shared_lane_density_500m',
        'cycleway_dedicated_length_500m', 'cycleway_dedicated_density_500m',
        'bike_parking_spots_500m'
    ],
    'cycle_infra_1000m': [
        'bike_lane_length_1000m', 'bike_lane_density_1000m',
        'cycleway_track_length_1000m', 'cycleway_track_density_1000m',
        'cycleway_lane_length_1000m', 'cycleway_lane_density_1000m',
        'cycleway_opposite_length_1000m', 'cycleway_opposite_density_1000m',
        'cycleway_opposite_lane_length_1000m', 'cycleway_opposite_lane_density_1000m',
        'cycleway_opposite_track_length_1000m', 'cycleway_opposite_track_density_1000m',
        'cycleway_shared_lane_length_1000m', 'cycleway_shared_lane_density_1000m',
        'cycleway_dedicated_length_1000m', 'cycleway_dedicated_density_1000m',
        'bike_parking_spots_1000m'
    ],
    'land_use_250m': land_use_250m,
    'amenities_250m': ['shops_250m', 'education_250m', 'hospitals_250m', 'hotels_250m'],
    'amenities_500m': ['shops_500m', 'education_500m', 'hospitals_500m', 'hotels_500m'],
    'amenities_1000m': ['shops_1000m', 'education_1000m', 'hospitals_1000m', 'hotels_1000m'],
    'infrastructure_250m': ['pt_infra_count_250m', 'bus_stop_count_250m', 'railway_stop_count_250m', 'traffic_signals_250m', 'crossings_250m', 'intersections_250m'],
    'infrastructure_500m': ['pt_infra_count_500m', 'bus_stop_count_500m', 'railway_stop_count_500m', 'traffic_signals_500m', 'crossings_500m', 'intersections_500m'],
    'infrastructure_1000m': ['pt_infra_count_1000m', 'bus_stop_count_1000m', 'railway_stop_count_1000m', 'traffic_signals_1000m', 'crossings_1000m', 'intersections_1000m'],
    'demographics': ['pop_dens_2023'],
    'accident_data': ['total_road_cacausualties_2014'],
    'target_columns': ['Daily Count', 'passing_bikes', 'rented_bikes', 'returned_bikes']
}
# Combine into feature list
selected_features = (
    feature_groups['meteorological'] +
    feature_groups['proximity_traffic'] +
    #feature_groups['amenities_250m'] +
    #feature_groups['amenities_500m'] +
    feature_groups['demographics'] +
    feature_groups['land_use_250m'] +
    #feature_groups['infrastructure_250m'] +
    #feature_groups['infrastructure_500m'] +
    feature_groups['cycle_infra_250m'] +
    #feature_groups['cycle_infra_500m'] +
    feature_groups['spatial_temporal']
)

for c in selected_features + ['Predicted Daily Count']: data_df[c] = pd.to_numeric(data_df[c], errors='coerce')
# Aggregate selected features (mean per grid)
feature_agg = data_df.groupby('grid')[selected_features].mean().reset_index()

# Merge aggregated features into merged_gdf
merged_gdf = merged_gdf.merge(feature_agg, on='grid', how='left')



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
# Select predictors (log_total_bikes + aggregated features)
X = merged_gdf[['log_total_bikes'] + selected_features]

# Handle missing data (fill or drop)
X = X.fillna(0)  # or use X.dropna() if preferred

# Add intercept
X = sm.add_constant(X)

# Target variable
y = merged_gdf['log_accidents']

# Fit the model
model = sm.OLS(y, X).fit()

# Output summary
print(model.summary())


from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Exclude constant and target
X_no_const = X.drop(columns='const')

vif_df = pd.DataFrame()
vif_df["feature"] = X_no_const.columns
vif_df["VIF"] = [variance_inflation_factor(X_no_const.values, i) for i in range(X_no_const.shape[1])]
print(vif_df.sort_values(by="VIF", ascending=False))
