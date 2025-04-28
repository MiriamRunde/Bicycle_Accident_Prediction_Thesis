import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestRegressor
import warnings
import os

warnings.filterwarnings("ignore")

output_dir = "Acc_RF_model_plots"
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Load and preprocess data
# --------------------------
df = pd.read_csv("full_daily_grid_level_with_accidents.csv", low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df['week'] = df['Date'].dt.isocalendar().week
df['year'] = df['Date'].dt.year

# Define feature groups
feature_groups = {
    "spatial_temporal": [
        "grid_mid_lat", "grid_mid_lon", "HECTARES",
        "day_of_week", "day_of_month", "month", "weekend",
        "season_spring", "season_summer", "is_holiday"
    ],
    "meteorological": ["tavg", "tmin", "tmax", "wdir", "wspd", "wpgt", "pres"],
    "proximity_traffic": ["distance_to_center_m", "max_speed"],
    "amenities_250m": ["shops_250m", "education_250m", "hospitals_250m", "hotels_250m"],
    "amenities_500m": ["shops_500m", "education_500m", "hospitals_500m", "hotels_500m"],
    "demographics": ["2023", "Aged 0-15_2013", "Aged 16-64_2013", "Aged 65+_2013", "pop_dens_2023"],
    "land_use_250m": [
        "landuse_residential_pct_250m", "landuse_commercial_pct_250m", "landuse_industrial_pct_250m",
        "landuse_retail_pct_250m", "landuse_park_pct_250m", "landuse_forest_pct_250m",
        "landuse_meadow_pct_250m", "landuse_farmland_pct_250m", "landuse_grass_pct_250m",
        "landuse_cemetery_pct_250m", "landuse_allotments_pct_250m",
        "landuse_recreation_ground_pct_250m", "landuse_water_pct_250m",
        "landuse_wood_pct_250m", "landuse_construction_pct_250m"
    ],
    "infrastructure_250m": [
        "pt_infra_count_250m", "bus_stop_count_250m", "railway_stop_count_250m",
        "traffic_signals_250m", "crossings_250m", "intersections_250m"
    ],
    "infrastructure_500m": [
        "pt_infra_count_500m", "bus_stop_count_500m", "railway_stop_count_500m",
        "traffic_signals_500m", "crossings_500m", "intersections_500m"
    ],
    "cycle_infra_250m": [
        "bike_lane_length_250m", "bike_lane_density_250m",
        "cycleway_track_length_250m", "cycleway_track_density_250m",
        "cycleway_lane_length_250m", "cycleway_lane_density_250m",
        "cycleway_opposite_length_250m", "cycleway_opposite_density_250m",
        "cycleway_dedicated_length_250m", "cycleway_dedicated_density_250m",
        "bike_parking_spots_250m"
    ],
    "cycle_infra_500m": [
        "bike_lane_length_500m", "bike_lane_density_500m",
        "cycleway_track_length_500m", "cycleway_track_density_500m",
        "cycleway_lane_length_500m", "cycleway_lane_density_500m",
        "cycleway_opposite_length_500m", "cycleway_opposite_density_500m",
        "cycleway_dedicated_length_500m", "cycleway_dedicated_density_500m",
        "bike_parking_spots_500m"
    ]
}

features = [feat for group in feature_groups.values() for feat in group]

# Aggregate to weekly level
agg_dict = {feat: 'mean' for feat in features}
agg_dict['accident'] = 'sum'
weekly_df = df.groupby(['grid', 'year', 'week']).agg(agg_dict).reset_index()

if 'Predicted Daily Count' in df.columns:
    # Use sum of predicted daily counts (better for weekly exposure)
    predicted_volume = df.groupby(['grid', 'year', 'week'])['Predicted Daily Count'].sum().reset_index(name='sum_predicted_weekly_count')
    weekly_df = weekly_df.merge(predicted_volume, on=['grid', 'year', 'week'], how='left')
    
    # Relative risk calculation with summed weekly exposure
    weekly_df['accidents_per_1000_bikes'] = 1000 * weekly_df['accident'] / weekly_df['sum_predicted_weekly_count']
    weekly_df['accidents_per_1000_bikes'] = weekly_df['accidents_per_1000_bikes'].replace([np.inf, -np.inf], 0).fillna(0)

    #weekly_df['accidents_per_1000_bikes'] = weekly_df['accidents_per_1000_bikes'].replace([np.inf, -np.inf], 0).fillna(0)
else:
    raise ValueError("'Predicted Daily Count' column is required to compute relative risk.")

# --------------------------
# Forecasting setup: Train on weeks 1–40, test on 41–52
# --------------------------
train_weeks = (weekly_df['week'] >= 1) & (weekly_df['week'] <= 30)
test_weeks = (weekly_df['week'] >= 31) & (weekly_df['week'] <= 52)

X_train = weekly_df[train_weeks][features]
y_train = weekly_df[train_weeks]['accidents_per_1000_bikes']
X_test = weekly_df[test_weeks][features]
y_test = weekly_df[test_weeks]['accidents_per_1000_bikes']

imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=features)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=features)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_imputed, y_train)
weekly_df.loc[test_weeks, 'predicted_accidents_per_1000_bikes'] = model.predict(X_test_imputed)

# --------------------------
# Evaluate and print metrics
# --------------------------
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))

def print_metrics(y_true, y_pred):
    print("Evaluation on Test Set (Weeks 41–52):")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}")
    print(f"R²: {r2_score(y_true, y_pred):.3f}")
    print(f"SMAPE: {smape(y_true, y_pred):.2f}%")

print_metrics(y_test, weekly_df.loc[test_weeks, 'predicted_accidents_per_1000_bikes'])

# --------------------------
# Weekly prediction heatmaps (Weeks 31–52)
# --------------------------
grid_gdf = gpd.read_file("shapefiles/grid_cells_enriched.geojson")
viz_df = grid_gdf.merge(weekly_df, on="grid", how="left")

# Calculate the global min and max for the color scale
vmin = min(weekly_df['accidents_per_1000_bikes'].min(), weekly_df['predicted_accidents_per_1000_bikes'].min())
vmax = max(weekly_df['accidents_per_1000_bikes'].max(), weekly_df['predicted_accidents_per_1000_bikes'].max())

for week in range(31, 53):  # Iterate from week 31 to 52
    week_df = viz_df[viz_df['week'] == week]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Actual Risk Plot
    week_df.plot(
        column='accidents_per_1000_bikes', cmap='YlGnBu', legend=True, ax=ax[0],
        vmin=vmin, vmax=vmax  # Use the global color scale
    )
    ax[0].set_title(f"Actual Risk - Week {week}")
    ax[0].axis('off')

    # Predicted Risk Plot
    week_df.plot(
        column='predicted_accidents_per_1000_bikes', cmap='YlGnBu', legend=True, ax=ax[1],  # Same colormap as Actual Risk
        vmin=vmin, vmax=vmax  # Use the global color scale
    )
    ax[1].set_title(f"Predicted Risk - Week {week}")
    ax[1].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/forecast_map_week_{week}.png", dpi=300)
    plt.close()

# --------------------------
# Save prediction dataframe
# --------------------------
weekly_df.to_csv("weekly_accident_risk_predictions_rf_forecast.csv", index=False)