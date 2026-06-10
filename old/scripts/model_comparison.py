import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error


# ----------------------------------------
# 1. Load and preprocess data
# ----------------------------------------
df = pd.read_csv("features_with_counts.csv", low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df['day_of_week'] = df['Date'].dt.dayofweek
df['season'] = df['Date'].dt.month % 12 // 3
df['season'] = df['season'].map({0: 'winter', 1: 'spring', 2: 'summer', 3: 'fall'})
df = pd.concat([df, pd.get_dummies(df['season'], prefix='season')], axis=1)

uk_key_holidays = pd.to_datetime([
    '2022-01-01', '2022-12-25', '2022-12-26',
    '2023-01-01', '2023-12-25', '2023-12-26'
])
df['is_holiday'] = df['Date'].isin(uk_key_holidays).astype(int)
df['weekend'] = (df['day_of_week'] >= 5).astype(int)
df['day_of_month'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df = df[df['GroupType'] == 'SG'].copy()

# Interpolate weather
for col in ['tavg', 'tmin', 'tmax', 'wspd', 'pres', 'wdir', 'wpgt']:
    df[col] = df[col].interpolate(method='linear', limit_direction='both')

# Clean max_speed
df['max_speed'] = df['max_speed'].str.replace('mph', '', regex=True).str.strip()
df['max_speed'] = pd.to_numeric(df['max_speed'], errors='coerce')

# One-hot encoding
df = pd.get_dummies(df, columns=['Borough', 'Names'], prefix=['Borough', 'Names'], drop_first=True)

# Exclude specific grids
excluded_grids = [138, 355]
df = df[~df['grid'].isin(excluded_grids)]

print(f"Excluded grids: {excluded_grids}")
print(f"Remaining grids: {df['grid'].nunique()} grids")

# Check for columns containing the value 'none'
columns_with_none = [col for col in df.columns if df[col].astype(str).str.contains('none', na=False).any()]

if columns_with_none:
    print(f"Columns containing 'none': {columns_with_none}")
    # Replace 'none' with NaN
    for col in columns_with_none:
        df[col] = df[col].replace('none', np.nan)
else:
    print("No columns contain the value 'none'.")

# Optional: Handle NaN values (e.g., fill with 0 or interpolate)
df.fillna(0, inplace=True)  # Replace NaN with 0 (or use another strategy)

# ----------------------------------------
# 2. Select features
# ----------------------------------------

# Define base lists first
land_use_250m = [
    "landuse_residential_pct_250m", "landuse_commercial_pct_250m", "landuse_industrial_pct_250m",
    "landuse_retail_pct_250m", "landuse_park_pct_250m", "landuse_forest_pct_250m",
    "landuse_meadow_pct_250m", "landuse_farmland_pct_250m", "landuse_grass_pct_250m",
    "landuse_cemetery_pct_250m", "landuse_allotments_pct_250m",
    "landuse_recreation_ground_pct_250m", "landuse_water_pct_250m",
    "landuse_wood_pct_250m", "landuse_construction_pct_250m"
]

# Now build the dictionary
feature_groups = {
    "spatial_temporal": ["grid_mid_lat", "grid_mid_lon", 
                         "Latitude", "Longitude", "HECTARES",
                         "day_of_week", "day_of_month", "month", "weekend",
                         "season_spring", "season_summer", "season_fall", 
                         "season_winter", "is_holiday"
    ],
    
    "meteorological": [
        "tavg", "tmin", "tmax", "wdir", "wspd", "wpgt", "pres"
    ],
    "proximity_traffic": [
        "distance_to_center_m", "max_speed"
    ],
    "cycle_infra_250m": [
        "cycleway_type",
        "bike_lane_length_250m", "bike_lane_density_250m",
        "cycleway_track_length_250m", "cycleway_track_density_250m",
        "cycleway_lane_length_250m", "cycleway_lane_density_250m",
        "cycleway_opposite_length_250m", "cycleway_opposite_density_250m",
        "cycleway_opposite_lane_length_250m", "cycleway_opposite_lane_density_250m",
        "cycleway_opposite_track_length_250m", "cycleway_opposite_track_density_250m",
        "cycleway_shared_lane_length_250m", "cycleway_shared_lane_density_250m",
        "cycleway_dedicated_length_250m", "cycleway_dedicated_density_250m",
        "bike_parking_spots_250m"
    ],
    "cycle_infra_500m": [
        "bike_lane_length_500m", "bike_lane_density_500m",
        "cycleway_track_length_500m", "cycleway_track_density_500m",
        "cycleway_lane_length_500m", "cycleway_lane_density_500m",
        "cycleway_opposite_length_500m", "cycleway_opposite_density_500m",
        "cycleway_opposite_lane_length_500m", "cycleway_opposite_lane_density_500m",
        "cycleway_opposite_track_length_500m", "cycleway_opposite_track_density_500m",
        "cycleway_shared_lane_length_500m", "cycleway_shared_lane_density_500m",
        "cycleway_dedicated_length_500m", "cycleway_dedicated_density_500m",
        "bike_parking_spots_500m"
    ],
    "cycle_infra_1000m": [
        "bike_lane_length_1000m", "bike_lane_density_1000m",
        "cycleway_track_length_1000m", "cycleway_track_density_1000m",
        "cycleway_lane_length_1000m", "cycleway_lane_density_1000m",
        "cycleway_opposite_length_1000m", "cycleway_opposite_density_1000m",
        "cycleway_opposite_lane_length_1000m", "cycleway_opposite_lane_density_1000m",
        "cycleway_opposite_track_length_1000m", "cycleway_opposite_track_density_1000m",
        "cycleway_shared_lane_length_1000m", "cycleway_shared_lane_density_1000m",
        "cycleway_dedicated_length_1000m", "cycleway_dedicated_density_1000m",
        "bike_parking_spots_1000m"
    ],
    "land_use_250m": [
        "landuse_residential_pct_250m", "landuse_commercial_pct_250m", "landuse_industrial_pct_250m",
        "landuse_retail_pct_250m", "landuse_park_pct_250m", "landuse_forest_pct_250m",
        "landuse_meadow_pct_250m", "landuse_farmland_pct_250m", "landuse_grass_pct_250m",
        "landuse_cemetery_pct_250m", "landuse_allotments_pct_250m",
        "landuse_recreation_ground_pct_250m", "landuse_water_pct_250m",
        "landuse_wood_pct_250m", "landuse_construction_pct_250m"
    ],
    "amenities_250m": ["shops_250m", "education_250m", "hospitals_250m", "hotels_250m"],
    "amenities_500m": ["shops_500m", "education_500m", "hospitals_500m", "hotels_500m"],
    "amenities_1000m": ["shops_1000m", "education_1000m", "hospitals_1000m", "hotels_1000m"],
    "infrastructure_250m": [
        "pt_infra_count_250m", "bus_stop_count_250m", "railway_stop_count_250m",
        "traffic_signals_250m", "crossings_250m", "intersections_250m"
    ],
    "infrastructure_500m": [
        "pt_infra_count_500m", "bus_stop_count_500m", "railway_stop_count_500m",
        "traffic_signals_500m", "crossings_500m", "intersections_500m"
    ],
    "infrastructure_1000m": [
        "pt_infra_count_1000m", "bus_stop_count_1000m", "railway_stop_count_1000m",
        "traffic_signals_1000m", "crossings_1000m", "intersections_1000m"
    ],
    "demographics": [
        "2023", "Aged 0-15_2013", "Aged 16-64_2013", "Aged 65+_2013",
        "pop_dens_2023", "Mean_inclome_2012/13"
    ],
    "accident_data": ["total_road_cacausualties_2014"],
    "target_columns": ["Daily Count", "passing_bikes", "rented_bikes", "returned_bikes"]
}


print(feature_groups)

print("Meteorological features:", feature_groups["meteorological"])

selected_features = (
    feature_groups["meteorological"] +
    ["grid_mid_lat", "grid_mid_lon"] +
    feature_groups["proximity_traffic"] +
    feature_groups["amenities_250m"] + 
    #feature_groups["amenities_500m"] + 
    #feature_groups["amenities_1000m"] +
    feature_groups["demographics"] +
    #feature_groups["accident_data"] +
    feature_groups["land_use_250m"] + 
    feature_groups["infrastructure_250m"] + 
    #feature_groups["infrastructure_500m"] + 
    #feature_groups["infrastructure_1000m"] +
    feature_groups["cycle_infra_250m"] + 
    #feature_groups["cycle_infra_500m"] + 
    #feature_groups["cycle_infra_1000m"] +
    feature_groups["spatial_temporal"] +
    feature_groups["demographics"] +
    feature_groups["spatial_temporal"]
)



duplicates = [feature for feature in selected_features if selected_features.count(feature) > 1]
print("Duplicate features:", set(duplicates))

# Check for columns with NaNs in the selected features
nan_columns = [col for col in selected_features if df[col].isna().any()]

# Print the columns with NaNs
if nan_columns:
    print("Columns with NaNs:", nan_columns)
else:
    print("No columns with NaNs in the selected features.")

# ----------------------------------------
# 3. Scale target per grid
# ----------------------------------------
scaler_dict = {}
df['Scaled_Count'] = 0.0
for grid_val in df['grid'].unique():
    scaler = StandardScaler()
    mask = df['grid'] == grid_val
    scaled = scaler.fit_transform(df.loc[mask, ['Daily Count']]).flatten()
    df.loc[mask, 'Scaled_Count'] = scaled
    scaler_dict[grid_val] = scaler

global_scaler = StandardScaler()
global_scaler.fit(df[['Daily Count']])

# ----------------------------------------
# 4. Baseline models
# ----------------------------------------
X = df[selected_features]
y = df['Scaled_Count']
groups = df['grid']


# Convert all dummy/season columns to float64
for col in ['season_spring', 'season_summer', 'season_fall', 'season_winter']:
    X[col] = X[col].astype(float)



logo = LeaveOneGroupOut()

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))

def evaluate(name, yt, yp):
    print(f"{name}: MAE={mean_absolute_error(yt, yp):.3f}, RMSE={np.sqrt(mean_squared_error(yt, yp)):.3f}, R2={r2_score(yt, yp):.3f}, SMAPE={smape(yt, yp):.2f}%")


# Define minimal hyperparameter grids for each model
param_grids = {
    'RandomForest': {
        'n_estimators': [200],  # Reduced from larger ranges
        'max_depth': [None],    # Minimal options
        'min_samples_split': [2, 5]  # Minimal options
    },
    'GBR': {
        'n_estimators': [50, 100],  # Reduced from larger ranges
        'learning_rate': [0.1],     # Fixed to 0.1
        'max_depth': [3, 5]         # Minimal options
    },
    'XGBoost': {
        'n_estimators': [50, 100],  # Reduced from larger ranges
        'learning_rate': [0.1],     # Fixed to 0.1
        'max_depth': [3, 5]         # Minimal options
    }
}

# Initialize models
models = {
    'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0),
    'LinearRegression': LinearRegression(),  # No tuning required
    'RandomForest': RandomForestRegressor(random_state=42),
    'GBR': GradientBoostingRegressor(random_state=42)
}

non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Non-numeric columns: {non_numeric_cols}")


# Perform hyperparameter tuning
best_models = {}
for name, model in tqdm(models.items(), desc="Hyperparameter Tuning"):
    print(f"\n▶ Tuning hyperparameters for {name}...")
    if name in param_grids:
        grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
        if name == 'XGBoost':
            X_temp = X.copy()
            X_temp = X_temp.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float64)
            X_temp = X_temp.values  # Convert to NumPy array
        else:
            X_temp = X
        grid_search.fit(X_temp, y)
        best_models[name] = grid_search.best_estimator_
        print(f"✅ Best parameters for {name}: {grid_search.best_params_}")
    else:
        best_models[name] = model.fit(X, y)
        print("⚠️ No hyperparameter grid defined — using default model.")


# Store predictions and true values for each model
all_results = {}

for name, model in tqdm(best_models.items(), desc="Evaluating Models"):
    preds_logo = []
    y_true_logo = []
    
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[test_idx]

        X_tr = X_tr.apply(pd.to_numeric, errors='coerce')
        X_val = X_val.apply(pd.to_numeric, errors='coerce')

        if name == 'XGBoost':
            X_tr = X_tr.to_numpy(dtype=np.float64)
            X_val = X_val.to_numpy(dtype=np.float64)

        model.fit(X_tr, y_tr)
        y_pred_scaled = model.predict(X_val)

        heldout_grid = groups.iloc[test_idx].iloc[0]
        scaler = scaler_dict.get(heldout_grid, global_scaler)
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = df.iloc[test_idx]['Daily Count'].values

        preds_logo.extend(y_pred)
        y_true_logo.extend(y_true)

    # Store for this model
    all_results[name] = {
        "y_true": np.array(y_true_logo),
        "y_pred": np.array(preds_logo)
    }

    print(f"\n{name} - LOGO CV Evaluation (Inverse-Scaled):")
    evaluate(name, np.array(y_true_logo), np.array(preds_logo))


    # Removed holdout evaluation (since X_test, y_test are undefined)


import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ----------------------------------------
# 5. Plot results
# ----------------------------------------
results = []
for name, res in all_results.items():
    y_true = res["y_true"]
    y_pred = res["y_pred"]
    smape_val = smape(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)
    results.append((name, smape_val, mae_val, r2_val))

results_df = pd.DataFrame(results, columns=["Model", "SMAPE", "MAE", "R2"]).sort_values(by="SMAPE")
print(results_df)


# Define custom hex colors for each model
custom_colors = {
    "LinearRegression": "#3b0f70",  # Example: Bright orange
    "RandomForest": "#ca3e72",       # Example: Bright green
    "GBR": "#f7705c",  # Example: Bright blue
    "XGBoost": "#fe9f6d",            # Example: Bright pink
    "SVR": "#fde7a9"                 # Example: Bright yellow
}

# Create a dictionary to map models to colors
model_colors = {model: custom_colors[model] for model in results_df["Model"]}

plt.figure(figsize=(16, 5))

# SMAPE Plot
plt.subplot(1, 3, 1)
plt.bar(results_df["Model"], results_df["SMAPE"], color=[model_colors[model] for model in results_df["Model"]], edgecolor="black")
plt.title("SMAPE Comparison", fontsize=16, fontweight="bold")
plt.ylabel("SMAPE (%)", fontsize=12)
plt.xticks(rotation=45, fontsize=10)

# MAE Plot
plt.subplot(1, 3, 2)
plt.bar(results_df["Model"], results_df["MAE"], color=[model_colors[model] for model in results_df["Model"]], edgecolor="black")
plt.title("MAE Comparison", fontsize=16, fontweight="bold")
plt.ylabel("MAE", fontsize=12)
plt.xticks(rotation=45, fontsize=10)

# R² Plot
plt.subplot(1, 3, 3)
plt.bar(results_df["Model"], results_df["R2"], color=[model_colors[model] for model in results_df["Model"]], edgecolor="black")
plt.title("R² Score Comparison", fontsize=16, fontweight="bold")
plt.ylabel("R²", fontsize=12)
plt.xticks(rotation=45, fontsize=10)

plt.tight_layout()
plt.savefig("baseline_model_comparison_custom_colors.png", dpi=300)
plt.show()

# ----------------------------------------
# 6. Calculate Mean and Global SMAPE
# ----------------------------------------

# Global SMAPE (calculated across all predictions)
global_smape = smape(np.array(y_true_logo), np.array(preds_logo))

# Mean SMAPE (average of SMAPE values per grid)
grid_smape = []
for train_idx, test_idx in logo.split(X, y, groups):
    heldout_grid = groups.iloc[test_idx].iloc[0]
    X_test = X.iloc[test_idx]
    y_test = df.iloc[test_idx]['Daily Count']
    y_pred_scaled = best_models["RandomForest"].predict(X_test)  # Example: Random Forest
    scaler = scaler_dict.get(heldout_grid, global_scaler)
    y_pred = scaler.inverse_transform(np.array(y_pred_scaled).reshape(-1, 1)).flatten()
    grid_smape.append(smape(y_test, y_pred))

mean_smape = np.mean(grid_smape)

# Print both metrics
print(f"\nGlobal SMAPE: {global_smape:.2f}%")
print(f"Mean SMAPE (Per Grid): {mean_smape:.2f}%")

# Save metrics to a text file
with open("smape_metrics.txt", "w") as f:
    f.write(f"Global SMAPE: {global_smape:.2f}%\n")
    f.write(f"Mean SMAPE (Per Grid): {mean_smape:.2f}%\n")