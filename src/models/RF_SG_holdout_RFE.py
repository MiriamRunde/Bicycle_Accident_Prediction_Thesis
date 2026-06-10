import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.feature_selection import RFECV
from sklearn.base import clone
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from tqdm import tqdm
import os
from sklearn.feature_selection import RFE

output_dir = "rf_results"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------
# 1. Load and preprocess data
# ---------------------------------

df = pd.read_csv("features_with_counts.csv", delimiter=",", low_memory=False)
# Parse dates and engineer temporal features
df['Date'] = pd.to_datetime(df['Date'])
df['day_of_week'] = df['Date'].dt.dayofweek
df['day_of_month'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
# Season as categorical
season_map = {12: 'winter', 1: 'winter', 2: 'winter',
              3: 'spring', 4: 'spring', 5: 'spring',
              6: 'summer', 7: 'summer', 8: 'summer',
              9: 'fall', 10: 'fall', 11: 'fall'}
df['season'] = df['Date'].dt.month.map(season_map)
df = pd.concat([df, pd.get_dummies(df['season'], prefix='season')], axis=1)
# UK key holidays
uk_key_holidays = pd.to_datetime([
    '2022-01-01', '2022-12-25', '2022-12-26',
    '2023-01-01', '2023-12-25', '2023-12-26'
])
df['is_holiday'] = (df['Date'].isin(uk_key_holidays)).astype(int)
df['weekend'] = (df['day_of_week'] >= 5).astype(int)

# Filter to group type SG
df = df[df['GroupType'] == 'SG'].copy()

# Interpolate weather variables
weather_cols = ['tavg', 'tmin', 'tmax', 'wspd', 'pres', 'wdir', 'wpgt']
for col in weather_cols:
    df[col] = df[col].interpolate(method='linear', limit_direction='both')
# Clean max_speed
if 'max_speed' in df.columns:
    df['max_speed'] = pd.to_numeric(
        df['max_speed'].str.replace('mph', '', regex=True).str.strip(),
        errors='coerce'
    )

# One-hot encode Borough and Names
for col in ['Borough', 'Names']:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], prefix=[col], drop_first=True)

# Replace 'none' strings with NaN then fill
for c in df.columns:
    if df[c].dtype == object and df[c].str.contains('none', na=False).any():
        df[c] = df[c].replace('none', np.nan)
df.fillna(0, inplace=True)

# ------------------------
# 2. Define feature groups
# ------------------------
land_use_250m = [
    'landuse_residential_pct_250m', 'landuse_commercial_pct_250m', 'landuse_industrial_pct_250m',
    'landuse_retail_pct_250m', 'landuse_park_pct_250m', 'landuse_forest_pct_250m',
    'landuse_meadow_pct_250m', 'landuse_farmland_pct_250m', 'landuse_grass_pct_250m',
    'landuse_cemetery_pct_250m', 'landuse_allotments_pct_250m',
    'landuse_recreation_ground_pct_250m', 'landuse_water_pct_250m',
    'landuse_wood_pct_250m', 'landuse_construction_pct_250m'
]
feature_groups = {
    'spatial_temporal': [
        'grid_mid_lat', 'grid_mid_lon', 'Latitude', 'Longitude', 'HECTARES',
        'day_of_week', 'day_of_month', 'month', 'weekend',
        'season_winter', 'season_spring', 'season_summer', 'season_fall', 'is_holiday'
    ],
    'meteorological': ['tavg', 'tmin', 'tmax', 'wspd', 'pres', 'wdir', 'wpgt'],
    'proximity_traffic': ['distance_to_center_m', 'max_speed'],
    'cycle_infra_250m': [
        'cycleway_type', 'bike_lane_length_250m', 'bike_lane_density_250m',
        'cycleway_track_length_250m', 'cycleway_track_density_250m',
        'cycleway_lane_length_250m', 'cycleway_lane_density_250m',
        'cycleway_opposite_length_250m', 'cycleway_opposite_density_250m',
        'cycleway_opposite_lane_length_250m', 'cycleway_opposite_lane_density_250m',
        'cycleway_opposite_track_length_250m', 'cycleway_opposite_track_density_250m',
        'cycleway_shared_lane_length_250m', 'cycleway_shared_lane_density_250m',
        'cycleway_dedicated_length_250m', 'cycleway_dedicated_density_250m',
        'bike_parking_spots_250m'
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
    'demographics': ['2023', 'Aged 0-15_2013', 'Aged 16-64_2013', 'Aged 65+_2013', 'pop_dens_2023', 'Mean_inclome_2012/13'],
    'accident_data': ['total_road_cacausualties_2014'],
    'target_columns': ['Daily Count', 'passing_bikes', 'rented_bikes', 'returned_bikes']
}
# Combine into feature list
selected_features = (
    feature_groups['meteorological'] +
    feature_groups['proximity_traffic'] +
    feature_groups['amenities_250m'] +
    feature_groups['amenities_500m'] +
    feature_groups['demographics'] +
    feature_groups['land_use_250m'] +
    feature_groups['infrastructure_250m'] +
    feature_groups['infrastructure_500m'] +
    feature_groups['cycle_infra_250m'] +
    feature_groups['cycle_infra_500m'] +
    feature_groups['spatial_temporal']
)

# Exclude specific grids and drop NA target
excluded_grids = [138, 355]
df = df[~df['grid'].isin(excluded_grids)].copy()

df.dropna(subset=selected_features + ['Daily Count'], inplace=True)

# Scale target per grid
scaler_dict = {}
df['Scaled_Count'] = 0.0
for grid in df['grid'].unique():
    scaler = StandardScaler()
    mask = df['grid'] == grid
    df.loc[mask, 'Scaled_Count'] = scaler.fit_transform(df.loc[mask, ['Daily Count']]).flatten()
    scaler_dict[grid] = scaler
# Global scaler for fallback
global_scaler = StandardScaler().fit(df[['Daily Count']])

# Prepare X, y, groups
X = df[selected_features]
y = df['Scaled_Count']
groups = df['grid']

logo = LeaveOneGroupOut()

# ------------------------
# Feature Selection: RFE
# ------------------------
print("🔍 Running RFE...")

# Base estimator
rf_base = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

# RFE setup (e.g., select top 30 features)
rfe = RFE(estimator=rf_base, n_features_to_select=30, step=1, verbose=1)
rfe.fit(X, y)

# Get selected features
selected_rfe = X.columns[rfe.support_].tolist()
print(f"✅ RFE selected {len(selected_rfe)} features")

print("Selected features:", selected_rfe)

# Subset data
X_sel = X[selected_rfe]

# ---------------------------------
# 4. Hyperparameter Tuning: RandomizedSearchCV
# ---------------------------------
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]
}
search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=20,
    scoring=mae_scorer,
    cv=logo.split(X_sel, y, groups),
    verbose=2,
    n_jobs=-1,
    random_state=42
)
print("🔍 Running hyperparameter tuning...")
search.fit(X_sel, y)
print("Best params:", search.best_params_)
print("Best MAE (neg):", search.best_score_)
# Set best estimator
best_rf = clone(search.best_estimator_)

# ---------------------------------
# 5. Retrain and Evaluate
# ---------------------------------
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2*np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)+1e-10))

# SMAPE and MAE per grid
grid_metrics = []
for train, test in tqdm(logo.split(X_sel, y, groups), total=len(groups.unique())):
    grid_id = groups.iloc[test].iat[0]
    model = clone(best_rf)
    model.fit(X_sel.iloc[train], y.iloc[train])
    pred_scaled = model.predict(X_sel.iloc[test])
    scaler = scaler_dict.get(grid_id, global_scaler)
    preds = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    true = df.iloc[test]['Daily Count'].values
    smape_val = smape(true, preds)
    mae_val = mean_absolute_error(true, preds)
    grid_metrics.append({'Grid': grid_id, 'SMAPE': smape_val, 'MAE': mae_val})

# Save grid-specific metrics
grid_metrics_df = pd.DataFrame(grid_metrics)
grid_metrics_df.to_csv(os.path.join(output_dir, "grid_metrics.csv"), index=False)

for metric in grid_metrics:
    print(f"Grid {metric['Grid']} — SMAPE: {metric['SMAPE']:.2f}%, MAE: {metric['MAE']:.2f}")

# Global metrics
y_preds, y_trues = [], []
for tr, te in tqdm(logo.split(X_sel, y, groups), total=len(groups.unique())):
    m = clone(best_rf)
    m.fit(X_sel.iloc[tr], y.iloc[tr])
    ps = m.predict(X_sel.iloc[te])
    sc = scaler_dict.get(groups.iloc[te].iat[0], global_scaler)
    ps_inv = sc.inverse_transform(ps.reshape(-1,1)).flatten()
    y_preds.extend(ps_inv)
    y_trues.extend(df.iloc[te]['Daily Count'].values)

print("✅ Final Model — SMAPE: {:.2f}%, MAE: {:.2f}".format(
    smape(np.array(y_trues), np.array(y_preds)),
    mean_absolute_error(y_trues, y_preds)
))

# Save metrics
with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write(f"SMAPE: {smape(np.array(y_trues), np.array(y_preds)):.2f}%\n")
    f.write(f"MAE: {mean_absolute_error(y_trues, y_preds):.2f}\n")

# Save predictions
pd.DataFrame({'True': y_trues, 'Predicted': y_preds}).to_csv(
    os.path.join(output_dir, "predictions.csv"), index=False
)

# ---------------------------------
# 6. Diagnostics & Insights
# ---------------------------------
# Feature importance
final_model = clone(best_rf)
final_model.fit(X_sel, y)
fi = pd.DataFrame({'Feature': X_sel.columns, 'Importance': final_model.feature_importances_})
fi = fi.sort_values('Importance', ascending=False)
print("Top 10 Features by Importance:\n", fi.head(10))

# Save feature importance
fi.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

# Residuals vs True
residuals = np.array(y_trues) - np.array(y_preds)
plt.figure(figsize=(10,6))
plt.scatter(y_trues, residuals, alpha=0.5)
plt.axhline(0, linestyle='--', color='red')
plt.xlabel('True Daily Count')
plt.ylabel('Residuals')
plt.title('Residuals vs True Values')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residuals_vs_true.png"), dpi=300)

# Permutation importance
perm = permutation_importance(final_model, X_sel, y, n_repeats=10, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({'Feature': X_sel.columns, 'Importance': perm.importances_mean}).sort_values('Importance', ascending=False)
print("Top 10 Permutation Importances:\n", perm_df.head(10))

# Save permutation importances
perm_df.to_csv(os.path.join(output_dir, "permutation_importance.csv"), index=False)

# Partial dependence for top 3
top3 = perm_df['Feature'].head(3).tolist()
PartialDependenceDisplay.from_estimator(final_model, X_sel, top3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "partial_dependence.png"), dpi=300)
