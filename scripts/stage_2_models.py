# --- Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import geopandas as gpd
import os
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

# --- Setup ---
output_dir = "Cycling_Safety_Temporal_Holdout"
os.makedirs(output_dir, exist_ok=True)

# --- Load Data ---
df = pd.read_csv("full_daily_grid_level_with_accidents.csv", low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df['week'] = df['Date'].dt.isocalendar().week
df['year'] = df['Date'].dt.year

# --- SMAPE ---
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))

# --- Feature Groups ---
feature_groups = {
    "spatial_temporal": ["grid_mid_lat", "grid_mid_lon", "HECTARES", "day_of_week", "day_of_month", "month", "weekend", "season_spring", "season_summer", "is_holiday"],
    "meteorological": ["tavg", "tmin", "tmax", "wdir", "wspd", "wpgt", "pres"],
    "proximity_traffic": ["distance_to_center_m", "max_speed"],
    "amenities_250m": ["shops_250m", "education_250m", "hospitals_250m", "hotels_250m"],
    "demographics": ["2023", "Aged 0-15_2013", "Aged 16-64_2013", "Aged 65+_2013", "pop_dens_2023"],
    "land_use_250m": ["landuse_residential_pct_250m", "landuse_commercial_pct_250m", "landuse_industrial_pct_250m", "landuse_retail_pct_250m", "landuse_park_pct_250m", "landuse_forest_pct_250m", "landuse_meadow_pct_250m", "landuse_farmland_pct_250m", "landuse_grass_pct_250m", "landuse_cemetery_pct_250m", "landuse_allotments_pct_250m", "landuse_recreation_ground_pct_250m", "landuse_water_pct_250m", "landuse_wood_pct_250m", "landuse_construction_pct_250m"],
    "infrastructure_250m": ["pt_infra_count_250m", "bus_stop_count_250m", "railway_stop_count_250m", "traffic_signals_250m", "crossings_250m", "intersections_250m"],
    "cycle_infra_250m": ["bike_lane_length_250m", "bike_lane_density_250m", "cycleway_track_length_250m", "cycleway_track_density_250m", "cycleway_lane_length_250m", "cycleway_lane_density_250m", "cycleway_opposite_length_250m", "cycleway_opposite_density_250m", "cycleway_dedicated_length_250m", "cycleway_dedicated_density_250m", "bike_parking_spots_250m"]
}

selected_features = sum(feature_groups.values(), [])
for c in selected_features + ['Predicted Daily Count']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# --- Weekly Aggregation ---
agg_dict = {feat: 'mean' for feat in selected_features}
agg_dict['accident'] = 'sum'
weekly_df = df.groupby(['grid', 'year', 'week']).agg(agg_dict).reset_index()
weekly_df['sum_predicted_weekly_count'] = df.groupby(['grid', 'year', 'week'])['Predicted Daily Count'].sum().values
weekly_df['accidents_per_1000_bikes'] = 1000 * weekly_df['accident'] / weekly_df['sum_predicted_weekly_count']
weekly_df['accidents_per_1000_bikes'].replace([np.inf, -np.inf], 0, inplace=True)
weekly_df['accidents_per_1000_bikes'].fillna(0, inplace=True)
weekly_df['accident_occurrence'] = (weekly_df['accident'] > 0).astype(int)
weekly_df = weekly_df[~weekly_df['grid'].isin([138, 355])]

# --- Model Setup ---
X = weekly_df[selected_features]
y_clf = weekly_df['accident_occurrence']
y_reg = weekly_df['accidents_per_1000_bikes']

splits = [{"train": (1, 23), "test": (27, 52)}, {"train": (1, 26), "test": (27, 52)}]

param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}

# --- Temporal Splits Loop ---
for idx, split in enumerate(splits, 1):
    print(f"\n=== Temporal Split {idx}: Train Weeks {split['train']} | Test Weeks {split['test']} ===")

    train_mask = (weekly_df['week'] >= split['train'][0]) & (weekly_df['week'] <= split['train'][1])
    test_mask = (weekly_df['week'] >= split['test'][0]) & (weekly_df['week'] <= split['test'][1])

    X_train, X_test = X[train_mask], X[test_mask]
    y_clf_train, y_clf_test = y_clf[train_mask], y_clf[test_mask]
    y_reg_train, y_reg_test = y_reg[train_mask], y_reg[test_mask]

    imp = SimpleImputer(strategy='mean')
    X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=selected_features)
    X_test_imp = pd.DataFrame(imp.transform(X_test), columns=selected_features)

    # --- Classification ---
    clf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, scoring='f1', cv=3, n_jobs=-1)
    clf_grid.fit(X_train_imp, y_clf_train)
    clf_best = clf_grid.best_estimator_
    y_clf_pred = clf_best.predict(X_test_imp)

    # --- Confusion Matrix ---
    conf_matrix = confusion_matrix(y_clf_test, y_clf_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=['No Accident', 'Accident'],
        yticklabels=['No Accident', 'Accident'],
        annot_kws={"size": 22}  # Increase font size for numbers inside the matrix
    )
    plt.xlabel('Predicted Label', fontsize=22)
    plt.ylabel('True Label', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.title(f'Confusion Matrix - Split {idx}', fontsize=22, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_split_{idx}.png", dpi=300)
    plt.close()

    # --- Regression ---
    reg_grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1)
    reg_grid.fit(X_train_imp, y_reg_train)
    reg_best = reg_grid.best_estimator_
    y_reg_pred = reg_best.predict(X_test_imp)

    # Save predictions for Split 2
    if idx == 2:
        weekly_df.loc[test_mask, 'predicted_accidents_per_1000_bikes_split2'] = y_reg_pred

    # --- True vs Predicted Plot ---
    # Calculate absolute error for coloring
    errors = np.abs(y_reg_test - y_reg_pred)

    plt.figure(figsize=(10,8))
    sc = plt.scatter(y_reg_test, y_reg_pred, c=errors, cmap='viridis', alpha=0.8, edgecolors='k', linewidths=0.5)
    plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2, label="Perfect Prediction")

    plt.xlabel('True Accident Risk', fontsize=18)
    plt.ylabel('Predicted Accident Risk', fontsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f'True vs. Predicted Accident Risk (for 1000 bike rides)- Split {idx}', fontsize=20, fontweight="bold")

    cbar = plt.colorbar(sc)
    cbar.set_label('Absolute Error', fontsize=16)

    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/true_vs_predicted_regression_split_{idx}.png", dpi=300)
    plt.close()


# --- Map Comparison for Split 2 ---
grid_gdf = gpd.read_file("shapefiles/grid_cells_enriched.geojson")
viz_df = grid_gdf.merge(weekly_df, on="grid", how="left").dropna(subset=['week'])

vmin = min(viz_df['accidents_per_1000_bikes'].min(), viz_df['predicted_accidents_per_1000_bikes_split2'].min())
vmax = max(viz_df['accidents_per_1000_bikes'].max(), viz_df['predicted_accidents_per_1000_bikes_split2'].max())

for week in sorted(viz_df['week'].dropna().unique()):
    if week < 27:
        continue

    week_df = viz_df[viz_df['week'] == week]
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    week_df.plot(column='accidents_per_1000_bikes', cmap='YlGnBu', legend=True, ax=ax[0], vmin=vmin, vmax=vmax,
                 legend_kwds={'label': "Accidents per 1000 Bikes", 'shrink': 0.7})
    ax[0].set_title(f"Actual Risk - Week {week}", fontsize=22, fontweight='bold')
    ax[0].axis('off')

    week_df.plot(column='predicted_accidents_per_1000_bikes_split2', cmap='YlGnBu', legend=True, ax=ax[1], vmin=vmin, vmax=vmax,
                 legend_kwds={'label': "Predicted Accidents per 1000 Bikes", 'shrink': 0.7})
    ax[1].set_title(f"Predicted Risk (Split 2) - Week {week}", fontsize=22, fontweight='bold')
    ax[1].axis('off')

    plt.suptitle(f"Comparison of Actual vs. Predicted Accident Risk - Week {week}", fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/map_comparison_split2_week_{week}.png", dpi=300)
    plt.close()
