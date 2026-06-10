import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
from tqdm import tqdm
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

output_dir = "Acc_RF_model_plots"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------
# Define feature groups and selection
# -----------------------------------------
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

selected_features = (
    feature_groups['meteorological'] +
    feature_groups['proximity_traffic'] +
    feature_groups['amenities_250m'] +
    #feature_groups['amenities_500m'] +
    feature_groups['demographics'] +
    feature_groups['land_use_250m'] +
    feature_groups['infrastructure_250m'] +
    #feature_groups['infrastructure_500m'] +
    feature_groups['cycle_infra_250m']+ 
    #feature_groups['cycle_infra_500m'] +
    feature_groups['spatial_temporal']
)


# -----------------------------------------
# Load and preprocess data (weekly level)
# -----------------------------------------
df = pd.read_csv("full_daily_grid_level_with_accidents.csv", low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df['week'] = df['Date'].dt.isocalendar().week
df['year'] = df['Date'].dt.year

agg_dict = {feat: 'mean' for feat in selected_features}
agg_dict['accident'] = 'sum'

# Aggregate feature values and accidents
weekly_df = df.groupby(['grid', 'year', 'week']).agg(agg_dict).reset_index()

# Sum of predicted daily counts per week (exposure)
df_pred = df.groupby(['grid', 'year', 'week'])['Predicted Daily Count'].sum().reset_index(name='sum_predicted_weekly_count')

# Merge sum of predictions into weekly_df
weekly_df = weekly_df.merge(df_pred, on=['grid', 'year', 'week'], how='left')

# Compute relative risk (using summed predicted counts!)
weekly_df['relative_risk_pct'] = 100 * weekly_df['accident'] / weekly_df['sum_predicted_weekly_count']

# Clean up infinities or NaNs
weekly_df['relative_risk_pct'].replace([np.inf, -np.inf], 0, inplace=True)
weekly_df['relative_risk_pct'].fillna(0, inplace=True)

# Drop problematic grids
weekly_df = weekly_df[~weekly_df['grid'].isin([138, 355])]

print(weekly_df)
plt.figure(figsize=(8, 5))
weekly_df['relative_risk_pct'].hist(bins=30)
plt.title("Distribution of Weekly Relative Risk (%)")
plt.xlabel("Relative Risk (%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# -----------------------------------------
# Train-test split setup
# -----------------------------------------
df_numeric = weekly_df[selected_features].apply(pd.to_numeric, errors='coerce')

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))

def evaluate(model, X, y, name, threshold=0.5):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    smape_val = smape(y, preds)
    
    # Binarize predictions and true values for classification metrics
    y_binary = (y > threshold).astype(int)
    preds_binary = (preds > threshold).astype(int)
    f1 = f1_score(y_binary, preds_binary)
    precision = precision_score(y_binary, preds_binary, zero_division=0)
    recall = recall_score(y_binary, preds_binary, zero_division=0)

    non_zero_mask = y > 0
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "smape": smape_val,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

    if non_zero_mask.sum() > 0:
        y_nz = y[non_zero_mask]
        preds_nz = preds[non_zero_mask]
        metrics.update({
            "mae_nz": mean_absolute_error(y_nz, preds_nz),
            "rmse_nz": np.sqrt(mean_squared_error(y_nz, preds_nz)),
            "r2_nz": r2_score(y_nz, preds_nz),
            "smape_nz": smape(y_nz, preds_nz)
        })

    print(f"{name}:\n"
          f"  All:     MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}, "
          f"SMAPE={metrics['smape']:.2f}%, F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")

    if 'mae_nz' in metrics:
        print(f"  Non-Zero:MAE={metrics['mae_nz']:.3f}, RMSE={metrics['rmse_nz']:.3f}, R2={metrics['r2_nz']:.3f}, "
              f"SMAPE={metrics['smape_nz']:.2f}%")
    else:
        print("  Non-Zero: No non-zero values in test set.")

    return preds, metrics

def plot_predictions(y_true, y_pred, title, filename, metrics):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, label="Data Points")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label="Ideal Fit")
    plt.xlabel("True Relative Risk (%)")
    plt.ylabel("Predicted")
    plt.title(title)

    # Add metrics as text on the plot
    metrics_text = (
        f"MAE={metrics['mae']:.3f}\n"
        f"RMSE={metrics['rmse']:.3f}\n"
        f"R²={metrics['r2']:.3f}\n"
        f"SMAPE={metrics['smape']:.2f}%\n"
        f"F1={metrics['f1']:.3f}\n"
        f"Precision={metrics['precision']:.3f}\n"
        f"Recall={metrics['recall']:.3f}"
    )

    if 'mae_nz' in metrics:
        metrics_text += (
            f"\nNon-Zero:\n"
            f"MAE={metrics['mae_nz']:.3f}\n"
            f"RMSE={metrics['rmse_nz']:.3f}\n"
            f"R²={metrics['r2_nz']:.3f}\n"
            f"SMAPE={metrics['smape_nz']:.2f}%"
        )

    plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

splits = [
    {"train_weeks": (1, 26), "test_weeks": (27, 52)},
    {"train_weeks": (1, 20), "test_weeks": (21, 40)},
    {"train_weeks": (13, 39), "test_weeks": (40, 52)},
    {"train_weeks": (1, 30), "test_weeks": (31, 52)}, 
    {"train_weeks": (1, 40), "test_weeks": (41, 52)}
]

for split_idx, split in enumerate(splits, start=1):
    print(f"\n=== Split {split_idx}: Train Weeks {split['train_weeks']} | Test Weeks {split['test_weeks']} ===")
    train_weeks = (weekly_df['week'] >= split["train_weeks"][0]) & (weekly_df['week'] <= split["train_weeks"][1])
    test_weeks = (weekly_df['week'] >= split["test_weeks"][0]) & (weekly_df['week'] <= split["test_weeks"][1])
    X_train = df_numeric[train_weeks]
    y_train = weekly_df.loc[train_weeks, 'relative_risk_pct']
    X_test = df_numeric[test_weeks]
    y_test = weekly_df.loc[test_weeks, 'relative_risk_pct']

    imp = SimpleImputer(strategy='mean')
    X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=selected_features)
    X_test_imp = pd.DataFrame(imp.transform(X_test), columns=selected_features)

    print("\nTraining and evaluating default Random Forest...")
    default_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    default_rf.fit(X_train_imp, y_train)
    default_preds, metrics = evaluate(default_rf, X_test_imp, y_test, f"Default RF (Split {split_idx})")

    plot_predictions(
        y_test,
        default_preds,
        f"Default RF: True vs Predicted (Split {split_idx})",
        f"default_rf_predictions_split_{split_idx}.png",
        metrics
    )
