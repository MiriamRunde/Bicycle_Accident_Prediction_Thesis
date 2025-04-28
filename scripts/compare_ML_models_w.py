import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
from tqdm import tqdm

from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

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

selected_features = sum(feature_groups.values(), [])

# -----------------------------------------
# Load and preprocess data (weekly level)
# -----------------------------------------
df = pd.read_csv("full_daily_grid_level_with_accidents.csv", low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df['week'] = df['Date'].dt.isocalendar().week
df['year'] = df['Date'].dt.year

agg_dict = {feat: 'mean' for feat in selected_features}
agg_dict['accident'] = 'sum'
weekly_df = df.groupby(['grid', 'year', 'week']).agg(agg_dict).reset_index()
df_pred = df.groupby(['grid', 'year', 'week'])['Predicted Daily Count'].mean().reset_index(name='avg_predicted_daily_count')
weekly_df = weekly_df.merge(df_pred, on=['grid', 'year', 'week'], how='left')
weekly_df['relative_risk_pct'] = 100 * weekly_df['accident'] / (weekly_df['avg_predicted_daily_count'] * 7)
weekly_df['relative_risk_pct'].replace([np.inf, -np.inf], 0, inplace=True)
weekly_df['relative_risk_pct'].fillna(0, inplace=True)
weekly_df = weekly_df[~weekly_df['grid'].isin([138, 355])]

weekly_df[selected_features] = weekly_df[selected_features].apply(pd.to_numeric, errors='coerce')
X = weekly_df[selected_features]
y = weekly_df['relative_risk_pct']

# Time-based train-test split
early_weeks = weekly_df['week'] <= 40
late_weeks = weekly_df['week'] > 40

X_train = X[early_weeks]
y_train = y[early_weeks]
X_test = X[late_weeks]
y_test = y[late_weeks]

imp = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(imp.fit_transform(X_train), columns=selected_features)
X_test = pd.DataFrame(imp.transform(X_test), columns=selected_features)

logo = LeaveOneGroupOut()

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))

def evaluate(name, yt, yp):
    print(f"{name}: MAE={mean_absolute_error(yt, yp):.3f}, RMSE={np.sqrt(mean_squared_error(yt, yp)):.3f}, R2={r2_score(yt, yp):.3f}, SMAPE={smape(yt, yp):.2f}%")


from sklearn.model_selection import GridSearchCV


models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GBR': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0, random_state=42)
}


param_grids = {
    'RandomForest': {
        'n_estimators': [100],  # Keep fixed
        'max_depth': [5, 10, None]
    },
    'GBR': {
        'n_estimators': [100],
        'max_depth': [3, 5]
    },
    'XGBoost': {
        'n_estimators': [100],
        'max_depth': [3, 5]
    }
}

tuned_models = {}

for name, model in models.items():
    if name in param_grids:
        print(f"Tuning {name}...")
        grid = GridSearchCV(model, param_grids[name], cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        tuned_models[name] = grid.best_estimator_
        print(f"Best params for {name}: {grid.best_params_}")
    else:
        tuned_models[name] = model.fit(X_train, y_train)


groups = weekly_df[early_weeks]['grid'].values  # Use only training grids for LOGO

for name, model in tqdm(tuned_models.items(), desc="Training and Evaluating Models"):
    model.fit(X_train, y_train)

    preds_logo = []
    y_true_logo = []

    for train_idx, test_idx in logo.split(X_train, y_train, groups=groups):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        preds_logo.extend(y_pred)
        y_true_logo.extend(y_val)

    print(f"\n{name} - LOGO CV Evaluation:")
    evaluate(name, np.array(y_true_logo), np.array(preds_logo))


#vizualize error metrics for different models
def plot_error_metrics(metrics):
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.15
    index = np.arange(len(metrics))

    for i, (model_name, model_metrics) in enumerate(metrics.items()):
        ax.bar(index + i * bar_width, model_metrics['mae'], bar_width, label=model_name)

    ax.set_xlabel('Models')
    ax.set_ylabel('MAE')
    ax.set_title('Mean Absolute Error for Different Models')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(list(metrics.keys()))
    ax.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('error_metrics.png')