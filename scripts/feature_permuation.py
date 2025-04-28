import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import warnings
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

# --- 1. Load and preprocess data ---
df = pd.read_csv("full_daily_grid_level_with_accidents.csv", low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df['week'] = df['Date'].dt.isocalendar().week
df['year'] = df['Date'].dt.year

selected_features = [col for col in df.columns if col not in ['Date', 'grid', 'year', 'week', 'accident', 'Predicted Daily Count']]
for c in selected_features + ['Predicted Daily Count']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

agg_dict = {feat: 'mean' for feat in selected_features}
agg_dict['accident'] = 'sum'
weekly_df = df.groupby(['grid', 'year', 'week']).agg(agg_dict).reset_index()
weekly_df['sum_predicted_daily_count'] = df.groupby(['grid', 'year', 'week'])['Predicted Daily Count'].sum().values
weekly_df['relative_risk_pct'] = 100 * weekly_df['accident'] / weekly_df['sum_predicted_daily_count']
weekly_df['relative_risk_pct'].replace([np.inf, -np.inf], 0, inplace=True)
weekly_df['relative_risk_pct'].fillna(0, inplace=True)
weekly_df['accident_occurrence'] = (weekly_df['accident'] > 0).astype(int)
weekly_df = weekly_df[~weekly_df['grid'].isin([138, 355])]

# --- 2. Prepare features ---
clf_features = selected_features + ['sum_predicted_daily_count']
X_clf = weekly_df[[col for col in clf_features if col in weekly_df.columns]].dropna(axis=1, how='all')
X_reg = weekly_df[[col for col in selected_features if col in weekly_df.columns]].dropna(axis=1, how='all')

y_clf = weekly_df['accident_occurrence']
y_reg = weekly_df['relative_risk_pct']

# --- 3. Impute ---
imp_clf = SimpleImputer(strategy='mean')
X_clf_imp = pd.DataFrame(imp_clf.fit_transform(X_clf), columns=X_clf.columns, index=X_clf.index)

imp_reg = SimpleImputer(strategy='mean')
X_reg_imp = pd.DataFrame(imp_reg.fit_transform(X_reg), columns=X_reg.columns, index=X_reg.index)

# --- 4. Train models ---
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_clf_imp, y_clf)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_reg_imp, y_reg)

# --- 5. Baseline metrics ---
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))

baseline_f1 = f1_score(y_clf, clf_model.predict(X_clf_imp))
baseline_smape = smape(y_reg, reg_model.predict(X_reg_imp))
print(f"Baseline F1: {baseline_f1:.4f}")
print(f"Baseline SMAPE: {baseline_smape:.2f}")

# --- 6. Group-based permutation importance ---
# Define groups
groups = {
    'Weather': [c for c in X_clf.columns if c in ['tavg', 'tmin', 'tmax', 'wdir', 'wspd', 'wpgt', 'pres']],
    'Landuse': [c for c in X_clf.columns if 'landuse_' in c],
    'Infrastructure': [c for c in X_clf.columns if 'bike_lane' in c or 'cycleway' in c or 'shops_' in c or 'intersections' in c],
    'Demographics': [c for c in X_clf.columns if 'pop_dens' in c or 'income' in c or 'Aged' in c],
    'Borough': [c for c in X_clf.columns if 'Borough_' in c],
    'Volume': ['sum_predicted_daily_count'] if 'sum_predicted_daily_count' in X_clf.columns else []
}

# Initialize
results_clf = {}
results_reg = {}

for group, cols in groups.items():
    if not cols:
        continue

    # --- Classification ---
    X_clf_perm = X_clf_imp.copy()
    X_clf_perm[cols] = np.random.permutation(X_clf_perm[cols].values)
    perm_f1 = f1_score(y_clf, clf_model.predict(X_clf_perm))
    results_clf[group] = baseline_f1 - perm_f1

    # --- Regression ---
    X_reg_perm = X_reg_imp.copy()
    common_cols = [c for c in cols if c in X_reg_perm.columns]
    if common_cols:
        X_reg_perm[common_cols] = np.random.permutation(X_reg_perm[common_cols].values)
        perm_smape = smape(y_reg, reg_model.predict(X_reg_perm))
        results_reg[group] = perm_smape - baseline_smape

# --- 7. Plot ---
output_dir = "Cycling_Safety_Permutation_Importance"
os.makedirs(output_dir, exist_ok=True)

# Classification plot
plt.barh(list(results_clf.keys()), list(results_clf.values()))
plt.xlabel("F1 Score Drop")
plt.title("Group-based Permutation Importance (Classification)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "permutation_importance_classification.png"))
plt.close()

# Regression plot
plt.barh(list(results_reg.keys()), list(results_reg.values()))
plt.xlabel("SMAPE Increase")
plt.title("Group-based Permutation Importance (Regression)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "permutation_importance_regression.png"))
plt.close()
