# Wheels in Motion, Risk in Mind?

### A Two-Stage Machine-Learning Framework for Extrapolating Bike Volume and Exposure-Based Accident Risk Prediction

> **M.Sc. Data Science for Public Policy** · Hertie School · April 2025  
> **Supervisor**: Prof. Dr. Lynn Kaack  
> 📍 Study area: London Borough of Camden

---

## Abstract

Accurate estimates of cyclist exposure are essential for meaningful safety analysis, yet permanent count networks often lack the spatial and temporal coverage needed to normalize accident data. This thesis introduces a two-stage machine-learning framework to forecast bicycle volumes and predict exposure-adjusted accident risk in the London Borough of Camden.

In **Stage 1**, a Random Forest regressor is trained on 2022 daily counts from 28 camera-based counters and over 130 contextual, infrastructural, and meteorological features to predict daily bicycle volumes at **250 × 250 m resolution** — achieving a global SMAPE of 21% and explaining over 85% of variance in held-out locations.

In **Stage 2**, these predicted volumes are used as the normalising factor in a **weekly relative accident risk regression** model (R² = 0.84), and as an input to a **binary accident occurrence classifier** (F1 = 0.99 in temporal holdout). The model uncovers low-volume, high-risk hotspots that raw crash counts would obscure.

Finally, a log–log regression provides modest support for the **"safety in numbers"** hypothesis.

---

## The Two-Stage Pipeline

```
Stage 1: Cycling Volume Model
  Input:  weather + land use + cycling infrastructure + demographics + …
  Method: Random Forest regression, Leave-One-Grid-Out cross-validation
  Output: predicted daily cycling counts per 250×250m grid cell
                          ↓
Stage 2: Accident Risk Models
  Target A (Classification): accident occurrence this week? (0/1)   → F1 = 0.99
  Target B (Regression):     accidents per 1,000 predicted cyclists  → R² = 0.84
  Validation: temporal holdout (train on earlier weeks, test on later)
```

---

## Key Results

### Model Comparison — Stage 2 Regression (accidents per 1,000 bikes)

| Model | R² | SMAPE |
|---|---|---|
| **Random Forest** | **0.82** | **0.10%** |
| XGBoost | 0.81 | 199% |
| Gradient Boosting | 0.78 | 199% |
| Linear Regression | −0.11 | 200% |

> **Note on SMAPE:** The large gap in SMAPE between RF and other models is partly an artefact of the zero-inflated target distribution (see [Limitations](#limitations)). R² is the more reliable comparison metric here.

### Temporal Holdout Evaluation — Stage 2

| Task | F1 | Precision | Recall | R² | Non-zero SMAPE |
|---|---|---|---|---|---|
| Classification (split 2) | **0.99** | 0.99 | 0.99 | — | — |
| Regression (split 2) | — | — | — | **0.84** | 24% |
| Regression (split 3) | — | — | — | **0.86** | 27% |

### Stage 1 — Cycling Volume Forecast

| Metric | Value |
|---|---|
| Global SMAPE | 21.6% |
| Mean SMAPE per grid cell | 10.4% |
| Variance explained (held-out) | >85% |
| Count stations used for training | 28 |
| Coverage of permanent counters | ~10% of area |

---

## Selected Visualisations

### Weekly Risk Forecast Maps (Predicted vs. Actual, Split 2)

<p align="center">
  <img src="plots/stage2_safety/temporal_holdout_maps/map_comparison_split2_week_36.png" width="700" alt="Map comparison week 36">
</p>

### SHAP Feature Importance — Classification & Regression

<p float="left">
  <img src="plots/stage2_safety/shap/shap_summary_classification_dot_full.png" width="48%" alt="SHAP classification dot plot">
  <img src="plots/stage2_safety/shap/shap_summary_regression_dot_full.png" width="48%" alt="SHAP regression dot plot">
</p>

### Permutation Importance

<p float="left">
  <img src="plots/stage2_safety/feature_importance/permutation_importance_classification.png" width="48%">
  <img src="plots/stage2_safety/feature_importance/permutation_importance_regression.png" width="48%">
</p>

### Baseline Model Comparison

<p align="center">
  <img src="plots/stage2_safety/baseline_model_comparison_custom_colors.png" width="650" alt="Baseline model comparison">
</p>

### Partial Dependence Plot

<p align="center">
  <img src="plots/stage2_safety/partial_dependence.png" width="650" alt="Partial dependence plot">
</p>

---

## Feature Groups

Features are engineered at **250 m** and **500 m** buffer radii around each grid cell:

| Group | Examples |
|---|---|
| Spatial / Temporal | Grid centroid lat/lon, week of year, day of week, season, public holidays |
| Meteorological | Avg/min/max temperature, wind speed/direction, pressure |
| Proximity & Traffic | Distance to city centre, road speed limit |
| Amenities | Shops, schools, hospitals, hotels |
| Demographics | Population density, age bands (2013 & 2023) |
| Land Use | % residential, commercial, retail, park, forest, water … |
| Public Transport Infra | Bus stops, rail stops, traffic signals, crossings, intersections |
| Cycling Infrastructure | Bike lane length/density, dedicated cycleways, bike parking |

---

## Repository Structure

```
thesis_clean/
│
├── src/
│   ├── models/
│   │   ├── RF_SG_holdout_RFECV.py      # Stage 1: volume model (LOGO-CV + RFECV)
│   │   ├── RF_SG_holdout_RFE.py        # Stage 1: volume model (RFE variant)
│   │   ├── stage_2_models.py           # Stage 2: safety classification + regression
│   │   ├── rf_model.py                 # Stage 2: relative risk RF (temporal splits)
│   │   ├── compare_ML_models_w.py      # Baseline model comparison
│   │   ├── compare_risk.py             # Risk comparison analysis
│   │   ├── check_safety_in_numbers.py  # Safety-in-numbers OLS analysis
│   │   ├── feature_permuation.py       # Permutation importance
│   │   └── model_comparison.py         # Model comparison utilities
│   └── viz/
│       ├── EDA.py
│       ├── create_base_maps.py
│       ├── map_counting_stations.py
│       ├── feat_imp.py
│       ├── smape_viz.py
│       └── count_compare_metrics.py
│
├── data/
│   └── shapefiles/                     # Camden boundary + enriched grid GeoJSON
│
├── plots/
│   ├── stage1_volume/
│   │   ├── forecast_maps/              # Weekly volume forecast maps
│   │   └── split_predictions/          # Prediction scatter plots per split
│   └── stage2_safety/
│       ├── classification/             # Confusion matrices, ROC/PR curves
│       ├── regression/                 # Scatter plots, residuals
│       ├── shap/                       # SHAP summary plots
│       ├── feature_importance/         # Permutation importance plots
│       ├── temporal_holdout_maps/      # Side-by-side predicted vs actual maps
│       ├── baseline_model_comparison_custom_colors.png
│       └── partial_dependence.png
│
├── results/
│   ├── stage1/                         # Volume model metrics + predictions CSV
│   └── stage2/
│       ├── temporal_holdout/           # Metrics TXTs + hyperparameter logs
│       ├── accident_model_metrics.csv
│       └── Grid_SMAPE_Results.csv
│
├── docs/                               # GitHub Pages portfolio site
│   ├── index.html
│   └── images/                         # Curated plots for the website
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Stage 1 — cycling volume model (spatial generalisation with LOGO-CV)
python src/models/RF_SG_holdout_RFECV.py

# Stage 2 — accident risk models (temporal holdout)
python src/models/stage_2_models.py

# Baseline model comparison
python src/models/compare_ML_models_w.py

# Visualisations
python src/viz/EDA.py
python src/viz/create_base_maps.py
```

> **Data note**: The full daily grid-level dataset (`full_daily_grid_level_with_accidents.csv`) is not tracked in git due to its size. Contact the author for access.

---

## Limitations

Acknowledging these openly is part of honest science:

**Stage 2 — Classification F1 = 0.99 reflects spatial memorisation more than temporal prediction.** Static features (land use, infrastructure, demographics) do not change week to week, so the model largely learns *which grid cells are structurally prone to accidents* rather than *when* accidents will occur. A more demanding evaluation would hold out entire geographic areas unseen during training, not just future time periods. The result is still useful for planning (identifying high-risk locations), but should not be interpreted as a dynamic weekly forecast.

**Stage 1 — Per-grid scaling is fitted before cross-validation.** The StandardScaler for each counting station is fitted on all of that station's data before Leave-One-Grid-Out splits. This slightly optimises the evaluation: in true deployment, no historical volume data would exist for a new location, requiring the global scaler fallback. The practical impact on SMAPE is small but the reported performance is marginally optimistic.

**SMAPE is unstable on zero-inflated targets.** When both true and predicted values are near zero, SMAPE collapses to ~0, artificially deflating the global metric. The non-zero SMAPE (~24–27%) is the more meaningful performance figure for the Stage 2 regression model.

---

## Policy Relevance

The framework shows that reliable volume estimates can be produced even when permanent count data cover just **10% of an area**, using open data on weather, land use, and demographics. The resulting exposure-adjusted risk maps support targeted interventions aligned with **Vision Zero** goals.

---

## Portfolio Site

👉 **[View the portfolio](https://miriamrunde.github.io/thesis_clean/)**

To activate GitHub Pages: `Settings → Pages → Source: Deploy from a branch → Branch: main, /docs`

---

## Author

**Miriam Runde** · [miriam.runde@googlemail.com](mailto:miriam.runde@googlemail.com)  
M.Sc. Data Science for Public Policy · Hertie School · 2025
