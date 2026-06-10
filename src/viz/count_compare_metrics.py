import pandas as pd
import matplotlib.pyplot as plt

# 1) Read your CSV
df = pd.read_csv('count_model_metrics.csv')

# 2) Drop the LinearRegression outlier (SMAPE ≈ 165) so we only see the ~22–24% models
df = df[df['SMAPE'] < 100]

# 3) (Optional) If you want a specific order, you can do:
desired_order = ['RandomForest', 'GBR', 'XGBoost']
df = df.set_index('Model').loc[desired_order].reset_index()

# 4) Extract for plotting
models = df['Model'].tolist()
smape  = df['SMAPE'].tolist()
mae    = df['MAE'].tolist()
r2     = df['R2'].tolist()

# 5) Define the 4 “brand” colors and only use as many as you need
palette = ['#b73779', '#f8765c', '#febb81' ]
colors  = palette[:len(models)]

# 6) Make the 3‑panel figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# SMAPE panel
axes[0].bar(models, smape, color=colors)
axes[0].set_title('SMAPE Comparison', fontsize=16, fontweight='bold')
axes[0].set_ylabel('SMAPE (%)', fontsize=14)
axes[0].tick_params(axis='x', rotation=30)

# MAE panel
axes[1].bar(models, mae, color=colors)
axes[1].set_title('MAE Comparison', fontsize=16, fontweight='bold')
axes[1].set_ylabel('MAE', fontsize=14)
axes[1].tick_params(axis='x', rotation=30)

# R² panel
axes[2].bar(models, r2, color=colors)
axes[2].set_title(r'$R^2$ Score Comparison', fontsize=16, fontweight='bold')
axes[2].set_ylabel(r'$R^2$', fontsize=14)
axes[2].tick_params(axis='x', rotation=30)

# 7) Tidy up
plt.tight_layout()
plt.show()




# 1) Load your data
df = pd.read_csv('accident_model_metrics.csv')

# 2) (Optional) filter out any extreme SMAPE values, e.g. >100%
#df = df[df['SMAPE'] < 100]

# 3) (Optional) enforce a particular model order
order = ['RandomForest', 'GBR', 'XGBoost', 'LinearRegression']
df = df.set_index('Model').loc[order].reset_index()

# 4) Extract columns for plotting
models = df['Model'].tolist()
smape  = df['SMAPE'].tolist()
mae    = df['MAE'].tolist()
r2     = df['R2'].tolist()

# 5) Define the four “brand” colors
palette = ['#26828e', '#35b779', '#b5de2b' ]
colors  = palette[:len(models)]

# 6) Build a 1×3 figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# SMAPE
axes[0].bar(models, smape, color=colors)
axes[0].set_title('SMAPE Comparison', fontsize=16, fontweight='bold')
axes[0].set_ylabel('SMAPE (%)', fontsize=14)
axes[0].tick_params(axis='x', rotation=30)

# MAE
axes[1].bar(models, mae, color=colors)
axes[1].set_title('MAE Comparison', fontsize=16, fontweight='bold')
axes[1].set_ylabel('MAE', fontsize=14)
axes[1].tick_params(axis='x', rotation=30)

# R²
axes[2].bar(models, r2, color=colors)
axes[2].set_title(r'$R^2$ Score Comparison', fontsize=16, fontweight='bold')
axes[2].set_ylabel(r'$R^2$', fontsize=14)
axes[2].tick_params(axis='x', rotation=30)

# 7) Final layout
plt.tight_layout()
plt.show()




import pandas as pd
import matplotlib.pyplot as plt

# Read the predictions file
df = pd.read_csv('rfecv_tuned_predictions.csv')

# Define colors to match the bar chart style
scatter_color = '#b73779'   # Peach color for scatter
line_color = 'black'      # Rose color for the perfect line

# Scatter plot: True vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(df['True'], df['Predicted'], color=scatter_color, alpha=0.7, label='Predictions')
plt.plot([df['True'].min(), df['True'].max()], 
         [df['True'].min(), df['True'].max()], 
         color=line_color, linestyle='--', label='Perfect Prediction')
plt.xlabel('True Values', fontsize=22)
plt.ylabel('Predicted Values', fontsize=22)
plt.title('True vs Predicted Daily Counts', fontsize=22, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('rf_results/grid_metrics.csv')

# Sort for SMAPE and MAE
df_smape = df.sort_values('SMAPE', ascending=False)
df_mae = df.sort_values('MAE', ascending=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar plot for SMAPE
axes[0].bar(df_smape['Grid'].astype(str), df_smape['SMAPE'], color='#b73779', edgecolor='black')
axes[0].set_title('SMAPE for Grids (Ordered by Magnitude)', fontweight='bold', fontsize= 20)
axes[0].set_xlabel('Grid', fontsize=14)
axes[0].set_ylabel('SMAPE (%)', fontsize=22)
axes[0].tick_params(axis='x', rotation=90, labelsize=12)  # Larger font size for x-axis
axes[0].tick_params(axis='y', labelsize=15)  # Larger font size for y-axis

# Mean line for SMAPE
mean_smape = df['SMAPE'].mean()
axes[0].axhline(mean_smape, color='black', linestyle='--', linewidth=2, label=f'Mean SMAPE: {mean_smape:.2f}')
axes[0].legend(fontsize=15)

# Bar plot for MAE
axes[1].bar(df_mae['Grid'].astype(str), df_mae['MAE'], color='#e95462', edgecolor='black')
axes[1].set_title('MAE for Grids (Ordered by Magnitude)', fontweight='bold', fontsize=20)
axes[1].set_xlabel('Grid', fontsize=14)
axes[1].set_ylabel('MAE', fontsize=22)
axes[1].tick_params(axis='x', rotation=90, labelsize=12)  # Larger font size for x-axis
axes[1].tick_params(axis='y', labelsize=15)  # Larger font size for y-axis

# Mean line for MAE
mean_mae = df['MAE'].mean()
axes[1].axhline(mean_mae, color='black', linestyle='--', linewidth=2, label=f'Mean MAE: {mean_mae:.2f}')
axes[1].legend(fontsize=15)

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt

# Sample grouped feature importance values
feature_groups = ['Weather', 'Temporal', 'Grid Location', 'Bike Activity', 'Infrastructure', 'Demographics', 'Land Use', 'Traffic']
importances = [0.39, 0.28, 0.12, 0.11, 0.05, 0.03, 0.01, 0.01]

# Configure figure and font sizes
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 18,  # Base font size
                     'axes.titlesize': 22,
                     'axes.labelsize': 20,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16})

# Create horizontal bar plot
y_positions = range(len(feature_groups))
plt.barh(y_positions, importances, color=plt.cm.viridis(importances))

# Labels and title
plt.yticks(y_positions, feature_groups)
plt.xlabel('Importance', fontsize=20)
plt.title('Grouped Feature Importance (Random Forest)', fontsize=22)

# Invert y-axis to have highest importance on top
plt.gca().invert_yaxis()

# Layout adjustment and display
plt.tight_layout()
plt.show()






import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('rf_results/predictions.csv')
print

# Assuming the columns are named 'true' and 'pred' (adjust if different)
df['residuals'] = df['Predicted'] - df['True']

# Plot
plt.figure(figsize=(10,6))

# Customize color here
scatter_color = '#fea973'  # Change this to any color you like (e.g., 'green', '#FF5733')

plt.scatter(df['True'], df['residuals'], alpha=0.5, color=scatter_color, edgecolors='#fa7d5e')
plt.axhline(0, color='red', linestyle='--', linewidth=2)

plt.title('Residuals vs True Values', fontsize=22, fontweight='bold')
plt.xlabel('True Daily Count', fontsize= 20)
plt.ylabel('Residuals',fontsize= 20)
plt.show()
