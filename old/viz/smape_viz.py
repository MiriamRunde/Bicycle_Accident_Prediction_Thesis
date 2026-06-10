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