import matplotlib.pyplot as plt

# Sample grouped feature importance values
feature_groups = ['Weather', 'Temporal', 'Grid Location', 'Bike Activity', 'Infrastructure', 'Demographics', 'Land Use', 'Traffic']
importances = [0.39, 0.28, 0.12, 0.11, 0.05, 0.03, 0.01, 0.01]

# Configure figure and font sizes
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 20,  # Base font size
                     'axes.titlesize': 22,
                     'axes.labelsize': 20,
                     'xtick.labelsize': 18,
                     'ytick.labelsize': 18})

# Create horizontal bar plot with original plasma colormap
y_positions = range(len(feature_groups))
plt.barh(y_positions, importances, color=plt.cm.plasma(importances))

# Labels and title
plt.yticks(y_positions, feature_groups)
plt.xlabel('Importance', fontsize=20)
plt.title('Grouped Feature Importance (Random Forest)', fontsize=22, fontweight='bold')

# Invert y-axis to have highest importance on top
plt.gca().invert_yaxis()

# Layout adjustment and display
plt.tight_layout()
plt.show()
