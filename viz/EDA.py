import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('features_with_counts.csv')

# Filter rows where 'SiteID' contains 'SG'
df_filtered = df[df['SiteID'].str.contains('SG', na=False)]
# Exclude SG 15, 26, 31, 32, 33
df_filtered = df_filtered[~df_filtered['SiteID'].isin(['SG015', 'SG020', 'SG031'])]

# Prepare data for boxplot grouped by 'SiteID'
groups = sorted(df_filtered['SiteID'].unique())
data = [df_filtered[df_filtered['SiteID'] == grp]['Daily Count'] for grp in groups]

# Create the boxplot
plt.figure(figsize=(14, 6))
box = plt.boxplot(data, labels=range(1, len(groups) + 1), patch_artist=True, showfliers=True)

# Apply colors from the 'magma' colormap
cmap = plt.cm.get_cmap('magma', len(groups))
for patch, color in zip(box['boxes'], cmap.colors):
    patch.set_facecolor(color)

# Final plot formatting
plt.xticks(rotation=90, fontsize=18)
plt.xlabel('Station Group (1 to 28)', fontsize=22)
plt.ylabel('Daily Count', fontsize=22)
plt.yticks(rotation=90, fontsize=18)
plt.title('Daily Count by Station Group', fontsize=22, fontweight="bold")
plt.tight_layout()
plt.show()
