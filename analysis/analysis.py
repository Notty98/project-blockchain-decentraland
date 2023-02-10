import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../dataset/DCL_summary.csv')

# remove the features that cannot be used in the correlation analysis
df.drop('is_road', inplace=True, axis=1)
df.drop('is_plaza', inplace=True, axis=1)
df.drop('estate_size', inplace=True, axis=1)
df.drop('x', inplace=True, axis=1)
df.drop('y', inplace=True, axis=1)


corr_spearman = df.corr(method='spearman')
corr_pearson = df.corr(method='pearson')
corr_kendall = df.corr(method='kendall')

fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(8, 16))

ax1, ax2, ax3 = axes

ax1.set_title('spearman')
ax2.set_title('pearson')
ax3.set_title('kendall')

# create a heatmap for each correlation method
im1 = sns.heatmap(corr_spearman, xticklabels=corr_spearman.columns, yticklabels=corr_spearman.columns, cmap='RdBu_r', ax=ax1)
im2 = sns.heatmap(corr_pearson, xticklabels=corr_pearson.columns, yticklabels=corr_pearson.columns, cmap='RdBu_r', ax=ax2)
im3 = sns.heatmap(corr_kendall, xticklabels=corr_kendall.columns, yticklabels=corr_kendall.columns, cmap='RdBu_r', ax=ax3)

plt.show()
