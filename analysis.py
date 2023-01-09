import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./DCL_summary.csv')

corr_spearman = df.corr(method='spearman')
corr_pearson = df.corr(method='pearson')
corr_kendall = df.corr(method='kendall')

fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(8, 16))

ax1, ax2, ax3 = axes

ax1.set_title('spearman')
ax2.set_title('pearson')
ax3.set_title('kendall')

# create a heatmap for each correlation method
im1 = sns.heatmap(corr_spearman, xticklabels=corr_spearman.columns, yticklabels=corr_spearman.columns, cmap='RdBu', ax=ax1)
im2 = sns.heatmap(corr_pearson, xticklabels=corr_pearson.columns, yticklabels=corr_pearson.columns, cmap='RdBu', ax=ax2)
im3 = sns.heatmap(corr_kendall, xticklabels=corr_kendall.columns, yticklabels=corr_kendall.columns, cmap='RdBu', ax=ax3)

plt.show()
