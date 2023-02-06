import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./DCL_summary.csv')

# remove the features that cannot be used in the correlation analysis
df.drop('is_road', inplace=True, axis=1)
df.drop('is_plaza', inplace=True, axis=1)
df.drop('estate_size', inplace=True, axis=1)
df.drop('x', inplace=True, axis=1)
df.drop('y', inplace=True, axis=1)

# create the spearmann correlation matrix
corr_spearman = df.corr(method='spearman')

# create a heatmap with the features name and the reversed color (Red with high correlation and Blue with low correlation)
sns.heatmap(corr_spearman, xticklabels=corr_spearman.columns, yticklabels=corr_spearman.columns, cmap='RdBu_r')

plt.show()