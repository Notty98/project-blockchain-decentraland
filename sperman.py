import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./DCL_summary.csv')

df.drop('is_road',inplace=True,axis=1)
df.drop('is_plaza',inplace=True,axis=1)
df.drop('x',inplace=True,axis=1)
df.drop('y',inplace=True,axis=1)

corr_spearman = df.corr(method='spearman')

sns.heatmap(corr_spearman, xticklabels=corr_spearman.columns, yticklabels=corr_spearman.columns, cmap='RdBu')

plt.show()