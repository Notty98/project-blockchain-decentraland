import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./DCL_summary.csv')
corr_spearman = df.corr(method='spearman')

sns.heatmap(corr_spearman, xticklabels=corr_spearman.columns, yticklabels=corr_spearman.columns, cmap='RdBu')

plt.show()