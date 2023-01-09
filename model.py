import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

raw_data = pd.read_csv('./DCL_summary.csv')

# create a matrix with only the column that have a positive correlation with the price (see analysis.py)
X = raw_data.loc[:, ['transactions', 'ETH', 'ETH_7d', 'COIN', 'COIN_7d', 'google', 'tweet_meta', 'tweet_pojo']]
y = raw_data.loc[:, ['price_usd']] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
model = xgb.XGBRegressor(objective='reg:tweedie', learning_rate='0.5', n_estimators=1000, max_depth=32, min_child_weight=10)
model.fit(X_train, y_train)

# TEST
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("MSE : % f" %(mse))

print("RMSE: % f" %(np.sqrt(mse)))

mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE : % f" %(mape))

mae = mean_absolute_error(y_test, y_pred)
print("MAE : % f" %(mae))

r2 = r2_score(y_test, y_pred)
print("R2 : % f" %(r2))

# Create a scatter plot of the predictions and the true values
plt.scatter(y_test, y_pred)

# Add a line for perfect prediction
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('XGBoost Predictions')

plt.show()