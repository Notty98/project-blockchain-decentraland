import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

# update font size
plt.rcParams.update({'font.size': 20})

raw_data = pd.read_csv('../dataset/DCL_summary.csv')

# create a matrix with the relevant features of the dataset
X = raw_data.loc[:, ['transactions','ETH','ETH_7d','COIN','COIN_7d','google','tweet_meta','tweet_pojo']]
y = raw_data.loc[:, ['price_usd']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
model = xgb.XGBRegressor(gamma= 0.5, learning_rate= 0.05, max_depth= 25, min_child_weight= 1, n_estimators= 500, objective = 'reg:gamma', subsample= 0.9)
model.fit(X_train, y_train)

importance = model.get_booster().get_score(importance_type='weight')
print(importance)

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

# Create a scatter plot of the approximation and the true values
plt.scatter(y_test, y_pred)

# Add a line for perfect prediction
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

plt.xlabel('True Values (USD)')
plt.ylabel('Approximation (USD)')
plt.title('XGBoost Approximation')

plt.show()