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
#X = raw_data.loc[:, ['transactions', 'ETH','ETH_7d','COIN', 'COIN_7d', 'google', 'tweet_meta', 'tweet_pojo']] #best correlation
#X = raw_data.loc[:, ['transactions', 'ETH','ETH_7d','COIN', 'COIN_7d', 'google', 'tweet_meta', 'tweet_pojo']] #best correlation + position
#X = raw_data.loc[:, ['transactions','ETH_7d','COIN_7d', 'tweet_meta', 'tweet_pojo', 'x','y','estate_size','traffic_cum_sum']] #best correlation without noise
#X = raw_data.loc[:, ['transactions','price_usd_avg_7d','x','y','is_road','is_plaza','estate_size','traffic_cum_sum','traffic_max','traffic_avg','ETH','ETH_7d','COIN','COIN_7d','google','tweet_meta','tweet_pojo']] #all features
X = raw_data.loc[:, ['transactions','ETH','ETH_7d','COIN_7d', 'tweet_meta', 'tweet_pojo', 'x','y','estate_size','traffic_cum_sum']] #best correlation without noise + ETH
#X = raw_data.loc[:, ['transactions','ETH_7d','COIN','COIN_7d', 'tweet_meta', 'tweet_pojo', 'x','y','estate_size','traffic_cum_sum']] #best correlation without noise + COIN
#X = raw_data.loc[:, ['transactions','ETH','ETH_7d','COIN','COIN_7d', 'tweet_meta', 'tweet_pojo', 'x','y','estate_size','traffic_cum_sum']] #best correlation without noise + COIN & ETH
#X = raw_data.loc[:, ['transactions','ETH','COIN', 'tweet_meta', 'tweet_pojo', 'x','y','estate_size','traffic_cum_sum']] #best correlation without noise - _7d
y = raw_data.loc[:, ['price_usd']] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
model = xgb.XGBRegressor(objective='reg:tweedie', learning_rate='0.5', n_estimators=1000, max_depth=32, min_child_weight=10)
model.fit(X_train, y_train)

#model1= xgb.XGBRegressor(objective='reg:tweedie',gamma = 0.0, max_depth = 3, min_child_weight= 0.001, n_estimators = 30, subsample = 0.9)
#model1= xgb.XGBRegressor(objective='reg:gamma',gamma = 0.1,learning_rate=0.5, max_depth = 3, min_child_weight= 0.001, n_estimators = 50, subsample = 0.9)
model1= xgb.XGBRegressor(objective='reg:squarederror',gamma = 0.01,learning_rate=0.05, max_depth = 9, min_child_weight= 0.001, n_estimators = 150, subsample = 0.8)
model1.fit(X_train,y_train)

importance = model.get_booster().get_score(importance_type='weight')
print(importance)

# TEST
y_pred = model.predict(X_test)
y_pred1 = model1.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print("MSE : % f" %(mse))

print("RMSE: % f" %(np.sqrt(mse)))

mse_1 = mean_squared_error(y_test, y_pred1)
print("MSE model1: % f" %(mse_1))

print("RMSE model1: % f" %(np.sqrt(mse_1)))

mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE : % f" %(mape))

mape_1 = mean_absolute_percentage_error(y_test, y_pred1)
print("MAPE model1: % f" %(mape_1))

mae = mean_absolute_error(y_test, y_pred)
print("MAE : % f" %(mae))

mae_1 = mean_absolute_error(y_test, y_pred1)
print("MAE model1: % f" %(mae_1))

r2 = r2_score(y_test, y_pred)
print("R2 : % f" %(r2))

r2_1 = r2_score(y_test, y_pred1)
print("R2 model1: % f" %(r2_1))

# Create a scatter plot of the predictions and the true values
plt.scatter(y_test, y_pred)

# Add a line for perfect prediction
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('XGBoost Predictions')

plt.show()