import xgboost as xgb
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import numpy as np 

raw_data = pd.read_csv('./DCL_summary.csv')

# create a matrix with only the column that have a positive correlation with the price (see analysis.py)
X = raw_data.loc[:, ['transactions', 'ETH', 'ETH_7d', 'COIN', 'COIN_7d', 'google', 'tweet_meta', 'tweet_pojo']]
y = raw_data.loc[:, ['price_usd']] # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(data = X, label=y)

""" domain_params_dicts = {
    #reg:linear deprecated
    'objective' : ['reg:squarederror','reg:squaredlogerror','reg:gamma','reg:tweedie'],

    'n_estimators': [30, 50, 70, 100, 150, 200, 300],

    'max_depth': [3, 5, 7, 9],

    'min_child_weight': [0.001, 0.1, 1, 5, 10, 20],

    'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],

    'n_estimators': [30, 50, 70, 100, 150, 200, 300],

    'subsample': [i/10 for i in range(4,10)],

    'learning_rate' : [0.5, 0.25 , 0.05, 0.005]

} """

domain_params_dicts = {
    #reg:linear deprecated
    'objective' : ['reg:squarederror','reg:squaredlogerror','reg:gamma','reg:tweedie'],

    'n_estimators': [50, 100, 150],

    'max_depth': [10, 100, 150],

    'min_child_weight': [0.001, 0.1, 1],

    'gamma': [0.01, 0.1, 0.5],

    'n_estimators': [10, 50, 100],

    'subsample': [i/10 for i in range(4,10)],

    'learning_rate' : [0.5, 0.25 , 0.05]

}

gsc = GridSearchCV(
            estimator=xgb.XGBRegressor(),
            param_grid=domain_params_dicts,
            cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

grid_result = MultiOutputRegressor(gsc).fit(X_train, y_train)
best_params = grid_result.estimators_[0].best_params_
print(grid_result.estimators_)
print(best_params)


