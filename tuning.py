import xgboost as xgb
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
import pandas as pd
import numpy as np 

raw_data = pd.read_csv('./DCL_summary.csv')

# create a matrix with only the column that have a positive correlation with the price (see analysis.py)
X = raw_data.loc[:, ['transactions', 'ETH','ETH_7d','COIN', 'COIN_7d', 'google', 'tweet_meta', 'tweet_pojo']]
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
    'objective' : ['reg:squaredlogerror','reg:gamma','reg:tweedie', 'reg:squarederror', 'reg:logistic', 'reg:poisson', 'reg:quantile'],

    'n_estimators': [50, 250, 500],

    'max_depth': [25, 100, 250],

    'min_child_weight': [0.001, 0.1, 1],

    'gamma': [0.001, 0.01, 0.5],

    'subsample': [i/10 for i in range(4,10)],

    'learning_rate' : [0.5, 0.05 , 0.005]

}

#mean_squared_error = make_scorer(mean_squared_error, greater_is_better=False)
r2 = make_scorer(r2_score)

gsc = GridSearchCV(
            estimator=xgb.XGBRegressor(),
            param_grid=domain_params_dicts,
            cv=3, scoring=['neg_mean_squared_error', 'r2'], refit='r2', verbose=0, n_jobs=-1)

grid_result = MultiOutputRegressor(gsc).fit(X_train, y_train)
best_params = grid_result.estimators_[0].best_params_
print(grid_result.estimators_)
print(best_params)


