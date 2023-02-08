import xgboost as xgb
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer, r2_score
import pandas as pd

# load the dataset
raw_data = pd.read_csv('./DCL_summary.csv')

# remove the row that doesn't have traffic
# for index, row in raw_data.iterrows():
#     if row['traffic_avg'] == 0:
#         raw_data.drop(index, axis=0, inplace=True)

# create the set of features (X) that will be used to approximate the price_usd (y)
X = raw_data.loc[:, ['transactions','x','y','estate_size','traffic_cum_sum','traffic_max','traffic_avg','ETH','ETH_7d','COIN','COIN_7d','google','tweet_meta','tweet_pojo']]
y = raw_data.loc[:, ['price_usd']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(data = X, label=y)

# define the values of parameters that need to be tuned
domain_params_dicts = {
    #reg:linear deprecated
    'objective' : ['reg:squaredlogerror','reg:gamma','reg:tweedie', 'reg:squarederror', 'count:poisson'],

    'n_estimators': [50, 250, 500],

    'max_depth': [25, 100, 250, 500],

    'min_child_weight': [0.001, 0.1, 1],

    'gamma': [0.001, 0.01, 0.5],

    'subsample': [i/10 for i in range(4,10)],

    'learning_rate' : [0.5, 0.05 , 0.005]

}

# define the accuracy
r2 = make_scorer(r2_score)

# define the SearchCV to obtain the best value that have the best accuracy and the least loss
gsc = GridSearchCV(
            estimator=xgb.XGBRegressor(),
            param_grid=domain_params_dicts,
            cv=3, 
            scoring=['neg_mean_squared_error', 'r2'], 
            refit='r2', 
            verbose=0, 
            n_jobs=-1)

grid_result = MultiOutputRegressor(gsc).fit(X_train, y_train)
best_params = grid_result.estimators_[0].best_params_

print(grid_result.estimators_)
print(best_params)


