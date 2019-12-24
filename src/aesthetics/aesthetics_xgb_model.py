import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from scipy.stats import uniform, randint

use_splitted_db = False
use_short_term = False
hyperparameter_tunning = False
RANDOM_STATE = 44

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# Path to file
if use_splitted_db:
    train_embeddings_file_path = '../../data/corpus/devset/dev-set/train_aesthetics_embeddings_splitted_annotations.csv'
    test_embeddings_file_path = '../../data/corpus/devset/dev-set/test_aesthetics_embeddings_splitted_annotations.csv'
else:
    train_embeddings_file_path = '../../data/corpus/devset/dev-set/train_aesthetics_embeddings_annotations.csv'
    test_embeddings_file_path = '../../data/corpus/devset/dev-set/test_aesthetics_embeddings_annotations.csv'

df_train_embeddings = pd.read_csv(train_embeddings_file_path)
df_test_embeddings = pd.read_csv(test_embeddings_file_path)

X_train = df_train_embeddings.iloc[:,3:87].to_numpy()
X_test = df_test_embeddings.iloc[:,3:87].to_numpy()

if use_short_term:
    y_train = df_train_embeddings['short_term'].to_numpy()
    y_test = df_test_embeddings['short_term'].to_numpy()
else:
    y_train = df_train_embeddings['long_term'].to_numpy()
    y_test = df_test_embeddings['long_term'].to_numpy()


y_pred = []

if hyperparameter_tunning:

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE)
    params = {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3), # default 0.1 
        "max_depth": randint(2, 6), # default 3
        "n_estimators": randint(100, 150), # default 100
        "subsample": uniform(0.6, 0.4)
    }

    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=RANDOM_STATE, n_iter=200, cv=3, verbose=2, n_jobs=-1, return_train_score=True)
    search.fit(X_train, y_train)
    report_best_scores(search.cv_results_, 1)

    y_pred = search.predict(X_test)
else:
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE)

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
spearman = spearmanr(y_test, y_pred)

print ('XGB MSE ' + str(mse))
print ('XGB ' + str(spearman))

