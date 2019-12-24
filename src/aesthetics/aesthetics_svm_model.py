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

clf = svm.LinearSVR(random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
spearman = spearmanr(y_test, y_pred)

print ('SVR MSE ' + str(mse))
print ('SVR ' + str(spearman))
