import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
from scipy.stats import uniform, randint

for db_type in  [True, False]:
    for label_type in ['short_term', 'long_term']:
        use_splitted_db = db_type
        label = label_type
        RANDOM_STATE = 44

        # Path to file
        if use_splitted_db:
            train_embeddings_file_path = '../../data/corpus/devset/dev-set/train_saliency_embeddings_splitted_annotations.csv'
            test_embeddings_file_path = '../../data/corpus/devset/dev-set/test_saliency_embeddings_splitted_annotations.csv'
        else:
            train_embeddings_file_path = '../../data/corpus/devset/dev-set/train_saliency_embeddings_annotations.csv'
            test_embeddings_file_path = '../../data/corpus/devset/dev-set/test_saliency_embeddings_annotations.csv'

        df_train_embeddings = pd.read_csv(train_embeddings_file_path)
        df_test_embeddings = pd.read_csv(test_embeddings_file_path)

        X_train = df_train_embeddings.iloc[:,3:87].to_numpy()
        X_test = df_test_embeddings.iloc[:,3:87].to_numpy()

        y_train = df_train_embeddings[label].to_numpy()
        y_test = df_test_embeddings[label].to_numpy()

        clf = svm.LinearSVR(random_state=RANDOM_STATE, max_iter=4000)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        spearman = spearmanr(y_test, y_pred)
        pearson = pearsonr(y_test, y_pred)

        print ('')
        print ('')
        print ('DB ' + str(use_splitted_db) + ' Label ' + str(label))
        print ('SVR MSE ' + str(mse))
        print ('SVR Spearman ' + str(spearman))
        print ('SVR Pearson' + str(pearson))

        df_test_embeddings["y_pred"] = y_pred
        avg_test = df_test_embeddings.groupby(["0"]).mean()
        mse = mean_squared_error(avg_test[label], avg_test["y_pred"])
        spearman = spearmanr(avg_test[label], avg_test["y_pred"])
        pearson = pearsonr(avg_test[label], avg_test["y_pred"])

        print('SVR MSE VIDEO ' + str(mse))
        print('SVR VIDEO ' + str(spearman))
        print('SVR VIDEO pearson' + str(pearson))