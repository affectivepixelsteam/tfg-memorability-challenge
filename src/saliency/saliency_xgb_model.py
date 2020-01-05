import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
from scipy.stats import uniform, randint

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

for db_type in  [True, False]:
    for label_type in ['short_term', 'long_term']:
        use_splitted_db = db_type
        label = label_type
        hyperparameter_tunning = False
        RANDOM_STATE = 44

        # Path to file
        if use_splitted_db:
            train_embeddings_file_path = '../../data/corpus/devset/dev-set/train_saliency_embeddings_splitted_annotations.csv'
            test_embeddings_file_path = '../../data/corpus/devset/dev-set/test_saliency_embeddings_splitted_annotations.csv'
        else:
            train_embeddings_file_path = '../../data/corpus/devset/dev-set/train_saliency_embeddings_annotations.csv'
            test_embeddings_file_path = '../../data/corpus/devset/dev-set/test_saliency_embeddings_annotations.csv'

        predictions_file_path = '../../data/corpus/devset/dev-set/ground-truth/train_predictions.csv'

        df_train_embeddings = pd.read_csv(train_embeddings_file_path)
        df_test_embeddings = pd.read_csv(test_embeddings_file_path)
        df_predictions = pd.read_csv(predictions_file_path)

        X_train = df_train_embeddings.iloc[:,3:87].to_numpy()
        X_test = df_test_embeddings.iloc[:,3:87].to_numpy()

        y_train = df_train_embeddings[label].to_numpy()
        y_test = df_test_embeddings[label].to_numpy()


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
            # Best result from hyperparameter:
            # Model with rank: 1
            # Mean validation score: 0.056 (std: 0.010)
            # Parameters: {'colsample_bytree': 0.8711391407057438, 'gamma': 0.20338619366637994, 'learning_rate': 0.04551319656966865, 'max_depth': 5, 'n_estimators': 123, 'subsample': 0.785745815782349}

            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', 
            colsample_bytree=0.8711391407057438,
            gamma=0.20338619366637994,
            learning_rate=0.04551319656966865,
            max_depth=5,
            n_estimators=123,
            subsample=0.785745815782349,
            random_state=RANDOM_STATE)

            xgb_model.fit(X_train, y_train)
            y_pred = xgb_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        spearman = spearmanr(y_test, y_pred)
        pearson = pearsonr(y_test, y_pred)

        print ('')
        print ('')
        print ('DB ' + str(use_splitted_db) + ' Label ' + str(label))
        print ('XGB MSE ' + str(mse))
        print ('XGB ' + str(spearman))
        print ('XGB Pearson' + str(pearson))

        df_test_embeddings["y_pred"] = y_pred
        avg_test = df_test_embeddings.groupby(["0"]).mean()
        mse = mean_squared_error(avg_test[label], avg_test["y_pred"])
        spearman = spearmanr(avg_test[label], avg_test["y_pred"])
        pearson = pearsonr(avg_test[label], avg_test["y_pred"])

        print('XGB MSE VIDEO ' + str(mse))
        print('XGB VIDEO ' + str(spearman))
        print('XGB VIDEO pearson' + str(pearson))

        y_train_pred = xgb_model.predict(X_train)
        df_predictions["saliency_xgb_" + label] = y_train_pred
        df_predictions.to_csv(predictions_file_path, index=False)