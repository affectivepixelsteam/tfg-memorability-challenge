# Now that we got the embeddings for all of the words that we have in captions, we are going to define the model and train it.

import sys
sys.path.insert(0, '../../')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import pandas as pd
import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.optimizers import Adagrad
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from scipy.stats import uniform, randint
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

import src.metrics.metrics_regression as metrics_regression

# Allow GPU growth
# INITIAL CONFIG FOR SHARING GPU RESOURCES
config = tf.ConfigProto()

# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True

# config.gpu_options.per_process_gpu_memory_fraction=0.33
#config.allow_soft_placement=True
#config.log_device_placement=False

sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# Hyperparameters
use_tuner = False
classifier = True
EPOCHS = 20
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2

if classifier:
    METRICS = ['binary_accuracy']
    LOSS = 'binary_crossentropy'
    ACTIVATION_FUNCTION = 'sigmoid'
else:
    METRICS = [metrics_regression.spearman_rank_correlation, metrics_regression.PearsonCorrelation4keras]
    LOSS = 'mean_squared_error'
    ACTIVATION_FUNCTION = 'sigmoid'

LSTM_NEURONS = 20
RANDOM_STATE = 44

# label = 'long-term_memorability'
# MEAN_LABEL = 0.7628

label = 'short-term_memorability'
MEAN_LABEL = 0.8544

# Parameters: {'dropout': 0.3393013324064793, 'epochs': 21, 'learning_rate': 0.030109132551140977, 'lstm_neurons': 11, 'recurrent_dropout': 0.31691807115857495}

# Path to file
train_captions_file_path = '../../data/corpus/devset/dev-set/train_dev-set_video-captions-cleanup_splitted.csv'
test_captions_file_path = '../../data/corpus/devset/dev-set/test_dev-set_video-captions-cleanup_splitted.csv'
embeddings_file_path = '../../data/corpus/devset/dev-set/embeddings-for-each-word.csv'
train_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/train_ground-truth_dev-set_splitted.csv'
test_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/test_ground-truth_dev-set_splitted.csv'

# Path model and weights
model_save_path = '../../models/word2vec/word2vec-model.json'
weight_save_path = '../../models/word2vec/word2vec-weight.h5'

# Path for image
img_file_path = '../../figures/word2vec_train_loss_class.png'

# Load dataframes
df_embeddings = pd.read_csv(embeddings_file_path)
df_train_captions = pd.read_csv(train_captions_file_path)
df_test_captions = pd.read_csv(test_captions_file_path)
df_train_ground_truth = pd.read_csv(train_ground_truth_file_path)
df_test_ground_truth = pd.read_csv(test_ground_truth_file_path)

# So first we get every word in our captions and stored them without repetition.
# Filter whitespaces too.
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n ')
tokenizer.fit_on_texts(df_train_captions.captions.tolist())

# captions to sequece
train_captions_sequece = tokenizer.texts_to_sequences(df_train_captions.captions)
test_captions_sequence = tokenizer.texts_to_sequences(df_test_captions.captions)

# add paddings.
train_captions_sequece = pad_sequences(train_captions_sequece)
test_captions_sequence = pad_sequences(test_captions_sequence)

# Now get the embedding matrix.
embedding_matrix = df_embeddings.iloc[:,1:].to_numpy()

# Get X_train, Y_train, X_test, Y_test

def to_classifier(x) :
    if x > MEAN_LABEL:
        return 1
    return 0

if classifier:
    y_train = df_train_ground_truth[label].apply(to_classifier).to_numpy()
    y_test = df_test_ground_truth[label].apply(to_classifier).to_numpy()
else:
    y_train = df_train_ground_truth[label].to_numpy()
    y_test = df_test_ground_truth[label].to_numpy()

# Y = np.zeros((Y_values.shape[0], 2))

# for i, y in np.ndenumerate(Y_values):
#     if y > 0.77:
#         Y[i] = [1, 0]
#     else:
#         Y[i] = [0, 1] 

X_train = train_captions_sequece
X_test = test_captions_sequence

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

def build_model(lstm_neurons=11, learning_rate=0.01, dropout=0.2, recurrent_dropout=0.2):
    # Now define the model
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0],
                        embedding_matrix.shape[1],
                        weights=[embedding_matrix],
                        trainable=False))
    model.add(LSTM(units=lstm_neurons, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=False))
    model.add(Dense(1, activation=ACTIVATION_FUNCTION))

    model.compile(optimizer= Adagrad(lr=learning_rate), loss=LOSS, metrics=METRICS)
    return model

if use_tuner:

    params = {
        "dropout": uniform(0.0, 0.4),
        "recurrent_dropout": uniform(0.0, 0.4),
        "lstm_neurons": randint(5, 50),
        "learning_rate": uniform(0.03, 0.3), # default 0.1
        "epochs": randint(20, 70)
    } 

    keras_wrapper = KerasRegressor(build_fn=build_model, batch_size=BATCH_SIZE)
    search = RandomizedSearchCV(estimator=keras_wrapper, param_distributions=params, random_state=RANDOM_STATE, n_iter=200, cv=3, verbose=1, n_jobs=-1, return_train_score=True)
    search.fit(X_train, y_train)
    report_best_scores(search.cv_results_, 1)

    y_pred = search.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    spearman = spearmanr(y_test, y_pred)

    print ('XGB MSE ' + str(mse))
    print ('XGB ' + str(spearman))
else:
    model = build_model(lstm_neurons=LSTM_NEURONS)

    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_split=VALIDATION_SPLIT)

    if classifier:
        score, accuracy = model.evaluate(X_test, y_test,
                                    batch_size=BATCH_SIZE)

        print('Test score with LSTM:', score)
        print('Test accuracy with LSTM:', accuracy)
    else:
        score, spearman, pearson = model.evaluate(X_test, y_test,
                                    batch_size=BATCH_SIZE)

        print('Test score with LSTM:', score)
        print('Test spearman with LSTM:', spearman)
        print('Test pearson with LSTM:', pearson)

    with open(model_save_path, 'w+') as save_file:
        save_file.write(model.to_json())

    model.save_weights(weight_save_path)

    # Plotting lost
    plt.suptitle('Optimizer : adagrad', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.plot(history.history['loss'], color='b', label='Training Loss')
    plt.legend(loc='upper right')

    # Save img
    plt.savefig(img_file_path)

    plt.show()
