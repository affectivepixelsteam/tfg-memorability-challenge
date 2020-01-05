# Try to implement a model with tf-idf as embeddings.
import sys
sys.path.insert(0, '../../')

import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense

from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt

import src.metrics.metrics_regression as metrics_regression

# Hyperparameters
use_tuner = False
classifier = False
EPOCHS = 20
BATCH_SIZE = 128
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

# Path to file
train_captions_file_path = '../../data/corpus/devset/dev-set/train_dev-set_video-captions-cleanup.csv'
test_captions_file_path = '../../data/corpus/devset/dev-set/test_dev-set_video-captions-cleanup.csv'
embeddings_file_path = '../../data/corpus/devset/dev-set/tf-idf_embeddings.csv'
train_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/train_ground-truth_dev-set.csv'
test_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/test_ground-truth_dev-set.csv'

# Path model and weights
model_save_path = '../../models/tf-idf/tf-idf_model.json'
weight_save_path = '../../models/tf-idf/tf-idf_weight.h5'

# Path for image
img_file_path = '../../figures/tf-idf_train_loss_class.png'

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
test_captions_sequece = tokenizer.texts_to_sequences(df_test_captions.captions)

# add paddings.
train_captions_sequece = pad_sequences(train_captions_sequece)
test_captions_sequece = pad_sequences(test_captions_sequece)

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

X_train = train_captions_sequece
X_test  = test_captions_sequece

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
    
# Now define the model
model = build_model(LSTM_NEURONS)
model.summary()


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

# Save model
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
