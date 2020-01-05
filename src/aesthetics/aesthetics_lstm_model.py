
# Implement a basic LSTM model to predict memorability from saliency
import sys
sys.path.insert(0, '../../')

import pandas as pd
import numpy as np

import tensorflow as tf

from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras import backend as K

from sklearn.model_selection import train_test_split 

import src.metrics.metrics_regression as metrics_regression

import matplotlib.pyplot as plt

# Hyperparameters
SEQUENCE_LENGTH = 168
LSTM_UNITS = 50
EMBEDDING_LENGTH = 84
EPOCHS = 20
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
METRICS = [metrics_regression.spearman_rank_correlation, metrics_regression.PearsonCorrelation4keras]
loss = 'mean_squared_error'
label = 'long-term_memorability'

# Path to file
train_embeddings_file_path = '../../data/corpus/devset/dev-set/train_aesthetics_embeddings_annotations.csv'
test_embeddings_file_path = '../../data/corpus/devset/dev-set/test_aesthetics_embeddings_annotations.csv'
train_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/train_ground-truth_dev-set.csv'
test_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/test_ground-truth_dev-set.csv'

# Path model and weights
model_save_path = '../../models/aesthetics/aesthetics-lstm-model.json'
weight_save_path = '../../models/aesthetics/aesthetics-lstm-weight.h5'

# Path for image
img_file_path = '../../figures/aesthetics_lstm_train_loss_class.png'


def data_to_input(df_embeddings, df_ground_truth):
    i = 0
    df_input = np.zeros((len(df_ground_truth), SEQUENCE_LENGTH, EMBEDDING_LENGTH))
    for video_name in df_ground_truth['video'].apply(lambda x : x.split('.')[0]).values:
        video_embeddings = df_embeddings.loc[df_embeddings['0'] == video_name]
        video_embeddings = video_embeddings.iloc[:,3:87].to_numpy()
        df_input[i] = video_embeddings
        i += 1

    return df_input

def tf_pearson(y_true, y_pred):
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]
    
# Load dataframes
df_train_embeddings = pd.read_csv(train_embeddings_file_path)
df_test_embeddings = pd.read_csv(test_embeddings_file_path)
df_train_ground_truth = pd.read_csv(train_ground_truth_file_path)
df_test_ground_truth = pd.read_csv(test_ground_truth_file_path)

# Sort dataframes
# df_train_embeddings = df_train_embeddings.sort_values(['0', '1'], ascending=[True, True])
# df_test_embeddings = df_test_embeddings.sort_values(['0', '1'], ascending=[True, True])
df_train_ground_truth = df_train_ground_truth.sort_values('video', ascending=True)
df_test_ground_truth = df_test_ground_truth.sort_values('video', ascending=True)

# Get X_train, Y_train, X_test, Y_test
X_train = data_to_input(df_train_embeddings, df_train_ground_truth)
X_test = data_to_input(df_test_embeddings, df_test_ground_truth)

def to_classifier(x) :
    if x > 0.77:
        return 1
    return 0

y_train = df_train_ground_truth[label].to_numpy()
y_test = df_test_ground_truth[label].to_numpy()

# Y = np.zeros((Y_values.shape[0], 2))

# for i, y in np.ndenumerate(Y_values):
#     if y > 0.77:
#         Y[i] = [1, 0]
#     else:
#         Y[i] = [0, 1] 

# Now define the model
model = Sequential()
model.add(LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
model.add(Dense(1, activation='linear'))

adagrad = Adagrad(lr = 0.01)
model.compile(optimizer=adagrad, loss=loss, metrics=METRICS)

history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=VALIDATION_SPLIT)

model.summary()
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
