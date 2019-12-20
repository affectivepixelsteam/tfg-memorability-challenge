
# Implement a basic LSTM model to predict memorability from saliency
import pandas as pd
import numpy as np
import numpy.random as rand
import math, os, random
import datetime

import tensorflow as tf
from tensorflow import set_random_seed

import keras
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras import backend as K
import src.utils.nw_config_functions as nw_config

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

config_ses=tf.ConfigProto()
config_ses.gpu_options.allow_growth=True
#config_ses.gpu_options.per_process_gpu_memory_fraction=0.8
config_ses.allow_soft_placement=True
config_ses.log_device_placement=False

# Hyperparameters
SEQUENCE_LENGTH = 168
EMBEDDING_LENGTH = 84
EPOCHS = 150#15
BATCH_SIZE = 224#
VALIDATION_SPLIT = 0.2
COMPILER = 'adam' #adagrad, rmsprop
LEARNING_RATE = 0.0001
NUM_NEURONS = 50
LOSS = 'bxentropy' #bxentropy
METRICS = ['accuracy'] #accuracy
classification = True
SEED = 10
# SAME SEED
random.seed(SEED)
rand.seed(SEED)
set_random_seed(SEED)

# Path to file
train_embeddings_file_path = '../../data/corpus/devset/dev-set/train_saliency_embeddings_splitted.csv'
test_embeddings_file_path = '../../data/corpus/devset/dev-set/test_saliency_embeddings_splitted.csv'
train_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/train_ground-truth_dev-set_splitted.csv'
test_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/test_ground-truth_dev-set_splitted.csv'
save_processed_files = "../../data/corpus/devset/dev-set/numpy_files/"

# Path model and weights
model_save_path = '../../models/saliency/saliency-lstm-model_NEWCRIS.json'
weight_save_path = '../../models/saliency/saliency-lstm-weight_NEWCRIS.h5'
current_time = (datetime.datetime.now()).strftime("%Y%M%d_%H%M%S")
log_dir = '../../models/LOGS_saliency/'+current_time

headers = ["EPOCHS","BATCH_SIZE", "VAL_SPLIT", "COMPILER", "LEARNING_RATE", "NUM_NEURONS", "LOSS", "SEED"]
df_parameters = pd.DataFrame([[EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, COMPILER, LEARNING_RATE, NUM_NEURONS, LOSS, SEED]], columns=headers)

if(not os.path.exists(log_dir)):
    os.mkdir(log_dir)
    df_parameters.to_csv(log_dir+"/info_parameters.csv", index=False)

# Path for image
img_file_path = '../../figures/saliency_lstm_train_loss_class_NEWCRIS.png'


def data_to_input(df_embeddings, df_ground_truth):
    i = 0
    df_input = np.zeros((len(df_ground_truth), SEQUENCE_LENGTH, EMBEDDING_LENGTH))
    for video_name in df_ground_truth['video'].apply(lambda x : x.split('.')[0]).values:
        video_embeddings = df_embeddings.loc[df_embeddings['0'] == video_name]
        video_embeddings = video_embeddings.iloc[:,3:].to_numpy()
        df_input[i] = video_embeddings
        i += 1

    return df_input

def tf_pearson(y_true, y_pred):
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]

def step_decay(epoch):
   initial_lrate = LEARNING_RATE
   drop = 0.5
   epochs_drop = 25#settings.epochs_dr #10.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   print("new lr:", str(lrate))
   return lrate


def to_classifier(x) :
    if x > 0.77:
        return 1
    return 0

if(os.path.isfile(save_processed_files+"X_train.npy")):
    #load data
    X_train = np.load(save_processed_files+"X_train.npy")
    y_train = np.load(save_processed_files + "y_train.npy")
    X_test = np.load(save_processed_files + "X_test.npy")
    y_test = np.load(save_processed_files + "y_test.npy")
else:
    # Load dataframes
    df_train_embeddings = pd.read_csv(train_embeddings_file_path)
    df_test_embeddings = pd.read_csv(test_embeddings_file_path)
    df_train_ground_truth = pd.read_csv(train_ground_truth_file_path)
    df_test_ground_truth = pd.read_csv(test_ground_truth_file_path)

    # Sort dataframes
    df_train_embeddings = df_train_embeddings.sort_values(['0', '1'], ascending=[True, True])
    df_test_embeddings = df_test_embeddings.sort_values(['0', '1'], ascending=[True, True])
    df_train_ground_truth = df_train_ground_truth.sort_values('video', ascending=True)
    df_test_ground_truth = df_test_ground_truth.sort_values('video', ascending=True)

    # Get X_train, Y_train, X_test, Y_test
    X_train = data_to_input(df_train_embeddings, df_train_ground_truth)
    X_test = data_to_input(df_test_embeddings, df_test_ground_truth)

    y_train = df_train_ground_truth['long-term_memorability'].to_numpy()
    y_test = df_test_ground_truth['long-term_memorability'].to_numpy()

    #SAVE FILES
    np.save(save_processed_files+"X_train.npy", X_train)
    np.save(save_processed_files+"y_train.npy", y_train)
    np.save(save_processed_files+"X_test.npy", X_test)
    np.save(save_processed_files+"y_test.npy", y_test)

if(classification):
    y_train_new = []
    y_test_new = []
    for label in y_train:
        y_train_new += [to_classifier(label)]
    for label in y_test:
        y_test_new += [to_classifier(label)]
    y_train = np.asarray(y_train_new)
    y_test = np.asarray(y_test_new)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y_train.reshape(-1, 1))
    y_train_oneHot = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test_oneHot = enc.transform(y_test.reshape(-1, 1)).toarray()
    NEURONS_LAST_LAYER = 2
    ACTIVATION_LAST_LAYER = 'softmax'
else:
    y_test_oneHot = y_test
    y_train_oneHot = y_train
    NEURONS_LAST_LAYER = 1
    ACTIVATION_LAST_LAYER = 'linear'


print (X_train[0,:,:])
print (y_train[0])
print (X_train.shape)
print(X_test.shape)
print (y_train.shape)
print(y_test.shape)



# Y = np.zeros((Y_values.shape[0], 2))

# for i, y in np.ndenumerate(Y_values):
#     if y > 0.77:
#         Y[i] = [1, 0]
#     else:
#         Y[i] = [0, 1]

# Now define the model
#initial_input = Input(shape=(SEQUENCE_LENGTH,EMBEDDING_LENGTH))
input_layer = keras.layers.Input(shape=(SEQUENCE_LENGTH,EMBEDDING_LENGTH),name="input")
x = LSTM(NUM_NEURONS, dropout=0, recurrent_dropout=0.0, return_sequences=False, name="LSTM_0")(input_layer)
output = Dense(NEURONS_LAST_LAYER, activation=ACTIVATION_LAST_LAYER, name="output_layer")(x)
model = keras.models.Model([input_layer], output)
model.summary()
#model = Sequential()
# model.add(LSTM(SEQUENCE_LENGTH, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
# model.add(Dense(1, activation='linear'))
#keras.models.Model([input_layer], output)
# adagrad = Adagrad(learning_rate=0.05)

#CALLBACKS
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',#val_loss
                                              patience=40, verbose=2, mode='auto',
                                              restore_best_weights=True)  # EARLY STOPPING

lrate_callback = keras.callbacks.LearningRateScheduler(step_decay)#DECAY LEARNING RATE #más info sobre learning rate decay methods in: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
tensorboard = TensorBoard(log_dir=log_dir,update_freq='epoch',write_images=True,write_graph=False) #CONTROL THE TRAINING

optim, lossFunct = nw_config.get_optim_and_lossFunct(optimizer = COMPILER, learning_rate = LEARNING_RATE, lossFunction=LOSS)
model.compile(optimizer=optim, loss=lossFunct, metrics=METRICS)
history = model.fit(X_train, y_train_oneHot,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=VALIDATION_SPLIT,callbacks=[tensorboard,lrate_callback,earlyStopping])
#in order to see the info plotted by the tensorboard callback, open a terminal and type: tensorboard --logdir=/....Ruta aquí.../tfg-memorability-challenge/models/LOGS_saliencyLOGS_saliency/

model.summary()
score, acc = model.evaluate(X_test, y_test_oneHot,
                            batch_size=BATCH_SIZE)

print('Test score with LSTM:', score)
print('Test accuracy with LSTM:', acc)


with open(model_save_path, 'w+') as save_file:
    save_file.write(model.to_json())

model.save_weights(weight_save_path)

# Plotting loss
plt.suptitle('Optimizer : adagrad', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.plot(history.history['loss'], color='b', label='Training Loss')
plt.legend(loc='upper right')

# Save img
plt.savefig(img_file_path)

plt.show()