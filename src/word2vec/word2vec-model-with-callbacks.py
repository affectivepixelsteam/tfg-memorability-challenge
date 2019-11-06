
# Now that we got the embeddings for all of the words that we have in captions, we are going to define the model and train it.

import pandas as pd
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
import os

from keras.layers import Embedding, Input
from keras.layers import LSTM
from keras.layers import Dense

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import utils.simple_callback as simple_callback
from keras.callbacks import TensorBoard

# Path to file
ROOT_PATH = '../'
captions_file_path = ROOT_PATH+'corpus/devset/dev-set/dev-set_video-captions-cleanup.csv'
embeddings_file_path = ROOT_PATH+'corpus/devset/dev-set/embeddings-for-each-word.csv'
ground_truth_file_path = ROOT_PATH+'corpus/devset/dev-set/ground-truth/ground-truth_dev-set.csv'

model_save_path = ROOT_PATH+'models/word2vec/word2vec-model.json'
weight_save_path = ROOT_PATH+'models/word2vec/word2vec-weight.h5'
#model parameters:
batch_size = 64
timesteps = 40
emb_dim = 300
units_lstm = 100
epochs = 10

df_embeddings = pd.read_csv(embeddings_file_path)
df_captions = pd.read_csv(captions_file_path)
df_ground_truth = pd.read_csv(ground_truth_file_path)

# So first we get every word in our captions and stored them without repetition.
# Filter whitespaces too.
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n ')
tokenizer.fit_on_texts(df_captions.captions.tolist())

# captions to sequece
captions_sequece = tokenizer.texts_to_sequences(df_captions.captions)

# add paddings.
captions_sequece = pad_sequences(captions_sequece, maxlen=timesteps)

# Now get the embedding matrix.
embedding_matrix = df_embeddings.iloc[:,1:].to_numpy()

# Get X_train, Y_train, X_test, Y_test

def to_classifier(x) :
    if x > 0.77:
        return 1
    return 0

Y = df_ground_truth['long-term_memorability'].apply(to_classifier).to_numpy()

X = captions_sequece

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=123)

# Now define the model

initial_input = Input(shape=(timesteps,))
input_emb = Embedding(embedding_matrix.shape[0],
                    embedding_matrix.shape[1],
                    input_length=timesteps,
                    weights=[embedding_matrix],
                    trainable=False, name="emb_layer", mask_zero=True)(initial_input)

x = LSTM(units_lstm, dropout=0.2, recurrent_dropout=0.2, return_sequences=False, name="lstm_layer0")(input_emb)
output = Dense(1, activation='sigmoid', name="output_layer")(x)

model = keras.models.Model([initial_input], output)


model.compile(optimizer='adagrad', loss='binary_crossentropy',metrics=['binary_accuracy'] )#metrics=['binary_accuracy']

#DESCOMENTAR LO DE DEBAJO SÓLO PARA DEBUG & AÑADIR EN fit EL CALLBACK callb A LA LISTA DE callbacks
# El truco este funciona para versdiones muy específicas de tf y keras, en mis pruebas funcionó con: tf==1.13.1 & keras==2.2.4
# output_layers = ['emb_layer','lstm_layer0', 'output_layer']
# output_names_in_callback = ['emb_v','lstmOut0_v', 'lastlayer_v']
# model.metrics_names += output_names_in_callback
# model.metrics_tensors += [layer.output for layer in model.layers if layer.name in output_layers]
# callb = simple_callback.simple_Callback(output_names_in_callback)

tensorboard = TensorBoard(log_dir=os.path.join((ROOT_PATH+'/models/'), "logs/"), update_freq='epoch', write_images=True,write_graph=False)
#Para ver los resultados, en un terminal te metes en tu entorno y vas a la carpeta:ROOT_PATH+'/models/   y luego metes en el terminal el comando:  tensboard --logdir=logs


model.summary()

history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2, verbose=1, callbacks=[tensorboard])#,callbacks=[callb, tensorboard]


score, acc= model.evaluate(X_test, y_test)

print('Test score with LSTM:', score)
print('Test accuracy with LSTM:', acc)

# Save model and weights
#...
