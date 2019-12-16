
# Implement a basic LSTM model to predict memorability from saliency
import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense

from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt

# Hyperparameters
SEQUENCE_LENGTH = 50

# Path to file
train_embeddings_file_path = ''
test_embeddings_file_path = ''
train_ground_truth_file_path = ''
test_ground_truth_file_path = ''

# Path model and weights
model_save_path = '../../models/aesthetics/aesthetics-lstm-model.json'
weight_save_path = '../../models/aesthetics/aesthetics-lstm-weight.h5'

# Path for image
img_file_path = '../../figures/aesthetics_lstm_train_loss_class.png'

# Load dataframes
df_train_embeddings = pd.read_csv(train_embeddings_file_path)
df_test_embeddings = pd.read_csv(test_embeddings_file_path)
df_train_ground_truth = pd.read_csv(train_ground_truth_file_path)
df_test_ground_truth = pd.read_csv(test_ground_truth_file_path)

# Now get the embedding matrix.
X_train = df_train_embeddings.iloc[:,2:].to_numpy()
X_test = df_test_embeddings.iloc[:,2:].to_numpy()

# Get X_train, Y_train, X_test, Y_test

def to_classifier(x) :
    if x > 0.77:
        return 1
    return 0

y_train = df_train_ground_truth['long-term_memorability'].apply(to_classifier).to_numpy()
y_test = df_test_ground_truth['long-term_memorability'].apply(to_classifier).to_numpy()

# Y = np.zeros((Y_values.shape[0], 2))

# for i, y in np.ndenumerate(Y_values):
#     if y > 0.77:
#         Y[i] = [1, 0]
#     else:
#         Y[i] = [0, 1] 

# Now define the model
model = Sequential()
model.add(LSTM(SEQUENCE_LENGTH, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2)


score, acc = model.evaluate(X_test, y_test,
                            batch_size=32)

print('Test score with LSTM:', score)
print('Test accuracy with LSTM:', acc)


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
