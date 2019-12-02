
# Now that we got the embeddings for all of the words that we have in captions, we are going to define the model and train it.

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

X_train = train_captions_sequece
X_test = test_captions_sequence

# Now define the model
model = Sequential()
model.add(Embedding(embedding_matrix.shape[0],
                    embedding_matrix.shape[1],
                    weights=[embedding_matrix],
                    trainable=False))
model.add(LSTM(embedding_matrix.shape[1], dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
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
