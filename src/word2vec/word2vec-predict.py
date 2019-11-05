import pandas as pd

from keras.models import model_from_json

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

model_path = '../models/word2vec/word2vec-model.json'
weight_path = '../models/word2vec/word2vec-weight.h5'
captions_file_path = '../corpus/devset/dev-set/dev-set_video-captions-cleanup.csv'
ground_truth_file_path = '../corpus/devset/dev-set/ground-truth/ground-truth_dev-set.csv'

# load json and create model
with open(model_path, 'r') as model_file:
    model = model_file.read()

model = model_from_json(model)

# load weights into new model
model.load_weights(weight_path)
print("Loaded model from disk")


# load some captions data
df_captions = pd.read_csv(captions_file_path)
df_ground_truth = pd.read_csv(ground_truth_file_path)

# So first we get every word in our captions and stored them without repetition.
# Filter whitespaces too.
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n ')
tokenizer.fit_on_texts(df_captions.captions.tolist())

# captions to sequece
captions_sequece = tokenizer.texts_to_sequences(df_captions.captions)

# add paddings.
captions_sequece = pad_sequences(captions_sequece)
 
Y = df_ground_truth['short-term_memorability'].to_numpy()

# predict
model.compile(optimizer='adagrad', loss='mean_squared_error', metrics=["mean_squared_error"])

ypred = model.predict(captions_sequece[0])

print ('ypred:' + str(ypred) + 'yreal:' + str(Y[0]))
