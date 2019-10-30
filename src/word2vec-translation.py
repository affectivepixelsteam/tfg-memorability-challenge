
# Now that we got the embeddings for all of the words that we have in captions. We are get the sequence and save it in the csv file

import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Path to file
captions_file_path = '../corpus/devset/dev-set/dev-set_video-captions-cleanup.csv'
embeddings_file_path = '../corpus/devset/dev-set/embeddings-for-each-word.csv'

embeddings_file = pd.read_csv(embeddings_file_path)
captions_file = pd.read_csv(captions_file_path)

# So first we get every word in our captions and stored them without repetition.
# Filter whitespaces too.
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n ')
tokenizer.fit_on_texts(captions_file.captions.tolist())

# captions to sequece
captions_sequece = tokenizer.texts_to_sequences(captions_file.captions)

# add paddings.
captions_sequece = pad_sequences(captions_sequece)

# Save sequence in dataframe
i = 0
for column in captions_sequece.T:
    captions_file["emb" + str(i)] = column
    i += 1



# Save dataframe
save_file_path = '../corpus/devset/dev-set/dev-set_video-captions_with-embedding.csv'
captions_file.to_csv(save_file_path)
