import pandas as pd
import numpy as np

import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

from keras.preprocessing.text import Tokenizer

print('Starting...')

# Path to file
captions_file_path = '../corpus/devset/dev-set/dev-set_video-captions-cleanup.csv'

# Load word2vec Google's pretrained model
word_vectors = KeyedVectors.load_word2vec_format('../corpus/models/GoogleNews-vectors-negative300.bin', binary=True)
print ('Word2vec model loaded')

# Load captions file
captions_file = pd.read_csv(captions_file_path)

# So first we get every word in our captions and stored them without repetition.
# Filter whitespaces too.
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n ')
tokenizer.fit_on_texts(captions_file.captions.tolist())
word_index = tokenizer.word_index

# length of vectors from word2vec model.
EMBEDDING_DIM = 300

# Create matrix that will contain the embedding for each word.
vocabulary_size = len(word_index) + 1
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

# Now we get the embedding for each word.
for word, i in word_index.items():
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        # So, in case the word from captions doesn't exists in word2vec model (which will be pretty strange) assign to that word a random embedding.
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)

# Remove word_vectors since it is not needed anymore.
del(word_vectors)

# Convert numpy array to pandas dataframe
embedding_file = pd.DataFrame(embedding_matrix)

print('Got embedding matrix')

# Save embeddings for each word.
save_file_path = '../corpus/devset/dev-set/embeddings-for-each-word.csv'
embedding_file.to_csv(save_file_path)

print('Saved')
