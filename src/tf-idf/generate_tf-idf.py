import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# First we will create two documents. One will contain the words from the higher memorable videos and the other one will have the words from the
# lowest memorable videos.
# Then we will use tf-idf to retrieve which words appears more in one document or the other and try to implement a neural network which is able
# to classify well between higher and lower memorable videos.

# Function to remove stopwords from documents
def stopwords_remover(document):
    english_stopwords = stopwords.words('english')
    words = [word for word in document.split() if word.lower() not in english_stopwords]
    result = ' '.join(words)
    return result

# Paths to dataset
splitted_captions_file_path = '../../data/corpus/devset/dev-set/dev-set_video-captions-cleanup_splitted.csv'
tf_idf_embeddings_file_path = '../../data/corpus/devset/dev-set/tf-idf_embeddings.csv'

# Load it
df_captions = pd.read_csv(splitted_captions_file_path)

# Split dataset in two halfs. First half are the higher memorable videos and the other half are the lower ones.
df_high_memorable, df_low_memorable = np.split(df_captions, [int(0.5*len(df_captions))])

# Get both documents and remove stopwords from it
high_memorable_captions = df_high_memorable["captions"].str.cat(sep=' ')
low_memorable_captions = df_low_memorable["captions"].str.cat(sep=' ')

# Use tf-idf from sklearn
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([high_memorable_captions, low_memorable_captions])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df_tfidf = pd.DataFrame(denselist, columns=feature_names).T

# Now tokenize our captions to transform each word to its respective integer
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n ')
tokenizer.fit_on_texts(df_captions.captions.tolist())

# words to integer
words_as_integers = []
for word in df_tfidf.index.values:
    int_word = tokenizer.texts_to_sequences([word])
    words_as_integers.append(int_word[0][0])

# replace index of words to a index of integers
df_tfidf.index = words_as_integers

# Append zero integer which is added as padding
df_tfidf = df_tfidf.append(pd.DataFrame([[0.0, 0.0]]))

df_tfidf = df_tfidf.sort_index()

# Save embeddings
df_tfidf.to_csv(tf_idf_embeddings_file_path)