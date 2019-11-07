# Show most common words in captions file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Path of captions file
captions_file_path = '../../data/corpus/devset/dev-set/dev-set_video-captions-cleanup.csv'
img_file_path = '../../figures/most_common_words.png'

# Read csv
captions_file = pd.read_csv(captions_file_path)

# Concat all captions rows into one row and then split it to get an array of all words.
captions_splitted = captions_file["captions"].str.cat(sep=' ').split(' ')

# Use Counter to count.
count = Counter(captions_splitted)

# Plotting. Black magic
labels, values = zip(*count.most_common(100))
indexes = np.arange(len(labels))
width = 1

plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
# Rotate labels so they don't overlap and we can understand what is written.
plt.xticks(rotation=90)

# Save img
plt.savefig(img_file_path)

# Show it.
plt.show()

