import pandas as pd
import numpy as np

# Files
train_embeddings_file_path = '../../data/corpus/devset/dev-set/train_aesthetics_embeddings.csv'
train_embeddings_splitted_file_path = '../../data/corpus/devset/dev-set/train_aesthetics_embeddings_splitted.csv'
test_embeddings_file_path = '../../data/corpus/devset/dev-set/test_aesthetics_embeddings.csv'
test_embeddings_splitted_file_path = '../../data/corpus/devset/dev-set/test_aesthetics_embeddings_splitted.csv'
train_ground_truth_splitted_file_path = '../../data/corpus/devset/dev-set/ground-truth/train_ground-truth_dev-set_splitted.csv'
test_ground_truth_splitted_file_path = '../../data/corpus/devset/dev-set/ground-truth/test_ground-truth_dev-set_splitted.csv'

# Dataframes
df_train_ground_truth_splitted = pd.read_csv(train_ground_truth_splitted_file_path)
df_train_embeddings = pd.read_csv(train_embeddings_file_path)
df_test_ground_truth_splitted = pd.read_csv(test_ground_truth_splitted_file_path)
df_test_embeddings = pd.read_csv(test_embeddings_file_path)

df_train_aesthetics_splitted = df_train_embeddings[df_train_embeddings['0'].isin(df_train_ground_truth_splitted.video.apply(lambda x : x.split('.')[0]))]
df_test_aesthetics_splitted = df_test_embeddings[df_test_embeddings['0'].isin(df_test_ground_truth_splitted.video.apply(lambda x : x.split('.')[0]))]


print (df_train_aesthetics_splitted)

df_train_aesthetics_splitted.loc[:, '1'] = df_train_aesthetics_splitted['1'].apply(lambda x: x.split('_')[1].split('.')[0])
df_train_aesthetics_splitted['1'] = df_train_aesthetics_splitted['1'].astype(int)
df_test_aesthetics_splitted.loc[:, '1'] = df_test_aesthetics_splitted['1'].apply(lambda x: x.split('_')[1].split('.')[0])
df_test_aesthetics_splitted['1'] = df_test_aesthetics_splitted['1'].astype(int)

df_train_aesthetics_splitted = df_train_aesthetics_splitted.sort_values(['0', '1'], ascending=[True, True])
df_test_aesthetics_splitted = df_test_aesthetics_splitted.sort_values(['0', '1'], ascending=[True, True])

print (df_train_aesthetics_splitted)

# Save data.
df_train_aesthetics_splitted.to_csv(train_embeddings_splitted_file_path, index=False)
df_test_aesthetics_splitted.to_csv(test_embeddings_splitted_file_path, index=False)
