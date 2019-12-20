import pandas as pd
import numpy as np

def push_annotations(df_embeddings, df_annotations):
    list_video_names = df_annotations['video'].values
    for video_name in list_video_names:
        video_name_without_ext = video_name.split('.')[0]
        df_embeddings.loc[df_embeddings['0'] == video_name_without_ext, 'long_term'] = df_annotations.loc[df_annotations['video'] == video_name, 'long-term_memorability'].iloc[0]
        df_embeddings.loc[df_embeddings['0'] == video_name_without_ext, 'short_term'] = df_annotations.loc[df_annotations['video'] == video_name, 'short-term_memorability'].iloc[0]

    return df_embeddings

# Path to file
train_embeddings_file_path = '../../data/corpus/devset/dev-set/train_saliency_embeddings_splitted.csv'
test_embeddings_file_path = '../../data/corpus/devset/dev-set/test_saliency_embeddings_splitted.csv'
train_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/train_ground-truth_dev-set_splitted.csv'
test_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/test_ground-truth_dev-set_splitted.csv'

# Save paths
train_embeddings_annotations_file_path = '../../data/corpus/devset/dev-set/train_saliency_embeddings_splitted_annotations.csv'
test_embeddings_annotations_file_path = '../../data/corpus/devset/dev-set/test_saliency_embeddings_splitted_annotations.csv'


df_train_embeddings = pd.read_csv(train_embeddings_file_path)
df_test_embeddings = pd.read_csv(test_embeddings_file_path)
df_train_ground_truth = pd.read_csv(train_ground_truth_file_path)
df_test_ground_truth = pd.read_csv(test_ground_truth_file_path)

df_train_embeddings = push_annotations(df_train_embeddings, df_train_ground_truth)
df_test_embeddings = push_annotations(df_test_embeddings, df_test_ground_truth)

df_train_embeddings = df_train_embeddings.sort_values(['0', '1'], ascending=[True, True])
df_test_embeddings = df_test_embeddings.sort_values(['0', '1'], ascending=[True, True])

df_train_embeddings.to_csv(train_embeddings_annotations_file_path, index=False)
df_test_embeddings.to_csv(test_embeddings_annotations_file_path, index=False)