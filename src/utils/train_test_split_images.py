# We are going to split the images folders in train and test.
import pandas as pd
import os

# Captions output file paths
train_captions_file_path = '../../data/corpus/devset/dev-set/train_dev-set_video-captions-cleanup.csv'
test_captions_file_path = '../../data/corpus/devset/dev-set/test_dev-set_video-captions-cleanup.csv'

# Images folder
images_folder_folder_path = '/media/marcoscollado/pgth06a/saliency/'
train_images_folder_path = '/media/marcoscollado/pgth06a/saliency/train/'
test_images_folder_path = '/media/marcoscollado/pgth06a/saliency/test/'

df_train_ground_truth = pd.read_csv(train_captions_file_path)
df_test_ground_truth = pd.read_csv(test_captions_file_path)

for index, row in df_train_ground_truth.iterrows():
    video_name = row['id']
    video_name = video_name.split('.')[0]
    print('Moving:' + video_name)
    os.renames(os.path.join(images_folder_folder_path, video_name), os.path.join(train_images_folder_path, video_name))

for index, row in df_test_ground_truth.iterrows():
    video_name = row['id']
    video_name = video_name.split('.')[0]
    print('Moving:' + video_name)
    os.renames(os.path.join(images_folder_folder_path, video_name), os.path.join(test_images_folder_path, video_name))

