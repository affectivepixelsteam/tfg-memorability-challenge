# So here what we are going to do is to split our dev-set in test and train sets. Since our video set is so large what we
# are actually going to do is to split the ground-truth csv so anytime we want to get our train test just take the videos
# that are contained in the train_ground-truth.csv. Same happens with test_ground-truth.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
import os

# So we are going to get the train and test set from the splitted dataset and the whole dataset.
folder_path = "/mnt/pgth06a/AVA/images-dataset/images"


# Split params
test_size = 0.2
round_state = 123

# Get images
list_images = os.listdir(folder_path)
# list_images = np.random.shuffle(list_images)

# Split 
train_path = os.path.join(folder_path, "train")
test_path = os.path.join(folder_path, "test")
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

train, test = train_test_split(list_images, test_size=test_size, random_state=round_state)

for image in train:
    print('Moving:' + image)
    os.renames(os.path.join(folder_path, image), os.path.join(train_path, image))

for image in test:
    print('Moving:' + image)
    os.renames(os.path.join(folder_path, image), os.path.join(test_path, image))