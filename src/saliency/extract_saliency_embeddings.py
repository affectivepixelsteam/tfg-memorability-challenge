import pandas as pd
import numpy as np
import os

from keras.models import model_from_json
from keras.models import Model
from PIL import Image
from skimage import transform

train_image_path = '/mnt/pgth06a/saliency/train/'
test_image_path = '/mnt/pgth06a/saliency/test/'
train_embeddings_output = '../../data/corpus/devset/dev-set/train_saliency_embeddings_splitted.csv'
test_embeddings_output = '../../data/corpus/devset/dev-set/test_saliency_embeddings_splitted.csv'
model_path = '../../models/saliency/saliency_autoencoder_model.json'
weight_path = '../../models/saliency/saliency_autoencoder_weight.h5'

EMBEDDING_LENGTH = 84


def getVideoNames(isTrain):
    # Path
    if isTrain:
        ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/train_ground-truth_dev-set_splitted.csv'
    else:
        ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/test_ground-truth_dev-set_splitted.csv'
    
    # load dataframes
    df_ground_truth = pd.read_csv(ground_truth_file_path)

    return df_ground_truth['video'].values


def process_image(np_image):
    """
      Preprocess the image to be fed into the model.
    """
    np_image = np.array(np_image).astype('float32')/255
    np_image = np_image.T
    np_image = transform.resize(np_image, (384, 224, 1))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


# load json and create model
with open(model_path, 'r') as model_file:
    model = model_file.read()

complete_model = model_from_json(model)

# load weights into new model
complete_model.load_weights(weight_path)
print("Loaded model from disk")

# Generate new model in which output is the bottleneck layer output so we can extract the embedding
bottleneck_layer = "dense_1"
intermediate_output_model = Model(
    inputs=complete_model.input, outputs=complete_model.get_layer(bottleneck_layer).output)


# Get video names
list_video_names = getVideoNames(False)

# Create array in which we will save the embeddings
data = []

# For each video, lookup for its folder and get from every image in its folder the embedding. Finally, save the embedding in the csv
for video_name in list_video_names:
  video_name = video_name.split('.')[0]
  video_folder = os.path.join(test_image_path, video_name)

  images = [f for f in os.listdir(video_folder) if os.path.isfile(os.path.join(video_folder, f))]

  for image in images:
    input_image = Image.open(os.path.join(video_folder, image))
    pixels = process_image(input_image)

    # Get the embedding
    embedding = intermediate_output_model.predict(pixels)

    this_data = list(embedding[0])

    # Save it in the array
    this_data.insert(0, video_name)
    this_data.insert(1, image)

    data.append(this_data)

# Save data.
df = pd.DataFrame(data)
df.to_csv(test_embeddings_output)