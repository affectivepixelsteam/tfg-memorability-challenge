import os
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2
from PIL import Image
from skimage import transform

# Mobilenet data.​
ava_weights = '../../models/aesthetics/fold_0.h5'
num_classes = 1

# Embeddings
train_embeddings_output = '../../data/corpus/devset/dev-set/train_aesthetics_embeddings.csv'
test_embeddings_output = '../../data/corpus/devset/dev-set/test_aesthetics_embeddings.csv'

# Images path
train_image_path = '/mnt/pgth06a/tfg-memorability-challenge/train/'
test_image_path = '/mnt/pgth06a/tfg-memorability-challenge/test/'


def process_image(np_image):
    """
      Preprocess the image to be fed into the model.
    """
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def getVideoNames(isTrain):
    # Path
    if isTrain:
        ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/train_ground-truth_dev-set.csv'
    else:
        ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/test_ground-truth_dev-set.csv'
    
    # load dataframes
    df_ground_truth = pd.read_csv(ground_truth_file_path)

    return df_ground_truth['video'].values

def mobilenet(num_classes, embedding_size=84, load_from=None):
	"""Pretrained mobilenetV2 model w/o last layer.
		Args:
			num_classes (int): Size of output layer.
			embedding_size (int): Embedding layer size [default: 84]
			load_from (str): Path to loadable weights [default: None].
		Return:
			net: Ready-to-plug pretrained model.
	"""
	basenet = MobileNetV2(
		input_shape=(224, 224, 3),
		weights='imagenet',
		include_top=False,
		pooling='avg')
	basenet_output = basenet.output
	prediction = Dense(embedding_size,
		activation='sigmoid')(basenet_output)
	prediction = Dense(num_classes,
		activation='sigmoid')(prediction)
	net = Model(inputs=basenet.input,
		outputs=prediction)
	if load_from:
		net.load_weights(load_from)
	return net


mobilenet = mobilenet(num_classes=num_classes, load_from=ava_weights)

mobilenet.summary()

bottleneck_layer = 'dense_1'
embedding_model = Model(inputs=mobilenet.input,
                        outputs=mobilenet.get_layer(bottleneck_layer).output)


# Get video names
list_video_names = getVideoNames(False)

# Create array in which we will save the embeddings
data = []

# For each video, lookup for its folder and get from every image in its folder the embedding. Finally, save the embedding in the csv
for video_name in list_video_names:
    video_name = video_name.split('.')[0]
    video_folder = os.path.join(test_image_path, video_name)
    images = [f for f in os.listdir(video_folder) if os.path.isfile(
        os.path.join(video_folder, f))]

    for image in images:
        input_image = Image.open(os.path.join(video_folder, image))
        pixels = process_image(input_image)

        # Get the embedding
        embedding = embedding_model.predict(pixels)
        this_data = list(embedding[0])

        # Save it in the array
        this_data.insert(0, video_name)
        this_data.insert(1, image)

        data.append(this_data)

# Save data.
df = pd.DataFrame(data)
df.to_csv(test_embeddings_output)
