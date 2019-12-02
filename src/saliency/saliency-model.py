# What we are going to do is to retrieve saliency features for these videos. How?
# First we will use autoencoders with conv networks for retrieve features of each frame in the video
# Then these features of each frame in video we will be loaded to a lstm network in order to retrieve
# the features for each video.

import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Deconvolution2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam
from keras.backend.tensorflow_backend import set_session

# Path
train_videos_folder_path = "/media/marcoscollado/gth10b/saliency"

# Path model and weights
model_save_path = '../../models/saliency/saliency_model.json'
weight_save_path = '../../models/saliency/saliency_weight.h5'

# Path for image
img_file_path = '../../figures/saliency-autoencoder_train_loss.png'

# Hyperparameters
img_size = (384, 224)
input_size = (384, 224, 1)
pooling_size = (2,2)

# Allow GPU growth
# INITIAL CONFIG FOR SHARING GPU RESOURCES
config = tf.ConfigProto()

# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True

# config.gpu_options.per_process_gpu_memory_fraction=0.33
#config.allow_soft_placement=True
#config.log_device_placement=False

sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

train_im = ImageDataGenerator(
               rescale=1./255,
               shear_range=0.2,
               horizontal_flip=False)

def train_images():
    train_generator = train_im.flow_from_directory (
            train_videos_folder_path, 
             target_size=img_size,
             color_mode='rgb',
             batch_size=100,
             shuffle = True,
             class_mode='categorical')
    x =  train_generator
    return x[0][0], x[0][1]

#ENCODER
inp = Input(input_size)
e = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inp)
e = MaxPooling2D(pooling_size)(e)
e = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(e)
e = MaxPooling2D(pooling_size)(e)
e = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(e)
l = Flatten()(e)
l = Dense(5076, activation='softmax')(l)

#DECODER
d = Reshape((94,54,1))(l)
d = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(d)
d = BatchNormalization()(d)
d = Conv2DTranspose(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(d)
decoded = Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(d)

ae = Model(inp, decoded)
ae.summary()


# compile it using adam optimizer
ae.compile(optimizer="adam", loss="mse")

#Train it by providing training images
history = ae.fit(train_images, train_images, epochs=2)

# Save model
with open(model_save_path, 'w+') as save_file:
    save_file.write(ae.to_json())
ae.save_weights(weight_save_path)

print("Saved model")

# Plotting lost
plt.suptitle('Optimizer : adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.plot(history.history['loss'], color='b', label='Training Loss')
plt.legend(loc='upper right')

# Save img
plt.savefig(img_file_path)

plt.show()