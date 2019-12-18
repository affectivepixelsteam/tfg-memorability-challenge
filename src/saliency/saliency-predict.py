import pandas as pd
import numpy as np

from keras.models import model_from_json
from PIL import Image
from skimage import transform
import matplotlib.pyplot as plt

image_path = '/mnt/pgth06a/saliency/'
model_path = '../../models/saliency/saliency_model.json'
weight_path = '../../models/saliency/saliency_weight.h5'

# load json and create model
with open(model_path, 'r') as model_file:
    model = model_file.read()

model = model_from_json(model)

# load weights into new model
model.load_weights(weight_path)
print("Loaded model from disk")

# Load image
def process_image(np_image):
   np_image = np.array(np_image).astype('float32')/255
   np_image = np_image.T
   np_image = transform.resize(np_image, (384, 224, 1))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

input_image = Image.open(image_path + "test/video212/0020.png")
pixels = process_image(input_image)

# Show that normalization went well.
print("Shape ", pixels.shape)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

# Predict
ypred = model.predict(pixels)

# Show images
f, axarr = plt.subplots(1,2)

axarr[0].imshow(ypred[0,:,:,0], cmap="gray")
axarr[1].imshow(pixels[0,:,:,0], cmap="gray")

plt.show()

