import os
from keras.layers import Dense
from keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2

def mobilenet(num_classes, load_from=None):
	"""Pretrained mobilenetV2 model w/o last layer.
		Args:
			num_classes (int): Size of output layer.
			load_from (str): Path to loadable weights.
		Return:
			net: Ready-to-plug pretrained model.
	"""
	basenet = MobileNetV2(
		input_shape=(224, 224, 3),
		weights='imagenet',
		include_top=False,
		pooling='avg')
	basenet_output = basenet.output
	prediction = Dense(num_classes,
		activation='sigmoid')(basenet_output)
	net = Model(inputs=basenet.input,
		outputs=prediction)
	if load_from:
		net.load_weights(load_from)
	return net