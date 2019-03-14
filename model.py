import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf
import keras


def init():
	loaded_model = keras.models.load_model('scribber.h5')
	print("Loaded Model from disk")
	graph = tf.get_default_graph()

	return loaded_model,graph