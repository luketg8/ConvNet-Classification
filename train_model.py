"""
References:
    Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
Links:
    [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.datasets import cifar10

import initialise_model

def train():
	# Data loading and preprocessing
	(X, Y), (X_test, Y_test) = cifar10.load_data()
	X, Y = shuffle(X, Y)
	Y = to_categorical(Y, 10)
	Y_test = to_categorical(Y_test, 10)

	print("testing")

	# Train using classifier
	model = tflearn.DNN(initialise_model.create_network('adam'), tensorboard_verbose=0, checkpoint_path='cifar10.tfl.ckpt')

	#train the algorithm and take checkpoints every epoch
	model.fit(X, Y, n_epoch=1, shuffle=True, validation_set=(X_test, Y_test), snapshot_epoch=True,
	          show_metric=True, batch_size=122, run_id='cifar10_cnn')

	#export the model
	model.save('cifar.tflearn')

def load(model, image):
	model.load('cifar.tflearn')
	print(model.predict([image]))

if __name__ == "__load__":
	load()

if __name__ == "__train__":
	train()