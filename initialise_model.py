import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

def create_network(optimiser):
	# Real-time data preprocessing
	img_prep = ImagePreprocessing()
	img_prep.add_samplewise_stdnorm()
	img_prep.add_featurewise_stdnorm()

	# Real-time data augmentation to add variance to the data
	img_aug = ImageAugmentation()
	img_aug.add_random_blur (sigma_max=3)
	img_aug.add_random_flip_leftright()
	img_aug.add_random_rotation(max_angle=25.)
	# Convolutional network building
	network = input_data(shape=[None, 32, 32, 3],
	                     data_preprocessing=img_prep,
	                     data_augmentation=img_aug)
	network = conv_2d(network, 32, 3, activation='relu')
	network = max_pool_2d(network, 2)
	network = conv_2d(network, 64, 3, activation='relu')
	network = conv_2d(network, 64, 3, activation='relu')
	network = max_pool_2d(network, 2)
	network = fully_connected(network, 512, activation='relu')
	network = dropout(network, 0.5)
	network = fully_connected(network, 10, activation='softmax')
	network = regression(network, optimizer=optimiser,
	                     loss='categorical_crossentropy',
	                     learning_rate=0.002)

	return network

if __name__ == "__create_network__":
	create_network()