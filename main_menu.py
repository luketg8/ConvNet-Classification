import Tkinter as tk
import train_model as training
import tflearn
from tflearn.layers.core import input_data
import scipy
import numpy as np

import initialise_model

def load_network():
	model = tflearn.DNN(initialise_model.create_network('adam'), tensorboard_verbose=0)
	return model

def evaluate_image(img):
	'''testing
	#cv2.imshow("plane", img)
	#cv2.waitKey()'''

	model = load_network();

	# Scale the image
	img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')

	training.load(model, img)

menu = tk.Tk()
tk.Button(menu, text="train", command=training.train).pack();
tk.Button(menu, text="load model", command=lambda: evaluate_image(scipy.ndimage.imread("plane.jpg", mode="RGB"))).pack();
menu.mainloop()

if __name__ == "__load_network__":
	load_network()
