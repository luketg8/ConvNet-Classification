import Tkinter as tk
import train_model as training
import tflearn
from tflearn.layers.core import input_data

import initialise_model

def load_network():
	model = tflearn.DNN(initialise_model.create_network('adam'), tensorboard_verbose=0)
	return model

menu = tk.Tk()
tk.Button(menu, text="train", command=training.train).pack();
tk.Button(menu, text="load model", command=lambda: training.load(load_network())).pack();
menu.mainloop()

if __name__ == "__load_network__":
	load_network()
