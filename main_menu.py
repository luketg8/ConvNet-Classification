import Tkinter as tk
import train_model as training

menu = tk.Tk()
tk.Button(menu, text="train", command=training.train).pack();
menu.mainloop()
