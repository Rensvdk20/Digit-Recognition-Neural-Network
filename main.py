import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import tkinter as tk
from tkinter import *
import PIL
from PIL import Image, ImageDraw, ImageOps
import threading

# Load the trained model
model = load_model("digit_recognizer_model.h5")

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Digit Recognizer (CNN)")
        self.canvas = tk.Canvas(self, width=200, height=200, bg="white")
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.button_clear.grid(row=1, column=0, pady=2, padx=2)

        self.prediction_label = tk.Label(self, text="", font=("Helvetica", 24))
        self.prediction_label.grid(row=0, column=0, padx=10, pady=10, sticky=NW)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.image = PIL.Image.new("L", (200, 200), (255))
        self.draw = ImageDraw.Draw(self.image)

        self.drawing = False
        self.prediction_pending = False
        self.timer_id = None

    def paint(self, event):
        self.drawing = True
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
        self.draw.line([x1, y1, x2, y2], fill="black", width=5)

        if not self.prediction_pending:
            self.schedule_prediction()

    def on_release(self, event):
        self.drawing = False
        if self.timer_id:
            self.after_cancel(self.timer_id)
            self.timer_id = None
        self.update_prediction()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, 200, 200), fill=(255))
        self.prediction_label.config(text="")

    def predict_digit(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img = img.convert('L')
        img = np.array(img)
        img = tf.keras.utils.normalize(img, axis=1)
        img = np.array(img).reshape(-1, 28, 28, 1)
        predictions = model.predict(img)
        digit = np.argmax(predictions)
        return digit

    def schedule_prediction(self):
        self.prediction_pending = True
        self.timer_id = self.after(1000, self.run_prediction)

    def run_prediction(self):
        if self.drawing:
            threading.Thread(target=self.async_prediction).start()
            self.schedule_prediction()
        else:
            self.prediction_pending = False

    def async_prediction(self):
        digit = self.predict_digit()
        self.prediction_label.config(text=str(digit))

    def update_prediction(self):
        threading.Thread(target=self.async_prediction).start()
        self.prediction_pending = False

app = App()
app.mainloop()