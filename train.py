import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np

# Dataset
# Contains 28x28 pixel grayscale images of handwritten digits (0-9)
mnist = tf.keras.datasets.mnist

# Load the dataset
# Split the dataset into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values from 0-255 (RGB) to 0-1 (Grayscale)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Set the image size to 28x28 pixels
# Reshape the training and testing sets to include a channel dimension (1 for grayscale)
# -1 means the number of samples is inferred from the data
IMG_SIZE = 28
x_trainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Creating the neural network (Sequential API instead of functional)
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = x_trainr.shape[1:])) # 2D convulational layer with 64 filters, 3x3 size
model.add(Activation("relu")) # Activation function to make the model non-linear and setting all values < 0 to 0
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(32))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation("softmax"))

# Compile and train the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_trainr, y_train, epochs=5, validation_split=0.3)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_testr, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# Save the model
model.save("digit_recognizer_model.h5")