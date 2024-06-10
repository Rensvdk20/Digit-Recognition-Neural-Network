# Digit Recognition

This python code makes use of a Convolutional Neural Network (CNN). The system is trained to recognize handwritten digits (0-9) from the MNIST dataset. The code includes a script to train the model `train.py` and a GUI `main.py` for live digit prediction.

## Requirements

To run the code in this repository, you need to have the following libraries installed:

- TensorFlow
- NumPy
- Pillow

You can install these dependencies using the following command:

```bash
pip install tensorflow numpy pillow
```

## Setup
- Clone this repository:

    ```bash
    git clone https://github.com/Rensvdk20/Digit-Recognition-Neural-Network
    ```

- Run the `train.py` script to train the model:

    ```bash
    python train.py
    ```

   This script will load the MNIST dataset, normalize the images, define and train the CNN, and save the trained model as `digit_recognizer_model.h5`

## Usage

- Run the `main.py` script to start the GUI application:

    ```bash
    python main.py
    ```

   This script will launch a GUI where you can draw a digit and see a live prediction of what digit you drew.
