import os

import tensorflow as tf
from tensorflow import keras

import numpy as np

class algorithm:
    def __init__(self, model) -> None:
        self.model_path = '../saved_model' + model
        if (os.path.exists(self.model_path)):
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            self.model = tf.keras.Sequential()