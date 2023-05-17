import os

import tensorflow as tf

import numpy as np

class algorithm:
    def __init__(self, model):
        self.model_path = '../saved_model' + model
        if (os.path.exists(self.model_path)):
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense()
            ])
    
    def __del__(self):
        self.model.save(self.model_path)
    
    def __str__(self):
        return(self.model.predict())
