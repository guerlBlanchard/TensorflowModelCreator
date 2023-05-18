import os

import tensorflow as tf

import numpy as np
import pandas as pd

class algorithm:
    def __init__(self, model):
        self.model_path = '../saved_model/' + model
        if (os.path.exists(self.model_path)):
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            print("The given path does not exit, would you like to create a new model? (y/n)")
            if (input(">> ") == 'y'):
                self.setModel()
            else:
                exit
    
    def __del__(self):
        if (self.model != None):
            self.model.save(self.model_path)
    
    def __str__(self, predictSet):
        return(self.model.predict(predictSet))
    
    def setModel(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=7),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def handleMissing(self, dataSet: pd.DataFrame) -> pd.DataFrame:
        if (dataSet.isnull().sum() == 0):
            return (dataSet)
        print("The given file has missing values, please enter ow you wish to handle them:")
        print("\t1 - Drop the values")
        print("\t2 - Replace with the most frequent values")
        print("\t3 - Replace with the most average values")
        option = input(">> ")
        if option == '1':
            return (dataSet.dropna())
        elif option == '2':
            return (dataSet.fillna(dataSet.mode().iloc[0]))
        elif option == '3':
            return (dataSet.fillna(dataSet.mean()))
        elif option == 'EXIT':
            exit()
        else:
            print('\033[93m' + "Incorect value, please enter a correct value or type EXIT to stop" + '\033[0m')
            return(self.handleMissing(dataSet))

    def train(self, trainingSet):
        trainingData = self.handleMissing(pd.read_csv(trainingSet))

            
