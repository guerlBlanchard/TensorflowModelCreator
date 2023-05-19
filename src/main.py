

import os

import tensorflow as tf

import numpy as np
import pandas as pd

class algorithm:
    model: tf.keras.Model
    def __init__(self, model):
        self.model_path = '../saved_model/' + model
        print(self.model_path)
        if (os.path.exists(self.model_path)):
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            print("The given path does not exit, would you like to create a new model? (y/n)")
            if (input(">> ") == "y"):
                self.setModel()
            else:
                exit

    def saveModel(self):
        self.model.save(self.model_path)

    
    def setModel(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=6),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def handleMissing(self, dataSet: pd.DataFrame) -> pd.DataFrame:
        print("The given file has missing values, please enter ow you wish to handle them:")
        print("\t1 - Drop the values")
        print("\t2 - Replace with the most frequent values")
        print("\t3 - Replace with the most average values")
        print("\t4 - Show missing values amount")
        option = input(">> ")
        if option == '1':
            return (dataSet.dropna())
        elif option == '2':
            return (dataSet.fillna(dataSet.mode().iloc[0]))
        elif option == '3':
            return (dataSet.fillna(dataSet.mean()))
        elif option == '4':
            print(dataSet.isna().sum())
            return(self.handleMissing(dataSet))
        elif option == 'EXIT':
            exit()
        else:
            print('\033[93m' + "Incorect value, please enter a correct value or type EXIT to stop" + '\033[0m')
            return(self.handleMissing(dataSet))
        
    def encode(self, dataSet: pd.DataFrame) -> pd.DataFrame:
        # for column in dataSet.columns:
        #     if dataSet[column].dtype != 'int64' and dataSet[column].dtype != 'float64':
        dataSet['Sex'] = dataSet['Sex'].replace({'male': 0, 'female': 1})
        dataSet['Embarked'] = dataSet['Embarked'].replace({'C': 0, 'Q': 1, 'S': 2})
        return dataSet

    def train(self, trainingSet):
        trainingData = self.handleMissing(pd.read_csv(trainingSet))
        trainingData = self.encode(trainingData)
        print(trainingData)
        inputData = trainingData[["Pclass", "Age", "Sex", "SibSp", "Parch", "Embarked"]]
        self.model.fit(inputData, trainingData['Survived'], epochs=100, batch_size=32)



if __name__ == "__main__":
    titanic = algorithm('model1.h5')
    titanic.train('../Datasets/train.csv')
    titanic.saveModel()
