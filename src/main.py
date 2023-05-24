import os
import readline

import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class algorithm:
    model: tf.keras.Model 
    modelPath:str = "../saved_model/"
    # columns: list[str]

    def __init__(self, savedModel:str=None):
        if savedModel is None:
            return
        self.modelPath += savedModel
        if (os.path.exists(self.modelPath)):
            self.model = tf.keras.models.load_model(self.modelPath)
            print("Previous model has been loaded")

    def __str__(self) -> str:
        return (self.model.summary())

    def saveModel(self):
        print("Do you wish to save this model? Yes/[Any]")
        if (input(">> ") == 'Yes'):
            print("Enter a savefile name")
            savefile = input(">> ")
            if os.path.exists(self.modelPath + savefile + ".h5"):
                print("This savefile name is already taken. Do you wish to overide this savefile? Yes/[Any]")
                if (input(">> ") == 'Yes'):
                    self.model.save(self.modelPath + savefile + ".h5")
            else:
                self.model.save(self.modelPath + savefile + ".h5")
    
    def setModel(self):
        print("Creating new model")
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=(6,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    def selectInput(self, dataSet: pd.DataFrame) -> pd.DataFrame:
        print("Please input the names of the columns you wish to use as an input separated by a semicolon ( ; )")
        print(dataSet.columns.values.tolist())
        columns = input(">> ")
        self.columns = columns.split(';')
        return (dataSet[self.columns])
    
    def selectTarget(self, dataSet: pd.DataFrame) -> pd.DataFrame:
        print("Please input the names of the column you wish to use as a Target value")
        print(dataSet.columns.values.tolist())
        column = input(">> ")
        return (dataSet[column])

    def handleMissing(self, dataSet: pd.DataFrame) -> pd.DataFrame:
        for column in dataSet.columns:
            if dataSet[column].isna().sum() != 0:
                print("The column named {} has missing {} values, please enter ow you wish to handle them:".format(column, dataSet[column].isna().sum()))
                print("\t1 - Drop the values")
                print("\t2 - Replace with the most frequent values")
                print("\t3 - Replace with the most average values")
                print("\t4 - Ignore")
                option = input(">> ")
                if option == "1":
                    dataSet.dropna(subset=[column], inplace=True)
                elif option == "2":
                    dataSet[column].fillna(dataSet[column].mean(), inplace=True)
                elif option == "3":
                    dataSet[column].fillna(dataSet[column].mode()[0], inplace=True)
                elif option == "4":
                    continue
                elif option == "EXIT":
                    exit()
                else:
                    print("\033[93m" + "Incorect value, please enter a correct value or type EXIT to stop" + "\033[0m")
        return (dataSet)
        
    def encode(self, dataSet: pd.DataFrame) -> pd.DataFrame:
        for column in dataSet.columns:
            if dataSet[column].dtype != "int64" and dataSet[column].dtype != "float64":
                print("Encoding the {} column".format(column))
                if (len(dataSet[column].unique()) / len(dataSet[column]) * 100 >= 50):
                    print("The column {} has too many varying values, please encode them manualy or augment the training data".format(column))
                    exit
                encoding_dict = {value: index for index, value in enumerate(dataSet[column].unique())}
                dataSet[column] = dataSet[column].map(encoding_dict)
        return dataSet

    def train(self, trainingSet):
        dataSet = self.handleMissing(pd.read_csv(trainingSet))
        inputData = self.encode(self.selectInput(dataSet))
        dataSet.drop(inputData.columns, axis=1)
        targetData = self.selectTarget(dataSet)
        print(inputData)
        print(targetData)
        input("Press ENTER to create your model")
        self.setModel()
        input("Press ENTER to train your model")
        trainX, validX, trainY, validY = train_test_split(inputData, targetData, test_size=0.1, random_state=42)
        self.model.fit(trainX, trainY, epochs=1000, batch_size=32, validation_data=(validX, validY))
        print("Overall Evaluation:")
        loss, acc = self.model.evaluate(validX, validY)
        print("Loss: {} || Accuracy: {}".format(loss, acc))

    def predict(self, testingSet):
        dataSet = pd.read_csv(testingSet)
        if self.columns:
            inputData = self.encode(self.handleMissing(dataSet[self.columns]))
        else:
            inputData = self.encode(self.handleMissing(self.selectInput(dataSet[self.columns])))
        predictions = self.model.predict(inputData)
        print(f"Estimated test probability: {np.sum(predictions) / len(predictions):.4f}")


if __name__ == "__main__":
    titanic = algorithm()
    titanic.train("../Datasets/train.csv")
    titanic.predict("../Datasets/test.csv")
    titanic.saveModel()
