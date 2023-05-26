import os
import readline

import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class algorithm:
    # model definition
    model: tf.keras.Model 
    modelPath:str = "../saved_model/"
    modelTrainHistory: tf.keras.callbacks.History

    # dataset
    datasetInput: pd.DataFrame
    datasetTarget: pd.DataFrame

    # model recommendations
    inputLayerUnitsRecommendation: int = 0

    # columns: list[str]

    def __init__(self, savedModel:str=None):
        if savedModel is None:
            return
        self.modelPath += savedModel
        if (os.path.exists(self.modelPath)):
            self.model = tf.keras.models.load_model(self.modelPath)
            print("Previous model has been loaded")

    def __str__(self) -> str:
        self.model.summary()
        self.plotHistory()
        return ("")

    def plotHistory(self):
        plt.figure(figsize=(20, 6))
        plt.subplot(1, 3, 1)
        plt.plot(self.modelTrainHistory.epoch, self.modelTrainHistory.history['accuracy'], label='Training Accuracy')
        plt.plot(self.modelTrainHistory.epoch, self.modelTrainHistory.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Epochs vs Accuracy')
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.plot(self.modelTrainHistory.history['accuracy'], self.modelTrainHistory.history['val_accuracy'], label='Training Accuracy')
        plt.xlabel('acc')
        plt.ylabel('val acc')
        plt.title('Validation vs Train Accuracy')
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(self.modelTrainHistory.epoch, self.modelTrainHistory.history['loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.tight_layout()
        plt.show()

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
        print("Please input the amount of units you wish you input layer has (Recommended: {})".format(self.inputLayerUnitsRecommendation))
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation="relu", input_shape=(self.datasetInput.shape[1], )),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(self.datasetTarget.shape[1], activation="softmax")
        ])
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        print("Model has been created")

    def setDataset(self, datasetPath:str):
        dataSet = self.handleMissing(pd.read_csv(datasetPath))
        self.datasetInput = self.encode(self.selectInput(dataSet))
        dataSet.drop(self.datasetInput.columns, axis=1)
        self.datasetTarget = self.selectTarget(dataSet)

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
        if dataSet[column].dtype != "int64" and dataSet[column].dtype != "float64":
            return pd.get_dummies(dataSet[column])
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
                if (len(dataSet[column].unique()) / len(dataSet[column]) * 100 >= 10):
                    print("The column {} has too many varying values, please encode them manualy or augment the training data".format(column))
                    continue
                encoding_dict = {value: index for index, value in enumerate(dataSet[column].unique())}
                dataSet[column] = dataSet[column].map(encoding_dict)
                self.inputLayerUnitsRecommendation += len(dataSet[column].unique())
            elif dataSet[column].dtype == "int64" or dataSet[column].dtype == "float64":
                if (len(dataSet[column].unique()) / len(dataSet[column]) * 100 >= 10):
                    dataSet[column] = (dataSet[column] - dataSet[column].min()) / (dataSet[column].max() - dataSet[column].min())
                    self.inputLayerUnitsRecommendation += 1
                else:
                    self.inputLayerUnitsRecommendation += len(dataSet[column].unique())
            else:
                print("Column {} if a {} type that has yet to be handled".format(column, dataSet[column].dtype))
                self.inputLayerUnitsRecommendation += 10
        return dataSet

    def train(self):
        print(self.datasetInput)
        print(self.datasetTarget)
        input("Press ENTER to create your model")
        self.setModel()
        input("Press ENTER to train your model")
        trainX, validX, trainY, validY = train_test_split(self.datasetInput, self.datasetTarget, test_size=0.1, random_state=42)
        self.modelTrainHistory = self.model.fit(trainX, trainY, epochs=100, batch_size=32, validation_data=(validX, validY))
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
    iris = algorithm()
    iris.setDataset("../Datasets/Iris/Iris.csv")
    iris.train()
    print(iris)
    # titanic.predict("../Datasets/test.csv")
    # titanic.saveModel()
