import os

import tensorflow as tf

import numpy as np
import pandas as pd

class algorithm:
    model: tf.keras.Model
    def __init__(self):
        print("Please input the name of the model you wish to create/load")
        model_name = input(">> ")
        self.model_path = "../saved_model/" + model_name
        print(self.model_path)
        if (os.path.exists(self.model_path)):
            print("This model already exits, do you wish to overwride it (O) or load it (L)")
            if (input(">> ") == 'L'):
                self.model = tf.keras.models.load_model(self.model_path)
                print("Previous model has been loaded")
            else:
                self.setModel()
        else:
            self.setModel()


    def saveModel(self):
        self.model.save(self.model_path)

    
    def setModel(self):
        print("Creating new model")
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=(6,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    def selectInput(self, dataSet: pd.DataFrame) -> pd.DataFrame:
        print("Please input the names of the columns you wish to use as an input separated by a semicolon ( ; )")
        print(dataSet.columns.values.tolist())
        columns = input(">> ")
        return (dataSet[columns.split(';')])


    def handleMissing(self, dataSet: pd.DataFrame) -> pd.DataFrame:
        print("The given file has missing values, please enter ow you wish to handle them:")
        print("\t1 - Drop the values")
        print("\t2 - Replace with the most frequent values")
        print("\t3 - Replace with the most average values")
        print("\t4 - Show missing values amount")
        option = input(">> ")
        if option == "1":
            return (dataSet.dropna())
        elif option == "2":
            return (dataSet.fillna(dataSet.mode().iloc[0]))
        elif option == "3":
            return (dataSet.fillna(dataSet.mean()))
        elif option == "4":
            print(dataSet.isna().sum())
            return(self.handleMissing(dataSet))
        elif option == "EXIT":
            exit()
        else:
            print("\033[93m" + "Incorect value, please enter a correct value or type EXIT to stop" + "\033[0m")
            return(self.handleMissing(dataSet))
        
    def encode(self, dataSet: pd.DataFrame) -> pd.DataFrame:
        for column in dataSet.columns:
            if dataSet[column].dtype != "int64" and dataSet[column].dtype != "float64":
                if (sum(dataSet[column].unique()) / sum(dataSet[column]) >= 50):
                    print("The column {} has too many varying values, please encode them manualy or augment the training data".format(column))
                    exit
                
        # dataSet["Sex"] = dataSet["Sex"].replace({"male": 0, "female": 1})
        # dataSet["Embarked"] = dataSet["Embarked"].replace({"C": 0, "Q": 1, "S": 2})
        return dataSet

    def train(self, trainingSet):
        dataSet = pd.read_csv(trainingSet)
        inputData = self.selectInput(dataSet)
        inputData = self.handleMissing(inputData)
        inputData = self.encode(inputData)
        print(inputData)
        self.model.fit(inputData, dataSet["Survived"], epochs=1000, batch_size=32)

    def predict(self, testingSet):
        testingData = self.handleMissing(pd.read_csv(testingSet))
        testingData = self.encode(testingData)
        inputData = testingData[["Pclass", "Age", "Sex", "SibSp", "Parch", "Embarked"]]
        predictions = self.model.predict(inputData)
        print(f"Estimated test probability: {np.sum(predictions) / len(predictions):.4f}")


if __name__ == "__main__":
    titanic = algorithm()
    titanic.train("../Datasets/train.csv")
    titanic.predict("../Datasets/test.csv")
    titanic.saveModel()
