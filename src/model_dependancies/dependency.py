import os
from typing import Tuple
try:
    import readline
except ImportError:
    import pyreadline as readline

import pandas as pd

class DependencyModel:
    datasetInput: pd.DataFrame
    datasetTarget: pd.DataFrame

    def __init__(self):
        pass
    
    
    #default methods
    def inputCommand(self, Autocomplete : 'list[str]' = []) -> str:
        if Autocomplete == []:
            return (input(">> "))
        def completer(text, state):
            options = [cmd for cmd in Autocomplete if cmd.startswith(text)]
            if state < len(options):
                return options[state]
            else:
                return None
        readline.parse_and_bind("tab: complete")
        readline.set_completer(completer)
        return (input(">> "))


    # default dataset Methods
    def setDataset(self, datasetPath:str):
        dataSet = self.handleMissing(pd.read_csv(datasetPath))
        self.datasetInput = self.encode(self.selectInput(dataSet))
        dataSet = dataSet.drop(self.datasetInput.columns, axis=1)
        self.datasetTarget = self.selectTarget(dataSet)

    def selectInput(self, dataSet: pd.DataFrame) -> pd.DataFrame:
        print("Please input the names of the columns you wish to use as an input separated by a semicolon ( ; )")
        print(dataSet.columns.values.tolist())
        columns = self.inputCommand(dataSet.columns.values.tolist())
        self.columns = columns.split(';')
        return (dataSet[self.columns])
    
    def selectTarget(self, dataSet: pd.DataFrame) -> pd.DataFrame:
        print("Please input the names of the column you wish to use as a Target value")
        print(dataSet.columns.values.tolist())
        column = self.inputCommand(dataSet.columns.values.tolist())
        return (dataSet[[column]])

    #model methods
    def build_model(self):
        pass

    def compile_model(self):
        pass

    def train_model(self):
        pass

    def evaluate_model(self):
        pass

