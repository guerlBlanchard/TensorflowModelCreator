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


    #model methods
    def build_model(self):
        pass

    def compile_model(self):
        pass

    def train_model(self):
        pass

    def evaluate_model(self):
        pass

