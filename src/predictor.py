from torch import load, tensor
from src.deep_model import Deep
import numpy as np

class Prediction:
    def __init__(self, filename):
        self.model = Deep()
        self.model.load_state_dict(load(filename)['state_dict'])

    def predict(self, X):
        X = tensor(X.astype(np.float32).values)
        self.model.eval()
        return self.model(X)