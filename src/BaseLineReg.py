import numpy as np


class BaseLineReg:
    def __init__(self):
        self._model = None

    def fit(self, x, y):
        self._model = np.mean(y)

    def predict(self, x):
        return self._model

