import numpy as np

from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier


def euclidean_distance(a, b):
    return np.dot(a - b, a - b)


class KNN(object):
    def __init__(self, k=1):
        self._model_x = None
        self._model_y = None
        self._k = k
        self._num_obs = None

    def __repr__(self):
        return '{}<{}>'.format(self.__class__.__name__, self._k)

    def fit(self, x, y):
        self._model_x = x.copy()
        self._model_y = y.copy()
        self._num_obs = self._model_x.shape[0]

    def predict(self, x):
        pred = []
        for i in range(x.shape[0]):
            distance = []
            for j in range(self._num_obs):
                distance.append(euclidean_distance(x[i], self._model_x[j]))

            top_k_index = np.argpartition(distance, -self._k)[-self._k:]
            top_k_label = self._model_y[top_k_index]
            pred.append(np.mean(top_k_label))

        return np.array(pred)
