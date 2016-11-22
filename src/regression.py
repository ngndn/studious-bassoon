import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from gradient import gradient_descent, linear, sigmoid


class LinearRegressionCustom(object):

    def __init__(self):
        self._theta = None

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

    def fit(self, x, y, alpha=0.001):
        self._theta = gradient_descent(x, y, alpha, linear)

    def predict(self, x):
        return x @ self._theta


class PolynomialRegression(object):

    def __init__(self, degree):
        self._degree = degree
        self._model = PolynomialFeatures(degree=degree)

    def __repr__(self):
        return '{}<{}>'.format(self.__class__.__name__, self._degree)

    def fit(self, x, y):
        self._model.fit(x, y)
        x = self._model.transform(x)
        self._theta = np.linalg.inv(x.T @ x) @ x.T @ y

    def predict(self, x):
        x = self._model.transform(x)
        return x @ self._theta
