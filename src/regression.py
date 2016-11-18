from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from gradient import gradient_descent, linear
from LinearRegressionBaseline import LinearRegressionBaseline
from LinearRegressionSelfmade import LinearRegressionSelfmade
from PolynomialRegression import PolynomialRegression


class LinearRegressionSelfmade():
    def __init__(self):
        self._theta = None

    def fit(self, x, y, alpha=0.01):
        self._theta = gradient_descent(x, y, alpha, linear)

    def predict(self, x):
        return x @ self._theta


class PolynomialRegression(object):

    def __init__(self, degree):
        self._model = PolynomialFeatures(degree=degree)

    def fit(self, x, y):
        x = self._model.fit_transform(x)
        self._theta = np.linalg.inv(x.T @ x) @ x.T @ y

    def predict(self, x):
        x = self._model.fit_transform(x)
        return x @ self._theta
