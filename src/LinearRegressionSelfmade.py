from gradient import gradient_descent, linear


class LinearRegressionSelfmade():
    def __init__(self):
        self._theta = None

    def fit(self, x, y, alpha=0.01):
        self._theta = gradient_descent(x, y, alpha, linear)

    def predict(self, x):
        return x @ self._theta
