class KNN():
    def __int__(self, k=1):
        self._model_x = None
        self._model_y = None
        self._k = k

    def fit(self, x, y):
        self._model_x = x.copy()
        self._model_y = y.copy()

    def predict(self, x, y):
        pass