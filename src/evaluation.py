import logging
import sys

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, LeaveOneOut

from classification import KNeighborsClassifier, RandomForestClassifier
from regression import LinearRegression, PolynomialRegression

logger = logging.getLogger()


class Baseline(object):

    def __init__(self):
        self._model = None

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

    def fit(self, x, y):
        self._model = np.mean(y)

    def predict(self, x):
        return np.full(x.shape[0], self._model)


def _load(task_name):
    if task_name == 'regression':
        outcome = 'vote'
    elif task_name == 'classification':
        outcome = 'rating'
    else:
        raise NotImplementedError

    data_train = pd.read_csv(
        '../data/{}_dataset_training.csv'.format(task_name),
        index_col=0
    )
    x_train = data_train.drop(outcome, axis=1).as_matrix()
    y_train = getattr(data_train, outcome).as_matrix()
    x_test = pd.read_csv(
        '../data/{}_dataset_testing.csv'.format(task_name),
        index_col=0
    )
    x_test = x_test.as_matrix()
    y_test = pd.read_csv(
        '../data/{}_dataset_testing_solution.csv'.format(task_name),
        index_col=0
    )
    y_test = getattr(y_test, outcome).as_matrix()
    return data_train, x_train, y_train, x_test, y_test


def _plot(data, filename, **params):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('ticks')

    xlabel = params.pop('xlabel', r'top $k$ features')
    ylabel = params.pop('ylabel', r'score')
    title = params.pop('title', r'Score for Top $k$ Features')

    data.plot(**params)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontweight='bold', y=1.02)
    sns.despine(offset=0, trim=False)
    plt.savefig('../doc/figures/{}'.format(filename), dpi=300)


def _score(x, y, model, score_func, cv=10, round=False):
    """
    Return model MSE scores for x (observations), y (outcomes) and a specified
    model.  Model must have methods `fit` and `predict`.

    Parameters
    ----------
    x : numpy.matrix
    y : numpy.matrix
    model : classifier / predictor
    score_func : callable scoring function
    cv : int
        Number of folds for cross validation
    round : bool
        Whether or not the score_func expects labels, or probabilities for each
        prediction.


    Returns
    -------
    score : float

    """
    if isinstance(cv, int):
        cv = KFold(n_splits=10)
    elif isinstance(cv, str) and cv == 'loo':
        cv = LeaveOneOut()

    scores = []
    for train_index, test_index in cv.split(x):
        x_train, x_test = x[train_index, :], x[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred = np.round(y_pred) if round else y_pred
        scores.append(score_func(y_test, y_pred))

    return np.mean(scores)


def log_bernoulli_loss(true, pred):
    # Remove total values that result in infinity.
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    return -np.mean(true * np.log(pred) + (1 - true) * np.log(1 - pred))


def mean_squared_error(true, pred):
    return np.mean((pred - true) ** 2)


def evaluate(model, x, index, name, round=False, negative=False):
    """
    Evaluate predictions using input X and MODEL.  Optionally round values to
    0 decimals.  Always remove negative values when predicting count problems.

    Save CSV for submission.

    """
    y = model.predict(x)
    if round:
        y = np.around(y)

    if not negative:
        y[y < 0] = 0

    y = pd.Series(y, index=index, name='rating')
    y.to_csv('../{}_submission.csv'.format(name), header=True)


def run(models, data, score_func, name=None, submit=False, round=False,
        training=True):
    """
    Run model evaluation for MODELS and test the best fit.  Additionally, save
    CSV of test predictions for submission to Kaggle.

    Parameters
    ----------
    models : [model]
        Family of similar models.
    data : [x_train, y_train, x_test, y_test]
    name : str
    submit : bool
    round : bool
    training : bool
        Evaluate model using cross-validation on training set, or train model
        and report performance on test set.

    """
    results = pd.Series()
    for (model, params) in models:
        model_name = model.__class__.__name__
        if params is not None:
            model_name = '{}<{}>'.format(model_name, params)

        if training:
            logger.info('Validating {}…'.format(model_name))
            score = _score(data[0], data[1], model, score_func, round=round)
            logger.info('Mean Score (CV): {}'.format(score))
        else:
            logger.info('Testing {}…'.format(model_name))
            model.fit(*data[:2])
            y_pred = model.predict(data[2])
            y_pred = np.round(y_pred) if round else y_pred
            score = score_func(data[3], y_pred)
            logger.info('Score (test)  : {}'.format(score))

        results.set_value(model_name, score)

    # Kaggel:
    if submit:
        if name is None:
            name = 'linear'

        evaluate(data[2], model, name)

    return results


def run_regression(num_features=None, training=True):
    # Prepare data
    data_train, x_train, y_train, x_test, y_test = _load('regression')

    # Select all features by default
    if num_features is None:
        num_features = x_train.shape[1]

    # Feature selection for regression on source data
    results = pd.DataFrame()
    for k in range(1, num_features + 1):
        logger.info('Selecting {} features for regression...\n'.format(k))
        fs = SelectKBest(score_func=f_regression, k=k).fit(x_train, y_train)
        for score, feature in sorted(zip(fs.scores_, data_train.columns))[-k:]:
            logger.info('{} ({:0.2f})'.format(feature, score))

        # Select features
        xtr = fs.transform(x_train)
        xte = fs.transform(x_test)

        # Add bias term
        # xtr = np.append(np.ones((xtr.shape[0], 1)), xtr, axis=1)
        # xte = np.append(np.ones((xte.shape[0], 1)), xte, axis=1)

        # Run model comparison
        logger.info('\nScoring models...\n')
        models = [
            (Baseline(), None),
            (LinearRegression(), None),
            (PolynomialRegression(2), 2),
        ]
        data = [xtr, y_train, xte, y_test]
        scores = run(
            models,
            data,
            score_func=mean_squared_error,
            round=False,
            training=training
        )
        logger.info('\n')  # Done
        scores.name = k
        results = pd.concat([results, scores], axis=1)

    return results.transpose()


def run_classification(num_features=None, training=True, score_func=None):
    # Prepare data
    data_train, x_train, y_train, x_test, y_test = _load('classification')

    # Select all features by default
    if num_features is None:
        num_features = x_train.shape[1]

    # Select default scoring function
    if score_func is None:
        score_func = 'accuracy'

    if score_func == 'accuracy':
        score_func = accuracy_score
        round = True
    elif score_func == 'f1_score':
        score_func = f1_score
        round = True
    elif score_func == 'log_loss':
        score_func = log_bernoulli_loss
        round = False
    else:
        NotImplementedError

    # Feature selection for regression on source data
    results = pd.DataFrame()
    for k in range(1, num_features + 1):
        # Feature selection for classification on source data
        logger.info('Selecting {} features for classification...\n'.format(k))
        fs = SelectKBest(score_func=chi2, k=k).fit(x_train, y_train)
        for score, feature in sorted(zip(fs.scores_, data_train.columns))[-k:]:
            logger.info('{} ({:0.2f})'.format(feature, score))

        # Select features
        xtr = fs.transform(x_train)
        xte = fs.transform(x_test)

        # Run model comparison
        logger.info('\nScoring models...\n')
        models = [
            (Baseline(), None),
            (KNeighborsClassifier(1), 1),
            (KNeighborsClassifier(5), 5),
            (RandomForestClassifier(n_estimators=10), 10),
            (RandomForestClassifier(n_estimators=50), 50),
        ]
        data = [xtr, y_train, xte, y_test]
        scores = run(
            models,
            data,
            score_func=score_func,
            round=round,
            training=training
        )
        logger.info('\n')  # Done
        scores.name = k
        results = pd.concat([results, scores], axis=1)

    return results.transpose()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'reg':
            run_regression()
        elif sys.argv[1] == 'cls':
            run_classification()

    else:
        # run_regression()
        # run_classification()
        pass
