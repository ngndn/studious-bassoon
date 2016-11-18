import numpy as np


def log_bernoulli_loss(predicted, true):
    return (true * np.log(predicted)) + ((1 - true) * np.log(1 - predicted))


def mean_square_error(predicted, true):
    return (predicted - true)**2