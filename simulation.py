from numpy import *
from numpy.random import randn, seed
from scipy.stats import bernoulli
import algo
import numpy as np


def gen_data(m=100, d=10, c=.5, v=1):
    """
    Generates synthetic datasets for binary classification from a normal distribution.

    Parameters
    ----------
    m: positive integer
        no. of examples in the positive and negative set
    d: positive integer
        no. of dimensions/features
    c: real number
        center of normal distribution
    v: positive real number
        variance of normal distribution

    Returns
    -------
    X_train: array of shape = [2*m, d]
             Training examples
    y_train: array of shape = [2*m]
            Training labels
    X_test: array of shape = [2*m, d]
             Test examples
    y_train: array of shape = [2*m]
            Test labels
    """

    Xp = v * randn(m, d) + c
    yp = ones(m)
    Xn = v * randn(m, d) - c
    yn = -1 * ones(m)

    X_train = vstack((Xp, Xn))
    y_train = hstack((yp, yn))

    Xp = v * randn(m, d) + c
    yp = ones(m)
    Xn = v * randn(m, d) - c
    yn = -1 * ones(m)

    X_test = vstack((Xp, Xn))
    y_test = hstack((yp, yn))

    return X_train, y_train, X_test, y_test


def sim_t(X, y, z, p=1):
    """
    Generates labels from simulated annotators for the input dataset.

    Parameters
    ----------
    X: array of shape = [n_samples, n_features]
        Input examples
    y: array of shape = [n_samples]
        Input original labels
    z: array of shape = [n_annotators]
        Annotators' expertise
        +1 if annotator labels all examples correctly
        -1 if annotator flips all labels
        0 if annotator generates noisy labels
    p: positive integer
        Noise parameter (see paper)

    Returns
    -------
    Y: array of shape = [n_samples, n_annotators]
        Simulated labels
    """

    m, n = X.shape
    nt = len(z)  # no. of teachers
    # noise_level = 1.# higher values result in low disagreement
    # first train a linear model, obtain scores
    cparam = 2 ** array([-14., -12., -10., -8., -6., -4., -2., -1., 0., 1., 2., 4., 6., 8., 10., 12., 14.])
    model = algo.train(X, y, 1. / cparam)
    f = model.predict(X)
    f = f * 1. / max(abs(f))  # scale to [-10, 10]
    f = 10 * f  # scores closer to +/-10 will get a prob 1
    f = 2 * (1 - (1. / (1 + np.exp(p * -0.25 * abs(f)))))

    Y = empty((m, nt))  # noisy labels
    for i in arange(m):
        # with prob=f/2, flip m coins
        Y[i] = -y[i] * (2 * bernoulli.rvs(f[i] / 2., size=nt) - 1)
        # with prob=f, flip all coins
        if sign(sum(Y[i])) == sign(y[i]):
            Y[i] = -Y[i] * (2 * bernoulli.rvs(f[i] / 1.) - 1)

    return Y
