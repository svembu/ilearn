from numpy import *
from sklearn import metrics


# disagreement score
def dis_score(Y, alpha=1):
    """
    Computes disagreement score for the input noisy labels (see equation 2 in the paper).

    Parameters
    ----------
    Y: array of shape = [n_samples, n_annotators]
        noisy labels
    alpha: positive integer
        example reweighting parameter (see paper)

    Returns
    -------
    disagreement score
    """
    m, nt = Y.shape
    d = empty(m)
    for i in arange(m):
        d[i] = sum([(Y[i][p] - Y[i][q]) ** 2 for p in arange(nt) for q in arange(p, nt)])
    d = d * 1. / max(d)
    d = 1. / (1 + exp(alpha * d))
    return d


def auc(yp, yt):
    """
    Computes area under the ROC curve.

    Parameters
    ----------
    yp: array of shape = [n_samples]
        predicted scores
    yt: array of shape = [n_samples]
        true labels

    Returns
    -------
    area under the ROC curve
    """
    fpr, tpr, thresholds = metrics.roc_curve(yt, yp)
    return metrics.auc(fpr, tpr)


def mse(yp, yt):
    """
    Computes mean squared error.

    Parameters
    ----------
    yp: array of shape = [n_samples]
        predicted scores
    yt: array of shape = [n_samples]
        true labels

    Returns
    -------
    mean squared error
    """

    return sum((yp - yt) ** 2) * 1. / len(yp)


def has_converged(z_old, z_new, tol=1e-5):
    """
    Tests convergence.

    Parameters
    ----------
    z_old: array of shape = [n_annotators]
        old parameters
    z_new: array of shape = [n_annotators]
        new parameters
    tol: positive real number
        tolerance parameter

    Returns
    -------
    True if converged, False otherwise
    """
    conv = False
    if mean(abs(z_old - z_new)) < tol:
        conv = True
    return conv
