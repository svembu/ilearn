from numpy import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import Ridge
import util
import constants


def ilearn(X_train, Y_train, X_test, y_test, cparam, alpha, iact):
    """
    Trains a model on multiple noisy labels using the method described in the paper.

    Parameters
    ----------
    X_train: array of shape = [n_samples, n_features]
        Training examples
    Y_train: array of shape = [n_samples, n_annotators]
        Noisy training labels
    X_test: array of shape = [n_samples, n_features]
        Test examples
    y_train: array of shape = [n_samples]
        Test labels
    cparam: array
        list of regularization parameters
    alpha: positive real number
       example reweighting parameter (see paper)
    iact: positive real number
       binary variable, 1/True to run the algorithm in the interactive mode,
       0/False to run the algorithm in non-interactive mode

    Returns
    -------
    AUC of the trained model computed on the test set
    """

    dis = ones(X_train.shape[0])

    z_new = ones(Y_train.shape[1])  # expertise scores of annotators
    y_train = sum(Y_train * outer(ones(Y_train.shape[0]), z_new), 1) * 1. / sum(z_new)

    auc_out = 0
    for itr in range(constants.MAX_ITRS):
        z = z_new

        if iact:
            dis = util.dis_score(Y_train, alpha)

        model = train(X_train, y_train, cparam, dis)
        f = model.predict(X_test)
        auc_out = util.auc(f, y_test)

        # re-estimate z and labels
        f = model.predict(X_train)
        dis_wts = outer(dis, ones(len(z)))
        zz = mean(dis_wts * (outer(f, ones(len(z))) - Y_train) ** 2, 0)
        z_new = 1. / zz

        y_train = sum(Y_train * outer(ones(len(y_train)), z_new), 1) * 1. / sum(z_new)

        if util.has_converged(z, z_new, tol=1e-7):
            break

    return auc_out


def train(X, y, cparam, wts=None):
    """
     Trains ridge regression model on the input data set.
     Uses cross-validation to select the regularization parameter
     and returns the model retrained with the best parameter.

     Parameters
     ----------
     X: array of shape = [n_samples, n_features]
         Input examples
     y: array of shape = [n_samples]
         labels
     cparam: array
         list of regularization parameters
     wts: array of shape = [n_samples]
         example weights

     Returns
     -------
     ridge regression model (scik)
     """

    err = []

    for c in cparam:
        err.append(cv(X, y, c, wts, constants.NFOLDS))

    bparam = cparam[argmin(err)]

    model = Ridge(alpha=bparam)

    if wts is not None:
        model.fit(X, y, sample_weight=wts)
    else:
        model.fit(X, y)

    return model


def cv(X, y, c=1, wts=None, nfolds=10):
    """
    Runs nfold cross-validation on the input data set. Uses ridge regression
    as the training algorithm.

    Parameters
    ----------
    X: array of shape = [n_samples, n_features]
        Input examples
    y: array of shape = [n_samples]
        labels
    c: positive real number
        regularization parameter
    wts: array of shape = [n_samples]
        example weights
    nfolds: scalar
        no. of folds in cross-validation

    Returns
    -------
    average mean squared error (cross-validation error)
    """

    kf = StratifiedKFold(sign(y), n_folds=nfolds)

    err = []
    for tr_ids, te_ids in kf:
        model = Ridge(alpha=c)

        if wts is not None:
            model.fit(X[tr_ids], y[tr_ids], sample_weight=wts[tr_ids])
        else:
            model.fit(X[tr_ids], y[tr_ids])

        f = model.predict(X[te_ids])
        err.append(util.mse(f, y[te_ids]))
    return mean(err)