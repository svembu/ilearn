from numpy import *
from numpy.random import randint, seed
import sys
import util
import constants
import algo
import simulation as sim


def run(m, n, alpha=2, p=1):
    """
    Runs an experiment on simulated data.

    Parameters
    ----------
    m: positive integer
        Number of positive (=negative) training examples.
        An equal number of test examples is generated.
    d: positive integer
        number of features/dimensions
    alpha: positive integer
        example reweighting parameter (see paper)
    p: positive integer
        noise parameter (see paper)

    Returns
    -------
    1 if the experiment is successful,
    -1 if the simulated labels are too noisy (see comments).
    """

    # generate training data
    X_train, y_train, X_test, y_test = sim.gen_data(m, n)

    # regularization parameters
    cparam = 2 ** array([-14., -12., -10., -8., -6., -4., -2., -1., 0., 1., 2., 4., 6., 8., 10., 12., 14.])

    # initial example weights
    wts = ones(X_train.shape[0])

    # simulate annotators
    z = array([1, 1, 0, 0, 0, 0, 0, 0, -1, -1])  # annotators expertise
    Y_train = sim.sim_t(X_train, y_train, z, p)

    # train on noisy data using majority voted labels
    y_train = sign(sum(Y_train, 1))
    clf = algo.train(X_train, y_train, 1. / cparam)
    f1 = clf.predict(X_test)

    if util.auc(f1, y_test) < 0.5:
        return -1  # too noisy labels...

    # train ilearn model in non-interactive mode
    iauc1 = algo.ilearn(X_train, Y_train, X_test, y_test, 1. / cparam, alpha, iact=0)

    # train ilearn model in interactive mode
    iauc2 = algo.ilearn(X_train, Y_train, X_test, y_test, 1. / cparam, alpha, iact=1)

    print 'NEW TRIAL'
    print 'AUC of non-interactive learning algorithm: {}'.format(iauc1)
    print 'AUC of interactive learning algorithm: {}'.format(iauc2)
    print '\n'

    return 1


if __name__ == "__main__":

    alpha = int(sys.argv[1])  # example reweighting parameter
    p = int(sys.argv[2])  # noise parameter

    m = 250  # no. of positive (=negative) training examples
    n = 10  # no. of features

    print 'example reweighting parameter (alpha): {}'.format(alpha)
    print 'noise parameter (p): {}'.format(p)

    seeds = unique(randint(1e4, size=constants.NTRIALS * 10))
    ctr = 0
    for s in seeds:
        seed(s)
        if ctr == constants.NTRIALS: break
        ret = run(m, n)
        if ret == -1: continue
        ctr += 1
