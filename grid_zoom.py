import itertools
import numpy as np
import pandas as pd
from asd import ASDReg, ASDLogEvi, PostCovInv, LOWER_BOUND_DELTA_TEMPORAL

def nextHyperInGridFromHyper(hyper, ds, n):
    rng = []
    for x, d in zip(hyper, ds):
        rng.append(np.linspace(x-d, x+d, n))
    return rng

def nextHyperInGridFromBounds(bounds, n):
    """
    for each lower and upper bound in bounds,
    choose n equally spaced values within that range
    returns the cartesian product of all the resulting combos
    """
    rng = []
    for i, (lb, ub) in enumerate(bounds):
        rng.append(np.linspace(lb, ub, n))
    return rng

def ASDHyperGrid(XX, XY, YY, p, q, Ds, n=5, hyper0=None, ds=None):
    """
    evaluates the evidence at each hyperparam
        within a grid of valid hyperparams
    """
    evis = []
    if hyper0 is not None:
        grid = nextHyperInGridFromHyper(hyper0, ds, n)
    else:
        grid = nextHyperInGridFromBounds(asd_theta_bounds(Ds), n)
    for hyper in itertools.product(*grid):
        print hyper
        ro, ssq = hyper[:2]
        deltas = hyper[2:]
        Reg = ASDReg(ro, zip(Ds, deltas))
        SigmaInv = PostCovInv(np.linalg.inv(Reg), XX, ssq)
        evi = ASDLogEvi(XX, YY, XY, Reg, SigmaInv, ssq, p, q)
        print evi
        print '-----'
        evis.append((hyper, evi))
    return evis

def next_zoom(hyper0, ds0, hyper1, n):
    grid = [np.linspace(x-d, x+d, n) for x, d in zip(hyper0, ds0)]
    inds = [np.where(xs==x)[0][0] for xs, x in zip(grid, hyper1)]
    ds1 = [(xs[ind] - xs[ind-1]) if ind-1 >= 0 else (xs[ind+1] - xs[ind]) for ind, xs in zip(inds, grid)]
    return np.abs(np.array(ds1))

def grid_zoom(X, Y, D, hyper0, delta0=None, nbins=4, nzooms=4, outfile='out/evidences.csv'):
    """
    hyper0 is array - center (starting) hyperparameter
    delta0 is array - distance above and below each hyperparameter to include in grid
    nbins is int - number of bins in each dimension (default = hyper0 - 1e-3)
    nzooms is int - number of recursions

    (0.18593049482814772, 1.0013184627777783, 3.6758187813580245)
    $ python main.py ~/Desktop/design.npy --ro 0.18593049482814772 --ssq 1.0013184627777783 --delta 3.6758187813580245
        Initial scores...
        ro=0.185930494828, ssq=1.00131846278, delta=3.67581878136
        evidence=835929962.586
        neg. log likelihood=34109514.9469
    """
    p, q = X.shape
    XX = X.T.dot(X)
    XY = X.T.dot(Y)
    YY = Y.T.dot(Y)

    delta0 = np.array(hyper0) - 1e-3 if delta0 is None else delta0

    nhypers = len(hyper0)
    deltas = np.zeros([nzooms, nhypers])
    centers = np.zeros([nzooms+1, nhypers])
    evidences = np.zeros([nzooms*(nbins**nhypers), nhypers+1])
    deltas[0,:] = np.array(delta0)
    centers[0,:] = np.array(hyper0)

    for i in xrange(nzooms):
        delta = deltas[i,:]
        center = centers[i,:]
        print delta
        print center
        print '-----'

        # hs = itertools.product(*[np.linspace(x-d, x+d, nbins) for x, d in zip(center, delta)])
        # evis = [(h, np.random.rand()) for h in hs]
        evis = ASDHyperGrid(XX, XY, YY, p, q, [D], n=nbins, hyper0=center, ds=delta)
        eviM = max(evis, key=lambda x: x[-1])
        print 'Max grid evidence: {0}'.format(eviM)

        for j, (h, e) in enumerate(evis):
            evidences[i*(nbins**nhypers)+j,:] = np.hstack([np.array(h), np.array([e])])

        nextCenter = eviM[0]
        centers[i+1,:] = nextCenter
        keepLowerBound = lambda d, c, lb=0.0: d if (c - d) > lb else (c - lb - 1e-5)
        if i+1 < nzooms:
            deltas[i+1,:] = next_zoom(center, delta, nextCenter, nbins)*0.95 # perturb slightly
            deltas[i+1,-2] = keepLowerBound(deltas[i+1,-2], nextCenter[-2])
            deltas[i+1,-1] = keepLowerBound(deltas[i+1,-1], nextCenter[-1], LOWER_BOUND_DELTA_TEMPORAL)
        print '======'
    pd.DataFrame(evidences).to_csv('out/evidences.csv')
