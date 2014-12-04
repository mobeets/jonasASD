import numpy as np
import scipy.optimize
from asd import ASDLogEvi, ASDEviGradient, ASDReg, PostCovInv, PostMean, MeanInvCov, LOWER_BOUND_DELTA_TEMPORAL

def ASD(X, Y, Ds, theta0=None, jac=True, isLog=True, method='TNC'): # 'TNC' 'CG', 'SLSQP', 'L-BFGS-B'
    """
    X - (p x q) matrix with inputs in rows
    Y - (p, 1) matrix with measurements
    Ds - [(q, q), ...] matrices containing squared distances between all input points
    
    Implelements the ASD regression descrived in:
        M. Sahani and J. F. Linden.
        Evidence optimization techniques for estimating stimulus-response functions.
        In S. Becker, S. Thrun, and K. Obermayer, eds.
        Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003
    """
    theta0 = (1.0, 0.1) + (2.0,)*len(Ds) if theta0 is None else theta0
    if isLog:
        theta0 = np.log(theta0)
        bounds = [(1.0, 3.0), (0.0, 10.0)] + [(0.0, 10.0)]*len(Ds)
    else:
        bounds = [(-20.0, 20.0), (10e-6, 10e6)] + [(LOWER_BOUND_DELTA_TEMPORAL, 1e5)]*len(Ds)
    p, q = X.shape
    YY = Y.T.dot(Y)
    XY = X.T.dot(Y)
    XX = X.T.dot(X)

    def objfcn(hyper, jac=jac, verbose=True):
        if isLog:
            hyper = np.exp(hyper)
        if verbose:
            print hyper
        ro, ssq = hyper[:2]
        deltas = hyper[2:]
        Reg = ASDReg(ro, zip(Ds, deltas))
        SigmaInv = PostCovInv(np.linalg.inv(Reg), XX, ssq)
        evi = ASDLogEvi(XX, YY, XY, Reg, SigmaInv, ssq, p, q)
        if not jac:
            return -evi
        mu = PostMean(SigmaInv, XY, ssq)
        sse = (Y - X.dot(mu)**2).sum()
        der_evi = ASDEviGradient(hyper, p, q, Ds, mu, np.linalg.inv(SigmaInv), Reg, sse)
        if verbose:
            print -np.array(der_evi)
        return -evi, -np.array(der_evi)

    options = {'disp': 5}
    theta = scipy.optimize.minimize(objfcn, theta0, bounds=bounds, method=method, jac=jac, options=options)
    if not theta['success']:
        print theta
    hyper = theta['x']
    if isLog:
        hyper = np.exp(hyper)
    print hyper
    ro, ssq = hyper[:2]
    deltas = hyper[2:]
    Reg = ASDReg(ro, zip(Ds, deltas))
    mu, _ = MeanInvCov(XX, XY, Reg, ssq)
    return mu, Reg, hyper
