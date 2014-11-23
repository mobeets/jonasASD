import numpy as np
import scipy.optimize
from asd import ASDEvi, ASDReg, PostCov, PostMean

def ASD(X, Y, Ds, theta0=None, method='L-BFGS-B'): # 'SLSQP'
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
    bounds = [(-20.0, 20.0), (10e-6, 10e6)] + [(1e-5, 10e3)]*len(Ds)
    p, q = X.shape
    XY = X.T.dot(Y)
    XX = X.T.dot(X)

    def objfcn(hyper):
        ro, ssq = hyper[:2]
        deltas = hyper[2:]
        Reg = ASDReg(ro, zip(Ds, deltas))
        sigma = PostCov(np.linalg.inv(Reg), XX, ssq)
        return -ASDEvi(X, Y, Reg, sigma, ssq, p, q)

    theta = scipy.optimize.minimize(objfcn, theta0, bounds=bounds, method=method, jac=False)
    if not theta['success']:
        print theta
    hyper = theta['x']
    print hyper
    ro, ssq = hyper[:2]
    deltas = hyper[2:]
    Reg = ASDReg(ro, zip(Ds, deltas))
    sigma = PostCov(np.linalg.inv(Reg), XX, ssq)
    mu = PostMean(sigma, XY, ssq)
    return mu, Reg, hyper