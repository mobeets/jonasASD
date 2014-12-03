import numpy as np

# LOWER_BOUND_DELTA_TEMPORAL = 0.025904
LOWER_BOUND_DELTA_TEMPORAL = 0.12 # less than about 0.30 is indistinguishable

def PostCov(RegInv, XX, ssq):
    return np.linalg.inv((XX/ssq) + RegInv)

def PostMean(sigma, XY, ssq):
    return sigma.dot(XY)/ssq

logDet = lambda x: np.linalg.slogdet(x)[1]
def ASDEvi(X, Y, Reg, sigma, ssq, p, q):
    """
    log evidence
    """
    z1 = 2*np.pi*sigma
    z2 = 2*np.pi*ssq*np.eye(q)
    z3 = 2*np.pi*Reg
    logZ = logDet(z1) - logDet(z2) - logDet(z3)
    B = (np.eye(p)/ssq) - (X.dot(sigma).dot(X.T))/(ssq**2)
    return 0.5*(logZ - Y.T.dot(B).dot(Y))

def ASDLogEvi(X, Y, Reg, Sigma, ssq, p, q):
    A0 = X.T.dot(X).dot(Reg)/ssq + np.eye(q)
    A = 8*np.pi*ssq*A0
    B0 = np.eye(p)/ssq - X.T.dot(Sigma).dot(X)/(ssq**2)
    B = Y.dot(B0).dot(Y.T)
    return -(logDet(A) + B)/2.0

rinv = lambda A, y: np.linalg.solve(A.T, y.T).T

def ASDEviGradient(hyper, p, q, Ds, mu, sigma, Reg, sse):
    """
    gradient of log evidence w.r.t. hyperparameters
    """
    ro, ssq = hyper[:2]
    deltas = hyper[2:]
    Z = rinv(Reg, Reg - sigma - np.outer(mu, mu))
    der_ro = np.trace(Z)/2.0
    
    # the below two lines are not even used!
    v = -p + np.trace(np.eye(q) - rinv(Reg, sigma))
    der_ssq = sse/(ssq**2) + v/ssq

    der_deltas = []
    for (D, d) in zip(Ds, deltas):
        der_deltas.append(-np.trace(rinv(Reg, Z.dot(Reg * D/(d**3))))/2.0)
    return np.array((der_ro, der_ssq) + tuple(der_deltas))

def ASDLogLikelihood(Y, X, mu, ssq):
    sse = ((Y - X.dot(mu))**2).sum()
    return -sse/(2.0*ssq) - np.log(2*np.pi*ssq)/2.0

def RidgeReg(ro, q):
    return np.exp(-ro)*np.eye(q)

def ASDReg(ro, ds):
    """
    ro - float
    ds - list of tuples [(D, d), ...]
        D - (q x q) squared distance matrix in some stimulus dimension
        d - float, the weighting of D
    """
    vs = 0.0
    for D, d in ds:
        if not hasattr(vs, 'shape'):
            vs = np.zeros(D.shape)
        vs += D/(d**2)
    return np.exp(-ro-0.5*vs)

def MeanCov(X, Y, Reg, ro, ssq):
    XX = X.T.dot(X)
    XY = X.T.dot(Y)
    sigma = PostCov(np.linalg.inv(Reg), XX, ssq)
    mu = PostMean(sigma, XY, ssq)
    return mu, sigma

def evidence(X, Y, D, (ro, ssq, delta)):
    p, q = X.shape
    Reg = ASDReg(ro, [(D, delta)])
    _, sigma = MeanCov(X, Y, Reg, ro, ssq)
    return ASDEvi(X, Y, Reg, sigma, ssq, p, q)

def loglikelihood(X, Y, D, (ro, ssq, delta)):
    Reg = ASDReg(ro, [(D, delta)])
    mu, sigma = MeanCov(X, Y, Reg, ro, ssq)
    return ASDLogLikelihood(Y, X, mu, ssq)

def scores(X0, Y0, X1, Y1, D, (ro, ssq, delta)):
    evi = evidence(X0, Y0, D, (ro, ssq, delta))
    ll = loglikelihood(X1, Y1, D, (ro, ssq, delta))
    return evi, -ll
