import numpy as np

# LOWER_BOUND_DELTA_TEMPORAL = 0.025904
LOWER_BOUND_DELTA_TEMPORAL = 0.12 # less than about 0.30 is indistinguishable

logDet = lambda x: np.linalg.slogdet(x)[1]
linv = lambda A, y: np.linalg.solve(A, y)
rinv = lambda A, y: np.linalg.solve(A.T, y.T).T

def PostCovInv(RegInv, XX, ssq):
    return XX/ssq + RegInv

def PostMean(SigmaInv, XY, ssq):
    return linv(SigmaInv, XY)/ssq

def ASDLogEvi(XX, YY, XY, Reg, SigmaInv, ssq, p, q):
    """
    XX is X.T.dot(X) - m x m
    YY is Y.T.dot(Y) - 1 x 1
    XY is X.T.dot(Y) - m x 1
    """
    A = -logDet(Reg.dot(XX)/ssq + np.eye(q)) - p*np.log(2*np.pi*ssq)
    B = YY/ssq - XY.T.dot(linv(SigmaInv, XY))/(ssq**2)
    return (A - B)/2.0

def ASDEviGradient(hyper, p, q, Ds, mu, Sigma, Reg, sse):
    """
    gradient of log evidence w.r.t. hyperparameters
    """
    ro, ssq = hyper[:2]
    deltas = hyper[2:]
    Z = rinv(Reg, Reg - Sigma - np.outer(mu, mu))
    der_ro = np.trace(Z)/2.0
    
    v = -p + q - np.trace(rinv(Reg, Sigma))
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

def ASDInit(X, Y, D, (ro, ssq, delta)):
    XX = X.T.dot(X)
    XY = X.T.dot(Y)
    YY = Y.T.dot(Y)
    p, q = X.shape
    Reg = ASDReg(ro, [(D, delta)])
    return XX, XY, YY, p, q, Reg

def MeanInvCov(XX, XY, Reg, ssq):
    SigmaInv = PostCovInv(np.linalg.inv(Reg), XX, ssq)
    return PostMean(SigmaInv, XY, ssq), SigmaInv

def evidence(X, Y, D, (ro, ssq, delta)):
    XX, XY, YY, p, q, Reg = ASDInit(X, Y, D, (ro, ssq, delta))
    SigmaInv = PostCovInv(np.linalg.inv(Reg), XX, ssq)
    return ASDLogEvi(XX, YY, XY, Reg, SigmaInv, ssq, p, q)

def loglikelihood(X, Y, D, (ro, ssq, delta)):
    XX, XY, YY, p, q, Reg = ASDInit(X, Y, D, (ro, ssq, delta))
    mu, _ = MeanInvCov(XX, XY, Reg, ssq)
    return ASDLogLikelihood(Y, X, mu, ssq)

def scores(X0, Y0, X1, Y1, D, (ro, ssq, delta)):
    evi = evidence(X0, Y0, D, (ro, ssq, delta))
    ll = loglikelihood(X1, Y1, D, (ro, ssq, delta))
    return evi, -ll
