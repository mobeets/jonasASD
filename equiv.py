import numpy as np
from asd import ASDReg as reg
from dio import temporal_distance as sqdist

fr = lambda r, d, n: reg(r, [(sqdist(n)**0.5, np.sqrt(1/d)/np.sqrt(2.0))])
fr2 = lambda r, d, n: np.exp(-r)*(np.exp(-sqdist(n)**0.5)**d)
cm = lambda r, d, n: np.allclose(fr(r,d,n), fr2(r,d,n))


finv = lambda r, d, n: np.linalg.inv(fr(r,d,n))
# finv2 = lambda r, d, n: np.exp(r)*(np.linalg.inv(np.exp(-sqdist(n)))**d)
finv2 = lambda r, d, n: np.exp(r)*np.linalg.inv(np.exp(-sqdist(n)**0.5)**d)
cm2 = lambda r, d, n: np.allclose(finv(r,d,n), finv2(r,d,n))

N = 10
print 'Reg pass: ',
print np.array([cm(np.random.uniform(-20, 20), np.random.uniform(1e-2, 20), np.random.randint(1, 30)) for i in xrange(N)]).all()
print 'RevInv pass: ',
print cm2(0.0, 4.0, 30)


print np.linalg.matrix_rank(np.exp(-sqdist(500)**0.5))

def f(r, n):
    r = np.exp(r)
    D = np.exp(-sqdist(n))
    for d in np.arange(20):
        # print np.linalg.matrix_rank(r*(D**(d+1e-2)))
        print np.linalg.cond(r*(D**(1/(d+1e-2))))
    # return r*(D**d)
f(0.0, 30)
# 1/(dp*np.sqrt(2.0)**2)