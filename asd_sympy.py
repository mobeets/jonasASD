import numpy as np
import sympy as sp
"""
http://docs.sympy.org/latest/tutorial/intro.html

TODO: implement evidence expression, and try to integrate/differentiate?
"""

logDet = lambda x: np.linalg.slogdet(x)[1]

nt, nw = 10, 2
ssq = sp.symbols('ssq')
X = sp.MatrixSymbol('X', nt, nw)
Y = sp.MatrixSymbol('Y', nt, 1)
C = sp.MatrixSymbol('C', nw, nw)
I = sp.Identity(nt)

S = ((X.T*X)/ssq + C.inverse()).inverse()
e1 = sp.det(2*sp.pi*S) / (sp.det(2*sp.pi*ssq*sp.Identity(1)) * sp.det(2*sp.pi*C))
e2 = I/ssq - (X*S*X.T)/(ssq**2)
e3 = -0.5*Y.T*e2*Y
evidence = sp.sqrt(e1) * sp.exp(e3[0,0])

e4 = sp.log(sp.Determinant(S)) - sp.log(sp.Determinant(C)) - sp.log(2*sp.pi*ssq)
-log(2*pi*ssq) - log(Determinant(C)) + log(Determinant(S))

logevi = 0.5*e4 + e3
