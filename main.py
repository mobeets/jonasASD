#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt
from dio import load
from solve import ASD
from grid_zoom import grid_zoom
from asd import scores, MeanInvCov, ASDReg, ASDInit

def main(infile, trainPct, nLags, (ro, ssq, delta), doScore, doPlot, doSolve, doFit, doGridZoom):
    X0, Y0, X1, Y1, D = load(infile, nLags, trainPct)
    print 'Loaded {0} rows from {1}'.format(X0.shape[0] + X1.shape[0], infile)
    print 'Training on {0} rows'.format(X0.shape[0])
    print
    muOLS, muRdg = None, None
    if doFit:
        OLS = sklearn.linear_model.LinearRegression(fit_intercept=False, normalize=False).fit(X0, Y0)
        # OLS = sklearn.linear_model.BayesianRidge(fit_intercept=False).fit(X1, Y1)
        muOLS = OLS.coef_
        Rdg = sklearn.linear_model.BayesianRidge(fit_intercept=False).fit(X0, Y0)
        muRdg = Rdg.coef_
        print 'Ridge: --ssq {0} --ro {1}'.format(1/Rdg.alpha_, np.log(Rdg.lambda_))
    if doScore:
        print 'Initial scores...'
        evi, nll = scores(X0, Y0, X1, Y1, D, (ro, ssq, delta))
        print 'ro={0}, ssq={1}, delta={2}'.format(ro, ssq, delta)
        print 'evidence={0}'.format(evi)
        print 'neg. log likelihood={0}'.format(nll)
        print
    if doPlot:
        XX, XY, YY, p, q, Reg = ASDInit(X0, Y0, D, (ro, ssq, delta))
        mu, SigmaInv = MeanInvCov(XX, XY, Reg, ssq)
        plt.plot([0, X0.shape[1]], [0.0, 0.0], '--', color='k', alpha=0.5)
        plt.plot(mu, '-', alpha=0.5, lw=2, label='ASD')
        # err = np.sqrt(np.diag(np.linalg.inv(SigmaInv)))
        # plt.errorbar(xrange(len(mu)), mu, yerr=err)
        for lbl, m in [('OLS', muOLS), ('Ridge', muRdg)]:
            if m is not None:
                plt.plot(m, '-', alpha=0.5, lw=2, label=lbl)
        if muOLS is not None or muRdg is not None:
            plt.legend()
        plt.show()
    if doSolve:
        print 'Solving...'
        mu, Reg, hyper = ASD(X0, Y0, [D], (ro, ssq, delta))
        ro, ssq, delta = hyper
        evi, nll = scores(X0, Y0, X1, Y1, D, hyper)
        print 'ro={0}, ssq={1}, delta={2}'.format(ro, ssq, delta)
        print 'evidence={0}'.format(evi)
        print 'neg. log likelihood={0}'.format(nll)
    if doGridZoom:
        grid_zoom(X0, Y0, D, (ro, ssq, delta))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, default=None)
    parser.add_argument("--score", action='store_true', default=False, help="calculate evidence")
    parser.add_argument("--solve", action='store_true', default=False, help="minimizes evidence using scipy.optimize")
    parser.add_argument("--gridzoom", action='store_true', default=False, help="minimizes evidence by gridding hyperparams and zooming")
    parser.add_argument("--plot", action='store_true', default=False, help="plots the solution")
    parser.add_argument("--fit", action='store_true', default=False, help="fit OLS and Ridge")
    parser.add_argument('--ro', type=float, required=True, default=None)
    parser.add_argument('--ssq', type=float, required=True, default=None)
    parser.add_argument('--delta', type=float, required=True, default=None)
    parser.add_argument('-p', type=float, default=0.1, help="percent of data to use in training")
    parser.add_argument('-m', type=int, default=0, help="# of lags to use")
    args = parser.parse_args()
    main(args.infile, args.p, args.m, (args.ro, args.ssq, args.delta), args.score, args.plot, args.solve, args.fit, args.gridzoom)
