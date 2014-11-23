import argparse
from dio import load
from asd import scores
from solve import ASD
from grid_zoom import grid_zoom

def main(infile, trainPct, nLags, (ro, ssq, delta), doMinimize, doGridZoom):
    X0, Y0, X1, Y1, D = load(infile, nLags, trainPct)
    print 'Loaded {0} rows from {1}'.format(X0.shape[0] + X1.shape[0], infile)
    print 'Training on {0} rows'.format(X0.shape[0])
    print
    print 'Initial scores...'
    evi, nll = scores(X0, Y0, X1, Y1, D, (ro, ssq, delta))
    print 'ro={0}, ssq={1}, delta={2}'.format(ro, ssq, delta)
    print 'evidence={0}'.format(evi)
    print 'neg. log likelihood={0}'.format(nll)
    print
    if doMinimize:
        print 'Solving...'
        mu, Reg, hyper = ASD(X0, Y0, [D], (ro, ssq, delta))
        evi, nll = scores(X0, Y0, X1, Y1, D, hyper)
        print 'ro={0}, ssq={1}, delta={2}'.format(ro, ssq, delta)
        print 'evidence={0}'.format(evi)
        print 'neg. log likelihood={0}'.format(nll)
    if doGridZoom:
        grid_zoom(X0, Y0, D, (ro, ssq, delta))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, default=None)
    parser.add_argument("--minimize", action='store_true', default=False, help="minimizes evidence using scipy.optimize")
    parser.add_argument("--gridzoom", action='store_true', default=False, help="minimizes evidence by gridding hyperparams and zooming")
    parser.add_argument('--ro', type=float, required=True, default=None)
    parser.add_argument('--ssq', type=float, required=True, default=None)
    parser.add_argument('--delta', type=float, required=True, default=None)
    parser.add_argument('-p', type=float, default=0.1, help="percent of data to use in training")
    parser.add_argument('-m', type=int, default=0, help="# of lags to use")
    args = parser.parse_args()
    main(args.infile, args.p, args.m, (args.ro, args.ssq, args.delta), args.minimize, args.gridzoom)
