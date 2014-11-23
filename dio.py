import numpy as np
import scipy.spatial.distance

def split(X, Y, N, M, startM=0, front=True):
    """
    N is int - number of trials
    M is int - number of lags
    """
    if front:
        X = X[:N, startM:startM+M]
        Y = Y[:N]
    else:
        X = X[-(N+1):, startM:startM+M]
        Y = Y[-(N+1):]
    return X, Y

def load_raw(infile, keep_ones=True):
    assert infile.endswith('.npy')
    xs = np.load(infile)
    end_ind = -1 if keep_ones else -2
    X = xs[:,:end_ind][:,::-1] # all but last columns; reverse columns
    Y = xs[:,-1] # last column
    return X, Y

def sqdist(xy):
    return scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(xy, 'euclidean'))**2

def temporal_distance(nt):
    xy = np.array(zip(np.arange(nt), np.zeros(nt))) # distances between lags in time is just 1s
    return sqdist(xy) # distance matrix

def load(infile, nLags, trainPct):
    X, Y = load_raw(infile, keep_ones=False)
    if trainPct == 0.0:
        N = X.shape[0]
    elif trainPct > 0.0:
        N = int(trainPct*X.shape[0])
    if nLags == 0:
        nLags = X.shape[1]
    X0, Y0 = split(X, Y, N, nLags, front=True)
    X1, Y1 = split(X, Y, X.shape[0] - N, nLags, front=False)
    D = temporal_distance(X0.shape[1])
    return X0, Y0, X1, Y1, D