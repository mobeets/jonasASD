import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def color_list(n, cmap=None, gray=True):
    if gray:
        colors = [str(i) for i in np.linspace(0, 1, n)]
    else:
        cm = plt.get_cmap("Reds" if cmap is None else cmap)
        colors = [cm(i) for i in np.linspace(0, 1, n)]
    return colors*(n/len(colors)) + colors[:n%len(colors)]

def load(infile, lastind=None):
    df = pd.read_csv(infile)
    # df = df[df['1'] > 0]
    # df = df[df['2'] > 0]
    df = df[df['3'] > -2000]
    df = df[df['3'] < 0]
    return df if lastind is None else df.ix[:lastind]

def plot(df, xkey='0', ykey='1', zkey='3', grpkey='2', gray=False):
    colors = color_list(len(df[grpkey].unique()), None, gray)
    for i, (grp, dfc) in enumerate(df.groupby(grpkey)):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(dfc[xkey].values, dfc[ykey].values, dfc[zkey].values, c=colors[i])#, zdir='y')
        plt.xlabel('delta')
        plt.ylabel('ssq')
        ax.set_zlabel('log evidence')
        plt.show()

def main():
    df = load('out/evidences.csv')
    plot(df, gray=True)

if __name__ == '__main__':
    main()
