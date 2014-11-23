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
    # df = df[df['3'] > -5000]
    return df if lastind is None else df.ix[:lastind]

def plot(ax, df, xkey='2', ykey='1', zkey='3', grpkey='0', gray=False):
    colors = color_list(len(df[grpkey].unique()), None, gray)
    for i, (grp, dfc) in enumerate(df.groupby(grpkey)):
        ax.scatter(dfc[xkey].values, dfc[ykey].values, dfc[zkey].values, c=colors[i])#, zdir='y')
    plt.xlabel('delta')
    plt.ylabel('ssq')
    ax.set_zlabel('log evidence')

def main():
    fig = plt.figure()
    df = load('out/evidences.csv')
    ax = fig.gca(projection='3d')
    plot(ax, df, gray=True)
    plt.show()

if __name__ == '__main__':
    main()
