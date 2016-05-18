import numpy as np

import pandas as pd
from sklearn import datasets

import core


np.random.seed(42)


class ConvergenceError(Exception):
    pass


def init_ord(n_samples):
    """Random ordering of samples. The ordering is only generated once per 
    run to allow different methods to use the same ordering"""
    global _ord
    try:
        _ord
    except NameError:
        _ord = np.arange(n_samples)
        np.random.shuffle(_ord)


def init_noise(n_samples, var=0.25):
    """Artifical gaussian noise used for stochastic simulations"""
    global _noise
    try:
        _noise
    except NameError:
        _noise = var * np.random.randn(n_samples, 2)
        
    
def cast_as_func(arg):
    if hasattr(arg, '__call__'):
        return arg
    else:
        return lambda k: arg


def flatten(listoflist):
    return [item for sublist in listoflist for item in sublist]


def load_wine_dataset():
    print 'Loading Wine dataset...'
    header = ['label','Alcohol','Malic acid','Ash','Alcalinity of ash',
              'Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
              'Proanthocyanins','Color intensity','Hue',
              'OD280/OD315 of diluted wines','Proline']
    wine_df = pd.read_csv('data/wine.data.txt', sep=',')
    wine_df.columns = header
    wine_df = wine_df[wine_df['label']>1]
    wine_df['label'] -= 2
    h1 = 'Hue'
    h2 = 'OD280/OD315 of diluted wines'
    wine_df = wine_df[['label', h1, h2]]
    wine_arr = np.array(wine_df)
    wine_arr[:,1:] = (wine_arr[:,1:] - wine_arr[:,1:].min(axis=0)) / (wine_arr[:,1:].max(axis=0) - wine_arr[:,1:].min(axis=0))
    return wine_arr

def load_iris_dataset():
    iris = datasets.load_iris()
    mask = iris.target < 2
    iris_arr = np.array([iris.target,iris.data[:,2]]).T[mask]
    order = np.arange(len(iris_arr))
    np.random.shuffle(order)
    return iris_arr[order]


def squareloss(x, y, label, feat):
    return 0.5*((x*feat + y - label)**2)


def squareloss_grad(x, y, label, feat):
    diff = (x*feat + y - label)
    return np.array([feat*diff, diff])


def build_wine_logreg_func(stoch=False):
    global wine_data
    try:
        wine_data
    except NameError:
        wine_data = load_wine_dataset()
    def func(x, y):
        return squareloss(x, y, wine_data[:,0], wine_data[:,1]).mean()
    def grad(x, y):
        return squareloss_grad(x, y, wine_data[:,0], wine_data[:,1]).mean(axis=1)
    if stoch:
        def stoch_func(x, y, i):
            return squareloss(x, y, wine_data[i,0], wine_data[i,1])
        def stoch_grad(x, y, i):
            return squareloss_grad(x, y, wine_data[i,0], wine_data[i,1])
        return core.StochasticFunction2dWrapper(func, grad, stoch_func, stoch_grad, len(wine_data))
    return core.Function2dWrapper(func, grad)


def build_iris_logreg_func(stoch=False):
    global iris_arr
    try:
        iris_arr
    except NameError:
        iris_arr = load_iris_dataset()
    def func(x, y):
        return squareloss(x, y, iris_arr[:,0], iris_arr[:,1]).mean()
    def grad(x, y):
        return squareloss_grad(x, y, iris_arr[:,0], iris_arr[:,1]).mean(axis=1)
    if stoch:
        def stoch_func(x, y, i):
            return squareloss(x, y, iris_arr[i,0], iris_arr[i,1])
        def stoch_grad(x, y, i):
            return squareloss_grad(x, y, iris_arr[i,0], iris_arr[i,1])
        return core.StochasticFunction2dWrapper(func, grad, stoch_func, stoch_grad, len(iris_arr))
    return core.Function2dWrapper(func, grad)

if __name__ == "__main__":
    import matplotlib; 
    matplotlib.use('TkAgg'); 
    from matplotlib import pyplot as plt
    func = build_wine_logreg_func()
    x = np.linspace(-50, 10, num=100)
    y = np.linspace(-10, 10, num=100)
    X, Y = np.meshgrid(x, y)
    ff = np.vectorize(func._func)
    Z = ff(X,Y)
    plt.figure()
    plt.contourf(X, Y, Z, levels=np.linspace(Z.min(),Z.max(),100))
    h1 = 'Hue'
    h2 = 'OD280/OD315 of diluted wines'
    mask = wine_data[:,0]==1
    plt.show()

