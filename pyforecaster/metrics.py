import numpy as np


def feature_importance(x, m, n_epsilon=10):
    # isotropic spherical sampling
    n_o, n_f = x.shape()
    eps = np.random.randn(n_epsilon, n_o, n_f)
    preds = np.dstack([m.predict((x + eps_i)) for eps_i in eps])
    std = np.std(preds, axis=-1)


def err(x, t):
    return x-t


def squerr(x, t):
    return err(x, t)**2


def rmse(x, t, agg_index=None):
    agg_index = x.index if agg_index is None else agg_index
    return squerr(x, t).groupby(agg_index).mean() ** 0.5


def mape(x, t, agg_index=None):
    agg_index = x.index if agg_index is None else agg_index
    return (err(x, t)/(t + 1e-5)).abs().groupby(agg_index).mean()


def summary_score(x, t, score=rmse, agg_index=None):
    return score(x, t, agg_index)