import numpy as np
import pandas as pd


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


def nmae(x, t, agg_index=None):
    agg_index = x.index if agg_index is None else agg_index
    return (err(x, t) / (t.mean(axis=1).values.reshape(-1,1) + 1)).abs().groupby(agg_index).mean()


def summary_score(x, t, score=rmse, agg_index=None):
    return score(x, t, agg_index)


def summary_scores(x, t, scores, agg_indexes: pd.DataFrame, mask=None, n_quantiles=10):

    # mask[mask] transform a boolean mask in a 1.0-NaN mask
    x = x if mask is None else x * mask[mask]
    t = t if mask is None else t * mask[mask]

    # quantize non-integer data
    if np.any(agg_indexes.dtypes != 'int'):
        agg_indexes.loc[:, agg_indexes.dtypes != int] = pd.concat(
            [pd.qcut(agg_indexes.loc[:, k], n_quantiles) for k in agg_indexes.columns if agg_indexes[k].dtype != int], axis=1)

    scores_df = {}
    for s in scores:
        index_scores = {k: summary_score(x, t, s, pd.Index(v)) for k, v in agg_indexes.items()}
        scores_df[s.__name__] = pd.concat(index_scores, axis=0)
    return scores_df