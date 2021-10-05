import numpy as np
import pandas as pd
from itertools import permutations

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


def make_scorer(metric):
    def scorer(estimator, X, y):
        y_hat = estimator.predict(X)
        if not isinstance(y_hat, pd.DataFrame):
            y_hat = pd.DataFrame(y_hat, index=y.index)
        score = metric(y_hat, y)
        return score.mean().mean()
    return scorer

def summary_score(x, t, score=rmse, agg_index=None):
    if isinstance(x, pd.DataFrame) and isinstance(t, pd.DataFrame):
        x.columns = t.columns
    elif isinstance(x, pd.DataFrame):
        t = pd.DataFrame(t, index=x.index, columns=x.columns)
    elif isinstance(t, pd.DataFrame):
        x = pd.DataFrame(x, index=t.index, columns=t.columns)

    return score(x, t, agg_index)


def summary_scores(x, t, scores, idxs: pd.DataFrame, mask=None, n_quantiles=10):

    agg_indexes = idxs.copy()

    # all vars that are pd.DataFrame must have the same index
    pd_indexes = [a.index for a in [x,t,mask] if isinstance(a, pd.DataFrame)]
    assert np.all([a[0]==a[1] for a in permutations(pd_indexes,2)]), 'some dataframe do not have the same index. ' \
                                                                     'This could be a sign that something is wrong. ' \
                                                                     'Please, check.'

    # mask[mask] transform a boolean mask in a 1.0-NaN mask
    x = x if mask is None else x * mask[mask].values
    t = t if mask is None else t * mask[mask].values

    # quantize non-integer data
    if np.any(agg_indexes.dtypes != 'int'):
        qcuts = [pd.qcut(agg_indexes.loc[:, k], n_quantiles, duplicates='drop') for k in agg_indexes.columns if agg_indexes[k].dtype != int and len(agg_indexes[k].value_counts())>=n_quantiles]
        if len(qcuts)>0:
            replacement_idxs = [agg_indexes[k].dtype != int and len(agg_indexes[k].value_counts())>=n_quantiles for k in agg_indexes.columns]
            agg_indexes.loc[:, replacement_idxs] = pd.concat(qcuts, axis=1)

    scores_df = {}
    for s in scores:
        index_scores = {k: summary_score(x, t, s, pd.Index(v)) for k, v in agg_indexes.items()}
        scores_df[s.__name__] = pd.concat(index_scores, axis=0)
    return scores_df