import numpy as np
import pandas as pd
from itertools import permutations


def chose_axis(x, agg_index):
    return 0 if len(x) == len(agg_index) else 1


def err(x, t):
    return x-t


def squerr(x, t):
    return err(x, t)**2


def rmse(x, t, agg_index=None):
    agg_index = x.index if agg_index is None else agg_index
    return squerr(x, t).groupby(agg_index, axis=chose_axis(x, agg_index)).mean() ** 0.5


def mape(x, t, agg_index=None):
    agg_index = x.index if agg_index is None else agg_index
    return (err(x, t)/(t + 1e-5)).abs().groupby(agg_index, axis=chose_axis(x, agg_index)).mean()


def nmae(x, t, agg_index=None):
    agg_index = x.index if agg_index is None else agg_index
    return (err(x, t) / (t.mean(axis=1).values.reshape(-1,1) + 1)).abs().groupby(agg_index, axis=chose_axis(x, agg_index)).mean()


def make_scorer(metric):
    def scorer(estimator, X, y):
        y_hat = estimator.predict(X)
        if not isinstance(y_hat, pd.DataFrame):
            y_hat = pd.DataFrame(y_hat, index=y.index, columns=y.columns)
        else:
            y_hat.columns = y.columns
        # default behaviour is to reduce metric over forecasting horizon.
        # If just one step ahead is forecasted, metric is reduced on samples
        agg_index = np.zeros(len(y)) if len(y.shape)<2 else np.zeros(y.shape[1])
        score = metric(y_hat, y, agg_index=agg_index)
        return score.mean()
    return scorer


def summary_score(x, t, score=rmse, agg_index=None):
    if isinstance(x, pd.DataFrame) and isinstance(t, pd.DataFrame):
        x.columns = t.columns
    elif isinstance(x, pd.DataFrame):
        t = pd.DataFrame(t, index=x.index, columns=x.columns)
    elif isinstance(t, pd.DataFrame):
        x = pd.DataFrame(x, index=t.index, columns=t.columns)

    return score(x, t, agg_index)


def summary_scores(x, t, metrics, idxs: pd.DataFrame, mask=None, n_quantiles=10):

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
    for m in metrics:
        index_scores = {k: summary_score(x, t, m, pd.Index(v)) for k, v in agg_indexes.items()}
        scores_df[m.__name__] = pd.concat(index_scores, axis=0)
    return scores_df