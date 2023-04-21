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


def nmae(x, t, agg_index=None, inter_normalization=True):
    agg_index = x.index if agg_index is None else agg_index
    offset = t.abs().mean(axis=1).quantile(0.5) * 0.01 if inter_normalization else 0
    return (err(x, t) / (t.abs().mean(axis=1).values.reshape(-1,1) + offset)).abs().groupby(agg_index, axis=chose_axis(x, agg_index)).mean()


def quantile_scores(q_hat, t, alphas=None, agg_index=None):
    """
    :param q_hat: matrix of quanitles, preferred form (n_t, n_sa, n_quantiles)
    :param t: target matrix or DataFrame (n_t, n_sa)
    :param alphas: vector of quantiles, if none assume linear spacing from 0 1
    :param agg_index:
    :return:
    """
    agg_index = t.index if agg_index is None else agg_index

    quantile_axis = 2 if q_hat.shape[1] == t.shape[1] else 1
    n_quantiles = q_hat.shape[quantile_axis]
    alphas = np.linspace(0, 1, n_quantiles) if alphas is None else alphas
    qscore_alpha, reliability_alpha = {}, {}
    for a, alpha in enumerate(alphas):
        if quantile_axis == 1:
            err_alpha = q_hat[:, a, :] - t.values
        else:
            err_alpha = q_hat[:, :, a] - t.values
        I = (err_alpha > 0).astype(int)
        qs_a = (I - alpha) * err_alpha
        qs_a = pd.DataFrame(qs_a, index=t.index, columns=t.columns).groupby(agg_index, axis=chose_axis(t, agg_index)).mean()

        qscore_alpha[alpha] = qs_a
        reliability_alpha[alpha] = pd.DataFrame(I, index=t.index, columns=t.columns).groupby(agg_index, axis=chose_axis(t, agg_index)).mean()

    qscore = pd.concat(qscore_alpha, axis=1)
    reliability = pd.concat(reliability_alpha, axis=1)

    return qscore, reliability


def crps(q_hat, t, alphas=None, agg_index=None, collapse_quantile_axis=True):
    qscore, _ = quantile_scores(q_hat, t, alphas=alphas, agg_index=agg_index)

    # collapse quantile axis
    if collapse_quantile_axis:
        qscore = qscore.groupby(axis=1, level=1).mean()

    return qscore


def reliability(q_hat, t, alphas=None, agg_index=None, get_score=False):
    alphas = np.linspace(0, 1, q_hat.shape[2]) if alphas is None else alphas
    _, reliability = quantile_scores(q_hat, t, alphas=alphas, agg_index=agg_index)

    # subtract to reliability dataframe the name of the first level of the multiindex dataframe
    if get_score:
        for alpha in reliability.columns.get_level_values(0).unique():
            reliability[alpha] = reliability[alpha] - alpha
    return reliability



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
        return np.mean(score.mean()) if len(score.shape)>1 else score.mean()
    return scorer


def summary_score(x, t, score=rmse, agg_index=None):
    if len(x.shape) < 3:
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