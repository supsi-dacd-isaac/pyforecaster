import numpy as np
import pandas as pd
from itertools import permutations
from pyforecaster.forecaster import ScenarioGenerator

def chose_axis(x, agg_index):
    return 0 if len(x) == len(agg_index) else 1


def err(x, t):
    return x-t


def squerr(x, t):
    return err(x, t)**2


def rmse(x, t, agg_index=None, **kwargs):
    agg_index = np.ones_like(x.index) if agg_index is None else agg_index
    res = squerr(x, t)
    axis = chose_axis(x, agg_index)
    if axis == 0:
        res = res.groupby(agg_index, observed=False).mean() ** 0.5
    else:
        res = res.T.groupby(agg_index, observed=False).mean() ** 0.5
    return res


def mape(x, t, agg_index=None, **kwargs):
    agg_index = np.ones_like(x.index) if agg_index is None else agg_index
    res = (err(x, t)/(np.abs(t) + 1e-5)).abs()
    axis = chose_axis(x, agg_index)
    if axis == 0:
        res = res.groupby(agg_index, observed=False).mean()
    else:
        res = res.T.groupby(agg_index, observed=False).mean()
    return res


def nmae(x, t, agg_index=None, inter_normalization=True, **kwargs):
    agg_index = np.ones_like(x.index) if agg_index is None else agg_index
    offset = t.abs().mean(axis=1).quantile(0.5) * 0.01 + 1e-12 if inter_normalization else 1e-12
    axis = chose_axis(x, agg_index)
    res = (err(x, t) / (t.abs().mean(axis=1).values.reshape(-1,1) + offset)).abs()
    if axis == 0:
        res = res.groupby(agg_index, observed=False).mean()
    else:
        res = res.T.groupby(agg_index, observed=False).mean()
    return res


def quantile_scores(q_hat, t, alphas=None, agg_index=None, **kwargs):
    """
    :param q_hat: matrix of quanitles, preferred form (n_t, n_sa, n_quantiles)
    :param t: target matrix or DataFrame (n_t, n_sa)
    :param alphas: vector of quantiles, if none assume linear spacing from 0 1
    :param agg_index:
    :return:
    """
    agg_index = np.ones_like(t.index) if agg_index is None else agg_index
    q_hat = ScenarioGenerator().quantiles_to_numpy(q_hat) if isinstance(q_hat, pd.DataFrame) else q_hat
    quantile_axis = 2 if q_hat.shape[1] == t.shape[1] else 1
    n_quantiles = q_hat.shape[quantile_axis]
    if alphas is None:
        print('warning: alphas not specified, assuming linear spacing from 0 to 1')
    alphas = np.linspace(0, 1, n_quantiles) if alphas is None else alphas
    qscore_alpha, reliability_alpha = {}, {}
    axis = chose_axis(t, agg_index)
    for a, alpha in enumerate(alphas):
        if quantile_axis == 1:
            err_alpha = q_hat[:, a, :] - t.values
        else:
            err_alpha = q_hat[:, :, a] - t.values
        I = (err_alpha > 0).astype(int)
        qs_a = (I - alpha) * err_alpha

        if axis == 0:
            qs_a = pd.DataFrame(qs_a, index=t.index, columns=t.columns).groupby(agg_index, observed=False) .mean()
            reliability_alpha[alpha] = pd.DataFrame(I, index=t.index, columns=t.columns).groupby(agg_index, observed=False) .mean()

        else:
            qs_a = pd.DataFrame(qs_a, index=t.index, columns=t.columns).T.groupby(agg_index, observed=False) .mean()
            reliability_alpha[alpha] = pd.DataFrame(I, index=t.index, columns=t.columns).T.groupby(agg_index, observed=False) .mean()

        qscore_alpha[alpha] = qs_a

    qscore = pd.concat(qscore_alpha, axis=1, names=['alpha'])
    reliability = pd.concat(reliability_alpha, axis=1, names=['alpha'])

    return qscore, reliability


def crps(q_hat, t, alphas=None, agg_index=None, collapse_quantile_axis=True, **kwargs):
    qscore, _ = quantile_scores(q_hat, t, alphas=alphas, agg_index=agg_index)

    # collapse quantile axis
    if collapse_quantile_axis:
        qscore = qscore.T.groupby(level=1).mean().T

    return qscore


def reliability(q_hat, t, alphas=None, agg_index=None, get_score=False, **kwargs):
    alphas = np.linspace(0, 1, q_hat.shape[2]) if alphas is None else alphas
    _, reliability = quantile_scores(q_hat, t, alphas=alphas, agg_index=agg_index)

    # subtract to reliability dataframe the name of the first level of the multiindex dataframe
    if get_score:
        for alpha in reliability.columns.get_level_values(0).unique():
            reliability[alpha] = reliability[alpha] - alpha
    return reliability



def make_scorer(metric):
    if '__name__' in  dir(metric):
        name = metric.__name__
    elif 'func' in  dir(metric):
        name = metric.func.__name__
    def scorer(estimator, X, y):
        kwargs = {}
        if name  in ["crps", "reliability"]:
            y_hat = estimator.predict_quantiles(X)
            if 'q_vect' in dir(estimator):
                alphas = estimator.q_vect
                kwargs['alphas'] = alphas
        else:
            y_hat = estimator.predict(X)
            if not isinstance(y_hat, pd.DataFrame):
                y_hat = pd.DataFrame(y_hat, index=y.index, columns=y.columns)
            else:
                y_hat.columns = y.columns
        # default behaviour is to reduce metric over forecasting horizon.
        # If just one step ahead is forecasted, metric is reduced on samples
        agg_index = np.zeros(len(y)) if len(y.shape)<2 else np.zeros(y.shape[1])
        score = metric(y_hat, y, agg_index=agg_index, **kwargs)
        return np.mean(score.mean()) if len(score.shape)>1 else score.mean()
    return scorer


def summary_score(x, t, score=rmse, agg_index=None):
    if len(x.shape) < 3:
        # if x is a pd.DataFrame and its columns are multiindex, just make sure t is a df
        if isinstance(x, pd.DataFrame) and x.columns.nlevels > 1:
            if not isinstance(t, pd.DataFrame):
                t = pd.DataFrame(t, index=x.index)
        elif isinstance(x, pd.DataFrame) and isinstance(t, pd.DataFrame):
            x.columns = t.columns
        elif isinstance(x, pd.DataFrame):
            t = pd.DataFrame(t, index=x.index, columns=x.columns)
        elif isinstance(t, pd.DataFrame):
            x = pd.DataFrame(x, index=t.index, columns=t.columns)

    return score(x, t, agg_index=agg_index)


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
    if np.any(~agg_indexes.dtypes.isin([np.dtype('int'), np.dtype('int32'), np.dtype('int8'), np.dtype('int64'), np.dtype('object')])):
        qcuts = [pd.qcut(agg_indexes.loc[:, k], n_quantiles, duplicates='drop') for k in agg_indexes.columns if agg_indexes[k].dtype != int and len(agg_indexes[k].value_counts())>=n_quantiles]
        if len(qcuts)>0:
            replacement_idxs = [agg_indexes[k].dtype != int and len(agg_indexes[k].value_counts())>=n_quantiles for k in agg_indexes.columns]
            agg_indexes.loc[:, replacement_idxs] = pd.concat(qcuts, axis=1)

    scores_df = {}
    for m in metrics:
        if '__name__' in dir(m):
            name = m.__name__
        elif 'func' in dir(m):
            name = m.func.__name__
        else:
            name = 'metric'
        index_scores = {k: summary_score(x, t, m, pd.Index(v)) for k, v in agg_indexes.items()}
        scores_df[name] = pd.concat(index_scores, axis=0)
    return scores_df