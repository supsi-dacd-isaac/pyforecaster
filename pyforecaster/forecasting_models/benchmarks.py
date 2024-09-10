from pyforecaster.forecaster import ScenarioGenerator
import pandas as pd
import numpy as np
from pyforecaster.forecasting_models.holtwinters import hankel

class Persistent(ScenarioGenerator):
    def __init__(self, target_col, n_sa=1, q_vect=None, val_ratio=None, nodes_at_step=None,
                 conditional_to_hour=False, **scengen_kwgs):
        super().__init__(q_vect, nodes_at_step=nodes_at_step, val_ratio=val_ratio,
                         conditional_to_hour=conditional_to_hour, **scengen_kwgs)
        self.n_sa = n_sa
        self.target_col = target_col

    def fit(self, x:pd.DataFrame, y:pd.DataFrame=None):
        x, y, x_val, y_val = self.train_val_split(x, x[[self.target_col]])
        hw_target = hankel(y_val.iloc[1:].values, self.n_sa)
        super().fit(x_val.iloc[:-self.n_sa, :], pd.DataFrame(hw_target, index=x_val.index[:hw_target.shape[0]]))
        return self

    def predict(self, x:pd.DataFrame, **kwargs):
        y_hat = x[self.target_col].values.reshape(-1, 1) * np.ones((1, self.n_sa))
        return pd.DataFrame(y_hat, index=x.index)

    def _predict_quantiles(self, x:pd.DataFrame, **kwargs):
        preds = np.expand_dims(self.predict(x), -1) * np.ones((1, 1, len(self.q_vect)))
        if self.conditional_to_hour:
            for h in np.unique(x.index.hour):
                preds[x.index.hour == h, :, :] += np.expand_dims(self.err_distr[h], 0)
        else:
            preds += np.expand_dims(self.err_distr, 0)

        return preds


class SeasonalPersistent(ScenarioGenerator):
    def __init__(self, target_col, seasonality=1, n_sa=1, q_vect=None, val_ratio=None, nodes_at_step=None,
                 conditional_to_hour=False, **scengen_kwgs):
        super().__init__(q_vect, nodes_at_step=nodes_at_step, val_ratio=val_ratio,
                         conditional_to_hour=conditional_to_hour, **scengen_kwgs)
        self.seasonality = seasonality
        self.n_sa = n_sa
        self.target_col = target_col
        self.state = None

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None):
        x, y, x_val, y_val = self.train_val_split(x, x[[self.target_col]])
        hw_target = hankel(y_val.iloc[1:].values, self.n_sa)
        self.state = x[[self.target_col]].iloc[-self.seasonality:].values
        super().fit(x_val.iloc[:-self.n_sa, :], pd.DataFrame(hw_target, index=x_val.index[:hw_target.shape[0]]))
        return self

    def predict(self, x: pd.DataFrame, **kwargs):
        values = np.hstack([self.state.ravel(), x[self.target_col].values])
        y_hat = hankel(values, self.n_sa)
        return pd.DataFrame(y_hat[:len(x), :], index=x.index)

    def _predict_quantiles(self, x: pd.DataFrame, **kwargs):
        preds = np.expand_dims(self.predict(x), -1) * np.ones((1, 1, len(self.q_vect)))
        if self.conditional_to_hour:
            for h in np.unique(x.index.hour):
                preds[x.index.hour == h, :, :] += np.expand_dims(self.err_distr[h], 0)
        else:
            preds += np.expand_dims(self.err_distr, 0)

        return preds


class DiscreteDistr(ScenarioGenerator):
    def __init__(self, period='1d', q_vect=None, val_ratio=None, nodes_at_step=None,
                 conditional_to_hour=False, **scengen_kwgs):
        super().__init__(q_vect, nodes_at_step=nodes_at_step, val_ratio=val_ratio,
                         conditional_to_hour=conditional_to_hour, **scengen_kwgs)
        self.n_sa = None
        self.period = period
        self.target_names = None
        self.y_distributions = None
        self.support = None

    def fit(self, x:pd.DataFrame, y:pd.DataFrame):
        self.n_sa = y.shape[1]
        # infer sampling time
        sampling_time = pd.infer_freq(x.index)

        # retrieve support of target distribution
        support = np.unique(y.values)
        self.support = support
        print('support of target distribution:', support)

        # Create a new column for the time within a day (hours and minutes)
        time_of_day = y.index.floor(sampling_time).time

        # Group by this time of day and calculate the mean across all days
        mean_by_time_of_day = {}
        for v in support:
            mean_by_time_of_day[v] = y.groupby(time_of_day).agg(lambda x: np.sum(x==v))

        y_distrib = pd.concat(mean_by_time_of_day, axis=1)
        # swap levels of the multiindex
        y_distrib = y_distrib.swaplevel(0, 1, axis=1).sort_index(axis=1)

        # normalize the distribution for each variable in level 0
        for t in y_distrib.columns.levels[0]:
            y_distrib[t] = y_distrib[t] / (y_distrib[t].sum(axis=1).values.reshape(-1, 1)+1e-12)

        # store the distribution
        self.y_distributions = y_distrib
        self.target_names = list(y_distrib.columns.levels[0])
        self.target_cols = [f'{t}_t+{i}' for t in self.target_names for i in range(1, self.n_sa+1)]
        return self

    def predict(self, x, **kwargs):
        return (self.predict_probabilities(x) * np.tile(self.support.reshape(1, -1), self.n_sa)).groupby(level=0, axis=1).sum()

    def predict_probabilities(self, x, **kwargs):
        # infer sampling time
        sampling_time = pd.infer_freq(x.index)
        # Create a new column for the time within a day (hours and minutes)
        time_of_day = pd.Series(x.index.floor(sampling_time).time)

        # retrieve the distribution for the time of day
        y_hat = {}
        for t in self.target_names:
            pres_t = [time_of_day.map(self.y_distributions[t].loc[:, i]).rename(f'{i}') for i in
                      self.y_distributions[t].columns]
            y_hat[t] = pd.concat(pres_t, axis=1)

        return pd.concat(y_hat, axis=1)

