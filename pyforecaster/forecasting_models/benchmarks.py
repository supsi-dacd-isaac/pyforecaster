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

    def predict_quantiles(self, x:pd.DataFrame, **kwargs):
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

    def predict_quantiles(self, x: pd.DataFrame, **kwargs):
        preds = np.expand_dims(self.predict(x), -1) * np.ones((1, 1, len(self.q_vect)))
        if self.conditional_to_hour:
            for h in np.unique(x.index.hour):
                preds[x.index.hour == h, :, :] += np.expand_dims(self.err_distr[h], 0)
        else:
            preds += np.expand_dims(self.err_distr, 0)

        return preds


class DiscreteDistr(ScenarioGenerator):
    def __init__(self, period='1d', n_sa=1, q_vect=None, val_ratio=None, nodes_at_step=None,
                 conditional_to_hour=False, **scengen_kwgs):
        super().__init__(q_vect, nodes_at_step=nodes_at_step, val_ratio=val_ratio,
                         conditional_to_hour=conditional_to_hour, **scengen_kwgs)
        self.n_sa = n_sa
        self.period = period

    def fit(self, x:pd.DataFrame, y:pd.DataFrame):
        # infer sampling time
        sampling_time = pd.infer_freq(x.index)
        # aggregate by