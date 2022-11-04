import numpy as np
import pandas as pd

from abc import abstractmethod
from lightgbm import LGBMRegressor, Dataset, train
from sklearn.linear_model import RidgeCV, LinearRegression
from pyforecaster.scenarios_generator import ScenGen


def train_val_split(x, y, val_ratio):
    n_val = int(x.shape[0] * val_ratio)
    x_val, y_val = x.iloc[-n_val:, :], y.iloc[-n_val:, :]
    x, y = x.iloc[:n_val, :], y.iloc[:n_val, :]
    return x, y, x_val, y_val

class ScenarioGenerator:
    def __init__(self, q_vect=None, **scengen_kwgs):
        self.q_vect = np.hstack([0.01, np.linspace(0,1,11)[1:-1], 0.99]) if q_vect is None else q_vect
        self.scengen = ScenGen(**scengen_kwgs)

    def fit(self, x:pd.DataFrame, y:pd.DataFrame):
        self.scengen.fit(y, x)

    @abstractmethod
    def predict(self, x, **kwargs):
        pass

    @abstractmethod
    def predict_quantiles(self, x, **kwargs):
        pass

    def predict_scenarios(self, x, n_scen=100, **predict_q_kwargs):
        # retrieve quantiles from child class
        quantiles = self.predict_quantiles(x, **predict_q_kwargs)
        scenarios = self.scengen.predict(quantiles, n_scen=n_scen, x=x, kind='scenarios', q_vect=self.q_vect)
        return scenarios

    def predict_trees(self, x, n_scen=100, scenarios_per_step=None, **predict_q_kwargs):

        # retrieve quantiles from child class
        quantiles = self.predict_quantiles(x, **predict_q_kwargs)

        trees = self.scengen.predict(quantiles, n_scen=n_scen, x=x, kind='tree', q_vect=self.q_vect,
                                         scenarios_per_step=scenarios_per_step)

        # if we predicted just one step just return a nx object, not a list
        if len(trees) == 1:
            trees = trees[0]
        return trees


class LinearForecaster(ScenarioGenerator):
    def __init__(self, q_vect=None, val_ratio=None, kind='linear', **scengen_kwgs):
        super().__init__(q_vect, **scengen_kwgs)
        self.m = None
        self.err_distr = None
        self.kind = kind
        self.val_ratio = val_ratio

    def fit(self, x, y):
        if self.val_ratio is not None:
            x, y, x_val, y_val = train_val_split(x, y, self.val_ratio)

        super().fit(x, y)
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, pd.DataFrame):
            y = y.values
        if self.kind == 'linear':
            self.m = LinearRegression().fit(x, y)
        elif self.kind == 'ridge':
            self.m = RidgeCV(alphas=10 ** np.linspace(-2, 8, 9)).fit(x, y)

        if self.val_ratio is None:
            preds = self.predict(x)
            self.err_distr = np.quantile(preds-y, self.q_vect, axis=0).T
        else:
            preds = self.predict(x_val)
            self.err_distr = np.quantile(preds - y_val, self.q_vect, axis=0).T
        return self

    def predict(self, x, **kwargs):
        if isinstance(x, pd.DataFrame):
            x = x.values
        return self.m.predict(x)

    def predict_quantiles(self, x, **kwargs):
        if isinstance(x, pd.DataFrame):
            x = x.values
        preds = self.predict(x)
        return np.expand_dims(preds, -1) + np.expand_dims(self.err_distr, 0)


class LGBForecaster(ScenarioGenerator):
    def __init__(self, lgb_pars, q_vect=None, val_ratio=None, **scengen_kwgs):
        super().__init__(q_vect, **scengen_kwgs)
        self.m = []
        self.lgb_pars = {"objective": "regression",
                         "max_depth": 20,
                         "num_leaves": 100,
                         "learning_rate": 0.1,
                         "verbose": -1,
                         "metric": "l2",
                         "min_data": 4,
                         "num_threads": 8}
        self.lgb_pars.update(lgb_pars)
        self.err_distr = None
        self.q_vect = np.linspace(0,1,11)[1:-1]
        self.val_ratio = val_ratio

    def fit(self, x, y):
        if self.val_ratio is not None:
            x, y, x_val, y_val = train_val_split(x, y, self.val_ratio)

        super().fit(x, y)
        for i in range(y.shape[1]):
            lgb_data = Dataset(x, y.iloc[:, i].values.ravel())
            m = train(self.lgb_pars, lgb_data)
            self.m.append(m)

        if self.val_ratio is None:
            preds = self.predict(x)
            self.err_distr = np.quantile(preds.values-y.values, self.q_vect, axis=0).T
        else:
            preds = self.predict(x_val)
            self.err_distr = np.quantile(preds.values - y_val.values, self.q_vect, axis=0).T

        return self

    def predict(self, x, **kwargs):
        preds = []
        for m in self.m:
            preds.append(m.predict(x).reshape(-1, 1))
        return pd.DataFrame(np.hstack(preds), index=x.index)

    def predict_quantiles(self, x, **kwargs):
        preds = self.predict(x)
        return np.expand_dims(preds, -1) + np.expand_dims(self.err_distr, 0)


