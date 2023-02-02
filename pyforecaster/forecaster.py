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
    def __init__(self, q_vect=None, nodes_at_step=None, **scengen_kwgs):
        self.q_vect = np.hstack([0.01, np.linspace(0,1,11)[1:-1], 0.99]) if q_vect is None else q_vect
        self.scengen = ScenGen(q_vect=self.q_vect, nodes_at_step=nodes_at_step, **scengen_kwgs)
        self.online_tree_reduction = scengen_kwgs['online_tree_reduction'] if 'online_tree_reduction' in \
                                                                              scengen_kwgs.keys() else True

    def fit(self, x:pd.DataFrame, y:pd.DataFrame):
        self.scengen.fit(y, x)

    @abstractmethod
    def predict(self, x, **kwargs):
        pass

    @abstractmethod
    def predict_quantiles(self, x, **kwargs):
        pass

    def predict_scenarios(self, x, n_scen=100, random_state=None, **predict_q_kwargs):
        # retrieve quantiles from child class
        quantiles = self.predict_quantiles(x, **predict_q_kwargs)
        scenarios = self.scengen.predict_scenarios(quantiles, n_scen=n_scen, x=x, random_state=random_state)
        return scenarios

    def predict_trees(self, x, n_scen=100, nodes_at_step=None, init_obs=None, random_state=None,
                      **predict_q_kwargs):
        """
        :param x:
        :param n_scen:
        :param scenarios_per_step:
        :param init_obs: if init_obs is not None it should be a vector with length x.shape[0]. init_obs are then used as
                         first observation of the tree. This is done to build a tree with the last realisation of the
                         forecasted value as root node. Useful for stochastic control.
        :param predict_q_kwargs:
        :return:
        """

        if self.online_tree_reduction:
            # retrieve quantiles from child class
            quantiles = self.predict_quantiles(x, **predict_q_kwargs)

            trees = self.scengen.predict_trees(quantiles=quantiles, n_scen=n_scen, x=x,
                                         nodes_at_step=nodes_at_step, init_obs=init_obs,
                                         random_state=random_state)
        else:
            predictions = self.predict(x, **predict_q_kwargs)

            trees = self.scengen.predict_trees(predictions=predictions, n_scen=n_scen, x=x, init_obs=init_obs,
                                         random_state=random_state)

        # if we predicted just one step just return a nx object, not a list
        if len(trees) == 1:
            trees = trees[0]
        return trees


class LinearForecaster(ScenarioGenerator):
    def __init__(self, q_vect=None, val_ratio=None, kind='linear', **scengen_kwgs):
        super().__init__(q_vect, **scengen_kwgs)
        self.m = None
        self.err_distr = {}
        self.kind = kind
        self.val_ratio = val_ratio

    def fit(self, x:pd.DataFrame, y:pd.DataFrame):
        if self.val_ratio is not None:
            x, y, x_val, y_val = train_val_split(x, y, self.val_ratio)

        #if isinstance(x, pd.DataFrame):
        #    x = x.values
        #if isinstance(y, pd.DataFrame):
        #    y = y.values
        if self.kind == 'linear':
            self.m = LinearRegression().fit(x, y)
        elif self.kind == 'ridge':
            self.m = RidgeCV(alphas=10 ** np.linspace(-2, 8, 9)).fit(x, y)

        if self.val_ratio is None:
            preds = self.predict(x)
            errs = pd.DataFrame(preds.values-y.values, index=x.index)
            super().fit(x, errs)
        else:
            preds = self.predict(x_val)
            errs = pd.DataFrame(preds.values-y_val.values, index=x_val.index)
            super().fit(x_val, errs)

        self.err_distr = {}
        for h in np.unique(x.index.hour):
            self.err_distr[h] = np.quantile(errs.loc[errs.index.hour == h, :], self.q_vect, axis=0).T

        return self

    def predict(self, x:pd.DataFrame, **kwargs):
        return pd.DataFrame(self.m.predict(x), index=x.index)

    def predict_quantiles(self, x:pd.DataFrame, **kwargs):
        preds = np.expand_dims(self.predict(x), -1) * np.ones((1, 1, len(self.q_vect)))
        for h in np.unique(x.index.hour):
            preds[x.index.hour == h, :, :] += np.expand_dims(self.err_distr[h], 0)
        return preds


class LGBForecaster(ScenarioGenerator):
    def __init__(self, lgb_pars=None, q_vect=None, val_ratio=None, **scengen_kwgs):
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
        if lgb_pars is not None:
            self.lgb_pars.update(lgb_pars)
        self.err_distr = {}
        self.val_ratio = val_ratio

    def fit(self, x, y):
        if self.val_ratio is not None:
            x, y, x_val, y_val = train_val_split(x, y, self.val_ratio)

        for i in range(y.shape[1]):
            lgb_data = Dataset(x, y.iloc[:, i].values.ravel())
            m = train(self.lgb_pars, lgb_data)
            self.m.append(m)

        if self.val_ratio is None:
            preds = self.predict(x)
            errs = pd.DataFrame(preds.values-y.values, index=x.index)
            super().fit(x, errs)
        else:
            preds = self.predict(x_val)
            errs = pd.DataFrame(preds.values - y_val.values, index=x_val.index)
            super().fit(x_val, errs)

        self.err_distr = {}
        for h in np.unique(x.index.hour):
            self.err_distr[h] = np.quantile(errs.loc[errs.index.hour == h], self.q_vect, axis=0).T

        return self

    def predict(self, x, **kwargs):
        preds = []
        for m in self.m:
            preds.append(m.predict(x).reshape(-1, 1))
        return pd.DataFrame(np.hstack(preds), index=x.index)

    def predict_quantiles(self, x, **kwargs):
        preds = np.expand_dims(self.predict(x), -1) * np.ones((1, 1, len(self.q_vect)))
        for h in np.unique(x.index.hour):
            preds[x.index.hour == h, :, :] += np.expand_dims(self.err_distr[h], 0)
        return preds


