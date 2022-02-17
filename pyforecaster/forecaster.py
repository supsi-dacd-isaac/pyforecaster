import numpy as np
import pandas as pd
import logging
from typing import Union
from . import COPULA_MAP
from scipy.interpolate import interp1d
from .scenred import scenred
from abc import abstractmethod
from lightgbm import LGBMRegressor, Dataset, train
from sklearn.linear_model import RidgeCV, LinearRegression


class ScenarioGenerator:
    def __init__(self, **scengen_kwgs):
        self.scengen = ScenGen(**scengen_kwgs)

    def fit(self, x, y):
        self.scengen.fit(y, x)

    @abstractmethod
    def predict(self, x, **kwargs):
        pass

    @abstractmethod
    def predict_quantiles(self, x, **kwargs):
        pass

    def predict_scenarios(self, x, n_scen=100, q_vect=None, **predict_q_kwargs):
        # retrieve quantiles from child class
        quantiles = self.predict_quantiles(x, **predict_q_kwargs)
        scenarios = self.scengen.predict(quantiles, n_scen=n_scen, x=x, kind='scenarios', q_vect=q_vect)
        return scenarios

    def predict_trees(self, x, n_scen=100, q_vect=None, scenarios_per_step=None, **predict_q_kwargs):

        # retrieve quantiles from child class
        quantiles = self.predict_quantiles(x, **predict_q_kwargs)

        trees = self.scengen.predict(quantiles, n_scen=n_scen, x=x, kind='tree', q_vect=q_vect,
                                         scenarios_per_step=scenarios_per_step)

        # if we predicted just one step just return a nx object, not a list
        if len(trees) == 1:
            trees = trees[0]
        return trees


class LinearForecaster(ScenarioGenerator):
    def __init__(self, kind='linear', **scengen_kwgs):
        super().__init__(**scengen_kwgs)
        self.m = None
        self.err_distr = None
        self.q_vect = np.linspace(0,1,11)[1:-1]
        self.kind = kind

    def fit(self, x, y):
        super().fit(x, y)
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, pd.DataFrame):
            y = y.values
        if self.kind == 'linear':
            self.m = LinearRegression().fit(x, y)
        elif self.kind == 'ridge':
            self.m = RidgeCV(alphas=10 ** np.linspace(-2, 8, 9)).fit(x, y)
        preds = self.predict(x)
        self.err_distr = np.quantile(preds-y, self.q_vect, axis=0).T
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
    def __init__(self, lgb_pars, **scengen_kwgs):
        super().__init__(**scengen_kwgs)
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

    def fit(self, x, y):
        super().fit(x, y)
        for i in range(y.shape[1]):
            lgb_data = Dataset(x, y.iloc[:, i].values.ravel())
            m = train(self.lgb_pars, lgb_data)
            self.m.append(m)

        preds = self.predict(x)
        self.err_distr = np.quantile(preds.values-y.values, self.q_vect, axis=0).T
        return self

    def predict(self, x, **kwargs):
        preds = []
        for m in self.m:
            preds.append(m.predict(x).reshape(-1, 1))
        return pd.DataFrame(np.hstack(preds), index=x.index)

    def predict_quantiles(self, x, **kwargs):
        preds = self.predict(x)
        return np.expand_dims(preds, -1) + np.expand_dims(self.err_distr, 0)

class ScenGen:
    def __init__(self, copula_type: str = 'HourlyGaussianCopula', **copula_kwargs):
        self.logger = get_logger()
        self.copula = COPULA_MAP[copula_type](**copula_kwargs)

    def fit(self, y: pd.DataFrame, x: pd.DataFrame = None):
        """
        Estimate a multivariate copula linking timesteps of forecasts
        :param y: pd.DataFrame of targets. Columns must refer to successive time-steps
        :param x: pd.DataFrame of covariates from which the copula depends on
        :return: multivariate copulas for different hour of the day
        """
        if x is not None:
            assert isinstance(x.index, pd.DatetimeIndex), 'index of the series/dataframe must be a pd.DatetimeIndex'

        # estimate covariances between time-steps, conditional on the hour of the day
        self.copula.fit(y, x)
        return self

    def predict(self, quantiles: Union[pd.DataFrame, np.ndarray], n_scen: int = 100, x: pd.DataFrame = None,
                kind='tree', q_vect=None, scenarios_per_step=None):
        """
        :param quantiles: predicted quantiles from the forecaster. quantiles has the following structure, depending on
                          type:
                          pd.DataFrame: multiindex DataFrame, index: timestamp, outer level (level 0): quantiles,
                                                inner level (level 1): step ahead.
                          np.ndarray: (n_obs, n_steps, n_quantiles)

        :param n_scen:
        :param x: features, pd.DataFrame with shape (n_obs, n_features)
        :param kind: kind of output required, could be in {'scenarios', 'tree'}
        :param q_vect: the alpha values to which the quantiles DataFrame/matrix refers to. If not passed, it is assumed
                       that quantiles are equally spaced and generated by:
                       np.linspace(0,1,n_quantiles+2)[1:-1]
        :return:
        """

        if isinstance(quantiles, pd.DataFrame):
            if x is None:
                x = pd.DataFrame(columns=[0], index=quantiles.index)
            quantiles = np.dstack([quantiles[q].values for q in quantiles.columns.get_level_values(0).unique()])
        else:
            assert x is not None, 'if quantiles are not pd.DataFrame, x must be passed, and must be a pd.DataFrame ' \
                                  'with DatetimeIndex index'
        # if scenarios_per_step not passed, assume linear function from 1 to n_scen
        if scenarios_per_step is None:
            scenarios_per_step = np.linspace(1, n_scen, quantiles.shape[1], dtype=int)

        if q_vect is None:
            self.logger.warning('q_vect was not passed, inferring it using equally-spaced quantiles')
            q_vect = np.linspace(0, 1, quantiles.shape[-1] + 2)[1:-1]

        assert len(q_vect) == quantiles.shape[-1], 'q_vect must contain the alpha values of quantiles third ' \
                                                        'dimension. q_vect length: {} differs from quantile third ' \
                                                        'dimension: {}'.format(len(q_vect), quantiles.shape[-1])

        # copula samples of dimension n_obs, n_steps, n_scen
        copula_samples = self.copula.sample(x, n_scen)

        scenarios = np.zeros((quantiles.shape[0], quantiles.shape[1], n_scen))

        # for each temporal observation, generate scenarios from copula samples
        for t, copula_sample_t, quantiles_t in zip(range(len(copula_samples)), copula_samples, quantiles):
            scenarios[t, :] = interp_scens(copula_sample_t, quantiles_t, q_vect)

        if kind == 'scenarios':
            return scenarios
        elif kind == 'tree':
            trees = []
            for scenarios_t in scenarios:
                [S_init0, P_sn, J_sn, Me_sn, nx_tree] = scenred(scenarios_t, nodes=scenarios_per_step)
                trees.append(nx_tree)
            return trees
        else:
            raise ValueError('prediction kind not recognized, should be in [scenarios, tree]')


def interp_scens(copula_samples, quantiles_mat, q_vect):
    """
    Interpolate at the sampled copula_sample quantile the quantile_mat from the forecaster, for a single observation /
    time prediction.
    For each timestep we draw n_scen scenarios, so the interpolation function can be reused.
    for each observation, we have the copula_sample matrix (n_steps, n_scen) and the
    quantile matrix (n_step, n_quantile)

    :param copula_samples: (n_steps, n_scen)
    :param quantiles_mat: forecasted quantiles matrix (n_step, n_quantile)
    :param q_vect: alpha levels of the quantiles returned by the forecaster
    :return:
    """
    # for each step
    scenarios = np.zeros_like(copula_samples)
    for step, quantiles_h in enumerate(quantiles_mat):
        # interp function for the ith step ahead
        q_interpfun_at_step = interp1d(q_vect, quantiles_h, fill_value='extrapolate')
        scenarios[step, :] = q_interpfun_at_step(copula_samples[step,:])
    return scenarios


def get_logger(level=logging.INFO):
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s')
    logger.setLevel(level)
    return logger