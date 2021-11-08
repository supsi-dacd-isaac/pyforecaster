import numpy as np
import pandas as pd
import logging
from typing import Union
from pyforecaster import COPULA_MAP
from pyforecaster import copula
from scipy.interpolate import interp1d


class ScenGen:
    def __init__(self, q_vect, copula_type: str = 'HourlyGaussianCopula', **copula_kwargs):
        self.logger = get_logger()
        self.copula = COPULA_MAP[copula_type](**copula_kwargs)
        self.q_vect = q_vect

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

    def predict(self, quantiles: Union[pd.DataFrame, np.ndarray], n_scen: int = 100, x: pd.DataFrame = None):
        """
        :param quantiles: predicted quantiles from the forecaster. quantiles has the following structure, depending on
                          type:
                          pd.DataFrame: multiindex DataFrame, index: timestamp, outer level (level 0): quantiles,
                                                inner level (level 1): step ahead.
                          np.ndarray: (n_obs, n_steps, n_quantiles)

        :param n_scen:
        :param x: features, pd.DataFrame with shape (n_obs, n_features)
        :return:
        """

        if isinstance(quantiles, pd.DataFrame):
            if x is None:
                x = pd.DataFrame(columns=[0], index=quantiles.index)
            quantiles = np.dstack([quantiles[q].values for q in self.q_vect])
        else:
            assert x is not None, 'if quantiles are not pd.DataFrame, x must be passed, and must be a pd.DataFrame ' \
                                  'with DatetimeIndex index'


        assert len(self.q_vect) == quantiles.shape[-1], 'q_vect must contain the alpha values of quantiles third ' \
                                                        'dimension. q_vect length: {} differs from quantile third ' \
                                                        'dimension: {}'.format(len(self.q_vect), quantiles.shape[-1])

        # copula samples of dimension n_obs, n_steps, n_scen
        copula_samples = self.copula.sample(x, n_scen)

        scenarios = np.zeros((quantiles.shape[0], quantiles.shape[1], n_scen ))

        # for each temporal observation, generate scenarios from copula samples
        for t, copula_sample_t, quantiles_t in zip(range(len(copula_samples)), copula_samples, quantiles):
            scenarios[t, :] = interp_scens(copula_sample_t, quantiles_t, self.q_vect)

        return scenarios


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
