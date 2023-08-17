from typing import Union
from scipy.interpolate import interp1d
from .scenred import superimpose_signal_to_tree
from pyforecaster.utilities import get_logger
import pandas as pd
import pyforecaster.dictionaries
import numpy as np
from inspect import signature
from copy import deepcopy
from multiprocessing import cpu_count
import concurrent
from tqdm import tqdm
from functools import partial

class ScenGen:
    def __init__(self, copula_type: str = 'HourlyGaussianCopula', tree_type:str = 'ScenredTree',
                 online_tree_reduction=True, q_vect=None, nodes_at_step=None, max_iterations=100,
                 parallel_preds=False, **kwargs):
        self.logger = get_logger()
        self.copula_type = copula_type
        self.tree_type = tree_type
        self.copula_class = pyforecaster.dictionaries.COPULA_MAP[copula_type]
        self.tree_class = pyforecaster.dictionaries.TREE_MAP[tree_type]
        copula_kwargs = {p: kwargs[p] for p in signature(self.copula_class).parameters.keys() if p in kwargs}
        tree_kwargs = {p: kwargs[p] for p in signature(self.tree_class).parameters.keys() if p in kwargs}
        self.copula = self.copula_class(**copula_kwargs)
        self.tree = self.tree_class(nodes_at_step=nodes_at_step, **tree_kwargs)
        self.online_tree_reduction = online_tree_reduction
        self.trees = {}
        self.k_max = max_iterations
        self.q_vect = np.hstack([0.01, np.linspace(0,1,11)[1:-1], 0.99]) if q_vect is None else q_vect
        self.parallel_preds = parallel_preds

    def fit(self, y: pd.DataFrame, x: pd.DataFrame = None, n_scen=100, **copula_kwargs):
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

        # pre-fit trees if required
        if not self.online_tree_reduction:
            if self.copula_type == 'HourlyGaussianCopula':
                err_distr = {}
                for h in np.unique(x.index.hour):
                    filt_h = y.index.hour == h
                    err_distr[h] = np.expand_dims(np.quantile(y.loc[filt_h, :], self.q_vect, axis=0).T, 0)

                for h in range(24):
                    #scenarios = self.copula.sample(y_h.iloc[[0], :], n_scen, **copula_kwargs)
                    filt_h = y.index.hour == h
                    y_h = y.loc[filt_h, :]
                    scenarios = self.sample_scenarios(y_h, n_scen, err_distr[h], init_obs=None, **copula_kwargs)
                    start_tree = self.trees[0] if h>0 else None
                    self.trees[h], _, _, _ = self.tree.gen_tree(np.squeeze(scenarios), k_max=self.k_max, start_tree=start_tree)
            else:
                raise NotImplementedError('pre-fit currently not available for copulas other than HourlyGaussianCopula')

        return self

    def predict_scenarios(self, quantiles: Union[pd.DataFrame, np.ndarray]=None, n_scen: int = 100,
                          x: pd.DataFrame = None, init_obs=None, **copula_kwargs):
        """
        :param quantiles: predicted quantiles from the forecaster. quantiles has the following structure, depending on
                          type:
                          pd.DataFrame: multiindex DataFrame, index: timestamp, outer level (level 0): quantiles,
                                                inner level (level 1): step ahead.
                          np.ndarray: (n_obs, n_steps, n_quantiles)

        :param n_scen:
        :param x: features, pd.DataFrame with shape (n_obs, n_features)
        :param kind: kind of output required, could be in {'scenarios', 'tree'}
        :param init_obs: if init_obs is not None it should be a vector with length x.shape[0]. init_obs are then used as
                         first observation of the tree. This is done to build a tree with the last realisation of the
                         forecasted value as root node. Useful for stochastic control.
        :param copula_kwargs: contains kwargs for the sampling of the copula e.g. random_state
        :return:
        """

        if isinstance(quantiles, pd.DataFrame):
            if x is None:
                x = pd.DataFrame(columns=[0], index=quantiles.index)
            quantiles = np.dstack([quantiles[q].values for q in quantiles.columns.get_level_values(0).unique()])
        else:
            assert x is not None, 'if quantiles are not pd.DataFrame, x must be passed, and must be a pd.DataFrame ' \
                                  'with DatetimeIndex index'

        assert len(self.q_vect) == quantiles.shape[-1], 'q_vect must contain the alpha values of quantiles third ' \
                                                        'dimension. q_vect length: {} differs from quantile third ' \
                                                        'dimension: {}'.format(len(self.q_vect), quantiles.shape[-1])

        scenarios = self.sample_scenarios(x, n_scen, quantiles, init_obs, **copula_kwargs)
        return scenarios

    def predict_trees(self, predictions:pd.DataFrame=None, quantiles: Union[pd.DataFrame, np.ndarray] = None, n_scen: int = 100,
                          x: pd.DataFrame = None, init_obs=None, nodes_at_step=None, **copula_kwargs):
        trees = []
        if self.online_tree_reduction:
            assert quantiles is not None, 'if online_tree_reduction quantiles must be passed'
            if isinstance(quantiles, pd.DataFrame):
                if x is None:
                    x = pd.DataFrame(columns=[0], index=quantiles.index)
                quantiles = np.dstack([quantiles[q].values for q in quantiles.columns.get_level_values(0).unique()])
            else:
                assert x is not None, 'if quantiles are not pd.DataFrame, x must be passed, and must be a pd.DataFrame ' \
                                      'with DatetimeIndex index'
            scenarios = self.sample_scenarios(x, n_scen, quantiles, init_obs, **copula_kwargs)
            for scenarios_t in scenarios:
                nx_tree, _, _, _ = self.tree.gen_tree(scenarios_t, k_max=self.k_max, nodes_at_step=nodes_at_step)
                trees.append(nx_tree)
        else:
            assert predictions is not None, 'if online_tree_reduction is false, predictions must be passed'

            if self.parallel_preds:
                with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()-1) as executor:
                    trees = [i for i in tqdm(
                        executor.map(partial(tree_gen, trees=self.trees), predictions.iterrows()),
                        total=predictions.shape[0])]

            else:
                for t in predictions.iterrows():
                    nx_tree = tree_gen(t, self.trees)
                    trees.append(nx_tree)

        return trees

    def sample_scenarios(self, x, n_scen, quantiles, init_obs, **copula_kwargs):
        # copula samples of dimension n_obs, n_steps, n_scen
        copula_samples = self.copula.sample(x, n_scen, **copula_kwargs)

        if init_obs is not None:
            scenarios = np.zeros((quantiles.shape[0], quantiles.shape[1] + 1, n_scen))
            init_obs = np.atleast_1d(init_obs)
            assert len(init_obs) == scenarios.shape[0], 'length of init_obs was {}, while should have been equal to ' \
                                                        '{}'.format(len(init_obs), scenarios.shape[0])
            for t, init_o in enumerate(init_obs):
                scenarios[t, 0, :] = init_o + np.random.rand(n_scen)*1e-6
            # for each temporal observation, generate scenarios from copula samples
            for t, copula_sample_t, quantiles_t in zip(range(len(copula_samples)), copula_samples, quantiles):
                scenarios[t, 1:, :] = interp_scens(copula_sample_t, quantiles_t, self.q_vect)
        else:
            scenarios = np.zeros((quantiles.shape[0], quantiles.shape[1], n_scen))
            # for each temporal observation, generate scenarios from copula samples
            for t, copula_sample_t, quantiles_t in zip(range(len(copula_samples)), copula_samples, quantiles):
                scenarios[t, :] = interp_scens(copula_sample_t, quantiles_t, self.q_vect)
        return scenarios


def tree_gen(prediction_tuple, trees=None):
    hour = prediction_tuple[0].hour
    nx_tree = deepcopy(trees[hour])
    nx_tree = superimpose_signal_to_tree(prediction_tuple[1], nx_tree)
    return nx_tree

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