from quantile_forest import RandomForestQuantileRegressor
import pandas as pd
from pyforecaster.utilities import get_logger
from tqdm import tqdm
from pyforecaster.forecaster import ScenarioGenerator, encode_categorical
import numpy as np
import concurrent.futures
from time import time
from functools import partial

class QRF(ScenarioGenerator):
    def __init__(self, n_estimators=100, q_vect=None, val_ratio=None, nodes_at_step=None,
                 metadata_features=None, n_single=0, red_frac_multistep=0.5, tol_period='1h',
                keep_last_n_lags=0, keep_last_seconds=0, criterion="squared_error", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0,
                 max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None,
                 random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, parallel=True,
                 max_parallel_workers=8, **scengen_kwgs):
        """
        :param n_single: number of single models, should be less than number of step ahead predictions. The rest of the
                         steps ahead are forecasted by a global model
        :param red_frac_multistep: reduce the observations used for training the global model

        :param metadata_features: list of features that shouldn't be pruned
        :param learning_rate:
        :param tol_period:
        :param q_vect:
        :param scengen_kwgs:
        """

        super().__init__(q_vect, val_ratio=val_ratio, nodes_at_step=nodes_at_step, **scengen_kwgs)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_single = n_single
        self.red_frac_multistep = red_frac_multistep
        self.tol_period = tol_period
        self.models = []
        self.multi_step_model = None
        self.n_multistep = None
        self.logger = get_logger()
        self.metadata_features = metadata_features if metadata_features is not None\
            else []
        self.keep_last_n_lags = keep_last_n_lags
        self.keep_last_seconds = keep_last_seconds
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.max_samples = max_samples
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_samples_leaf = max_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.ccp_alpha = ccp_alpha
        self.parallel = parallel
        self.max_parallel_workers = max_parallel_workers

        self.qrf_pars = {
            "n_estimators":n_estimators,
            "bootstrap": bootstrap,
            "oob_score": oob_score,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": verbose,
            "warm_start": warm_start,
            "max_samples": max_samples,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_samples_leaf": max_samples_leaf,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "max_features": max_features ,
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease,
            "default_quantiles":q_vect if q_vect is not None else 'mean',
            "criterion":criterion,
             "ccp_alpha":ccp_alpha
        }

    def _fit(self, i, x, y, sample_weight=None):
        x_i = self.dataset_at_stepahead(x, i, self.metadata_features, formatter=self.formatter,
                                        logger=self.logger, method='periodic', keep_last_n_lags=self.keep_last_n_lags,
                                        keep_last_seconds=self.keep_last_seconds,
                                        tol_period=self.tol_period)
        model = RandomForestQuantileRegressor(**self.qrf_pars).fit(x_i, y.iloc[:, i], sample_weight=sample_weight)
        return model

    @encode_categorical
    def fit(self, x, y, sample_weight=None):
        x, y, x_val, y_val = self.train_val_split(x, y)
        if self.parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_parallel_workers) as executor:
                self.models = [i for i in tqdm(executor.map(partial(self._fit, x=x, y=y, sample_weight=sample_weight), range(self.n_single)),total=self.n_single)]
        else:
            for i in tqdm(range(self.n_single)):
                model = self._fit(i, x, y, sample_weight=sample_weight)
                self.models.append(model)

        n_sa = y.shape[1]
        self.n_multistep = n_sa - self.n_single
        if self.n_multistep>0:
            t_0 = time()
            qrf_pars_global = self.qrf_pars.copy()
            if 'n_jobs' in qrf_pars_global and qrf_pars_global['n_jobs'] is not None and qrf_pars_global['n_jobs'] > 0:
                qrf_pars_global['n_jobs'] *= self.max_parallel_workers
            self.multi_step_model = RandomForestQuantileRegressor(**qrf_pars_global).fit(x, y.iloc[:, -self.n_multistep:])
            self.logger.info('QRF multistep fitted in {:0.2e} s, x shape: [{}, {}]'.format(time() - t_0,
                                                                                                     x.shape[0],
                                                                                                     x.shape[1]))
        super().fit(x_val, y_val)
        return self

    @encode_categorical
    def predict(self, x, **kwargs):
        preds = []
        period = kwargs['period'] if 'period' in kwargs else '24h'
        if self.parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
                preds = [i for i in tqdm(executor.map(partial(self._predict, x=x, period=period, **kwargs), range(self.n_single)),total=self.n_single)]

        else:
            for i in range(self.n_single):
                p = self._predict(i, x, period, **kwargs)
                preds.append(p)
        x_pd = x
        if self.n_multistep>0:
            preds.append(self.predict_multi_step(x, quantiles=list(kwargs['quantiles']) if 'quantiles' in kwargs else 'mean'))
        preds = np.dstack(preds)
        if 'quantiles' in kwargs:
            if str(kwargs['quantiles']) == 'mean':
                preds = pd.DataFrame(np.atleast_2d(np.squeeze(preds)), index=x.index, columns=self.target_cols)
            else:
                if len(preds.shape) == 2:
                    preds = np.expand_dims(preds, 0)
                preds = np.swapaxes(preds, 1, 2)
                preds = self.quantiles_to_df(preds, index=x.index, q_vect=kwargs['quantiles'])
        else:
            preds = pd.DataFrame(np.atleast_2d(np.squeeze(preds)), index=x.index, columns=self.target_cols)
        y_hat = self.anti_transform(x, preds)
        return y_hat

    def _predict(self, i, x, period, **kwargs):
        x_i = self.dataset_at_stepahead(x, i, self.metadata_features, formatter=self.formatter,
                                        logger=self.logger, method='periodic', keep_last_n_lags=self.keep_last_n_lags,
                                        keep_last_seconds=self.keep_last_seconds,
                                        tol_period=self.tol_period, period=period)
        p = self.models[i].predict(x_i, quantiles=list(kwargs['quantiles']) if 'quantiles' in kwargs else 'mean')
        if len(p.shape) == 1:
            p = np.expand_dims(p, 1)
        return p

    def predict_multi_step(self, x, quantiles='mean'):
        preds = self.multi_step_model.predict(x, quantiles)
        if len(preds.shape) == 1:
            preds = np.expand_dims(preds, 1)
        if len(preds.shape) == 2:
            preds = np.expand_dims(preds, 1)
        else:
            preds = np.swapaxes(preds, 1, 2)
        return preds

    @staticmethod
    def dataset_at_stepahead(df, target_col_num, metadata_features, formatter, logger, method='periodic', keep_last_n_lags=1, period="24h",
                             tol_period='1h', keep_last_seconds=0):
        if formatter is None:
            logger.warning('dataset_at_stepahead returned the unmodified dataset since there is no self.formatter')
            return df
        else:
            return formatter.prune_dataset_at_stepahead(df, target_col_num, metadata_features=metadata_features, method=method,
                                                        period=period, keep_last_n_lags=keep_last_n_lags, keep_last_seconds=keep_last_seconds,
                                                        tol_period=tol_period)

    @staticmethod
    def quantiles_to_numpy(q_hat: pd.DataFrame):
        n_taus = len(q_hat.columns.get_level_values(1).unique())
        q_hat = q_hat.values
        q_hat = np.reshape(q_hat, (q_hat.shape[0], n_taus, -1))
        q_hat = np.swapaxes(q_hat, 1, 2)
        return q_hat

    def predict_quantiles(self, x, dataframe=True, **kwargs):
        preds = self.predict(x, quantiles=list(kwargs['quantiles']) if 'quantiles' in kwargs else self.q_vect)
        if dataframe is False:
            return self.quantiles_to_numpy(preds)
        return preds
