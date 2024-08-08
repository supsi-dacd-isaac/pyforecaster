import pandas as pd
from pyforecaster.utilities import get_logger
from tqdm import tqdm
from pyforecaster.forecaster import ScenarioGenerator
from lightgbm import LGBMRegressor
import numpy as np
import concurrent.futures
from time import time
from multiprocessing import Pool, cpu_count
from functools import partial

class LGBMHybrid(ScenarioGenerator):
    def __init__(self, device_type='cpu', max_depth=20, n_estimators=100, num_leaves=100, learning_rate=0.1, min_child_samples=20,
                 n_jobs=1, objective='regression', tol_period='1h', colsample_bytree=1,
                 colsample_bynode=1, verbose=-1, metric='l2', n_single=1,
                 red_frac_multistep=1, q_vect=None, val_ratio=None, nodes_at_step=None,
                 formatter=None, metadata_features=None, keep_last_n_lags=0, keep_last_seconds=0, parallel=False,
                 **scengen_kwgs):
        """
        :param n_single: number of single models, should be less than number of step ahead predictions. The rest of the
                         steps ahead are forecasted by a global model
        :param red_frac_multistep: reduce the observations used for training the global model
        :param formatter: the formatter used to produce the x,y df. Needed for step ahead feature pruning
        :param metadata_features: list of features that shouldn't be pruned
        :param n_estimators:
        :param learning_rate:
        :param tol_period:
        :param q_vect:
        :param scengen_kwgs:
        """

        super().__init__(q_vect, val_ratio=val_ratio, nodes_at_step=nodes_at_step, **scengen_kwgs)
        self.parallel = parallel
        self.device_type = device_type
        self.n_single = n_single
        self.red_frac_multistep = red_frac_multistep
        self.tol_period = tol_period

        self.max_depth = max_depth
        self.objective = objective
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.metric = metric
        self.min_child_samples = min_child_samples
        self.n_jobs = n_jobs
        self.colsample_bytree = colsample_bytree
        self.colsample_bynode = colsample_bynode

        self.models = []
        self.multi_step_model = None
        self.n_multistep = None
        self.formatter = formatter
        self.logger = get_logger()
        self.metadata_features = metadata_features if metadata_features is not None\
            else []
        self.keep_last_n_lags = keep_last_n_lags
        self.keep_last_seconds = keep_last_seconds


    def get_lgb_pars(self):
        lgb_pars = {"device_type": self.device_type,
                    "objective": self.objective,
                    "max_depth": self.max_depth,
                    "n_estimators": self.n_estimators,
                    "num_leaves": self.num_leaves,
                    "learning_rate": self.learning_rate,
                    "verbose": self.verbose,
                    "metric": self.metric,
                    "min_child_samples": self.min_child_samples,
                    "n_jobs": self.n_jobs,
                    "colsample_bytree": self.colsample_bytree,
                    "colsample_bynode": self.colsample_bynode
                    }
        return lgb_pars

    def fit(self, x, y):
        lgb_pars = self.get_lgb_pars()
        x, y, x_val, y_val = self.train_val_split(x, y)
        if self.parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()-2) as executor:
                self.models = list(executor.map(partial(self.fit_single, x=x, y=y, lgb_pars=lgb_pars), np.arange(self.n_single)))
        else:
            for i in tqdm(range(self.n_single)):
                x_i = self.dataset_at_stepahead(x, i,  self.metadata_features, formatter=self.formatter,
                                                logger=self.logger, method='periodic', keep_last_n_lags=self.keep_last_n_lags,
                                                keep_last_seconds=self.keep_last_seconds,
                                                tol_period=self.tol_period)
                self.models.append(LGBMRegressor(**lgb_pars).fit(x_i, y.iloc[:, i]))

        n_sa = y.shape[1]
        self.n_multistep = n_sa - self.n_single
        if self.n_multistep>0:
            x_pd = x.copy()
            x_pd.reset_index(drop=True, inplace=True)
            red_frac = np.maximum(1, self.red_frac_multistep*self.n_multistep)/self.n_multistep
            n_batch = int(len(x_pd)*red_frac)
            n_long = n_batch*self.n_multistep
            rand_idx = np.random.choice(x_pd.index, n_long).reshape(self.n_multistep, -1)

            x_long = []
            y_long = []
            for i, sa in enumerate(np.arange(self.n_single, self.n_single + self.n_multistep)):
                x_i = pd.concat([x_pd.loc[rand_idx[i], :].reset_index(drop=True), pd.Series(np.ones(n_batch) * i)], axis=1)
                y_long.append(y.iloc[rand_idx[i], sa])
                x_long.append(x_i)

            x_long = pd.concat(x_long, axis=0)
            y_long = pd.concat(y_long)

            t_0 = time()
            self.multi_step_model = LGBMRegressor(**lgb_pars).fit(x_long, y_long)
            self.logger.info('LGBMHybrid multistep fitted in {:0.2e} s, x shape: [{}, {}]'.format(time() - t_0,
                                                                                                     x.shape[0],
                                                                                                     x.shape[1]))
        super().fit(x_val, y_val)
        return self

    def fit_single(self, i, x, y, lgb_pars):
        x_i = self.dataset_at_stepahead(x, i, self.metadata_features, formatter=self.formatter,
                                        logger=self.logger, method='periodic', keep_last_n_lags=self.keep_last_n_lags,
                                        keep_last_seconds=self.keep_last_seconds,
                                        tol_period=self.tol_period)
        m = LGBMRegressor(**lgb_pars).fit(x_i, y.iloc[:, i])
        return m
    def predict(self, x, **kwargs):
        preds = []
        period = kwargs['period'] if 'period' in kwargs else '24h'
        for i in range(self.n_single):
            x_i = self.dataset_at_stepahead(x, i, self.metadata_features, formatter=self.formatter,
                                            logger=self.logger, method='periodic', keep_last_n_lags=self.keep_last_n_lags,
                                            keep_last_seconds=self.keep_last_seconds,
                                            tol_period=self.tol_period, period=period)
            preds.append(self.models[i].predict(x_i).reshape(-1, 1))
        x_pd = x
        if self.n_multistep>0:
            preds.append(self.predict_parallel(x_pd))
        preds = pd.DataFrame(np.hstack(preds), index=x.index, columns=self.target_cols)
        y_hat = self.anti_transform(x, preds)
        return y_hat

    def predict_single(self, i, x):
        x = pd.concat([x.reset_index(drop=True), pd.Series(np.ones(len(x)) * i)], axis=1)
        return self.multi_step_model.predict(x,num_threads=1).reshape(-1, 1)

    def predict_parallel(self, x):
        if self.parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
                y_hat = list(executor.map(partial(self.predict_single, x=x), np.arange(self.n_multistep)))
        else:
            y_hat = []
            for i in range(self.n_multistep):
                y_hat.append(self.predict_single(i, x))
        return np.hstack(y_hat)

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

    def predict_quantiles(self, x, **kwargs):
        preds = np.expand_dims(self.predict(x), -1) * np.ones((1, 1, len(self.q_vect)))
        for h in np.unique(x.index.hour):
            preds[x.index.hour == h, :, :] += np.expand_dims(self.err_distr[h], 0)
        return preds


class LGBEnergyAware(LGBMHybrid):
    def fit(self, x, y):
        lgb_pars = self.get_lgb_pars()
        x, y, x_val, y_val = self.train_val_split(x, y)
        e_unbalance = pd.DataFrame()
        for i in tqdm(range(self.n_single)):
            x_i = self.dataset_at_stepahead(x, i,  self.metadata_features, formatter=self.formatter,
                                            logger=self.logger, method='periodic', keep_last_n_lags=self.keep_last_n_lags,
                                            keep_last_seconds=self.keep_last_seconds,
                                            tol_period=self.tol_period)
            if i > 0:
                e_feature = e_unbalance.sum(axis=1)
                e_feature.name = 'e_unbalance'
                x_i = pd.concat([x_i, e_feature], axis=1)
            self.models.append(LGBMRegressor(**lgb_pars).fit(x_i, y.iloc[:, i]))

            # retrieve energy unbalance
            preds_i = self.models[i].predict(x_i)
            x_i_no_ctrl=x_i.copy()
            x_i_no_ctrl[[c for c in x_i_no_ctrl.columns if 'force_off' in c and '-' in c]] = 0
            preds_i_no_ctrl = self.models[i].predict(x_i_no_ctrl)
            e_unbalance = pd.concat([e_unbalance, pd.Series(preds_i_no_ctrl-preds_i, index=x_i.index)], axis=1)

        super().fit(x_val, y_val)
        return self

    def predict(self, x, **kwargs):
        preds = []
        e_unbalance = pd.DataFrame()
        for i in range(self.n_single):
            x_i = self.dataset_at_stepahead(x, i, self.metadata_features, formatter=self.formatter,
                                            logger=self.logger, method='periodic', keep_last_n_lags=self.keep_last_n_lags,
                                            keep_last_seconds=self.keep_last_seconds,
                                            tol_period=self.tol_period)
            if i > 0:
                e_feature = e_unbalance.sum(axis=1)
                e_feature.name = 'e_unbalance'
                x_i = pd.concat([x_i, e_feature], axis=1)
            x_i_no_ctrl = x_i.copy()
            preds.append(self.models[i].predict(x_i).reshape(-1, 1))

            # retrieve energy unbalance
            x_i_no_ctrl[[c for c in x_i_no_ctrl.columns if 'force_off' in c and '-' in c]] = 0

            preds_i_no_ctrl = self.models[i].predict(x_i_no_ctrl)

            e_unbalance = pd.concat([e_unbalance, pd.Series(preds_i_no_ctrl-preds[i].ravel(), index=x_i.index)], axis=1)
        preds = pd.DataFrame(np.hstack(preds), index=x.index, columns=self.target_cols)
        y_hat = self.anti_transform(x, preds)
        return y_hat
