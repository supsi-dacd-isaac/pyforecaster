import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
from abc import abstractmethod
from lightgbm import LGBMRegressor, Dataset, train
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import LabelEncoder
from pyforecaster.scenarios_generator import ScenGen
from pyforecaster.utilities import get_logger
from inspect import signature


def encode_categorical(func):
    def wrapper(self, x: pd.DataFrame, *args, **kwargs):
        # Initialize a protected dictionary to store encoders if it doesn't exist yet
        if not hasattr(self, '_le'):
            self._le = {}

        # Check if x contains columns that are not numbers and encode them
        for column in x.select_dtypes(include=['object', 'category']).columns:
            if column not in self._le:
                # Create and fit a new encoder for the column if it's the first encounter
                le = LabelEncoder()
                x[column] = le.fit_transform(x[column].astype(str))
                self._le[column] = le  # Store the encoder for future use
            else:
                # Use the existing encoder to transform the data
                le = self._le[column]
                # Check for unseen categories
                unique_values = set(x[column].astype(str))
                unseen_values = unique_values - set(le.classes_)
                if unseen_values:
                    raise ValueError(f"Unseen categories {unseen_values} encountered in column '{column}'.")
                x[column] = le.transform(x[column].astype(str))

        # Call the original function with preprocessed data
        return func(self, x, *args, **kwargs)

    return wrapper



class ScenarioGenerator(object):
    def __init__(self, q_vect=None, nodes_at_step=None, val_ratio=None, logger=None, n_scen_fit=100,
                 additional_node=False, formatter=None, conditional_to_hour=True, **scengen_kwgs):
        self.q_vect = np.hstack([0.01, np.linspace(0,1,11)[1:-1], 0.99]) if q_vect is None else q_vect
        self.scengen = ScenGen(q_vect=self.q_vect, nodes_at_step=nodes_at_step, additional_node=additional_node, **scengen_kwgs)
        self.val_ratio = val_ratio
        self.err_distr = {}
        self.logger = get_logger(name='Forecaster', level='WARNING') if logger is None else logger
        self.n_scen_fit = n_scen_fit
        self.additional_node = additional_node
        self.formatter = formatter
        self.conditional_to_hour = conditional_to_hour
        self.target_cols = None

    @property
    def online_tree_reduction(self):
        return self.scengen.online_tree_reduction

    @online_tree_reduction.setter
    def online_tree_reduction(self, value):
        self.scengen.online_tree_reduction = value

    def set_params(self, **kwargs):
        [self.__setattr__(k, v) for k, v in kwargs.items() if k in self.__dict__.keys()]

    def get_params(self, **kwargs):
        return {k: getattr(self, k) for k in signature(self.__class__).parameters.keys() if k in self.__dict__.keys()}

    def fit(self, x:pd.DataFrame, y:pd.DataFrame):
        y = self.anti_transform(x, y)
        self.target_cols = y.columns
        preds = self.predict(x)
        errs = pd.DataFrame(y.values-preds.values, index=x.index)
        self.scengen.fit(errs, x, n_scen=self.n_scen_fit)

        self.err_distr = {}
        if self.conditional_to_hour:
            self.err_distr = {}
            hours = np.arange(24)
            if len(np.unique(x.index.hour)) != 24:
                print('not all hours are there in the training set, building unconditional confidence intervals')
                for h in hours:
                    self.err_distr[h] = np.quantile(errs, self.q_vect, axis=0).T
            else:
                for h in hours:
                    self.err_distr[h] = np.quantile(errs.loc[errs.index.hour == h, :], self.q_vect, axis=0).T
        else:
            self.err_distr = np.quantile(errs, self.q_vect, axis=0).T

    @abstractmethod
    def predict(self, x, **kwargs):
        pass

    def anti_transform(self, x, y_hat):
        if self.formatter is not None and self.formatter.denormalizing_fun is not None:
            y_hat = self.formatter.denormalize(x, y_hat)
        return y_hat

    @abstractmethod
    def _predict_quantiles(self, x, **kwargs):
        pass

    def quantiles_to_df(self, q_hat:np.ndarray, index, q_vect=None):
        level_0_labels = self.target_cols
        level_1_labels = self.q_vect if q_vect is None else q_vect
        q_hat = np.reshape(q_hat, (q_hat.shape[0], q_hat.shape[1] * q_hat.shape[2]))
        q_hat = pd.DataFrame(q_hat, index=index, columns=pd.MultiIndex.from_product([level_0_labels, level_1_labels]))
        q_hat.columns.names = ['target', 'quantile']
        return q_hat

    @staticmethod
    def quantiles_to_numpy(q_hat:pd.DataFrame):
        n_taus = len(q_hat.columns.get_level_values(1).unique())
        q_hat = q_hat.values
        q_hat = np.reshape(q_hat, (q_hat.shape[0], -1, n_taus))
        return q_hat

    def predict_quantiles(self, x, dataframe=True, **kwargs):
        q_hat = self._predict_quantiles(x, **kwargs)
        if isinstance(q_hat, np.ndarray) and dataframe:
            # create multiindex dataframe
            q_hat = self.quantiles_to_df(q_hat, x.index)
        if isinstance(q_hat, pd.DataFrame) and not dataframe:
            q_hat = self.quantiles_to_numpy(q_hat)

        return q_hat

    def predict_pmf(self, x, discrete_prob_space, **predict_q_kwargs):
        """
        Return probability mass function of the target variable, obtained from quantile predictions
        :param x:
        :param predict_q_kwargs:
        :return:
        """
        quantiles = self.predict_quantiles(x, dataframe=False, **predict_q_kwargs)
        pmf = np.zeros((quantiles.shape[0], quantiles.shape[1], len(discrete_prob_space)-1))
        for i in range(quantiles.shape[0]):
            for j in range(quantiles.shape[1]):
                pmf[i, j, :] = np.histogram(quantiles[i, j, :], bins=discrete_prob_space)[0]/len(self.q_vect)
        return pmf

    def predict_scenarios(self, x, n_scen=None, random_state=None, **predict_q_kwargs):
        n_scen = self.n_scen_fit if n_scen is None else n_scen
        # retrieve quantiles from child class
        quantiles = self.predict_quantiles(x, dataframe=False, **predict_q_kwargs)
        scenarios = self.scengen.predict_scenarios(quantiles, n_scen=n_scen, x=x, random_state=random_state)
        q_from_scens = np.rollaxis(np.quantile(scenarios, self.q_vect, axis=-1), 0, 3)
        mean_abs_dev = np.abs(q_from_scens - quantiles).mean(axis=0).mean(axis=0)
        self.logger.info('mean abs deviations of re-estimated quantiles from scenarios: {}'.format(mean_abs_dev))
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
            quantiles = self.predict_quantiles(x, dataframe=False, **predict_q_kwargs)

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

    def train_val_split(self, x, y):
        if self.val_ratio is not None:
            n_val = int(x.shape[0] * self.val_ratio)
            n_tr = x.shape[0] - n_val
            x_val, y_val = x.iloc[n_tr:, :], y.iloc[n_tr:, :]
            x, y = x.iloc[:n_tr, :], y.iloc[:n_tr, :]
        else:
            x_val, y_val = x, y
        return x, y, x_val, y_val



class LinearForecaster(ScenarioGenerator):
    def __init__(self, q_vect=None, val_ratio=None, nodes_at_step=None, kind='linear', **scengen_kwgs):
        super().__init__(q_vect, nodes_at_step=nodes_at_step, val_ratio=val_ratio, **scengen_kwgs)
        self.m = None
        self.kind = kind

    @encode_categorical
    def fit(self, x:pd.DataFrame, y:pd.DataFrame):
        x, y, x_val, y_val = self.train_val_split(x, y)
        if self.kind == 'linear':
            self.m = LinearRegression().fit(x, y)
        elif self.kind == 'ridge':
            self.m = RidgeCV(alphas=10 ** np.linspace(-2, 8, 9)).fit(x, y)
        else:
            raise ValueError('kind must be either linear or ridge')
        super().fit(x_val, y_val)
        return self

    @encode_categorical
    def predict(self, x:pd.DataFrame, **kwargs):
        y_hat = pd.DataFrame(self.m.predict(x), index=x.index, columns=self.target_cols)
        y_hat = self.anti_transform(x, y_hat)
        return y_hat

    def _predict_quantiles(self, x:pd.DataFrame, **kwargs):
        preds = np.expand_dims(self.predict(x), -1) * np.ones((1, 1, len(self.q_vect)))
        for h in np.unique(x.index.hour):
            preds[x.index.hour == h, :, :] += np.expand_dims(self.err_distr[h], 0)
        return preds


class LGBForecaster(ScenarioGenerator):
    def __init__(self, device_type='cpu', max_depth=20, n_estimators=100, num_leaves=100, learning_rate=0.1, min_child_samples=20,
                 n_jobs=4, n_jobs_predict=0, objective='regression', verbose=-1, metric='l2', colsample_bytree=1, colsample_bynode=1, q_vect=None, val_ratio=None, nodes_at_step=None, **scengen_kwgs):
        super().__init__(q_vect, val_ratio=val_ratio, nodes_at_step=nodes_at_step, **scengen_kwgs)
        self.m = []
        self.device_type = device_type
        self.objective = objective
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.metric = metric
        self.min_child_samples = min_child_samples
        self.colsample_bytree=colsample_bytree
        self.colsample_bynode = colsample_bynode
        self.n_jobs = n_jobs
        self.n_jobs_predict = n_jobs_predict
        self.lgb_pars = {"device_type": self.device_type,
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
                    "colsample_bynode": self.colsample_bynode}

    def fit(self, x, y):
        x, y, x_val, y_val = self.train_val_split(x, y)
        for i in tqdm(range(y.shape[1]), 'fitting LGB models'):
            lgb_data = Dataset(x, y.iloc[:, i].values.ravel())
            m = train(self.lgb_pars, lgb_data, num_boost_round=self.lgb_pars['n_estimators'])
            self.m.append(m)

        super().fit(x_val, y_val)
        return self

    def predict(self, x, **kwargs):
        preds = []
        for m in self.m:
            preds.append(m.predict(x, num_threads=self.n_jobs_predict).reshape(-1, 1))
        y_hat = pd.DataFrame(np.hstack(preds), index=x.index, columns=self.target_cols)
        y_hat = self.anti_transform(x, y_hat)
        return y_hat

    def _predict_quantiles(self, x, **kwargs):
        preds = np.expand_dims(self.predict(x), -1) * np.ones((1, 1, len(self.q_vect)))
        for h in np.unique(x.index.hour):
            preds[x.index.hour == h, :, :] += np.expand_dims(self.err_distr[h], 0)
        return preds


