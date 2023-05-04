import unittest

import optuna
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from pyforecaster.trainer import hyperpar_optimizer, retrieve_cv_results, base_storage_fun
from pyforecaster.metrics import make_scorer, nmae
from lightgbm import LGBMRegressor, Dataset, train


class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.t = 100
        self.n = 10
        self.x = pd.DataFrame(np.sin(np.arange(self.t)*10*np.pi/self.t).reshape(-1,1) * np.random.randn(1, self.n), index=pd.date_range('01-01-2020', '01-05-2020', self.t))
        self.y = self.x  @ np.random.randn(self.n, 5)
        self.y.iloc[10, :] =0
        self.logger =logging.getLogger()
        logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                            filename=None)

    @staticmethod
    def param_space_fun(trial):
        param_space = {'model__l1_ratio': trial.suggest_uniform('l1_ratio', 0, 1-1e-5),
                       'model__alpha': trial.suggest_uniform('alpha', 0, 10)}
        return param_space


    def test_trainer_lgb(self):
        def param_space_fun(trial):
            param_space = {'learning_rate': trial.suggest_float('learning_rate', 0, 1 - 1e-5)}
            return param_space

        linreg = ElasticNet()
        model = Pipeline([('model', linreg)])
        model = LGBMRegressor()
        n_trials = 20
        n_folds = 5
        cv_idxs = []
        for i in range(n_folds):
            tr_idx = np.random.randint(0, 2, len(self.x.index), dtype=bool)
            te_idx = ~tr_idx
            cv_idxs.append((tr_idx, te_idx))

        study, replies = hyperpar_optimizer(self.x, self.y.iloc[:, [0]], model, n_trials=n_trials, metric=nmae, cv=(f for f in cv_idxs),
                                   param_space_fun= param_space_fun,
                                   hpo_type='one_fold')
        optuna.visualization.matplotlib.plot_contour(study, [k for k in study.best_params.keys()])
        trials_df = retrieve_cv_results(study)
        assert trials_df['value'].isna().sum() == 0


    def test_trainer(self):
        linreg = ElasticNet()
        model = Pipeline([('model', linreg)])

        n_trials = 20
        n_folds = 5
        cv_idxs = []
        for i in range(n_folds):
            tr_idx = np.random.randint(0, 2, len(self.x.index), dtype=bool)
            te_idx = ~tr_idx
            cv_idxs.append((tr_idx, te_idx))

        study, replies = hyperpar_optimizer(self.x, self.y, model, n_trials=n_trials, metric=nmae, cv=(f for f in cv_idxs),
                                   param_space_fun=self.param_space_fun,
                                   hpo_type='one_fold')
        optuna.visualization.matplotlib.plot_contour(study, [k for k in study.best_params.keys()])
        trials_df = retrieve_cv_results(study)
        assert trials_df['value'].isna().sum() == 0

    def test_trainer_savepoints(self):
        linreg = ElasticNet()
        model = Pipeline([('model', linreg)])

        n_trials = 20
        n_folds = 5
        cv_idxs = []
        for i in range(n_folds):
            tr_idx = np.random.randint(0, 2, len(self.x.index), dtype=bool)
            te_idx = ~tr_idx
            cv_idxs.append((tr_idx, te_idx))

        study, replies = hyperpar_optimizer(self.x, self.y, model, n_trials=n_trials, metric=nmae, cv=(f for f in cv_idxs),
                                   param_space_fun=self.param_space_fun,
                                   hpo_type='full', storage_fun=base_storage_fun)
        assert len(replies) == len(study.trials_dataframe())


if __name__ == '__main__':
    unittest.main()
