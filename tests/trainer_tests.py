import unittest

import optuna
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from pyforecaster.trainer import hyperpar_optimizer, retrieve_cv_results
from pyforecaster.metrics import make_scorer, nmae


class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.t = 500
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

    def test_trainer(self):
        linreg = ElasticNet()
        model = Pipeline([('model', linreg)])

        n_trials = 20
        n_folds = 5
        study = hyperpar_optimizer(self.x, self.y, model, n_trials=n_trials, scoring=make_scorer(nmae), cv=n_folds,
                           param_space_fun=self.param_space_fun,
                           hpo_type='full')
        optuna.visualization.matplotlib.plot_contour(study, [k for k in study.best_params.keys()])
        trials_df = retrieve_cv_results(study)
        assert trials_df['value'].isna().sum() == 0

if __name__ == '__main__':
    unittest.main()
