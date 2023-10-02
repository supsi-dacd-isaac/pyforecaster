import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from pyforecaster.forecasting_models.neural_forecasters import PICNN
from pyforecaster.trainer import hyperpar_optimizer
from pyforecaster.formatter import Formatter
from pyforecaster.metrics import nmae
from os import makedirs
from os.path import exists
import jax.numpy as jnp

class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.read_pickle('tests/data/test_data.zip').droplevel(0, 1)
        self.logger =logging.getLogger()
        logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                            filename=None)
        formatter = Formatter(logger=self.logger).add_transform(['all'], lags=np.arange(144),
                                                                    relative_lags=True)
        formatter.add_transform(['all'], ['min', 'max'], agg_bins=[1, 2, 15, 20])
        formatter.add_target_transform(['all'], lags=-np.arange(144))

        self.x, self.y = formatter.transform(self.data.iloc[:40000])
        self.x = (self.x - self.x.mean(axis=0)) / (self.x.std(axis=0)+0.01)
        self.y = (self.y - self.y.mean(axis=0)) / (self.y.std(axis=0)+0.01)

    def test_ffnn(self):
        # normalize inputs
        x = (self.x - self.x.mean(axis=0)) / (self.x.std(axis=0)+0.01)
        y = (self.y - self.y.mean(axis=0)) / (self.y.std(axis=0)+0.01)

        n_tr = int(len(x) * 0.8)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        savepath_tr_plots = 'tests/results/ffnn_tr_plots'

        # if not there, create directory savepath_tr_plots
        if not exists(savepath_tr_plots):
            makedirs(savepath_tr_plots)


        optimization_vars = x_tr.columns[:100]


        m = PICNN(learning_rate=1e-3,  batch_size=1000, load_path=None, n_hidden_x=200, n_hidden_y=200,
               n_out=y_tr.shape[1], n_layers=3, optimization_vars=optimization_vars).fit(x_tr,
                                                                                           y_tr,
                                                                                           n_epochs=2,
                                                                                           savepath_tr_plots=savepath_tr_plots,
                                                                                           stats_step=40)

        y_hat_1 = m.predict(x_te.iloc[:100, :])
        m.save('tests/results/ffnn_model.pk')
        n = PICNN(load_path='tests/results/ffnn_model.pk')
        y_hat_2 = n.predict(x_te.iloc[:100, :])

        assert np.all(np.sum(y_hat_1-y_hat_2) == 0)


    def test_optimization(self):

        # normalize inputs
        x = (self.x - self.x.mean(axis=0)) / (self.x.std(axis=0)+0.01)
        y = (self.y - self.y.mean(axis=0)) / (self.y.std(axis=0)+0.01)

        n_tr = int(len(x) * 0.8)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        savepath_tr_plots = 'tests/results/ffnn_tr_plots'

        # if not there, create directory savepath_tr_plots
        if not exists(savepath_tr_plots):
            makedirs(savepath_tr_plots)


        optimization_vars = x_tr.columns[:100]


        m = PICNN(learning_rate=1e-3,  batch_size=1000, load_path=None, n_hidden_x=200, n_hidden_y=200,
               n_out=y_tr.shape[1], n_layers=3, optimization_vars=optimization_vars).fit(x_tr,
                                                                                           y_tr,
                                                                                           n_epochs=10,
                                                                                           savepath_tr_plots=savepath_tr_plots,
                                                                                           stats_step=40)

        objective = lambda y_hat, ctrl: jnp.mean(y_hat ** 2) + jnp.sum(ctrl**2)
        y_hat_opt, ctrl_opt, v_opt = m.optimize(x_te.iloc[[100], :], objective=objective,n_iter=200)
        y_hat = m.predict(x_te.iloc[[100], :])

        plt.figure()
        plt.plot(y_hat_opt.values.ravel(), label='y_hat_opt')
        plt.plot(y_te.iloc[100, :].values.ravel(), label='y_te')
        plt.plot(y_hat.values.ravel(), label='y_hat')
        plt.legend()

    def test_hyperpar_optimization(self):

        model = PICNN(optimization_vars=self.x.columns[:10], n_out=self.y.shape[1])

        n_folds = 2
        cv_idxs = []
        for i in range(n_folds):
            tr_idx = np.random.randint(0, 2, len(self.x.index), dtype=bool)
            te_idx = ~tr_idx
            cv_idxs.append((tr_idx, te_idx))

        study, replies = hyperpar_optimizer(self.x, self.y, model, n_trials=1, metric=nmae,
                                            cv=(f for f in cv_idxs),
                                            param_space_fun=None,
                                            hpo_type='full')


if __name__ == '__main__':
    unittest.main()

