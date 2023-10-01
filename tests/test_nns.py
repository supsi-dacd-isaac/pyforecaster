import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from pyforecaster.forecasting_models.neural_forecasters import PICNN
from pyforecaster.formatter import Formatter
from os import makedirs
from os.path import exists
import jax.numpy as jnp

class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.read_pickle('tests/data/test_data.zip').droplevel(0, 1)
        self.logger =logging.getLogger()
        logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                            filename=None)

    def test_ffnn(self):
        formatter = Formatter(logger=self.logger).add_transform(['all'], lags=np.arange(144),
                                                                    relative_lags=True)
        formatter.add_transform(['all'], ['min', 'max'], agg_bins=[1, 2, 15, 20])
        formatter.add_target_transform(['all'], lags=-np.arange(144))

        x, y = formatter.transform(self.data.iloc[:40000])
        # normalize inputs
        x = (x - x.mean(axis=0)) / (x.std(axis=0)+0.01)
        y = (y - y.mean(axis=0)) / (y.std(axis=0)+0.01)

        n_tr = int(len(x) * 0.8)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        savepath_tr_plots = 'tests/results/ffnn_tr_plots'

        # if not there, create directory savepath_tr_plots
        if not exists(savepath_tr_plots):
            makedirs(savepath_tr_plots)


        optimization_vars = x_tr.columns[:100]


        m = PICNN(learning_rate=1e-3,  batch_size=1000, load_path=None, n_hidden_x=200, n_hidden_y=200,
               n_out=y_tr.shape[1], n_layers=3, optimization_vars=optimization_vars).train(x_tr,
                                                                                           y_tr, x_te, y_te,
                                                                                           n_epochs=1,
                                                                                           savepath_tr_plots=savepath_tr_plots,
                                                                                           stats_step=40)

        y_hat_1 = m.predict(x_te.iloc[:100, :])
        m.save('tests/results/ffnn_model.pk')
        n = PICNN(load_path='tests/results/ffnn_model.pk')
        y_hat_2 = n.predict(x_te.iloc[:100, :])

        assert np.all(np.sum(y_hat_1-y_hat_2) == 0)


    def test_optimization(self):
        formatter = Formatter(logger=self.logger).add_transform(['all'], lags=np.arange(144),
                                                                    relative_lags=True)
        formatter.add_transform(['all'], ['min', 'max'], agg_bins=[1, 2, 15, 20])
        formatter.add_target_transform(['all'], lags=-np.arange(144))

        x, y = formatter.transform(self.data.iloc[:40000])
        # normalize inputs
        x = (x - x.mean(axis=0)) / (x.std(axis=0)+0.01)
        y = (y - y.mean(axis=0)) / (y.std(axis=0)+0.01)

        n_tr = int(len(x) * 0.8)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        savepath_tr_plots = 'tests/results/ffnn_tr_plots'

        # if not there, create directory savepath_tr_plots
        if not exists(savepath_tr_plots):
            makedirs(savepath_tr_plots)


        optimization_vars = x_tr.columns[:100]


        m = PICNN(learning_rate=1e-3,  batch_size=1000, load_path=None, n_hidden_x=200, n_hidden_y=200,
               n_out=y_tr.shape[1], n_layers=3, optimization_vars=optimization_vars).train(x_tr,
                                                                                           y_tr, x_te, y_te,
                                                                                           n_epochs=1,
                                                                                           savepath_tr_plots=savepath_tr_plots,
                                                                                           stats_step=40)

        objective = lambda y_hat: jnp.mean(y_hat ** 2)
        y_hat_opt, ctrl_opt, v_opt = m.optimize(x_te.iloc[[100], :], objective=objective,n_iter=200)
        y_hat = m.predict(x_te.iloc[[100], :])

        plt.figure()
        plt.plot(y_hat_opt.values.ravel(), label='y_hat_opt')
        plt.plot(y_te.iloc[100, :].values.ravel(), label='y_te')
        plt.plot(y_hat.values.ravel(), label='y_hat')
        plt.legend()

if __name__ == '__main__':
    unittest.main()

