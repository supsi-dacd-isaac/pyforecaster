import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from pyforecaster.forecasting_models.neural_forecasters import PQICNN, PICNN, RecStablePICNN, NN
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

        m = NN(learning_rate=1e-3,  batch_size=1000, load_path=None, n_hidden_x=200,
               n_out=y_tr.shape[1], n_layers=3).fit(x_tr,y_tr, n_epochs=1, savepath_tr_plots=savepath_tr_plots, stats_step=40)

        y_hat_1 = m.predict(x_te.iloc[:100, :])


    def test_picnn(self):
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


        m_1 = PICNN(learning_rate=1e-3, batch_size=1000, load_path=None, n_hidden_x=200, n_hidden_y=200,
                  n_out=y_tr.shape[1], n_layers=3, optimization_vars=optimization_vars).fit(x_tr,
                                                                                            y_tr,
                                                                                            n_epochs=1,
                                                                                            savepath_tr_plots=savepath_tr_plots,
                                                                                            stats_step=40)

        y_hat_1 = m_1.predict(x_te)
        m_1.save('tests/results/ffnn_model.pk')

        n = PICNN(load_path='tests/results/ffnn_model.pk')
        y_hat_2 = n.predict(x_te.iloc[:100, :])


        m = RecStablePICNN(learning_rate=1e-3, batch_size=1000, load_path=None, n_hidden_x=200, n_hidden_y=200,
                  n_out=1, n_layers=3, optimization_vars=optimization_vars).fit(x_tr,y_tr.iloc[:, [0]],
                                                                                            n_epochs=1,
                                                                                            savepath_tr_plots=savepath_tr_plots,
                                                                                            stats_step=40)
        m.predict(x_te.iloc[:100, :])

        print(np.sum(y_hat_1-y_hat_2).sum())
        assert np.all(np.mean((y_hat_1-y_hat_2).abs()) <= 1e-6)


    def no_test_pqicnn(self):
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


        optimization_vars = x_tr.columns[:-1]


        m_1 = PQICNN(learning_rate=1e-2, batch_size=1000, load_path=None, n_hidden_x=200, n_hidden_y=200,
                  n_out=y_tr.shape[1], n_layers=4, optimization_vars=optimization_vars,stopping_rounds=100).fit(x_tr,
                                                                                            y_tr,
                                                                                            n_epochs=30,
                                                                                            savepath_tr_plots=savepath_tr_plots,
                                                                                            stats_step=100,rel_tol=-1)
        y_hat_1 = m_1.predict(x_te)

         # check convexity of the PICNN
        rnd_idxs = np.random.choice(x_tr.shape[0], 1)
        rand_opt_vars = np.random.choice(optimization_vars, 10)
        for cc in rand_opt_vars:
            x = x_tr.iloc[rnd_idxs, :]
            x = pd.concat([x] * 100, axis=0)
            x[cc] = np.linspace(-1, 1, 100)
            y_hat = m_1.predict(x)
            d = np.diff(np.sign(np.diff(y_hat.values[:, :96], axis=0)), axis=0)
            approx_second_der = np.round(np.diff(y_hat.values[:, :96], 2, axis=0), 5)
            approx_second_der[approx_second_der == 0] = 0  # to fix the sign
            is_convex = not np.any(np.abs(np.diff(np.sign(approx_second_der), axis=0)) > 1)
            print('output is convex w.r.t. input {}: {}'.format(cc, is_convex))
            plt.figure(layout='tight')
            plt.plot(np.tile(x[cc].values.reshape(-1, 1), 96), y_hat.values[:, :96], alpha=0.3)
            plt.xlabel(cc)
            plt.show()
            plt.savefig('wp3/results/figs/convexity/{}.png'.format(cc), dpi=300)

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
               n_out=y_tr.shape[1], n_layers=3, optimization_vars=optimization_vars, inverter_learning_rate=1e-3,
                  augment_ctrl_inputs=True, layer_normalization=True, unnormalized_inputs=optimization_vars).fit(x_tr, y_tr,
                                                                          n_epochs=3,
                                                                          savepath_tr_plots=savepath_tr_plots,
                                                                          stats_step=40)

        objective = lambda y_hat, ctrl: jnp.mean(y_hat ** 2) + 0.01*jnp.sum(ctrl**2)
        ctrl_opt, inputs_opt, y_hat_opt, v_opt = m.optimize(x_te.iloc[[100], :], objective=objective,n_iter=50)
        y_hat = m.predict(x_te.iloc[[100], :].copy())

        x_freeze = x_te.iloc[[100], :].copy()
        x_freeze[optimization_vars] = ctrl_opt.ravel()

        y_hat_opt_2 = m.predict(x_freeze)

        plt.figure()
        plt.plot(y_hat_opt.values.ravel(), label='y_hat_opt')
        plt.plot(y_hat_opt_2.values.ravel(), label='y_hat_opt_2')
        plt.plot(y_te.iloc[100, :].values.ravel(), label='y_te')
        plt.plot(y_hat.values.ravel(), label='y_hat')
        plt.legend()
        assert  (y_hat_opt_2-y_hat_opt).sum().sum() == 0

    def test_hyperpar_optimization(self):

        model = PICNN(optimization_vars=self.x.columns[:10], n_out=self.y.shape[1], n_epochs=6)

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

