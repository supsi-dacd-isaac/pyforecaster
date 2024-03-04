import logging
import unittest
from os import makedirs
from os.path import exists, join

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from jax import vmap

from pyforecaster.forecasting_models.neural_models.ICNN import PICNN, RecStablePICNN, PIQCNN, PIQCNNSigmoid, \
    StructuredPICNN, LatentStructuredPICNN, latent_pred
from pyforecaster.forecasting_models.neural_models.INN import CausalInvertibleNN
from pyforecaster.forecasting_models.neural_models.base_nn import NN, FFNN
from pyforecaster.formatter import Formatter
from pyforecaster.trainer import hyperpar_optimizer

from pyforecaster.forecaster import LinearForecaster
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

        formatter = Formatter(logger=self.logger,augment=False).add_transform(['all'], lags=-np.arange(288),
                                                                    relative_lags=True)
        self.e, _ = formatter.transform(self.data.iloc[:40000],time_features=False)
        self.e = (self.e - self.e.mean(axis=0)) / (self.e.std(axis=0)+0.01)


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

        savepath_tr_plots = 'tests/results/figs/convexity'

        # if not there, create directory savepath_tr_plots
        if not exists(savepath_tr_plots):
            makedirs(savepath_tr_plots)


        optimization_vars = x_tr.columns[:100]


        m_1 = PICNN(learning_rate=1e-3, batch_size=500, load_path=None, n_hidden_x=20, n_hidden_y=20,
                  n_out=y_tr.shape[1], n_layers=3, optimization_vars=optimization_vars,probabilistic=True, probabilistic_loss_kind='crps', rel_tol=-1,
                    val_ratio=0.2,z_min=-1*jnp.ones(y_tr.shape[1]),z_max=jnp.ones(y_tr.shape[1])).fit(x_tr, y_tr,n_epochs=2,stats_step=50,savepath_tr_plots=savepath_tr_plots)

        y_hat_1 = m_1.predict(x_te)
        m_1.save('tests/results/ffnn_model.pk')

        rnd_idxs = np.random.choice(x_te.shape[0], 1)
        for r in rnd_idxs:
            y_hat = m_1.predict(x_te.iloc[[r], :])
            q_hat = m_1.predict_quantiles(x_te.iloc[[r], :])
            plt.figure()
            plt.plot(y_te.iloc[r, :].values.ravel(), label='y_true')
            plt.plot(y_hat.values.ravel(), label='y_hat')
            plt.plot(np.squeeze(q_hat), label='q_hat', color='orange', alpha=0.3)
            plt.legend()

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


    def test_causal_df(self):
        # normalize inputs
        x = (self.x - self.x.mean(axis=0)) / (self.x.std(axis=0) + 0.01)
        y = (self.y - self.y.mean(axis=0)) / (self.y.std(axis=0) + 0.01)
        x = x.iloc[:, -10:]
        n_tr = int(len(x) * 0.8)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        savepath_tr_plots = 'tests/results/ffnn_tr_plots'

        # if not there, create directory savepath_tr_plots
        if not exists(savepath_tr_plots):
            makedirs(savepath_tr_plots)


        optimization_vars = x_tr.columns[:-1]
        causal_df = pd.DataFrame(np.tril(np.ones((len(optimization_vars), y_tr.shape[1]))), index=optimization_vars, columns=y_tr.columns)

        m_1 = PICNN(learning_rate=1e-3, batch_size=1000, load_path=None, n_hidden_x=200, n_hidden_y=200,
                    n_out=y_tr.shape[1], n_latent=20, n_layers=3, optimization_vars=optimization_vars, probabilistic=False, rel_tol=-1).fit(x_tr,
                                                                                              y_tr,
                                                                                              n_epochs=10,
                                                                                              savepath_tr_plots=savepath_tr_plots,
                                                                                              stats_step=5)
        # check convexity of the PICNN
        #rnd_idxs = np.random.choice(x_tr.shape[0], 1)
        #rand_opt_vars = np.random.choice(optimization_vars, 10)
        #for cc in rand_opt_vars:
        #    x = x_tr.iloc[rnd_idxs, :]
        #    x = pd.concat([x] * 100, axis=0)
        #    x[cc] = np.linspace(-1, 1, 100)
        #    y_hat = m_1.predict(x)
        #    d = np.diff(np.sign(np.diff(y_hat.values[:, :96], axis=0)), axis=0)
        #    approx_second_der = np.round(np.diff(y_hat.values[:, :96], 2, axis=0), 5)
        #    approx_second_der[approx_second_der == 0] = 0  # to fix the sign
        #    is_convex = not np.any(np.sign(approx_second_der) < -0)
        #    print('output is convex w.r.t. input {}: {}'.format(cc, is_convex))
        #    plt.figure(layout='tight')
        #    plt.plot(np.tile(x[cc].values.reshape(-1, 1), 96), y_hat.values[:, :96], alpha=0.3)
        #    plt.xlabel(cc)
        #    plt.savefig(join(savepath_tr_plots, '{}.png'.format(cc)), dpi=300)
    def test_pqicnn(self):

        # normalize inputs
        x = (self.x - self.x.mean(axis=0)) / (self.x.std(axis=0)+0.01)
        y = (self.y - self.y.mean(axis=0)) / (self.y.std(axis=0)+0.01)

        n_tr = int(len(x) * 0.8)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        savepath_tr_plots = 'tests/results/figs/convexity'

        # if not there, create directory savepath_tr_plots
        if not exists(savepath_tr_plots):
            makedirs(savepath_tr_plots)


        optimization_vars = x_tr.columns[:10]
        optimizer = optax.adamw(learning_rate=1e-3)

        m_1 = PIQCNN(learning_rate=1e-3, batch_size=1000, load_path=None, n_hidden_x=200, n_hidden_y=200,
                  n_out=y_tr.shape[1], n_layers=4, optimization_vars=optimization_vars,stopping_rounds=100, optimizer=optimizer).fit(x_tr,
                                                                                            y_tr,
                                                                                            n_epochs=1,
                                                                                            savepath_tr_plots=savepath_tr_plots,
                                                                                            stats_step=80,rel_tol=-1)

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
            plt.savefig(join(savepath_tr_plots, '{}.png'.format(cc)), dpi=300)

    def test_piqcnn_sigmoid(self):

        # normalize inputs
        x_cols = ['all_lag_101', 'all_lag_062', 'ghi_6', 'all_lag_138',
         'all_lag_090']
        #x_cols = np.random.choice(self.x.columns, 5)
        x = (self.x[x_cols] - self.x[x_cols].mean(axis=0)) / (self.x[x_cols].std(axis=0)+0.01)
        y = (self.y - self.y.mean(axis=0)) / (self.y.std(axis=0)+0.01)

        n_tr = int(len(x) * 0.8)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        savepath_tr_plots = 'tests/results/figs/convexity'

        # if not there, create directory savepath_tr_plots
        if not exists(savepath_tr_plots):
            makedirs(savepath_tr_plots)


        optimization_vars = x_tr.columns[:-2]
        optimizer = optax.adamw(learning_rate=1e-3)

        m_1 = PIQCNNSigmoid(learning_rate=1e-2, batch_size=200, load_path=None, n_hidden_x=200, n_hidden_y=200,
                  n_out=y_tr.shape[1], n_layers=4, optimization_vars=optimization_vars,stopping_rounds=100, optimizer=optimizer, layer_normalization=True).fit(x_tr,
                                                                                            y_tr,
                                                                                            n_epochs=1,
                                                                                            savepath_tr_plots=savepath_tr_plots,
                                                                                            stats_step=200,rel_tol=-1)

        # check convexity of the PICNN
        rnd_idxs = np.random.choice(x_tr.shape[0], 1)
        rand_opt_vars = np.random.choice(optimization_vars, 5)
        for cc in rand_opt_vars:
            x = x_tr.iloc[rnd_idxs, :]
            x = pd.concat([x] * 100, axis=0)
            x[cc] = np.linspace(-1, 1, 100)
            y_hat = m_1.predict(x)
            d = np.diff(np.sign(np.diff(y_hat.values, axis=0)), axis=0)
            approx_second_der = np.round(np.diff(y_hat.values, 2, axis=0), 5)
            approx_second_der[approx_second_der == 0] = 0  # to fix the sign
            is_convex = not np.any(np.abs(np.diff(np.sign(approx_second_der), axis=0)) > 1)
            print('output is convex w.r.t. input {}: {}'.format(cc, is_convex))
            plt.figure(layout='tight')
            plt.plot(np.tile(x[cc].values.reshape(-1, 1), y_hat.shape[1]), y_hat.values, alpha=0.3)
            plt.xlabel(cc)
            plt.savefig(join(savepath_tr_plots, '{}.png'.format(cc)), dpi=300)

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
                                                                          n_epochs=1,
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

        n_folds = 1
        cv_idxs = []
        for i in range(n_folds):
            tr_idx = np.random.randint(0, 2, len(self.x.index[:1000]), dtype=bool)
            te_idx = ~tr_idx
            cv_idxs.append((tr_idx, te_idx))

        objective = lambda y, ctrl: (y ** 2).mean()
        def custom_metric(x, t, agg_index=None, inter_normalization=True, **kwargs):
            obj = x.apply(lambda x: objective(x, None), axis=1)
            obj_hat = t.apply(lambda x: objective(x, None), axis=1)
            rank = np.argsort(obj)
            rank_hat = np.argsort(obj_hat)
            corr = np.corrcoef(rank, rank_hat)[0, 1]
            return np.array(corr)

        study, replies = hyperpar_optimizer(self.x.iloc[:1000, :], self.y.iloc[:1000, :], model, n_trials=1, metric=custom_metric,
                                            cv=(f for f in cv_idxs),
                                            param_space_fun=None,
                                            hpo_type='full')

    def test_structured_picnn_sigmoid(self):

        #x_cols = np.random.choice(self.x.columns, 5)
        x = (self.x - self.x.mean(axis=0)) / (self.x.std(axis=0)+0.01)
        y = (self.y - self.y.mean(axis=0)) / (self.y.std(axis=0)+0.01)

        n_tr = int(len(x) * 0.8)

        objective = lambda y, ctrl: jnp.mean(y**2)

        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        savepath_tr_plots = 'tests/results/figs/convexity'

        # if not there, create directory savepath_tr_plots
        if not exists(savepath_tr_plots):
            makedirs(savepath_tr_plots)


        optimization_vars = x_tr.columns[:20]

        m_1 = StructuredPICNN(learning_rate=1e-4, batch_size=100, load_path=None, n_hidden_x=512, n_hidden_y=250,
                  n_out=y_tr.shape[1], n_layers=3, optimization_vars=optimization_vars,stopping_rounds=2
                              , layer_normalization=True, objective_fun=objective, probabilistic=True, probabilistic_loss_kind='crps', distribution='log-normal', normalize_target=False).fit(x_tr,
                                                                                            y_tr,
                                                                                            n_epochs=6,
                                                                                            savepath_tr_plots=savepath_tr_plots,
                                                                                            stats_step=200, rel_tol=-1)

        objs = vmap(objective,in_axes=(0, 0))(y_te.values, x_te.values)
        rnd_idxs = np.random.choice(x_te.shape[0], 5000, replace=False)
        rnd_idxs =  rnd_idxs[np.argsort(objs[rnd_idxs])]

        fig, ax = plt.subplots(2, 1, figsize=(5, 10))
        ax[0].plot(m_1.predict(x_te.iloc[rnd_idxs, :], return_obj=True)[1].values.ravel(), label='y_hat', linewidth=1, color='orange')
        ax[0].plot(objs[rnd_idxs], label='y_true', linewidth=1, color='darkblue')
        ax[1].scatter(objs[rnd_idxs], m_1.predict(x_te.iloc[rnd_idxs, :], return_obj=True)[1].values.ravel(), s=1)
        ax[0].plot(np.squeeze(m_1.predict_quantiles(x_te.iloc[rnd_idxs, :], return_obj=True)), color='orange', alpha=0.3, linewidth=1)


        rnd_idxs = np.random.choice(x_te.shape[0], 5000, replace=False)
        _, obj, sigma = m_1.predict(x_te.iloc[rnd_idxs, :], return_obj=True, return_sigma=True)
        qs = np.squeeze(m_1.predict_quantiles(x_te.iloc[rnd_idxs, :], return_obj=True))
        rnd_idxs =  np.argsort(sigma.values.ravel())

        fig, ax = plt.subplots(2, 1, figsize=(5, 10))
        ax[0].plot(obj.values[rnd_idxs], label='y_hat', linewidth=1,
                   color='orange')
        ax[0].plot(objs[rnd_idxs], label='y_true', linewidth=1, color='darkblue')
        ax[1].scatter(objs[rnd_idxs], m_1.predict(x_te.iloc[rnd_idxs, :], return_obj=True)[1].values.ravel(), s=1)
        ax[0].plot(qs[rnd_idxs], color='orange',
                   alpha=0.3, linewidth=1)

        ordered_idx = np.argsort(np.abs(objs[rnd_idxs] - m_1.predict(x_te.iloc[rnd_idxs, :], return_obj=True)[1].values.ravel()))


        for r in rnd_idxs[ordered_idx[-10:]]:
            y_hat = m_1.predict(x_te.iloc[[r], :])
            #q_hat = m_1.predict_quantiles(x_te.iloc[[r], :])
            plt.figure()
            plt.plot(y_te.iloc[r, :].values.ravel(), label='y_true')
            plt.plot(y_hat.values.ravel(), label='y_hat')
            #plt.plot(np.squeeze(q_hat), label='q_hat', color='orange', alpha=0.3)
            plt.legend()


    def test_latent_picnn(self):

        # normalize inputs
        x = (self.x - self.x.mean(axis=0)) / (self.x.std(axis=0)+0.01)
        y = (self.y - self.y.mean(axis=0)) / (self.y.std(axis=0)+0.01)

        x = x.iloc[:, :10]
        n_tr = int(len(x) * 0.8)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        savepath_tr_plots = 'tests/results/ffnn_tr_plots'

        # if not there, create directory savepath_tr_plots
        if not exists(savepath_tr_plots):
            makedirs(savepath_tr_plots)

        optimization_vars = x_tr.columns[:5]

        m = LatentStructuredPICNN(n_latent=5, learning_rate=1e-3,  batch_size=1000, load_path=None, n_hidden_x=200,
               n_out=y_tr.shape[1], n_layers=3, optimization_vars=optimization_vars, inverter_learning_rate=1e-3,
                  augment_ctrl_inputs=True, layer_normalization=True, unnormalized_inputs=optimization_vars,
                                  n_embeddings=10, n_first_decoder=100).fit(x_tr, y_tr,
                                                                          n_epochs=1,
                                                                          savepath_tr_plots=savepath_tr_plots,
                                                                          stats_step=20)

        objective = lambda y_hat, ctrl: jnp.mean(y_hat ** 2)# + boxconstr(ctrl, 100, -100)
        m.inverter_optimizer = optax.adabelief(learning_rate=1e-1)
        ctrl_opt, inputs_opt, y_hat_opt, v_opt, _ ,_, _ = m.optimize(x_te.iloc[[0], :], objective=objective,n_iter=500)

        #e_optobj_convexity_test(x_te, m, optimization_vars)
        #io_convexity_test(x_te, m, optimization_vars)
        #eo_convexity_test(x_te, m, optimization_vars)

        rnd_idxs = np.random.choice(x_te.shape[0], 1)
        rnd_idxs = [0]
        for r in rnd_idxs:
            y_hat = m.predict(x_te.iloc[[r], :])
            plt.figure()
            plt.plot(y_te.iloc[r, :].values.ravel(), label='y_true')
            plt.plot(y_hat.values.ravel(), label='y_hat')
            plt.legend()
        plt.plot(y_hat_opt.values.ravel())
        plt.show()


    def test_invertible_causal_nn(self):

        n_tr = int(len(self.x) * 0.8)
        e_tr, e_te = [self.e.iloc[:n_tr, :].copy(), self.e.iloc[n_tr:, :].copy()]

        savepath_tr_plots = 'tests/results/ffnn_tr_plots'

        # if not there, create directory savepath_tr_plots
        if not exists(savepath_tr_plots):
            makedirs(savepath_tr_plots)

        m_lin = LinearForecaster().fit(e_tr.iloc[:, :144], e_tr.iloc[:, 144:])
        y_hat = m_lin.predict(e_te.iloc[:, :144])
        """

        m = CausalInvertibleNN(learning_rate=1e-2,  batch_size=1000, load_path=None, n_hidden_x=e_tr.shape[1],
                               n_layers=3, normalize_target=False, n_epochs=20, stopping_rounds=20, rel_tol=-1).fit(e_tr, e_tr)

        n_predict = 1000
        z = m.predict(e_te.iloc[:n_predict, :])

        # invert transform
        x_hat = m.invert(z)

        plt.figure()
        plt.hist(z.values.ravel(), bins=100, alpha=0.5)
        plt.hist(e_te.iloc[:n_predict, :].values.ravel(), bins=100, alpha=0.5)
        plt.show()

        fig, ax = plt.subplots(2, 1, layout='tight')
        ax[0].plot(e_te.iloc[:n_predict, :].values, alpha=0.2)
        ax[1].plot(z.values, alpha=0.2)


        fig, ax = plt.subplots(2, 1)
        ax[0].plot(np.quantile(e_te.iloc[:n_predict, :].values, np.linspace(0, 1, 10), axis=0).T, alpha=0.5, color='r')
        ax[1].plot(np.quantile(z.values, np.linspace(0, 1, 10), axis=0).T, alpha=0.5, color='r')


        fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)
        ax[0].plot(x_hat)
        ax[1].plot(e_te.iloc[:n_predict, :].values)
        ax[2].plot(x_hat - e_te.iloc[:n_predict, :].values)

        plt.close('all')

        z_tr = m.predict(e_tr)
        z_te = m.predict(e_te)
        m_lin = LinearForecaster().fit(z_tr.iloc[:, :144], z_tr.iloc[:, 144:])
        z_hat_e = pd.DataFrame(np.hstack([z_te.iloc[:, :144], m_lin.predict(z_te.iloc[:, :144])]), columns=e_te.columns)
        y_hat_e = m.invert(z_hat_e)
        y_invert = m.invert(z_te)

        fig, ax = plt.subplots(1, 1)
        for i in range(50):
            plt.cla()
            ax.plot(z_te.iloc[i, 144:].values.ravel())
            ax.plot(z_hat_e.iloc[i, 144:].values.ravel())
            plt.pause(0.01)


        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        for i in range(100):
            plt.cla()
            ax.plot(e_te.iloc[i, 144:].values)
            ax.plot(y_hat.iloc[i, :].values, linewidth=1)
            ax.plot(y_hat_e.iloc[i, 144:].values, linewidth=1)
            ax.plot(y_invert.iloc[i, 144:].values, linestyle='--')
            plt.pause(1e-6)
        """
        m = FFNN(n_layers=1, learning_rate=1e-3, batch_size=100, load_path=None, n_out=144, rel_tol=-1, stopping_rounds=20).fit(e_tr.iloc[:, :144], e_tr.iloc[:, 144:])
        y_hat = m.predict(e_te.iloc[:, :144])

        m = CausalInvertibleNN(learning_rate=1e-2, batch_size=200, load_path=None, n_hidden_x=e_tr.shape[1],
                               n_layers=4, normalize_target=False, n_epochs=5, stopping_rounds=20, rel_tol=-1,
                               end_to_end=True, n_hidden_y=300, n_prediction_layers=3).fit(e_tr, e_tr)



        z_hat_ete = m.predict(e_te)

        embeddings_hat = m.predict_batch(m.pars, e_te.values)[:, :e_te.shape[1] // 2]
        embeddings_future = m.predict_batch(m.pars, e_te.values)[:, e_te.shape[1] // 2:]

        plt.figure()
        plt.plot(embeddings_hat[:, 0], alpha=0.3)
        plt.plot(embeddings_future[:, 0], linestyle='--', alpha=0.3)
        plt.plot(y_hat.iloc[:, 0].values, linestyle='--', alpha=0.3)

        np.mean((z_hat_ete.values- e_te.iloc[:, 144:].values)**2)
        np.mean((y_hat.values- e_te.iloc[:, 144:].values)**2)

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        for i in range(2):
            plt.cla()
            ax.plot(e_te.iloc[i, 144:].values)
            ax.plot(y_hat.iloc[i, :].values, linewidth=1)
            ax.plot(z_hat_ete.iloc[i, :].values, linestyle='--')
            plt.pause(1e-6)



def boxconstr(x, ub, lb):
    return jnp.sum(jnp.maximum(0, x - ub)**2 + jnp.maximum(0, lb - x)**2)

def e_optobj_convexity_test(df, forecaster, ctrl_names, **objective_kwargs):
    x_names = [c for c in df.columns if not c in ctrl_names]
    rand_idxs = np.random.choice(len(df), 1)
    for idx in rand_idxs:
        x  = df[x_names].iloc[idx, :]
        ctrl_embedding = np.tile(np.random.randn(forecaster.n_embeddings, 1), 20).T

        plt.figure()
        for ctrl_e in range(forecaster.n_embeddings):
            ctrls = ctrl_embedding.copy()
            ctrls[:, ctrl_e] = np.linspace(-1, 1, 20)
            preds = np.hstack([forecaster._objective(c, np.atleast_2d(x.values.ravel()), **objective_kwargs) for c in ctrls])
            approx_second_der = np.round(np.diff(preds, 2, axis=0), 5)
            approx_second_der[approx_second_der == 0] = 0  # to fix the sign
            is_convex = np.all(np.sign(approx_second_der) >= 0)
            print('output is convex w.r.t. input {}: {}'.format(ctrl_e, is_convex))
            plt.plot(ctrls[:, ctrl_e], preds, alpha=0.3)
            plt.pause(0.001)
        plt.show()

def io_convexity_test(df, forecaster, ctrl_names, **objective_kwargs):
    rand_idxs = np.random.choice(len(df), 1)
    for idx in rand_idxs:
        df_i  = df.iloc[[idx]*20, :].copy()
        plt.figure()
        for ctrl_e in range(len(ctrl_names)):
            df_e = df_i.copy()
            df_e[ctrl_names[ctrl_e]] = np.linspace(-1, 1, 20)
            preds = np.hstack([forecaster.predict(df_e.loc[[i], :]) for i in df_e.index])
            approx_second_der = np.round(np.diff(preds, 2, axis=0), 5)
            approx_second_der[approx_second_der == 0] = 0  # to fix the sign
            is_convex = np.all(np.sign(approx_second_der) >= 0)
            print('output is convex w.r.t. input {}: {}'.format(ctrl_e, is_convex))
            plt.plot(preds, alpha=0.3)
            plt.pause(0.001)
        plt.show()

def eo_convexity_test(df, forecaster, ctrl_names, **objective_kwargs):
    x_names = [c for c in df.columns if not c in ctrl_names]
    rand_idxs = np.random.choice(len(df), 1)
    for idx in rand_idxs:
        x  = df[x_names].iloc[idx, :]
        ctrl_embedding = np.tile(np.random.randn(forecaster.n_embeddings, 1), 20).T

        plt.figure()
        for ctrl_e in range(forecaster.n_embeddings):
            ctrls = ctrl_embedding.copy()
            ctrls[:, ctrl_e] = np.linspace(-1, 1, 20)

            preds = np.dstack([latent_pred(forecaster.pars, forecaster.model, x, c) for c in ctrls])
            approx_second_der = np.round(np.diff(preds, 2, axis=0), 5)
            approx_second_der[approx_second_der == 0] = 0  # to fix the sign
            is_convex = np.all(np.sign(approx_second_der) >= 0)
            print('output is convex w.r.t. input {}: {}'.format(ctrl_e, is_convex))
            plt.plot(preds, alpha=0.3)
            plt.pause(0.001)
        plt.show()

if __name__ == '__main__':
    unittest.main()

