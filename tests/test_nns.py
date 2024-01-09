import unittest
import matplotlib.pyplot as plt
import optax
import pandas as pd
import numpy as np
import logging
from pyforecaster.forecasting_models.neural_forecasters import PICNN, RecStablePICNN, NN, PIQCNN, PIQCNNSigmoid, StructuredPICNN
from pyforecaster.trainer import hyperpar_optimizer
from pyforecaster.formatter import Formatter
from pyforecaster.metrics import nmae
from os import makedirs
from os.path import exists, join
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

        savepath_tr_plots = 'tests/results/figs/convexity'

        # if not there, create directory savepath_tr_plots
        if not exists(savepath_tr_plots):
            makedirs(savepath_tr_plots)


        optimization_vars = x_tr.columns[:100]


        m_1 = PICNN(learning_rate=1e-3, batch_size=500, load_path=None, n_hidden_x=20, n_hidden_y=20,
                  n_out=y_tr.shape[1], n_layers=3, optimization_vars=optimization_vars,probabilistic=True, probabilistic_loss_kind='crps', rel_tol=-1,
                    val_ratio=0.2).fit(x_tr, y_tr,n_epochs=2,stats_step=50,savepath_tr_plots=savepath_tr_plots)

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

        n_tr = int(len(x) * 0.8)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        savepath_tr_plots = 'tests/results/ffnn_tr_plots'

        # if not there, create directory savepath_tr_plots
        if not exists(savepath_tr_plots):
            makedirs(savepath_tr_plots)

        optimization_vars = x_tr.columns[:2]
        causal_df = pd.DataFrame(np.tril(np.ones((len(optimization_vars), y_tr.shape[1]))), index=optimization_vars, columns=y_tr.columns)

        m_1 = PICNN(learning_rate=1e-3, batch_size=1000, load_path=None, n_hidden_x=200, n_hidden_y=200,
                    n_out=y_tr.shape[1], n_layers=3, optimization_vars=optimization_vars, causal_df = causal_df, probabilistic=True).fit(x_tr,
                                                                                              y_tr,
                                                                                              n_epochs=1,
                                                                                              savepath_tr_plots=savepath_tr_plots,
                                                                                              stats_step=10)
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

        m_1 = StructuredPICNN(learning_rate=1e-4, batch_size=100, load_path=None, n_hidden_x=250, n_hidden_y=250,
                  n_out=y_tr.shape[1], n_layers=3, optimization_vars=optimization_vars,stopping_rounds=100
                              , layer_normalization=True, objective_fun=objective, probabilistic=True, probabilistic_loss_kind='crps', normalize_target=False).fit(x_tr,
                                                                                            y_tr,
                                                                                            n_epochs=2,
                                                                                            savepath_tr_plots=savepath_tr_plots,
                                                                                            stats_step=500, rel_tol=-1)
        from jax import vmap
        objs = vmap(objective,in_axes=(0, 0))(y_te.values, x_te.values)
        rnd_idxs = np.random.choice(x_te.shape[0], 5000, replace=False)
        rnd_idxs =  rnd_idxs[np.argsort(objs[rnd_idxs])]

        fig, ax = plt.subplots(2, 1, figsize=(5, 10))
        ax[0].plot(objs[rnd_idxs], label='y_true')
        ax[0].plot(m_1.predict(x_te.iloc[rnd_idxs, :], return_obj=True)[1].values.ravel(), label='y_hat')
        ax[1].scatter(objs[rnd_idxs], m_1.predict(x_te.iloc[rnd_idxs, :], return_obj=True)[1].values.ravel(), s=1)
        ax[0].plot(np.squeeze(m_1.predict_quantiles(x_te.iloc[rnd_idxs, :], return_obj=True)), color='orange', alpha=0.3)


        ordered_idx = np.argsort(np.abs(objs[rnd_idxs] - m_1.predict(x_te.iloc[rnd_idxs, :], return_obj=True)[1].values.ravel()))


        for r in rnd_idxs[ordered_idx[-10:]]:
            y_hat = m_1.predict(x_te.iloc[[r], :])
            #q_hat = m_1.predict_quantiles(x_te.iloc[[r], :])
            plt.figure()
            plt.plot(y_te.iloc[r, :].values.ravel(), label='y_true')
            plt.plot(y_hat.values.ravel(), label='y_hat')
            #plt.plot(np.squeeze(q_hat), label='q_hat', color='orange', alpha=0.3)
            plt.legend()


if __name__ == '__main__':
    unittest.main()

