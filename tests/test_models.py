import unittest

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

from ray.tune import quniform

from pyforecaster.forecasting_models.holtwinters import HoltWinters, HoltWintersMulti
from pyforecaster.forecasting_models.fast_adaptive_models import Fourier_es, FK, FK_multi
from pyforecaster.forecasting_models.random_fourier_features import RFFRegression, AdditiveRFFRegression, BrutalRegressor
from pyforecaster.forecasting_models.randomforests import QRF
from pyforecaster.forecasting_models.gradientboosters import LGBMHybrid
from pyforecaster.forecasting_models.statsmodels_wrapper import ExponentialSmoothing
from pyforecaster.forecaster import LinearForecaster, LGBForecaster
from pyforecaster.plot_utils import plot_quantiles
from pyforecaster.formatter import Formatter
from pyforecaster.forecasting_models.neural_models.base_nn import FFNN
from pyforecaster.plot_utils import ts_animation
from pyforecaster.forecasting_models.benchmarks import Persistent, SeasonalPersistent

class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.t = 10000
        self.n = 3
        self.n_sa = 24
        self.periods = [self.n_sa, self.n_sa*7]

        self.x = pd.DataFrame(np.sin(np.arange(self.t)*2*np.pi/self.periods[0]).reshape(-1,1) * np.random.randn(1, self.n), index=pd.date_range('01-01-2020', '01-05-2020', self.t)) + 20
        self.x = self.x + pd.DataFrame(np.sin(np.arange(self.t)*2*np.pi/self.periods[1]).reshape(-1,1) * np.random.randn(1, self.n), index=pd.date_range('01-01-2020', '01-05-2020', self.t))
        self.y = pd.DataFrame((self.x.values  @ np.random.randn(self.n, 1)), columns=['target'], index=self.x.index)

        randpeak = np.random.rand(self.t) > 0.98
        randpeak = randpeak + np.roll(randpeak, 1) + np.roll(randpeak, 2)
        self.y_difficult = self.y.abs() + (randpeak.astype(int) * (self.y.abs().max()[0]-self.y.abs().mean()[0])*2 ).reshape(-1,1)

        self.data = pd.read_pickle('tests/data/test_data.zip').droplevel(0, 1)
        self.logger =logging.getLogger()
        logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                            filename=None)

    def test_hw(self):
        n_tr = int(len(self.x)*0.8)
        x, y = self.x, self.y
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        hw = HoltWinters(periods=self.periods, n_sa=self.n_sa, optimization_budget=10, q_vect=np.arange(11)/10, target_name='target').fit(pd.concat([x_tr, y_tr], axis=1), y_tr)
        #hw.reinit(x_tr['target'])
        y_hat = hw.predict(pd.concat([x_te,y_te], axis=1))

        ts_animation([y_hat], names=['y_hat', 'target'], target=y_te.values, frames=100, repeat=False)

    def test_fast_linreg(self):

        formatter = Formatter(logger=self.logger).add_transform(['all'], lags=np.arange(24),
                                                                    relative_lags=True)
        formatter.add_transform(['all'], ['min', 'max'], agg_bins=[1, 2, 15, 20])
        formatter.add_target_transform(['all'], lags=-np.arange(6))
        x, y = formatter.transform(self.data.iloc[:1000])
        x.columns = x.columns.astype(str)
        y.columns = y.columns.astype(str)
        n_tr = int(len(x) * 0.8)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        formatter_fast = Formatter(logger=self.logger).add_transform(['all'], lags=np.arange(24),
                                                                    relative_lags=True)
        x_fast, y_fast = formatter.transform(self.data.iloc[:1000])
        x_fast.columns = x_fast.columns.astype(str)
        y_fast.columns = y_fast.columns.astype(str)
        n_tr = int(len(x_fast) * 0.8)
        x_fast_tr, x_fast_te, y_fast_tr, y_fast_te = [x_fast.iloc[:n_tr, :].copy(), x_fast.iloc[n_tr:, :].copy(), y_fast.iloc[:n_tr].copy(),
                                  y_fast.iloc[n_tr:].copy()]


        m_lin = LinearForecaster(val_ratio=0.2, fit_intercept=False, normalize=False).fit(x_tr, y_tr)
        m_fast_lin = FFNN(val_ratio=0.2, learning_rate=0.01, n_layers=[200, 200, 200, y_tr.shape[1]], batch_size=400).fit(x_fast_tr, y_fast_tr, n_epochs=100)

        y_hat = m_lin.predict(x_te)
        y_hat_fast = m_fast_lin.predict(x_fast_te)

        s_a = 5
        y_te.iloc[:, s_a].plot()
        y_hat.iloc[:, s_a].plot()
        (y_hat_fast.iloc[:, s_a]).plot()

    def test_linreg_with_categorical_features(self):

        formatter = Formatter(logger=self.logger).add_transform(['all'], lags=np.arange(24), relative_lags=True)
        formatter.add_transform(['all'], ['min', 'max'], agg_bins=[1, 2, 15, 20])
        formatter.add_target_transform(['all'], lags=-np.arange(6))
        x, y = formatter.transform(self.data.iloc[:1000])
        x['cat'] = np.random.choice(['cat', 'dog'], len(x))
        x.columns = x.columns.astype(str)
        y.columns = y.columns.astype(str)
        n_tr = int(len(x) * 0.8)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        m_lin = LinearForecaster(val_ratio=0.2, fit_intercept=False, normalize=False).fit(x_tr, y_tr)
        y_hat = m_lin.predict(x_te)

        s_a = 5
        y_te.iloc[:, s_a].plot()
        y_hat.iloc[:, s_a].plot()

    def test_hw_difficult(self):

        n_tr = int(len(self.x) * 0.5)
        x, y = self.x, self.y_difficult
        y = np.maximum(0, y -np.quantile(y, 0.45))
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        hw = HoltWinters(periods=self.periods, n_sa=self.n_sa, optimization_budget=5, q_vect=np.arange(11)/10,
                         target_name='target').fit(y_tr)
        y_hat = hw.predict(y_te)

        hw_multi = HoltWintersMulti(periods=self.periods, n_sa=self.n_sa, optimization_budget=50, q_vect=np.arange(11)/10,
                         target_name='target', models_periods=np.array([1,2,3,5, 10, 24]), constraints=[0, np.inf]).fit(y_tr,y_tr)
        y_hat_multi = hw_multi.predict(y_te)

        ts_animation([y_hat, y_hat_multi], names=['y_hat', 'y_hat_multi', 'target'], target=y_te.values, frames=100, repeat=False)


    def test_hw_multi(self):
        self.data = self.data.resample('1h').mean()
        df_tr, df_te = self.data.iloc[:1200], self.data.iloc[1200:1500]
        steps_day = 24
        fks_multi = FK_multi(n_predictors=4, n_sa=steps_day, m=steps_day*7,
                             target_name='all', periodicity=steps_day*2,
                             optimize_hyperpars=True, optimize_submodels_hyperpars=True, optimization_budget=5, targets_names=df_tr.columns[:2], verbose=False, diagnostic_plots=False).fit(df_tr)
        hw_multi = HoltWintersMulti(periods=[steps_day, steps_day * 7], n_sa=steps_day, optimization_budget=5, q_vect=np.arange(11) / 10,
                         target_name='all', models_periods=np.array([1,2,steps_day]), targets_names=df_tr.columns[:6]).fit(df_tr)


        fes = Fourier_es(n_sa=steps_day, m=steps_day*7, target_name='all', optimization_budget=5).fit(df_tr, df_tr['all'])


        fks = FK(n_sa=steps_day, m=steps_day, target_name='all',
                 optimization_budget=2).fit(df_tr, df_tr['all'])



        hw = HoltWinters(periods=[steps_day, steps_day * 7], n_sa=steps_day, optimization_budget=10, q_vect=np.arange(11) / 10,
                         target_name='all').fit(df_tr, df_tr['all'])



        y_hat = hw.predict(df_te)
        y_hat_multi = hw_multi.predict(df_te)
        y_hat_fes = fes.predict(df_te)
        y_hat_fks = fks.predict(df_te)

        y_hat_fks_multi = fks_multi.predict(df_te)
        y_hat_fks_multi_q = fks_multi.predict_quantiles(df_te)

        ys = [y_hat, y_hat_multi, y_hat_fes, y_hat_fks, y_hat_fks_multi]
        ts_animation(ys, target = df_te['all'].values, names = ['hw', 'hw_multi', 'fes', 'fks', 'fks_multi', 'target'], frames = 120, interval = 1, step = 1, repeat = False)

    def test_linear_val_split(self):

        formatter = Formatter(logger=self.logger).add_transform(['all'], lags=np.arange(24),
                                                                    relative_lags=True)
        formatter.add_transform(['all'], ['min', 'max'], agg_bins=[1, 2, 15, 20])
        formatter.add_target_transform(['all'], lags=-np.arange(6))

        x, y = formatter.transform(self.data.iloc[:1000])
        n_tr = int(len(x) * 0.99)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        m_lin = LinearForecaster(val_ratio=0.2).fit(x_tr, y_tr)
        y_hat = m_lin.predict(x_te)

        m_lgb = LGBForecaster(val_ratio=0.5, lgb_pars={'num_leaves':20}).fit(x_tr, y_tr)
        y_hat_lgb = m_lgb.predict(x_te)
        q = m_lgb.predict_quantiles(x_te)

        plt.close('all')
        plot_quantiles([y_te, y_hat, y_hat_lgb], q, ['y_te', 'y_hat', 'y_hat_lgb'])
        plt.close('all')

    def test_qrf(self):
        formatter = Formatter(logger=self.logger).add_transform(['all'], lags=np.arange(24),
                                                                    relative_lags=True)
        formatter.add_transform(['all'], ['min', 'max'], agg_bins=[1, 2, 15, 20])
        formatter.add_target_transform(['all'], lags=-np.arange(6))
        formatter.add_target_normalizer(['all'], 'mean', agg_freq='3d', name='a_movingavg')
        formatter.add_target_normalizer(['all'], 'std', agg_freq='3d', name='a_movingstd')

        #m_lin = LinearForecaster(val_ratio=0.2, formatter=formatter).fit(x_tr, y_tr)
        #y_hat_nonorm = m_lin.predict(x_te)
        #q_nonorm = m_lin.predict_quantiles(x_te)

        #m_lgb = LGBForecaster(val_ratio=0.5, lgb_pars={'num_leaves':20}, formatter=formatter).fit(x_tr, y_tr)
        #y_hat_lgb = m_lgb.predict(x_te)
        #mae = lambda x, y: np.abs(x-y).mean().mean()
        #print('MAE lin:', mae(y_te, y_hat_nonorm))


        formatter.add_normalizing_fun(expr="(df[t] - df['a_movingavg']) / (df['a_movingstd'] + 1)",
                                      inv_expr="df[t]*(df['a_movingstd']+1) + df['a_movingavg']")
        x, y_norm = formatter.transform(self.data.iloc[:1000])
        n_tr = int(len(x) * 0.9)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y_norm.iloc[:n_tr].copy(), y_norm.iloc[n_tr:].copy()]

        qrf = QRF(val_ratio=0.2, formatter=formatter, n_jobs=4, n_single=6).fit(x_tr, y_tr)
        y_hat = qrf.predict(x_te)
        q = qrf.predict_quantiles(x_te)
        y_te = formatter.denormalize(x_te, y_te)
         #plot_quantiles([y_te, y_hat], q, ['y_te', 'y_hat', 'y_hat_qrf'])

        qrf = QRF(val_ratio=0.2, formatter=formatter, n_jobs=4, n_single=2).fit(x_tr, y_tr)
        y_hat = qrf.predict(x_te)
        q = qrf.predict_quantiles(x_te, quantiles=[0.1, 0.9])
        q = qrf.predict_quantiles(x_te, quantiles=[0.5])

        #plot_quantiles([y_te, y_hat], q, ['y_te', 'y_hat', 'y_hat_qrf'])

        y_hat = qrf.predict(x_te.iloc[[0], :])
        q = qrf.predict(x_te.iloc[[0], :])



    def test_antinormalize(self):
        formatter = Formatter(logger=self.logger, augment=False).add_transform(['all'], lags=np.arange(144),
                                                                    relative_lags=True)
        formatter.add_transform(['all'], ['min', 'max'], agg_bins=[1, 2, 15, 20])
        formatter.add_target_transform(['all'], lags=-np.arange(1, 145))

        formatter.add_target_normalizer(['all'], 'mean', agg_freq='3d', name='a_movingavg')
        formatter.add_target_normalizer(['all'], 'std', agg_freq='3d', name='a_movingstd')

        x, y = formatter.transform(self.data)

        n_tr = int(len(x) * 0.9)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        #m_lin = LinearForecaster(val_ratio=0.2, formatter=formatter).fit(x_tr, y_tr)
        #y_hat_nonorm = m_lin.predict(x_te)
        #q_nonorm = m_lin.predict_quantiles(x_te)

        #m_lgb = LGBForecaster(val_ratio=0.5, lgb_pars={'num_leaves':20}, formatter=formatter).fit(x_tr, y_tr)
        #y_hat_lgb = m_lgb.predict(x_te)
        #mae = lambda x, y: np.abs(x-y).mean().mean()
        #print('MAE lin:', mae(y_te, y_hat_nonorm))


        formatter.add_normalizing_fun(expr="(df[t] - df['a_movingavg']) / (df['a_movingstd'] + 1)",
                                      inv_expr="df[t]*(df['a_movingstd']+1) + df['a_movingavg']")
        x, y_norm = formatter.transform(self.data)
        y = formatter.denormalize(x, y_norm)

        x_tr, x_te, y_tr = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y_norm.iloc[:n_tr].copy()]
        m_lin = LinearForecaster(val_ratio=0.2, formatter=formatter).fit(x_tr, y_tr)
        y_hat = m_lin.predict(x_te)
        q = m_lin.predict_quantiles(x_te)
        plot_quantiles([y_te, y_hat], q, ['y_te', 'y_hat_lin'], n_rows=100)
        mae = lambda x, y: np.abs(x-y).mean().mean()
        print('MAE lin:', mae(y_te, y_hat))

    def test_statsmodels_wrappers(self):
        self.data = self.data.resample('1h').mean()
        data_tr = self.data.iloc[:100]
        data_te = self.data.iloc[100:300]
        m = ExponentialSmoothing(target_name='all', q_vect=np.arange(31)/30, nodes_at_step=None, val_ratio=0.8, n_sa=24,
                                 seasonal=24).fit(data_tr)
        y_hat = m.predict(data_te)
        q_hat = m.predict_quantiles(data_te, dataframe=False)
        y_plot = pd.concat({'y_{:02d}'.format(i): data_te['all'].shift(-i) for i in range(24)}, axis=1)
        #plot_quantiles([y_plot, y_hat], q_hat, ['y_te', 'y_hat'], n_rows=300)

        discr_prob = m.predict_pmf(data_te, discrete_prob_space=np.linspace(500, 1200, 10))
        plt.matshow(discr_prob[2].T )
        import seaborn as sb

        i = 1
        plt.figure(figsize=(10, 6))
        x_bins = np.arange(24)
        y_bins = np.linspace(500, 1200, 10)
        extent = [x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()]
        plt.imshow(discr_prob[i].T, aspect='auto', extent=extent, origin='lower', cmap='plasma')

        # Overlay the time series on top
        plt.plot(x_bins, y_plot.values[i, :], color='black', linewidth=2, label="Time Series")
        plt.plot(x_bins, q_hat[i, :, :], color='black', linewidth=2, label="Time Series")

    def test_persistence(self):
        self.data = self.data.resample('1h').mean()
        data_tr = self.data.iloc[:2100]
        data_te = self.data.iloc[2100:2400]
        m = Persistent(target_col='all', n_sa=24, q_vect=np.arange(11)/10, val_ratio=0.8, conditional_to_hour=False).fit(data_tr)
        y_hat = m.predict(data_te)
        q_hat = m.predict_quantiles(data_te)
        y_plot = pd.concat({'y_{:02d}'.format(i): data_te['all'].shift(-i) for i in range(24)}, axis=1)
        plot_quantiles([y_plot, y_hat], q_hat, ['y_te', 'y_hat'], n_rows=300)

        m = SeasonalPersistent(target_col='all', seasonality=24, n_sa=24, q_vect=np.arange(11) / 10, val_ratio=0.8,
                       conditional_to_hour=True).fit(data_tr)
        y_hat = m.predict(data_te)
        q_hat = m.predict_quantiles(data_te)
        y_plot = pd.concat({'y_{:02d}'.format(i): data_te['all'].shift(-i) for i in range(24)}, axis=1)
        plot_quantiles([y_plot, y_hat], q_hat, ['y_te', 'y_hat'], n_rows=300)


if __name__ == '__main__':
    unittest.main()
