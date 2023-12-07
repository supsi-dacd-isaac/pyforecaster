import unittest

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from pyforecaster.forecasting_models.holtwinters import HoltWinters, HoltWintersMulti
from pyforecaster.forecasting_models.randomforests import QRF
from pyforecaster.forecaster import LinearForecaster, LGBForecaster
from pyforecaster.plot_utils import plot_quantiles
from pyforecaster.formatter import Formatter
from pyforecaster.forecasting_models.neural_forecasters import FFNN

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

        n_plot = 50
        for i in range(n_plot):
            plt.cla()
            plt.plot(y_te.iloc[i+1:i+self.periods[1]+1].values)
            plt.plot(y_hat[i, :])
            plt.pause(0.0001)
        plt.close('all')

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

    def test_hw_difficult(self):

        n_tr = int(len(self.x) * 0.5)
        x, y = self.x, self.y_difficult
        y = np.maximum(0, y -np.quantile(y, 0.45))
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        hw = HoltWinters(periods=self.periods, n_sa=self.n_sa, optimization_budget=5, q_vect=np.arange(11)/10,
                         target_name='target').fit(None, y_tr)
        y_hat = hw.predict(y_te)

        hw_multi = HoltWintersMulti(periods=self.periods, n_sa=self.n_sa, optimization_budget=50, q_vect=np.arange(11)/10,
                         target_name='target', models_periods=np.array([1,2,3,5, 10, 24]), constraints=[0, np.inf]).fit(None, y_tr)
        y_hat_multi = hw_multi.predict(y_te)

        n_plot = 50
        for i in range(n_plot):
            plt.cla()
            plt.plot(y_te.iloc[i+1:i+self.periods[1]+1].values)
            plt.plot(y_hat[i, :], label='y_hat')
            plt.plot(y_hat_multi[i, :], label='yhat constrained')
            plt.legend()
            plt.pause(0.0001)
        plt.close('all')


    def test_hw_multi(self):
        n_tr = int(len(self.x) * 0.5)
        df_tr, df_te = self.data.iloc[:n_tr], self.data.iloc[n_tr:]

        hw = HoltWinters(periods=[144, 144 * 7], n_sa=144, optimization_budget=5, q_vect=np.arange(11) / 10,
                         target_name='all').fit(df_tr, df_tr['all'])
        hw_multi = HoltWintersMulti(periods=[144, 144 * 7], n_sa=144, optimization_budget=50, q_vect=np.arange(11) / 10,
                         target_name='all', models_periods=np.array([1,2,144])).fit(df_tr, df_tr['all'])

        y_hat = hw.predict(df_te)
        y_hat_multi = hw_multi.predict(df_te)

        n_plot = 50
        for i in range(n_plot):
            plt.cla()
            plt.plot(df_te['all'].iloc[i + 1:i + self.periods[1] + 1].values, label='observed')
            plt.plot(y_hat[i, :], label='HW')
            plt.plot(y_hat_multi[i, :], label='HW multi')
            plt.legend()
            plt.pause(0.0001)

    def test_linear_val_split(self):

        formatter = Formatter(logger=self.logger).add_transform(['all'], lags=np.arange(24),
                                                                    relative_lags=True)
        formatter.add_transform(['all'], ['min', 'max'], agg_bins=[1, 2, 15, 20])
        formatter.add_target_transform(['all'], lags=-np.arange(6))

        x, y = formatter.transform(self.data.iloc[:5000])
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

        x, y = formatter.transform(self.data.iloc[:1000])
        x.columns = x.columns.astype(str)
        y.columns = y.columns.astype(str)
        n_tr = int(len(x) * 0.99)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        qrf = QRF(val_ratio=0.2, formatter=formatter, n_jobs=4, n_single=6).fit(x_tr, y_tr)
        y_hat = qrf.predict(x_te)
        q = qrf.predict_quantiles(x_te)

         #plot_quantiles([y_te, y_hat], q, ['y_te', 'y_hat', 'y_hat_qrf'])

        qrf = QRF(val_ratio=0.2, formatter=formatter, n_jobs=4, n_single=2).fit(x_tr, y_tr)
        y_hat = qrf.predict(x_te)
        q = qrf.predict_quantiles(x_te)

        #plot_quantiles([y_te, y_hat], q, ['y_te', 'y_hat', 'y_hat_qrf'])

        y_hat = qrf.predict(x_te.iloc[[0], :])
        q = qrf.predict(x_te.iloc[[0], :])

if __name__ == '__main__':
    unittest.main()
