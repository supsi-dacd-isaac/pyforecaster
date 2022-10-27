import unittest

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from pyforecaster.forecasting_models.holtwinters import HoltWinters, HoltWintersMulti


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

        hw = HoltWinters(periods=self.periods, n_sa=self.n_sa, optimization_budget=100, q_vect=np.arange(11)/10, target_name='target').fit(pd.concat([x_tr, y_tr], axis=1), y_tr)
        #hw.reinit(x_tr['target'])
        y_hat = hw.predict(pd.concat([x_te,y_te], axis=1))

        n_plot = 50
        for i in range(n_plot):
            plt.cla()
            plt.plot(y_te.iloc[i+1:i+self.periods[1]+1].values)
            plt.plot(y_hat[i, :])
            plt.pause(0.0001)

    def test_hw_difficult(self):

        n_tr = int(len(self.x) * 0.5)
        x, y = self.x, self.y_difficult
        y = np.maximum(0, y -np.quantile(y, 0.45))
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        hw = HoltWinters(periods=self.periods, n_sa=self.n_sa, optimization_budget=50, q_vect=np.arange(11)/10,
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

    def test_hw_multi(self):
        n_tr = int(len(self.x) * 0.5)
        df_tr, df_te = self.data.iloc[:n_tr], self.data.iloc[n_tr:]

        hw = HoltWinters(periods=[144, 144 * 7], n_sa=144, optimization_budget=50, q_vect=np.arange(11) / 10,
                         target_name='all').fit(df_tr, df_tr['all'])
        hw_multi = HoltWintersMulti(periods=[144, 144 * 7], n_sa=144, optimization_budget=50, q_vect=np.arange(11) / 10,
                         target_name='all', models_periods=np.array([1,2,3,4,6,10,20,40,96,144])).fit(df_tr, df_tr['all'])

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



if __name__ == '__main__':
    unittest.main()
