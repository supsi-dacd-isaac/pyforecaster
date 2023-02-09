import unittest

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from pyforecaster.forecasting_models.gradientboosters import LGBMHybrid
from pyforecaster.forecaster import LGBForecaster, LinearForecaster
from pyforecaster.plot_utils import plot_quantiles
from pyforecaster.formatter import Formatter


class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.read_pickle('tests/data/test_data.zip').droplevel(0, 1)
        self.logger =logging.getLogger()
        logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                            filename=None)

    def test_linear_val_split(self):

        formatter = Formatter(logger=self.logger).add_transform(['all'], lags=np.arange(24),
                                                                    relative_lags=True)
        formatter.add_transform(['all'], ['min', 'max'], agg_bins=[1, 2, 15, 24])
        formatter.add_target_transform(['all'], lags=-np.arange(24)-1)

        x, y = formatter.transform(self.data.iloc[:3000])
        n_tr = int(len(x) * 0.99)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        m_lin = LinearForecaster(q_vect=np.linspace(0.1, 0.9, 11), cov_est_method='vanilla').fit(x_tr, y_tr)
        y_hat_lin = m_lin.predict(x_te)
        q = m_lin.predict_quantiles(x_te)

        m_lgbhybrid = LGBMHybrid(red_frac_multistep=0.1,  val_ratio=0.3, lgb_pars={'num_leaves': 100, 'n_estimators': 100, 'learning_rate':0.05}, n_single=1).fit(x_tr, y_tr)
        y_hat_lgbh = m_lgbhybrid.predict(x_te)
        q = m_lgbhybrid.predict_quantiles(x_te)

        m_lgb = LGBForecaster(lgb_pars={'num_leaves': 10, 'n_estimators': 100, 'learning_rate':0.05}).fit(x_tr, y_tr)
        y_hat_lgb = m_lgb.predict(x_te)

        plt.close('all')
        plot_quantiles([y_te, y_hat_lin, y_hat_lgbh, y_hat_lgb], q, ['y_te', 'y_lin', 'y_lgbhybrid_1', 'y_hat_lgb'])


if __name__ == '__main__':
    unittest.main()
