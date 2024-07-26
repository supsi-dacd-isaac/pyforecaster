import unittest

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from pyforecaster.forecasting_models.gradientboosters import LGBMHybrid
from pyforecaster.forecaster import LGBForecaster, LinearForecaster
from pyforecaster.plot_utils import plot_quantiles, plot_trees
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
        formatter.add_target_transform(['all'], lags=-np.arange(24)-1)

        x, y = formatter.transform(self.data.resample('1h').mean())
        n_tr = int(len(x) * 0.7)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        m_lin = LinearForecaster(q_vect=np.linspace(0.1, 0.9, 11), cov_est_method='vanilla').fit(x_tr, y_tr)
        y_hat_lin = m_lin.predict(x_te)
        q = m_lin.predict_quantiles(x_te)

        m_lgbhybrid = LGBMHybrid(red_frac_multistep=0.1,  val_ratio=0.3, lgb_pars={'num_leaves': 300, 'n_estimators': 10, 'learning_rate':0.05},
                                 n_single=10, parallel=True, formatter=formatter, metadata_features=['minuteofday', 'utc_offset', 'dayofweek', 'hour'],tol_period='1h', keep_last_seconds=3600).fit(x_tr, y_tr)
        y_hat_lgbh = m_lgbhybrid.predict(x_te)
        q = m_lgbhybrid.predict_quantiles(x_te)

        m_lgb = LGBForecaster(lgb_pars={'num_leaves': 10, 'n_estimators': 10, 'learning_rate':0.05}, parallel=True).fit(x_tr, y_tr)
        y_hat_lgb = m_lgb.predict(x_te)

        # plot_quantiles([y_hat_lgbh.iloc[:100, :]], q[:100, :, :], ['y_hat_lgb'], n_rows=100, repeat=True)
        plot_quantiles([y_hat_lin.iloc[:100, :]], q[:100, :, :], ['y_hat_lgb'], n_rows=100, repeat=False)


    def do_not_test_linear_val_split(self):

        formatter = Formatter(logger=self.logger).add_transform(['all'], lags=np.arange(144),
                                                                    relative_lags=True)
        formatter.add_transform(['all'], lags=np.arange(144)+144*6)
        formatter.add_transform(['all'], ['min', 'max'], agg_bins=[1, 2, 15, 24, 48, 144])
        formatter.add_target_transform(['all'], lags=-np.arange(144)-1)

        x, y = formatter.transform(self.data)
        n_tr = int(len(x) * 0.8)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        m_lin = LinearForecaster(q_vect=np.linspace(0.01, 0.99, 11), cov_est_method='vanilla', val_ratio=0.6, online_tree_reduction=False,
                                 nodes_at_step=np.logspace(0.5, 2, 144,base=5).astype(int)).fit(x_tr, y_tr)
        y_hat_lgbh = m_lin.predict(x_te)
        q = m_lin.predict_quantiles(x_te)

        trees = m_lin.predict_trees(x_te.iloc[:500, :])
        plot_trees(trees, [y_te.iloc[:500, :].values], frames=500,
                              ax_labels={'x':"step [10 min]", "y": 'P [kW]'}, layout='tight', figsize=(8, 4),
                              legend_kwargs={'loc':'upper right'}, savepath='linear_preds_tree.mp4')
        plt.close('all')
        plot_quantiles([y_te, y_hat_lgbh], q, ['y_te', 'y_hat'], n_rows=500,
                              ax_labels={'x':"step [10 min]", "y": 'P [kW]'}, layout='tight', figsize=(8, 4),
                              legend_kwargs={'loc':'upper right'}, savepath='linear_preds.mp4')


        formatter.plot_transformed_feature(self.data, 'all',
                             ax_labels={'x':"step [1 day]", "y": 'P [kW]'}, layout='tight', figsize=(8, 4),
                             legend_kwargs={'loc':'upper right'}, frames=500, savepath='transformed_features.mp4')
if __name__ == '__main__':
    unittest.main()
