import os
import unittest


import pandas as pd
import numpy as np
import pyforecaster.formatter as pyf
from pyforecaster.reconciliation.reconciliation import HierarchicalReconciliation
import logging
from pyforecaster.metrics import crps, reliability, rmse


class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.t = 100
        self.n = 4
        self.x = pd.DataFrame(
            np.sin(np.arange(self.t) * 5 * np.pi / self.t).reshape(-1, 1) * np.random.randn(1, self.n),
            index=pd.date_range('01-01-2020', '01-05-2020', self.t, tz='Europe/Zurich'))


        self.logger =logging.getLogger()
        logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                            filename=None)

    def test_global_formatter(self):
        formatter = pyf.Formatter(logger=self.logger).add_transform([0, 1], ['mean', 'max'], agg_freq='2h',
                                                                    lags=[-1, -2, -10])
        formatter.add_target_transform(['target'], lags=np.arange(24))
        x_tr, y_tr = formatter.transform(self.x, global_form=True)

        assert x_tr.isna().sum().sum() == 0 and y_tr.isna().sum().sum() == 0 and y_tr.shape[0] == x_tr.shape[0]

    def test_hierarchical(self):
        formatter = pyf.Formatter(logger=self.logger)
        formatter.add_target_transform(['target'], lags=np.arange(24))
        hierarchy = {'s_0': ['s_1', 's_2'], 's_1': [0, 1], 's_2': [2, 3]}
        hierarchy_flatten = HierarchicalReconciliation.unroll_dict(hierarchy)
        for k, v in hierarchy_flatten.items():
            self.x[k] = self.x[v].sum(axis=1)
        x_tr, y_tr = formatter.transform(self.x, global_form=True)
        hr = HierarchicalReconciliation(hierarchy, n_scenarios=10, q_vect=np.linspace(0.1, 0.9, 5)).fit(x_tr, y_tr)

        y_hat = hr.predict(x_tr)
        q_hat = hr.predict_quantiles(x_tr)

        rmses = hr.compute_kpis(y_hat, x_tr, y_tr, rmse)
        crpss = hr.compute_kpis(q_hat, x_tr, y_tr, crps, alphas=hr.q_vect)
        reliabilities = hr.compute_kpis(q_hat, x_tr, y_tr, reliability, alphas=hr.q_vect)


if __name__ == '__main__':
    unittest.main()
