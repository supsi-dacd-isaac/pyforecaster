import unittest
import pandas as pd
import numpy as np
import pyforecaster.formatter as pyf
import logging
from pyforecaster.plot_utils import ts_animation
import matplotlib.pyplot as plt
import seaborn as sb


class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.t = 500
        self.n = 10
        self.x = pd.DataFrame(
            np.sin(np.arange(self.t) * 10 * np.pi / self.t).reshape(-1, 1) * np.random.randn(1, self.n),
            index=pd.date_range('01-01-2020', '01-05-2020', self.t))

        times = pd.date_range('01-01-2020', '01-05-2020', freq='20min')
        self.x2 = pd.DataFrame(
            np.sin(np.arange(len(times)) * 10 * np.pi / len(times)).reshape(-1, 1) * np.random.randn(1, self.n),
            index=times)
        self.x3 = pd.DataFrame((np.arange(len(times)) % 20).reshape(-1,1) * np.random.rand(1, 3), index=times)

        self.logger =logging.getLogger()
        logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                            filename=None)

    def test_transformer(self):

        tr = pyf.Transformer([0], ['mean'], logger=self.logger)
        _ = tr.transform(self.x)
        tr = pyf.Transformer([0], ['mean'], agg_freq='2h', logger=self.logger)
        _ = tr.transform(self.x)
        tr = pyf.Transformer([0], ['mean'], agg_freq='2h', lags=[1, 2, -1, -2], logger=self.logger)
        _ = tr.transform(self.x)
        tr = pyf.Transformer([0, 1], ['mean', 'max'], '3h', [-1, -2], logger=self.logger)
        _ = tr.transform(self.x)
        tr = pyf.Transformer([0, 1], ['mean', 'max', lambda x: np.sum(x**2)-1], '3h', [-1, -2], logger=self.logger)
        _ = tr.transform(self.x)
        tr = pyf.Transformer([0], agg_freq='3h', lags=[-1, -2, -3], logger=self.logger)
        _ = tr.transform(self.x, augment=False)

        tr = pyf.Transformer([0], ['mean'], '3h', [-1], logger=self.logger)
        x_tr = tr.transform(self.x, augment=False)

        assert x_tr.shape[1] == 1

    def test_formatter(self):
        formatter = pyf.Formatter(logger=self.logger).add_transform([0, 1, 2, 3], ['mean', 'max'], agg_freq='2h', lags=[-1,-2, -10])
        formatter.add_target_transform([3], lags=np.arange(10))
        formatter.plot_transformed_feature(self.x, 3)
        formatter.plot_transformed_feature(self.x, 0)
        x_tr, y_tr = formatter.transform(self.x)
        assert x_tr.isna().sum().sum() == 0 and y_tr.isna().sum().sum() == 0 and y_tr.shape[0] == x_tr.shape[0]

    def test_tkfcv_simulate_transform(self):
        formatter = pyf.Formatter(logger=self.logger).add_transform([0, 1, 2, 3], ['mean', 'max'], agg_freq='2h',
                                                                    lags=[-1, -2, -10])
        formatter.add_target_transform([3], lags=np.arange(10))
        folds_df = formatter.time_kfcv(self.x.index, 4, 3, x=self.x)
        for fold_name in folds_df.stack().columns:
            assert np.sum(folds_df[fold_name]['tr']) + np.sum(folds_df[fold_name]['te']) < len(self.x) - 1

    def test_tkfcv_pretransform(self):
        formatter = pyf.Formatter(logger=self.logger).add_transform([0, 1, 2, 3], ['mean', 'max'], agg_freq='2h', lags=[-1,-2, -10], relative_lags=True)
        formatter.add_target_transform([3], lags=np.arange(10))
        formatter.transform(self.x2)
        folds_df = formatter.time_kfcv(self.x2.index, 4, 3)
        for fold_name in folds_df.stack().columns:
            assert np.sum(folds_df[fold_name]['tr']) + np.sum(folds_df[fold_name]['te']) < len(self.x2) -1

    def test_prune_at_stepahead(self):
        formatter = pyf.Formatter(logger=self.logger).add_transform([0, 1, 2, 3], ['mean', 'max'], agg_freq='2h',
                                                                    lags=-np.arange(24*3), relative_lags=False)
        formatter.add_target_transform([3], lags=np.arange(10))
        x_transformed, y_transformed = formatter.transform(self.x2)
        crosspattern = pd.DataFrame()
        for i in range(10):
            x_i = formatter.prune_dataset_at_stepahead(x_transformed, i, method='periodic', period='24H', tol_period='10m')
            crosspattern = crosspattern.combine_first(pd.DataFrame(1, index=x_i.columns, columns=[i]))
        sb.heatmap(crosspattern)

    def test_nonanticipativity(self):

        formatter = pyf.Formatter(logger=self.logger).add_transform([0, 1, 2], ['max'], agg_freq='40min', lags=-1-np.arange(2), relative_lags=True)
        formatter.add_target_transform([2], lags=-np.arange(30)-1)
        x_transformed, y_transformed = formatter.transform(self.x3)
        formatter.plot_transformed_feature(self.x3, 2)


if __name__ == '__main__':
    unittest.main()
