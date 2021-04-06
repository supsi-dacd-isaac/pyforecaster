import unittest
import pandas as pd
import numpy as np
import pyforecaster.pyforecaster as pyf
import logging
from pyforecaster.plot_utils import ts_animation


class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.t = 150
        self.n = 10
        self.x = pd.DataFrame(np.sin(np.arange(self.t)*4*np.pi/self.t).reshape(-1,1) * np.random.randn(1, self.n), index=pd.date_range('01-01-2020', '01-02-2020', self.t))
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
        tr = pyf.Transformer([0, 1], ['mean', 'max', lambda x: x**2-1], '3h', [-1, -2], logger=self.logger)
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

if __name__ == '__main__':
    unittest.main()
