import unittest
import pandas as pd
import numpy as np
import pyforecaster.pyforecaster as pyf
import logging


class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.t = 25
        self.n = 10
        self.x = pd.DataFrame(np.random.randn(self.t, self.n), index=pd.date_range('01-01-2020', '01-02-2020', self.t))
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



if __name__ == '__main__':
    unittest.main()
