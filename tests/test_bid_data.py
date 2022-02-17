import unittest
import pandas as pd
import numpy as np
from pyforecaster.stat_utils import bootstrap
import logging
from pyforecaster.big_data_utils import reduce_mem_usage


class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.N = 1000
        self.t = 10000
        self.df = pd.DataFrame(np.random.randn(self.t, self.N))
        self.logger =logging.getLogger()

    def test_reduce_mem(self):
        df = reduce_mem_usage(self.df, logger=self.logger)
        assert len(df) == len(self.df)


if __name__ == '__main__':
    unittest.main()
