import unittest
import pandas as pd
import numpy as np
from pyforecaster.big_data_utils import reduce_mem_usage
import logging


class TestBigData(unittest.TestCase):
    def setUp(self) -> None:
        N = 1000
        t = 10000
        self.df = pd.DataFrame(np.random.randn(t, N))
        self.logger =logging.getLogger()
        logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                            filename=None)

    def test_bootstrap(self):
        print(self.df.memory_usage().sum())
        dfa = reduce_mem_usage(self.df, logger=self.logger, parallel=True)
        print(dfa.memory_usage().sum())


if __name__ == '__main__':
    unittest.main()
