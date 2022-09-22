import unittest
import pandas as pd
import numpy as np
from pyforecaster.stat_utils import bootstrap
import logging


class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.t = 500
        self.n = 10
        self.x = pd.DataFrame(np.sin(np.arange(self.t)*10*np.pi/self.t).reshape(-1,1) * np.random.randn(1, self.n), index=pd.date_range('01-01-2020', '01-05-2020', self.t))
        self.logger =logging.getLogger()
        logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                            filename=None)

    def test_bootstrap(self):
        summary = bootstrap(self.x[0], 'mean')
        assert len(summary) == 2


if __name__ == '__main__':
    unittest.main()
