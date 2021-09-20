import unittest
import pandas as pd
import numpy as np
import pyforecaster.metrics as pyme
import logging


class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.t = 500
        self.n = 10
        times = pd.date_range('01-01-2020', '01-06-2020', freq='20min')
        self.target = pd.DataFrame(
            np.sin(np.arange(len(times)) * 10 * np.pi / len(times)).reshape(-1, 1) * np.random.randn(1, self.n),
            index=times)
        self.x = self.target + np.random.randn(len(self.target)).reshape(-1, 1)
        self.logger =logging.getLogger()
        logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                            filename=None)

    def test_summaryscore(self):
        agg_index = self.x.index.hour
        scores = pyme.summary_score(self.x, self.target, score=pyme.rmse, agg_index=agg_index)
        from pyforecaster.plot_utils import plot_summary_score
        plot_summary_score(scores, colorbar_label='rmse')
        assert scores.shape == (len(agg_index.unique()), self.target.shape[1])


if __name__ == '__main__':
    unittest.main()
