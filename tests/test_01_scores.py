import unittest
import pandas as pd
import numpy as np
import pyforecaster.metrics as pyme
import logging
from pyforecaster.forecaster import LinearForecaster


class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.t = 500
        self.n = 10
        times = pd.date_range('01-01-2020', '01-06-2020', freq='20min')
        self.target = pd.DataFrame(
            np.sin(np.arange(len(times)) * 10 * np.pi / len(times)).reshape(-1, 1) * np.random.randn(1, self.n),
            index=times)
        self.x = self.target + np.random.randn(*self.target.shape)
        self.logger =logging.getLogger()
        logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                            filename=None)

    def test_probabilit_metrics(self):
        q_vect = [0.1, 0.5, 0.9]
        m = LinearForecaster(q_vect=q_vect).fit(self.x, self.target)
        q = m.predict_quantiles(self.x, quantiles=q_vect)
        crps, reliability = pyme.quantile_scores(q, self.target, alphas=q_vect)
        pyme.reliability(q, self.target, alphas=q_vect, get_score=True)
        pyme.crps(q, self.target, alphas=q_vect)

    def test_summaryscore(self):
        agg_index = self.x.index.hour
        scores = pyme.summary_score(self.x, self.target, score=pyme.rmse, agg_index=agg_index)
        from pyforecaster.plot_utils import plot_summary_score
        plot_summary_score(scores, colorbar_label='rmse')
        assert scores.shape == (len(agg_index.unique()), self.target.shape[1])

    def test_summarscores(self):

        agg_index = pd.DataFrame({'hour':self.x.index.hour, 'weekday':self.x.index.weekday})
        print(agg_index.dtypes)
        mask = self.x > 0.1
        scores = pyme.summary_scores(self.x, self.target, metrics=[pyme.rmse, pyme.mape, pyme.nmae],
                                     idxs=agg_index, mask=mask)
        print(scores)
        print([s.shape[0]  for s in scores.values()])
        print([np.sum([len(v.value_counts()) for k, v in agg_index.items()]) for s in scores.values()])

        assert np.all([s.shape[0] == np.sum([len(v.value_counts()) for k, v in agg_index.items()])
                       for s in scores.values()])

    def test_summaryscore_2(self):
        agg_index = self.x.index.hour
        scores = pyme.summary_score(self.x, self.target, score=pyme.rmse, agg_index=agg_index)
        scores.columns = scores.columns.astype(str) + 'd'
        from pyforecaster.plot_utils import plot_summary_score
        plot_summary_score(scores, colorbar_label='rmse', numeric_xticks=True)
        assert scores.shape == (len(agg_index.unique()), self.target.shape[1])


if __name__ == '__main__':
    unittest.main()
