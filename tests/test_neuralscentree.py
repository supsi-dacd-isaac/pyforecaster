import unittest
import pandas as pd
from pyforecaster.neural_gas_tree import NeuralGas, NeuralDiffTree
from pyforecaster.forecaster import ScenGen
import numpy as np
from scipy.stats import multivariate_normal, norm, weibull_min


class TestScenarios(unittest.TestCase):
    def setUp(self) -> None:
        self.ng = NeuralGas(savepath='figs/neural_gas', init='quantiles', base_tree='quantiles')
        self.ndt = NeuralDiffTree(savepath='figs/diff_tree', init='quantiles', base_tree='quantiles')
        n_days = 10
        self.t = 24 * n_days
        self.lags = 24
        t_index = pd.date_range('01-01-2020', '01-10-2020', self.t)
        signal = pd.DataFrame(np.sin(np.arange(self.t)*2*n_days*np.pi/self.t).reshape(-1,1), index=t_index) \
                 + 0.1 * np.random.randn(self.t, 1)
        self.target = pd.concat([signal.shift(l) for l in -np.arange(self.lags)], axis=1)
        self.x = pd.concat([self.target.iloc[:,0].shift(l) for l in np.arange(1,3)], axis=1)
        self.x = pd.concat([pd.Series(self.x.index.minute + self.x.index.hour*60, index=self.x.index), self.x], axis=1)
        self.x.fillna(0, inplace=True)
        self.x = self.x.loc[~self.target.isna().any(axis=1)]
        self.target = self.target.loc[~self.target.isna().any(axis=1)]
        self.target.columns = ['target_{}'.format(i) for i in range(self.lags)]
        self.q_vect = np.linspace(0, 1, 11)[1:-1]

        self.quantiles_df = pd.concat({q: self.target + norm.ppf(q) * np.linspace(0.3, 1, self.lags)
                                       for q in self.q_vect}, axis=1)

    def test_neural_gas(self):
        sg = ScenGen(cov_est_method='shrunk').fit(self.target)
        rand_idx = np.random.choice(len(self.quantiles_df), 5)
        scenarios = np.squeeze(sg.predict(self.quantiles_df.iloc[[0, 15]], 50, kind='scenarios', q_vect=self.q_vect))
        #tree = self.ndt.gen_tree(np.hstack([scenarios[0], scenarios[1]]), k_max=100)
        #tree = self.ng.gen_tree(np.hstack([scenarios[0], scenarios[1]]), k_max=100)
        #tree = self.ndt.gen_tree(scenarios[1], k_max=100)


if __name__ == '__main__':
    unittest.main()
