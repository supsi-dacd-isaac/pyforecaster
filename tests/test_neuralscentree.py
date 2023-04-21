import unittest
import pandas as pd
from pyforecaster.tree_builders import NeuralGas, DiffTree, ScenredTree, QuantileTree
from pyforecaster.forecaster import ScenGen, LinearForecaster
from pyforecaster.scenred import plot_from_graph
import numpy as np
from scipy.stats import multivariate_normal, norm, weibull_min
import matplotlib.pyplot as plt

class TestScenarios(unittest.TestCase):
    def setUp(self) -> None:
        self.ng = NeuralGas(savepath='figs/neural_gas', init='quantiles', base_tree='quantiles')
        self.ndt = DiffTree(savepath='figs/diff_tree', init='quantiles', base_tree='quantiles')
        self.srt = ScenredTree()
        self.qt = QuantileTree()
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

    def test_tree_gens(self):
        sg = ScenGen(cov_est_method='shrunk', online_tree_reduction=True, q_vect=self.q_vect).fit(self.target)
        scenarios = np.squeeze(sg.predict_scenarios(self.quantiles_df.iloc[[0, 15]], 100))
        #tree_ng, _, _, _ = self.ng.gen_tree(np.hstack([scenarios[0], scenarios[1]]), k_max=10, do_plot=False)
        #tree_d, _, _, _  = self.ndt.gen_tree(np.hstack([scenarios[0], scenarios[1]]), k_max=10, do_plot=False)
        tree_sr, _, _, _ = self.srt.gen_tree(np.hstack([scenarios[0], scenarios[1]]), k_max=10)
        tree_q, _, _, _ = self.qt.gen_tree(np.hstack([scenarios[0], scenarios[1]]), k_max=10)

        #plot_from_graph(tree_ng)
        #plot_from_graph(tree_d, ax=plt.gca(), color='r')
        #plot_from_graph(tree_sr, ax=plt.gca(), linestyle='--')
        #plot_from_graph(tree_q)

    def test_lin_forecaster_offline_difftree(self):
        lf = LinearForecaster(online_tree_reduction=False, tree_type='QuantileTree').fit(self.x, self.target)

if __name__ == '__main__':
    unittest.main()
