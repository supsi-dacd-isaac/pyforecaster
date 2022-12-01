import unittest
import pandas as pd
import numpy as np
from pyforecaster.forecaster import ScenGen, LinearForecaster
import logging
from scipy.stats import multivariate_normal, norm, weibull_min
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from pyforecaster.plot_utils import jointplot
from pyforecaster.scenred import scenred, plot_graph, plot_from_graph


class TestScenarios(unittest.TestCase):
    def setUp(self) -> None:
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

        # create normal predictions with perfect mean knowledge
        self.quantiles_df = pd.concat({q: self.target + norm.ppf(q) * np.linspace(0.3, 1, self.lags)
                                       for q in self.q_vect}, axis=1)
        self.quantiles_np = np.dstack([self.quantiles_df[q].values for q in self.q_vect])

        self.logger = logging.getLogger()

    def test_copulas(self):
        sg = ScenGen(cov_est_method='vanilla').fit(self.target)
        sg_gl = ScenGen(cov_est_method='glasso_cv').fit(self.target)
        assert 1 == 1

    def test_proof_of_concept(self):
        # sample from multivariate normal with known covariance
        mu = [0, 0]
        cov = np.array([[1, 0.9], [0.9, 1]])
        mvr = multivariate_normal(mu, cov)
        samples = mvr.rvs(1000)           # random var
        q_samples = norm.cdf(samples)    # cdfs

        # disjoint pdfs
        rv1 = weibull_min(1.79)
        rv2 = weibull_min(4)
        fig, ax = plt.subplots(1, 1)
        x = np.linspace(0, 4, 100)
        ax.plot(x, rv1.pdf(x), 'k-', lw=2, label='frozen pdf')
        ax.plot(x, rv2.pdf(x), 'k-', lw=2, label='frozen pdf')

        #fig0, _ = plt.subplots(1, 1, figsize=(10,5))
        #fig1, fig2 = fig0.subfigures(1, 2)

        #jointplot(pd.DataFrame(samples, columns=['x', 'y']), 'x', 'y', fig=fig1)

        # inverse cdfs; this must be turned into inverse quantiles
        #join_rv = np.vstack([rv1.ppf(q_samples[:, 0]), rv2.ppf(q_samples[:, 1])])
        #jointplot(pd.DataFrame(join_rv.T, columns=['x', 'y']), 'x', 'y', fig=fig2)
        assert 1 == 1

    def test_scen_gen_df(self):
        sg = ScenGen(cov_est_method='shrunk').fit(self.target)
        rand_idx = np.random.choice(len(self.quantiles_df), 5)
        scenarios = sg.predict(self.quantiles_df.iloc[rand_idx], 20, kind='scenarios', q_vect=self.q_vect)

        cm = plt.get_cmap('viridis', 4)
        for i, randix in enumerate(rand_idx):
            plt.figure()
            for q in range(int(len(self.q_vect)/2)):
                plt.fill_between(range(self.lags), self.quantiles_df[self.q_vect[q]].iloc[randix,:], self.quantiles_df[self.q_vect[-q-1]].iloc[randix,:],color=cm(q), alpha=0.2)
            plt.plot(np.squeeze(scenarios[i, :, :]), alpha=0.3)
            plt.plot(self.target.iloc[randix, :].values, linestyle='--')
        assert 1 == 1

    def test_scenred(self):
        sg = ScenGen(cov_est_method='shrunk').fit(self.target)
        rand_idx = np.random.choice(len(self.quantiles_df), 2)
        scenarios = sg.predict(self.quantiles_df.iloc[rand_idx], 100, kind='scenarios', q_vect=self.q_vect)
        scenarios_per_step = np.linspace(1, 40, len(scenarios[0]), dtype=int)

        for i, rand_i in enumerate(rand_idx):
            [S_init0, P_sn, J_sn, Me_sn, gn] = scenred(scenarios[i], nodes=scenarios_per_step)
            plot_graph(gn)
            plot_from_graph(gn)
            cm = plt.get_cmap('viridis', 4)
            for q in range(int(len(self.q_vect) / 2)):
                plt.fill_between(range(self.lags), self.quantiles_df[self.q_vect[q]].iloc[rand_i, :],
                                 self.quantiles_df[self.q_vect[-q - 1]].iloc[rand_i, :], color=cm(q), alpha=0.2)

            plt.plot(self.target.iloc[rand_i, :].values, linestyle='--')

        trees = sg.predict(self.quantiles_df.iloc[rand_idx], 100, kind='tree', q_vect=self.q_vect)

        assert 1 == 1

    def test_forecaster(self):
        lf = LinearForecaster().fit(self.x, self.target)
        preds = lf.predict(self.x)
        rand_idx = np.random.choice(np.arange(len(self.x)), 2)
        q_preds = lf.predict_quantiles(self.x.iloc[rand_idx,:])
        scenarios = lf.predict_scenarios(self.x.iloc[rand_idx,:])
        trees = lf.predict_trees(self.x.iloc[:2,:], scenarios_per_step=np.linspace(1,20,scenarios.shape[1], dtype=int))

        for i, rand_i in enumerate(rand_idx):
            plot_graph(trees[i])
            plot_from_graph(trees[i])
            cm = plt.get_cmap('viridis', 4)
            for q in range(int(len(self.q_vect) / 2)):
                plt.fill_between(range(self.lags), q_preds[i, :, q],
                                 q_preds[i, :, -q - 1], color=cm(q), alpha=0.2)

            plt.plot(self.target.iloc[rand_i, :].values, linestyle='--')
            plt.plot(preds[rand_idx[i], :])
        assert 1==1


if __name__ == '__main__':
    unittest.main()

