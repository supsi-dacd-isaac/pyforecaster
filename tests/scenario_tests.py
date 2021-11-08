import unittest
import pandas as pd
import numpy as np
from pyforecaster.scenarios import ScenGen
import logging
from scipy.stats import multivariate_normal, norm, weibull_min
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from pyforecaster.plot_utils import jointplot


class TestScenarios(unittest.TestCase):
    def setUp(self) -> None:
        n_days = 10
        self.t = 24 * n_days
        self.lags = 24
        t_index = pd.date_range('01-01-2020', '01-10-2020', self.t)
        self.x = pd.DataFrame(np.sin(np.arange(self.t)*2*n_days*np.pi/self.t).reshape(-1,1), index=t_index) \
                 + 0.1 * np.random.randn(self.t, 1)
        self.target = pd.concat([self.x.shift(l) for l in np.arange(self.lags)], axis=1)
        self.target = self.target.loc[~self.target.isna().any(axis=1)]
        self.target.columns = ['target_{}'.format(i) for i in range(self.lags)]
        self.q_vect = np.linspace(0.1, 0.9, 9).round(2)

        # create normal predictions with perfect mean knowledge
        self.quantiles_df = pd.concat({q: self.target + norm.ppf(q) * np.linspace(0.3, 1, self.lags)
                                       for q in self.q_vect}, axis=1)
        self.quantiles_np = np.dstack([self.quantiles_df[q].values for q in self.q_vect])

        self.logger = logging.getLogger()

    def test_copulas(self):
        sg = ScenGen(cov_est_method='vanilla', q_vect=self.q_vect).fit(self.target)
        sg_gl = ScenGen(cov_est_method='glasso_cv', q_vect=self.q_vect).fit(self.target)
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

        fig0, _ = plt.subplots(1, 1, figsize=(10,5))
        fig1, fig2 = fig0.subfigures(1, 2)

        jointplot(pd.DataFrame(samples, columns=['x', 'y']), 'x', 'y', fig=fig1)

        # inverse cdfs; this must be turned into inverse quantiles
        join_rv = np.vstack([rv1.ppf(q_samples[:, 0]), rv2.ppf(q_samples[:, 1])])
        jointplot(pd.DataFrame(join_rv.T, columns=['x', 'y']), 'x', 'y', fig=fig2)
        assert 1 == 1

    def test_scen_gen_df(self):
        sg = ScenGen(q_vect=self.q_vect, cov_est_method='shrunk').fit(self.target)
        rand_idx = np.random.choice(len(self.quantiles_df), 5)
        scenarios = sg.predict(self.quantiles_df.iloc[rand_idx], 20)


        cm = plt.get_cmap('viridis', 4)
        for i, randix in enumerate(rand_idx):
            plt.figure()
            for q in range(int(len(self.q_vect)/2)):
                plt.fill_between(range(self.lags), self.quantiles_df[self.q_vect[q]].iloc[randix,:], self.quantiles_df[self.q_vect[-q-1]].iloc[randix,:],color=cm(q), alpha=0.2)
            plt.plot(np.squeeze(scenarios[i, :, :]), alpha=0.3)
            plt.plot(self.target.iloc[randix, :].values, linestyle='--')
        assert 1 == 1


if __name__ == '__main__':
    unittest.main()

