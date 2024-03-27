import pandas as pd
import numpy as np
import logging
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV, ShrunkCovariance
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt
from pyforecaster.utilities import get_logger


class GaussianCopula:
    def __init__(self, cov_est_method='vanilla', logger=None):
        self.pars = None
        self.cov_est_method = cov_est_method
        self.logger = logger if logger is not None else get_logger(name='Copula', level=logging.WARNING)
        self.dim = None

    def fit(self, y, x=None, do_plot=True):
        self.dim = y.shape[1]
        return self

    def sample(self, x: pd.DataFrame, n: int):
        """
        Return samples from a multivariate random variable whose properties depends on x
        :param x: pd.DataFrame of conditional variables
        :param n: number of samples to draw
        :return: samples of the multivariate copula, dimension n_obs, n_steps, n_scen
        """
        pass


class HourlyGaussianCopula(GaussianCopula):
    def __init__(self, cov_est_method='vanilla', logger=None):
        super().__init__(cov_est_method, logger)

    def fit(self, y, x=None,  do_plot=False):
        self.dim = y.shape[1]
        est_covs = {}
        alpha_glasso = 0.01
        self.logger.info('estimating gaussian copula, conditional on day-hour')
        if self.cov_est_method == 'glasso_cv':
            # optimize alpha in CV on the hour with most observations only
            hour_filt = y.index.hour == y.index.hour.value_counts().argmax()
            x_h = y.loc[hour_filt, :]
            glcv = GraphicalLassoCV().fit(x_h)
            alpha_glasso = glcv.alpha_

        for h in np.unique(y.index.hour):
            hour_filt = y.index.hour == h
            x_h = y.loc[hour_filt, :]
            if self.cov_est_method == 'vanilla':
                # force positive semidefinite matrix
                est_cov = np.corrcoef(x_h.T) + np.eye(self.dim)*1e-6
            elif self.cov_est_method == 'shrunk':
                sh = ShrunkCovariance().fit(x_h)
                std = np.std(x_h.values, axis=0)
                est_cov = sh.covariance_ / np.outer(std, std)
            elif self.cov_est_method in ['glasso', 'glasso_cv']:
                self.logger.info('alpha glasso: {}'.format(alpha_glasso))
                gl = GraphicalLasso(alpha=alpha_glasso).fit(x_h)
                std = np.std(x_h.values, axis=0)
                est_cov = gl.covariance_ / np.outer(std, std)
            else:
                raise ValueError('cov_est_method not recognized')
            est_covs[h] = est_cov

        if do_plot:
            n = np.ceil(len(est_covs) ** 0.5).astype(int)
            fig, ax = plt.subplots(n, n)
            [a.imshow(cov) for a, cov in zip(ax.ravel(), est_covs.values())]
        self.pars = est_covs
        return self

    def sample(self, x: pd.DataFrame, n_scen: int, random_state=None):
        """
        Return samples from a multivariate random variable whose properties depends on x
        :param x: pd.DataFrame of conditional variables
        :param n_scen: number of samples (scenarios) to draw
        :return: samples of the multivariate copula, dimension n_obs, n_steps, n_scen
        """
        index = x.index

        copula_samples = np.zeros((len(index), self.dim, n_scen))
        # retrieve covariances:
        for h in np.unique(index.hour):
            cov_h = self.pars[h]
            mu = np.zeros(self.dim)
            mvr = multivariate_normal(mu, cov_h)
            hour_filt = index.hour == h
            if sum(hour_filt) == 0:
                continue
            else:
                mvr_samples = mvr.rvs((np.sum(hour_filt), n_scen), random_state=random_state)
                # if sum(hour_filt) == 1, mvr_sample has lower dimensionality
                if np.sum(hour_filt) == 1:
                    mvr_samples = np.expand_dims(mvr_samples, 0)
                # if we're predicting only one target, mvr.rvs has lower dimensionality
                if len(mvr_samples.shape) == 2:
                    mvr_samples = np.expand_dims(mvr_samples, -1)
                mvr_samples = np.swapaxes(mvr_samples, 1, 2)
                copula_samples[hour_filt, :, :] = norm.cdf(mvr_samples)
        return copula_samples


class ConditionalGaussianCopula(GaussianCopula):
    def __init__(self, cov_est_method='vanilla', conditional_on='hour', logger=None):
        self.conditional_on = conditional_on
        super().__init__(cov_est_method, logger)

    def fit(self, y, x=None, do_plot=True):
        self.dim = y.shape[1]
        return self

    def sample(self, x: pd.DataFrame, n: int):
        """
        Return samples from a multivariate random variable whose properties depends on x
        :param x: pd.DataFrame of conditional variables
        :param n: number of samples to draw
        :return: samples of the multivariate copula, dimension n_obs, n_steps, n_scen
        """
        pass
