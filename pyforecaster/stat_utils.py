from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from scipy.stats import norm

from pyforecaster.plot_utils import basic_setup


def bootstrap(series, stat_fun, save_plots_dir=None, n_sampling=4000):
    """
    :param df: a one column df or time series containing a random variable
    :param stat_fun: string or function, used to retrieve the statistic on each pool/sample
    :param save_plots_dir: if passed, save diagnostic plots in this directory
    :return:
    """
    assert isinstance(series, pd.Series), 'series must be a pd.Series instance, you passed {}'.format(type(series))

    n = len(series)
    samples = np.random.choice(series.index, (n_sampling, n), replace=True)

    sample_at = np.arange(100, n_sampling, 500).astype(int)
    history = pd.DataFrame({})
    summary = pd.DataFrame({})

    if save_plots_dir:
        fig, ax = basic_setup((1,1), 4, 3, b=0.2, l=0.2, style='seaborn-paper')
        colors = plt.get_cmap('plasma_r', len(sample_at))

    for i, n_s in enumerate(sample_at):
        stats = []
        for s in samples[:n_s]:
            stats.append(series.loc[s].agg(stat_fun))

        if save_plots_dir:
            qqplot((stats-np.mean(stats))/np.std(stats), ax, label=n_s, c=colors(i))

        stats_s = pd.DataFrame(np.vstack([np.array(stats), n_s * np.ones(len(stats), dtype=int)]).T, columns=['stats', 'bootsrtap samples'])
        summary = pd.concat([summary, pd.DataFrame(np.array([np.mean(stats), np.std(stats)]).reshape(1,-1), columns=['mean', 'std'], index=[n_s])])
        history = pd.concat([history, stats_s], axis=0)

    history['bootsrtap samples'] = history['bootsrtap samples'].astype(int)

    if save_plots_dir:
        ax.legend()
        plt.savefig(join(save_plots_dir, 'bootstrap_rectified_qq_plot.pdf'))

        pal = "Set2"
        fig, ax = basic_setup((1,1), 4, 3, b=0.2, l=0.2, style='seaborn-paper')
        #pt.half_violinplot(data=history, x='bootsrtap samples', y='stats', palette=pal, bw=.2)
        sb.stripplot(data=history, x='bootsrtap samples', y='stats', s=2, palette=pal)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        xlimm = np.abs(np.max(ax.get_xlim()))
        ax.set_xlim(ax.get_xlim()[0]-xlimm*0.05, ax.get_xlim()[1]+xlimm*0.05)
        plt.savefig(join(save_plots_dir, 'bootstrap_stripplot.pdf'))

    return summary.iloc[-1, :]


def qqplot(x, ax=None, label=None, alpha=0.5, c='grey'):
    alpha_levels = np.linspace(0.01, 0.99, 30)
    x_q = np.quantile(x, alpha_levels)
    norm_q = norm.ppf(alpha_levels)
    if ax is None:
        plt.plot(norm_q, x_q-norm_q, label=label, alpha=alpha, c=c)
    else:
        ax.plot(norm_q, x_q-norm_q, label=label, alpha=alpha, c=c)