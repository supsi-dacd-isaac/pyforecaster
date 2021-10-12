import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import seaborn as sb
import pandas as pd
import matplotlib.dates as dates
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import matplotlib.colors as colors


def basic_setup(subplots_tuple, width, height, b=0.15, l=0.15, w=0.22, r=None, style ='seaborn', **kwargs):
    plt.style.use(style)
    fig, ax = plt.subplots(subplots_tuple[0], subplots_tuple[1], figsize=(width, height), **kwargs)
    plt.subplots_adjust(bottom=b, left=l, wspace=w, right=r)
    return fig, ax


def remove_ticks(axes, coord='y', target='labels'):
    """
    :param axes: ax or list of axes
    :param coord:
    :param target:
    :return:
    """
    axes = axes.ravel() if "__len__" in dir(axes) else [axes]
    for ax in axes:
        if target=='ticks':
            if coord == 'y':
                ax.set_yticks([])
            elif coord == 'x':
                ax.set_xticks([])
            elif coord == 'both':
                ax.set_xticks([])
                ax.set_yticks([])
        elif target=='labels':
            if coord == 'y':
                ax.set_yticklabels([])
            elif coord == 'x':
                ax.set_xticklabels([])
            elif coord == 'both':
                ax.set_xticklabels([])
                ax.set_yticklabels([])


def plot_summary_score(df, width=4.5, height=3, x_label='step ahead [-]', y_label='aggregation [-]',
                       colorbar_label='score',  b=0.15, l=0.2, w=0.22, font_scale=0.8, interval_to_ints=True,
                       numeric_xticks=False, rotation_deg=None, ax=None, **kwargs):

    if interval_to_ints and np.all([isinstance(i, pd.Interval) for i in df.index]):
        int_intervals = [pd.Interval(i.left.astype(int), i.right.astype(int)) for i in df.index]
        df.index = int_intervals

    sb.set(font_scale=font_scale)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
    plt.subplots_adjust(bottom=b, left=l, wspace=w)
    sb.heatmap(data=df, ax=ax, cbar_kws={'label': colorbar_label}, **kwargs)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if rotation_deg:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=rotation_deg, va='top')

    # take only 7 xticks
    step = int(len(df.columns) / 7)
    ax.set_xticks(ax.get_xticks()[::step])
    if numeric_xticks:
        ax.set_xticks(np.linspace(0, len(df.columns), 7))
        ax.set_xticklabels(np.linspace(0, len(df.columns), 7,dtype=int))

    fig = plt.gcf()
    return fig, ax


def plot_multiple_summary_scores(dfs, width, height, logscale=False, **kwargs):
    if logscale:
        dfs = {k:df-1 for k, df in dfs.items()}

    v_min = np.min([f.min().min() for f in dfs.values()])
    v_max = np.max([f.max().max() for f in dfs.values()])
    #fig, ax = plt.subplots(1, len(dfs), figsize=(width, height))
    fig, ax = basic_setup((1, len(dfs)), width, height, w=0.05, l=0, b=0.2, r=0.85)
    for i, df in enumerate(dfs.values()):
        if i == len(dfs)-1:
            position = ax[i].get_position().get_points()
            cb_ax = fig.add_axes([position[1][0] + 0.02, position[0][1], 0.02, position[1][1] - position[0][1]])
            kwargs.update(cbar_ax=cb_ax, cbar=True)
        else:
            kwargs.update(cbar=False)
        if logscale:
            plot_summary_score(df, norm=colors.SymLogNorm(vmin=v_min, vmax=v_max, linthresh=0.01), ax=ax[i],
                               cmap='RdBu_r', **kwargs)
        else:
            plot_summary_score(df, vmin=v_min, vmax=v_max, ax=ax[i], **kwargs)
        ax[i].set_title(list(dfs.keys())[i])

    remove_ticks(ax[1:], coord='y', target='labels')
    [a.set_ylabel('') for a in ax[1:]]
    return fig, ax



def ts_animation(ys:list, ts:list, names:list, frames=150):
    "plot the first n_rows of the two y_te and y_hat matrices"
    fig, ax = plt.subplots(1)
    lines = []
    f_min = np.min([np.min(y) for y in ys])
    f_max = np.max([np.max(y) for y in ys])

    def init():
        for y, t in zip(ys, ts):
            l, = ax.plot(t/np.timedelta64(3600*24,'s'), y[0, :], linestyle='none', marker='.')
            lines.append(l)
        for y, t in zip(ys, ts):
            l, = ax.plot(t/np.timedelta64(3600*24,'s'), y[0, :], alpha=0.2, linewidth=1)
            lines.append(l)
        ax.set_ylim(f_min - np.abs(f_min) * 0.1, f_max + np.abs(f_max) * 0.1)

        return lines

    def animate(i):
        for y, l, t in zip(ys, lines, ts):
            l.set_data(t/np.timedelta64(3600*24,'s'), y[i, :])
        for y, l, t in zip(ys, lines[len(ys):], ts):
            l.set_data(t/np.timedelta64(3600*24,'s'), y[i, :])
        return lines

    plt.pause(1e-5)
    ani = animation.FuncAnimation(fig, animate, init_func=init,  blit=False, frames=np.minimum(ys[0].shape[0]-1, frames), interval=100, repeat=False)

    return ani

