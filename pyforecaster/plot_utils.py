import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import seaborn as sb
import pandas as pd
import matplotlib.dates as dates
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import networkx as nx
from pyforecaster.scenred import plot_from_graph
from pyforecaster.forecaster import ScenarioGenerator

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


def jointplot(df, x: str, y: str, grid_steps=4, style='seaborn', fig=None):

    if fig is None:
        fig = plt.figure()
    plt.style.use(style)

    gs = GridSpec(grid_steps, grid_steps)

    ax_scatter = fig.add_subplot(gs[1:grid_steps, 0:(grid_steps-1)])
    ax_hist_x = fig.add_subplot(gs[0, 0:(grid_steps-1)], frameon=False)
    ax_hist_y = fig.add_subplot(gs[1:grid_steps, (grid_steps-1)],frameon=False)
    #plt.subplots_adjust(wspace=0.3, hspace=0.3)

    cm = plt.get_cmap('viridis',10)
    kde_col = cm(4)
    sb.kdeplot(data=df, x=x, y=y, ax=ax_scatter, levels=10, cmap='mako', shade_lowest=False, alpha=0.5)
    sb.scatterplot(data=df, x=x, y=y, ax=ax_scatter, s=10)
    sb.histplot(data=df, x=x, ax=ax_hist_x, kde=True, color=kde_col, bins=30)
    sb.histplot(data=df, y=y, ax=ax_hist_y, kde=True, color=kde_col, bins=30)

    [remove_ticks(a, b) for a, b in zip([ax_hist_x, ax_hist_y], [x, y])]
    ax_hist_y.set_ylabel('')
    ax_hist_x.set_xlabel('')

    return fig


def plot_summary_score(df, width=4.5, height=3, x_label='step ahead [-]', y_label='aggregation [-]',
                       colorbar_label='score',  b=0.15, l=0.2, w=0.22, font_scale=0.8, interval_to_ints=True,
                       numeric_xticks=False, rotation_deg=None, ax=None, label_specs=None, **kwargs):

    if interval_to_ints and np.all([isinstance(i, pd.Interval) for i in df.index]):
        int_intervals = [pd.Interval(i.left.astype(int), i.right.astype(int)) for i in df.index]
        df.index = int_intervals
    if isinstance(label_specs, dict):
        for k, v in label_specs.items():
            idx = df.index if k=='y' else df.columns

            if 'round' in v.keys():
                if v['round'] == 'int':
                    idx = [pd.Interval(int(i.left), int(i.right)) for i in idx]
                else:
                    idx = [pd.Interval(np.round(i.left, v['round']), np.round(i.right,v['round'])) for i in idx]

            if 'unify' in v.keys():
                if v['unify'] == 'left':
                    idx = [i.left for i in idx]
                elif v['unify'] == 'mean':
                    idx = [np.mean(i.left + i.right) for i in idx]


            if k == 'y':
                df.index = idx
            else:
                df.columns = idx

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
        ax.set_xticklabels(np.linspace(0, len(df.columns), 7, dtype=int))

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


def hist_2d(data, value, x, y, plot=True, qs=None, **basic_setup_kwargs):
    if qs is None:
        qs = np.linspace(0.1, 0.9, 11)
    hist = data[value].groupby([pd.cut(data[x], bins=data[x].quantile(qs), duplicates='drop'),
                                pd.cut(data[y], bins=data[y].quantile(qs), duplicates='drop')])\
        .mean().unstack(fill_value=0)
    if plot:
        fig, ax = plot_summary_score(hist, **basic_setup_kwargs)
        return fig, ax
    return hist


def ts_animation(ys:list,  ts=None, names=None, frames=150, interval=1, step=1, repeat=False, target=None):
    "plot the first n_rows of the two y_te and y_hat matrices"
    ts = [np.arange(len(ys[0][0]))]*len(ys) if ts is None else ts
    fig, ax = plt.subplots(1, figsize=(6, 4))
    lines = []
    f_min = np.min([np.min(y) for y in ys])
    f_max = np.max([np.max(y) for y in ys])

    def init():
        for y, t in zip(ys, ts):
            l, = ax.plot(t, y[0, :], alpha=0.8, linewidth=1)
            lines.append(l)
        if target is not None:
            l, = ax.plot(ts[0], target[:len(ts[0])], alpha=0.8, linewidth=1)
            lines.append(l)
        ax.set_ylim(f_min - np.abs(f_min) * 0.1, f_max + np.abs(f_max) * 0.1)
        plt.legend(names, loc='upper left')
        return lines

    def animate(i):
        for y, l, t in zip(ys, lines, ts):
            l.set_data(t, y[i*step, :])
        if target is not None:
            lines[-1].set_data(ts[0], target[i*step:i*step+len(ts[0])])

        return lines


    ani = animation.FuncAnimation(fig, animate, init_func=init,  blit=False, frames=np.minimum(ys[0].shape[0]-1, frames), interval=interval, repeat=repeat)

    return ani



def plot_trees(ys:list, y_gt=None, times=None, frames=150, ax_labels=None, legend_kwargs={},
                   remove_spines=True, savepath=None, **kwargs):
    "plot the first n_rows of the two y_te and y_hat matrices"
    fig, ax = plt.subplots(1, **kwargs)
    f_min = np.min([np.min(np.array(list(nx.get_node_attributes(y, 'v').values()))) for y in ys])
    f_max = np.max([np.max(np.array(list(nx.get_node_attributes(y, 'v').values()))) for y in ys])
    lines = None
    lines = plot_from_graph(ys[0], alpha=0.2, linewidth=1, ax=ax, color='r')
    if ax_labels is not None:
        for k, v in ax_labels.items():
            if k == 'x':
                ax.set_xlabel(v)
            elif k == 'y':
                ax.set_ylabel(v)
    if remove_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    if y_gt is not None:
        t = np.arange(len(y_gt[0][0])) if times is None else times
        lines_ground_truth = []
        for y in y_gt:
            l, = ax.plot(t, y[0, :])
            lines_ground_truth.append(l)
    ax.set_ylim(f_min - np.abs(f_min) * 0.1, f_max + np.abs(f_max) * 0.1)

    def animate(i):
        _ = plot_from_graph(ys[i], lines, alpha=0.2, ax=ax, color='r')
        if y_gt is not None:
            for y, l in zip(y_gt, lines_ground_truth):
                l.set_data(t, y[i, :])
        return
    if savepath is not None:
        writervideo = animation.FFMpegWriter(fps=60)
        animation.FuncAnimation(fig, animate, blit=False, frames=np.minimum(len(ys) - 1, frames), interval=100,
                                repeat=False).save(savepath, writer=writervideo)
    else:
        return animation.FuncAnimation(fig, animate,  blit=False, frames=np.minimum(len(ys)-1, frames), interval=100,
                                       repeat=False)


def ts_animation_bars(ys:list, start_t:list, end_t:list, frames=150, ax_labels=None, legend_kwargs={},
                   remove_spines=True, savepath=None, **kwargs):
    "plot the first n_rows of the two y_te and y_hat matrices"
    fig, ax = plt.subplots(1, **kwargs)
    lines = []
    f_min = np.min([np.min(y) for y in ys])
    f_max = np.max([np.max(y) for y in ys])
    cm = plt.get_cmap('Set1')
    if ax_labels is not None:
        for k, v in  ax_labels.items():
            if k == 'x':
                ax.set_xlabel(v)
            elif k == 'y':
                ax.set_ylabel(v)
    if remove_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    def init():
        for y, s_t, e_t, idx in zip(ys, start_t, end_t, range(len(ys))):
            l = [ax.stairs(y_j.ravel(), np.array([s_t.iloc[j]/np.timedelta64(3600*24,'s'), e_t.iloc[j]/np.timedelta64(3600*24,'s')]), color=cm(idx)) for j, y_j in enumerate(y[0, :])]
            lines.append(l)
        ax.set_ylim(f_min - np.abs(f_min) * 0.1, f_max + np.abs(f_max) * 0.1)

        return [item for sublist in lines for item in sublist]

    def animate(i):
        for y, l, s_t, e_t in zip(ys, lines, start_t, end_t):
            for j, l_j in enumerate(l):
                l_j.set_data(y[i, j].ravel(), np.array([s_t.iloc[j]/np.timedelta64(3600*24,'s'), e_t.iloc[j]/np.timedelta64(3600*24,'s')]))
        ax.legend(**legend_kwargs)
        return [item for sublist in lines for item in sublist]

    if savepath is not None:
        writervideo = animation.FFMpegWriter(fps=60)
        animation.FuncAnimation(fig, animate, init_func=init, blit=False, frames=np.minimum(ys[0].shape[0] - 1, frames),
                                interval=100, repeat=False).save(savepath, writer=writervideo)
    else:
        return animation.FuncAnimation(fig, animate, init_func=init, blit=False,
                                       frames=np.minimum(ys[0].shape[0] - 1, frames), interval=100, repeat=False)

def plot_quantiles(signals, qs, labels, n_rows=50, interval=1, step=1, repeat=False, ax_labels=None, legend_kwargs={},
                   remove_spines=True, savepath=None, **kwargs):
    n_max = np.minimum(signals[0].shape[0], int(n_rows*step))
    n_rows = np.minimum(int(np.floor(signals[0].shape[0]/step)), n_rows)
    qs = ScenarioGenerator().quantiles_to_numpy(qs) if isinstance(qs, pd.DataFrame) else qs
    fig, ax = plt.subplots(1, **kwargs)
    signals = signals if isinstance(signals, list) else [signals]
    t = np.arange(signals[0].shape[1])
    lines= []
    for i, s in enumerate(signals):
        signals[i] = s.values if isinstance(s, pd.DataFrame) else s
        line, = ax.plot(signals[i][0, :], lw=2, label = labels[i])
        lines.append(line)
    lineq = ax.plot(np.squeeze(qs[0, :, :]), 'r', lw=2, alpha=0.3)
    ax.set_ylim(np.min(qs[:n_max]), np.max(qs[:n_max])*1.05)

    if ax_labels is not None:
        for k, v in  ax_labels.items():
            if k == 'x':
                ax.set_xlabel(v)
            elif k == 'y':
                ax.set_ylabel(v)
    if remove_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


    def animate(i):
        i = i * step
        for l, s in zip(lines, signals):
            l.set_data(t, s[i, :])
        [lineq[j].set_data(t, qsi) for j, qsi in enumerate(qs[i, :, :].T)]
        ax.legend(**legend_kwargs)
        return (*lines, *lineq, )

    def init():
        lines[0].set_data([], [])
        plt.legend()
        return (lines[0],)


    if savepath is not None:
        writervideo = animation.FFMpegWriter(fps=60)
        animation.FuncAnimation(fig, animate, init_func=init, frames=n_rows, interval=interval, blit=True,
                                repeat=repeat).save(savepath, writer=writervideo)
    else:
        return animation.FuncAnimation(fig, animate, init_func=init, frames=n_rows, interval=interval, blit=True,
                                  repeat=repeat)


def plot_scenarios_from_multilevel(scens, i=0, ax=None):
    ax = plt.gca() if ax is None else ax
    sb.lineplot(scens.stack().loc[(scens.stack().index[i], slice(None)), :].reset_index(1).melt(id_vars='step'),
        hue='scenario', x='step', y='value', ax=ax)
    return ax
