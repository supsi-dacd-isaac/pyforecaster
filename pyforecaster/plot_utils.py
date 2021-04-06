import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import pandas as pd
import matplotlib.dates as dates
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


def ts_animation(ys:list, ts:list, names:list, frames=150):
    "plot the first n_rows of the two y_te and y_hat matrices"
    fig, ax = plt.subplots(1)
    lines = []
    f_min = np.min([np.min(y) for y in ys])
    f_max = np.max([np.max(y) for y in ys])

    def init():
        for y, t in zip(ys, ts):
            l, = ax.plot(t.total_seconds()/3600/24, y[0, :], linestyle='none', marker='.')
            lines.append(l)
        ax.set_ylim(f_min - np.abs(f_min) * 0.1, f_max + np.abs(f_max) * 0.1)
        # ax.legend(names)
        return lines

    def animate(i):
        for y, l, t in zip(ys, lines, ts):
            l.set_data(t.total_seconds()/3600/24, y[i, :])
        return lines

    plt.pause(1e-5)
    ani = animation.FuncAnimation(fig, animate, init_func=init,  blit=False, frames=np.minimum(ys[0].shape[0]-1, frames), interval=100, repeat=False)

    return ani

