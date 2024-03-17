import pandas as pd
import numpy as np
import wget
import matplotlib.pyplot as plt
from matplotlib import animation

data = pd.read_pickle('tests/data/test_data.zip')['y'][['all']]
data /= data.std()

def ts_animation(y_te, y_hat, n_rows=50, labels=None):
  "plot the first n_rows of the two y_te and y_hat matrices"
  if labels is None:
    labels = ['1', '2']
  fig, ax = plt.subplots(1);
  y_min = np.minimum(np.min(y_hat[:2*n_rows]), np.min(y_te[:2*n_rows]))
  y_max = np.maximum(np.max(y_hat[:2*n_rows]), np.max(y_te[:2*n_rows]))
  line1, = ax.plot(y_hat[0], lw=2, label=labels[0])
  line2, = ax.plot(y_hat[0], lw=2, label=labels[1])
  plt.legend()
  ax.set_ylim(y_min, y_max)
  n_sa = y_hat.shape[1]

  def animate(i):
    line1.set_data(np.arange(n_sa),y_te[i:i+n_sa])
    line2.set_data(np.arange(n_sa),y_hat[i,:])
    return (line1,line2)

  def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return (line1,line2,)

  ani = animation.FuncAnimation(fig, animate, init_func=init, frames=n_rows, interval=10,
                                blit=True)
  return ani


def modified_holt_winters(y, s_init=None, h=1, alpha=0.8, beta=0.1, gamma=0.1, omega=0.9, m=24, return_s=False):
  """
  h: steps ahead to be predicted
  m: period of the seasonality
  """
  l, l_past = 0, 0
  s = s_init if s_init is not None else np.zeros(m)
  b = 0
  eps = 0
  for i, y_i in enumerate(y):
    index = i%m
    eps = omega * (y_i-l_past-b-s[index]) + (1-omega) * eps
    s[index] = gamma*(y_i-l_past-b-eps) + (1-gamma)*s[index]
    l = alpha*(y_i-s[index]-eps) + (1-alpha)*(l_past+b)
    b = beta*(l-l_past) + (1-beta)*b
    l_past = l

  preds = l + b*np.arange(h) + np.hstack([s[index:], s[:index]])[:h] + eps*(h-np.arange(h))**2/h**2
  if return_s:
    return preds, s
  else:
    return preds


m = 24
y_hat = []
seasonal_state = []
s_init = data['all'].iloc[:m].copy().values
for i in range(600):
  preds, s = modified_holt_winters(data['all'].iloc[:1+i].copy().values, h=24, alpha=0.5, beta=0.5, gamma=0.5, m=24, return_s=True, s_init=s_init)
  y_hat.append(preds)
  seasonal_state.append(s)

y_hat = np.vstack(y_hat)
seasonal_state = np.vstack(seasonal_state)


ts_animation(data['all'].values, y_hat, 300, labels=['ground truth', 'predictions'])


# Fourier Holt-Winters: in this Holt-Winters forecasters the states are the Fourier coefficients

def get_basis(t, l, n_h):
  """
  Get the first n_h sine and cosine basis functions and the projection
  matrix P
  """
  trigonometric = np.vstack([np.vstack([np.sin(2*np.pi*k*t/l), np.cos(2*np.pi*k*t/l)]) for k in np.arange(n_h)+1])
  P = np.vstack([np.ones(len(t))/np.sqrt(2), trigonometric]).T * np.sqrt(2 / l)
  return P


def fourier_hw(y, h=1, alpha=0.8, m=24, omega=0.99, n_harmonics=3, return_coeffs=False):
    """
    :param y:
    :param h:
    :param alpha:
    :param m:
    :return:
    """
    # precompute basis over all possible periods
    Ps_past = []
    Ps_future = []
    for i in range(m):
        t = np.arange(m)+i
        t_f = np.arange(m+h)+i
        n_harmonics = np.minimum(n_harmonics, m // 2)
        P = get_basis(t, m, n_harmonics)
        Ps_past.append(P)
        P_f = get_basis(t_f, m, n_harmonics)
        P_f = P_f[m:, :]
        Ps_future.append(P_f)
    coeffs = np.zeros(2*n_harmonics+1)
    preds = [[0]]
    coeffs_t_history = []
    coeffs[0] = y[0]*np.sqrt(m)
    eps = 0
    for i in range(1, len(y)-1):
        P = Ps_past[i%m]
        P_f = Ps_future[i%m]
        start = np.maximum(0, i-m)
        w = y[start:i].copy()
        eps = omega * (w[-1] - preds[-1][0]) + (1-omega) * eps
        w[-1] = w[-1] -eps
        coeffs_t = P[-len(w):,:].T@w
        coeffs = alpha*coeffs_t + (1-alpha)*coeffs

        preds.append((P_f@coeffs).ravel() + eps*(h-np.arange(h))**2/h**2)
        if return_coeffs:
            coeffs_t_history.append(coeffs)
    if return_coeffs:
        return np.vstack(preds[1:]), np.vstack(coeffs_t_history)
    return np.vstack(preds[1:])


m = 24*6*3
y_hat, coeffs_t = fourier_hw(data['all'].iloc[:2100].copy().values, h=24*6, alpha=0.9, omega=0.5, n_harmonics=20, m=m, return_coeffs=True)

ts_animation(data['all'].values, y_hat, 2000, labels=['ground truth', 'predictions'])
ts_animation(data['all'].values, coeffs_t[:, :], 2000, labels=['ground truth', 'coeffs'])