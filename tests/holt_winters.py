import pandas as pd
import numpy as np
import wget
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
from pyforecaster.utilities import kalman_update, kalman_predict, kalman
from numba import njit
from numba.typed import List
from pyforecaster.forecasting_models.holtwinters import Fourier_es as FES
from pyforecaster.forecasting_models.holtwinters import FK_multi, FK

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

  ani = animation.FuncAnimation(fig, animate, init_func=init, frames=n_rows, interval=1,
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


m = 24*6
y_hat = []
seasonal_state = []
s_init = data['all'].iloc[:m].copy().values
for i in range(1600):
  preds, s = modified_holt_winters(data['all'].iloc[:1+i].copy().values, h=m, alpha=0.01, beta=0.01, gamma=0.01, omega=0.1, m=m, return_s=True, s_init=s_init)
  y_hat.append(preds)
  seasonal_state.append(s)

y_hat = np.vstack(y_hat)
seasonal_state = np.vstack(seasonal_state)


#ts_animation(data['all'].values, y_hat, 1600, labels=['ground truth', 'predictions'])


# Fourier Holt-Winters: in this Holt-Winters forecasters the states are the Fourier coefficients

def get_basis(t, l, n_h):
  """
  Get the first n_h sine and cosine basis functions and the projection
  matrix P
  """
  trigonometric = np.vstack([np.vstack([np.sin(2*np.pi*k*t/l), np.cos(2*np.pi*k*t/l)]) for k in np.arange(n_h)+1])
  P = np.vstack([np.ones(len(t))/np.sqrt(2), trigonometric]).T * np.sqrt(2 / l)
  return P


class Fourier_es:


    def __init__(self, h=1, alpha=0.8, m=24, omega=0.99, n_harmonics=3):
        """
        :param y:
        :param h:
        :param alpha:
        :param m:
        :return:
        """
        assert m>0, 'm must be positive'
        assert 0<alpha<1, 'alpha must be in (0, 1)'
        assert 0<omega<1, 'omega must be in (0, 1)'
        assert n_harmonics>0, 'n_harmonics must be positive'

        # precompute basis over all possible periods
        self.alpha = alpha
        self.omega = omega
        self.h=h
        self.m = m
        Ps_past = []
        Ps_future = []
        self.n_harmonics = np.minimum(n_harmonics, m // 2)
        for i in range(m):
            t = np.arange(m)+i
            t_f = np.arange(m+h)+i
            P = get_basis(t, m, self.n_harmonics)
            Ps_past.append(P)
            P_f = get_basis(t_f, m, self.n_harmonics)
            P_f = P_f[m:, :]
            Ps_future.append(P_f)
        Ps_future_typed = List()
        for arr in Ps_future:
            Ps_future_typed.append(arr)
        Ps_past_typed = List()
        for arr in Ps_past:
            Ps_past_typed.append(arr)

        self.Ps_future = Ps_future_typed
        self.Ps_past = Ps_past_typed
        self.coeffs = None
        self.eps = None
        self.last_1sa_preds = 0
    def predict(self, y, return_coeffs=False, start_from=0):
        if self.coeffs is None:
            coeffs = np.zeros(2 * self.n_harmonics + 1)
            coeffs[0] = y[0]*np.sqrt(m)
            eps = 0
            preds = [[0]]
        else:
            eps = self.eps
            coeffs = self.coeffs
            preds = [[y[start_from]+eps]]

        coeffs_t_history = []
        for i in range(start_from, len(y)):
            P = self.Ps_past[(i)%self.m]
            P_f = self.Ps_future[(i-self.h)%self.m]
            start = np.maximum(0, i-self.m+1)

            w = y[start:i+1].copy()
            #eps = self.omega * (w[-1] - preds[-1][0]) + (1-self.omega) * eps
            eps = self.omega * (w[-1] - self.last_1sa_preds) + (1-self.omega) * eps
            #w[-1] = w[-1] -eps
            coeffs_t = P[-len(w):,:].T@w
            coeffs = self.alpha*coeffs_t + (1-self.alpha)*coeffs
            last_preds = (P_f@coeffs).ravel()
            self.last_1sa_preds = last_preds[0]
            preds.append((last_preds + eps*(self.h-np.arange(self.h))**2/self.h**2).ravel() )
            if return_coeffs:
                coeffs_t_history.append(coeffs)
        self.coeffs = coeffs
        self.eps = eps
        if return_coeffs:
            return np.vstack(preds[1:]), np.vstack(coeffs_t_history)
        return np.vstack(preds[1:])

    def __getstate__(self):
        return (self.coeffs, self.eps,self.last_1sa_preds)
    def __setstate__(self, state):
        self.coeffs, self.eps, self.last_1sa_preds = state

from pyforecaster.forecaster import LinearForecaster
from pyforecaster.formatter import Formatter
fr = Formatter().add_transform(['all'], ['mean'], lags=np.arange(1, 144))
fr.add_target_transform(['all'], ['mean'], lags=-np.arange(1, 145))
x, y = fr.transform(data, time_features=False)

n_tr = 10000
n_te = 10000

lf = LinearForecaster().fit(x.iloc[:n_tr,:], y.iloc[:n_tr,:])

y_hat_lin = lf.predict(x.iloc[n_tr:n_tr+n_te,:])
ts_animation(y.iloc[n_tr:n_tr+n_te,0].values, y_hat_lin.values, 2000, labels=['ground truth', 'predictions'])


m = int(24*6)


y_hat, coeffs_t = Fourier_es(h=24*6, alpha=0.5, omega=0.9, n_harmonics=80, m=m).predict(data['all'].values[:2100], return_coeffs=True)
ts_animation(x.iloc[:, 0].values, y_hat, 2000, labels=['ground truth', 'predictions'])

ts_animation(x.iloc[:, 0].values, coeffs_t[:, :], 2000, labels=['ground truth', 'coeffs'])

fes = FES(n_sa=24*6, alpha=0.8, omega=0.99, n_harmonics=80, m=m, target_name='all').fit(data.iloc[:2100, :])
y_hat = fes.predict(data.iloc[2100:4100, :])

ts_animation(x.iloc[2100:4100, 0].values, y_hat, 2000, labels=['ground truth', 'predictions'])


"""
mae = lambda x, y: np.mean(np.abs(x-y))
mae(y.iloc[n_tr:n_tr+n_te,:].values, y_hat_lin.values)
mae(y.iloc[n_tr:n_tr+n_te,:].values, y_hat[n_tr:n_tr+n_te, :])
"""

# Fourier-Kalman exponential smoothing


from filterpy.kalman import KalmanFilter


class Fourier_kexp(Fourier_es):
    def __init__(self, h=1, alpha=0.8, m=24, omega=0.99, n_harmonics=3):
        super().__init__(h, alpha, m, omega, n_harmonics)
        n_harmonics = np.minimum(n_harmonics, m // 2)

        coeffs = np.zeros(2 * n_harmonics + 1)

        # precompute basis over all possible periods
        my_filter = KalmanFilter(dim_x=n_harmonics * 2 + 1, dim_z=n_harmonics * 2 + 1)
        my_filter.x = coeffs.copy()

        my_filter.F = np.eye(n_harmonics * 2 + 1)

        my_filter.H = np.eye(n_harmonics * 2 + 1)
        my_filter.P = np.eye(n_harmonics * 2 + 1) * 1000
        my_filter.R = np.eye(n_harmonics * 2 + 1) * 0.01
        my_filter.Q = 0.01
        self.filter = my_filter
        self.n_harmonics = n_harmonics
        self.coeffs_t_history = []
    def predict(self, y, return_coeffs=False, start_from=0):
        if self.coeffs is None:
            coeffs = np.zeros(2 * self.n_harmonics + 1)
            coeffs[0] = y[0]*np.sqrt(m)
            eps = 0
            preds = [[0]]
        else:
            eps = self.eps
            coeffs = self.coeffs
            preds = [[y[start_from]+eps]]

        coeffs_t_history = []
        for i in range(start_from, len(y)):
            P = self.Ps_past[(i)%self.m]
            P_f = self.Ps_future[(i-self.h)%self.m]
            start = np.maximum(0, i-self.m+1)

            w = y[start:i+1].copy()
            #eps = self.omega * (w[-1] - preds[-1][0]) + (1-self.omega) * eps
            eps = self.omega * (w[-1] - self.last_1sa_preds) + (1-self.omega) * eps
            #w[-1] = w[-1] -eps
            coeffs_t = P[-len(w):,:].T@w
            self.filter.predict()
            self.filter.update(coeffs_t)
            coeffs = self.filter.x

            last_preds = (P_f@coeffs).ravel()
            self.last_1sa_preds = last_preds[0]
            preds.append((last_preds + eps*(self.h-np.arange(self.h))**2/self.h**2).ravel() )
            self.coeffs_t_history.append(coeffs)
            if i % m == 0 and i >= m:
                self.filter.Q = np.corrcoef(np.vstack(self.coeffs_t_history).T)
                self.filter.R = np.corrcoef(np.vstack(self.coeffs_t_history).T) * 0.01
                self.filter.P = np.eye(self.n_harmonics * 2 + 1) * 1000

        self.coeffs = coeffs
        self.eps = eps
        if return_coeffs:
            return np.vstack(preds[1:]), np.vstack(self.coeffs_t_history)
        return np.vstack(preds[1:])

    def __getstate__(self):
        return (self.coeffs, self.eps,self.last_1sa_preds)
    def __setstate__(self, state):
        self.coeffs, self.eps, self.last_1sa_preds = state



@njit
def update_predictions(coeffs_t_history, start_from, y, Ps_past, Ps_future, h, m, omega, last_1sa_preds, eps, x, P, F, Q, H, R, n_harmonics):
    preds = []
    for i in range(start_from, len(y)):
        P_past = Ps_past[i % m]
        P_f = Ps_future[(i - h) % m]
        start = max(0, i - m + 1)

        w = y[start:i + 1].copy()
        eps = omega * (w[-1] - last_1sa_preds) + (1 - omega) * eps
        coeffs_t = np.dot(P_past[-len(w):, :].T, w)
        x, P = kalman(x, P, F, Q, coeffs_t, H, R)  # Assuming kalman is a Numba-compatible function
        coeffs = x

        last_preds = np.dot(P_f, coeffs).ravel()
        last_1sa_preds = last_preds[0]
        preds.append((last_preds + eps * (h - np.arange(h)) ** 2 / h ** 2).ravel())
        coeffs_t_history[:, i] = coeffs

        if i % m == 0 and i >= m:
            Q = np.corrcoef(coeffs_t_history[:, :i])
            R = Q * 0.01
            P = np.eye(n_harmonics * 2 + 1) * 1000

    return preds, last_1sa_preds, eps, x, P, Q, R, coeffs_t_history

class Fourier_kexp_custom(Fourier_es):
    def __init__(self, h=1, alpha=0.8, m=24, omega=0.99, n_harmonics=3):
        super().__init__(h, alpha, m, omega, n_harmonics)
        n_harmonics = np.minimum(n_harmonics, m // 2)

        coeffs = np.zeros(2 * n_harmonics + 1)

        # precompute basis over all possible periods
        self.x = coeffs.copy()

        self.F = np.eye(n_harmonics * 2 + 1)

        self.H = np.eye(n_harmonics * 2 + 1)
        self.P = np.eye(n_harmonics * 2 + 1) * 1000
        self.R = np.eye(n_harmonics * 2 + 1) * 0.01
        self.Q = np.eye(n_harmonics * 2 + 1) * 0.01
        self.n_harmonics = n_harmonics
        self.coeffs_t_history = []
    def predict(self, y, return_coeffs=False, start_from=0):
        if self.coeffs is None:
            coeffs = np.zeros(2 * self.n_harmonics + 1)
            coeffs[0] = y[0]*np.sqrt(self.m)
            eps = 0
            preds = [[0]]
        else:
            eps = self.eps
            coeffs = self.coeffs
            preds = [[y[start_from]+eps]]

        if len(self.coeffs_t_history)>0:
            coeffs_t_history = np.vstack([self.coeffs_t_history, np.zeros((len(y) - start_from, self.n_harmonics * 2 + 1))])
        else:
            coeffs_t_history = np.zeros((len(y) - start_from, self.n_harmonics * 2 + 1))
        preds_updated, self.last_1sa_preds, self.eps, self.x, self.P, self.Q, self.R, coeffs_t_history = update_predictions(coeffs_t_history.T, start_from, y, self.Ps_past, self.Ps_future, self.h, self.m, self.omega, self.last_1sa_preds, eps, self.x, self.P, self.F, self.Q, self.H, self.R,
                           self.n_harmonics)

        preds = preds + preds_updated
        self.coeffs_t_history = coeffs_t_history.T
        self.coeffs = coeffs_t_history.T[-1, :]
        self.eps = eps
        if return_coeffs:
            return np.vstack(preds[1:]), coeffs_t_history.T
        return np.vstack(preds[1:])

    def __getstate__(self):
        return (self.coeffs, self.eps,self.last_1sa_preds)
    def __setstate__(self, state):
        self.coeffs, self.eps, self.last_1sa_preds = state


def fourier_kexp(y, h=1, alpha=0.8, m=24, omega=0.99, n_harmonics=3, return_coeffs=False):
    """
    :param y:
    :param h:
    :param alpha:
    :param m:
    :return:
    """

    n_harmonics = np.minimum(n_harmonics, m // 2)

    coeffs = np.zeros(2*n_harmonics+1)
    preds = [[0]]
    coeffs_t_history = []
    coeffs[0] = y[0]*np.sqrt(m)

    # precompute basis over all possible periods
    my_filter = KalmanFilter(dim_x=n_harmonics*2+1, dim_z=n_harmonics*2+1)
    my_filter.x = coeffs.copy()

    my_filter.F = np.eye(n_harmonics*2+1)

    my_filter.H = np.eye(n_harmonics*2+1)
    my_filter.P = np.eye(n_harmonics*2+1) * 1000
    my_filter.R = np.eye(n_harmonics*2+1) * 0.01
    my_filter.Q = 0.01

    Ps_past = []
    Ps_future = []
    for i in range(m):
        t = np.arange(m)+i
        t_f = np.arange(m+h)+i
        P = get_basis(t, m, n_harmonics)
        Ps_past.append(P)
        P_f = get_basis(t_f, m, n_harmonics)
        P_f = P_f[m:, :]
        Ps_future.append(P_f)

    eps = 0
    for i in range(1, len(y)):
        P = Ps_past[i%m]
        P_f = Ps_future[i%m]
        start = np.maximum(0, i-m)
        w = y[start:i].copy()
        eps = omega * (w[-1] - preds[-1][0]) + (1-omega) * eps
        w[-1] = w[-1] -eps
        coeffs_t = P[-len(w):,:].T@w

        my_filter.predict()
        my_filter.update(coeffs_t)
        coeffs = my_filter.x

        preds.append((P_f@coeffs).ravel() + eps*(h-np.arange(h))**2/h**2)
        if return_coeffs:
            coeffs_t_history.append(coeffs_t)
        if i%m==0 :
            my_filter.Q = np.corrcoef(np.vstack(coeffs_t_history).T)
            my_filter.R = np.corrcoef(np.vstack(coeffs_t_history).T)*0.01
            my_filter.P = np.eye(n_harmonics*2+1) * 1000


    if return_coeffs:
        return np.vstack(preds[1:]), np.vstack(coeffs_t_history)
    return np.vstack(preds[1:])

m = 336
data_te = x.iloc[:, 0].values[:2100].copy()
data_te[500:] -= 5
data_te[1000:] += 5
"""
y_hat_kexp, coeffs_t = fourier_kexp(data_te, h=24*6, alpha=0.9, omega=0.9, n_harmonics=20, m=m, return_coeffs=True)
ts_animation(data_te, y_hat_kexp, 2000, labels=['ground truth', 'predictions'])
ts_animation(data_te, coeffs_t, 2000, labels=['ground truth', 'predictions'])

y_hat_kexp, coeffs_t = Fourier_kexp(h=24*6, alpha=0.9, omega=0.9, n_harmonics=20, m=m).predict(data_te, return_coeffs=True)
ts_animation(data_te, y_hat_kexp, 2000, labels=['ground truth', 'predictions'])

y_hat_kexp, coeffs_t = Fourier_kexp_custom(h=24*6, alpha=0.9, omega=0.9, n_harmonics=10, m=m).predict(x.iloc[:6100, :]['all'].values, return_coeffs=True)

fks = FKS(n_sa=24*6, alpha=0.9, omega=0.9, n_harmonics=10, m=m, target_name='all').fit(x.iloc[:2100, :], return_coeffs=False)
y_hat_kexp_fast = fks.predict(x.iloc[2100:6100, :])

from pyforecaster.plot_utils import ts_animation
ts_animation([y_hat_kexp[2100:6100, :], y_hat_kexp_fast], names=['kexp custom', 'kexp fast', 'target'], target=x['all'].iloc[2100:6100], frames=100, interval=0.1, step=3)
"""

def fourier_kexp_2(y, h=1, n_predictors=4,  m_max=24, omega=0.9, alpha=0.9, n_harmonics=3, return_coeffs=False):
    """
    :param y:
    :param h:
    :param alpha:
    :param m:
    :return:
    """


    n_harmonics = np.minimum(n_harmonics, m_max // 2)
    ms = np.linspace(1, m_max, n_predictors+1).astype(int)[1:]

    preds = [[0]]
    coeffs_t_history = []


    # precompute basis over all possible periods
    my_filter = KalmanFilter(dim_x=n_predictors, dim_z=n_predictors)
    my_filter.x = np.ones(n_predictors) / n_predictors

    my_filter.F = np.eye(n_predictors)
    my_filter.H = np.eye(n_predictors)
    my_filter.P = np.eye(n_predictors) * 1000
    my_filter.R = np.eye(n_predictors)
    my_filter.Q = 0.1

    models = [Fourier_kexp_custom(h=h, alpha=alpha, omega=omega, n_harmonics=n_harmonics, m=ms[i]) for i in range(n_predictors)]
    states = [models[j].__getstate__() for j in range(n_predictors)]


    preds_models = [[] for i in range(n_predictors)]
    for i in tqdm(np.arange(len(y))):

        [models[j].__setstate__(states[j]) for j in range(n_predictors)]
        preds_t = [models[j].predict(y[:i+1].copy(), return_coeffs=False, start_from=i).ravel() for j in range(n_predictors)]
        for j in range(n_predictors):
            preds_models[j].append(preds_t[j])
        if i>=h:
            for j in range(n_predictors):
                preds_models[j].pop(0)
            # average last point error over different prediction times for all the models
            avg_err = [np.mean([np.abs(p[-(k+1)]-y[i]) for k, p in enumerate(preds_models[j])]) for j in range(n_predictors)]
            coeffs_t = np.exp(-np.array(avg_err)) / np.exp(-np.array(avg_err)).sum()
        else:
            coeffs_t = np.ones(n_predictors) / n_predictors
        states = [models[j].__getstate__() for j in range(n_predictors)]

        my_filter.predict()
        my_filter.update(coeffs_t)
        coeffs = my_filter.x
        coeffs = coeffs / np.abs(coeffs).sum()
        preds.append(np.vstack(preds_t).T@coeffs)

        if return_coeffs:
            coeffs_t_history.append(coeffs_t)
        if i%m==0 and i>=m:
            my_filter.Q = np.corrcoef(np.vstack(coeffs_t_history).T)
            my_filter.R = np.corrcoef(np.vstack(coeffs_t_history).T)*10
            my_filter.P = np.eye(n_predictors) * 10

    if return_coeffs:
        return np.vstack(preds[1:]), np.vstack(coeffs_t_history)
    return np.vstack(preds[1:])



class fourier_kexp_2_custom:
    """
    :param y:
    :param h:
    :param alpha:
    :param m:
    :return:
    """

    def __init__(self, h=1, n_predictors=4,  m_max=24, omega=0.9, alpha=0.9, n_harmonics=3):

        n_harmonics = np.minimum(n_harmonics, m_max // 2)
        ms = np.linspace(1, m_max, n_predictors + 1).astype(int)[1:]
        self.n_predictors = n_predictors
        self.h = h

        # precompute basis over all possible periods
        self.x = np.ones(n_predictors) / n_predictors

        self.F = np.eye(n_predictors)
        self.H = np.eye(n_predictors)
        self.P = np.eye(n_predictors) * 1000
        self.R = np.eye(n_predictors)
        self.Q = 0.1

        self.models = [Fourier_kexp_custom(h=h, alpha=alpha, omega=omega, n_harmonics=n_harmonics, m=ms[i]) for i in
                  range(n_predictors)]

    def predict(self,y, return_coeffs=False, start_from=0):
        preds = [[0]]
        coeffs_t_history = []

        self.states = [self.models[j].__getstate__() for j in range(self.n_predictors)]

        preds_models = List()
        for _ in range(self.n_predictors):
            preds_models.append(List([np.array([], dtype=np.float64)]))

        preds_models = [[] for i in range(self.n_predictors)]
        for i in tqdm(np.arange(len(y))):

            [self.models[j].__setstate__(self.states[j]) for j in range(self.n_predictors)]
            preds_t = [self.models[j].predict(y[:i+1].copy(), return_coeffs=False, start_from=i).ravel() for j in range(self.n_predictors)]
            for j in range(self.n_predictors):
                preds_models[j].append(preds_t[j])
            if i >= self.h:
                for j in range(self.n_predictors):
                    preds_models[j].pop(0)
                # average last point error over different prediction times for all the models
                avg_err = [np.mean([np.abs(p[-(k + 1)] - y[i]) for k, p in enumerate(preds_models[j])]) for j in
                           range(self.n_predictors)]
                coeffs_t = np.exp(-np.array(avg_err)) / np.exp(-np.array(avg_err)).sum()
            else:
                coeffs_t = np.ones(self.n_predictors) / self.n_predictors
            states = [self.models[j].__getstate__() for j in range(self.n_predictors)]

            self.states = [self.models[j].__getstate__() for j in range(self.n_predictors)]

            self.x, self.P = kalman(self.x, self.P, self.F, self.Q, coeffs_t, self.H, self.R)

            coeffs = self.x
            coeffs = coeffs / np.abs(coeffs).sum()
            preds.append(np.vstack(preds_t).T@coeffs)

            if return_coeffs:
                coeffs_t_history.append(coeffs_t)
            if i%m==0 and i>=m:
                self.Q = np.corrcoef(np.vstack(coeffs_t_history).T)
                self.R = np.corrcoef(np.vstack(coeffs_t_history).T)*10
                self.P = np.eye(self.n_predictors) * 10

        if return_coeffs:
            return np.vstack(preds[1:]), np.vstack(coeffs_t_history)
        return np.vstack(preds[1:])

data.iloc[11500:12500]-=5
x, y = fr.transform(data, time_features=False)

n_tr = 10000
n_te = 2800

m = 24*6*7
n_sa =24*9
periodicity = 24*6
fks = FK(n_sa=n_sa, alpha=0.8, omega=0.99, n_harmonics=10, m=336, target_name='all', n_predictors=3, periodicity=periodicity).fit(x.iloc[:2000, :], return_coeffs=True)
y_hat_kexp_multi_fast = fks.predict(x.iloc[2100:6100,:])
from pyforecaster.plot_utils import ts_animation
ts_animation([y_hat_kexp_multi_fast], names=['kexp fast', 'target'], target=x['all'].iloc[2100:6100], frames=300, step=2)


fks = FK_multi(n_sa=n_sa, alpha=0.8, omega=0.99, n_harmonics=10, m=m, target_name='all', n_predictors=3, periodicity=periodicity).fit(x.iloc[:2100, :], return_coeffs=True)
y_hat_kexp_multi_fast = fks.predict(x.iloc[2100:6100,:])

kexp2c = fourier_kexp_2_custom(h=n_sa, omega=0.99, alpha=0.8, n_harmonics=10, m_max=m, n_predictors=3)
y_hat_kexp_multi, coeffs_t = kexp2c.predict(x.iloc[:6100,0].values, return_coeffs=True)

ts_animation([y_hat_kexp_multi[2100:6100, :], y_hat_kexp_multi_fast], names=['kexp custom', 'kexp fast', 'target'], target=x['all'].iloc[2100:6100], frames=100, interval=0.1, step=2)

