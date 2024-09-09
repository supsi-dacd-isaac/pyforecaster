import numpy as np
import pandas as pd
from pyforecaster.forecaster import ScenarioGenerator
from pyforecaster.utilities import kalman
from pyforecaster.forecasting_models.holtwinters import tune_hyperpars, hankel
from copy import deepcopy
from abc import abstractmethod

def get_basis(t, l, n_h, frequencies=None):
  """
  Get the first n_h sine and cosine basis functions and the projection
  matrix P
  """
  frequencies = np.arange(n_h)+1 if frequencies is None else frequencies
  trigonometric = np.vstack([np.vstack([np.sin(2*np.pi*k*t/l), np.cos(2*np.pi*k*t/l)]) for k in frequencies])
  P = np.vstack([np.ones(len(t))/np.sqrt(2), trigonometric]).T * np.sqrt(2 / l)
  return P

def generate_recent_history(y, w, start, i):
    """
    :param y: target array during training / testing
    :param w: window containing most recent history
    :param start: int, if there was an offset in the start if y
    :param i: current step w.r.t. step zero
    :return:
    """
    w = np.copy(w)
    m = len(w)  # lookback steps
    # deal with the prediction case: no y, we use stored window values
    y_w = y[start:i + 1]
    if len(y_w) == m:
        w = y_w
    else:
        w = np.roll(w, -len(y_w))
        w[-len(y_w):] = y_w
    return w


class StatefulForecaster(ScenarioGenerator):
    """
    Stateful Forecasters' predictions are a function of the past history seen by the model.
    They are more general than ARIMA/state-space/exponential smoothing models, as they can be any model that can be
    trained in a stateful way.
    In particular, the model of this class are able to:
    1) generate a past history vector, w, which is initially set to the first value observed when running the model, and
       is then populated as the predictions are made.
    2) train hyperparameters on multiple time series, by using the same hyperparameters for all time series, or by
       training them individually.
    """
    def __init__(self, target_name='target', targets_names=None, n_sa=1, m=24, val_ratio=0.8, optimize_hyperpars=True,
                 optimization_budget=100,  verbose=True, nodes_at_step=None,
                 q_vect=None, **scengen_kwgs):
        """
        :param target_name:
        :param targets_names:
        :param n_sa:
        :param alpha:
        :param m:
        :param omega:
        :param n_harmonics:
        :param val_ratio:
        :param nodes_at_step:
        :param q_vect:
        :param periodicity:
        :param optimize_hyperpars:
        :param optimization_budget:
        :param diagnostic_plots:
        :param verbose:
        :param scengen_kwgs:
        """

        assert m>0, 'm must be positive'
        self.targets_names = [target_name] if targets_names is None else targets_names
        self.init_pars = {'target_name': target_name, 'targets_names': self.targets_names, 'n_sa': n_sa, 'm': m,
                          'val_ratio': val_ratio, 'optimize_hyperpars': optimize_hyperpars,
                          'optimization_budget': optimization_budget, 'verbose': verbose,
                          'nodes_at_step': nodes_at_step, 'q_vect': q_vect}
        self.init_pars.update(scengen_kwgs)
        self.optimize_hyperpars = optimize_hyperpars
        self.optimization_budget = optimization_budget
        self.target_name = target_name
        self.m = m
        self.n_sa = n_sa
        self.val_ratio = val_ratio
        self.verbose = verbose

        self.states = {'w':None}
        self.hyperpar_lims = {}
        super().__init__(q_vect, nodes_at_step=nodes_at_step, val_ratio=val_ratio, **scengen_kwgs)

    @abstractmethod
    def reinit_pars(self):
        self.states['w'] = None

    @staticmethod
    def generate_recent_history(y, w, start, i):
        """
        :param y: target array during training / testing
        :param w: window containing most recent history
        :param start: int, if there was an offset in the start if y
        :param i: current step w.r.t. step zero
        :return:
        """
        w = np.copy(w)
        m = len(w)  # lookback steps
        # deal with the prediction case: no y, we use stored window values
        y_w = y[start:i + 1]
        if len(y_w) == m:
            w = y_w
        else:
            w = np.roll(w, -len(y_w))
            w[-len(y_w):] = y_w
        return w

    def tune_hyperpars(self, x_pd):
        pars_opt = tune_hyperpars(x_pd, self.__class__, hyperpars=self.hyperpar_lims, n_trials=self.optimization_budget,
                                  return_model=False, model_init_kwargs=self.init_pars)
        self.set_params(**pars_opt)
        self.reinit_pars()
        return self


    def fit(self, x_pd, y_pd=None, **kwargs):
        if self.optimize_hyperpars:
            self.tune_hyperpars(x_pd)

        y_present = x_pd[self.target_name].values
        x = x_pd.values

        # exclude the last n_sa, we need them to create the target
        preds = self.run(x, y_present, start_from=0, fit=True)[:-self.n_sa]
        #preds = self.predict(x_pd)[:-self.n_sa]

        # hankelize the target
        hw_target = hankel(y_present[1:], self.n_sa)
        resid = hw_target - preds
        self.err_distr = np.quantile(resid, self.q_vect, axis=0).T
        self.target_cols = [f'{self.target_name}_t+{i}' for i in range(1, self.n_sa + 1)]
        return self

    @abstractmethod
    def run(self, x, y, return_coeffs=False, start_from=0, fit=True):
        pass

class Fourier_es(StatefulForecaster):

    def __init__(self, target_name='target', targets_names=None, n_sa=1, m=24, val_ratio=0.8, optimize_hyperpars=True,
                 optimization_budget=100,  verbose=False, nodes_at_step=None,
                 q_vect=None, alpha=0.8, omega=0.99, n_harmonics=3, periodicity=None, **scengen_kwgs):
        """
        :param y:
        :param h:
        :param alpha:
        :param m:
        :return:
        """
        assert 0<alpha<1, 'alpha must be in (0, 1)'
        assert 0<omega<1, 'omega must be in (0, 1)'
        assert n_harmonics>0, 'n_harmonics must be positive'

        super().__init__(target_name=target_name, targets_names=targets_names, n_sa=n_sa, m=m, val_ratio=val_ratio,
                         optimize_hyperpars=optimize_hyperpars, optimization_budget=optimization_budget,
                         verbose=verbose, nodes_at_step=nodes_at_step, q_vect=q_vect, **scengen_kwgs)

        self.init_pars.update({'alpha': alpha, 'omega': omega, 'n_harmonics': n_harmonics, 'periodicity': periodicity})

        self.periodicity = periodicity if periodicity is not None else n_sa
        self.alpha = alpha
        self.omega = omega
        self.n_harmonics = n_harmonics
        self.n_harmonics_int = int(np.minimum(n_harmonics, m // 2))
        self.P_future = None

        self.hyperpar_lims = {'alpha': [0, 1], 'omega': [0, 1], 'n_harmonics': [1, m//2]}
        self.states = {'x': None, 'w':np.zeros(m), 'eps':0, 'last_1sa_preds':0}
        self.reinit_pars()


    def reinit_pars(self):
        self.states['x'] = None
        self.states['w'] = np.zeros(self.m)
        self.states['eps'] = 0
        self.states['last_1sa_preds'] = 0
        self.store_basis()

    def store_basis(self):
        self.n_harmonics_int = int(np.minimum(self.n_harmonics, self.m // 2))
        t_f = np.arange(2 * self.m + np.maximum(self.n_sa, self.periodicity))
        self.P_future = get_basis(t_f, self.m, self.n_harmonics_int)


    def predict(self, x_pd, **kwargs):
        x = x_pd.values
        y = x_pd[self.target_name].values
        return self.run(x, y, start_from=0, fit=False)


    def run(self, x, y, return_coeffs=False, start_from=0, fit=True):
        states = deepcopy(self.states)
        x = states['x']
        eps = states['eps']
        last_1sa_preds = states['last_1sa_preds']
        w_init = states['w']

        if x is None:
            x = np.zeros(2 * self.n_harmonics_int + 1)
            x[0] = y[0]*np.sqrt(self.m)
            preds = [[0]]
        else:
            preds = [[y[start_from]+eps]]

        coeffs_t_history = []
        last_1sa_preds = np.copy(last_1sa_preds)

        for i in range(start_from, len(y)):
            P_past = self.P_future[i % self.m:(i % self.m + self.m), :]
            # this is copy-pasting the Fourier smoothing in the last periodicity
            P_f = self.P_future[i % self.m + self.m - self.periodicity:i % self.m + self.m -self.periodicity + self.n_sa, :]
            start = np.maximum(0, i-self.m+1)

            w = self.generate_recent_history(y, w_init, start, i)

            #eps = self.omega * (w[-1] - preds[-1][0]) + (1-self.omega) * eps
            eps_obs = w[-1] - last_1sa_preds
            eps = self.omega * eps_obs + (1-self.omega) * eps
            coeffs_t = P_past[-len(w):,:].T@w
            x = self.alpha*coeffs_t + (1-self.alpha)*x
            last_preds = (P_f@x).ravel()
            last_1sa_preds = last_preds[0]
            preds.append((last_preds + eps*(self.n_sa-np.arange(self.n_sa))**2/self.n_sa**2).ravel() )
            if return_coeffs:
                coeffs_t_history.append(x)

        # only store states if we are fitting
        if fit:
            self.__setstate__({'x': x, 'w': w, 'eps': eps, 'last_1sa_preds': last_1sa_preds})

        if return_coeffs:
            return np.vstack(preds[1:]), np.vstack(coeffs_t_history)
        return np.vstack(preds[1:])

    def _predict_quantiles(self, x, **kwargs):
        preds = self.predict(x)
        return np.expand_dims(preds, -1) + np.expand_dims(self.err_distr, 0)

    def __getstate__(self):
        return self.states

    def __setstate__(self, states):
        self.states.update(states)



#@njit
def update_predictions(coeffs_t_history, start_from, y, Ps_future, period, h, m, omega, last_1sa_preds, eps, x, P, F, Q, H, R, n_harmonics, w_init):
    preds = []
    for i in range(start_from, len(y)):
        P_past = Ps_future[i % m:(i % m + m), :]
        # this is copy-pasting the Fourier smoothing in the last periodicity
        P_f = Ps_future[i % m + m  -period: i % m + m -period+h, :]

        start = max(0, i - m + 1)

        # deal with the prediction case: no y, we use stored window values
        y_w = y[start:i + 1].copy()
        if len(y_w) == m:
            w = y_w
        else:
            w = np.roll(w_init, -len(y_w))
            w[-len(y_w):] = y_w


        coeffs_t = np.dot(P_past[-len(w):, :].T, w)
        x, P = kalman(x, P, F, Q, coeffs_t, H, R)  # Assuming kalman is a Numba-compatible function
        coeffs = x

        last_preds = np.dot(P_f, coeffs).ravel()
        last_1sa_preds = last_preds[0].copy()

        eps = omega * (w[-1] - last_1sa_preds) + (1 - omega) * eps
        preds.append((last_preds + eps * (h - np.arange(h)) ** 2 / h ** 2).ravel())
        coeffs_t_history[:, i] = coeffs

        if i % m == 0 and i >= m:
            Q = np.corrcoef(coeffs_t_history[:, :i])
            R = Q * 0.01
            #P = np.eye(n_harmonics * 2 + 1) * 1000

    return preds, last_1sa_preds, eps, x, P, Q, R, coeffs_t_history, w



class FK(StatefulForecaster):


    def __init__(self, target_name='target', targets_names=None, n_sa=1, alpha=0.8, m=24, omega=0.99, n_harmonics=3,
                 val_ratio=0.8, nodes_at_step=None, q_vect=None, periodicity=None, optimize_hyperpars=True,
                 optimization_budget=100, r=0.1, q=0.1, verbose=True, **scengen_kwgs):
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

        super().__init__(target_name=target_name, targets_names=targets_names, n_sa=n_sa, m=m, val_ratio=val_ratio,
                         optimize_hyperpars=optimize_hyperpars, optimization_budget=optimization_budget,
                         verbose=verbose, nodes_at_step=nodes_at_step, q_vect=q_vect, **scengen_kwgs)

        self.targets_names = [target_name] if targets_names is None else targets_names
        self.init_pars.update({'alpha': alpha, 'omega': omega, 'n_harmonics': n_harmonics, 'periodicity': periodicity,
                               'r': r, 'q': q})
        self.init_pars.update(scengen_kwgs)
        self.periodicity = periodicity if periodicity is not None else n_sa
        if self.periodicity < n_sa:
            print('WARNING: periodicity is smaller than n_sa, this may lead to suboptimal results.')

        self.alpha = alpha
        self.omega = omega
        self.n_harmonics = n_harmonics
        self.r = r
        self.q = q
        self.F = None
        self.H = None
        self.P_future = None
        self.coeffs_t_history = []

        self.hyperpar_lims = {'alpha': [0, 1], 'omega': [0, 1], 'n_harmonics': [1, self.m // 2], 'r': [0.001, 0.8],
                              'q': [0.001, 0.8]}
        self.states = {'x': None, 'P': None, 'R': None, 'Q': None, 'w':np.zeros(m), 'eps':None, 'last_1sa_preds':0}
        self.reinit_pars()


    def reinit_pars(self):
        self.n_harmonics_int = int(np.minimum(self.n_harmonics, self.m // 2))
        # precompute basis over all possible periods
        self.F = np.eye(self.n_harmonics_int * 2 + 1)
        self.H = np.eye(self.n_harmonics_int * 2 + 1)

        self.states['x'] = np.zeros(2 * self.n_harmonics_int + 1)
        self.states['P'] = np.eye(self.n_harmonics_int * 2 + 1) * 1000
        self.states['R'] = np.eye(self.n_harmonics_int * 2 + 1) * self.r
        self.states['Q'] = np.eye(self.n_harmonics_int * 2 + 1) * self.q
        self.states['w'] = np.zeros(self.m)
        self.states['eps'] = None
        self.states['last_1sa_preds'] = 0
        self.P_future = None
        self.store_basis()

    def store_basis(self):
        t_f = np.arange(2 * self.m + np.maximum(self.n_sa, self.periodicity))
        self.P_future = get_basis(t_f, self.m, self.n_harmonics_int)


    def predict(self, x_pd, **kwargs):
        x = x_pd.values
        y = x_pd[self.target_name].values
        return self.run(x, y, start_from=0, fit=False)


    def run(self, x, y, return_coeffs=False, start_from=0, fit=True):
        states = deepcopy(self.states)
        Q = states['Q']
        R = states['R']
        P = states['P']
        x = states['x']
        w = states['w']
        eps = states['eps']
        last_1sa_preds = states['last_1sa_preds']

        if eps is None:
            eps = 0
            preds = [[0]]
        else:
            preds = [[y[start_from]+eps]]

        if len(self.coeffs_t_history)>0:
            coeffs_t_history = np.vstack([self.coeffs_t_history, np.zeros((len(y) - start_from, self.n_harmonics_int * 2 + 1))])
        else:
            coeffs_t_history = np.zeros((len(y) - start_from, self.n_harmonics_int * 2 + 1))
        (preds_updated, last_1sa_preds, eps,
         x, P, Q, R, coeffs_t_history, w) = update_predictions(coeffs_t_history.T, start_from, y, self.P_future,
                                                               self.periodicity, self.n_sa, self.m, self.omega,
                                                               last_1sa_preds, eps, x, P, self.F, Q, self.H, R,
                                                               self.n_harmonics_int, w)
        if fit:
            self.__setstate__({'x': x, 'P': P, 'R': R, 'Q': Q, 'w': w, 'eps': eps, 'last_1sa_preds': last_1sa_preds})

        preds = preds + preds_updated
        self.coeffs_t_history = coeffs_t_history.T

        if return_coeffs:
            return np.vstack(preds[1:]), coeffs_t_history.T
        return np.vstack(preds[1:])

    def _predict_quantiles(self, x, **kwargs):
        preds = self.predict(x)
        return np.expand_dims(preds, -1) + np.expand_dims(self.err_distr, 0)

    def __getstate__(self):
        return self.states

    def __setstate__(self, states):
        self.states.update(states)


class FK_multi(StatefulForecaster):
    """
        Multistep ahead forecasting with multiple Fourier-Kalman regressors
    """

    def __init__(self, target_name='target', targets_names=None, n_sa=1,  n_predictors=4, alphas=None, m=24, omegas=None,
                 ns_harmonics=None, val_ratio=0.8, nodes_at_step=None, q_vect=None, periodicity=None,
                 base_predictor=Fourier_es, optimize_hyperpars=False, optimize_submodels_hyperpars=True,
                 optimization_budget=100, r=0.1, q=0.1, verbose=True, submodels_pars=None, **scengen_kwgs):
        """
        :param y:
        :param h: this is the numebr of steps ahead to be predicted.
        :param alpha:
        :param m:
        :return:
        """
        n_predictors = int(n_predictors)
        self.base_predictor = base_predictor

        super().__init__(target_name=target_name, targets_names=targets_names, n_sa=n_sa, m=m, val_ratio=val_ratio,
                         optimize_hyperpars=optimize_hyperpars, optimization_budget=optimization_budget,
                         verbose=verbose, nodes_at_step=nodes_at_step, q_vect=q_vect, **scengen_kwgs)
        # submodels' init pars are set to the father init class pars
        self.submodels_init_pars = deepcopy(self.init_pars)
        self.submodels_pars = submodels_pars

        self.init_pars.update({'n_predictors': n_predictors, 'alphas': alphas, 'omegas': omegas,
                          'ns_harmonics': ns_harmonics, 'periodicity': periodicity, 'r': r, 'q': q,
                          'optimize_submodels_hyperpars': optimize_submodels_hyperpars,
                          'base_predictor': base_predictor, 'submodels_pars':submodels_pars})
        self.init_pars.update(scengen_kwgs)

        self.optimize_submodels_hyperpars = optimize_submodels_hyperpars
        self.periodicity = periodicity if periodicity is not None else n_sa
        assert self.periodicity < m, 'Periodicity must be smaller than history m'
        if self.periodicity < n_sa:
            print('WARNING: periodicity is smaller than n_sa, this may lead to suboptimal results.')
        self.n_predictors = n_predictors
        self.r = r
        self.q = q
        self.coeffs_t_history = []
        self.H = None
        self.F = None
        self.models = None
        self.states = {'x':None, 'P':None, 'R':None, 'Q':None}
        self.reinit_pars()


    def reinit_pars(self):
        self.submodels_init_pars.update({'optimize_hyperpars': self.optimize_submodels_hyperpars,
                                         'verbose':True})
        # create models with increasing history length
        ms = np.linspace(1, self.m, self.n_predictors + 1).astype(int)[1:]
        ms = np.maximum(ms, self.n_sa)

        submodels_pars = [self.submodels_init_pars.copy() for i in range(self.n_predictors)]
        if np.any([m<self.periodicity for m in ms]):
            print('The history of the predictors are: {}'.format(ms))
            print('But periodicity is {}'.format(self.periodicity))
            print('I am going to set the history of the predictors with m<periodicity to the periodicity')
            ms = np.maximum(ms, self.periodicity)
        for i in range(self.n_predictors):
            submodels_pars[i].update({'m':ms[i]})

        # precompute basis over all possible periods
        self.F = np.eye(self.n_predictors)
        self.H = np.eye(self.n_predictors)

        self.states['x'] = np.ones(self.n_predictors) / self.n_predictors
        self.states['P'] = np.eye(self.n_predictors) * 1000
        self.states['R'] = np.eye(self.n_predictors) * self.r
        self.states['Q'] = np.eye(self.n_predictors) * self.q


        if self.submodels_pars is not None:
            self.models = [self.base_predictor(**self.submodels_pars[i]) for i in range(self.n_predictors)]
        else:
            self.models = [self.base_predictor(**submodels_pars[i]) for i in range(self.n_predictors)]

    def fit(self, x_pd, y_pd=None, **kwargs):
        if self.optimize_hyperpars:
            pars_opt = tune_hyperpars(x_pd, FK_multi, hyperpars={'fake':[0, 1]},
                            n_trials=1, return_model=False, parallel=False, model_init_kwargs=self.init_pars)
            self.set_params(**pars_opt)
            self.reinit_pars()

        preds, _ = self.run(x_pd, fit=True)

        # exclude the last n_sa, we need them to create the target
        preds = preds[:-self.n_sa]

        # hankelize the target
        hw_target = hankel(x_pd[self.target_name].values[1:], self.n_sa)
        resid = hw_target - preds
        self.err_distr = np.quantile(resid, self.q_vect, axis=0).T
        self.target_cols = ['{}_{}'.format(self.target_name, t) for t in np.arange(self.n_sa)]
        return self

    def predict(self, x_pd, **kwargs):
        return self.run(x_pd, fit=False, return_coeffs=True)[0]


    def run(self, x_pd, return_coeffs=True, fit=True):
        preds = [[0]]
        coeffs_t_history = []
        states = deepcopy(self.states)
        Q = states['Q']
        R = states['R']
        P = states['P']
        x = states['x']

        # set stored states
        if fit:
            for j in range(self.n_predictors):
                self.models[j].fit(x_pd)

        preds_experts = np.dstack([ self.models[j].predict(x_pd) for j in range(self.n_predictors)])


        for i, idx in enumerate(x_pd.index):
            if i >= self.n_sa:
                # average last point error over different prediction times for all the models
                last_obs = x_pd[self.target_name].iloc[i]
                # sometimes persistence error is too small, sometimes signal could be small. We normalize by the average
                norm_avg_err = [np.mean([np.abs(preds_experts[i-sa, sa, predictor]-last_obs) for sa in range(self.n_sa)]) for predictor in range(self.n_predictors)]
                norm_avg_err = np.array(norm_avg_err) / np.mean(norm_avg_err)
                coeffs_t = np.exp(-np.array(norm_avg_err)) / np.exp(-np.array(norm_avg_err)).sum()
            else:
                coeffs_t = x
            try:
                x, P = kalman(x, P, self.F, Q, coeffs_t, self.H, R)
            except Exception as e:
                print(e)
            if not np.any(np.isnan(x)):
                coeffs = x
            else:
                coeffs = coeffs_t

            coeffs = np.abs(coeffs / np.abs(coeffs).sum())
            preds.append(preds_experts[i] @ coeffs)

            if return_coeffs:
                coeffs_t_history.append(coeffs_t)
            if i % self.m == 0 and i >= self.m:
                Q = np.corrcoef(np.vstack(coeffs_t_history).T)
                R = np.corrcoef(np.vstack(coeffs_t_history).T) * 10
                #P = np.eye(self.n_predictors) * 1000

        if fit:
            # if we were fitting, store states. Do nothing if we're predicting
            # self.states = [self.models[j].__getstate__() for j in range(self.n_predictors)]
            self.__setstate__({'Q':Q, 'R':R, 'P':P, 'x':x})
        if return_coeffs:
            return np.vstack(preds[1:]), np.vstack(coeffs_t_history)
        return np.vstack(preds[1:])

    def _predict_quantiles(self, x, **kwargs):
        preds = self.predict(x)
        return np.expand_dims(preds, -1) + np.expand_dims(self.err_distr, 0)

    def __getstate__(self):
        return self.states

    def __setstate__(self, states):
        self.states.update(states)