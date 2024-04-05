import numpy as np
import numba
from tqdm import tqdm
from pyforecaster.forecaster import ScenarioGenerator
import pandas as pd
from pyforecaster.utilities import kalman


def hankel(x, n, generate_me=None):
    if generate_me is None:
        generate_me = np.arange(n)
    x = x.ravel()
    m = len(x) - n + 1
    w = np.arange(m)
    h = []
    for g in generate_me:
        h.append(x[w + g].reshape(-1, 1))
    return np.hstack(h)


class HoltWinters(ScenarioGenerator):
    def __init__(self, periods, target_name, q_vect=None, optimization_budget=800, n_sa=1, constraints=None,
                 **scengen_kwgs):
        """
        :param periods: vector of two seasonalities' periods e.g. [24, 7*24]
        :param optimization_budget: number of test point to optimize the parameters
        :param n_sa: number of steps ahead to be predicted
        :param q_vect: vector of quantiles
        """

        super().__init__(q_vect, **scengen_kwgs)
        self.periods = periods
        self.budget = optimization_budget
        self.n_sa = n_sa

        # HW parameters
        self.alpha = 1
        self.beta = 0
        self.gamma = [0.1, 0.1]

        # HW states
        self.a = None
        self.b = None
        self.s1 = None
        self.s2 = None

        self.exogenous_names = None
        self.m = None
        self.target_name = target_name

        self.coeffs_output = None
        self.coeffs_input = None
        self.y_hat_tr = None
        self.y_tr = None
        self.q_hat_err = None

        self.err_distr = None
        self.constraints = constraints

        super().__init__()

    def fit(self, x_pd, y_pd):
        """
        :param x: pd.DataFrame (n_obs, n_cov), pandas of covariates
        :param y: pd.DataFrame (n_obs, 1), target
        :return:
        """
        y = np.vstack(y_pd.values)

        # exclude the last n_sa, we need them to create the target
        hw_input = y[:-self.n_sa]

        # hankelize the target
        hw_target = hankel(y[1:], self.n_sa)

        # optimize HW parameters
        y_hat_hw = self.rand_search(hw_input.ravel(), hw_target)

        # see if there is some residual error which can be explained with the regressors
        resid = hw_target -y_hat_hw
        if x_pd is not None:
            x = x_pd.values
            self.coeffs_output = np.linalg.pinv(x[:-self.n_sa,:].T @ x[:-self.n_sa,:]) @ (x[:-self.n_sa,:].T @ resid)
            self.y_hat_tr = pd.DataFrame(y_hat_hw + x[:-self.n_sa, :] @ self.coeffs_output,
                                         index=y_pd.index[:-self.n_sa])
            resid = hw_target - self.y_hat_tr
        else:
            self.y_hat_tr = pd.DataFrame(y_hat_hw, index=y_pd.index[:-self.n_sa])

        self.y_tr = pd.DataFrame(hw_target, index=y_pd.index[:-self.n_sa])

        # reinit HW
        # concat past inputs and last row of target
        self.reinit(y)
        self.err_distr = np.quantile(resid, self.q_vect, axis=0).T
        return self

    def predict(self, x_pd, **kwargs):
        """
        Most probably n_pred=1, y_past is the previous observation of the target
        :param x_pd: pd.DataFrame (n_pred, n_cov)
        :return:
        """

        # create HW input
        y_past = np.vstack(x_pd[self.target_name].values)
        hw_input = y_past

        # predict the HW
        y_hat_hw, a, b, s1, s2 = self._run(hw_input.ravel(), self.alpha, 0, self.gamma)

        # set the states (it is thought that the forecaster is called in sequence on ordinate samples)
        self.a = a
        self.b = b
        self.s1 = s1
        self.s2 = s2

        # correct for covariates
        if len(x_pd.columns) > 1:
            x = x_pd.values
            y_hat = y_hat_hw + x @ self.coeffs_output
        else:
            y_hat = y_hat_hw

        self.y_hat_te = y_hat

        return y_hat

    def predict_quantiles(self, x, **kwargs):
        if isinstance(x, pd.DataFrame):
            x = x.values
        preds = self.predict(x)
        return np.expand_dims(preds, -1) + np.expand_dims(self.err_distr, 0)

    def rand_search(self, hw_input, hw_output):
        np.random.seed(0)
        test_from = int(len(hw_input)/10)

        test_pars = np.random.rand(self.budget,3)
        rmses = np.zeros(len(test_pars))
        '''
        res = forest_minimize(partial(self.run_wrapper, hw_input, hw_output),[(0.,1.), (0.,1.), (0.,1.)], n_calls=N)
        self.alpha, self.gamma[0], self.gamma[1] = res.x
        '''
        self.reinit(hw_input[:np.max(self.periods)])
        for i in tqdm(range(len(test_pars))):
            self.reinit(hw_input[:np.max(self.periods)])
            y_hat = self._run(hw_input.ravel(), test_pars[i][0], 1e-3, test_pars[i][1:3])[0]
            rmses[i] = self.rmse(y_hat[test_from:,:],hw_output[test_from:,:])
        best_idx = np.argmin(rmses)

        self.alpha, self.gamma[0], self.gamma[1] = test_pars[best_idx, :]

        y_hat_hw, a, b, s1, s2 = self._run(hw_input.ravel(), self.alpha, self.beta, self.gamma)

        self.a = a
        self.b = b
        self.s1 = s1
        self.s2 = s2

        return y_hat_hw

    def rmse(self,x,y):
        return np.mean(np.mean((x-y)**2, axis=1)**0.5)

    def get_states(self):
        states = {'a': self.a,
                  'b': self.b,
                  's1': self.s1,
                  's2': self.s2}
        return states

    def set_states(self, a=None, b=None,s1=None,s2=None):
        self.a = a if a is not None else self.a
        self.b = b if b is not None else self.b
        self.s1 = s1 if s1 is not None else self.s1
        self.s2 = s2 if s2 is not None else self.s2

    def reinit(self, y_pd):
        """
        Use it to reinit the states of the HW: it doesn't fit the parameters, just reinit the states
        :param y_pd: last values of target, should be longer than biggest periodicity
        :return:
        """
        if isinstance(y_pd, pd.DataFrame) or isinstance(y_pd, pd.Series):
            y = np.vstack(y_pd.values).ravel()
        else:
            y = np.vstack(y_pd).ravel()
        n_repeat = 20
        p = np.max(self.periods)
        for i in range(n_repeat):
            try:
                y_hat_hw, a, b, s1, s2 = self._run(y[-p:])
            except:
                print('dasd')
            self.a = a
            self.b = b
            self.s1 = s1
            self.s2 = s2

    def _run(self, Y, alpha=None, beta=None, gamma=None):
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma

        am1 = 0 if self.a is None or np.isnan(self.a) else self.a
        bm1 = 0 if self.b is None or np.isnan(self.b) else self.b
        s1_init = np.tile(Y[-1], self.periods[0]) if self.s1 is None or np.all(np.isnan(self.s1)) else self.s1
        s2_init = np.tile(Y[-1], self.periods[1]) if self.s2 is None or np.all(np.isnan(self.s2)) else self.s2

        horizon = np.asanyarray(1 + np.arange(self.n_sa),dtype=float)
        period = np.asanyarray(self.periods, dtype=float)
        alpha = np.asanyarray(alpha, dtype=float)
        beta = np.asanyarray(beta, dtype=float)
        gamma = np.asanyarray(gamma, dtype=float)

        Y = np.atleast_1d(np.asanyarray(Y, dtype=float).ravel())
        constraints = self.constraints if self.constraints is not None else [-np.inf, np.inf]
        constraints = np.array(constraints).ravel()
        y_hat,am1,bm1,s1mp,s2mp = run(horizon, period, alpha,
                                      beta, gamma, s1_init,
                                      s2_init, am1, bm1, Y,
                                      np.tile(Y.reshape(-1, 1), self.n_sa), np.zeros(self.n_sa, dtype=float),
                                      constraints)
        return y_hat,am1,bm1,s1mp,s2mp


@numba.jit(nopython=True)
def run(horizon, period, alpha, beta, gamma, s1mp, s2mp, am1, bm1, Y, y_tot, y_hat_i, constraints):
    """
    Predict additive
    :param y: y
    :type y: numpy ndarray
    """
    s1mp_shift = 0*s1mp
    s2mp_shift = 0*s2mp
    for j in np.arange(len(Y)):
        if np.isnan(Y[j]):
            a = alpha * (y_tot[j,0] - s1mp[0] - s2mp[0]) + (1 - alpha) * (am1 + bm1)
            s1 = gamma[0] * (y_tot[j,0] - a - s2mp[0]) + (1 - gamma[0]) * s1mp[0]
            s2 = gamma[1] * (y_tot[j,0] - a - s1mp[0]) + (1 - gamma[1]) * s2mp[0]
        else:
            a = alpha * (Y[j] - s1mp[0] - s2mp[0]) + (1 - alpha) * (am1 + bm1)
            s1 = gamma[0] * (Y[j] - a - s2mp[0]) + (1 - gamma[0]) * s1mp[0]
            s2 = gamma[1] * (Y[j] - a - s1mp[0]) + (1 - gamma[1]) * s2mp[0]
        s1 = constrainify(s1, constraints)
        s2 = constrainify(s2, constraints)

        b = beta * (a - am1) + (1 - beta) * bm1
        s1mp_shift[:-1] = s1mp[1:]
        s2mp_shift[:-1] = s2mp[1:]
        s1mp_shift[-1] = s1
        s2mp_shift[-1] = s2
        s1mp = s1mp_shift
        s2mp = s2mp_shift
        am1 = a
        bm1 = b
        for i, h in enumerate(horizon):
            y_tot[j,i] = a + h * b + s1mp[int((h - 1) % period[0])] + s2mp[int((h - 1) % period[1])]

    return y_tot, am1, bm1, s1mp, s2mp


@numba.jit(nopython=True)
def constrainify(x, constraints):
    x = np.minimum(np.maximum(constraints[0], x), constraints[1])
    return x


class HoltWintersMulti(ScenarioGenerator):
    def __init__(self, periods, target_name, q_vect=None, optimization_budget=800, n_sa=1, constraints=None,
                 models_periods=None, **scengen_kwgs):
        """
        :param periods: vector of two seasonalities' periods e.g. [24, 7*24]
        :param optimization_budget: number of test point to optimize the parameters
        :param n_sa: number of steps ahead to be predicted
        :param q_vect: vector of quantiles
        """

        super().__init__(q_vect, **scengen_kwgs)
        self.periods = periods
        self.budget = optimization_budget
        self.n_sa = n_sa
        self.models_periods = models_periods if models_periods is not None else np.arange(1, 1+n_sa)

        models = []
        for n in self.models_periods:
            models.append(HoltWinters(periods=periods, q_vect=q_vect,
                             n_sa=n, target_name=target_name, optimization_budget=optimization_budget))

        self.models = models

    def fit(self, x_pd, y_pd):
        err_distr = np.zeros((self.n_sa, len(self.q_vect)))
        k = 0
        for i,m in enumerate(self.models):
            m.fit(x_pd, y_pd)
            selection = np.arange(k, m.err_distr.shape[0])
            err_distr[selection, :] = m.err_distr[selection, :]
            k = m.err_distr.shape[1]

        # reinit HW
        # concat past inputs and last row of target
        self.reinit(y_pd)
        self.err_distr = err_distr

        return self

    def predict(self, x,  **kwargs):
        y_hat = np.zeros((x.shape[0], self.n_sa))
        k = 0
        for i,m in enumerate(self.models):
            y_hat_m = m.predict(x)
            selection = np.arange(k, y_hat_m.shape[1])
            y_hat[:, selection] = y_hat_m[:, selection]
            k = y_hat_m.shape[1]
        return y_hat

    def reinit(self, x):
        for i,m in enumerate(self.models):
            m.reinit(x)

    def predict_quantiles(self, x, **kwargs):
        preds = self.predict(x)
        return np.expand_dims(preds, -1) + np.expand_dims(self.err_distr, 0)


def get_basis(t, l, n_h, frequencies=None):
  """
  Get the first n_h sine and cosine basis functions and the projection
  matrix P
  """
  frequencies = np.arange(n_h)+1 if frequencies is None else frequencies
  trigonometric = np.vstack([np.vstack([np.sin(2*np.pi*k*t/l), np.cos(2*np.pi*k*t/l)]) for k in frequencies])
  P = np.vstack([np.ones(len(t))/np.sqrt(2), trigonometric]).T * np.sqrt(2 / l)
  return P


class Fourier_es(ScenarioGenerator):


    def __init__(self, target_name='target', n_sa=1, alpha=0.8, m=24, omega=0.99, n_harmonics=3, val_ratio=0.8, nodes_at_step=None, q_vect=None, periodicity=None, **scengen_kwgs):
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
        self.periodicity = periodicity if periodicity is not None else n_sa
        self.target_name = target_name
        # precompute basis over all possible periods
        self.alpha = alpha
        self.omega = omega
        self.n_sa=n_sa
        self.m = m
        self.n_harmonics = np.minimum(n_harmonics, m // 2)
        self.coeffs = None
        self.eps = None
        self.last_1sa_preds = 0
        self.w = np.zeros(m)
        self.P_past = None
        self.P_future = None
        self.store_basis()
        super().__init__(q_vect, nodes_at_step=nodes_at_step, val_ratio=val_ratio, **scengen_kwgs)

    def store_basis(self):
        t_f = np.arange(2 * self.m + np.maximum(self.n_sa, self.periodicity))
        self.P_future = get_basis(t_f, self.m, self.n_harmonics)
    def fit(self, x_pd, y_pd=None, **kwargs):
        y_present = x_pd[self.target_name].values
        x = x_pd.values
        self.run(x, y_present, start_from=0, fit=True)

        # exclude the last n_sa, we need them to create the target
        preds = self.predict(x_pd)[:-self.n_sa]

        # hankelize the target
        hw_target = hankel(y_present[1:], self.n_sa)
        resid = hw_target - preds
        self.err_distr = np.quantile(resid, self.q_vect, axis=0).T
        return self

    def predict(self, x_pd, **kwargs):
        x = x_pd.values
        y = x_pd[self.target_name].values
        return self.run(x, y, start_from=0, fit=False)


    def run(self, x, y, return_coeffs=False, start_from=0, fit=True):
        if self.coeffs is None:
            coeffs = np.zeros(2 * self.n_harmonics + 1)
            coeffs[0] = y[0]*np.sqrt(self.m)
            eps = 0
            preds = [[0]]
        else:
            eps = self.eps
            coeffs = self.coeffs
            preds = [[y[start_from]+eps]]

        coeffs_t_history = []
        last_1sa_preds = np.copy(self.last_1sa_preds)
        w = np.copy(self.w)
        for i in range(start_from, len(y)):
            P_past = self.P_future[i % self.m:(i % self.m + self.m), :]
            # this is copy-pasting the Fourier smoothing in the last periodicity
            P_f = self.P_future[i % self.m + self.m - self.periodicity:i % self.m + self.m -self.periodicity + self.n_sa, :]
            start = np.maximum(0, i-self.m+1)

            # deal with the prediction case: no y, we use stored window values
            y_w = y[start:i+1]
            if len(y_w)==self.m:
                w = y_w
            else:
                w = np.roll(np.copy(self.w), -len(y_w))
                w[-len(y_w):] = y_w

            #eps = self.omega * (w[-1] - preds[-1][0]) + (1-self.omega) * eps
            eps_obs = w[-1] - last_1sa_preds
            eps = self.omega * eps_obs + (1-self.omega) * eps
            coeffs_t = P_past[-len(w):,:].T@w
            coeffs = self.alpha*coeffs_t + (1-self.alpha)*coeffs
            last_preds = (P_f@coeffs).ravel()
            last_1sa_preds = last_preds[0]
            preds.append((last_preds + eps*(self.n_sa-np.arange(self.n_sa))**2/self.n_sa**2).ravel() )
            if return_coeffs:
                coeffs_t_history.append(coeffs)

        # only store states if we are fitting
        if fit:
            self.coeffs = coeffs
            self.eps = eps
            self.last_1sa_preds = last_1sa_preds
            self.w = w

        if return_coeffs:
            return np.vstack(preds[1:]), np.vstack(coeffs_t_history)
        return np.vstack(preds[1:])

    def predict_quantiles(self, x, **kwargs):
        preds = self.predict(x)
        return np.expand_dims(preds, -1) + np.expand_dims(self.err_distr, 0)

    def __getstate__(self):
        return (self.coeffs, self.eps, self.last_1sa_preds, self.w)
    def __setstate__(self, state):
        self.coeffs, self.eps, self.last_1sa_preds, self.w = state



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



class FK(ScenarioGenerator):


    def __init__(self, target_name='target', n_sa=1, alpha=0.8, m=24, omega=0.99, n_harmonics=3, val_ratio=0.8, nodes_at_step=None, q_vect=None, periodicity=None, **scengen_kwgs):
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

        self.periodicity = periodicity if periodicity is not None else n_sa
        if self.periodicity < n_sa:
            print('WARNING: periodicity is smaller than n_sa, this may lead to suboptimal results.')

        self.target_name = target_name
        # precompute basis over all possible periods
        self.alpha = alpha
        self.omega = omega
        self.n_sa=n_sa
        self.m = m

        self.eps = None
        self.last_1sa_preds = 0
        self.w = np.zeros(m)

        n_harmonics = np.minimum(n_harmonics, self.m // 2)

        self.n_harmonics = n_harmonics

        # precompute basis over all possible periods
        self.x = np.zeros(2 * n_harmonics + 1)

        self.F = np.eye(n_harmonics * 2 + 1)

        self.H = np.eye(n_harmonics * 2 + 1)
        self.P = np.eye(n_harmonics * 2 + 1) * 1000
        self.R = np.eye(n_harmonics * 2 + 1) * 0.01
        self.Q = np.eye(n_harmonics * 2 + 1) * 0.01

        self.coeffs_t_history = []
        self.P_future = None
        self.store_basis()
        super().__init__(q_vect, nodes_at_step=nodes_at_step, val_ratio=val_ratio, **scengen_kwgs)

    def store_basis(self):
        t_f = np.arange(2 * self.m + np.maximum(self.n_sa, self.periodicity))
        self.P_future = get_basis(t_f, self.m, self.n_harmonics)

    def fit(self, x_pd, y_pd=None, **kwargs):

        y_present = x_pd[self.target_name].values

        x = x_pd.values
        self.run(x, y_present, start_from=0, fit=True)

        # exclude the last n_sa, we need them to create the target
        preds = self.predict(x_pd)[:-self.n_sa]

        # hankelize the target
        hw_target = hankel(y_present[1:], self.n_sa)
        resid = hw_target - preds
        self.err_distr = np.quantile(resid, self.q_vect, axis=0).T
        return self

    def predict(self, x_pd, **kwargs):
        x = x_pd.values
        y = x_pd[self.target_name].values
        return self.run(x, y, start_from=0, fit=False)


    def run(self, x, y, return_coeffs=False, start_from=0, fit=True):
        if self.eps is None:
            eps = 0
            preds = [[0]]
        else:
            eps = self.eps
            preds = [[y[start_from]+eps]]

        if len(self.coeffs_t_history)>0:
            coeffs_t_history = np.vstack([self.coeffs_t_history, np.zeros((len(y) - start_from, self.n_harmonics * 2 + 1))])
        else:
            coeffs_t_history = np.zeros((len(y) - start_from, self.n_harmonics * 2 + 1))
        w_init = np.copy(self.w)
        preds_updated, last_1sa_preds, eps, x, P, Q, R, coeffs_t_history, w = update_predictions(coeffs_t_history.T, start_from, y, self.P_future, self.periodicity, self.n_sa, self.m, self.omega, self.last_1sa_preds, eps, self.x, self.P, self.F, self.Q, self.H, self.R,
                           self.n_harmonics, w_init)
        if fit:
            self.last_1sa_preds = last_1sa_preds
            self.eps = eps
            self.x = x
            self.P = P
            self.Q = Q
            self.R = R
            self.w = w

        preds = preds + preds_updated
        self.coeffs_t_history = coeffs_t_history.T

        if return_coeffs:
            return np.vstack(preds[1:]), coeffs_t_history.T
        return np.vstack(preds[1:])

    def predict_quantiles(self, x, **kwargs):
        preds = self.predict(x)
        return np.expand_dims(preds, -1) + np.expand_dims(self.err_distr, 0)

    def __getstate__(self):
        return (self.eps,self.last_1sa_preds, self.x, self.P, self.R, self.w)
    def __setstate__(self, state):
        self.eps, self.last_1sa_preds, self.x, self.P, self.R, self.w = state


class FK_multi(ScenarioGenerator):
    """
        Multistep ahead forecasting with multiple Fourier-Kalman regressors
    """

    def __init__(self, target_name='target', n_sa=1,  n_predictors=4, alpha=0.8, m=24, omega=0.99, n_harmonics=3, val_ratio=0.8, nodes_at_step=None, q_vect=None, periodicity=None,
                 base_predictor=Fourier_es, **scengen_kwgs):
        """
        :param y:
        :param h: this is the numebr of steps ahead to be predicted.
        :param alpha:
        :param m:
        :return:
        """
        self.periodicity = periodicity if periodicity is not None else n_sa
        assert self.periodicity < m, 'Periodicity must be smaller than history m'
        if self.periodicity < n_sa:
            print('WARNING: periodicity is smaller than n_sa, this may lead to suboptimal results.')

        self.n_sa = n_sa
        self.m = m
        self.target_name = target_name
        self.n_harmonics = n_harmonics

        n_harmonics = np.minimum(n_harmonics, m // 2)
        ms = np.linspace(1, m, n_predictors + 1).astype(int)[1:]
        ms = np.maximum(ms, n_sa)

        if np.any([m<self.periodicity for m in ms]):
            print('The history of the predictors are: {}'.format(ms))
            print('But periodicity is {}'.format(self.periodicity))
            print('I am going to set the history of the predictors with m<periodicity to the periodicity')
            ms = np.maximum(ms, self.periodicity)

        self.n_predictors = n_predictors

        # precompute basis over all possible periods
        self.x = np.ones(n_predictors) / n_predictors

        self.F = np.eye(n_predictors)
        self.H = np.eye(n_predictors)
        self.P = np.eye(n_predictors) * 1000
        self.R = np.eye(n_predictors)
        self.Q = 0.1

        self.models = [base_predictor(n_sa=n_sa, alpha=alpha, omega=omega, n_harmonics=n_harmonics, m=ms[i], target_name=target_name, periodicity=periodicity) for i in
                  range(n_predictors)]
        self.states = [self.models[j].__getstate__() for j in range(self.n_predictors)]
        self.coeffs_t_history = []

        super().__init__(q_vect, nodes_at_step=nodes_at_step, val_ratio=val_ratio, **scengen_kwgs)

    def fit(self, x_pd, y_pd=None, **kwargs):

        preds, _ = self.run(x_pd, fit=True)

        # exclude the last n_sa, we need them to create the target
        preds = preds[:-self.n_sa]

        # hankelize the target
        hw_target = hankel(x_pd[self.target_name].values[1:], self.n_sa)
        resid = hw_target - preds
        self.err_distr = np.quantile(resid, self.q_vect, axis=0).T
        return self

    def predict(self, x_pd, **kwargs):
        return self.run(x_pd, fit=False, return_coeffs=True)[0]


    def run(self, x_pd, return_coeffs=True, fit=True):
        preds = [[0]]
        coeffs_t_history = []
        Q = np.copy(self.Q)
        R = np.copy(self.R)
        P = np.copy(self.P)
        x = np.copy(self.x)

        # set stored states
        for j in range(self.n_predictors):
            self.models[j].__setstate__(self.states[j])

        preds_experts = np.dstack([self.models[j].run(x_pd.values, x_pd[self.target_name].values,
                                      return_coeffs=False, fit=True) for j in
                   range(self.n_predictors)])

        #from pyforecaster.plot_utils import ts_animation
        #ts_animation(np.rollaxis(preds_experts, -1),
        #             names=[str(i) for i in range(self.n_predictors)], target=x_pd['all'].values, frames=1000,
        #             interval=0.1, step=3)

        for i, idx in enumerate(tqdm(x_pd.index)):
            if i >= self.n_sa:
                # average last point error over different prediction times for all the models
                last_obs = x_pd[self.target_name].iloc[i]
                prev_obs = x_pd[self.target_name].iloc[i-1]
                # sometimes persistence error is too small, sometimes signal could be small. We normalize by the average
                norm_factor = (np.abs(last_obs - prev_obs) + np.abs(last_obs + prev_obs))/2
                norm_avg_err = [np.mean([np.abs(preds_experts[i-sa, sa, predictor]-last_obs) for sa in range(self.n_sa)]) / norm_factor for predictor in range(self.n_predictors)]
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
            self.states = [self.models[j].__getstate__() for j in range(self.n_predictors)]
            self.Q = Q
            self.R = R
            self.P = P
            self.x = x

        if return_coeffs:
            return np.vstack(preds[1:]), np.vstack(coeffs_t_history)
        return np.vstack(preds[1:])

    def predict_quantiles(self, x, **kwargs):
        preds = self.predict(x)
        return np.expand_dims(preds, -1) + np.expand_dims(self.err_distr, 0)

    def __getstate__(self):
        return (self.coeffs, self.eps,self.last_1sa_preds)
    def __setstate__(self, state):
        self.coeffs, self.eps, self.last_1sa_preds = state