import numpy as np
import numba
from tqdm import tqdm
from pyforecaster.forecaster import ScenarioGenerator
import pandas as pd


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

        am1 = 0 if self.a is None else self.a
        bm1 = 0 if self.b is None else self.b
        s1_init = np.tile(Y[-1], self.periods[0]) if self.s1 is None else self.s1
        s2_init = np.tile(Y[-1], self.periods[1]) if self.s2 is None else self.s2

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
