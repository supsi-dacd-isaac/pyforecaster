import numpy as np
import numba
from tqdm import tqdm
from pyforecaster.forecaster import ScenarioGenerator
import pandas as pd
from pyforecaster.utilities import kalman
from functools import partial
import matplotlib.pyplot as plt
from copy import deepcopy

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

def fit_sample(pars_dict,model_class, model_init_kwargs, x, return_model=False):
    targets_names = model_init_kwargs['targets_names'] if 'targets_names' in model_init_kwargs.keys() else None
    model_init_kwargs.update(pars_dict)
    model = model_class(**model_init_kwargs)
    if targets_names is not None and model_class.__name__ not in ["HoltWintersMulti", "FK_multi"]:
        subscores = []
        for c in targets_names:
            model_init_kwargs.update({'target_name': c, 'targets_names': None})
            model = model_class(**model_init_kwargs)
            # features are all x columns except the targets and current target c
            features_names = [n for n in x.columns if n not in set(targets_names) - {c}]
            model.target_name = c
            subscores.append(score_autoregressive(model, x[features_names], target_name=c, n_sa=model.n_sa))
        score = np.mean(subscores)
    else:
        model.fit(x)
    if return_model:
        return model
    else:
        return score

def tune_hyperpars(x, model_class, hyperpars, n_trials=100, return_model=True, parallel=True, model_init_kwargs=None):
    """
    :param x: pd.DataFrame (n, n_cov)
    :param y: pd.Series (n)
    :param model: ScenarioGenerator
    :param hyperpars: dict of hyperparameters to be tuned, keys=hyperpar name, values: 2 elements array with min-max
    :param n_trials: number of calls to the optimizer
    :param targets_names: list if not None, consider all the columns in targets_list to be different time series
                          to train on, in a global model fashion
    :return:
    """
    verbose = model_init_kwargs['verbose'] if 'verbose' in model_init_kwargs.keys() else False
    pars_cartridge = []
    for i in range(n_trials):
        trial = {}
        for k,v in hyperpars.items():
            v = np.array(v)
            if len(v.shape)<2:
                sample = np.random.rand() * (v[1] - v[0]) + v[0]
            else:
                sample = np.random.rand(v.shape[0]) * (v[:, 1] - v[:, 0]) + v[:, 0]
            trial[k] = sample
        pars_cartridge.append(trial)

    model_init_kwargs.update({'optimize_hyperpars':False})
    model_init_kwargs_0 = deepcopy(model_init_kwargs)


    if model_class.__name__ in ["HoltWintersMulti", "FK_multi"]:
        fitted_model = fit_sample(pars_cartridge[0], model_class, model_init_kwargs, x, return_model=True)
    else:
        if parallel:
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import cpu_count
            with ProcessPoolExecutor(max_workers=np.minimum(cpu_count(), 20)) as executor:
                scores = list(tqdm(executor.map(partial(fit_sample, model_class=model_class,
                                                   model_init_kwargs=model_init_kwargs, x=x),
                                           pars_cartridge), total=n_trials, desc='Tuning hyperpars for {}'.format(model_class.__name__)))
        else:
            scores = []
            for i in tqdm(range(n_trials), desc='Tuning hyperpars for {}'.format(model_class.__name__)):
                scores.append(fit_sample(pars_cartridge[i], model_class, model_init_kwargs, x))
        if verbose:
            plt.figure()
            t = np.linspace(0.01, 0.99, 30)
            plt.plot(np.quantile(scores, t), t)

    if model_class.__name__ in ["HoltWintersMulti"]:
        alphas = np.array([m.alpha for m in fitted_model.models])
        gammas_1 = ([m.gamma_1 for m in fitted_model.models])
        gammas_2 = ([m.gamma_2 for m in fitted_model.models])
        best_pars = {'alphas': alphas, 'gammas_1': gammas_1, 'gammas_2': gammas_2, 'optimize_submodels_hyperpars':False}
    elif model_class.__name__ in ["FK_multi"]:
        submodels_pars = [m.get_params() for m in fitted_model.models]
        best_pars = {'submodels_pars':submodels_pars, 'optimize_submodels_hyperpars':False}
    else:
        best_idx = np.argmin(scores)
        best_pars = pars_cartridge[best_idx]

    model_init_kwargs.update(best_pars)
    model = model_class(**model_init_kwargs)
    if return_model:
        if 'target_name' in model_init_kwargs.keys():
            model.target_name = model_init_kwargs_0['target_name']
        model.fit(x, x)
        return model
    else:
        return model.get_params()

def score_autoregressive(model, x, tr_ratio=0.7, target_name=None, n_sa=1):
    # training test split
    n_training = int(tr_ratio*len(x))
    x_tr, x_te = x.iloc[:n_training], x.iloc[n_training:]

    # create the target
    target = hankel(x_te[target_name].values[1:], n_sa)

    # fit and score
    model.fit(x_tr, x_tr)
    y_hat = model.predict(x_te)

    return np.mean((y_hat.values[:len(target), :] - target) ** 2)

class HoltWinters(ScenarioGenerator):
    def __init__(self, periods, target_name, targets_names=None, q_vect=None, val_ratio=None, nodes_at_step=None, optimization_budget=800, n_sa=1, constraints=None,
                 optimize_hyperpars = True, alpha=0.2, beta=0, gamma_1=0.1, gamma_2=0.1, verbose=True,
                 **scengen_kwgs):
        """
        :param periods: vector of two seasonalities' periods e.g. [24, 7*24]
        :param optimization_budget: number of test point to optimize the parameters
        :param n_sa: number of steps ahead to be predicted
        :param q_vect: vector of quantiles
        """
        self.init_pars = {'periods': periods, 'target_name': target_name, 'q_vect': q_vect, 'val_ratio': val_ratio, 'nodes_at_step': nodes_at_step,
                            'optimization_budget': optimization_budget, 'n_sa': n_sa, 'constraints': constraints, 'optimize_hyperpars': optimize_hyperpars,
                          'verbose':verbose, 'alpha': alpha, 'beta': beta, 'gamma_1': gamma_1, 'gamma_2': gamma_2}
        super().__init__(q_vect, nodes_at_step=nodes_at_step, val_ratio=val_ratio, **scengen_kwgs)
        self.targets_names = [target_name] if targets_names is None else targets_names
        self.init_pars.update({'targets_names':self.targets_names})
        self.periods = periods
        self.optimization_budget = optimization_budget
        self.n_sa = n_sa
        self.optimize_hyperpars = optimize_hyperpars
        self.verbose = verbose

        # HW parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

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

    def fit(self, x_pd, y_pd=None, **kwargs):
        if self.optimize_hyperpars:
            pars_opt = tune_hyperpars(x_pd, HoltWinters, hyperpars={'alpha':[0, 1], 'gamma_1':[0, 1], 'gamma_2':[0, 1]},
                            n_trials=self.optimization_budget, return_model=False, model_init_kwargs=self.init_pars)
            self.set_params(**pars_opt)

        y = x_pd[self.target_name].values
        x = x_pd.values

        y_hat_hw, self.a, self.b, self.s1, self.s2 = self._run(y)

        hw_target = hankel(y[1:], self.n_sa)

        # we cannot exploit predictinos without ground truth, trim them
        y_hat_hw = y_hat_hw[:hw_target.shape[0], :]

        resid = hw_target - y_hat_hw
        self.coeffs_output = np.linalg.pinv(x[:-self.n_sa,:].T @ x[:-self.n_sa,:]) @ (x[:-self.n_sa,:].T @ resid)
        self.y_hat_tr = pd.DataFrame(y_hat_hw + x[:-self.n_sa, :] @ self.coeffs_output,
                                        index=x_pd.index[:-self.n_sa])
        resid = hw_target - self.y_hat_tr

        self.y_tr = pd.DataFrame(hw_target, index=x_pd.index[:-self.n_sa])

        # reinit HW
        # concat past inputs and last row of target
        self.reinit(y)
        self.err_distr = np.quantile(resid, self.q_vect, axis=0).T
        self.target_cols = ['{}_{}'.format(self.target_name, t) for t in np.arange(self.n_sa)]
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
        y_hat_hw, a, b, s1, s2 = self._run(hw_input.ravel(), self.alpha, 0, self.gamma_1, self.gamma_2)

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

        return pd.DataFrame(y_hat, index=x_pd.index, columns=self.target_cols)

    def _predict_quantiles(self, x, **kwargs):
        preds = self.predict(x)
        return np.expand_dims(preds, -1) + np.expand_dims(self.err_distr, 0)

    def rand_search(self, hw_input, hw_output):
        np.random.seed(0)
        test_from = int(len(hw_input)/10)

        test_pars = np.random.rand(self.optimization_budget,3)
        rmses = np.zeros(len(test_pars))
        '''
        res = forest_minimize(partial(self.run_wrapper, hw_input, hw_output),[(0.,1.), (0.,1.), (0.,1.)], n_calls=N)
        self.alpha, self.gamma[0], self.gamma[1] = res.x
        '''
        self.reinit(hw_input[:np.max(self.periods)])
        for i in tqdm(range(len(test_pars))):
            self.reinit(hw_input[:np.max(self.periods)])
            y_hat = self._run(hw_input.ravel(), test_pars[i][0], 1e-3, test_pars[i][1], test_pars[i][2])[0]
            rmses[i] = self.rmse(y_hat[test_from:,:],hw_output[test_from:,:])
        best_idx = np.argmin(rmses)

        self.alpha, self.gamma_1, self.gamma_2 = test_pars[best_idx, :]

        y_hat_hw, a, b, s1, s2 = self._run(hw_input.ravel(), self.alpha, self.beta, self.gamma_1, self.gamma_2)

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

    def _run(self, Y, alpha=None, beta=None, gamma_1=None, gamma_2=None):
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        gamma_1 = self.gamma_1 if gamma_1 is None else gamma_1
        gamma_2 = self.gamma_2 if gamma_2 is None else gamma_2
        gamma = [gamma_1, gamma_2]
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
    def __init__(self, periods, target_name, targets_names=None, q_vect=None, val_ratio=None, nodes_at_step=None, optimization_budget=800,
                 optimize_hyperpars = True, optimize_submodels_hyperpars=True, n_sa=1, constraints=None,
                 models_periods=None, verbose=True, alphas=None, gammas_1=None, gammas_2=None, **scengen_kwgs):
        """
        :param periods: vector of two seasonalities' periods e.g. [24, 7*24]
        :param optimization_budget: number of test point to optimize the parameters
        :param n_sa: number of steps ahead to be predicted
        :param q_vect: vector of quantiles
        """

        super().__init__(q_vect, nodes_at_step=nodes_at_step, val_ratio=val_ratio, **scengen_kwgs)
        self.init_pars = {'periods': periods, 'target_name': target_name, 'q_vect': q_vect, 'val_ratio': val_ratio, 'nodes_at_step': nodes_at_step,
                            'optimization_budget': optimization_budget, 'n_sa': n_sa, 'constraints': constraints, 'optimize_hyperpars': optimize_hyperpars,
                          'verbose':verbose, 'alphas': alphas, 'gammas_1': gammas_1, 'gammas_2': gammas_2,
                          'models_periods': models_periods, 'optimize_submodels_hyperpars':optimize_submodels_hyperpars,
                          'targets_names': targets_names}

        self.targets_names = [target_name] if targets_names is None else targets_names
        self.periods = periods
        self.optimization_budget = optimization_budget
        self.optimize_submodels_hyperpars = optimize_submodels_hyperpars
        self.n_sa = n_sa
        self.models_periods = models_periods if models_periods is not None else np.arange(1, 1+n_sa)
        self.alphas = np.ones(len(self.models_periods)) * 0.01 if alphas is None else alphas
        self.gammas_1 = np.ones(len(self.models_periods)) * 0.01 if gammas_1 is None else gammas_1
        self.gammas_2 = np.ones(len(self.models_periods)) * 0.01 if gammas_2 is None else gammas_2
        self.optimize_hyperpars = optimize_hyperpars
        self.target_name = target_name
        self.models = None
        self.reinit_pars()

    def reinit_pars(self):
        models = []
        for i, n in enumerate(self.models_periods):
            models.append(HoltWinters(periods=self.periods, q_vect=self.q_vect,
                             n_sa=n, target_name=self.target_name, optimization_budget=self.optimization_budget,
                                      targets_names=self.targets_names, verbose=False, optimize_hyperpars=self.optimize_submodels_hyperpars,
                                      alpha=self.alphas[i], gamma_1=self.gammas_1[i], gamma_2=self.gammas_2[i]))

        self.models = models

    def fit(self, x_pd, y_pd=None):
        if self.optimize_hyperpars:
            pars_opt = tune_hyperpars(x_pd, HoltWintersMulti, hyperpars={'fake':[0, 1]}, n_trials=1, return_model=False, parallel=False, model_init_kwargs=self.init_pars)
            self.set_params(**pars_opt)
            self.reinit_pars()

        err_distr = np.zeros((self.n_sa, len(self.q_vect)))
        k = 0
        for i,m in enumerate(self.models):
            m.fit(x_pd, y_pd)
            selection = np.arange(k, m.err_distr.shape[0])
            err_distr[selection, :] = m.err_distr[selection, :]
            k = m.err_distr.shape[1]

        # reinit HW
        # concat past inputs and last row of target
        self.reinit(x_pd[self.target_name].values)
        self.err_distr = err_distr

        return self

    def predict(self, x,  **kwargs):
        y_hat = np.zeros((x.shape[0], self.n_sa))
        k = 0
        for i,m in enumerate(self.models):
            y_hat_m = m.predict(x)
            selection = np.arange(k, y_hat_m.shape[1])
            y_hat[:, selection] = y_hat_m.iloc[:, selection]
            k = y_hat_m.shape[1]
        return pd.DataFrame(y_hat, index=x.index, columns=['{}_{}'.format(self.target_name, t) for t in np.arange(self.n_sa)])

    def reinit(self, x):
        for i,m in enumerate(self.models):
            m.reinit(x)

    def _predict_quantiles(self, x, **kwargs):
        preds = self.predict(x)
        return np.expand_dims(preds, -1) + np.expand_dims(self.err_distr, 0)




