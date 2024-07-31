import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV, ShrunkCovariance
from pyforecaster.forecaster import LinearForecaster
from tqdm import tqdm
from pyforecaster.utilities import convert_multiindex_pandas_to_tensor, from_tensor_to_pdmultiindex
from pyforecaster.utilities import get_logger
import seaborn as sb
from pyforecaster.plot_utils import plot_scenarios_from_multilevel



class HierarchicalReconciliation:
    def __init__(self, hierarchy, method='minT', base_forecaster_class=LinearForecaster,
                 scenario_generation_method='unconditional', q_vect=None, n_scenarios=100, **forecaster_kwargs):
        self.hierarchy = hierarchy
        self.hierarchy_flatten = self.unroll_dict(hierarchy)
        self.method = method
        self.n_b = None
        self.n_u = None
        self.bottom_series = None
        self.aggregated_series = None
        self.a_matrix = None
        self.s_matrix = None
        self.g_matrix = None
        self.p_matrix = None
        self.ordered_cols = None
        self.base_forecaster_class = base_forecaster_class
        self.base_forecaster_kwargs = forecaster_kwargs
        self.models_dict = None
        self.steps = None
        self.scenario_generation_method = scenario_generation_method
        self.q_vect = np.linspace(0.01, 0.99, 11) if q_vect is None else q_vect
        self.logger = get_logger()
        self.n_scenarios = n_scenarios
        self.errs_hat = None
        self.errs_tilde = None
    def get_unrolled_hierarchy(self):
        return self.hierarchy_flatten

    def get_hierarchy_levels(self):
        l = pd.DataFrame({k: len(v) for k, v in self.hierarchy_flatten.items()}, index=[0])
        root = l.idxmax(1).values[0]
        # build levels starting from root
        levels = pd.DataFrame(self.explore_hierarchy(self.hierarchy, root), index=['level']).T
        return levels

    @staticmethod
    def explore_hierarchy(dict, node, level=0):
        if level == 0:
            levels = {node: level}
            level += 1
        else:
            levels = {}
        for n in dict[node]:
            levels[n] = level
            if n in dict.keys():
                levels.update(HierarchicalReconciliation.explore_hierarchy(dict, n, level=level+1))
        return levels


    @staticmethod
    def unroll_subdict(hierarchy, k):
        full_list = []
        for i in hierarchy[k]:
            if i in hierarchy.keys():
                sublist = HierarchicalReconciliation.unroll_subdict(hierarchy, i)
                full_list.append(sublist)
            else:
                full_list.append(np.array(i))
        return np.hstack(full_list)

    @staticmethod
    def unroll_dict(hierarchy):
        new_dict = {}
        for k, v in hierarchy.items():
            new_dict[k] = HierarchicalReconciliation.unroll_subdict(hierarchy, k)
        return new_dict

    def check_x(self, x):
        assert 'name' in x.columns, 'x must be in global_form format. Use the global_form option of the Formatter'
        # if aggregations are not present, create them
        assert set(self.hierarchy.keys()).issubset(set(x['name'].unique())), 'x must contain all aggregations'


    def get_predictions(self, x:pd.DataFrame, level='bottom'):
        if level == 'bottom':
            bottom_series = [c for c in x['name'].unique() if c not in self.hierarchy.keys()]
            y_hat = pd.concat({k: self.models_dict['bottom'].predict(x.loc[x['name'].isin([k])]
                                                                      .drop('name', axis=1)) for k in bottom_series},
                              axis=1)
        elif level == 'aggregations':
            y_hat = pd.concat({k: self.models_dict[k].predict(x.loc[x['name'].isin([k])].drop('name', axis=1)) for k in
                               self.hierarchy.keys()}, axis=1)
        elif level == 'all':
            y_hat_bottom = self.get_predictions(x, level='bottom')
            y_hat_aggr = self.get_predictions(x, level='aggregations')
            y_hat = pd.concat([y_hat_aggr, y_hat_bottom], axis=1)
        else:
            raise NotImplementedError('level must be either bottom or aggregations')

        return y_hat
    def get_quantiles(self, x: pd.DataFrame, level='bottom'):
        steps_ahead = len(self.steps)
        if level == 'bottom':
            bottom_series = [c for c in x['name'].unique() if c not in self.hierarchy.keys()]
            q_hat = pd.concat({k: from_tensor_to_pdmultiindex(
                    self.models_dict['bottom'].predict_quantiles(
                        x.loc[x['name'].isin([k])].drop('name', axis=1)),
                    x.loc[x['name'].isin([k])].index, np.arange(steps_ahead), self.q_vect) for k in
                    bottom_series}, axis=1)
        elif level == 'aggregations':
            q_hat = pd.concat(
                {k: from_tensor_to_pdmultiindex(
                    self.models_dict[k].predict_quantiles(x.loc[x['name'].isin([k])].drop('name', axis=1)),
                    x.loc[x['name'].isin([k])].index, np.arange(steps_ahead), self.q_vect) for k in
                    self.hierarchy.keys()},
                axis=1)
        elif level == 'all':
            q_hat_bottom = self.get_quantiles(x, level='bottom')
            q_hat_aggr = self.get_quantiles(x, level='aggregations')
            q_hat = pd.concat([q_hat_aggr, q_hat_bottom], axis=1)
        else:
            raise NotImplementedError('level must be either bottom or aggregations')

        return q_hat

    def get_target_matrix(self, x, y):
        bottom_series = [c for c in x['name'].unique() if c not in self.hierarchy.keys()]
        y_tr = y
        y_tr = pd.concat([pd.concat({k: y_tr.loc[x['name'].isin([k])] for k in self.hierarchy.keys()}, axis=1),
                          pd.concat({k: y_tr.loc[x['name'].isin([k])] for k in bottom_series}, axis=1)],
                         axis=1)
        return y_tr

    @staticmethod
    def get_scenarios(errs_tr, y_hat_te, n_scenarios, method='unconditional'):
        if method == 'unconditional':
            err_scenarios = np.zeros((y_hat_te.shape[0], y_hat_te.shape[1], n_scenarios))
            te_hour = y_hat_te.index.hour
            tr_hour = errs_tr.index.hour
            errs_tr_vals = errs_tr.values
            for h in range(24):
                h_idx_te = te_hour == h
                h_idx_tr = tr_hour == h
                errs_tr_h = errs_tr_vals[h_idx_tr, :]
                n_te = np.sum(h_idx_te)
                err_scenarios[h_idx_te, :, :] = np.dstack(
                    [errs_tr_h[np.random.choice(np.arange(len(errs_tr_h)), n_te), :] for n in range(n_scenarios)])
            # err_scenarios = err_scenarios.reshape(y_hat_te.shape[0], -1, n_scenarios)
            scenarios = np.expand_dims(y_hat_te.values, -1) + err_scenarios
        elif method == 'conditional':
            scenarios = 0
        else:
            raise NotImplementedError('method not yet implemented. Chose among conditional, unconditional')
        return scenarios
    def fit(self, x:pd.DataFrame, y:pd.DataFrame):
        """
        :return:
        """
        self.check_x(x)

        # train a separate global model for each level of the hierarchy
        models_dict = {}
        for k in self.hierarchy.keys():
            filt_k = x['name'] == k
            m = self.base_forecaster_class(**self.base_forecaster_kwargs)
            models_dict[k] = m.fit(x.loc[filt_k].drop('name', axis=1), y.loc[filt_k])

        # train a global model for the bottom series
        bottom_series = [c for c in x['name'].unique() if c not in self.hierarchy.keys()]
        filt_k = x['name'].isin(bottom_series)
        m = self.base_forecaster_class(**self.base_forecaster_kwargs)
        models_dict['bottom'] = m.fit(x.loc[filt_k].drop('name', axis=1), y.loc[filt_k])
        self.models_dict = models_dict

        # predict on the training set, bottom series and aggregations
        y_hat = self.get_predictions(x, level='all')

        # get the ground truth in matrix form
        y_tr = self.get_target_matrix(x, y)

        # fit reconciliation for all the steps ahead, keep errors, pre- and post-reconciliation
        self.steps = y_tr.columns.get_level_values(1).unique()
        errs_hat, errs_tilde = {}, {}
        target_names = y_hat.columns.get_level_values(1).unique()
        for t_name in tqdm(target_names):
            # ---------------------------- get base predictions, ground truth for sa ---------------------------------------
            y_hat_sa = y_hat.loc[:, y_hat.columns.get_level_values(1) == t_name].droplevel(1, 1)
            y_sa = y_tr.loc[:, y_hat.columns.get_level_values(1) == t_name].droplevel(1, 1)

            # ---------------------------- fit, predict -------------------------------------------------------------------
            self.fit_reconciliation(y_sa, y_hat_sa, self.hierarchy)
            y_tilde_sa = self.reconcile(y_hat_sa)

            # ---------------------------- retrieve error samples from the training set ------------------------------------

            errs_hat[t_name] = y_sa - y_hat_sa
            errs_tilde[t_name] = y_sa - y_tilde_sa

        self.errs_hat = pd.concat(errs_hat, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1)
        self.errs_tilde = pd.concat(errs_tilde, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1)
        return self

    def predict(self, x, method='reconciled'):
        self.check_x(x)
        # get predictions from base models, point forecasts and quantiles
        y_hat = self.get_predictions(x, level='all')
        if method == 'vanilla':
            return y_hat

        y_tilde = {}
        target_names = y_hat.columns.get_level_values(1).unique()
        for t_name in tqdm(target_names):
            # get reconciled forecasts at this step ahead
            y_hat_sa = y_hat.loc[:, y_hat.columns.get_level_values(1) == t_name].droplevel(1, 1)
            y_tilde[t_name] = self.reconcile(y_hat_sa)

        y_tilde = pd.concat(y_tilde, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1)

        return y_tilde

    def predict_quantiles(self, x, method='reconciled'):
        # get predictions from base models, point forecasts and quantiles

        if method == 'vanilla':
            return self.get_quantiles(x, level='all')
        elif method == 'reconciled':
            scens = self.predict_scenarios(x, method='reconciled')
            qs = scens.T.groupby(level=[0, 2]).quantile(self.q_vect).T
            return qs
        else:
            raise NotImplementedError('method not yet implemented. Chose among conditional, unconditional')


    def predict_scenarios(self, x, method='reconciled'):
        """
        Predict scenarios through (conditional) bootstrapping temporal traces of training set errors.
        :param x: features
        :param method: if 'vanilla', bootstrap base forecasts' errors, if 'reconciled', bootstraps errors of the
                       reconciled time series
        :return: multiindex dataframe with scenarios
        """

        y_hat = self.get_predictions(x, level='all')
        target_names = y_hat.columns.get_level_values(1).unique()
        if method == 'reconciled':
            y_tilde = pd.concat({t_name: self.reconcile(y_hat.loc[:, y_hat.columns.get_level_values(1) == t_name].
                                                    droplevel(1, 1)) for t_name in target_names}, axis=1)
            y_tilde = y_tilde.swaplevel(0, 1, axis=1).sort_index(axis=1)

        scens = []
        for h in range(24):
            h_idx_tr = self.errs_tilde.index.hour == h
            if method == 'vanilla':
                y_hat_h = y_hat.loc[h_idx_tr]
            elif method == 'reconciled':
                y_hat_h = y_tilde.loc[h_idx_tr]
            n_times = y_hat_h.shape[0]
            scens.append(pd.concat({s: y_hat_h + self.errs_tilde.loc[h_idx_tr]
                      .sample(n_times, replace=True)
                      .reset_index(drop=True)
                      .set_index(y_hat_h.index) for s in range(self.n_scenarios)},axis=1,names=['scenario', 'series', 'step'])\
                .swaplevel(0, 1, axis=1)\
                .sort_index(axis=1))

        return pd.concat(scens, axis=0).sort_index()
    def fit_reconciliation(self, y, y_hat, hierarchy:dict, cov_method='shrunk'):
        """
        hierarchy: dictionary containing {c:[list]} for all the upper time series in the errs dataframe

        """

        self.get_summation_matrix(y, hierarchy)

        self.ordered_cols = list(self.hierarchy_flatten.keys()) + self.bottom_series
        y = y[self.ordered_cols]
        y_hat = y_hat[self.ordered_cols]
        errs = y - y_hat

        # get forecast errors for the bottom and upper time series
        err_b = errs.loc[:, self.bottom_series]
        err_u = errs.loc[:, self.aggregated_series]

        y_rec = None
        # estimate covariances
        if self.method == 'dummy':
            precision = np.eye(self.n_b + self.n_u)
            self.p_matrix = np.linalg.inv(self.s_matrix.T @ precision @ self.s_matrix) @ (self.s_matrix.T @ precision)

        elif self.method == 'minT':
            cov, precision = self.estimate_covariance(errs.values, method=cov_method)
            self.p_matrix = np.linalg.inv(self.s_matrix.T @ precision @ self.s_matrix) @ (self.s_matrix.T @ precision)

        elif self.method == 'bayes':
            cov_b, precision_b = self.estimate_covariance(err_b.values, method=cov_method)
            if err_u.shape[1] < 2:
                cov_u = np.std(err_u)
            else:
                cov_u, precision_u = self.estimate_covariance(err_u.values, method=cov_method)
            self.g_matrix = cov_b @ self.a_matrix.T @ np.linalg.inv(cov_u + self.a_matrix @ cov_b @ self.a_matrix.T)
            # get posterior covariance of bottoms
            cov_b_post = cov_b - self.g_matrix.dot(cov_u + self.a_matrix.dot(cov_b.dot(self.a_matrix.T))).dot(self.g_matrix.T)
            cov_post = self.s_matrix.dot(cov_b_post.dot(self.s_matrix.T))
        else:
            raise NotImplementedError('method {} not implemented. Chose among minT, dummy, bayes'.format(self.method))

        return y_rec

    def reconcile(self, y_hat):
        if self.method in ['dummy', 'minT']:
            y_rec = self.s_matrix @ self.p_matrix @ y_hat[self.ordered_cols].T
        elif self.method == 'bayes':
            y_b = y_hat.loc[:, self.bottom_series]
            y_u = y_hat.loc[:, self.aggregated_series]
            y_b_rec = y_b.T + self.g_matrix @ (y_u.T - (self.a_matrix @ y_b.T))
            y_rec = self.s_matrix @ y_b_rec
        else:
            raise NotImplementedError('method {} not implemented. Chose among minT, dummy, bayes'.format(self.method))
        y_rec = y_rec.T
        y_rec.columns = y_hat.columns
        return y_rec

    def get_summation_matrix(self, y, hierarchy):
        self.bottom_series = [c for c in y.columns if c not in hierarchy.keys()]
        self.a_matrix = np.vstack([[1 if b in v else 0 for b in self.bottom_series] for v in self.hierarchy_flatten.values()])
        _, n_all = y.shape
        self.n_b = len(self.bottom_series)
        self.n_u = n_all - self.n_b
        I = np.eye(self.n_b)
        self.s_matrix = np.vstack([self.a_matrix, I])
        self.aggregated_series = [c for c in y.columns if c not in self.bottom_series]

    @staticmethod
    def estimate_covariance(x, method):
        """
        Covariance estimator wrapper
        :param x:
        :param method:
        :return:
        """
        cov = None
        if method == 'shrunk':
            cov = ShrunkCovariance().fit(x)
        elif method == 'glasso':
            cov = GraphicalLassoCV(cv=5, alphas=10, n_refinements=10).fit(x)
        else:
            ValueError('Covariance method not in [shrunk,glasso]')

        return cov.covariance_, cov.precision_

    def get_scenarios(self, errs_tr, y_hat_te, n_scenarios, method='unconditional'):
        if method == 'unconditional':
            err_scenarios = np.zeros((y_hat_te.shape[0], y_hat_te.shape[1], n_scenarios))
            te_hour = y_hat_te.index.hour
            tr_hour = errs_tr.index.hour
            errs_tr_vals = errs_tr.values
            for h in range(24):
                h_idx_te = te_hour == h
                h_idx_tr = tr_hour == h
                errs_tr_h = errs_tr_vals[h_idx_tr, :]
                n_te = np.sum(h_idx_te)
                err_scenarios[h_idx_te, :, :] = np.dstack(
                    [errs_tr_h[np.random.choice(np.arange(len(errs_tr_h)), n_te), :] for n in range(n_scenarios)])
            # err_scenarios = err_scenarios.reshape(y_hat_te.shape[0], -1, n_scenarios)
            scenarios = np.expand_dims(y_hat_te.values, -1) + err_scenarios
        elif method == 'conditional':
            scenarios = 0
        else:
            raise NotImplementedError('method not yet implemented. Chose among conditional, unconditional')
        return scenarios

    def compute_kpis(self, hat, x, y, metric, **metric_kwargs):
        y_mat = self.get_target_matrix(x, y)
        kpi = {}
        target_names = hat.columns.get_level_values(1).unique()
        for s in target_names:
            hat_s = convert_multiindex_pandas_to_tensor(hat.loc[:, (slice(None), [s], slice(None))].droplevel(1, 1))
            y_mat_s = y_mat.loc[:, (slice(None), [s])].droplevel(1, 1)
            kpi[s] = metric(hat_s, y_mat_s, **metric_kwargs).T
            if kpi[s].index.nlevels == 2:
                kpi[s] = kpi[s].unstack(level=0)

        kpi = pd.concat(kpi, axis=1).droplevel(1,1)
        kpi['level'] = self.get_hierarchy_levels()
        return kpi.groupby('level').mean()