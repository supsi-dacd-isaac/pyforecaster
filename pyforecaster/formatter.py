#import category_encoders
import functools
import logging
from functools import partial
from itertools import product
from multiprocessing import cpu_count
from typing import Union
import concurrent.futures

import holidays as holidays_api
import numpy as np
import pandas as pd
from long_weekends.long_weekends import spot_holiday_bridges
from scipy.stats import binned_statistic
from tqdm import tqdm

from pyforecaster.big_data_utils import fdf_parallel, reduce_mem_usage
from pyforecaster.plot_utils import ts_animation_bars
from pyforecaster.utilities import get_logger


class Formatter:
    """
    :param augment: if true, doesn't discard original columns of the dataset. Could be helpful to discard most
                        recent data if you don't have at prediction time.
    """
    def __init__(self, logger=None, augment=True, dt=None, n_parallel=None):
        self.logger = get_logger(level=logging.WARNING, name='Formatter') if logger is None else logger
        self.transformers = []
        self.fold_transformers = []
        self.target_transformers = []
        self.target_normalizers = []
        self.cv_gen = []
        self.augment = augment
        self.timezone = None
        self.dt = dt
        self.n_parallel = n_parallel if n_parallel is not None else cpu_count()
        self.normalizing_fun = None
        self.denormalizing_fun = None

    def add_time_features(self, x):
        tz = x.index[0].tz
        self.logger.debug('Adding time features. Time zone for the df: {}'.format(tz))
        if self.timezone is None:
            if tz is None:
                time_features = [x.index.hour, x.index.dayofweek, x.index.hour * 60 + x.index.minute]
                col_names = ['hour', 'dayofweek', 'minuteofday']
                self.timezone = False
            else:
                if x.shape[1] == 0:
                    utc_offset = pd.DataFrame(index=x.index, columns=[0]).apply(lambda x: x.name.utcoffset().
                                                                                total_seconds() / 3600, axis=1)
                else:
                    utc_offset = x.apply(lambda x: x.name.utcoffset().total_seconds()/3600, axis=1)
                time_features = [x.index.hour, x.index.dayofweek, x.index.hour * 60 + x.index.minute, utc_offset]
                col_names = ['hour', 'dayofweek', 'minuteofday', 'utc_offset']
                self.timezone = True
        else:
            if self.timezone:
                if tz is not None:
                    utc_offset = x.apply(lambda x: x.name.utcoffset().total_seconds()/3600, axis=1)
                else:
                    self.logger.warning('I am setting UTC offset feature to zero since the timeindex you passed was '
                                        'not time zone localized')
                    utc_offset = np.zeros(x.shape[0])
                time_features = [x.index.hour, x.index.dayofweek, x.index.hour * 60 + x.index.minute, utc_offset]
                col_names = ['hour', 'dayofweek', 'minuteofday', 'utc_offset']
            else:
                time_features = [x.index.hour, x.index.dayofweek, x.index.hour * 60 + x.index.minute]
                col_names = ['hour', 'dayofweek', 'minuteofday']
        time_df = pd.DataFrame(np.vstack(time_features).T, columns=col_names, index=x.index)

        x = pd.concat([x, time_df], axis=1)
        return x

    def add_holidays(self, x, state_code='CH', **kwargs):
        self.logger.debug('Adding holidays')
        holidays = holidays_api.country_holidays(country=state_code, years=x.index.year.unique(), **kwargs)
        bridges, long_weekends = spot_holiday_bridges(start=x.index[0]-pd.Timedelta('2D'), end=x.index[-1]+pd.Timedelta('2D'), holidays=pd.DatetimeIndex(holidays.keys()))
        bridges = np.array([b.date() for b in bridges])
        long_weekends = np.array([b.date() for b in long_weekends])


        x['holidays'] = 0
        x.loc[np.isin(x.index.date, bridges), 'holidays'] = 1
        x.loc[np.isin(x.index.date, long_weekends), 'holidays'] = 2
        x.loc[np.isin(x.index.date, list(holidays.keys())), 'holidays'] = 3
        return x

    def add_transform(self, names, functions=None, agg_freq=None, lags=None, relative_lags=False, agg_bins=None, **kwargs):
        transformer = Transformer(names, functions=functions, agg_freq=agg_freq, lags=lags, logger=self.logger,
                                  relative_lags=relative_lags, agg_bins=agg_bins, dt=self.dt, **kwargs)
        self.transformers.append(transformer)
        return self

    def add_target_transform(self, names, functions=None, agg_freq=None, lags=None, relative_lags=False, agg_bins=None):
        if lags is not None and np.any(np.array(lags) > 0):
            self.logger.critical('some lags are positive, which mean you are adding a target in the past. '
                                 'Is this intended?')
        transformer = Transformer(names, functions=functions, agg_freq=agg_freq, lags=lags, logger=self.logger,
                                  relative_lags=relative_lags, agg_bins=agg_bins, dt=self.dt)
        self.target_transformers.append(transformer)
        return self

    def add_normalizing_fun(self, expr, inv_expr):
        """
        When specifying a noramlization, an anti-transform expression must be provided
        :param expr: type: str
        :param inv_expr: type: str
        :return:
        """
        self.normalizing_fun = expr
        self.denormalizing_fun = inv_expr

    def add_target_normalizer(self, target, function:str=None, agg_freq:str=None, lags:list=None, relative_lags=False, agg_bins=None, name='mean'):
        if isinstance(target, str):
            target = [target]
        if isinstance(function, str):
            function = [function]
        if len(np.shape(lags))==0 and lags is not None:
            lags = [lags]


        if len(self.target_normalizers)>0:
            if np.any([n.name==name for n in self.target_normalizers]):
                self.logger.warning('You already have a target normalizer with this name. This normalizer is not added, choose another one')
                return

        assert len(target) == 1, 'only one target can be normalized per time. If you want to normalize two targets call this method twice'
        if lags is not None:
            assert len(lags) == 1, 'only one lag is admissible for the normalization.'
            assert np.all(lags>0), 'you cannot normalize with future values, all lags should be positive'

        transformer = Transformer(target, functions=function, agg_freq=agg_freq, lags=lags, logger=self.logger,
                                  relative_lags=relative_lags, agg_bins=agg_bins, dt=self.dt, name=name)

        self.target_normalizers.append(transformer)

    def transform(self, x, time_features=True, holidays=False, return_target=True, global_form=False, parallel=False,
                  reduce_memory=True, corr_reorder=False, **holidays_kwargs):
        """
        Takes the DataFrame x and applies the specified transformations stored in the transformers in order to obtain
        the pre-fold-transformed dataset: this dataset has the correct final dimensions, but fold-specific
        transformations like min-max scaling or categorical encoding are not yet applied.
        :param x: pd.DataFrame
        :param time_features: if True add time features
        :param holidays: if True add holidays as a categorical feature
        :param return_target: if True, returns also the transformed target. If False (e.g. at prediction time), returns
                             only x
        :param global_form: if True, assumes that columns of x which are not transformed are independent signals to be
                            forecasted with a global model. In this case, all target transform must refer to a "target"
                            column, which is the stacking of the independent signals. An additional column "name" is
                            added to the transformed dataset, which contains the name of the signal to be forecasted.
                            This is useful for stacking models.
        :param parallel: if True, parallelize the transformation of the dataset. This is useful if you have a lot of
                            signals to transform and you have a lot of cores. If you have a lot of signals but not a lot
                            of cores, you can set parallel=False and the transformation will be done in a single core
                            but with a single pass on the dataset. This is useful if you have a lot of signals but not
                            a lot of cores.
        :param reduce_memory: if True, reduce memory usage by casting float64 to float32 and int64 to int32
        :param corr_reorder: if True, reorder columns of the transformed dataset by correlation with the target

        :return x, target: the transformed dataset and the target DataFrame with correct dimensions
        """
        if global_form:
            dfs = self.global_form_preprocess(x)

            xs, ys = [], []
            if parallel:
                n_cpu = cpu_count()
                n_folds = np.ceil(len(dfs) / n_cpu).astype(int)
                # simulate transform on one fold single core to retrieve metadata (ray won't persist class attributes)
                self._simulate_transform(dfs[0])
                for i in tqdm(range(n_folds)):
                    x, y = fdf_parallel(f=partial(self._transform, time_features=time_features, holidays=holidays,
                                                  return_target=return_target, **holidays_kwargs),
                                        df=dfs[n_cpu * i:n_cpu * (i + 1)])
                    xs, ys = self.global_form_postprocess(x, y, xs, ys, reduce_memory=reduce_memory, corr_reorder=corr_reorder)
            else:
                for df_i in dfs:
                    x, y = self._transform(df_i, time_features=time_features, holidays=holidays,
                                           return_target=return_target, **holidays_kwargs)
                    xs, ys = self.global_form_postprocess(x, y, xs, ys, reduce_memory=reduce_memory, corr_reorder=corr_reorder)

            x = pd.concat(xs)
            target = pd.concat(ys)
        else:
            x, target = self._transform(x, time_features=time_features, holidays=holidays,
                                        return_target=return_target, **holidays_kwargs)
        return x, target

    @staticmethod
    def _transform_(tr, x):
        return tr.transform(x, augment=False)
    def _transform(self, x, time_features=True, holidays=False, return_target=True, parallel=False, **holidays_kwargs):
        """
        Takes the DataFrame x and applies the specified transformations stored in the transformers in order to obtain
        the pre-fold-transformed dataset: this dataset has the correct final dimensions, but fold-specific
        transformations like min-max scaling or categorical encoding are not yet applied.
        :param x: pd.DataFrame
        :param time_features: if True add time features
        :param holidays: if True add holidays as a categorical feature
        :param return_target: if True, returns also the transformed target. If False (e.g. at prediction time), returns
                             only x
        :return x, target: the transformed dataset and the target DataFrame with correct dimensions
        """
        original_columns = x.columns
        if np.any(x.isna()):
           self.logger.warning('There are {} nans in x, nans are not supported yet, '
                               'get over it. I have more important things to do.'.format(x.isna().sum()))

        target = pd.DataFrame(index=x.index)

        if len(self.transformers)>0:
            if parallel:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
                    x_tr = pd.concat([i for i in executor.map(partial(self._transform_, x=x), self.transformers)], axis=1)
                    x = pd.concat([x, x_tr], axis=1)
            else:
                for tr in self.transformers:
                    x = tr.transform(x)
        transformed_columns = [c for c in x.columns if c not in original_columns]

        if return_target:
            for tr in self.target_transformers:
                target = pd.concat([target, tr.transform(x, augment=False)], axis=1)

        # apply normalization to target if any and if return_target is True
        if len(self.target_normalizers)>0:
            normalizing_columns = [nr.name for nr in self.target_normalizers]
            x = self.add_normalizing_columns(x)

            # this is needed even if target is not returned, to normalize features correlated to the target
            target, x = self.normalize(x, target, return_target=return_target)
            transformed_columns = transformed_columns + normalizing_columns

        if return_target:
            # remove raws with nans to reconcile impossible dataset entries introduced by shiftin' around
            x = x.loc[~np.any(x[transformed_columns].isna(), axis=1) & ~np.any(target.isna(), axis=1)]
            target = target.loc[~np.any(x[transformed_columns].isna(), axis=1) & ~np.any(target.isna(), axis=1)]
        else:
            x = x.loc[~np.any(x[transformed_columns].isna(), axis=1)]
        if not self.augment:
            x = x[transformed_columns]
        # adding time features
        if time_features:
            x = self.add_time_features(x)

        if holidays:
            x = self.add_holidays(x, **holidays_kwargs)
        return x, target

    def add_normalizing_columns(self, x):

        # if we're doing the direct transform (normalization) we compute the normalizers and add them to the x df
        # compute normalizers if any
        normalizers = pd.concat([nr.transform(x, augment=False) for nr in self.target_normalizers], axis=1)

        # rename normalizers with tag names
        normalizers.columns = [nr.name for nr in self.target_normalizers]
        x = pd.concat([x, normalizers], axis=1)

        return x

    def normalize(self, x, y=None, normalizing_fun=None, antitransform=False, return_target=True):
        """
        Columns needed to compute the normaliztion factors are computed by the target transformers and returned in
        the original x dataframe. The normalizing_fun is a string expression that must be evaluated to normalize the
        target columns. The expression must contain the following variables:
        - df[t]
        - the name of the normalizing columns
        e.g. "(df[t] - df['target_movingavg']) / df['target_movingstd'] "

        :param x:
        :param y:
        :param normalizing_fun:
        :return:
        """

        normalizing_fun = self.normalizing_fun if normalizing_fun is None else normalizing_fun
        if self.normalizing_fun is None:
            self.logger.warning('You did not pass any normalization expression, ** no normalization will be applied **. '
                                '\bYou can set a normalization expression by calling Formatter.add_normalizing_fun '
                                '\bor by passing the noralizing_expr argument to this function')
            return y, x


        normalizers = x[[nr.name for nr in self.target_normalizers]]

        # get normalizers names
        target_to_norm_names = [nr.names for nr in self.target_normalizers]
        target_to_norm_names = [item for sublist in target_to_norm_names for item in sublist]

        # normalize the target if any
        if return_target:
            if isinstance(y.columns, pd.MultiIndex):
                # we are normalizing a Multiindex Dataframe containing quantiles, indicated in level=1
                for tau in y.columns.get_level_values(1).unique():
                    y_tau = y.loc[:, (slice(None), tau)]
                    y_tau = y_tau.droplevel(1, 1)
                    y.loc[:, (slice(None), tau)] =  self.normalize_target_inner(y_tau, normalizers, target_to_norm_names,
                                                normalizing_fun).values
            else:
                y = self.normalize_target_inner(y, normalizers, target_to_norm_names, normalizing_fun)

        # normalize the features related to the target
        for target_to_norm in np.unique(target_to_norm_names):
            for tr in self.transformers:
                # find df_n columns to normalize
                nr_columns = (tr.metadata['name'].isin([target_to_norm])).index
                for c in nr_columns:
                    x.loc[:, c] = self.normalizing_wrapper(normalizing_fun, x, c)


        return y, x

    def normalize_target_inner(self, y, normalizers, target_to_norm_names, normalizing_fun):
        # join target and normalizers in a single df
        df_n = pd.concat([y, normalizers], axis=1)

        for target_to_norm in np.unique(target_to_norm_names):
            for tr in self.target_transformers:
                nr_columns = (tr.metadata['name'].isin([target_to_norm])).index
                for c in nr_columns:
                    df_n.loc[:, c] = self.normalizing_wrapper(normalizing_fun, df_n, c)
        y = df_n[[c for c in y.columns]]
        return y

    def denormalize(self, x, y):
        if self.denormalizing_fun is None:
            self.logger.warning('You did not pass any denormalization expression, ** no denormalization will be applied **. '
                                '\bYou can set a denormalization expression by calling Formatter.add_normalizing_fun ')
            return y
        y, _ = self.normalize(x.copy(), y, normalizing_fun=self.denormalizing_fun)
        return y

    def normalizing_wrapper(self, normalizing_fun, df, t):
        return eval(normalizing_fun)


    def _simulate_transform(self, x=None):
        """
        This won't actually modify the dataframe, it will just populate the metqdata property of each transformer
        :param x:
        :return:
        """
        for tr in self.transformers:
            _ = tr.transform(x, simulate=True)
        for tr in self.target_transformers:
            _ = tr.transform(x, simulate=True)

    def plot_transformed_feature(self, x, feature, frames=100, ax_labels=None, legend_kwargs={},
                   remove_spines=True, **kwargs):
        x_tr, target = self.transform(x)
        all_feats = pd.concat([x_tr, target], axis=1)
        fs = []
        start_times = []
        end_times = []
        names = []
        for t in self.transformers + self.target_transformers:
            derived_features = t.metadata.loc[t.metadata['name'] == feature]
            if len(derived_features) > 0:
                for function in derived_features['function'].unique():
                    fs.append(all_feats[derived_features.index[derived_features['function']==function]].values)
                    start_times.append(t.metadata.loc[derived_features.index[derived_features['function']==function], 'start_time'])
                    end_times.append(
                        t.metadata.loc[derived_features.index[derived_features['function'] == function], 'end_time'])

        ani = ts_animation_bars(fs, start_times, end_times, frames=frames, ax_labels=ax_labels,
                                legend_kwargs=legend_kwargs, remove_spines=remove_spines, **kwargs)
        return ani

    def cat_encoding(self, df_train, df_test, target, encoder=None, cat_features=None):
        if cat_features is None:
            self.logger.debug('auto encoding categorical features')
            cat_features = self.auto_cat_feature_selection(df_train)

        self.logger.debug('encoding {} as categorical features'.format(cat_features))
        df_train[cat_features] = pd.Categorical(df_train[cat_features])
        df_test[cat_features] = pd.Categorical(df_test[cat_features])

        encoder = encoder(cols=cat_features)
        encoder.fit(df_train[cat_features], df_train[target])
        df_train = df_train.join(
            encoder.transform(df_train[cat_features]).add_suffix('_cb'))
        df_test = df_test.join(
            encoder.transform(df_test[cat_features]).add_suffix('_cb'))
        return df_train, df_test

    @staticmethod
    def auto_cat_feature_selection(df, max_num_class_cat=24):
        """
        :param df: pd.DataFrame
        :param max_num_class_cat: maximum number of unique classes after which the feature is considered categorical
        """
        cat_features = df.columns[df.nunique() <= max_num_class_cat]
        return cat_features

    def time_kfcv(self, time_index, n_k=4, multiplier_factor=1, x=None):
        """

        :param time_index: DateTimeIndex of the full dataset
        :param n_k: number of folds for the time cross validation
        :param multiplier_factor: if >1, the cross validation folds are augmented. A standard time 4-CV looks like:
                                  |####|xx--|----|----|
                                  |--xx|####|xx--|----|
                                  |----|--xx|####|xx--|
                                  |----|----|--xx|####|
                                  while with a multiplier factor of 2 it becomes:
                                  |####|xx--|----|----|
                                  |--xx|####|xx--|----|
                                  |----|--xx|####|xx--|
                                  |----|----|--xx|####|
                                  |x###|#xx-|----|----|
                                  |-xx#|###x|x---|----|
                                  |----|-xx#|###x|x---|
                                  |----|----|-xx#|###x|
        :return:
        """

        assert n_k >= 2, 'n_k must be bigger than 2, passed: {}'.format(n_k)
        t_start = time_index[0]
        t_end = time_index[-1]
        split_times = pd.date_range(t_start, t_end, n_k)
        time_window = split_times[1]-split_times[0]
        shift_times = pd.date_range(t_start, t_end - time_window, n_k * multiplier_factor)

        # deadband time requires to know sampling times of the signals and all the time transformations.
        # If the metadata attribute of self.transformers is empty, meaning that the transform method of the
        # formatter has not been called yet, a call to the _simulate_transform method is needed. This won't actually
        # modify the dataframe, it will just populate the metadata

        if any([t.metadata is None for t in self.transformers]):
            self.logger.warning('seems like the transform method has not been called yet. Calling _simulate_transform '
                                'to retrieve the info I need')
            if x is None:
                self.logger.critical('you must pass x argument to time_kfcv or make sure that the transform method has'
                                     ' already been called once')
            self._simulate_transform(x)

        deadband_time = np.max(np.hstack([t.metadata['start_time'].abs().max() for t in self.transformers]))
        self.logger.debug('deadband time: {}'.format(deadband_time))
        if deadband_time > split_times[1]-split_times[0]:
            self.logger.error('deadband time: {} is bigger than test fold timespan: {}. Try '
                              'lowering k'.format(deadband_time, split_times[1]-split_times[0]))
            raise
        self.logger.debug('adding {} test folds with length {}'.format(n_k*multiplier_factor, time_window))
        folds = {}
        for i, t_i in enumerate(shift_times):
            test_window = [t_i, time_window + t_i]
            tr_idxs = (time_index < test_window[0] - deadband_time) | (time_index > test_window[1] + deadband_time)
            te_idx = (time_index >= test_window[0]) & (time_index <= test_window[1])
            # self.logger.debug(self.print_fold(tr_idxs, te_idx))
            yield tr_idxs, te_idx
            #folds['fold_{:>02d}'.format(i)] = pd.DataFrame({'tr':tr_idxs,  'te': te_idx}, index=time_index)
        #return pd.concat(folds, axis=1)

    def cv_generator(self, folds):
        pass

    @staticmethod
    def print_fold(tr_idxs, te_idxs):
        buffers = ~(tr_idxs + te_idxs)
        if sum(buffers) > 0:
            sampling_step = np.quantile(np.diff(np.where(np.diff(buffers.astype(int)))[0]),0.1).astype(int)
            #sampling_step = int(np.max(np.convolve(~(tr_idxs + te_idxs), np.ones())))
            #sampling_step = np.minimum(sampling_step, int(len(tr_idxs) / 100))
        else:
            sampling_step = int(len(tr_idxs) / 100)
        fold = ''
        for s in tr_idxs[::sampling_step].astype(int) + 2*te_idxs[::sampling_step].astype(int):
            fold += 'x' * (s == 0) + '-' * (s == 1) + '|' * (s == 2)
        fold += '   -: train, |=test, x:skip'
        return fold

    def prune_dataset_at_stepahead(self, df, target_col_num, metadata_features, method='periodic', period='24h', tol_period='1h', keep_last_n_lags=0, keep_last_seconds=0):

        # retrieve referring_time of the given sa for the target from target_transformers
        target_start_times = []
        target_end_times = []
        for tt in self.target_transformers:
            target_start_times.append(tt.metadata.iloc[target_col_num, :]['start_time'])
            target_end_times.append(tt.metadata.iloc[target_col_num, :]['end_time'])

        target_start_time = pd.to_timedelta(target_start_times[0])
        target_end_time = pd.to_timedelta(target_end_times[0])

        metadata = pd.concat([t.metadata for t in self.transformers])
        if method == 'periodic':
            c1 = (target_start_time - metadata['start_time']) % pd.Timedelta(period) <= pd.Timedelta(tol_period)
            c2 = (target_end_time - metadata['end_time']) % pd.Timedelta(period) <= pd.Timedelta(tol_period)
            causality = metadata['end_time'] <= target_start_time
            metadata['keep'] = (c1 | c2)  # & causality
        elif method == 'up_to':
            metadata['keep'] = metadata['end_time'] <= target_end_time
        else:
            metadata['keep'] = True

        if keep_last_n_lags > 0:
            metadata['keep'] = metadata['keep'] | metadata['lag'].isin(np.arange(keep_last_n_lags))

        if keep_last_seconds > 0:
            close = (metadata['end_time'] <= pd.Timedelta(keep_last_seconds, unit='s')) | (
                        metadata['end_time'] <= pd.Timedelta(keep_last_seconds, unit='s'))
            predecessor = metadata['end_time'] <= target_end_time
            metadata['keep'] = metadata['keep'] | (close & predecessor)

        features = list(metadata.loc[metadata['keep']].index) + metadata_features
        return df[features]

    def plot_dataset_at_stepahead(self, df, metadata_features=None, method='periodic', period='24h', tol_period='1h', keep_last_n_lags=0, keep_last_seconds=0):

        n_target = np.sum([len(t.metadata) for t in self.target_transformers])
        metadata = pd.concat([t.metadata for t in self.transformers])

        import matplotlib.pyplot as plt
        plt.figure()
        for target_col_num in np.arange(n_target):
            x_i = self.prune_dataset_at_stepahead(df, target_col_num, metadata_features=metadata_features, method=method,
                                                 period=period, keep_last_n_lags=keep_last_n_lags,
                                                 keep_last_seconds=keep_last_seconds,
                                                 tol_period=tol_period)
            plt.cla()
            plt.plot(np.hstack([metadata['start_time'].min().total_seconds(), metadata['end_time'].max().total_seconds()]), np.hstack([0, 0]), linestyle='')
            k = 0
            for i, feature in enumerate(metadata['name'].unique()):
                metadata_filt = metadata.loc[metadata['name'] == feature].loc[x_i.columns]
                for _, var in metadata_filt.iterrows():
                    k += 1
                    plt.plot(np.hstack([var['start_time'].total_seconds(), var['end_time'].total_seconds()]), np.hstack([k, k]), label=feature, alpha=0.5)
            plt.legend()
            plt.title('target {}'.format(target_col_num))
            plt.pause(0.1)


    def rename_features_prediction_time(self, x, sa):
        """
        Rename features in x such that they contain the relative time w.r.t. the sa step ahead of prediction
        :param x:
        :param sa:
        :return:
        """
        metadata = pd.concat([t.metadata for t in self.transformers])
        target_metadata = pd.concat([t.metadata for t in self.target_transformers])

        # find time of prediction at step ahead sa
        prediction_time = target_metadata.loc[target_metadata['lag'] == -sa]['referring_time']
        assert len(prediction_time) == 1, 'Found {} targets at lag {}. This is not supported'.format(
            len(prediction_time), sa)
        metadata['new_referring_time'] = pd.TimedeltaIndex(metadata['referring_time'].values - prediction_time.values)

        new_names = metadata.reset_index()['index'].apply(lambda x: x.split('_lag')[0]) + metadata['new_referring_time'].apply(
            lambda x: hr_timedelta(x, zero_padding=True)).values
        new_names.index = metadata.index
        metadata.loc[:, 'new_name'] = new_names

        # if feature doesn't have an associated transformation, do not rename it and bypass it
        x.columns = [metadata['new_name'].loc[c] if c in metadata.index else c for c in x.columns]
        return x


    def get_time_lims(self, include_target=False, extremes=True):

        transformers = self.transformers + self.target_transformers if include_target else self.transformers
        if any([t.metadata is None for t in self.transformers]):
            self._simulate_transform()

        min_start_times = pd.DataFrame(pd.concat(
            [t.metadata.groupby('name').min()['start_time'] for t in transformers])).groupby(
            'name').min()

        max_end_times = pd.DataFrame(pd.concat(
            [t.metadata.groupby('name').max()['end_time'] for t in transformers])).groupby(
            'name').max()

        time_lims = pd.concat([min_start_times, max_end_times], axis=1)
        if extremes:
            time_lims = pd.DataFrame([time_lims['start_time'].min(), time_lims['end_time'].max()], index=['start_time', 'end_time']).T
        return time_lims

    def global_form_preprocess(self, x):
        assert np.unique([tr.names for tr in self.target_transformers]) == 'target', 'When using global_form option,' \
                                                                                     ' the only admissible target is' \
                                                                                     ' "target"'
        transformed_columns = [tr.names for tr in self.transformers]
        transformed_columns = [item for sublist in transformed_columns for item in sublist]
        transformed_columns = list(set(np.unique(transformed_columns)) - {'target'})
        # if x is multiindex pd.DataFrame do something
        if isinstance(x.columns, pd.MultiIndex):
            # find columns names at level 0 that contains the targets
            c_l_0 = x.columns.get_level_values(0).unique()
            private_cols_l0 = [c for c in c_l_0 if not np.all([str(t) in transformed_columns for t in x[c].columns])]
            shared_cols_l0 = list(set(c_l_0) - set(private_cols_l0))
            x_shared = x[shared_cols_l0].droplevel(0, 1)
            dfs = []
            for p in private_cols_l0:
                x_p = x[p]
                target_name_l1 = [c for c in x_p.columns if c not in transformed_columns]
                assert len(
                    target_name_l1) == 1, 'something went wrong, there should be only one target column. You must add a transform for all the non-target columns'
                target_name_l1 = target_name_l1[0]
                x_p = x_p.rename({target_name_l1: 'target'}, axis=1)
                dfs.append(pd.concat([x_p, x_shared, pd.DataFrame(p, columns=['name'], index=x.index)], axis=1))
        else:

            independent_targets = [c for c in x.columns if c not in transformed_columns]
            dfs = []
            for c in independent_targets:
                dfs.append(pd.concat(
                    [pd.DataFrame(x[c].rename(), columns=['target']), x[transformed_columns],
                     pd.DataFrame(c, columns=['name'], index=x.index)],
                    axis=1))
        return dfs

    def global_form_postprocess(self, x, y, xs, ys, reduce_memory=False, corr_reorder=False, use_ray=False,
                                parallel=False):

        if reduce_memory:
            x = reduce_mem_usage(x, use_ray=use_ray, parallel=parallel)
            y = reduce_mem_usage(y, use_ray=use_ray, parallel=parallel)

        # for all transformations
        for tr in self.transformers:
            # for all the features of the transformation
            for v in np.unique(tr.metadata.name):
                # reorder columns by correlation with first target
                transformed_cols_names = tr.metadata.loc[tr.metadata['name']==v].index
                if corr_reorder:
                    if y.empty:
                        transformed_cols_names_reordered = tr.reordered_cols_names_dict[v]
                        x.loc[:, transformed_cols_names] = x.loc[:, transformed_cols_names_reordered].values
                    else:
                        corr = x[transformed_cols_names].corrwith(y.iloc[:, 0])
                        transformed_cols_names_reordered = corr.sort_values(ascending=False).index
                        tr.reordered_cols_names_dict[v] = transformed_cols_names_reordered
                        x.loc[:, transformed_cols_names] = x.loc[:, transformed_cols_names_reordered].values
        xs.append(x)
        ys.append(y)
        return xs, ys
class Transformer:
    """
    Defines and applies transformations through rolling time windows and lags
    """
    Anyarray = Union[tuple, np.ndarray, list, None]
    def __init__(self, names, functions:Anyarray=None, agg_freq:Union[str, int, None]=None, lags:Anyarray=None, logger=None,
                 relative_lags:bool=False, agg_bins:Anyarray=None, nested=True, dt=None, name=None):
        """
        :param names: list of columns of the target dataset to which this transformer applies
        :param functions: list of functions
        :param agg_freq: str, frequency of aggregation
        :param lags: negative lag = FUTURE, positive lag = PAST. This is due to the fact that usually we are interested
                     in transformation of variables in the past when we're doing forecasting
        :param logger: auxiliary logger
        :param relative_lags: if True, lags are computed on the base of agg_freq
        :param agg_bins: if agg_bins is not None, ignore agg_freq and produce binned statistics of functions using the
                         values in the agg_bins array for the bins. [0, 2, 5] means two bins, the first one of two
                         elements (0, 1), the second one of 3 elements, (2, 3, 4).
        """
        assert isinstance(functions, list) or functions is None, 'functions must be a list of strings or functions'
        self.dt = dt
        self.name = name
        self.names = names
        self.functions = functions
        self.agg_freq = agg_freq
        self.lags = lags if lags is None else np.atleast_1d(np.array(lags))
        self.relative_lags = relative_lags
        self.logger = get_logger() if logger is None else logger
        self.transform_time = None  # time at which (lagged) transformations refer w.r.t. present time
        self.generated_features = None
        self.metadata = None
        self.reordered_cols_names_dict = {}
        self.nested = nested
        if agg_bins is not None:
            self.original_agg_bins = np.copy(np.sort(agg_bins)[::-1])
            assert np.all(np.diff(agg_bins) <= 0) or np.all(np.diff(agg_bins) >= 0), 'agg bins must be a monotone array'
            self.agg_bins = np.max(agg_bins) - np.sort(agg_bins)[::-1] # descending order
            if nested:
                self.transformers = []
                for i in range(len(self.original_agg_bins) - 1):
                    step_from = self.original_agg_bins[i]
                    step_to = self.original_agg_bins[i + 1]
                    freq = step_from - step_to
                    tr = Transformer(names, functions, agg_freq=freq, lags=[step_to], nested=False,
                                     logger=get_logger(logging.WARNING))
                    self.transformers.append(tr)
        else:
            self.agg_bins = None
            self.original_agg_bins = None


        if agg_freq is not None and agg_bins is not None:
            self.logger.warning('Transformer: agg_freq will be ignored since agg_bins is not None')

    def transform(self, x=None, augment=True, simulate=False):
        """
        Add transformations to the x pd.DataFrame, as specified by the Transformer's attributes

        :param x: pd.DataFrame
        :param augment: if True augments the original x DataFrame, otherwise returns just the transforms DataFrame
        :param simulate: if True do not perform any operation on the dataset, just populate the metadata property
                         with information on the name and time lags
        :return: transformed DataFrame
        """
        if simulate:
            if x is None:
                assert self.dt is not None, "self.dt must be set if you don't pass x while simulating"
        else:
            assert x is not None, "x must be passed if simulate is False"
            assert np.all(np.isin(self.names, x.columns)), "transformers names, {},  " \
                                                           "must be in x.columns:{}".format(self.names, x.columns)


            data = x.copy() if augment else pd.DataFrame(index=x.index)

        self.metadata = pd.DataFrame()
        for name in self.names:
            d = x[name].copy() if x is not None else None

            # infer sampling time
            dt = self.dt if self.dt is not None else x[name].index.to_series().diff().median()

            if self.agg_freq is None:
                self.agg_freq = dt
            elif isinstance(self.agg_freq, (np.int8, np.int32, np.int64, int)):
                self.agg_freq *= dt

            # agg_steps = number of steps on which aggregation stats are retrieved
            # agg_time = time range on which aggregation stats are retrieved
            if self.agg_bins is not None:
                # if agg_bins, the aggregation steps are the maximum specified in bins and lag time is multiple of dt
                agg_steps = np.max(self.agg_bins)-np.min(self.agg_bins)
                spacing_time = dt
            else:
                agg_steps = int(pd.Timedelta(self.agg_freq) / dt)
                spacing_time = dt * agg_steps if self.relative_lags else dt

            trans_names = [name]
            function_names = ['none']
            if self.functions:
                function_names = [get_fun_name(s) for s in self.functions]
                if self.agg_bins is None:
                    hr_agg_freq = self.agg_freq if isinstance(self.agg_freq, str) else hr_timedelta(self.agg_freq.to_timedelta64())
                    trans_names = ['{}_{}_{}'.format(name, hr_agg_freq, p) for p in function_names]

                    if not simulate:
                        d = d.rolling(self.agg_freq, min_periods=agg_steps).agg(self.functions)
                        d.columns = trans_names
                else:
                    if not simulate:
                        if self.nested:
                            trans_names = ['{}_{}_{}_{}'.format(name, p[1], self.original_agg_bins[p[0]],
                                                                self.original_agg_bins[p[0] + 1]) for p in
                                           product(np.arange(len(self.agg_bins) - 1), function_names)]
                            d = pd.DataFrame(d)
                            for i, tr in enumerate(self.transformers):
                                tr.names = [name]
                                d = tr.transform(d)
                            del d[name]
                            d.columns = trans_names
                        else:
                            trans_names = ['{}_{}_{}_{}'.format(name, p[0], self.original_agg_bins[p[1]],
                                                                self.original_agg_bins[p[1] + 1]) for p in
                                           product(function_names, np.arange(len(self.agg_bins) - 1))]
                            partial_res = []
                            for agg_fun in self.functions:
                                reducer = Reducer(bins=self.agg_bins, agg_fun=agg_fun)
                                indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=agg_steps)
                                d_rolled = d.rolling(window=indexer, min_periods=agg_steps).apply(reducer.reduce)
                                partial_res.append(np.vstack(reducer.stats).tolist())
                            d = pd.DataFrame(np.hstack(partial_res), index=d_rolled.index[~d_rolled.isna()], columns=trans_names)
                            d = d.shift(np.max(self.original_agg_bins)-1)
                    else:
                        if self.nested:
                            trans_names = ['{}_{}_{}_{}'.format(name, p[1], self.original_agg_bins[p[0]],
                                                                self.original_agg_bins[p[0] + 1]) for p in
                                           product(np.arange(len(self.agg_bins) - 1), function_names)]
                        else:
                            trans_names = ['{}_{}_{}_{}'.format(name, p[0], self.original_agg_bins[p[1]],
                                                                self.original_agg_bins[p[1] + 1]) for p in
                                           product(function_names, np.arange(len(self.agg_bins) - 1))]

            if self.lags is not None:
                trans_names = ['{}_lag_{:03d}'.format(p[1], p[0]) if p[0] >= 0 else '{}_lag_{:04d}'.format(p[1], p[0])
                               for p in product(self.lags, trans_names)]
                lag_steps = self.lags * agg_steps if self.relative_lags else self.lags
                assert len(lag_steps) == len(self.lags)
                if not simulate:
                    d = pd.concat([d.shift(l) for l in lag_steps], axis=1)
                    d.columns = trans_names
            else:
                lag_steps = np.array([0])
            if not simulate:
                data = pd.concat([data, d], axis=1)
                self.logger.debug('Added {} to the dataframe'.format(trans_names))

            if self.agg_bins is None:
                lags_and_fun = product([None] if self.lags is None else self.lags, function_names)
                lags_aux = np.array([lf[0] for lf in product([0] if self.lags is None else self.lags, function_names)])

                metadata_n = pd.DataFrame(lags_and_fun, columns=['lag', 'function'], index=trans_names)

                metadata_n['aggregation_time'] = self.agg_freq
                metadata_n['spacing_time'] = pd.Timedelta(spacing_time)

                metadata_n['start_time'] = - spacing_time * lags_aux - agg_steps * dt + dt
                metadata_n['end_time'] = - spacing_time * lags_aux + dt
            else:
                lags_expanded = np.outer(lag_steps, np.ones(len(self.agg_bins) - 1)).ravel()
                lags_and_fun =product(function_names, lags_expanded)
                metadata_n = pd.DataFrame(lags_and_fun, columns=['function', 'lag'], index=trans_names)
                metadata_n['aggregation_time'] = self.agg_freq
                metadata_n['spacing_time'] = pd.Timedelta(spacing_time)
                if self.nested:
                    metadata_n['start_time'] = [-dt * (self.original_agg_bins[i] + l) + dt for i, name, l in
                                                product(np.arange(len(self.agg_bins) - 1), function_names, lag_steps)]
                    metadata_n['end_time'] = [-dt * (self.original_agg_bins[i+1]+l) + dt for i, name, l in
                                                product(np.arange(len(self.agg_bins) - 1), function_names, lag_steps)]
                else:
                    metadata_n['start_time'] = [-dt * (self.original_agg_bins[i] + l) + dt for name, i, l in
                                                product(function_names, np.arange(len(self.agg_bins) - 1), lag_steps)]
                    metadata_n['end_time'] = [-dt * (self.original_agg_bins[i+1]+l) + dt for name, i, l in
                                              product(function_names, np.arange(len(self.agg_bins) - 1), lag_steps)]
            metadata_n['name'] = name
            metadata_n['relative_lags'] = self.relative_lags

            self.metadata = pd.concat([self.metadata, metadata_n])
        if not simulate:
            self.generated_features = set(data.columns) - set(x.columns)
            return data
        else:
            return None


class Reducer:
    def __init__(self, agg_fun, bins):
        """
        :param agg_fun: aggregation function
        :param bins: array/list of aggregation bins. [0, 2, 5] means two bins, the first one of two
                         elements (0, 1), the second one of 3 elements, (2, 3, 4).
        """
        self.stats = []
        self.bins = bins
        self.agg_fun = agg_fun

    def reduce(self, x):
        self.stats.append(binned_statistic(np.arange(len(x)), x, self.agg_fun, bins=self.bins).statistic)
        return 0

def hr_timedelta(t, zero_padding=False):
    """
    Timedelta64 to human-readable format
    :param t:
    :return:
    """
    if isinstance(t, pd.Timedelta):
        t = t.to_timedelta64()
    t = t.astype('timedelta64[s]').astype(int)
    sign_t = np.sign(t)
    t = np.abs(t)

    s = t % 60
    days = t // (3600 * 24)
    hours = (t - days * 3600 * 24) // 3600
    minutes = (t - days * 3600 * 24 - hours * 3600) // 60

    if zero_padding:
        time = '{:02d}d'.format(days) \
               + '{:02d}h'.format(hours) \
               + '{:02d}m'.format(minutes) \
               + '{:02d}s'.format(s) * int(s > 0)
    else:
        time = '{}d'.format(days) * int(days > 0)\
               + '{}h'.format(hours) * int(hours > 0)\
               + '{}m'.format(minutes) * int(minutes > 0) \
               + '{}s'.format(s) * int(s > 0)
    time = '-' + time if sign_t == -1 else time
    return time


def get_fun_name(f):
    if isinstance(f, str):
        return f
    elif "__name__" in dir(f):
        return f.__name__
    elif isinstance(f, functools.partial):
        return f.func.__name__
    else:
        raise ValueError('Type of function not recognized. It should either be str, have the __name__ attr, '
                         'or be a functools.partial instance')