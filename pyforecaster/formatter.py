#import category_encoders
import numpy as np
import pandas as pd
from itertools import product
from pyforecaster.plot_utils import ts_animation
import logging


def get_logger(level=logging.INFO):

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s')
    logger.setLevel(level)
    return logger


class Formatter:
    def __init__(self, logger=None):
        self.logger = get_logger() if logger is None else logger
        self.transformers = []
        self.fold_transformers = []
        self.target_transformers = []
        self.cv_gen = []

    def add_time_features(self, x):
        self.logger.info('Adding time features')
        x.loc[:, 'hour'] = x.index.hour
        x.loc[:, 'dayofweek'] = x.index.dayofweek
        x.loc[:,'minuteofday'] = x.index.hour * 60 + x.index.minute
        return x

    def add_transform(self, names, functions=None, agg_freq=None, lags=None, relative_lags=False):
        transformer = Transformer(names, functions=functions, agg_freq=agg_freq, lags=lags, logger=self.logger,
                                  relative_lags=relative_lags)
        self.transformers.append(transformer)
        return self

    def add_target_transform(self, names, functions=None, agg_freq=None, lags=None, relative_lags=False):
        if np.any(lags>0):
            self.logger.critical('some lags are positive, which mean you are adding a target in the past. '
                                 'Is this intended?')
        transformer = Transformer(names, functions=functions, agg_freq=agg_freq, lags=lags, logger=self.logger,
                                  relative_lags=relative_lags)
        self.target_transformers.append(transformer)
        return self

    def transform(self, x):
        """
        Takes the DataFrame x and applies the specified transformations stored in the transformers in order to obtain
        the pre-fold-transformed dataset: this dataset has the correct final dimensions, but fold-specific
        transformations like min-max scaling or categorical encoding are not yet applied.
        :param x: pd.DataFrame
        :return x, target: the transformed dataset and the target DataFrame with correct dimensions
        """
        if np.any(x.isna()):
           self.logger.warning('There are {} nans in x, nans are not supported yet, '
                               'get over it. I have more important things to do.'.format(x.isna().sum()))

        for tr in self.transformers:
            x = tr.transform(x)

        target = pd.DataFrame(index=x.index)
        for tr in self.target_transformers:
           target = pd.concat([target, tr.transform(x, augment=False)], axis=1)

        # remove raws with nans to reconcile impossible dataset entries introduced by shiftin' around
        x = x.loc[~np.any(x.isna(), axis=1) & ~np.any(target.isna(), axis=1)]
        target = target.loc[~np.any(x.isna(), axis=1) & ~np.any(target.isna(), axis=1)]

        # adding time features
        x = self.add_time_features(x)
        return x, target


    def _simulate_transform(self, x):
        """
        This won't actually modify the dataframe, it will just populate the metqdata property of each transformer
        :param x:
        :return:
        """
        for tr in self.transformers:
            _ = tr.transform(x, simulate=True)
        for tr in self.target_transformers:
            _ = tr.transform(x, simulate=True)

    def plot_transformed_feature(self, x, feature):
        x_tr, target = self.transform(x)
        all_feats = pd.concat([x_tr, target], axis=1)
        fs = []
        ts = []
        names = []
        for t in self.transformers + self.target_transformers:
            derived_features = t.metadata.loc[t.metadata['name'] == feature]
            if len(derived_features) > 0:
                for function in derived_features['function'].unique():
                    fs.append(all_feats[derived_features.index[derived_features['function']==function]].values)
                    ts.append(t.metadata.loc[derived_features.index[derived_features['function']==function], 'referring_time'])

        ani = ts_animation(fs, ts, names)
        return ani

    def cat_encoding(self, df_train, df_test, target, encoder=None, cat_features=None):
        if cat_features is None:
            self.logger.info('auto encoding categorical features')
            cat_features = self.auto_cat_feature_selection(df_train)

        self.logger.info('encoding {} as categorical features'.format(cat_features))
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
        split_times = pd.date_range(t_start, t_end, n_k+1)
        time_window = split_times[1]-split_times[0]
        shift_times = pd.date_range(t_start, t_end - time_window, n_k * multiplier_factor + 1)

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

        deadband_time = np.max(np.hstack([t.metadata['referring_time'].abs().max() for t in self.transformers]))
        self.logger.info('deadband time: {}'.format(deadband_time))
        if deadband_time > split_times[1]-split_times[0]:
            self.logger.error('deadband time: {} is bigger than test fold timespan: {}. Try '
                              'lowering k'.format(deadband_time, split_times[1]-split_times[0]))
            raise
        self.logger.info('adding {} test folds with length {}'.format(n_k*multiplier_factor, time_window))
        folds = {}
        for i, t_i in enumerate(shift_times):
            test_window = [t_i, time_window + t_i]
            tr_idxs = (time_index < test_window[0] - deadband_time) | (time_index > test_window[1] + deadband_time)
            te_idx = (time_index >= test_window[0]) & (time_index <= test_window[1])
            self.logger.info(self.print_fold(tr_idxs, te_idx))
            yield tr_idxs, te_idx
            #folds['fold_{:>02d}'.format(i)] = pd.DataFrame({'tr':tr_idxs,  'te': te_idx}, index=time_index)
        #return pd.concat(folds, axis=1)

    def cv_generator(self, folds):
        pass

    @staticmethod
    def print_fold(tr_idxs, te_idxs):
        buffers = ~(tr_idxs + te_idxs)
        if sum(buffers) > 0:
            sampling_step = int(np.max(np.convolve(~(tr_idxs + te_idxs), np.ones(sum(buffers)))))
            sampling_step = np.minimum(sampling_step, int(len(tr_idxs) / 100))
        else:
            sampling_step = int(len(tr_idxs) / 100)
        fold = ''
        for s in tr_idxs[::sampling_step].astype(int) + 2*te_idxs[::sampling_step].astype(int):
            fold += 'x' * (s == 0) + '-' * (s == 1) + '|' * (s == 2)
        fold += '   -: train, |=test, x:skip'
        return fold

    def prune_dataset_at_stepahead(self, df, sa, metadata_features, method='periodic', period='24H', tol_period='1H', sa_end=None, keep_last_n_lags=0):

        features = []
        # retrieve referring_time of the given sa for the target from target_transformers
        target_times = []
        for tt in self.target_transformers:
            target_times.append(tt.metadata.loc[tt.metadata['lag']==-sa, 'referring_time'])

        assert len(np.unique(target_times)) == 1, 'target for step ahead {} have different referring_times, ' \
                                             'not supported'.format(sa)
        target_time = pd.to_timedelta(target_times[0])[0]

        if method == 'periodic':
            # find signals with (target_time - referring_time) multiple of period
            for t in self.transformers:
                features += list(t.metadata.index[((target_time-t.metadata['referring_time']).abs() % pd.Timedelta(period)
                                                  <= pd.Timedelta(tol_period)) & (t.metadata['referring_time']<=target_time)])
        elif method == 'up_to':
            for t in self.transformers:
                features += list(t.metadata.index[t.metadata['referring_time'] <= target_time])

        if keep_last_n_lags > 0:
            last_lag_features = list(np.hstack([t.metadata.index[t.metadata['lag'].isin(np.arange(keep_last_n_lags))]
                                           for t in self.transformers]))
            features = np.unique(features + last_lag_features)

        features = np.unique(list(features) + metadata_features)
        return df[features]

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

class Transformer:
    """
    Defines and applies transformations through rolling time windows and lags
    """
    def __init__(self, names, functions=None, agg_freq=None, lags=None, logger=None, relative_lags=False):
        """
        :param names: list of columns of the target dataset to which this transformer applies
        :param functions:
        :param agg_freq:
        :param lags: negative lag = FUTURE, positive lag = PAST
        :param logger: auxiliary logger
        :param relative_lags: if True, lags are computed on the base of agg_freq
        """
        assert isinstance(functions, list) or functions is None, 'functions must be a list of strings or functions'
        self.names = names
        self.functions = functions
        self.agg_freq = agg_freq
        self.lags = lags
        self.relative_lags = relative_lags
        self.logger = get_logger() if logger is None else logger
        self.transform_time = None  # time at which (lagged) transformations refer w.r.t. present time
        self.generated_features = None
        self.metadata = None


    def transform(self, x, augment=True, simulate=False):
        """
        Add transformations to the x pd.DataFrame, as specified by the Transformer's attributes

        :param x: pd.DataFrame
        :param augment: if True augments the original x DataFrame, otherwise returns just the transforms DataFrame
        :param simulate: if True do not perform any operation on the dataset, just populate the metadata property
                         with information on the name and time lags
        :return: transformed DataFrame
        """
        assert np.all(np.isin(self.names, x.columns)), "transformers names, {},  " \
                                                       "must be in x.columns:{}".format(self.names, x.columns)

        data = x.copy() if augment else pd.DataFrame(index=x.index)
        self.metadata = pd.DataFrame()
        for name in self.names:
            lag_time = x[name].index.to_series().diff().median()

            if self.agg_freq is None:
                self.agg_freq = lag_time

            d = x[name].copy()
            min_periods = int(pd.Timedelta(self.agg_freq) / lag_time)

            trans_names = [name]
            function_names = ['none']
            if self.functions:
                function_names = [s if isinstance(s, str) else s.__name__ for s in self.functions]
                hr_agg_freq = self.agg_freq if isinstance(self.agg_freq, str) else hr_timedelta(self.agg_freq.to_timedelta64())
                trans_names = ['{}_{}_{}'.format(name, hr_agg_freq, p) for p in function_names]

                if not simulate:
                    d = d.rolling(self.agg_freq , min_periods=min_periods).agg(self.functions)
                    d.columns = trans_names

            if self.lags is not None:
                self.lags = np.array(self.lags)
                trans_names = ['{}_lag_{:03d}'.format(p[1], p[0]) if p[0] >= 0 else '{}_lag_{:04d}'.format(p[1], p[0])
                               for p in product(self.lags, trans_names)]
                lags = self.lags * min_periods if self.relative_lags else self.lags
                assert len(lags) == len(self.lags)
                if self.relative_lags:
                    lag_time *= min_periods
                if not simulate:
                    d = pd.concat([d.shift(l) for l in lags], axis=1)
                    d.columns = trans_names

            if not simulate:
                data = pd.concat([data, d], axis=1)
                self.logger.info('Added {} to the dataframe'.format(trans_names))

            lags_and_fun = product([0] if self.lags is None else self.lags, function_names)

            metadata_n = pd.DataFrame(lags_and_fun, columns=['lag', 'function'], index=trans_names)
            metadata_n['aggregation_time'] = self.agg_freq
            metadata_n['lag_time'] = pd.Timedelta(lag_time)
            metadata_n['referring_time'] = - metadata_n['lag_time'] * metadata_n['lag']
            metadata_n['name'] = name

            self.metadata = pd.concat([self.metadata, metadata_n])

        self.generated_features = set(data.columns) - set(x.columns)
        return data


def hr_timedelta(t, zero_padding=False):
    """
    Timedelta64 to human readable format
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
               + '{:02d}s'.format(s) * (s > 0)
    else:
        time = '{}d'.format(days) * (days > 0)\
               + '{}h'.format(hours) * (hours > 0)\
               + '{}m'.format(minutes) * (minutes > 0) \
               + '{}s'.format(s) * (s > 0)
    time = '-' + time if sign_t == -1 else time
    return time