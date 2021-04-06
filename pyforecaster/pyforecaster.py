import numpy as np
import pandas as pd
from itertools import product
from pyforecaster.plot_utils import ts_animation


class Formatter:
    def __init__(self, logger=None):
        self.logger = logger
        self.transformers = []
        self.target_transformers = []

    def add_transform(self, names, functions=None, agg_freq=None, lags=None):
        transformer = Transformer(names, functions=functions, agg_freq=agg_freq, lags=lags, logger=self.logger)
        self.transformers.append(transformer)
        return self

    def add_target_transform(self, names, functions=None, agg_freq=None, lags=None):
        transformer = Transformer(names, functions=functions, agg_freq=agg_freq, lags=lags, logger=self.logger)
        self.target_transformers.append(transformer)
        return self

    def transform(self, x):
        if self.logger:
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

        return x, target

    def plot_transformed_feature(self, x, feature):
        x_tr, target = self.transform(x)
        all_feats = pd.concat([x_tr, target], axis=1)
        fs = []
        ts = []
        names = []
        for t in self.transformers + self.target_transformers:
            if feature in t.transform_dict.keys():
                for n in t.transform_dict[feature]['names']:
                    fs.append(all_feats[n].values)
                    ts.append(t.transform_dict[feature]['times'])

        ani = ts_animation(fs, ts, names)
        return ani



class Transformer:
    """
    Defines and applies transformations through rolling time windows and lags
    """
    def __init__(self, names, functions=None, agg_freq=None, lags=None, logger=None):
        """
        :param names: list of columns of the target dataset to which this transformer applies
        :param functions:
        :param agg_freq:
        :param lags:
        :param logger: auxiliary logger
        """
        assert isinstance(functions, list) or functions is None, 'functions must be a list of strings or functions'
        self.names = names
        self.functions = functions
        self.agg_freq = agg_freq
        self.lags = lags
        self.logger = logger
        self.transform_time = None  # time at which (lagged) transformations refer w.r.t. present time
        self.generated_features = None
        self.transform_dict = {}

    def transform(self, x, augment=True):
        """
        Add transformations to the x pd.DataFrame, as specified by the Transformer's attributes

        :param x: pd.DataFrame
        :param augment: if True augments the original x DataFrame, otherwise returns just the transforms DataFrame
        :return: transformed DataFrame
        """
        assert np.all(np.isin(self.names, x.columns)), "transformers names, {},  " \
                                                       "must be in x.columns:{}".format(self.names, x.columns)

        data = x.copy() if augment else pd.DataFrame(index=x.index)

        for name in self.names:
            #inferred_freq = x[name].index.inferred_freq
            inferred_freq = str(int(x[name].index.to_series().diff().median().total_seconds())) + 'S'
            inferred_freq = inferred_freq if any([i.isdigit() for i in inferred_freq]) else '1' + inferred_freq
            d = x[name].copy()
            if self.functions:
                agg_freq = self.agg_freq if self.agg_freq else inferred_freq
                min_periods = int(pd.Timedelta(agg_freq) / pd.Timedelta(inferred_freq))
                d = d.rolling(agg_freq, min_periods=min_periods).agg(self.functions)
                trans_names = ['{}_{}{}'.format(name, agg_freq.upper(), p) for p in d.columns]
                d.columns = trans_names

            if self.lags is not None:
                lagged_names = ['{}_lag_{}'.format(p[1], p[0]) for p in
                                product(self.lags, d.columns if isinstance(d, pd.DataFrame) else [name])]
                d = pd.concat([d.shift(l) for l in self.lags], axis=1)
                d.columns = lagged_names
            d.columns = ['f'+n for n in d.columns]

            self.transform_dict[name] = {'names': [[n for n in d.columns if tn in n] for tn in
                                                   (trans_names if self.functions else [str(name)])],
                                         'times': pd.TimedeltaIndex(
                                             [l * pd.Timedelta(self.agg_freq if self.agg_freq else inferred_freq) for l
                                              in
                                              (self.lags if self.lags is not None else [1])])}
            if self.logger:
                self.logger.info('Added {} to the dataframe'.format(d.columns.tolist()))
            data = pd.concat([data, d], axis=1)



        self.generated_features = set(data.columns) - set(x.columns)
        return data


def format_dataset(x, target_transformer, *args, logger=None):
    """
    Apply several Transforsmers instances to the x pd.DataFrame to create the feature DataFrame, and target_transformer
    to create the target DataFrame

    :param x: pd.DataFrame
    :param target_transformer: target's transformer. Usually just specified by lags
    :param args: indefinite list of different transformers generating the features' DataFrame
    :param logger: auxiliary logger
    :return: feaures' and target's DataFrames
    """
    target = target_transformer.transform(x, augment=False)
    for t in args:
        x = t.transform(x)

    x = x.loc[~np.any(x.isna(), axis=1) & ~np.any(target.isna(), axis=1)]
    target = target.loc[~np.any(x.isna(), axis=1) & ~np.any(target.isna(), axis=1)]
    return x, target


def tr_te_split(data: pd.DataFrame, split_ratio: float=0.75):
    """
    Divides data into training and test sets
    :param data:
    :param split_ratio:
    :return:
    """
    n_tr = int(len(data)*split_ratio)
    return data.iloc[:n_tr, :], data.iloc[n_tr:, :]