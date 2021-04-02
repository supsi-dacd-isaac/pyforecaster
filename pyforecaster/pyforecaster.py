import numpy as np
import pandas as pd
from itertools import product


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
        self.names = names
        self.functions = functions
        self.agg_freq = agg_freq
        self.lags = lags
        self.logger = logger

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
            if self.functions:
                d = x[name].copy()
                inferred_freq = d.index.inferred_freq
                inferred_freq = inferred_freq if any([i.isdigit() for i in inferred_freq]) else '1' + inferred_freq
                agg_freq = self.agg_freq if self.agg_freq else inferred_freq
                min_periods = int(pd.Timedelta(agg_freq) / pd.Timedelta(inferred_freq))
                d = d.rolling(agg_freq, min_periods=min_periods).agg(self.functions)
                names = ['{}_{}{}'.format(name, agg_freq.upper(), p) for p in d.columns]
                d.columns = names
            else:
                d = x[[name]].copy()

            if self.lags:
                names = ['{}_lag_{}'.format(p[1], p[0]) for p in product(self.lags, d.columns)]
                d = pd.concat([d.shift(l) for l in self.lags], axis=1)
                d.columns = names
            if self.logger:
                self.logger.info('Added {} to the dataframe'.format(d.columns.tolist()))
            data = pd.concat([data, d], axis=1)
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
    target = target_transformer.transform(x)
    for t in args:
        out = t.transform(x)
        x = pd.concat([x, out], axis=1)

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