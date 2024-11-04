import os
import unittest

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyforecaster.formatter as pyf
import logging
import seaborn as sb
from time import time
import pickle as pk
from pyforecaster.big_data_utils import fdf_parallel
import os


class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.t = 500
        self.n = 10
        self.x = pd.DataFrame(
            np.sin(np.arange(self.t) * 10 * np.pi / self.t).reshape(-1, 1) * np.random.randn(1, self.n),
            index=pd.date_range('01-01-2020', '01-05-2020', self.t, tz='Europe/Zurich'))


        times = pd.date_range('01-01-2020', '01-05-2020', freq='20min')
        self.x2 = pd.DataFrame(
            np.sin(np.arange(len(times)) * 10 * np.pi / len(times)).reshape(-1, 1) * np.random.randn(1, self.n) + np.cumsum(np.random.randn(len(times))).reshape(-1, 1),
            index=times)
        self.x3 = pd.DataFrame((np.arange(len(times)) % 20).reshape(-1,1) * np.random.rand(1, 3), index=times)

        n_steps = 144 * 300
        self.x4 = pd.DataFrame(
            np.sin(np.arange(n_steps) * 10 * np.pi / n_steps).reshape(-1, 1) * np.random.randn(1,
                                                                                                     self.n) + np.cumsum(
                np.random.randn(n_steps)).reshape(-1, 1),
            index=pd.date_range('01-01-2020', '01-05-2020', n_steps))

        self.logger =logging.getLogger()
        logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                            filename=None)

    def test_transformer(self):

        tr = pyf.Transformer([0], ['mean'])
        _ = tr.transform(self.x)
        tr = pyf.Transformer([0], ['mean'], agg_freq='2h' )
        _ = tr.transform(self.x)
        tr = pyf.Transformer([0], ['mean'], agg_freq='2h', lags=[1, 2, -1, -2])
        _ = tr.transform(self.x)
        tr = pyf.Transformer([0, 1], ['mean', 'max'], '3h', [-1, -2])
        _ = tr.transform(self.x)
        tr = pyf.Transformer([0, 1], ['mean', 'max', lambda x: np.sum(x**2)-1], '3h', [-1, -2])
        _ = tr.transform(self.x)
        tr = pyf.Transformer([0], agg_freq='3h', lags=[-1, -2, -3])
        _ = tr.transform(self.x, augment=False)

        tr = pyf.Transformer([0], ['mean'], '3h', [-1])
        x_tr = tr.transform(self.x, augment=False)

        assert x_tr.shape[1] == 1

    def test_time_zone_features(self):
        formatter = pyf.Formatter().add_transform([0, 1, 2, 3], ['mean', 'max'], agg_freq='2h', lags=[-1,-2, -10])
        formatter.add_target_transform([3], lags=np.arange(10))
        x_tr, y_tr = formatter.transform(self.x)
        assert 'utc_offset' in x_tr.columns
        self.x.index = self.x.index.tz_localize(None)
        x_tr, y_tr = formatter.transform(self.x)
        assert 'utc_offset' in x_tr.columns
        assert np.all(x_tr['utc_offset'] == 0)

        formatter = pyf.Formatter().add_transform([0, 1, 2, 3], ['mean', 'max'], agg_freq='2h', lags=[-1,-2, -10])
        formatter.add_target_transform([3], lags=np.arange(10))
        x_tr, y_tr = formatter.transform(self.x)
        assert 'utc_offset' not in x_tr.columns
        x_tr, y_tr = formatter.transform(self.x)
        assert 'utc_offset' not in x_tr.columns
        self.x.index = self.x.index.tz_localize('Europe/Zurich')
        x_tr, y_tr = formatter.transform(self.x)
        assert 'utc_offset' not in x_tr.columns


    def test_formatter(self):
        formatter = pyf.Formatter().add_transform([0, 1, 2, 3], ['mean', 'max'], agg_freq='2h', lags=[-1,-2, -10])
        formatter.add_target_transform([3], lags=np.arange(10))
        formatter.plot_transformed_feature(self.x, 3)
        formatter.plot_transformed_feature(self.x, 0)
        x_tr, y_tr = formatter.transform(self.x)
        assert x_tr.isna().sum().sum() == 0 and y_tr.isna().sum().sum() == 0 and y_tr.shape[0] == x_tr.shape[0]

    def test_tkfcv_simulate_transform(self):
        formatter = pyf.Formatter().add_transform([0, 1, 2, 3], ['mean', 'max'], agg_freq='2h',
                                                                    lags=[-1, -2, -10])
        formatter.add_target_transform([3], lags=np.arange(10))
        folds_df = formatter.time_kfcv(self.x.index, 4, 3, x=self.x)
        for dfi in folds_df:
            assert np.sum(dfi[0]) + np.sum(dfi[1]) < len(self.x) -1

    def test_tkfcv_pretransform(self):
        formatter = pyf.Formatter().add_transform([0, 1, 2, 3], ['mean', 'max'], agg_freq='2h', lags=[-1,-2, -10], relative_lags=True)
        formatter.add_target_transform([3], lags=np.arange(10))
        formatter.transform(self.x2)
        folds_df = formatter.time_kfcv(self.x2.index, 4, 3)
        for dfi in folds_df:
            assert np.sum(dfi[0]) + np.sum(dfi[1]) < len(self.x2) -1

    def test_prune_at_stepahead(self):
        formatter = pyf.Formatter().add_transform([0, 1, 2, 3], ['mean', 'max'], agg_freq='2h',
                                                                    lags=np.arange(24*3), relative_lags=False)
        formatter.add_target_transform([3], lags=-np.arange(10)-1)
        x_transformed, y_transformed = formatter.transform(self.x2)
        crosspattern = pd.DataFrame()
        for i in range(10):
            x_i = formatter.prune_dataset_at_stepahead(x_transformed, i, metadata_features=[], method='periodic', period='24h', tol_period='10m')
            crosspattern = crosspattern.combine_first(pd.DataFrame(1, index=x_i.columns, columns=[i]))
        sb.heatmap(crosspattern)

    def test_nonanticipativity(self):
        """
        You should see 30 steps of future green target following two series of 2 points.
        :return:
        """
        formatter = pyf.Formatter().add_transform([0, 1, 2], ['mean', 'min'], agg_freq='80min', lags=np.arange(2), relative_lags=True)
        formatter.add_target_transform([2], lags=-np.arange(30)-1)
        x_transformed, y_transformed = formatter.transform(self.x3)
        formatter.plot_transformed_feature(self.x3, 2)
        plt.close('all')

    def test_aggregated_transform(self):
        """
        You should see min-max transform enveloping the signal
        :return:
        """
        formatter = pyf.Formatter().add_transform([0], lags=np.arange(10), agg_freq='20min',
                                                                    relative_lags=True)
        formatter.add_transform([0], ['min', 'max'], agg_bins=[-10, -15, -20], nested=False)
        formatter.add_target_transform([0], lags=-np.arange(30)-1)
        x_transformed, y_transformed = formatter.transform(self.x2)
        #formatter.plot_transformed_feature(self.x2, 0, frames=100)

        # test nested transform (much faster)
        formatter = pyf.Formatter().add_transform([0], lags=np.arange(10), agg_freq='20min',
                                                                    relative_lags=True)
        formatter.add_transform([0], ['min', 'max'], agg_bins=[-10, -15, -20], nested=True)
        formatter.add_target_transform([0], lags=-np.arange(30) - 1)
        x_transformed_nested, y_transformed = formatter.transform(self.x2)
        #formatter.plot_transformed_feature(self.x2, 0, frames=100)


        formatter = pyf.Formatter().add_transform([0], ['min', 'max'], agg_bins=[0, 1, 4, 10])
        formatter.add_transform([0], ['min', 'max'], agg_bins=[-10, -15, -20])
        formatter.add_transform([0], ['mean'], lags=np.arange(10), agg_freq='20min', relative_lags=True)
        formatter.add_target_transform([0], lags=-np.arange(30)-1)
        x_transformed, y_transformed = formatter.transform(self.x3)
        #formatter.plot_transformed_feature(self.x3, 0, frames=30)

        x_transformed, y_transformed = formatter.transform(self.x2)
        #formatter.plot_transformed_feature(self.x2, 0, frames=100)
        plt.close('all')

    def test_speed(self):
        t0 = time()
        formatter = pyf.Formatter().add_transform([0], ['mean'], lags=np.arange(10),
                                                                    relative_lags=True)
        formatter.add_transform([0], ['min', 'max'], agg_bins=np.hstack([144*7-np.arange(6)*144, 144-np.arange(int(144/6)+1)*6]))
        x_transformed, y_transformed = formatter.transform(self.x4)
        print('elapsed time to transform {} points into a {}x{} df: {}s '.format(self.x4.shape[0], *x_transformed.shape, time()-t0))


    def test_simulate_formatter(self):
        formatter = pyf.Formatter( dt = pd.Timedelta('15min')).add_transform([0], ['mean'], lags=np.arange(10),                                                                    relative_lags=True)
        formatter.add_transform([0, 1, 2], ['min', 'max'], agg_bins=np.hstack([144*7-np.arange(6)*144, 144-np.arange(int(144/6)+1)*6]))

        time_lims = formatter.get_time_lims(include_target=True, extremes=True)

        assert formatter.transformers[0].metadata is not None


    def test_pickability(self):
        """
        Pickability is needed to parallelize formatting operations
        """
        temp_file_path = 'test_pickle_temp.pk'
        formatter = pyf.Formatter().add_transform([0], lags=np.arange(10), agg_freq='20min',
                                                                    relative_lags=True)
        formatter.add_transform([0], ['min', 'max'], agg_bins=[-10, -15, -20])
        formatter.add_target_transform([0], lags=-np.arange(30)-1)

        with open(temp_file_path, 'wb') as f:
            pk.dump(formatter, f, protocol=pk.HIGHEST_PROTOCOL)

        with open(temp_file_path, 'rb') as f:
            formatter_pickled = pk.load(f)

        x_transformed, y_transformed = formatter_pickled.transform(self.x2)
        formatter_pickled.plot_transformed_feature(self.x2, 0, frames=100)

        with open('test_pickle_temp.pk', 'wb') as f:
            pk.dump(formatter, f, protocol=pk.HIGHEST_PROTOCOL)

        with open(temp_file_path, 'rb') as f:
            formatter_pickled = pk.load(f)
        x_transformed, y_transformed = formatter_pickled.transform(self.x2)
        formatter_pickled.plot_transformed_feature(self.x2, 0, frames=100)
        os.remove(temp_file_path)

    def test_parallel_transformations(self):
        formatter = pyf.Formatter().add_transform([0], lags=np.arange(10), agg_freq='20min',
                                                                    relative_lags=True)
        formatter.add_transform([0], ['min', 'max'], agg_bins=[-10, -15, -20])
        formatter.add_target_transform([0], lags=-np.arange(30)-1)

        print(self.x2.shape)
        # This gives some problems with the CI github actions tests. Uncomment if you want to test
        #res = fdf_parallel(f=formatter.transform, df=[self.x2, self.x2, self.x2, self.x2])
        #print(res[0].shape)
        #print(res[1].shape)

    def test_holidays(self):
        formatter = pyf.Formatter().add_transform([0], lags=np.arange(10), agg_freq='20min',
                                                                    relative_lags=True)
        formatter.add_transform([0], ['min', 'max'], agg_bins=[-10, -15, -20])
        df = formatter.transform(self.x2, time_features=True, holidays=True, subdiv='ZH')



    def test_global_multiindex(self):
        x_private = pd.DataFrame(np.random.randn(500, 15), index=pd.date_range('01-01-2020', '01-05-2020', 500, tz='Europe/Zurich'), columns=pd.MultiIndex.from_product([['b1', 'b2', 'b3'], ['a', 'b', 'c', 'd', 'e']]))
        x_shared =  pd.DataFrame(np.random.randn(500, 5), index=pd.date_range('01-01-2020', '01-05-2020', 500, tz='Europe/Zurich'), columns=pd.MultiIndex.from_product([['shared'], [0, 1, 2, 3, 4]]))

        df_mi = pd.concat([x_private, x_shared], axis=1)

        formatter = pyf.Formatter().add_transform([0,1 , 2, 3, 4], lags=np.arange(10), agg_freq='20min',
                                                                    relative_lags=True)
        formatter.add_transform(['a','b', 'c', 'd'], lags=np.arange(10),
                                                                    agg_freq='20min',
                                                                    relative_lags=True)
        formatter.add_target_transform(['target'], ['mean'], agg_bins=[-10, -15, -20])
        df = formatter.transform(df_mi, time_features=True, holidays=True, subdiv='ZH',global_form=True, parallel=False)

    def test_global_multiindex_with_col_reordering(self):
        x_private = pd.DataFrame(np.random.randn(500, 15), index=pd.date_range('01-01-2020', '01-05-2020', 500, tz='Europe/Zurich'), columns=pd.MultiIndex.from_product([['b1', 'b2', 'b3'], ['a', 'b', 'c', 'd', 'e']]))
        x_shared =  pd.DataFrame(np.random.randn(500, 5), index=pd.date_range('01-01-2020', '01-05-2020', 500, tz='Europe/Zurich'), columns=pd.MultiIndex.from_product([['shared'], [0, 1, 2, 3, 4]]))

        df_mi = pd.concat([x_private, x_shared], axis=1)

        formatter = pyf.Formatter().add_transform([0,1 , 2, 3, 4], lags=np.arange(10), agg_freq='20min',
                                                                    relative_lags=True)
        formatter.add_transform(['a','b', 'c', 'd'], lags=np.arange(10),
                                                                    agg_freq='20min',
                                                                    relative_lags=True)
        formatter.add_target_transform(['target'], ['mean'], agg_bins=[-10, -15, -20])
        df = formatter.transform(df_mi, time_features=True, holidays=True, subdiv='ZH',global_form=True, corr_reorder=True, parallel=False ,reduce_memory=False)


    def test_normalizers(self):
        df = pd.DataFrame(np.random.randn(100, 5), index=pd.date_range('01-01-2020', freq='20min', periods=100, tz='Europe/Zurich'), columns=['a', 'b', 'c', 'd', 'e'])
        formatter = pyf.Formatter().add_transform(['a', 'b'], lags=np.arange(1, 5), agg_freq='20min')
        formatter.add_target_transform(['a'], lags=-np.arange(1, 5), agg_freq='20min')
        formatter.add_target_normalizer(['a'], 'mean', agg_freq='10H', name='a_movingavg')
        formatter.add_target_normalizer(['a'], 'std', agg_freq='10H', name='a_movingstd')
        x, y = formatter.transform(df, time_features=True, holidays=True, subdiv='ZH')

        formatter.add_normalizing_fun(expr="(df[t] - df['a_movingavg']) / (df['a_movingstd'] + 1)",
                                      inv_expr="df[t]*(df['a_movingstd']+1) + df['a_movingavg']")
        x, y_norm = formatter.transform(df, time_features=True, holidays=True, subdiv='ZH')

        y_unnorm = formatter.denormalize(x, y_norm)

        # check if back-transform works
        assert (y_unnorm-y).sum().sum() < 1e-6


    def test_normalizers_complex(self):
        df = pd.DataFrame(np.random.randn(100, 5), index=pd.date_range('01-01-2020', freq='20min', periods=100, tz='Europe/Zurich'), columns=['a', 'b', 'c', 'd', 'e'])
        formatter = pyf.Formatter(augment=False).add_transform(['a', 'b'], lags=np.arange(1, 5), agg_freq='20min')
        formatter.add_target_transform(['a'], lags=-np.arange(1, 5), agg_freq='20min')
        formatter.add_target_normalizer(['a'], 'mean', agg_freq='10H', name='a_n')
        formatter.add_target_normalizer(['a'], 'std', agg_freq='5H', name='b_n')

        x, y = formatter.transform(df, time_features=True, holidays=True, subdiv='ZH')

        formatter.add_normalizing_fun(expr="np.exp(df[t]+df['a_n']) + df['b_n']", inv_expr="np.log(df[t]-df['b_n']) -df['a_n']")
        x, y_norm = formatter.transform(df, time_features=True, holidays=True, subdiv='ZH')
        y_unnorm = formatter.denormalize(x, y_norm)

        # check if back-transform works
        assert (y_unnorm-y).abs().sum().sum() < 1e-6


    def test_normalizers_impossible(self):
        x_private = pd.DataFrame(np.random.randn(100, 15),
                                 index=pd.date_range('01-01-2020', '01-05-2020', 100, tz='Europe/Zurich'),
                                 columns=pd.MultiIndex.from_product([['b1', 'b2', 'b3'], ['a', 'b', 'c', 'd', 'e']]))
        x_shared = pd.DataFrame(np.random.randn(100, 5),
                                index=pd.date_range('01-01-2020', '01-05-2020', 100, tz='Europe/Zurich'),
                                columns=pd.MultiIndex.from_product([['shared'], [0, 1, 2, 3, 4]]))

        df_mi = pd.concat([x_private, x_shared], axis=1)

        formatter = pyf.Formatter().add_transform([0, 1, 2, 3, 4], lags=np.arange(10), agg_freq='20min',
                                                  relative_lags=True)
        formatter.add_transform(['a', 'b', 'c', 'd'], lags=np.arange(10),
                                agg_freq='20min',
                                relative_lags=True)
        formatter.add_target_transform(['target'], ['mean'], agg_bins=[-10, -15, -20])

        formatter.add_target_normalizer(['target'], 'mean', agg_freq='10H', name='mean')
        formatter.add_target_normalizer(['target'], 'std', agg_freq='5H', name='std')

        x, y = formatter.transform(df_mi, time_features=True, holidays=True, subdiv='ZH',global_form=True)
        formatter.add_normalizing_fun("(df[t] - df['mean'])/(df['std']+1)", "df[t]*(df['std']+1) + df['mean']")
        x, y_norm = formatter.transform(df_mi, time_features=True, holidays=True, subdiv='ZH',global_form=True)

        xs = formatter.global_form_preprocess(df_mi)

if __name__ == '__main__':
    unittest.main()
