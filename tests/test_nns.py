import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from pyforecaster.forecasting_models.neural_forecasters import NN
from pyforecaster.formatter import Formatter
from os import makedirs
from os.path import exists


class TestFormatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.read_pickle('tests/data/test_data.zip').droplevel(0, 1)
        self.logger =logging.getLogger()
        logging.basicConfig(format='%(asctime)-15s::%(levelname)s::%(funcName)s::%(message)s', level=logging.INFO,
                            filename=None)

    def test_ffnn(self):
        formatter = Formatter(logger=self.logger).add_transform(['all'], lags=np.arange(144),
                                                                    relative_lags=True)
        formatter.add_transform(['all'], ['min', 'max'], agg_bins=[1, 2, 15, 20])
        formatter.add_target_transform(['all'], lags=-np.arange(144))

        x, y = formatter.transform(self.data.iloc[:40000])
        # normalize inputs
        x = (x - x.mean(axis=0)) / (x.std(axis=0)+0.01)
        y = (y - y.mean(axis=0)) / (y.std(axis=0)+0.01)

        n_tr = int(len(x) * 0.8)
        x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                  y.iloc[n_tr:].copy()]

        savepath_tr_plots = 'tests/results/ffnn_tr_plots'

        # if not there, create directory savepath_tr_plots
        if not exists(savepath_tr_plots):
            makedirs(savepath_tr_plots)

        parameters = {"n_hidden":255, "n_out":y_tr.shape[1], "n_layers":3, "batch_size":1000, "learning_rate":1e-3}

        optimization_vars = x_tr.columns[:5]
        m = NN(learning_rate=1e-4,  batch_size=1000, load_path=None, n_hidden_x=200, n_hidden_y=200,
               n_out=y_tr.shape[1], n_layers=3, optimization_vars=optimization_vars).train(x_tr,
                                                                                           y_tr, x_te, y_te,
                                                                                           n_epochs=10,
                                                                                           savepath_tr_plots=savepath_tr_plots,
                                                                                           stats_step=40)




if __name__ == '__main__':
    unittest.main()

