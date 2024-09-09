import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pyforecaster.forecaster import LinearForecaster
from pyforecaster.forecasting_models.gradientboosters import LGBMHybrid

from pyforecaster.formatter import Formatter
from pyforecaster.metrics import rmse

df = pd.read_pickle('tests/data/test_data.zip').droplevel(0, 1)

formatter = Formatter().add_transform(['all'], lags=np.flip(np.arange(24)), relative_lags=True)
formatter.add_transform(['all'], ['min', 'max'], agg_bins=np.flip([1, 2, 15, 20]))
formatter.add_target_transform(['all'], lags=-np.arange(6))

x, y = formatter.transform(df)

x.columns = x.columns.astype(str)
y.columns = y.columns.astype(str)

data = pd.concat([x, y], axis=1)
normalized_data = pd.DataFrame(StandardScaler().fit(data).transform(data.copy()), index=data.index, columns=data.columns)

discrepancy = ((normalized_data-normalized_data.mean())**2).sum(axis=1)
plt.hist(discrepancy, bins=100)

worst = discrepancy.sort_values(ascending=False).index[np.arange(10)]
median = discrepancy.sort_values().index[len(discrepancy)//2:len(discrepancy)//2+10]

plt.figure()
plt.plot(normalized_data.sample(100).T, color='blue', alpha=0.1)
plt.plot(normalized_data.loc[median].T, color='green')
plt.plot(normalized_data.loc[worst].T, color='black')


remove_ratio = np.linspace(0.001, 0.02, 10)
scores = {}
sorted_by_discrepancy = discrepancy.sort_values(ascending=False).index
for rr in remove_ratio:
    print(rr)
    n_tr = int(len(x)*0.8)
    x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                                y.iloc[n_tr:].copy()]
    internal_remove_ratio = int(len(x_tr)*rr)
    x_tr = x_tr.loc[~x_tr.index.isin(sorted_by_discrepancy[:internal_remove_ratio])]
    y_tr = y_tr.loc[~y_tr.index.isin(sorted_by_discrepancy[:internal_remove_ratio])]
    m = LGBMHybrid(formatter=formatter, n_single=y_tr.shape[1]).fit(x_tr, y_tr)
    y_hat = m.predict(x_te)
    y_hat.columns = y_te.columns
    scores[rr] = rmse(y_te, y_hat).mean(axis=1)
    print('rmse: {}'.format(scores[rr]))

plt.figure()
plt.plot(pd.Series(scores), marker='o')