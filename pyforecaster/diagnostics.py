import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from scipy.interpolate import griddata
from pyforecaster.formatter import Formatter
from pyforecaster.forecaster import LinearForecaster, LGBForecaster

data = pd.read_pickle('tests/data/test_data.zip').droplevel(0, 1)
data = data.resample('1h').mean()
formatter = Formatter().add_transform(['all'], lags=np.arange(24), relative_lags=True)
formatter.add_transform(['all'], ['min', 'max'], agg_bins=[1, 2, 15, 20])
formatter.add_target_transform(['all'], lags=-np.arange(1, 12))
x, y = formatter.transform(data)
n_tr = int(len(x) * 0.8)
x_tr, x_te, y_tr, y_te = [x.iloc[:n_tr, :].copy(), x.iloc[n_tr:, :].copy(), y.iloc[:n_tr].copy(),
                          y.iloc[n_tr:].copy()]
m_lgb = LinearForecaster().fit(x_tr, y_tr)
y_hat = m_lgb.predict(x_te)

for i in range(0, 24*10, 24):
    plt.figure()
    plt.plot(y_te.iloc[i], label='y_te')
    plt.plot(y_hat.iloc[i], label='y_hat')
    plt.show()

tr = umap.UMAP(n_neighbors=20,learning_rate=0.1).fit(y_tr)
y_te_tr = tr.transform(y_te)
y_hat_tr = tr.transform(y_hat)


plt.figure()
xy = y_hat_tr
uv = y_te_tr - y_hat_tr
plt.quiver(*xy.T, *uv.T,
           np.sum(uv ** 2, axis=1) ** 0.5, angles='xy')
grid_x, grid_y = np.meshgrid(np.linspace(xy[:, 0].min(), xy[:, 0].max(), 20),
                             np.linspace(xy[:, 1].min(), xy[:, 1].max(), 20))
# Interpolate components
u_grid = griddata(xy, uv[:, 0], (grid_x, grid_y), method='cubic')
v_grid = griddata(xy, uv[:, 1], (grid_x, grid_y), method='cubic')
plt.streamplot(grid_x, grid_y, u_grid, v_grid, color=np.sqrt(u_grid ** 2 + v_grid ** 2), cmap='viridis')
plt.show()



# plot a vectorial field using y_te.iloc[:, t] and y_hat.iloc[:, t] as starts and ends of the vectors
# color it proportionally to the error (length of vector)
tuples = np.random.randint(0, y_tr.shape[1], (10, 2))
for t in tuples:
    plt.figure()
    #plt.scatter(*y_te.iloc[:, t].values.T, s=1, c='b')
    #plt.scatter(*y_hat.iloc[:, t].values.T, s=10, c='r')
    xy = y_hat.iloc[:, t].values
    uv = y_te.iloc[:, t].values - y_hat.iloc[:, t].values
    plt.quiver(*xy.T, *uv.T, y_hat.index.hour, angles='xy', alpha=0.5)

    grid_x, grid_y = np.meshgrid(np.linspace(xy[:, 0].min(), xy[:, 0].max(), 20),
                                 np.linspace(xy[:, 1].min(), xy[:, 1].max(), 20))
    # Interpolate components
    u_grid = griddata(xy, uv[:, 0], (grid_x, grid_y), method='cubic')
    v_grid = griddata(xy, uv[:, 1], (grid_x, grid_y), method='cubic')
    plt.streamplot(grid_x, grid_y, u_grid, v_grid, color=np.sqrt(u_grid ** 2 + v_grid ** 2), cmap='viridis')

    plt.title('Error, tuple: {}'.format(t))
    plt.show()

