import numpy as np
import pandas as pd
from opfunu.name_based import CamelThreeHump, CamelSixHump, CosineMixture, Ackley01
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pyforecaster.forecasting_models.neural_forecasters import PICNN, PIQCNN, FFNN, PIQCNNSigmoid, LatentStructuredPICNN, predict_batch_latent_picnn
import jax.numpy as jnp
from os.path import join
from pyforecaster.forecasting_models.neural_forecasters import latent_pred, encode, decode
from jax import vmap


def make_dataset(fun, lb, ub, n_sample_per_dim=1000):
    X = np.linspace(lb[0], ub[0], n_sample_per_dim)
    Y = np.linspace(lb[1], ub[1], n_sample_per_dim)
    X, Y = np.meshgrid(X, Y)
    points = np.array([X, Y]).reshape(2, -1).T
    Z = np.hstack([fun(p) for p in points])
    data = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)])
    data = np.hstack([np.ones((data.shape[0], 1)), data])
    data = pd.DataFrame(data, columns=['const', 'x1', 'x2', 'y'])
    return data

def plot_surface(data, scatter=False, **kwargs):
    n_samples = int(np.sqrt(data.shape[0]))
    X = data['x1'].values.reshape(n_samples, n_samples)
    Y = data['x2'].values.reshape(n_samples, n_samples)
    Z = data['y'].values.reshape(n_samples, n_samples)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface.
    if scatter:
        surf = ax.scatter(X, Y, Z, c=Z, cmap=cm.plasma,
                          linewidth=0, antialiased=False, s=1, **kwargs)
    else:
        surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma,
                           linewidth=0, antialiased=False, **kwargs)

    return fig, ax

def train_test(test_fun, n_dim, forecaster_class=LatentStructuredPICNN, **forecaster_kwargs):
    # create dataset
    ub = np.ones(n_dim)
    lb = -np.ones(n_dim)
    data = make_dataset(test_fun(ndim=n_dim).evaluate, lb, ub, n_sample_per_dim=500)
    data = data.iloc[np.random.permutation(data.shape[0])]
    optimization_vars = ['x{}'.format(i) for i in range(1, n_dim+1)]
    x_names = ['const'] + optimization_vars
    m = forecaster_class(optimization_vars=optimization_vars, **forecaster_kwargs).fit(data[x_names], data[['y']])

    objective_fun = lambda x, ctrl: jnp.mean(x**2) #+ boxconstr(ctrl, ub, lb)
    print('minimum in the training set: {}'.format(data['y'].min()))
    sol = m.optimize(data[x_names].iloc[[0], :], objective=objective_fun,
                     n_iter=10000, recompile_obj=False, rel_tol=1e-12)

    eo_convexity_test(data[[c for c in data.columns if c != 'y']], m, optimization_vars)
    io_convexity_test(data[[c for c in data.columns if c !='y']], m, optimization_vars)
    e_optobj_convexity_test(data[[c for c in data.columns if c !='y']], m, optimization_vars)
    # find the global minimum in the latent space of the LatentStructuredPICNN
    for i in np.random.choice(data.shape[0], 10):
        sol = m.optimize(data[x_names].iloc[[i],:], objective=objective_fun,n_iter=10000, recompile_obj=False, rel_tol=1e-12)
        print(sol[0], sol[2].values.ravel(), sol[3])

    if n_dim == 2:
        savepath = 'wp3/global_optimization/'

        # scatter plot of the learned function m
        y_hat = m.predict(data[x_names])
        plot_surface(pd.concat([data[optimization_vars], y_hat], axis=1), scatter=True)
        plt.show()

        y_hat = m.predict(data[x_names])
        plot_surface(data, scatter=True)
        plt.savefig(join(savepath, '{}_ground_truth.png'.format(test_fun.__name__)))

        fig, ax = plot_surface(pd.concat([data[optimization_vars], y_hat], axis=1), scatter=True, alpha=0.2)
        ax.scatter(*np.hstack([sol[0], sol[2].values.ravel()]), c='r', s=1000, marker='*')



        for ii in range(0,360,36):
                ax.view_init(elev=10., azim=ii)
                plt.savefig(join(savepath, '{}_latent_optimization_{}.png'.format(test_fun.__name__, ii)))
        plt.show()

def boxconstr(x, ub, lb):
    return jnp.sum(jnp.maximum(0, x - ub)**2 + jnp.maximum(0, lb - x)**2)

def e_optobj_convexity_test(df, forecaster, ctrl_names, **objective_kwargs):
    x_names = [c for c in df.columns if not c in ctrl_names]
    rand_idxs = np.random.choice(len(df), 1)
    for idx in rand_idxs:
        x  = df[x_names].iloc[idx, :]
        ctrl_embedding = np.tile(np.random.randn(forecaster.n_embeddings, 1), 20).T

        plt.figure()
        for ctrl_e in range(forecaster.n_embeddings):
            ctrls = ctrl_embedding.copy()
            ctrls[:, ctrl_e] = np.linspace(-1, 1, 20)
            preds = np.hstack([forecaster._objective(c, np.atleast_2d(x.values.ravel()), **objective_kwargs) for c in ctrls])
            approx_second_der = np.round(np.diff(preds, 2, axis=0), 5)
            approx_second_der[approx_second_der == 0] = 0  # to fix the sign
            is_convex = np.all(np.sign(approx_second_der) >= 0)
            print('output is convex w.r.t. input {}: {}'.format(ctrl_e, is_convex))
            plt.plot(ctrls[:, ctrl_e], preds, alpha=0.3)
            plt.pause(0.001)
        plt.show()

def io_convexity_test(df, forecaster, ctrl_names, **objective_kwargs):
    rand_idxs = np.random.choice(len(df), 1)
    for idx in rand_idxs:
        df_i  = df.iloc[[idx]*20, :].copy()
        plt.figure()
        for ctrl_e in range(len(ctrl_names)):
            df_e = df_i.copy()
            df_e[ctrl_names[ctrl_e]] = np.linspace(-1, 1, 20)
            preds = np.hstack([forecaster.predict(df_e.loc[[i], :]) for i in df_e.index])
            approx_second_der = np.round(np.diff(preds, 2, axis=0), 5)
            approx_second_der[approx_second_der == 0] = 0  # to fix the sign
            is_convex = np.all(np.sign(approx_second_der) >= 0)
            print('output is convex w.r.t. input {}: {}'.format(ctrl_e, is_convex))
            plt.plot(preds, alpha=0.3)
            plt.pause(0.001)
        plt.show()

def eo_convexity_test(df, forecaster, ctrl_names, **objective_kwargs):
    x_names = [c for c in df.columns if not c in ctrl_names]
    rand_idxs = np.random.choice(len(df), 1)
    for idx in rand_idxs:
        x  = df[x_names].iloc[idx, :]
        ctrl_embedding = np.tile(np.random.randn(forecaster.n_embeddings, 1), 20).T

        plt.figure()
        for ctrl_e in range(forecaster.n_embeddings):
            ctrls = ctrl_embedding.copy()
            ctrls[:, ctrl_e] = np.linspace(-1, 1, 20)

            preds = np.hstack([latent_pred(forecaster.pars, forecaster.model, x, c) for c in ctrls])
            approx_second_der = np.round(np.diff(preds, 2, axis=0), 5)
            approx_second_der[approx_second_der == 0] = 0  # to fix the sign
            is_convex = np.all(np.sign(approx_second_der) >= 0)
            print('output is convex w.r.t. input {}: {}'.format(ctrl_e, is_convex))
            plt.plot(ctrls[:, ctrl_e], preds, alpha=0.3)
            plt.pause(0.001)
        plt.show()



def test1d(forecaster_class, **forecaster_kwargs):
    fun = lambda x:  (0.01 * (x * 10) ** 4 - ((x - 0.1) * 10) ** 2)/10
    x = np.linspace(-1, 1, 5000)

    data = pd.DataFrame(np.hstack([np.ones((x.shape[0], 1)), x.reshape(-1, 1), fun(x).reshape(-1, 1)]), columns=['const', 'x', 'y'])
    data = data.iloc[np.random.permutation(data.shape[0])]

    optimization_vars = ['x']
    x_names = ['const'] + optimization_vars
    m = forecaster_class(optimization_vars=optimization_vars, **forecaster_kwargs).fit(data[x_names], data[['y']])

    reduced_data = data.iloc[::50, :]
    # explore latent space

    e = vmap(lambda x, y: encode(m.pars, m.model, x, y), in_axes=(0, 0))
    d = vmap(lambda x, y: decode(m.pars, m.model, x, y), in_axes=(0, 0))
    l = vmap(lambda x, y: latent_pred(m.pars, m.model, x, y), in_axes=(0, 0))

    Z_hat_tr = m.predict(reduced_data[x_names]).values
    z, ctrl_embedding, ctrl_reconstruction = m.predict_batch_training(m.pars, [reduced_data[['const']].values, reduced_data[optimization_vars].values/m.input_scaler.scale_])

    E_tr = e(reduced_data[['const']].values, reduced_data[optimization_vars].values/m.input_scaler.scale_)

    e_max, e_min = np.max(E_tr, axis=0), np.min(E_tr, axis=0)
    X, Y = np.meshgrid(np.linspace(e_min[0], e_max[0], 10), np.linspace(e_min[1], e_max[1], 10))
    E_mesh = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
    Z_mesh = l(np.ones((len(E_mesh), 1)), E_mesh)

    x_reconstructed = d(np.ones((len(E_mesh), 1)), E_mesh)*m.input_scaler.scale_
    Z_oracle = np.hstack([fun(x) for x in x_reconstructed])

    x_tr_reconstructed = d(np.ones((len(E_tr), 1)), E_tr)*m.input_scaler.scale_
    Z_oracle_tr = np.hstack([fun(x) for x in x_tr_reconstructed])


    Z_from_latent = l(np.ones((len(E_tr), 1)), E_tr)


    # Plot the surface and the original points in latent space
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_mesh.reshape(10, 10), cmap=cm.plasma,
                            linewidth=0, antialiased=False, alpha=0.3)

    # fun from latent reconstruction tr points only
    ax.scatter(*E_tr.T, Z_from_latent, c='k', s=15, marker='v')

    # ground truth
    ax.scatter(*E_tr.T, data['y'].ravel()[::50], c='r', s=5, alpha=1)

    # preds
    ax.scatter(*E_tr.T, Z_hat_tr, c='orange', s=5, marker='o', alpha=0.3)

    # fun from ctrl reconstruction
    ax.scatter(*E_mesh.T, Z_oracle, c='b', s=1)

    # fun from latent reconstruction tr points only
    ax.scatter(*E_tr.T, Z_oracle_tr, c='b', s=20, marker='+', alpha=1)



    plt.show()



if __name__ == '__main__':
    forecaster_kwargs = dict(
    augment_ctrl_inputs = False, n_hidden_x = 3, n_latent=30, n_out = 1, batch_size = 2500,
    n_epochs = 500, stats_step = 20, n_layers = 3, n_encoder_layers = 3, n_decoder_layers = 3, learning_rate = 1e-3,
    stopping_rounds = 1000, init_type = 'normal', rel_tol = -2, layer_normalization = False, n_embeddings=2, normalize_target=False,
    unnormalized_inputs=['const'])

    test1d(forecaster_class=LatentStructuredPICNN, **forecaster_kwargs)

    train_test(Ackley01, 2, forecaster_class=LatentStructuredPICNN, **forecaster_kwargs)



    forecaster_kwargs = dict(
    augment_ctrl_inputs = True, n_hidden_x = 4, n_out = 1, batch_size = 1000,
    n_epochs = 10, stats_step = 100, n_layers = 4, learning_rate = 1e-2, n_latent=100,
    stopping_rounds = 10, init_type = 'normal', rel_tol = -1, layer_normalization = False, normalize_target=False)
    train_test(Ackley01, 2, forecaster_class=LatentStructuredPICNN, **forecaster_kwargs)


