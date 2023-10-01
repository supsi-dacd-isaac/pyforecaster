import jax.numpy as jnp
import pandas as pd
from flax import linen as nn
import pickle as pk

from jax import random, grad, jit, value_and_grad, vmap
import optax
from tqdm import tqdm
import numpy as np
from functools import partial
from os.path import join
import matplotlib.pyplot as plt
def positive_lecun_normal(key, shape, dtype=jnp.float32):
    # Start with standard lecun_normal initialization
    stddev = 1. / jnp.sqrt(shape[1])
    weights = random.normal(key, shape, dtype) * stddev
    # Ensure weights are non-negative
    return jnp.abs(weights)/10


class PICNNLayer(nn.Module):
    features_x: int
    features_y: int
    features_out: int
    n_layer: int = 0
    prediction_layer: bool = False
    @nn.compact
    def __call__(self, y, u, z):
        # Traditional NN component
        u_next = nn.Dense(features=self.features_x, name='u_dense')(u)
        u_next = nn.relu(u_next)

        # Input-Convex component without bias for the element-wise multiplicative interactions
        wzu = nn.Dense(features=self.features_out, use_bias=True, name='wzu')(u)
        wyu = nn.Dense(features=self.features_y, use_bias=True, name='wyu')(u)

        z_next = nn.Dense(features=self.features_out, use_bias=False, name='wz',kernel_init=positive_lecun_normal)(z * wzu)
        y_next = nn.Dense(features=self.features_out, use_bias=False, name='wy')(y * wyu)
        z_next = z_next + y_next + nn.Dense(features=self.features_out, use_bias=True, name='wuz')(u)
        if not self.prediction_layer:
            z_next = nn.relu(z_next)

        return u_next, z_next


class PartiallyICNN(nn.Module):
    num_layers: int
    features_x: int
    features_y: int
    features_out: int

    @nn.compact
    def __call__(self, x, y):
        u = x
        z = jnp.zeros(self.features_out)  # Initialize z_0 to be the same shape as y
        for i in range(self.num_layers):
            prediction_layer = i == self.num_layers -1
            u, z = PICNNLayer(features_x=self.features_x, features_y=self.features_y, features_out=self.features_out, n_layer=i, prediction_layer=prediction_layer)(y, u, z)
        return z




def reproject_weights(params):
    # Loop through each layer and reproject the input-convex weights
    for layer_name in params['params']:
        if 'PICNNLayer' in layer_name:
            params['params'][layer_name]['wzu']['kernel'] = jnp.maximum(0, params['params'][layer_name]['wzu']['kernel'])
            #params[layer_name]['wyu']['kernel'] = jnp.maximum(0, params[layer_name]['wyu']['kernel'])
    return params




class PICNN(object):
    "Partially input-convex neural network"
    learning_rate: float = 0.01
    inverter_learning_rate: float = 0.1
    batch_size: int = None
    load_path: str = None
    n_hidden_x: int = 100
    n_hidden_y: int = 100
    n_out: int = None
    n_layers: int = None
    optimization_vars: list = ()
    pars: dict = None
    target_columns: list = None
    def __init__(self, **kwargs):
        self.set_attr(kwargs)
        if self.load_path is not None:
            self.load(self.load_path)

        self.n_hidden_y = len(self.optimization_vars)
        model = self.set_arch()
        self.model = model

        self.optimizer = optax.adamw(learning_rate=self.learning_rate)
        self.inverter_optimizer = optax.adamw(learning_rate=self.inverter_learning_rate)
        @jit
        def loss_fn(params, x, y, target):
            predictions = model.apply(params, x, y)
            return jnp.mean((predictions - target) ** 2)
        @jit
        def train_step(params, optimizer_state, x_batch, y_batch, target_batch):
            values, grads = value_and_grad(loss_fn)(params, x_batch, y_batch, target_batch)
            updates, opt_state = self.optimizer.update(grads, optimizer_state, params)
            return optax.apply_updates(params, updates), opt_state, values

        @jit
        @partial(vmap, in_axes=(None, 0, 0))
        def predict_batch(pars, x, y):
            return model.apply(pars, x, y)

        self.train_step = train_step
        self.loss_fn = loss_fn
        self.predict_batch = predict_batch
        self.iterate = None

    @classmethod
    def get_class_properties_names(cls):
        return [key for key, value in cls.__dict__.items()
                if not callable(value)
                and not key.startswith('__')
                and not isinstance(value, (classmethod, staticmethod))]

    def get_class_properties(self):
        return {k: getattr(self, k) for k in self.get_class_properties_names()}

    def save(self, save_path):
        attrdict = self.get_class_properties()
        with open(save_path, 'wb') as f:
            pk.dump(attrdict, f, protocol=pk.HIGHEST_PROTOCOL)
    def set_attr(self, attrdict):
        [self.__setattr__(k, v) for k, v in attrdict.items()]

    def load(self, save_path):
        with open(save_path, 'rb') as f:
            attrdict = pk.load(f)
        self.set_attr(attrdict)

    @staticmethod
    def init_arch(nn_init, n_inputs_x=1, n_inputs_opt=1):
        "divides data into training and test sets "
        key1, key2 = random.split(random.key(0))
        x = random.normal(key1, (n_inputs_x, ))  # Dummy input data (for the first input)
        y = random.normal(key1, (n_inputs_opt, ))  # Dummy input data (for the second input)
        init_params = nn_init.init(key2, x, y)  # Initialization call

        return init_params

    def set_arch(self):
        model = PartiallyICNN(num_layers=self.n_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y, features_out=self.n_out)
        return model

    def train(self, inputs, target, inputs_te=None, target_test=None, n_epochs=10, savepath_tr_plots=None, stats_step=5000):
        self.target_columns = target.columns
        batch_size = self.batch_size if self.batch_size is not None else inputs.shape[0] // 10
        num_batches = inputs.shape[0] // batch_size

        x, y = self.get_inputs(inputs)
        target = target.values

        n_inputs_opt = len(self.optimization_vars)
        n_inputs_x =  inputs.shape[1] - n_inputs_opt
        pars = self.init_arch(self.model, n_inputs_opt=n_inputs_opt, n_inputs_x=n_inputs_x)
        opt_state = self.optimizer.init(pars)


        tr_loss, te_loss = [], []
        k = 0
        for epoch in range(n_epochs):
            for i in tqdm(range(num_batches)):
                rand_idx = np.random.choice(inputs.shape[0], batch_size)
                x_batch = x[rand_idx, :]
                y_batch = y[rand_idx, :]
                target_batch = target[rand_idx, :]

                pars, opt_state, values = self.train_step(pars, opt_state, x_batch, y_batch, target_batch)
                pars = reproject_weights(pars)

                if k % stats_step == 0 and k > 0 and inputs_te is not None and target_test is not None:
                    self.pars = pars
                    x_test = inputs_te[[c for c in inputs_te.columns if not c in self.optimization_vars]].values
                    y_test = inputs_te[self.optimization_vars].values
                    loss = self.loss_fn(pars, x_test[:batch_size, :], y_test[:batch_size, :], target_test.values[:batch_size, :])
                    te_loss.append(np.array(jnp.mean(loss)))
                    tr_loss.append(np.array(jnp.mean(values)))
                    if savepath_tr_plots is not None:
                        rand_idx_plt = np.random.choice(x_test.shape[0], 9)
                        self.training_plots(x_test[rand_idx_plt, :], y_test[rand_idx_plt, :], target_test.values[rand_idx_plt, :], savepath_tr_plots, k)
                    print('tr loss: {:0.2e}, te loss: {:0.2e}'.format(tr_loss[-1], te_loss[-1]))

                k += 1
            self.pars = pars
        return self
    def training_plots(self, x, y, target, savepath, k):
        n_instances = x.shape[0]
        y_hat = self.predict_batch(self.pars, x, y)
        # make the appropriate numbers of subplots disposed as a square
        fig, ax = plt.subplots(int(np.ceil(np.sqrt(n_instances))), int(np.ceil(np.sqrt(n_instances))))
        for i, a  in enumerate(ax.ravel()):
            a.plot(y_hat[i, :])
            a.plot(target[i, :])
            a.set_title('instance {}, iter {}'.format(i, k))
        plt.savefig(join(savepath, 'iteration_{}.png'.format(k)))

    def predict(self, inputs):
        x, y = self.get_inputs(inputs)
        y_hat = self.predict_batch(self.pars, x, y)
        return pd.DataFrame(y_hat, index=inputs.index, columns=self.target_columns)


    def optimize(self, inputs, objective, n_iter=200):
        inputs = inputs.copy()
        x, y = self.get_inputs(inputs)
        def _objective(y, x):
            return objective(self.model.apply(self.pars, x, y))

        if self.iterate is None:
            @jit
            def iterate(x, y, opt_state):
                for i in range(10):
                    values, grads = value_and_grad(_objective)(y, x)
                    updates, opt_state = self.inverter_optimizer.update(grads, opt_state, y)
                    y = optax.apply_updates(y, updates)
                return y, values
            self.iterate = iterate

        opt_state = self.optimizer.init(y)
        for i in range(n_iter//10):
            y, values = self.iterate(x, y, opt_state)
            print('iter {}, loss: {:0.2e}'.format((i+1)*10, values))


        inputs.loc[:, self.optimization_vars] = y
        target_opt = self.predict(inputs)
        return target_opt, y, values


    def get_inputs(self, inputs):
        x = inputs[[c for c in inputs.columns if not c in self.optimization_vars]].values
        y = inputs[self.optimization_vars].values
        return x, y