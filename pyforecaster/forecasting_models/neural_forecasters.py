import jax.numpy as jnp
import pandas as pd
from flax import linen as nn
import pickle as pk
from pyforecaster.forecaster import ScenarioGenerator
from jax import random, grad, jit, value_and_grad, vmap
import optax
from tqdm import tqdm
import numpy as np
from functools import partial
from os.path import join
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Union
from jax import jit
from inspect import getmro

def positive_lecun(key, shape, dtype=jnp.float32, init_type='normal'):
    # Start with standard lecun_normal initialization
    stddev = 1. / jnp.sqrt(shape[1])
    if init_type == 'normal':
        weights = random.normal(key, shape, dtype) * stddev
    elif init_type == 'uniform':
        weights = random.uniform(key, shape, dtype) * stddev
    else:
        raise NotImplementedError('init_type {} not implemented'.format(init_type))
    # Ensure weights are non-negative
    return jnp.abs(weights)/10


def identity(x):
    return x

class LinRegModule(nn.Module):
    n_out: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.n_out, name='dense')(x)
        return x

class FeedForwardModule(nn.Module):
    n_layers: Union[int, np.array, list]
    n_out: int=None
    n_neurons: int=None
    @nn.compact
    def __call__(self, x):
        if isinstance(self.n_layers, int):
            layers = np.ones(self.n_layers) * self.n_neurons
            layers[-1] = self.n_out
            layers = layers.astype(int)
        else:
            layers = self.n_layers
        for i, n in enumerate(layers):
            x = nn.Dense(features=n, name='dense_{}'.format(i))(x)
            if i < len(layers):
                x = nn.relu(x)
        return x

def jitting_wrapper(fun, model):
    return jit(partial(fun, model=model))

def loss_fn(params, inputs, targets, model=None):
    predictions = model(params, inputs)
    return jnp.mean((predictions - targets) ** 2)

def train_step(params, optimizer_state, inputs_batch, targets_batch, model=None, loss_fn=None):
    values, grads = value_and_grad(loss_fn)(params, inputs_batch, targets_batch)
    updates, opt_state = model.update(grads, optimizer_state, params)
    return optax.apply_updates(params, updates), opt_state, values

def predict_batch(pars, inputs, model=None):
    return model.apply(pars, inputs)

def predict_batch_picnn(pars, inputs, model=None):
    return model.apply(pars, *inputs)

class NN(ScenarioGenerator):
    scaler: StandardScaler = None
    learning_rate: float = 0.01
    batch_size: int = None
    load_path: str = None
    n_out: int = None
    n_layers: int = None
    n_hidden_x: int = None
    pars: dict = None
    n_epochs:int = 10
    savepath_tr_plots:str = None
    stats_step: int = 50
    rel_tol: float = 1e-4
    unnormalized_inputs: list = None
    to_be_normalized:list = None
    target_columns: list = None
    iterate = None
    def __init__(self, learning_rate: float = 0.01, batch_size: int = None, load_path: str = None,
                 n_hidden_x: int = 100, n_out: int = None, n_layers: int = 3, pars: dict = None, q_vect=None,
                 val_ratio=None, nodes_at_step=None, n_epochs: int = 10, savepath_tr_plots: str = None,
                 stats_step: int = 50, rel_tol: float = 1e-4, unnormalized_inputs=None, **scengen_kwgs):

        super().__init__(q_vect, val_ratio=val_ratio, nodes_at_step=nodes_at_step, **scengen_kwgs)
        self.set_attr({"learning_rate": learning_rate,
                       "batch_size": batch_size,
                       "load_path": load_path,
                       "n_hidden_x": n_hidden_x,
                       "n_out": n_out,
                       "n_layers": n_layers,
                       "pars": pars,
                       "n_epochs": n_epochs,
                       "savepath_tr_plots": savepath_tr_plots,
                       "stats_step": stats_step,
                       "rel_tol": rel_tol,
                       "unnormalized_inputs": unnormalized_inputs
                       })

        if self.load_path is not None:
            self.load(self.load_path)
        self.model = self.set_arch()
        self.optimizer = optax.adamw(learning_rate=self.learning_rate)


        self.predict_batch = vmap(jitting_wrapper(predict_batch, self.model), in_axes=(None, 0))
        self.loss_fn = jitting_wrapper(loss_fn, self.predict_batch)
        self.train_step = jitting_wrapper(partial(train_step, loss_fn=self.loss_fn), self.optimizer)

    def get_class_properties_names(cls):
        attributes = []
        cls = cls if isinstance(cls, type) else cls.__class__
        # Loop through the MRO (Method Resolution Order) to include parent classes
        for base_class in getmro(cls):
            attributes.extend(
                key for key, value in base_class.__dict__.items()
                if not callable(value)
                and not key.startswith('__')
                and not isinstance(value, (classmethod, staticmethod))
            )
        return list(set(attributes))  # Remove duplicates

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
    def init_arch(nn_init, n_inputs_x=1):
        "divides data into training and test sets "
        key1, key2 = random.split(random.key(0))
        x = random.normal(key1, (n_inputs_x,))  # Dummy input data (for the first input)
        init_params = nn_init.init(key2, x)  # Initialization call
        return init_params

    def set_arch(self):
        model = FeedForwardModule(n_layers=self.n_layers, n_neurons=self.n_hidden_x,
                              n_out=self.n_out)
        return model

    def fit(self, inputs, targets, n_epochs=None, savepath_tr_plots=None, stats_step=None, rel_tol=None):
        self.to_be_normalized = [c for c in inputs.columns if
                                 c not in self.unnormalized_inputs] if self.unnormalized_inputs is not None else inputs.columns
        rel_tol = rel_tol if rel_tol is not None else self.rel_tol
        n_epochs = n_epochs if n_epochs is not None else self.n_epochs
        stats_step = stats_step if stats_step is not None else self.stats_step
        self.scaler = StandardScaler().set_output(transform='pandas').fit(inputs[self.to_be_normalized])

        inputs, targets, inputs_val_0, targets_val = self.train_val_split(inputs, targets)
        training_len = inputs.shape[0]
        validation_len = inputs_val_0.shape[0]

        self.target_columns = targets.columns
        batch_size = self.batch_size if self.batch_size is not None else inputs.shape[0] // 10
        num_batches = inputs.shape[0] // batch_size

        inputs = self.get_normalized_inputs(inputs)
        inputs_val = self.get_normalized_inputs(inputs_val_0)
        inputs_len = [i.shape[1] for i in inputs] if isinstance(inputs, tuple) else inputs.shape[1]

        pars = self.init_arch(self.model, *np.atleast_1d(inputs_len))
        opt_state = self.optimizer.init(pars)

        tr_loss, val_loss = [], []
        k = 0
        finished = False
        for epoch in range(n_epochs):
            rand_idx_all = np.random.choice(training_len, training_len, replace=False)
            for i in tqdm(range(num_batches), desc='epoch {}/{}'.format(epoch, n_epochs)):
                rand_idx = rand_idx_all[i * batch_size:(i + 1) * batch_size]
                inputs_batch = [i[rand_idx, :] for i in inputs] if isinstance(inputs, tuple) else inputs[rand_idx, :]
                targets_batch = targets.values[rand_idx, :]

                pars, opt_state, values = self.train_step(pars, opt_state, inputs_batch, targets_batch)

                if k % stats_step == 0 and k > 0:
                    self.pars = pars

                    te_loss_i = self.loss_fn(pars, inputs_val, targets_val.values)
                    tr_loss_i = self.loss_fn(pars, inputs, targets.values)
                    val_loss.append(np.array(jnp.mean(te_loss_i)))
                    tr_loss.append(np.array(jnp.mean(tr_loss_i)))

                    self.logger.info('tr loss: {:0.2e}, te loss: {:0.2e}'.format(tr_loss[-1], val_loss[-1]))
                    if len(tr_loss) > 1:
                        if savepath_tr_plots is not None or self.savepath_tr_plots is not None:
                            savepath_tr_plots = savepath_tr_plots if savepath_tr_plots is not None else self.savepath_tr_plots

                            rand_idx_plt = np.random.choice(validation_len, 9)
                            self.training_plots([i[rand_idx_plt, :] for i in inputs_val] if isinstance(inputs_val, tuple) else inputs_val[rand_idx_plt, :],
                                                targets_val.values[rand_idx_plt, :], tr_loss, val_loss, savepath_tr_plots, k)

                        rel_te_err = (val_loss[-2] - val_loss[-1]) / np.abs(val_loss[-2] + 1e-6)
                        if rel_te_err < rel_tol:
                            finished = True
                            break
                k += 1
            if finished:
                break

            self.pars = pars
        super().fit(inputs_val_0, targets_val)
        return self

    def training_plots(self, inputs, target, tr_loss, te_loss, savepath, k):
        n_instances = target.shape[0]
        y_hat = self.predict_batch(self.pars, inputs)

        # make the appropriate numbers of subplots disposed as a square
        fig, ax = plt.subplots(int(np.ceil(np.sqrt(n_instances))), int(np.ceil(np.sqrt(n_instances))))
        for i, a in enumerate(ax.ravel()):
            a.plot(y_hat[i, :])
            a.plot(target[i, :])
            a.set_title('instance {}, iter {}'.format(i, k))
        plt.savefig(join(savepath, 'examples_iter_{:05d}.png'.format(k)))

        fig, ax = plt.subplots(1, 1)
        ax.plot(np.array(tr_loss), label='tr_loss')
        ax.plot(np.array(te_loss), label='te_loss')
        ax.legend()
        plt.savefig(join(savepath, 'losses_iter_{:05d}.png'.format(k)))

    def predict(self, inputs, **kwargs):
        x = self.get_normalized_inputs(inputs)
        y_hat = self.predict_batch(self.pars, x)
        return pd.DataFrame(y_hat, index=inputs.index, columns=self.target_columns)

    def get_normalized_inputs(self, inputs):
        inputs = inputs.copy()
        self.to_be_normalized = [c for c in inputs.columns if
                                 c not in self.unnormalized_inputs] if self.unnormalized_inputs is not None else inputs.columns
        normalized_inputs = self.scaler.transform(inputs[self.to_be_normalized])
        inputs.loc[:, self.to_be_normalized] = normalized_inputs.copy().values

        return inputs.values

    def get_denormalized_output(self, output):
        self.scaler.inverse_transform(output.values)


class FastLinReg(NN):
    def __init__(self, n_out=1, q_vect=None, n_epochs=10, val_ratio=None, nodes_at_step=None, learning_rate=1e-3,
                 scengen_dict={}, **model_kwargs):
        super().__init__(n_out=n_out, q_vect=q_vect, n_epochs=n_epochs, val_ratio=val_ratio, nodes_at_step=nodes_at_step, learning_rate=learning_rate,
                 nn_module=LinRegModule, scengen_dict=scengen_dict, **model_kwargs)

class FFNN(NN):
    def __init__(self, n_out=None, q_vect=None, n_epochs=10, val_ratio=None, nodes_at_step=None, learning_rate=1e-3,
                       scengen_dict={}, batch_size=None, **model_kwargs):
        super().__init__(n_out=n_out, q_vect=q_vect, n_epochs=n_epochs, val_ratio=val_ratio, nodes_at_step=nodes_at_step, learning_rate=learning_rate,
                 nn_module=FeedForwardModule, scengen_dict=scengen_dict, batch_size=batch_size,  **model_kwargs)


class PICNNLayer(nn.Module):

    features_x: int
    features_y: int
    features_out: int
    n_layer: int = 0
    prediction_layer: bool = False
    activation: callable = identity
    init_type: str = 'normal'
    augment_ctrl_inputs: bool = False
    layer_normalization: bool = False
    @nn.compact
    def __call__(self, y, u, z):
        if self.augment_ctrl_inputs:
            y = jnp.hstack([y, -y])

        # Input-Convex component without bias for the element-wise multiplicative interactions
        wzu = self.activation(nn.Dense(features=self.features_out, use_bias=True, name='wzu')(u))
        wzu = nn.relu(wzu)
        wyu = self.activation(nn.Dense(features=self.features_y, use_bias=True, name='wyu')(u))

        z_next = nn.Dense(features=self.features_out, use_bias=False, name='wz', kernel_init=partial(positive_lecun, init_type=self.init_type))(z * wzu)
        y_next = nn.Dense(features=self.features_out, use_bias=False, name='wy')(y * wyu)
        u_add = nn.Dense(features=self.features_out, use_bias=True, name='wuz')(u)

        if self.layer_normalization:
            y_next = nn.LayerNorm()(y_next)
            z_next = nn.LayerNorm()(z_next)
            u_add = nn.LayerNorm()(u_add)

        z_next = z_next + y_next + u_add
        if not self.prediction_layer:
            z_next = nn.relu(z_next)
            # Traditional NN component only if it's not the prediction layer
            u_next = nn.Dense(features=self.features_x, name='u_dense')(u)
            u_next = nn.relu(u_next)
            return u_next, z_next
        else:
            return None, z_next


class PartiallyICNN(nn.Module):
    num_layers: int
    features_x: int
    features_y: int
    features_out: int
    activation: callable = identity
    init_type: str = 'normal'
    augment_ctrl_inputs: bool = False
    layer_normalization:bool = False
    @nn.compact
    def __call__(self, x, y):
        u = x
        z = jnp.zeros(self.features_out)  # Initialize z_0 to be the same shape as y
        for i in range(self.num_layers):
            prediction_layer = i == self.num_layers -1
            u, z = PICNNLayer(features_x=self.features_x, features_y=self.features_y, features_out=self.features_out,
                              n_layer=i, prediction_layer=prediction_layer, activation=self.activation,
                              init_type=self.init_type, augment_ctrl_inputs=self.augment_ctrl_inputs,
                              layer_normalization=self.layer_normalization)(y, u, z)
        return z


def reproject_weights(params, rec_stable=False):
    # Loop through each layer and reproject the input-convex weights
    for layer_name in params['params']:
        if 'PICNNLayer' in layer_name:
            params['params'][layer_name]['wz']['kernel'] = jnp.maximum(0, params['params'][layer_name]['wz']['kernel'])
            if rec_stable:
                params['params'][layer_name]['wy']['kernel'] = jnp.maximum(0, params['params'][layer_name]['wy']['kernel'])

    return params

class PICNN(NN):
    inverter_learning_rate: float = 0.1
    n_hidden_y: int = 100
    optimization_vars: list = ()
    rec_stable: bool = False
    init_type: str = 'normal'
    augment_ctrl_inputs: bool = False
    layer_normalization: bool = False
    def __init__(self, learning_rate: float = 0.01, batch_size: int = None, load_path: str = None,
                 n_hidden_x: int = 100, n_out: int = None, n_layers: int = 3, pars: dict = None, q_vect=None,
                 val_ratio=None, nodes_at_step=None, n_epochs: int = 10, savepath_tr_plots: str = None,
                 stats_step: int = 50, rel_tol: float = 1e-4, unnormalized_inputs=None,
                 inverter_learning_rate: float = 0.1, optimization_vars: list = (), target_columns: list = None,
                 init_type='normal', augment_ctrl_inputs=False, layer_normalization=False,**scengen_kwgs):

        super().__init__(learning_rate, batch_size, load_path, n_hidden_x, n_out, n_layers, pars, q_vect, val_ratio,
                         nodes_at_step, n_epochs, savepath_tr_plots, stats_step, rel_tol, unnormalized_inputs,
                         **scengen_kwgs)
        self.set_attr({"inverter_learning_rate":inverter_learning_rate,
                       "optimization_vars":optimization_vars,
                       "target_columns":target_columns,
                       "init_type":init_type,
                       "augment_ctrl_inputs":augment_ctrl_inputs,
                       "layer_normalization":layer_normalization
                       })

        if load_path is not None:
            self.load(load_path)
        self.n_hidden_y = 2 * len(self.optimization_vars) if augment_ctrl_inputs else len(self.optimization_vars)
        self.model = self.set_arch()
        self.optimizer = optax.adamw(learning_rate=self.learning_rate)
        self.inverter_optimizer = optax.adabelief(learning_rate=self.inverter_learning_rate)

        self.predict_batch = vmap(jitting_wrapper(predict_batch_picnn, self.model), in_axes=(None, 0))
        self.loss_fn = jitting_wrapper(loss_fn, self.predict_batch)
        self.train_step = jitting_wrapper(partial(train_step, loss_fn=self.loss_fn), self.optimizer)

    def set_arch(self):
        model = PartiallyICNN(num_layers=self.n_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y,
                              features_out=self.n_out, init_type=self.init_type, augment_ctrl_inputs=self.augment_ctrl_inputs)
        return model

    @staticmethod
    def init_arch(nn_init, n_inputs_x=1, n_inputs_opt=1):
        "divides data into training and test sets "
        key1, key2 = random.split(random.key(0))
        x = random.normal(key1, (n_inputs_x, ))  # Dummy input data (for the first input)
        y = random.normal(key1, (n_inputs_opt, ))  # Dummy input data (for the second input)
        init_params = nn_init.init(key2, x, y)  # Initialization call
        return init_params
    def get_normalized_inputs(self, inputs):
        inputs = inputs.copy()
        self.to_be_normalized = [c for c in inputs.columns if c not in self.unnormalized_inputs] if self.unnormalized_inputs is not None else inputs.columns
        normalized_inputs = self.scaler.transform(inputs[self.to_be_normalized])
        inputs.loc[:, self.to_be_normalized] = normalized_inputs.copy().values

        x = inputs[[c for c in inputs.columns if not c in self.optimization_vars]].values
        y = inputs[self.optimization_vars].values
        return x, y
    def optimize(self, inputs, objective, n_iter=200, rel_tol=1e-4, recompile_obj=True, vanilla_gd=False, **objective_kwargs):
        rel_tol = rel_tol if rel_tol is not None else self.rel_tol
        inputs = inputs.copy()
        x, y = self.get_normalized_inputs(inputs)

        def _objective(y, x, **objective_kwargs):
            return objective(self.model.apply(self.pars, x, y), y, **objective_kwargs)

        # if the objective changes from one call to another, you need to recompile it. Slower but necessary
        if recompile_obj:
            @jit
            def iterate(x, y, opt_state, **objective_kwargs):
                for i in range(10):
                    values, grads = value_and_grad(partial(_objective, **objective_kwargs))(y, x)
                    if vanilla_gd:
                        y -= grads * 1e-1
                    else:
                        updates, opt_state = self.inverter_optimizer.update(grads, opt_state, y)
                        y = optax.apply_updates(y, updates)
                return y, values
            self.iterate = iterate
        else:
            if self.iterate is None:
                @jit
                def iterate(x, y, opt_state, **objective_kwargs):
                    for i in range(10):
                        values, grads = value_and_grad(partial(_objective, **objective_kwargs))(y, x)
                        if vanilla_gd:
                            y -= grads * 1e-1
                        else:
                            updates, opt_state = self.inverter_optimizer.update(grads, opt_state, y)
                            y = optax.apply_updates(y, updates)
                    return y, values

                self.iterate = iterate

        opt_state = self.inverter_optimizer.init(y)
        y, values_old = self.iterate(x, y, opt_state, **objective_kwargs)
        values_init = np.copy(values_old)

        # do 10 iterations at a time to speed up, check for convergence
        for i in range(n_iter//10):
            y, values = self.iterate(x, y, opt_state, **objective_kwargs)
            rel_improvement = (values_old - values) / (np.abs(values_old)+ 1e-12)
            values_old = values
            if rel_improvement < rel_tol:
                break

        print('optimization terminated at iter {}, final objective value: {:0.2e} '
                         'rel improvement: {:0.2e}'.format((i+1)*10, values,
                                                          (values_init-values)/(np.abs(values_init)+1e-12)))

        inputs.loc[:, self.optimization_vars] = y.ravel()
        inputs.loc[:, [c for c in inputs.columns if c not in  self.optimization_vars]] = x.ravel()
        inputs.loc[:, self.to_be_normalized] = self.scaler.inverse_transform(inputs[self.to_be_normalized].values)
        target_opt = self.predict(inputs)

        y_opt = inputs.loc[:, self.optimization_vars].values.ravel()
        return y_opt, inputs, target_opt, values

class PICNN_old(ScenarioGenerator):
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
    scaler = None
    n_epochs:int = 10
    savepath_tr_plots:str = None
    stats_step: int = 50
    rel_tol: float = 1e-4
    rec_stable: bool = False
    init_type: str = 'normal'
    unnormalized_inputs: list = None
    to_be_normalized:list = None
    augment_ctrl_inputs: bool = False
    layer_normalization: bool = False
    def __init__(self, learning_rate: float = 0.01, inverter_learning_rate: float = 0.1, batch_size: int = None,
                 load_path: str = None, n_hidden_x: int = 100, n_out: int = None,
                 n_layers: int = 3, optimization_vars: list = (), pars: dict = None, target_columns: list = None,
                 q_vect=None, val_ratio=None, nodes_at_step=None, n_epochs:int=10, savepath_tr_plots:str = None,
                 stats_step:int=50, rel_tol:float=1e-4, init_type='normal', unnormalized_inputs=None,
                 augment_ctrl_inputs=False, layer_normalization=False, **scengen_kwgs):
        super().__init__(q_vect, val_ratio=val_ratio, nodes_at_step=nodes_at_step, **scengen_kwgs)
        self.set_attr({"learning_rate": learning_rate,
                       "inverter_learning_rate": inverter_learning_rate,
                       "batch_size": batch_size,
                       "load_path": load_path,
                       "n_hidden_x": n_hidden_x,
                       "n_out": n_out,
                       "n_layers": n_layers,
                       "optimization_vars": optimization_vars,
                       "pars": pars,
                       "target_columns": target_columns,
                       "n_epochs": n_epochs,
                       "savepath_tr_plots":savepath_tr_plots,
                       "stats_step": stats_step,
                       "rel_tol":rel_tol,
                       "init_type":init_type,
                       "unnormalized_inputs":unnormalized_inputs,
                       "augment_ctrl_inputs":augment_ctrl_inputs,
                       "layer_normalization":layer_normalization
                       })
        if self.load_path is not None:
            self.load(self.load_path)

        self.n_hidden_y = 2*len(self.optimization_vars) if augment_ctrl_inputs else len(self.optimization_vars)
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
        model = PartiallyICNN(num_layers=self.n_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y,
                              features_out=self.n_out, init_type=self.init_type, augment_ctrl_inputs=self.augment_ctrl_inputs)
        return model

    def fit(self, inputs, targets, n_epochs=None, savepath_tr_plots=None, stats_step=None, rel_tol=None):
        self.to_be_normalized = [c for c in inputs.columns if c not in self.unnormalized_inputs] if self.unnormalized_inputs is not None else inputs.columns
        rel_tol = rel_tol if rel_tol is not None else self.rel_tol
        n_epochs = n_epochs if n_epochs is not None else self.n_epochs
        stats_step = stats_step if stats_step is not None else self.stats_step
        self.scaler = StandardScaler().set_output(transform='pandas').fit(inputs[self.to_be_normalized])

        inputs, targets, inputs_val, targets_val = self.train_val_split(inputs, targets)
        self.target_columns = targets.columns
        batch_size = self.batch_size if self.batch_size is not None else inputs.shape[0] // 10
        num_batches = inputs.shape[0] // batch_size

        x, y = self.get_normalized_inputs(inputs)
        x_val, y_val = self.get_normalized_inputs(inputs_val)

        targets = targets.values

        n_inputs_opt = len(self.optimization_vars)
        n_inputs_x =  inputs.shape[1] - n_inputs_opt
        pars = self.init_arch(self.model, n_inputs_opt=n_inputs_opt, n_inputs_x=n_inputs_x)
        opt_state = self.optimizer.init(pars)


        tr_loss, val_loss = [], []
        k = 0
        finished = False
        for epoch in range(n_epochs):
            rand_idx_all = np.random.choice(inputs.shape[0], inputs.shape[0], replace=False)
            for i in tqdm(range(num_batches), desc='epoch {}/{}'.format(epoch, n_epochs)):
                rand_idx = rand_idx_all[i*batch_size:(i+1)*batch_size]
                x_batch = x[rand_idx, :]
                y_batch = y[rand_idx, :]
                target_batch = targets[rand_idx, :]

                pars, opt_state, values = self.train_step(pars, opt_state, x_batch, y_batch, target_batch)
                pars = reproject_weights(pars, rec_stable=self.rec_stable)

                if k % stats_step == 0 and k > 0:
                    self.pars = pars

                    te_loss_i = self.loss_fn(pars, x_val, y_val, targets_val.values)
                    tr_loss_i = self.loss_fn(pars, x, y, targets)
                    val_loss.append(np.array(jnp.mean(te_loss_i)))
                    tr_loss.append(np.array(jnp.mean(tr_loss_i)))

                    self.logger.info('tr loss: {:0.2e}, te loss: {:0.2e}'.format(tr_loss[-1], val_loss[-1]))
                    if len(tr_loss) > 1:
                        if savepath_tr_plots is not None or self.savepath_tr_plots is not None:
                            savepath_tr_plots = savepath_tr_plots if savepath_tr_plots is not None else self.savepath_tr_plots
                            rand_idx_plt = np.random.choice(x_val.shape[0], 9)
                            self.training_plots(x_val[rand_idx_plt, :], y_val[rand_idx_plt, :],
                                                targets_val.values[rand_idx_plt, :], tr_loss, val_loss, savepath_tr_plots, k)

                        rel_tr_err = (tr_loss[-2] - tr_loss[-1]) / np.abs(tr_loss[-2] + 1e-6)
                        rel_te_err = (val_loss[-2] - val_loss[-1]) / np.abs(val_loss[-2] + 1e-6)
                        if rel_te_err<rel_tol:
                            finished = True
                            break
                k += 1
            if finished:
                break

            self.pars = pars
        super().fit(inputs_val, targets_val)
        return self
    def training_plots(self, x, y, target, tr_loss, te_loss, savepath, k):
        n_instances = x.shape[0]
        y_hat = self.predict_batch(self.pars, x, y)

        # make the appropriate numbers of subplots disposed as a square
        fig, ax = plt.subplots(int(np.ceil(np.sqrt(n_instances))), int(np.ceil(np.sqrt(n_instances))))
        for i, a  in enumerate(ax.ravel()):
            a.plot(y_hat[i, :])
            a.plot(target[i, :])
            a.set_title('instance {}, iter {}'.format(i, k))
        plt.savefig(join(savepath, 'examples_iter_{:05d}.png'.format(k)))

        fig, ax = plt.subplots(1, 1)
        ax.plot(np.array(tr_loss), label='tr_loss')
        ax.plot(np.array(te_loss), label='te_loss')
        ax.legend()
        plt.savefig(join(savepath, 'losses_iter_{:05d}.png'.format(k)))


    def predict(self, inputs, **kwargs):
        x, y = self.get_normalized_inputs(inputs)
        y_hat = self.predict_batch(self.pars, x, y)
        return pd.DataFrame(y_hat, index=inputs.index, columns=self.target_columns)


    def optimize(self, inputs, objective, n_iter=200, rel_tol=1e-4, recompile_obj=True, vanilla_gd=False, **objective_kwargs):
        rel_tol = rel_tol if rel_tol is not None else self.rel_tol
        inputs = inputs.copy()
        x, y = self.get_normalized_inputs(inputs)

        def _objective(y, x, **objective_kwargs):
            return objective(self.model.apply(self.pars, x, y), y, **objective_kwargs)

        # if the objective changes from one call to another, you need to recompile it. Slower but necessary
        if recompile_obj:
            @jit
            def iterate(x, y, opt_state, **objective_kwargs):
                for i in range(10):
                    values, grads = value_and_grad(partial(_objective, **objective_kwargs))(y, x)
                    if vanilla_gd:
                        y -= grads * 1e-1
                    else:
                        updates, opt_state = self.inverter_optimizer.update(grads, opt_state, y)
                        y = optax.apply_updates(y, updates)
                return y, values
            self.iterate = iterate
        else:
            if self.iterate is None:
                @jit
                def iterate(x, y, opt_state, **objective_kwargs):
                    for i in range(10):
                        values, grads = value_and_grad(partial(_objective, **objective_kwargs))(y, x)
                        if vanilla_gd:
                            y -= grads * 1e-1
                        else:
                            updates, opt_state = self.inverter_optimizer.update(grads, opt_state, y)
                            y = optax.apply_updates(y, updates)
                    return y, values

                self.iterate = iterate

        opt_state = self.inverter_optimizer.init(y)
        y, values_old = self.iterate(x, y, opt_state, **objective_kwargs)
        values_init = np.copy(values_old)

        # do 10 iterations at a time to speed up, check for convergence
        for i in range(n_iter//10):
            y, values = self.iterate(x, y, opt_state, **objective_kwargs)
            rel_improvement = (values_old - values) / (np.abs(values_old)+ 1e-12)
            values_old = values
            if rel_improvement < rel_tol:
                break

        print('optimization terminated at iter {}, final objective value: {:0.2e} '
                         'rel improvement: {:0.2e}'.format((i+1)*10, values,
                                                          (values_init-values)/(np.abs(values_init)+1e-12)))

        inputs.loc[:, self.optimization_vars] = y.ravel()
        inputs.loc[:, [c for c in inputs.columns if c not in  self.optimization_vars]] = x.ravel()
        inputs.loc[:, self.to_be_normalized] = self.scaler.inverse_transform(inputs[self.to_be_normalized].values)
        target_opt = self.predict(inputs)

        y_opt = inputs.loc[:, self.optimization_vars].values.ravel()
        return y_opt, inputs, target_opt, values


    def get_normalized_inputs(self, inputs):
        inputs = inputs.copy()
        self.to_be_normalized = [c for c in inputs.columns if c not in self.unnormalized_inputs] if self.unnormalized_inputs is not None else inputs.columns
        normalized_inputs = self.scaler.transform(inputs[self.to_be_normalized])
        inputs.loc[:, self.to_be_normalized] = normalized_inputs.copy().values

        x = inputs[[c for c in inputs.columns if not c in self.optimization_vars]].values
        y = inputs[self.optimization_vars].values
        return x, y

    def get_denormalized_output(self, output):
        self.scaler.inverse_transform(output.values)

class RecStablePICNN(PICNN):
    rec_stable = True
    def __init__(self, learning_rate: float = 0.01, batch_size: int = None, load_path: str = None,
                 n_hidden_x: int = 100, n_out: int = None, n_layers: int = 3, pars: dict = None, q_vect=None,
                 val_ratio=None, nodes_at_step=None, n_epochs: int = 10, savepath_tr_plots: str = None,
                 stats_step: int = 50, rel_tol: float = 1e-4, unnormalized_inputs=None,
                 inverter_learning_rate: float = 0.1, optimization_vars: list = (), target_columns: list = None,
                 init_type='normal', augment_ctrl_inputs=False, layer_normalization=False,**scengen_kwgs):

        super().__init__(learning_rate, batch_size, load_path, n_hidden_x, n_out, n_layers, pars, q_vect, val_ratio,
                         nodes_at_step, n_epochs, savepath_tr_plots, stats_step, rel_tol, unnormalized_inputs,
                         inverter_learning_rate, optimization_vars, target_columns, init_type, augment_ctrl_inputs,
                         layer_normalization, **scengen_kwgs)

    def set_arch(self):
        model = PartiallyICNN(num_layers=self.n_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y,
                              features_out=self.n_out, activation=nn.relu, init_type=self.init_type)
        return model