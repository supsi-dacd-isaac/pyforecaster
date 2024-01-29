import jax
import jax.numpy as jnp
import pandas as pd
from flax import linen as nn
import pickle as pk
from pyforecaster.forecaster import ScenarioGenerator
from jax import random, grad, jit, value_and_grad, vmap, custom_jvp
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
from jax import jvp
from scipy.special import erfinv

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

def reproject_weights(params, rec_stable=False, monotone=False):
    # Loop through each layer and reproject the input-convex weights
    for layer_name in params['params']:
        if 'PICNNLayer' in layer_name:
            if monotone:
                for name in {'wz', 'wy'} & set(params['params'][layer_name].keys()):
                    params['params'][layer_name][name]['kernel'] = jnp.maximum(0, params['params'][layer_name][name]['kernel'])
                #for name in {'wzu', 'wyu', 'wuz', 'u_dense'} & set(params['params'][layer_name].keys()):
                    #    params['params'][layer_name][name]['bias'] = jnp.maximum(0, params['params'][layer_name][name][
                #    'bias'])
            else:
                params['params'][layer_name]['wz']['kernel'] = jnp.maximum(0, params['params'][layer_name]['wz']['kernel'])
                if rec_stable:
                    params['params'][layer_name]['wy']['kernel'] = jnp.maximum(0, params['params'][layer_name]['wy']['kernel'])
    return params


def jitting_wrapper(fun, model, **kwargs):
    return jit(partial(fun, model=model, **kwargs))

def loss_fn(params, inputs, targets, model=None):
    predictions = model(params, inputs)
    return jnp.mean((predictions - targets) ** 2)

def embedded_loss_fn(params, inputs, targets, model=None):
    predictions, ctrl_embedding, ctrl_reconstruction = model(params, inputs)
    predictions_from_ctrl_reconstr, _, _ = model(params, [inputs[0], ctrl_reconstruction])
    target_loss = jnp.mean((predictions - targets) ** 2)
    ctrl_reconstruction_loss = jnp.mean((ctrl_reconstruction - inputs[1]) ** 2)
    obj_reconstruction_loss = jnp.mean((predictions - predictions_from_ctrl_reconstr) ** 2)
    return target_loss + ctrl_reconstruction_loss + obj_reconstruction_loss

def probabilistic_loss(y_hat, y, sigma_square, kind='maximum_likelihood', distribution='normal'):

    if kind == 'maximum_likelihood':
        if distribution == 'normal':
            ll = ((y_hat - y) ** 2) / sigma_square + jnp.log(sigma_square)
        elif distribution == 'log-normal':
            s = jnp.sign(sigma_square)
            sigma = jnp.sqrt(jnp.abs(sigma_square))
            ll = ((jnp.log(y_hat) - jnp.log(y)) ** 2) / (sigma**2) + jnp.log(sigma**2)
        loss = ll
    elif kind == 'crps':
        if distribution == 'normal':
            sigma = jnp.sqrt(sigma_square)
            mu = y_hat
            u = (y - mu) / sigma
            crps = sigma * ( u * (2 * jax.scipy.stats.norm.cdf(u) - 1) +
                             2 * jax.scipy.stats.norm.pdf(u) - 1 / jnp.sqrt(jnp.pi))
        elif distribution == 'log-normal':
            s = jnp.sign(sigma_square)
            sigma = jnp.sqrt(jnp.abs(sigma_square)) + 1e-10

            # we have predicted the mean of the log-normal distribution, we need to recover the parameter mu
            mu = jnp.log(y_hat**2 / jnp.sqrt(jnp.abs(sigma_square) + y_hat**2))
            sigma_square = jnp.log(1 + sigma**2 / y_hat**2)
            sigma = jnp.sqrt(sigma_square)

            # Standardize the observed value
            z = (jnp.log(jnp.maximum(1e-10, s*y + 2*jnp.exp(mu)*(s==-1))) - mu) / sigma

            # Compute the CRPS using the closed-form expression
            cdf_2 = jax.scipy.stats.norm.cdf(sigma / jnp.sqrt(2))
            cdf_1 = (s==-1) + s * jax.scipy.stats.norm.cdf(z - sigma)
            cdf = (s==-1) + s * jax.scipy.stats.norm.cdf(z)

            crps = y * (2 * cdf - 1) - 2 * jnp.exp(mu + jnp.abs(sigma_square) / 2) * (
                        cdf_1 + cdf_2 - 1)
        loss = crps
    return jnp.mean(loss)

def probabilistic_loss_fn(params, inputs, targets, model=None, kind='maximum_likelihood', distribution='normal'):
    out = model(params, inputs)
    predictions = out[:, :out.shape[1]//2]
    sigma_square = out[:, out.shape[1]//2:]
    ll = probabilistic_loss(predictions, targets, sigma_square, kind=kind, distribution=distribution)
    return ll


def train_step(params, optimizer_state, inputs_batch, targets_batch, model=None, loss_fn=None, **kwargs):
    values, grads = value_and_grad(loss_fn)(params, inputs_batch, targets_batch, **kwargs)
    updates, opt_state = model.update(grads, optimizer_state, params)
    return optax.apply_updates(params, updates), opt_state, values

def predict_batch(pars, inputs, model=None):
    return model.apply(pars, inputs)

def predict_batch_picnn(pars, inputs, model=None):
    return model.apply(pars, *inputs)

def predict_batch_latent_picnn(pars, inputs, model=None, mode='all'):
    z, ctrl_embedding, ctrl_reconstruction = model.apply(pars, *inputs)
    if mode == 'all':
        return z, ctrl_embedding, ctrl_reconstruction
    elif mode == 'prediction':
        return z
    elif mode == 'embedding':
        return ctrl_embedding
    elif mode == 'reconstruction':
        return ctrl_reconstruction



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
            if i < len(layers)-1:
                x = nn.relu(x)
        return x

class PICNNLayer(nn.Module):
    features_x: int
    features_y: int
    features_out: int
    features_latent: int
    n_layer: int = 0
    prediction_layer: bool = False
    activation: callable = nn.relu
    rec_activation: callable = identity
    init_type: str = 'normal'
    augment_ctrl_inputs: bool = False
    layer_normalization: bool = False
    z_min: jnp.array = None
    z_max: jnp.array = None
    @nn.compact
    def __call__(self, y, u, z):
        if self.augment_ctrl_inputs:
            y = jnp.hstack([y, -y])

        # Input-Convex component without bias for the element-wise multiplicative interactions
        wzu = nn.relu(nn.Dense(features=self.features_latent, use_bias=True, name='wzu')(u))
        wyu = self.rec_activation(nn.Dense(features=self.features_y, use_bias=True, name='wyu')(u))
        z_next = nn.Dense(features=self.features_out, use_bias=False, name='wz', kernel_init=partial(positive_lecun, init_type=self.init_type))(z * wzu)
        y_next = nn.Dense(features=self.features_out, use_bias=False, name='wy')(y * wyu)
        u_add = nn.Dense(features=self.features_out, use_bias=True, name='wuz')(u)


        if self.layer_normalization:
            y_next = nn.LayerNorm()(y_next)
            z_next = nn.LayerNorm()(z_next)
            u_add = nn.LayerNorm()(u_add)

        z_next = z_next + y_next + u_add
        if not self.prediction_layer:
            z_next = self.activation(z_next)
            # Traditional NN component only if it's not the prediction layer
            u_next = nn.Dense(features=self.features_x, name='u_dense')(u)
            u_next = self.activation(u_next)
            return u_next, z_next
        else:
            if self.z_min is not None:
                z_next = nn.sigmoid(z_next) * (self.z_max - self.z_min) + self.z_min
            return None, z_next

class NN(ScenarioGenerator):
    input_scaler: StandardScaler = None
    target_scaler: StandardScaler = None
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
    normalize_target: bool = True
    stopping_rounds: int = 5
    subtract_mean_when_normalizing: bool = False
    causal_df: pd.DataFrame = None
    reproject: bool = False
    rec_stable: bool = False
    monotone: bool = False
    probabilistic: bool = False
    def __init__(self, learning_rate: float = 0.01, batch_size: int = None, load_path: str = None,
                 n_hidden_x: int = 100, n_out: int = None, n_layers: int = 3, pars: dict = None, q_vect=None,
                 val_ratio=None, nodes_at_step=None, n_epochs: int = 10, savepath_tr_plots: str = None,
                 stats_step: int = 50, rel_tol: float = 1e-4, unnormalized_inputs=None, normalize_target=True,
                 stopping_rounds=5, subtract_mean_when_normalizing=False, causal_df=None, probabilistic=False,
                 probabilistic_loss_kind='maximum_likelihood',
                 **scengen_kwgs):

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
                       "unnormalized_inputs": unnormalized_inputs,
                       "normalize_target":normalize_target,
                       "stopping_rounds":stopping_rounds,
                       "subtract_mean_when_normalizing":subtract_mean_when_normalizing,
                       "causal_df":causal_df,
                       "probabilistic":probabilistic,
                       "probabilistic_loss_kind":probabilistic_loss_kind
                       })

        if self.load_path is not None:
            self.load(self.load_path)
        self.optimizer = None
        self.model = None
        self.loss_fn = None
        self.train_step = None
        self.predict_batch = None
        self.set_arch()


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

    def set_params(self, **kwargs):
        [self.__setattr__(k, v) for k, v in kwargs.items() if k in self.__dict__.keys()]
        self.set_arch()

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
        self.optimizer = optax.adamw(learning_rate=self.learning_rate)
        self.model = FeedForwardModule(n_layers=self.n_layers, n_neurons=self.n_hidden_x,
                              n_out=self.n_out)
        self.predict_batch = vmap(jitting_wrapper(predict_batch, self.model), in_axes=(None, 0))
        self.loss_fn = jitting_wrapper(probabilistic_loss_fn, self.predict_batch) if self.probabilistic else (
            jitting_wrapper(loss_fn, self.predict_batch))
        self.train_step = jitting_wrapper(partial(train_step, loss_fn=self.loss_fn), self.optimizer)


    def fit(self, inputs, targets, n_epochs=None, savepath_tr_plots=None, stats_step=None, rel_tol=None):
        self.to_be_normalized = [c for c in inputs.columns if
                                 c not in self.unnormalized_inputs] if self.unnormalized_inputs is not None else inputs.columns
        rel_tol = rel_tol if rel_tol is not None else self.rel_tol
        n_epochs = n_epochs if n_epochs is not None else self.n_epochs
        stats_step = stats_step if stats_step is not None else self.stats_step
        self.input_scaler = StandardScaler(with_mean=self.subtract_mean_when_normalizing).set_output(transform='pandas').fit(inputs[self.to_be_normalized])
        if self.normalize_target:
            self.target_scaler = StandardScaler().set_output(transform='pandas').fit(targets)

        inputs, targets, inputs_val_0, targets_val_0 = self.train_val_split(inputs, targets)
        training_len = inputs.shape[0]
        validation_len = inputs_val_0.shape[0]

        self.target_columns = targets.columns
        batch_size = self.batch_size if self.batch_size is not None else inputs.shape[0] // 10
        num_batches = inputs.shape[0] // batch_size

        inputs, targets = self.get_normalized_inputs(inputs, targets)
        inputs_val, targets_val = self.get_normalized_inputs(inputs_val_0, targets_val_0)
        inputs_len = [i.shape[1] for i in inputs] if isinstance(inputs, tuple) else inputs.shape[1]

        pars = self.init_arch(self.model, *np.atleast_1d(inputs_len))
        opt_state = self.optimizer.init(pars)

        tr_loss, val_loss = [np.inf], [np.inf]
        k = 0
        finished = False
        for epoch in range(n_epochs):
            rand_idx_all = np.random.choice(training_len, training_len, replace=False)
            for i in tqdm(range(num_batches),
                          desc='epoch {}/{}, val loss={:0.3e}'.format(epoch, n_epochs, val_loss[-1] if val_loss[-1] is not np.inf else np.nan)):
                rand_idx = rand_idx_all[i * batch_size:(i + 1) * batch_size]
                inputs_batch = [i[rand_idx, :] for i in inputs] if isinstance(inputs, tuple) else inputs[rand_idx, :]
                targets_batch = targets[rand_idx, :]

                pars, opt_state, values = self.train_step(pars, opt_state, inputs_batch, targets_batch)
                if self.reproject:
                    pars = reproject_weights(pars, rec_stable=self.rec_stable, monotone=self.monotone)

                if k % stats_step == 0 and k > 0:
                    old_pars = self.pars
                    self.pars = pars
                    rand_idx_val = np.random.choice(validation_len, np.minimum(batch_size, validation_len), replace=False)
                    inputs_val_sampled = [i[rand_idx_val, :] for i in inputs_val] if isinstance(inputs_val, tuple) else inputs_val[rand_idx_val, :]
                    te_loss_i = self.loss_fn(pars, inputs_val_sampled, targets_val[rand_idx_val, :])
                    tr_loss_i = self.loss_fn(pars, inputs_batch, targets_batch)
                    val_loss.append(np.array(jnp.mean(te_loss_i)))
                    tr_loss.append(np.array(jnp.mean(tr_loss_i)))

                    self.logger.info('tr loss: {:0.2e}, te loss: {:0.2e}'.format(tr_loss[-1], val_loss[-1]))
                    if len(tr_loss) > 2:
                        if savepath_tr_plots is not None or self.savepath_tr_plots is not None:
                            savepath_tr_plots = savepath_tr_plots if savepath_tr_plots is not None else self.savepath_tr_plots

                            rand_idx_plt = np.random.choice(validation_len, 9)
                            self.training_plots([i[rand_idx_plt, :] for i in inputs_val] if isinstance(inputs_val, tuple) else inputs_val[rand_idx_plt, :],
                                                targets_val[rand_idx_plt, :], tr_loss[1:], val_loss[1:], savepath_tr_plots, k)
                            plt.close("all")
                        rel_te_err = (val_loss[-2] - val_loss[-1]) / np.abs(val_loss[-2] + 1e-6)
                        last_improvement = k // stats_step - np.argwhere(np.array(val_loss) == np.min(val_loss)).ravel()[-1]
                        if rel_te_err < rel_tol or (k>self.stopping_rounds and last_improvement > self.stopping_rounds):
                            finished = True
                            break
                k += 1
            if finished:
                break
        if len(val_loss)>2:
            if val_loss[-1] > val_loss[-2]:
                pars = old_pars
        self.pars = pars

        return self

    def training_plots(self, inputs, target, tr_loss, te_loss, savepath, k):
        n_instances = target.shape[0]
        y_hat = self.predict_batch(self.pars, inputs)
        if self.probabilistic:
            q_hat = self.predict_quantiles(inputs, normalize=False)
        # make the appropriate numbers of subplots disposed as a square
        fig, ax = plt.subplots(int(np.ceil(np.sqrt(n_instances))), int(np.ceil(np.sqrt(n_instances))), figsize=(10, 10),
                               layout='tight')
        for i, a in enumerate(ax.ravel()):
            l = a.plot(y_hat[i, :target.shape[1]])
            if self.probabilistic:
                a.plot(q_hat[i, :target.shape[1], :], '--', color=l[0].get_color(), alpha=0.2)
            a.plot(target[i, :])
            a.set_title('instance {}, iter {}'.format(i, k))


        plt.savefig(join(savepath, 'examples_iter_{:05d}.png'.format(k)))

        fig, ax = plt.subplots(1, 1)
        ax.plot(np.array(tr_loss), label='tr_loss')
        ax.plot(np.array(te_loss), label='te_loss')
        ax.legend()
        plt.savefig(join(savepath, 'losses_iter_{:05d}.png'.format(k)))

    def predict(self, inputs, return_sigma=False, **kwargs):
        x, _ = self.get_normalized_inputs(inputs)
        y_hat = self.predict_batch(self.pars, x)
        y_hat = np.array(y_hat)
        if self.normalize_target:
            if self.probabilistic:
                y_hat[:, :y_hat.shape[1]//2] = self.target_scaler.inverse_transform(y_hat[:, :y_hat.shape[1]//2])
                y_hat[:, y_hat.shape[1] // 2:] = self.target_scaler.inverse_transform((y_hat[:, y_hat.shape[1] // 2:])**0.5)
            else:
                y_hat = self.target_scaler.inverse_transform(y_hat)
        if self.probabilistic:
            preds = pd.DataFrame(y_hat[:, :y_hat.shape[1] // 2], index=inputs.index, columns=self.target_columns)
            sigmas = pd.DataFrame(y_hat[:, y_hat.shape[1] // 2:], index=inputs.index, columns=self.target_columns)
            if return_sigma:
                return preds, sigmas
            else:
                return preds
        else:
            return pd.DataFrame(y_hat, index=inputs.index, columns=self.target_columns)

    def predict_quantiles(self, inputs, normalize=True, **kwargs):
        if normalize:
            mu_hat, sigma_hat = self.predict(inputs, return_sigma=True)
        else:
            x = inputs
            y_hat = self.predict_batch(self.pars, x)
            y_hat = np.array(y_hat)
            mu_hat = y_hat[:, :y_hat.shape[1]//2]
            sigma_hat = (y_hat[:, y_hat.shape[1] // 2:])** 0.5

        preds = np.expand_dims(mu_hat, -1) * np.ones((1, 1, len(self.q_vect)))
        for i, q in enumerate(self.q_vect):
            preds[:, :, i] += sigma_hat * np.sqrt(2) * erfinv(2*q-1)
        return preds

    def get_normalized_inputs(self, inputs, target=None):
        inputs = inputs.copy()
        self.to_be_normalized = [c for c in inputs.columns if
                                 c not in self.unnormalized_inputs] if self.unnormalized_inputs is not None else inputs.columns
        normalized_inputs = self.input_scaler.transform(inputs[self.to_be_normalized])
        inputs.loc[:, self.to_be_normalized] = normalized_inputs.copy().values

        if target is not None and self.normalize_target:
            target = target.copy()
            target = self.target_scaler.transform(target)
            target = target.values
        elif target is not None:
            target = target.values
        return inputs.values, target


class FFNN(NN):
    def __init__(self, n_out=None, q_vect=None, n_epochs=10, val_ratio=None, nodes_at_step=None, learning_rate=1e-3,
                       scengen_dict={}, batch_size=None, **model_kwargs):
        super().__init__(n_out=n_out, q_vect=q_vect, n_epochs=n_epochs, val_ratio=val_ratio, nodes_at_step=nodes_at_step, learning_rate=learning_rate,
                 nn_module=FeedForwardModule, scengen_dict=scengen_dict, batch_size=batch_size,  **model_kwargs)



class PartiallyICNN(nn.Module):
    num_layers: int
    features_x: int
    features_y: int
    features_out: int
    features_latent: int
    activation: callable = nn.relu
    rec_activation: callable = identity
    init_type: str = 'normal'
    augment_ctrl_inputs: bool = False
    layer_normalization:bool = False
    probabilistic: bool = False
    structured: bool = False
    z_max: jnp.array = None
    z_min: jnp.array = None
    @nn.compact
    def __call__(self, x, y):
        u = x.copy()
        z = jnp.zeros(self.features_latent)  # Initialize z_0 to be the same shape as y
        for i in range(self.num_layers):
            prediction_layer = i == self.num_layers -1
            features_out = self.features_out if prediction_layer else self.features_latent
            u, z = PICNNLayer(features_x=self.features_x, features_y=self.features_y, features_out=features_out,
                              features_latent=self.features_latent,
                              n_layer=i, prediction_layer=prediction_layer, activation=self.activation,
                              rec_activation=self.rec_activation, init_type=self.init_type,
                              augment_ctrl_inputs=self.augment_ctrl_inputs,
                              layer_normalization=self.layer_normalization, z_min=self.z_min,
                              z_max=self.z_max)(y, u, z)
        if self.probabilistic:
            u = x.copy()
            sigma_len = 1 if self.structured else self.features_out
            sigma = jnp.zeros(sigma_len)  # Initialize z_0 to be the same shape as y
            for i in range(self.num_layers):
                prediction_layer = i == self.num_layers - 1
                u, sigma = PICNNLayer(features_x=self.features_x, features_y=self.features_y,
                                  features_out=sigma_len,
                                  features_latent=self.features_latent,
                                  n_layer=i, prediction_layer=prediction_layer, activation=self.activation,
                                  rec_activation=self.rec_activation, init_type=self.init_type,
                                  augment_ctrl_inputs=self.augment_ctrl_inputs,
                                  layer_normalization=self.layer_normalization, z_min=self.z_min,
                              z_max=self.z_max)(y, u, sigma)
            sigma = nn.softplus(sigma) + 1e-10

            return jnp.hstack([z, sigma])
        return z


class LatentPartiallyICNN(nn.Module):
    num_layers: int
    features_x: int
    features_y: int
    features_out: int
    features_latent: int
    encoder_neurons: np.array = None
    decoder_neurons: np.array = None
    activation: callable = nn.relu
    rec_activation: callable = identity
    init_type: str = 'normal'
    augment_ctrl_inputs: bool = False
    layer_normalization:bool = False
    probabilistic: bool = False
    structured: bool = False
    z_max: jnp.array = None
    z_min: jnp.array = None


    def setup(self):
        ctrl_embedding_len = self.encoder_neurons[-1]
        features_y = 2*ctrl_embedding_len if self.augment_ctrl_inputs else ctrl_embedding_len

        self.encoder = PartiallyICNN(num_layers=self.num_layers, features_x=self.features_x, features_y=self.features_y,
                              features_out=self.encoder_neurons[-1], features_latent=self.features_latent, init_type=self.init_type,
                              augment_ctrl_inputs=self.augment_ctrl_inputs, probabilistic=self.probabilistic,
                                   z_min=self.z_min, z_max=self.z_max, name='encoder')

        self.decoder = PartiallyICNN(num_layers=self.num_layers, features_x=self.features_x, features_y=features_y,
                              features_out=self.decoder_neurons[-1], features_latent=self.features_latent, init_type=self.init_type,
                              augment_ctrl_inputs=self.augment_ctrl_inputs, probabilistic=self.probabilistic,
                                   z_min=self.z_min, z_max=self.z_max, name='decoder')


        self.picnn = PartiallyICNN(num_layers=self.num_layers, features_x=self.features_x, features_y=features_y,
                              features_out=self.features_out, features_latent=self.features_latent, init_type=self.init_type,
                              augment_ctrl_inputs=self.augment_ctrl_inputs, probabilistic=self.probabilistic,
                                   z_min=self.z_min, z_max=self.z_max, name='picnn')

    def __call__(self, x, y):

        ctrl_embedding = self.encoder(x, y)
        z = self.picnn(x, ctrl_embedding)
        ctrl_reconstruction = self.decoder(x, ctrl_embedding)


        return z, ctrl_embedding, ctrl_reconstruction

    def decode(self, x, ctrl_embedding):
        return self.decoder(x, ctrl_embedding)

    def latent_pred(self, x, ctrl_embedding):
        return self.picnn(x, ctrl_embedding)

def decode(params, model, x, ctrl_embedding):
    def decoder(lpicnn ):
        return lpicnn.decode(x, ctrl_embedding)

    return nn.apply(decoder, model)(params)

def latent_pred(params, model, x, ctrl_embedding):
    def _latent_pred(lpicnn ):
        return lpicnn.latent_pred(x, ctrl_embedding)

    return nn.apply(_latent_pred, model)(params)


class PartiallyIQCNN(nn.Module):
    num_layers: int
    features_x: int
    features_y: int
    features_out: int
    features_latent: int
    activation: callable = nn.softplus
    rec_activation: callable = identity
    init_type: str = 'normal'
    augment_ctrl_inputs: bool = False
    layer_normalization:bool = False
    probabilistic: bool = False
    z_max: jnp.array = None
    z_min: jnp.array = None

    def __call_wrapper__(self, y, x):
        u = x
        z = jnp.zeros(self.features_out)  # Initialize z_0 to be the same shape as y
        for i in range(self.num_layers):
            prediction_layer = i == self.num_layers -1
            u, z = PICNNLayer(features_x=self.features_x, features_y=self.features_y, features_out=self.features_out,
                              features_latent=self.features_latent,
                              n_layer=i, prediction_layer=prediction_layer, activation=self.activation,
                              init_type=self.init_type, augment_ctrl_inputs=self.augment_ctrl_inputs,
                              layer_normalization=self.layer_normalization, z_min=self.z_min,
                              z_max=self.z_max)(y, u, z)
        return z

    @nn.compact
    def __call__(self, x, y):
        qcvx_preds = jnp.abs(jvp(partial(self.__call_wrapper__, x=x),(y, ), (y,))[1])
        ex_preds = FeedForwardModule(n_layers=self.num_layers, n_neurons=self.features_x,
                              n_out=self.features_out)(x)
        z = qcvx_preds + ex_preds
        if self.probabilistic:
            return jnp.hstack([z[:self.features_out//2], nn.softplus(z[self.features_out//2:])])
        return z



def _my_jmp(model, params, ex_inputs, ctrl_inputs, M):
    """
    Compute the mean of the absolute value of the Jacobian (of the targets with respect to the control inputs)
    matrix multiplication with M. This is done for one instance, the function is then vectorized in cauasal_loss_fn
    :param model:
    :param params:
    :param ex_inputs:
    :param ctrl_inputs:
    :param M:
    :return:
    """
    return jnp.mean(jnp.abs(jax.jacfwd(lambda y: model.apply(params, ex_inputs, y))(ctrl_inputs) * M))


def causal_loss_fn(params, inputs, targets, model=None, causal_matrix=None):
    ex_inputs, ctrl_inputs = inputs[0], inputs[1]
    predictions = vmap(model.apply, in_axes=(None, 0, 0))(params, ex_inputs, ctrl_inputs)
    causal_loss =  vmap(_my_jmp, in_axes=(None, None, 0, 0, None))(model, params, ex_inputs, ctrl_inputs, causal_matrix.T)
    mse = jnp.mean((predictions - targets) ** 2)
    return mse + jnp.mean(causal_loss)

def probabilistic_causal_loss_fn(params, inputs, targets, model=None, causal_matrix=None, kind='maximum_likelihood'):
    ex_inputs, ctrl_inputs = inputs[0], inputs[1]
    out = vmap(model.apply, in_axes=(None, 0, 0))(params, ex_inputs, ctrl_inputs)
    predictions = out[:, :out.shape[1]//2]
    sigma_square = out[:, out.shape[1]//2:]
    causal_loss =  vmap(_my_jmp, in_axes=(None, None, 0, 0, None))(model, params, ex_inputs, ctrl_inputs, causal_matrix.T)
    ll = probabilistic_loss(predictions, targets, sigma_square, kind=kind)

    return ll + jnp.mean(causal_loss)




class PICNN(NN):
    reproject: bool = True
    rec_stable: bool = False
    inverter_learning_rate: float = 0.1
    n_hidden_y: int = 100
    optimization_vars: list = ()
    init_type: str = 'normal'
    augment_ctrl_inputs: bool = False
    layer_normalization: bool = False
    probabilistic_loss_kind: str = 'maximum_likelihood'
    distribution = 'normal'
    z_min: jnp.array = None
    z_max: jnp.array = None
    n_latent: int = 1
    def __init__(self, learning_rate: float = 0.01, batch_size: int = None, load_path: str = None,
                 n_hidden_x: int = 100, n_out: int = 1, n_latent:int = 1, n_layers: int = 3, pars: dict = None, q_vect=None,
                 val_ratio=None, nodes_at_step=None, n_epochs: int = 10, savepath_tr_plots: str = None,
                 stats_step: int = 50, rel_tol: float = 1e-4, unnormalized_inputs=None, normalize_target=True,
                 stopping_rounds=5, subtract_mean_when_normalizing=False, causal_df=None, probabilistic=False,
                 probabilistic_loss_kind='maximum_likelihood', distribution = 'normal', inverter_learning_rate: float = 0.1, optimization_vars: list = (),
                 target_columns: list = None, init_type='normal', augment_ctrl_inputs=False, layer_normalization=False,
                 z_min: jnp.array = None, z_max: jnp.array = None,
                 **scengen_kwgs):


        self.set_attr({"inverter_learning_rate":inverter_learning_rate,
                       "optimization_vars":optimization_vars,
                       "target_columns":target_columns,
                       "init_type":init_type,
                       "augment_ctrl_inputs":augment_ctrl_inputs,
                       "layer_normalization":layer_normalization,
                       "probabilistic_loss_kind":probabilistic_loss_kind,
                       "distribution": distribution,
                       "z_min": z_min,
                       "z_max": z_max,
                       "n_latent":n_latent
                       })
        self.n_hidden_y = 2 * len(self.optimization_vars) if augment_ctrl_inputs else len(self.optimization_vars)
        self.inverter_optimizer = optax.adabelief(learning_rate=self.inverter_learning_rate)

        super().__init__(learning_rate, batch_size, load_path, n_hidden_x, n_out, n_layers, pars, q_vect, val_ratio,
                         nodes_at_step, n_epochs, savepath_tr_plots, stats_step, rel_tol, unnormalized_inputs,
                         normalize_target, stopping_rounds, subtract_mean_when_normalizing, causal_df,
                         probabilistic, probabilistic_loss_kind, **scengen_kwgs)


    def set_arch(self):
        z_max = jnp.array(self.z_max) if self.z_max is not None else self.z_max
        z_min = jnp.array(self.z_min) if self.z_min is not None else self.z_min
        self.optimizer = optax.adamw(learning_rate=self.learning_rate)
        self.model = PartiallyICNN(num_layers=self.n_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y,
                              features_out=self.n_out, features_latent=self.n_latent, init_type=self.init_type,
                              augment_ctrl_inputs=self.augment_ctrl_inputs, probabilistic=self.probabilistic,
                                   z_min=z_min, z_max=z_max)

        self.predict_batch = vmap(jitting_wrapper(predict_batch_picnn, self.model), in_axes=(None, 0))
        if self.causal_df is not None:
            causal_df = (~self.causal_df.astype(bool)).astype(float)
            causal_matrix = np.tile(causal_df.values, 2) if self.probabilistic else causal_df.values
            self.loss_fn = jitting_wrapper(causal_loss_fn, self.model, causal_matrix=causal_matrix, kind=self.probabilistic_loss_kind) \
                if not self.probabilistic else jitting_wrapper(probabilistic_causal_loss_fn, self.model, causal_matrix=causal_matrix, kind=self.probabilistic_loss_kind)
        else:
            self.loss_fn = jitting_wrapper(loss_fn, self.predict_batch) if not self.probabilistic else jitting_wrapper(probabilistic_loss_fn, self.predict_batch, kind=self.probabilistic_loss_kind, distribution=self.distribution)

        self.train_step = jitting_wrapper(partial(train_step, loss_fn=self.loss_fn), self.optimizer)


    @staticmethod
    def init_arch(nn_init, n_inputs_x=1, n_inputs_opt=1):
        "divides data into training and test sets "
        key1, key2 = random.split(random.key(0))
        x = random.normal(key1, (n_inputs_x, ))  # Dummy input data (for the first input)
        y = random.normal(key1, (n_inputs_opt, ))  # Dummy input data (for the second input)
        init_params = nn_init.init(key2, x, y)  # Initialization call
        return init_params
    def get_normalized_inputs(self, inputs, target=None):
        inputs = inputs.copy()
        self.to_be_normalized = [c for c in inputs.columns if c not in self.unnormalized_inputs] if self.unnormalized_inputs is not None else inputs.columns
        normalized_inputs = self.input_scaler.transform(inputs[self.to_be_normalized])
        inputs.loc[:, self.to_be_normalized] = normalized_inputs.copy().values

        x = inputs[[c for c in inputs.columns if not c in self.optimization_vars]].values
        y = inputs[self.optimization_vars].values

        if target is not None and self.normalize_target:
            target = target.copy()
            target = self.target_scaler.transform(target)
            target = target.values
        elif target is not None:
            target = target.values
        return (x, y), target

    def optimize(self, inputs, objective, n_iter=200, rel_tol=1e-4, recompile_obj=True, vanilla_gd=False, **objective_kwargs):
        rel_tol = rel_tol if rel_tol is not None else self.rel_tol
        inputs = inputs.copy()
        normalized_inputs, _ = self.get_normalized_inputs(inputs)
        x, y = normalized_inputs
        def _objective(y, x, **objective_kwargs):
            return objective(self.predict_batch(self.pars, [x, y]), y, **objective_kwargs)

        # if the objective changes from one call to another, you need to recompile it. Slower but necessary
        if recompile_obj or self.iterate is None:
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
        inputs.loc[:, self.to_be_normalized] = self.input_scaler.inverse_transform(inputs[self.to_be_normalized].values)
        target_opt = self.predict(inputs)

        y_opt = inputs.loc[:, self.optimization_vars].values.ravel()
        return y_opt, inputs, target_opt, values

class PIQCNN(PICNN):
    reproject: bool = True
    rec_stable: bool = False
    def __init__(self, learning_rate: float = 0.01, batch_size: int = None, load_path: str = None,
                 n_hidden_x: int = 100, n_out: int = 1, n_latent:int = 1, n_layers: int = 3, pars: dict = None, q_vect=None,
                 val_ratio=None, nodes_at_step=None, n_epochs: int = 10, savepath_tr_plots: str = None,
                 stats_step: int = 50, rel_tol: float = 1e-4, unnormalized_inputs=None, normalize_target=True,
                 stopping_rounds=5, subtract_mean_when_normalizing=False, causal_df=None, probabilistic=False,
                 probabilistic_loss_kind='maximum_likelihood', distribution='normal',
                 inverter_learning_rate: float = 0.1, optimization_vars: list = (),
                 target_columns: list = None, init_type='normal', augment_ctrl_inputs=False, layer_normalization=False,
                 z_min: jnp.array = None, z_max: jnp.array = None,
                 **scengen_kwgs):

        super().__init__(learning_rate, batch_size, load_path, n_hidden_x, n_out, n_latent, n_layers, pars, q_vect, val_ratio,
                         nodes_at_step, n_epochs, savepath_tr_plots, stats_step, rel_tol, unnormalized_inputs,
                         normalize_target, stopping_rounds, subtract_mean_when_normalizing, causal_df, probabilistic,
                         probabilistic_loss_kind, distribution, inverter_learning_rate, optimization_vars, target_columns, init_type,
                         augment_ctrl_inputs, layer_normalization, z_min, z_max, **scengen_kwgs)


    def set_arch(self):
        z_max = jnp.array(self.z_max) if self.z_max is not None else self.z_max
        z_min = jnp.array(self.z_min) if self.z_min is not None else self.z_min
        self.optimizer = optax.adamw(learning_rate=self.learning_rate)
        self.model = PartiallyIQCNN(num_layers=self.n_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y,
                              features_out=self.n_out, features_latent=self.n_latent, init_type=self.init_type,
                               augment_ctrl_inputs=self.augment_ctrl_inputs, probabilistic=self.probabilistic,
                                    z_min=z_min, z_max=z_max)
        self.predict_batch = vmap(jitting_wrapper(predict_batch_picnn, self.model), in_axes=(None, 0))
        self.loss_fn = jitting_wrapper(probabilistic_loss_fn, self.predict_batch, kind=self.probabilistic_loss_kind, distribution=self.distribution) if self.probabilistic else (
            jitting_wrapper(loss_fn, self.predict_batch))
        self.train_step = jitting_wrapper(partial(train_step, loss_fn=self.loss_fn), self.optimizer)


class PIQCNNSigmoid(PICNN):
    reproject: bool = True
    rec_stable: bool = False
    monotone: bool = True

    def __init__(self, learning_rate: float = 0.01, batch_size: int = None, load_path: str = None,
                 n_hidden_x: int = 100, n_out: int = 1, n_latent:int = 1, n_layers: int = 3, pars: dict = None, q_vect=None,
                 val_ratio=None, nodes_at_step=None, n_epochs: int = 10, savepath_tr_plots: str = None,
                 stats_step: int = 50, rel_tol: float = 1e-4, unnormalized_inputs=None, normalize_target=True,
                 stopping_rounds=5, subtract_mean_when_normalizing=False, causal_df=None, probabilistic=False,
                 probabilistic_loss_kind='maximum_likelihood', distribution='normal', inverter_learning_rate: float = 0.1, optimization_vars: list = (),
                 target_columns: list = None, init_type='normal', augment_ctrl_inputs=False, layer_normalization=False,
                 z_min: jnp.array = None, z_max: jnp.array = None,
                 **scengen_kwgs):

        super().__init__(learning_rate, batch_size, load_path, n_hidden_x, n_out, n_latent, n_layers, pars, q_vect, val_ratio,
                         nodes_at_step, n_epochs, savepath_tr_plots, stats_step, rel_tol, unnormalized_inputs,
                         normalize_target, stopping_rounds, subtract_mean_when_normalizing, causal_df, probabilistic,
                         probabilistic_loss_kind, distribution, inverter_learning_rate, optimization_vars, target_columns, init_type,
                         augment_ctrl_inputs, layer_normalization, z_min, z_max, **scengen_kwgs)


    def set_arch(self):
        z_max = jnp.array(self.z_max) if self.z_max is not None else self.z_max
        z_min = jnp.array(self.z_min) if self.z_min is not None else self.z_min
        self.optimizer = optax.adamw(learning_rate=self.learning_rate)
        self.model = PartiallyICNN(num_layers=self.n_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y,
                              features_out=self.n_out, features_latent=self.n_latent, init_type=self.init_type,
                               augment_ctrl_inputs=self.augment_ctrl_inputs, activation=nn.sigmoid,
                               rec_activation=nn.sigmoid, probabilistic=self.probabilistic,z_min=z_min,
                              z_max=z_max)
        self.predict_batch = vmap(jitting_wrapper(predict_batch_picnn, self.model), in_axes=(None, 0))
        self.loss_fn = jitting_wrapper(probabilistic_loss_fn, self.predict_batch, kind=self.probabilistic_loss_kind, distribution=self.distribution) if self.probabilistic else (
            jitting_wrapper(loss_fn, self.predict_batch))
        self.train_step = jitting_wrapper(partial(train_step, loss_fn=self.loss_fn), self.optimizer)

class RecStablePICNN(PICNN):
    reproject: bool = True
    rec_stable = True
    def __init__(self, learning_rate: float = 0.01, batch_size: int = None, load_path: str = None,
                 n_hidden_x: int = 100, n_out: int = 1, n_latent:int = 1, n_layers: int = 3, pars: dict = None, q_vect=None,
                 val_ratio=None, nodes_at_step=None, n_epochs: int = 10, savepath_tr_plots: str = None,
                 stats_step: int = 50, rel_tol: float = 1e-4, unnormalized_inputs=None, normalize_target=True,
                 stopping_rounds=5, subtract_mean_when_normalizing=False, causal_df=None, probabilistic=False,
                 probabilistic_loss_kind='maximum_likelihood', distribution='normal', inverter_learning_rate: float = 0.1, optimization_vars: list = (),
                 target_columns: list = None, init_type='normal', augment_ctrl_inputs=False,
                 layer_normalization=False, z_min: jnp.array = None, z_max: jnp.array = None, **scengen_kwgs):

        super().__init__(learning_rate, batch_size, load_path, n_hidden_x, n_out, n_latent, n_layers, pars, q_vect, val_ratio,
                         nodes_at_step, n_epochs, savepath_tr_plots, stats_step, rel_tol, unnormalized_inputs,
                         normalize_target, stopping_rounds, subtract_mean_when_normalizing, causal_df, probabilistic,
                         probabilistic_loss_kind, distribution, inverter_learning_rate, optimization_vars, target_columns, init_type,
                         augment_ctrl_inputs, layer_normalization, z_min, z_max, **scengen_kwgs)

    def set_arch(self):
        z_max = jnp.array(self.z_max) if self.z_max is not None else self.z_max
        z_min = jnp.array(self.z_min) if self.z_min is not None else self.z_min
        self.optimizer = optax.adamw(learning_rate=self.learning_rate)
        self.model = PartiallyICNN(num_layers=self.n_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y,
                              features_out=self.n_out, features_latent=self.n_latent, activation=nn.relu,
                              init_type=self.init_type, probabilistic=self.probabilistic,z_min=z_min,
                              z_max=z_max)
        self.predict_batch = vmap(jitting_wrapper(predict_batch_picnn, self.model), in_axes=(None, 0))
        self.loss_fn = jitting_wrapper(probabilistic_loss_fn, self.predict_batch, kind=self.probabilistic_loss_kind, distribution=self.distribution) if self.probabilistic else (
            jitting_wrapper(loss_fn, self.predict_batch))
        self.train_step = jitting_wrapper(partial(train_step, loss_fn=self.loss_fn), self.optimizer)


def structured_loss_fn(params, inputs, targets, model=None, objective=None):
    predictions = model(params, inputs)
    structured_loss = jnp.mean((predictions - targets) ** 2)
    objs_hat = objective(predictions, inputs[1])
    objs = objective(targets, inputs[1])
    objective_loss = jnp.mean((objs-objs_hat)**2)
    monotonic_objective_loss = monotonic_objective_right(objs_hat, objs)
    #fourier_loss = jnp.mean((unnormalized_fourier_transform(predictions, 20) - unnormalized_fourier_transform(targets, 20))**2)
    return structured_loss +  monotonic_objective_loss + objective_loss

def structured_probabilistic_loss_fn(params, inputs, targets, model=None, kind='maximum_likelihood', objective=None, distribution='normal'):
    out = model(params, inputs)
    predictions = out[:, :-1]
    sigma_square = out[:, -1]
    objs_hat = objective(predictions, inputs[1])
    objs = objective(targets, inputs[1])

    obj_loss = probabilistic_loss(objs_hat, objs, sigma_square, kind=kind, distribution=distribution)

    structured_loss = jnp.mean((predictions-targets) ** 2)
    monotonic_objective_loss = monotonic_objective_right(objs_hat, objs)
    #fourier_loss = jnp.mean((unnormalized_fourier_transform(predictions, 20) - unnormalized_fourier_transform(targets, 20))**2)
    return structured_loss + monotonic_objective_loss + obj_loss

@partial(vmap, in_axes=(0, None))
def unnormalized_fourier_transform(predictions, n_freq):
    """
    Projects the predictions over sin and cos base functions, and returns the coefficients
    :param predictions: original predictions in time domain
    :param n_freq: number of harmonitcs to project on
    :return:
    """
    # project predictions on cos and sin functions
    t = np.arange(len(predictions))
    sin_bases = np.array([np.sin(2 * np.pi * i * t / t[-1]) for i in range(1, n_freq + 1)]).T
    cos_bases = np.array([np.cos(2 * np.pi * i * t / t[-1]) for i in range(1, n_freq + 1)]).T
    bases = np.hstack([sin_bases, cos_bases])
    unnormalized_fc = predictions @ bases

    return unnormalized_fc

def monotonic_objective_right(objs_hat, objs):
    rank = jnp.argsort(objs)
    rank_hat = jnp.argsort(objs_hat)
    return -jnp.mean(jnp.corrcoef(rank, rank_hat)[0, 1])

def monotonic_objective_relax(objs_hat, objs):
    key = random.key(0)
    random_pairs = random.choice(key, len(objs), (len(objs)*100, 2))
    d = objs[random_pairs[:, 0]] - objs[random_pairs[:, 1]]
    d_hat = objs_hat[random_pairs[:, 0]] - objs_hat[random_pairs[:, 1]]
    discordant = (d * d_hat) < 0
    return jnp.mean(d*discordant)


class StructuredPICNN(PICNN):
    reproject: bool = True
    rec_stable: bool = False
    monotone: bool = True
    objective_fun=None
    def __init__(self, learning_rate: float = 0.01, batch_size: int = None, load_path: str = None,
                 n_hidden_x: int = 100, n_out: int = 1, n_latent:int = 1, n_layers: int = 3, pars: dict = None, q_vect=None,
                 val_ratio=None, nodes_at_step=None, n_epochs: int = 10, savepath_tr_plots: str = None,
                 stats_step: int = 50, rel_tol: float = 1e-4, unnormalized_inputs=None, normalize_target=True,
                 stopping_rounds=5, subtract_mean_when_normalizing=False, causal_df=None, probabilistic=False,
                 probabilistic_loss_kind='maximum_likelihood', distribution='normal', inverter_learning_rate: float = 0.1, optimization_vars: list = (),
                 target_columns: list = None, init_type='normal', augment_ctrl_inputs=False, layer_normalization=False,
                 objective_fun=None, z_min: jnp.array = None, z_max: jnp.array = None, **scengen_kwgs):

        self.objective_fun = objective_fun
        self.objective = vmap(objective_fun, in_axes=(0, 0))

        super().__init__(learning_rate, batch_size, load_path, n_hidden_x, n_out, n_latent, n_layers, pars, q_vect, val_ratio,
                         nodes_at_step, n_epochs, savepath_tr_plots, stats_step, rel_tol, unnormalized_inputs,
                         normalize_target, stopping_rounds, subtract_mean_when_normalizing, causal_df, probabilistic,
                         probabilistic_loss_kind, distribution, inverter_learning_rate, optimization_vars, target_columns, init_type,
                         augment_ctrl_inputs, layer_normalization, z_min, z_max, **scengen_kwgs)

    def set_arch(self):
        z_max = jnp.array(self.z_max) if self.z_max is not None else self.z_max
        z_min = jnp.array(self.z_min) if self.z_min is not None else self.z_min
        self.optimizer = optax.adamw(learning_rate=self.learning_rate)
        self.model = PartiallyICNN(num_layers=self.n_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y,
                              features_out=self.n_out, features_latent=self.n_latent, init_type=self.init_type,
                               augment_ctrl_inputs=self.augment_ctrl_inputs, activation=nn.sigmoid,
                               rec_activation=nn.sigmoid, probabilistic=self.probabilistic, structured=True,
                                   z_min=z_min, z_max=z_max)
        self.predict_batch = vmap(jitting_wrapper(predict_batch_picnn, self.model), in_axes=(None, 0))
        self.loss_fn = jitting_wrapper(structured_loss_fn, self.predict_batch, objective=self.objective) if not self.probabilistic \
            else jitting_wrapper(structured_probabilistic_loss_fn, self.predict_batch, kind=self.probabilistic_loss_kind, objective=self.objective, distribution=self.distribution)
        self.train_step = jitting_wrapper(partial(train_step, loss_fn=self.loss_fn), self.optimizer)



    def predict(self, inputs, return_sigma=False, return_obj=False, **kwargs):
        x, _ = self.get_normalized_inputs(inputs)
        y_hat = self.predict_batch(self.pars, x)
        y_hat = np.array(y_hat)

        if self.normalize_target:
            if self.probabilistic:
                y_hat[:, :-1] = self.target_scaler.inverse_transform(y_hat[:, :y_hat.shape[1]//2])
                y_hat[:, -1] = self.target_scaler.inverse_transform((y_hat[:, y_hat.shape[1] // 2:])**0.5) # this is wrong, please do not normalize target when probabilistic
            else:
                y_hat = self.target_scaler.inverse_transform(y_hat)

        if self.probabilistic:
            preds = pd.DataFrame(y_hat[:, :-1], index=inputs.index, columns=self.target_columns)
            sigma = pd.DataFrame(np.sign(y_hat[:, -1]) * np.abs(y_hat[:, -1])**0.5, index=inputs.index, columns=['sigma'])
            objs = pd.DataFrame(self.objective(y_hat[:, :-1], x[1]), index=inputs.index, columns=['objective'])
            if return_obj and return_sigma:
                return preds, objs, sigma
            elif return_obj:
                return preds, objs
            elif return_sigma:
                return preds, sigma
            else:
                return preds

        else:
            preds = pd.DataFrame(y_hat, index=inputs.index, columns=self.target_columns)
            objs = pd.DataFrame(self.objective(y_hat, x[1]), index=inputs.index, columns=['objective'])

            if return_obj:
                return preds, objs
            else:
                return preds


    def predict_quantiles(self, inputs, normalize=True, **kwargs):
        if normalize:
            y_hat, objs_hat, sigma_hat = self.predict(inputs, return_sigma=True, return_obj=True)
            objs_hat = objs_hat.values
            s = np.sign(sigma_hat.values.reshape(-1, 1))
            sigma_hat = np.abs(sigma_hat.values.reshape(-1, 1))
        else:
            y_hat = self.predict_batch(self.pars, inputs)
            y_hat = np.array(y_hat)
            sigma_hat = (np.abs(y_hat[:, [-1]]))** 0.5
            s = np.sign(y_hat[:, [-1]])
            objs_hat = self.objective(y_hat[:, :-1], inputs[1]).reshape(-1, 1)


        preds = np.expand_dims(objs_hat, -1) * np.ones((1, 1, len(self.q_vect)))
        for i, q in enumerate(self.q_vect):
            if self.distribution == 'normal':
                qs = sigma_hat * np.sqrt(2) * erfinv(2 * q - 1)
                preds[:, :, i] += qs
            elif self.distribution == 'log-normal':
                sg_hat_square = jnp.log(1 + sigma_hat ** 2 / objs_hat ** 2)
                sg_hat = jnp.sqrt(sg_hat_square)

                mu_hat = jnp.log(objs_hat ** 2 / jnp.sqrt(sigma_hat ** 2 + objs_hat ** 2))
                qp = mu_hat + sg_hat * np.sqrt(2) * erfinv(2 * q - 1)
                pos_qs = np.exp(qp)
                qn = mu_hat + sg_hat * np.sqrt(2) * erfinv(2 * (1-q) - 1)
                neg_qs = 2 * jnp.exp(mu_hat) - np.exp(qn)
                preds[:, :, i] = (s==-1)*neg_qs + (s==1)*pos_qs

        return preds



class LatentStructuredPICNN(PICNN):
    reproject: bool = True
    rec_stable: bool = False
    monotone: bool = True
    objective_fun=None
    encoder_neurons: np.array = None
    decoder_neurons: np.array = None
    def __init__(self, learning_rate: float = 0.01, batch_size: int = None, load_path: str = None,
                 n_hidden_x: int = 100, n_out: int = 1, n_latent:int = 1, n_layers: int = 3, pars: dict = None, q_vect=None,
                 val_ratio=None, nodes_at_step=None, n_epochs: int = 10, savepath_tr_plots: str = None,
                 stats_step: int = 50, rel_tol: float = 1e-4, unnormalized_inputs=None, normalize_target=True,
                 stopping_rounds=5, subtract_mean_when_normalizing=False, causal_df=None, probabilistic=False,
                 probabilistic_loss_kind='maximum_likelihood', distribution='normal', inverter_learning_rate: float = 0.1, optimization_vars: list = (),
                 target_columns: list = None, init_type='normal', augment_ctrl_inputs=False, layer_normalization=False,
                 objective_fun=None, z_min: jnp.array = None, z_max: jnp.array = None,
                 n_first_encoder:int=10, n_last_encoder:int=10, n_encoder_layers:int=3,
                 n_first_decoder:int=10, n_decoder_layers:int=3,
                 **scengen_kwgs):

        self.set_attr({"encoder_neurons":np.linspace(n_first_encoder, n_last_encoder, n_encoder_layers).astype(int),
                       "decoder_neurons":np.linspace(n_first_decoder, len(optimization_vars), n_decoder_layers).astype(int),
                       "n_first_encoder":n_first_encoder,
                       "n_last_encoder":n_last_encoder,
                       "n_encoder_layers":n_encoder_layers,
                       "n_first_decoder":n_first_decoder,
                       "n_decoder_layers":n_decoder_layers
                       })

        super().__init__(learning_rate, batch_size, load_path, n_hidden_x, n_out, n_latent, n_layers, pars, q_vect, val_ratio,
                         nodes_at_step, n_epochs, savepath_tr_plots, stats_step, rel_tol, unnormalized_inputs,
                         normalize_target, stopping_rounds, subtract_mean_when_normalizing, causal_df, probabilistic,
                         probabilistic_loss_kind, distribution, inverter_learning_rate, optimization_vars, target_columns, init_type,
                         augment_ctrl_inputs, layer_normalization, z_min, z_max, **scengen_kwgs)
    def set_arch(self):
        z_max = jnp.array(self.z_max) if self.z_max is not None else self.z_max
        z_min = jnp.array(self.z_min) if self.z_min is not None else self.z_min
        self.optimizer = optax.adamw(learning_rate=self.learning_rate)
        self.model = LatentPartiallyICNN(num_layers=self.n_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y,
                              features_out=self.n_out, features_latent=self.n_latent, init_type=self.init_type,
                              augment_ctrl_inputs=self.augment_ctrl_inputs, probabilistic=self.probabilistic,
                                   z_min=z_min, z_max=z_max, encoder_neurons=self.encoder_neurons, decoder_neurons=self.decoder_neurons)

        self.predict_batch_training = vmap(jitting_wrapper(predict_batch_latent_picnn, self.model, mode='all'), in_axes=(None, 0))
        self.predict_batch = vmap(jitting_wrapper(predict_batch_latent_picnn, self.model, mode='prediction'), in_axes=(None, 0))
        self.loss_fn = jitting_wrapper(embedded_loss_fn, self.predict_batch_training) if not self.probabilistic else jitting_wrapper(probabilistic_loss_fn, self.predict_batch, kind=self.probabilistic_loss_kind, distribution=self.distribution)
        self.train_step = jitting_wrapper(partial(train_step, loss_fn=self.loss_fn), self.optimizer)


    def optimize(self, inputs, objective, n_iter=200, rel_tol=1e-4, recompile_obj=True, vanilla_gd=False, **objective_kwargs):
        rel_tol = rel_tol if rel_tol is not None else self.rel_tol
        inputs = inputs.copy()
        normalized_inputs, _ = self.get_normalized_inputs(inputs)
        x, y = normalized_inputs

        def _objective(ctrl_embedding, x, **objective_kwargs):
            preds = latent_pred(self.pars, self.model, x, ctrl_embedding)
            ctrl_reconstruct = decode(self.pars, self.model, x, ctrl_embedding)
            preds_reconstruct, _ , _ = self.predict_batch_training(self.pars, [x, ctrl_reconstruct])
            implicit_regularization_loss = jnp.mean((preds_reconstruct - preds)**2)
            return objective(preds, ctrl_reconstruct, **objective_kwargs) + implicit_regularization_loss
        self._objective = _objective
        # if the objective changes from one call to another, you need to recompile it. Slower but necessary
        if recompile_obj or self.iterate is None:
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

        _, ctrl_embedding, ctrl_reconstruct = self.predict_batch_training(self.pars, [x, y])
        opt_state = self.inverter_optimizer.init(ctrl_embedding)
        ctrl_embedding, values_old = self.iterate(x, ctrl_embedding, opt_state, **objective_kwargs)
        values_init = np.copy(values_old)


        # do 10 iterations at a time to speed up, check for convergence
        for i in range(n_iter//10):
            ctrl_embedding, values = self.iterate(x, ctrl_embedding, opt_state, **objective_kwargs)
            rel_improvement = (values_old - values) / (np.abs(values_old)+ 1e-12)
            values_old = values
            if rel_improvement < rel_tol:
                break
        print('optimization terminated at iter {}, final objective value: {:0.2e} '
                         'rel improvement: {:0.2e}'.format((i+1)*10, values,
                                                          (values_init-values)/(np.abs(values_init)+1e-12)))


        ctrl = decode(self.pars, self.model, x, ctrl_embedding)

        y_hat_from_latent = latent_pred(self.pars, self.model, x, ctrl_embedding)
        y_hat_from_ctrl_reconstructed, _ ,_ = self.predict_batch_training(self.pars, [x, ctrl])

        plt.plot(y_hat_from_latent.ravel())
        plt.plot(y_hat_from_ctrl_reconstructed.ravel())

        inputs.loc[:, self.optimization_vars] = ctrl.ravel()
        inputs.loc[:, [c for c in inputs.columns if c not in  self.optimization_vars]] = x.ravel()
        inputs.loc[:, self.to_be_normalized] = self.input_scaler.inverse_transform(inputs[self.to_be_normalized].values)
        target_opt = self.predict(inputs)

        y_opt = inputs.loc[:, self.optimization_vars].values.ravel()
        return y_opt, inputs, target_opt, values