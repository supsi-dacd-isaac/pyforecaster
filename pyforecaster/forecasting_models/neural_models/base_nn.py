import pickle as pk

from inspect import getmro
from os.path import join
from typing import Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from flax import linen as nn
from functools import partial
from jax import random, value_and_grad, vmap
from jax.scipy.special import erfinv
from scipy.special import erfinv
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pyforecaster.forecaster import ScenarioGenerator
from pyforecaster.forecasting_models.neural_models.neural_utils import jitting_wrapper, reproject_weights


def train_step(params, optimizer_state, inputs_batch, targets_batch, model=None, loss_fn=None, **kwargs):
    values, grads = value_and_grad(loss_fn)(params, inputs_batch, targets_batch, **kwargs)
    updates, opt_state = model.update(grads, optimizer_state, params)
    return optax.apply_updates(params, updates), opt_state, values

def predict_batch(pars, inputs, model=None):
    return model.apply(pars, inputs)


def loss_fn(params, inputs, targets, model=None):
    predictions = model(params, inputs)
    return jnp.mean((predictions - targets) ** 2)



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
