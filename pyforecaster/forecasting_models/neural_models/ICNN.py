from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import linen as nn
from jax import jit as jit
from jax import value_and_grad, jvp, vmap, random
from scipy.special import erfinv

from pyforecaster.forecasting_models.neural_models.base_nn import NN, train_step, loss_fn, probabilistic_loss_fn, \
    probabilistic_loss, FeedForwardModule
from pyforecaster.forecasting_models.neural_models.layers import PICNNLayer
from pyforecaster.forecasting_models.neural_models.neural_utils import identity, jitting_wrapper


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

def structured_loss_fn(params, inputs, targets, model=None, objective=None):
    predictions = model(params, inputs)
    structured_loss = jnp.mean((predictions - targets) ** 2)
    objs_hat = objective(predictions, inputs[1])
    objs = objective(targets, inputs[1])
    objective_loss = jnp.mean((objs-objs_hat)**2)
    monotonic_objective_loss = monotonic_objective_right(objs_hat, objs)
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
    return structured_loss + monotonic_objective_loss + obj_loss

def embedded_loss_fn(params, inputs, targets, model=None, full_model=None):
    predictions, ctrl_embedding, ctrl_reconstruction = model(params, inputs)
    predictions_from_ctrl_reconstr, _, _ = model(params, [inputs[0], ctrl_reconstruction])
    target_loss = jnp.mean((predictions - targets) ** 2)
    ctrl_reconstruction_loss = jnp.mean((ctrl_reconstruction - inputs[1]) ** 2)
    obj_reconstruction_loss = jnp.mean((predictions - predictions_from_ctrl_reconstr) ** 2)

    #output_grads_wrt_embedding = vmap(lambda ctrl_emb, exog: jax.jacfwd(lambda ctrl_emb: latent_pred(params, full_model, exog, ctrl_emb).ravel())(ctrl_emb), in_axes=(0, 0))(ctrl_embedding, inputs[0]).squeeze()
    #conic_loss = jnp.mean((output_grads_wrt_embedding-2*ctrl_embedding) ** 2)
    kl_gaussian_loss = jnp.mean(jnp.mean(ctrl_embedding**2, axis=0) + jnp.mean(ctrl_embedding, axis=0)**2
                                - jnp.log(jnp.mean(ctrl_embedding**2, axis=0)) - 1) # (sigma^2 + mu^2 - log(sigma^2) - 1)/2

    return target_loss + ctrl_reconstruction_loss + kl_gaussian_loss + 100*obj_reconstruction_loss


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




class PICNN_module(nn.Module):
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
        z = jnp.ones(self.features_latent)  # Initialize z_0 to be the same shape as y
        for i in range(self.num_layers):
            prediction_layer = i == self.num_layers -1
            features_out = self.features_out if prediction_layer else self.features_latent
            features_latent = self.features_latent if self.features_latent is not None else self.features_out
            u, z = PICNNLayer(features_x=self.features_x, features_y=self.features_y, features_out=features_out,
                              features_latent=features_latent,
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


class LatentPICNN_module(nn.Module):
    num_layers: int
    features_x: int
    features_y: int
    features_out: int
    features_latent: int
    n_encoder_layers: int
    n_decoder_layers:int
    n_embeddings: int
    n_control: int
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

        features_latent_encoder = self.features_latent if self.features_latent is not None else self.n_embeddings
        features_latent_decoder = self.features_latent if self.features_latent is not None else self.n_control
        features_latent_preds = self.features_latent if self.features_latent is not None else self.features_out


        features_y = self.n_embeddings*2 if self.augment_ctrl_inputs else self.n_embeddings
        self.encoder = PICNN_module(num_layers=self.n_encoder_layers, features_x=self.features_x, features_y=self.features_y,
                              features_out=self.n_embeddings, features_latent=features_latent_encoder, init_type=self.init_type,
                              augment_ctrl_inputs=self.augment_ctrl_inputs, probabilistic=self.probabilistic,
                                   z_min=self.z_min, z_max=self.z_max, name='PICNNLayer_encoder')

        self.decoder = PICNN_module(num_layers=self.n_decoder_layers, features_x=self.features_x, features_y=features_y,
                              features_out=self.n_control, features_latent=features_latent_decoder, init_type=self.init_type,
                              augment_ctrl_inputs=self.augment_ctrl_inputs, probabilistic=self.probabilistic, rec_activation=nn.relu,
                                   z_min=self.z_min, z_max=self.z_max, name='PICNNLayer_decoder_monotone')


        self.picnn = PICNN_module(num_layers=self.num_layers, features_x=self.features_x, features_y=features_y,
                              features_out=self.features_out, features_latent=features_latent_preds, init_type=self.init_type,
                              augment_ctrl_inputs=self.augment_ctrl_inputs, probabilistic=self.probabilistic,
                                   z_min=self.z_min, z_max=self.z_max, name='PICNNLayer_picnn')

    def __call__(self, x, y):

        ctrl_embedding = self.encoder(x, y)
        z = self.picnn(x, ctrl_embedding)
        ctrl_reconstruction = self.decoder(x, ctrl_embedding)


        return z, ctrl_embedding, ctrl_reconstruction

    def decode(self, x, ctrl_embedding):
        return self.decoder(x, ctrl_embedding)

    def encode(self, x, ctrl_embedding):
        return self.encoder(x, ctrl_embedding)

    def latent_pred(self, x, ctrl_embedding):
        return self.picnn(x, ctrl_embedding)


class PartiallyIQCNN_module(nn.Module):
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
        z = jnp.ones(self.features_out)  # Initialize z_0 to be the same shape as y
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
    inverter_class = None
    def __init__(self, learning_rate: float = 0.01, batch_size: int = None, load_path: str = None,
                 n_hidden_x: int = 100, n_out: int = 1, n_latent:int = 1, n_layers: int = 3, pars: dict = None, q_vect=None,
                 val_ratio=None, nodes_at_step=None, n_epochs: int = 10, savepath_tr_plots: str = None,
                 stats_step: int = 50, rel_tol: float = 1e-4, unnormalized_inputs=None, normalize_target=True,
                 stopping_rounds=5, subtract_mean_when_normalizing=False, causal_df=None, probabilistic=False,
                 probabilistic_loss_kind='maximum_likelihood', distribution = 'normal', inverter_learning_rate: float = 0.1, optimization_vars: list = (),
                 target_columns: list = None, init_type='normal', augment_ctrl_inputs=False, layer_normalization=False,
                 z_min: jnp.array = None, z_max: jnp.array = None, inverter_class=None,
                 **scengen_kwgs):

        inverter_class = optax.adabelief if inverter_class is None else inverter_class

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
                       "n_latent":n_latent,
                       "inverter_class":inverter_class
                       })

        self.n_hidden_y = 2 * len(self.optimization_vars) if augment_ctrl_inputs else len(self.optimization_vars)
        self.inverter_optimizer = inverter_class(learning_rate=self.inverter_learning_rate)

        super().__init__(learning_rate, batch_size, load_path, n_hidden_x, n_out, n_layers, pars, q_vect, val_ratio,
                         nodes_at_step, n_epochs, savepath_tr_plots, stats_step, rel_tol, unnormalized_inputs,
                         normalize_target, stopping_rounds, subtract_mean_when_normalizing, causal_df,
                         probabilistic, probabilistic_loss_kind, **scengen_kwgs)


    def set_arch(self):
        z_max = jnp.array(self.z_max) if self.z_max is not None else self.z_max
        z_min = jnp.array(self.z_min) if self.z_min is not None else self.z_min
        self.optimizer = optax.adamw(learning_rate=self.learning_rate)
        self.model = PICNN_module(num_layers=self.n_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y,
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



def decode(params, model, x, ctrl_embedding):
    def decoder(lpicnn ):
        return lpicnn.decode(x, ctrl_embedding)

    return nn.apply(decoder, model)(params)

def encode(params, model, x, ctrl_embedding):
    def encoder(lpicnn ):
        return lpicnn.encode(x, ctrl_embedding)

    return nn.apply(encoder, model)(params)


def latent_pred(params, model, x, ctrl_embedding):
    def _latent_pred(lpicnn ):
        return lpicnn.latent_pred(x, ctrl_embedding)

    return nn.apply(_latent_pred, model)(params)



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
        self.model = PartiallyIQCNN_module(num_layers=self.n_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y,
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
        self.model = PICNN_module(num_layers=self.n_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y,
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
        self.model = PICNN_module(num_layers=self.n_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y,
                              features_out=self.n_out, features_latent=self.n_latent, activation=nn.relu,
                              init_type=self.init_type, probabilistic=self.probabilistic,z_min=z_min,
                              z_max=z_max)
        self.predict_batch = vmap(jitting_wrapper(predict_batch_picnn, self.model), in_axes=(None, 0))
        self.loss_fn = jitting_wrapper(probabilistic_loss_fn, self.predict_batch, kind=self.probabilistic_loss_kind, distribution=self.distribution) if self.probabilistic else (
            jitting_wrapper(loss_fn, self.predict_batch))
        self.train_step = jitting_wrapper(partial(train_step, loss_fn=self.loss_fn), self.optimizer)



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
        self.model = PICNN_module(num_layers=self.n_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y,
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
    objective_fun=None
    encoder_neurons: np.array = None
    decoder_neurons: np.array = None
    n_embeddings: int = 10
    def __init__(self, learning_rate: float = 0.01, batch_size: int = None, load_path: str = None,
                 n_hidden_x: int = 100, n_out: int = 1, n_latent:int = None, n_layers: int = 3, pars: dict = None, q_vect=None,
                 val_ratio=None, nodes_at_step=None, n_epochs: int = 10, savepath_tr_plots: str = None,
                 stats_step: int = 50, rel_tol: float = 1e-4, unnormalized_inputs=None, normalize_target=True,
                 stopping_rounds=5, subtract_mean_when_normalizing=False, causal_df=None, probabilistic=False,
                 probabilistic_loss_kind='maximum_likelihood', distribution='normal', inverter_learning_rate: float = 0.1, optimization_vars: list = (),
                 target_columns: list = None, init_type='normal', augment_ctrl_inputs=False, layer_normalization=False,
                 objective_fun=None, z_min: jnp.array = None, z_max: jnp.array = None,
                 n_encoder_layers:int=3,
                 n_embeddings:int=10, n_decoder_layers:int=3,
                 **scengen_kwgs):

        self.set_attr({"n_embeddings":n_embeddings,
                       "n_encoder_layers":n_encoder_layers,
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
        self.model = LatentPICNN_module(num_layers=self.n_layers,n_encoder_layers=self.n_encoder_layers, n_decoder_layers=self.n_decoder_layers, features_x=self.n_hidden_x, features_y=self.n_hidden_y,
                              features_out=self.n_out, features_latent=self.n_latent, init_type=self.init_type,
                              augment_ctrl_inputs=self.augment_ctrl_inputs, probabilistic=self.probabilistic,
                                   z_min=z_min, z_max=z_max, n_embeddings=self.n_embeddings, n_control=len(self.optimization_vars))

        self.predict_batch_training = vmap(jitting_wrapper(predict_batch_latent_picnn, self.model, mode='all'), in_axes=(None, 0))
        self.predict_batch = vmap(jitting_wrapper(predict_batch_latent_picnn, self.model, mode='prediction'), in_axes=(None, 0))
        self.loss_fn = jitting_wrapper(embedded_loss_fn, self.predict_batch_training, full_model=self.model) if not self.probabilistic else jitting_wrapper(probabilistic_loss_fn, self.predict_batch, kind=self.probabilistic_loss_kind, distribution=self.distribution)
        self.train_step = jitting_wrapper(partial(train_step, loss_fn=self.loss_fn), self.optimizer)


    def optimize(self, inputs, objective, n_iter=200, rel_tol=1e-4, recompile_obj=True, vanilla_gd=False, **objective_kwargs):
        rel_tol = rel_tol if rel_tol is not None else self.rel_tol
        inputs = inputs.copy()
        normalized_inputs, _ = self.get_normalized_inputs(inputs)
        x, y = normalized_inputs

        def _preds_and_regularizations(ctrl_embedding, x):
            preds = latent_pred(self.pars, self.model, x, ctrl_embedding)
            ctrl_reconstruct = decode(self.pars, self.model, x, ctrl_embedding)
            preds_reconstruct, _ , _ = self.predict_batch_training(self.pars, [x, ctrl_reconstruct])
            # implicit_regularization_loss = jnp.mean((preds_reconstruct - preds)**2)
            ctrl_embedding_reconstruct = encode(self.pars, self.model, x, ctrl_reconstruct)
            implicit_regularization_on_ctrl_loss = jnp.mean((ctrl_embedding_reconstruct - ctrl_embedding)**2)
            regularization_loss = implicit_regularization_on_ctrl_loss #+ implicit_regularization_loss
            return preds, ctrl_reconstruct, regularization_loss

        def _objective(ctrl_embedding, x, weight=0, **objective_kwargs):
            preds, ctrl_reconstruct, regularization_loss = _preds_and_regularizations(ctrl_embedding, x)
            return objective(preds, ctrl_reconstruct, **objective_kwargs) + (weight * regularization_loss).ravel()[0]

        self._objective = _objective
        # if the objective changes from one call to another, you need to recompile it. Slower but necessary
        if recompile_obj or self.iterate is None:
            @jit
            def iterate(x, y, opt_state, lagrangian=0, **objective_kwargs):
                ctrl_embedding_history = []
                values_history = []
                for i in range(10):
                    ctrl_embedding_history.append(y)
                    values, grads = value_and_grad(partial(_objective, weight=lagrangian, **objective_kwargs))(y, x)
                    #lagrangian = lagrangian +  0.01 * self.inverter_learning_rate  * jnp.maximum(_preds_and_regularizations(y, x)[-1], 0)
                    if vanilla_gd:
                        y -= grads * 1e-1
                    else:
                        updates, opt_state = self.inverter_optimizer.update(grads, opt_state, y)
                        y = optax.apply_updates(y, updates)
                    values_history.append(values)
                return y, values, ctrl_embedding_history, values_history, lagrangian
            self.iterate = iterate

        _, ctrl_embedding, ctrl_reconstruct = self.predict_batch_training(self.pars, [x, y])
        opt_state = self.inverter_optimizer.init(ctrl_embedding)
        ctrl_embedding, values_old, ctrl_embedding_history_0, values_history_0, lagrangian = self.iterate(x, ctrl_embedding, opt_state, **objective_kwargs)
        values_init = np.copy(values_old)

        l = vmap(jit(partial(latent_pred, params=self.pars, model=self.model, x=x)), in_axes=(0))

        # do 10 iterations at a time to speed up, check for convergence
        ctrl_history = [np.vstack(ctrl_embedding_history_0)]
        sol_history = [np.vstack(l(ctrl_embedding=np.vstack(ctrl_embedding_history_0)))]
        lagrangian = jnp.array(0)
        init_constraint = 0
        for i in range(n_iter//10):
            ctrl_embedding, values, ctrl_embedding_history, values_history, lagrangian = self.iterate(x, ctrl_embedding, opt_state, lagrangian=lagrangian, **objective_kwargs)
            rel_improvement = (values_old - values) / (np.abs(values_old)+ 1e-12)
            values_old = values
            if i%10==0:
                print(values)

            if rel_improvement < rel_tol:
                break
            ctrl_history.append(np.vstack(ctrl_embedding_history))
            sol_history.append(np.vstack(l(ctrl_embedding=np.vstack(ctrl_embedding_history))))

        print('optimization terminated at iter {}, final objective value: {:0.2e} '
                         'rel improvement: {:0.2e}'.format((i+1)*10, values,
                                                          (values_init-values)/(np.abs(values_init)+1e-12)))


        ctrl = decode(self.pars, self.model, x, ctrl_embedding)

        y_hat_from_latent = l(ctrl_embedding=ctrl_embedding)
        y_hat_from_ctrl_reconstructed, _ ,_ = self.predict_batch_training(self.pars, [x, ctrl])


        inputs.loc[:, self.optimization_vars] = ctrl.ravel()
        inputs.loc[:, [c for c in inputs.columns if c not in  self.optimization_vars]] = x.ravel()
        inputs.loc[:, self.to_be_normalized] = self.input_scaler.inverse_transform(inputs[self.to_be_normalized].values)
        target_opt = self.predict(inputs)

        y_opt = inputs.loc[:, self.optimization_vars].values.ravel()
        return y_opt, inputs, target_opt, values, ctrl_embedding, np.vstack(ctrl_history), np.vstack(sol_history)
