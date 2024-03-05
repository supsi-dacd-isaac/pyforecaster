import jax
import pandas as pd
from flax import linen as nn
import jax.numpy as jnp
from pyforecaster.forecasting_models.neural_models.layers import CausalInvertibleLayer
from pyforecaster.forecasting_models.neural_models.base_nn import NN, predict_batch,  train_step, probabilistic_loss_fn, FeedForwardModule
import optax
from jax import vmap
from pyforecaster.forecasting_models.neural_models.neural_utils import jitting_wrapper
from functools import partial
import numpy as np

def identity(params, err):
    return err

def loss_fn(params, inputs, targets, model=None):
    predictions = model(params, inputs)
    kl_gaussian_loss = jnp.mean(jnp.mean(predictions**2, axis=0) + jnp.mean(predictions, axis=0)**2
                                - jnp.log(jnp.mean(predictions**2, axis=0)) - 1)
    return kl_gaussian_loss

def end_to_end_loss_fn(params, inputs, targets, model=None, embedder=None, inverter=identity):
    e_preds = model(params, inputs)
    e_target = embedder(params, jnp.hstack([inputs[:, targets.shape[1]:], targets]))[:, :targets.shape[1]]
    e_preds = inverter(params, e_preds)
    e_target = inverter(params, e_target)
    err = e_target - e_preds
    persistent_error = e_target - jnp.roll(e_target, 1)
    skill_score = jnp.mean(err**2) / jnp.mean(persistent_error**2)
    return skill_score


class CausalInvertibleModule(nn.Module):
    num_layers: int = 3
    features: int = 32

    def setup(self):
        self.layers = [CausalInvertibleLayer(prediction_layer=l==self.num_layers-1, features=self.features) for l in range(self.num_layers)]
    def __call__(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x

    def invert(self, y):
        for i in range(self.num_layers-1, -1, -1):
            y = self.layers[i].invert(y)
        return y

    def predict(self, x):
        for i in range(self.num_layers):
            x = self.layers[i].predict(x)
        return x

class EndToEndCausalInvertibleModule(nn.Module):
    num_embedding_layers: int = 3
    num_prediction_layers: int = 3
    features_embedding: int = 32
    features_prediction: int = 32
    n_out: int = 32
    activation: callable = nn.relu
    def setup(self):
        self.embedder = CausalInvertibleModule(num_layers=self.num_embedding_layers, features=self.features_embedding)
        self.predictor = FeedForwardModule(n_layers=np.hstack([(np.ones(self.num_prediction_layers-1)*self.features_prediction).astype(int), self.n_out]))
        self.invert_fun = jax.jit(partial(self.inverter, embedder=self.embedder))
    def __call__(self, x):
        embeddings = x.copy()
        embeddings = self.embedder(embeddings)
        preds = self.predictor(embeddings)
        return preds


    def invert(self, y):
        y = self.embedder.invert(y)
        return y

    def embed(self, x):
        return self.embedder.predict(x)

    @staticmethod
    def inverter(x, embedder):
        return embedder.invert(x)



class CausalInvertibleNN(NN):
    distribution = 'normal'
    end_to_end = False
    n_prediction_layers = 3
    n_hidden_y = 200
    def __init__(self, learning_rate: float = 0.01, batch_size: int = None, load_path: str = None,
                 n_hidden_x: int = 100, n_out: int = None, n_layers: int = 3, pars: dict = None, q_vect=None,
                 val_ratio=None, nodes_at_step=None, n_epochs: int = 10, savepath_tr_plots: str = None,
                 stats_step: int = 50, rel_tol: float = 1e-4, unnormalized_inputs=None, normalize_target=False,
                 stopping_rounds=5, subtract_mean_when_normalizing=False, causal_df=None, probabilistic=False,
                 probabilistic_loss_kind='maximum_likelihood', end_to_end='none', n_prediction_layers=3,
                 n_hidden_y=200,
                 **scengen_kwgs):

        self.set_attr({"end_to_end": end_to_end,
                       "n_prediction_layers": n_prediction_layers,
                       "num_embedding_layers": n_hidden_y})

        super().__init__(learning_rate, batch_size, load_path, n_hidden_x, n_out, n_layers, pars, q_vect, val_ratio,
                         nodes_at_step, n_epochs, savepath_tr_plots, stats_step, rel_tol, unnormalized_inputs,
                         normalize_target, stopping_rounds, subtract_mean_when_normalizing, causal_df, probabilistic,
                         probabilistic_loss_kind, **scengen_kwgs)

    def set_arch(self):
        self.optimizer = optax.adabelief(learning_rate=self.learning_rate)
        self.model = EndToEndCausalInvertibleModule(num_embedding_layers=self.n_layers, num_prediction_layers=self.n_prediction_layers, features_prediction=self.n_hidden_y, features_embedding=self.n_hidden_x, n_out=self.n_out) if (
            self.end_to_end in ['full', 'quasi']) else CausalInvertibleModule(num_layers=self.n_layers, features=self.n_hidden_x)

        self.predict_batch = vmap(jitting_wrapper(predict_batch, self.model), in_axes=(None, 0))
        if self.end_to_end in ['full', 'quasi']:
            self.loss_fn = jitting_wrapper(end_to_end_loss_fn, self.predict_batch,
                                           embedder=vmap(jitting_wrapper(embed, self.model), in_axes=(None, 0)),
                                           inverter=vmap(jitting_wrapper(invert, self.model), in_axes=(None, 0)) if
                                           self.end_to_end == 'full' else identity)
        else:
            self.loss_fn = jitting_wrapper(loss_fn, self.predict_batch)

        self.train_step = jitting_wrapper(partial(train_step, loss_fn=self.loss_fn), self.optimizer)


    def invert(self, y):
        y_np = invert(self.pars, y.values, self.model)
        if self.input_scaler is not None:
            y_np = self.input_scaler.inverse_transform(y_np)
        return pd.DataFrame(y_np, columns=y.columns)

    def fit(self, inputs, targets, n_epochs=None, savepath_tr_plots=None, stats_step=None, rel_tol=None):
        # You really don't want to normalize in here. Pre normalize your data if needed
        self.normalize_target = False
        self.unnormalized_inputs = inputs.columns

        if self.end_to_end in ['full', 'quasi']:
            assert targets.shape[1] == self.n_out, 'End to end models require the number of targets to be equal to n_out'

        return super().fit(inputs, targets, n_epochs, savepath_tr_plots, stats_step, rel_tol)
    def predict(self, inputs, return_sigma=False, **kwargs):
        x, _ = self.get_normalized_inputs(inputs)

        if self.end_to_end in ['full', 'quasi']:
            embeddings = embed(self.pars, x, self.model)
            embeddings_hat = self.predict_batch(self.pars, x)
            y_hat = invert(self.pars, embeddings_hat, self.model)

            # embedding-predicted embedding distributions
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize = (10, 6))
            ax[0].hist(np.array(embeddings.ravel()), bins=100, alpha=0.5, density=True, label='past embedding')
            ax[0].hist(np.array(embeddings_hat.ravel()), bins=100, alpha=0.5, density=True, label='forecasted embedding')
            ax[0].legend()

            # inputs-forecast distributions
            if self.target_scaler is not None:
                y_hat = self.target_scaler.inverse_transform(y_hat)
            try:
                ax[1].hist(np.array(y_hat.ravel()), bins=100, alpha=0.5, density=True, label='forecast ')
            except:
                print('Error in plotting forecast distribution')
            ax[1].hist(np.array(inputs.values.ravel()), bins=100, alpha=0.5, density=True, label='inputs')
            ax[1].legend()
        else:
            y_hat = self.predict_batch(self.pars, x)
            y_hat = np.array(y_hat)
            if self.normalize_target:
                y_hat = self.target_scaler.inverse_transform(y_hat)

        y_hat = pd.DataFrame(y_hat, index=inputs.index, columns=self.target_columns)

        return y_hat


def invert(params, y, model):
    def inverter(lpicnn):
        return lpicnn.invert(y)

    return nn.apply(inverter, model)(params)

def embed(params, y, model):
    def embedder(lpicnn):
        return lpicnn.embed(y)

    return nn.apply(embedder, model)(params)