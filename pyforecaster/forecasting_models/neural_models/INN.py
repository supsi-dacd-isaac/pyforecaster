import jax
import pandas as pd
from flax import linen as nn
import jax.numpy as jnp
from pyforecaster.forecasting_models.neural_models.layers import CausalInvertibleLayer
from pyforecaster.forecasting_models.neural_models.base_nn import NN, predict_batch, loss_fn, train_step, probabilistic_loss_fn, FeedForwardModule
import optax
from jax import vmap
from pyforecaster.forecasting_models.neural_models.neural_utils import jitting_wrapper
from functools import partial
import numpy as np
def loss_fn(params, inputs, targets, model=None):
    predictions = model(params, inputs)
    kl_gaussian_loss = jnp.mean(jnp.mean(predictions**2, axis=0) + jnp.mean(predictions, axis=0)**2
                                - jnp.log(jnp.mean(predictions**2, axis=0)) - 1)
    return kl_gaussian_loss

def end_to_end_loss_fn(params, inputs, targets, model=None):
    x = model(params, inputs)
    x_preds = x[:, :x.shape[1]//2]
    x_futures = x[:, x.shape[1]//2:]
    mse_loss = jnp.mean((x_preds - x_futures)**2)

    return mse_loss


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
    activation: callable = nn.relu
    def setup(self):
        self.embedder = CausalInvertibleModule(num_layers=self.num_embedding_layers, features=self.features_embedding)
        self.predictor = FeedForwardModule(n_layers=np.hstack([(np.ones(self.num_prediction_layers-1)*self.features_prediction).astype(int), self.features_embedding//2]))
        self.invert_fun = jax.jit(partial(self.inverter, embedder=self.embedder))
    def __call__(self, x):
        half_length = x.shape[0]//2
        embeddings = x.copy()

        embeddings = self.embedder(embeddings)
        e_past = embeddings[:half_length]
        e_future = embeddings[half_length:]

        e_pred = self.predictor(e_past)

        out = jnp.hstack([e_pred, e_future])

        #y_hat = self.invert_fun(jnp.hstack([e_past, e_pred]))[half_length:]

        #out = jnp.hstack([y_hat, x[half_length:]])

        return out


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
                 stats_step: int = 50, rel_tol: float = 1e-4, unnormalized_inputs=None, normalize_target=True,
                 stopping_rounds=5, subtract_mean_when_normalizing=False, causal_df=None, probabilistic=False,
                 probabilistic_loss_kind='maximum_likelihood', end_to_end=False, n_prediction_layers=3,
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
        self.optimizer = optax.adamw(learning_rate=self.learning_rate)
        self.model = EndToEndCausalInvertibleModule(num_embedding_layers=self.n_layers, num_prediction_layers=self.n_prediction_layers, features_prediction=self.n_hidden_y, features_embedding=self.n_hidden_x) if (
            self.end_to_end) else CausalInvertibleModule(num_layers=self.n_layers, features=self.n_hidden_x)

        self.predict_batch = vmap(jitting_wrapper(predict_batch, self.model), in_axes=(None, 0))
        loss_function = end_to_end_loss_fn if self.end_to_end else loss_fn
        self.loss_fn = jitting_wrapper(loss_function, self.predict_batch)
        self.train_step = jitting_wrapper(partial(train_step, loss_fn=self.loss_fn), self.optimizer)


    def invert(self, y):
        y_np = invert(self.pars, self.model, y.values)
        if self.input_scaler is not None:
            y_np = self.input_scaler.inverse_transform(y_np)
        return pd.DataFrame(y_np, columns=y.columns)

    def predict(self, inputs, return_sigma=False, **kwargs):
        x, _ = self.get_normalized_inputs(inputs)

        if self.end_to_end:
            embeddings = embed(self.pars, self.model, x)
            embeddings_hat = self.predict_batch(self.pars, x)[:, :inputs.shape[1]//2]
            y_embedded = np.hstack([embeddings[:, :inputs.shape[1]//2], embeddings_hat])
            y_np = invert(self.pars, self.model, y_embedded)

            # embedding-predicted embedding distributions
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize = (10, 6))
            ax[0].hist(np.array(embeddings[:, :inputs.shape[1]//2].ravel()), bins=100, alpha=0.5, label='past embedding')
            ax[0].hist(np.array(embeddings[:, inputs.shape[1]//2:].ravel()), bins=100, alpha=0.5, label='future embedding')
            ax[0].hist(np.array(embeddings_hat.ravel()), bins=100, alpha=0.5, label='predicted embedding')
            ax[0].legend()

            # inputs-forecast distributions
            try:
                ax[1].hist(np.array(y_np.ravel()), bins=100, alpha=0.5, label='forecast ')
            except:
                print('Error in plotting forecast distribution')
            ax[1].hist(np.array(inputs.values.ravel()), bins=100, alpha=0.9, label='inputs')
            ax[1].legend()
            if self.input_scaler is not None:
                y_np = self.input_scaler.inverse_transform(y_np)
            y_hat = pd.DataFrame(y_np[:, inputs.shape[1]//2:], index=inputs.index, columns=self.target_columns[:inputs.shape[1]//2])
        else:
            y_hat = self.predict_batch(self.pars, x)
            y_hat = np.array(y_hat)
            if self.normalize_target:
                y_hat = self.target_scaler.inverse_transform(y_hat)

            y_hat = pd.DataFrame(y_hat, index=inputs.index, columns=self.target_columns)

        return y_hat


def invert(params, model, y):
    def inverter(lpicnn):
        return lpicnn.invert(y)

    return nn.apply(inverter, model)(params)

def embed(params, model, y):
    def embedder(lpicnn):
        return lpicnn.embed(y)

    return nn.apply(embedder, model)(params)