from functools import partial

import jax.numpy as jnp
from flax import linen as nn

from pyforecaster.forecasting_models.neural_models.neural_utils import positive_lecun, identity


class PICNNLayer(nn.Module):
    features_x: int
    features_y: int
    features_out: int
    features_latent: int
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

        y_add_kernel_init = nn.initializers.lecun_normal() if self.rec_activation == identity else partial(positive_lecun, init_type=self.init_type)
        # Input-Convex component without bias for the element-wise multiplicative interactions
        wzu = nn.relu(nn.Dense(features=self.features_latent, use_bias=True, name='wzu')(u))
        wyu = self.rec_activation(nn.Dense(features=self.features_y, use_bias=True, name='wyu')(u))
        z_add = nn.Dense(features=self.features_out, use_bias=False, name='wz', kernel_init=partial(positive_lecun, init_type=self.init_type))(z * wzu)
        y_add = nn.Dense(features=self.features_out, use_bias=False, name='wy', kernel_init=y_add_kernel_init)(y * wyu)
        u_add = nn.Dense(features=self.features_out, use_bias=True, name='wuz')(u)


        if self.layer_normalization:
            y_add = nn.LayerNorm()(y_add)
            z_add = nn.LayerNorm()(z_add)
            u_add = nn.LayerNorm()(u_add)

        z_next = z_add + y_add + u_add
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

class CausalInvertibleLayer(nn.Module):
    features: int
    negative_slope:int = 0.01
    activation: callable = nn.leaky_relu(negative_slope=negative_slope)
    init_type: str = 'normal'
    layer_normalization: bool = False
    prediction_layer: bool = False

    def setup(self):
        pass
    def __call__(self, inputs):
        inner_pars = self.param(
            'kernel',
            self.kernel_init,
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )

        kernel = jnp.tril(inner_pars) + jnp.eye(self.features)

        if self.dot_general_cls is not None:
            dot_general = self.dot_general_cls()
        else:
            dot_general = self.dot_general
        y = dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        if self.prediction_layer:
            return y
        else:
            if self.layer_normalization:
                y = nn.LayerNorm()(y)
            return self.activation(y)

    def invert(self, y):
        if not self.prediction_layer:
            y = self.inverse_leaky_relu(y)
        return jnp.dot(y, jnp.linalg.inv(self.kernel))


    def inverse_leaky_relu(self, y):
        return jnp.where(y >= 0, y, y / self.negative_slope)
