from functools import partial

import jax.numpy as jnp
from flax import linen as nn
from jax import jit
import jax

from pyforecaster.forecasting_models.neural_models.neural_utils import positive_lecun, identity
from jax import lax

from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union, Dict,
)
Dtype = Any  # this could be a real type?
Array = Any
PRNGKey = Any
Shape = Tuple[int, ...]
DotGeneralT = Callable[..., Array]
PrecisionLike = Union[
    None,
    str,
    lax.Precision,
    Tuple[str, str],
    Tuple[lax.Precision, lax.Precision],
]

from flax.linen import initializers
default_kernel_init = initializers.lecun_normal()



@jit
def invcondiff(x: Array, a=1, c=1) -> Array:
    r"""Invertible, continuous and continuously differentiable activation function with full co-domain

        x<-a: -b ln(-x) - k
        -a<x<a: c x
        x>a:  b ln(x) + k

        with
            b = a c
            k =  a c (1 - ln(a))

    Args:
    x : input array
    """
    b = a * c
    k = a * c * (1 - jnp.log(a))
    left = -b * jnp.log(jnp.clip(-x, 1e-6)) - k
    middle = c * x
    right = b * jnp.log(jnp.clip(x, 1e-6)) + k
    return (x < -a) * left +  (x > a)*right +   middle*(jnp.logical_and(-a <= x, x <= a))


@jit
def invcondiff_inverse(y: jnp.ndarray, a=1, c=1) -> jnp.ndarray:
    b = a * c
    k = a * c * (1 - jnp.log(a))

    # Inverse for x < -a
    left_inv = -jnp.exp(-(y + k) / b)

    # Inverse for -a <= x <= a
    middle_inv = y / c

    # Inverse for x > a
    right_inv = jnp.exp((y - k) / b)

    # Determine which inverse to use based on the value of y
    # Note: We need to determine the appropriate ranges for y that correspond to the original x ranges
    y_left_range = y < -b * jnp.log(a) - k  # Corresponds to x < -a
    y_right_range = y > b * jnp.log(a) + k  # Corresponds to x > a
    return y_right_range * right_inv + y_left_range * left_inv + middle_inv * (y_right_range + y_left_range == 0)



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
    scaling_factor: float = 0.01
    #activation: callable = partial(nn.leaky_relu,negative_slope=negative_slope)
    activation: callable = invcondiff
    init_type: str = 'normal'
    layer_normalization: bool = False
    prediction_layer: bool = False
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = lax.Precision.HIGHEST
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    kernel = None
    bias = None
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
        initializers.zeros_init()
    )
    dot_general: DotGeneralT = lax.dot_general
    dot_general_cls: Any = None
    def setup(self):

        inner_pars = self.param(
            'kernel',
            self.kernel_init,
            ((self.features*(self.features+1))//2-self.features,1 ),
            self.param_dtype,
        )

        self.kernel = vector_to_lower_triangular(inner_pars, self.features)
        self.bias = self.param(
              'bias', self.bias_init, (self.features,), self.param_dtype
        )

    def __call__(self, inputs):

        #kernel = jnp.tanh(self.kernel) * self.scaling_factor + jnp.eye(self.features)
        kernel = jnp.tanh(self.kernel) * self.scaling_factor/self.features + jnp.eye(self.features)
        y = kernel @ inputs.T
        y += jnp.reshape(self.bias, (1,) * (y.ndim - 1) + (-1,))
        if self.prediction_layer:
            return y
        else:
            if self.layer_normalization:
                y = nn.LayerNorm()(y)
            return self.activation(y)

    def invert(self, y):
        if not self.prediction_layer:
            y = self.invert_activation(y)
        y -= jnp.reshape(self.bias, (1,) * (y.ndim - 1) + (-1,))
        kernel = jnp.tanh(self.kernel) * self.scaling_factor/self.features + jnp.eye(self.features)
        #return (jnp.linalg.inv(kernel) @ y.T).T
        f = jax.scipy.linalg.solve_triangular(kernel, y.T, lower=True).T
        #f = jnp.linalg.solve(kernel, y.T).T
        return f
    def predict(self, inputs):
        kernel = jnp.tanh(self.kernel)  * self.scaling_factor /self.features + jnp.eye(self.features)
        y = (kernel @ inputs.T).T
        y += jnp.reshape(self.bias, (1,) * (y.ndim - 1) + (-1,))
        if self.prediction_layer:
            return y
        else:
            if self.layer_normalization:
                y = nn.LayerNorm()(y)
            return self.activation(y)
    def invert_activation(self, y):
        name = self.activation.__name__ if '__name__' in dir(self.activation) else self.activation.func.__name__
        if name == "leaky_relu":
            return jnp.where(y >= 0, y, y / self.negative_slope)
        elif name == "softplus":
            return inverse_softplus(jnp.maximum(1e-6, y))
        elif name == "invcondiff":
            return invcondiff_inverse(y)
        else:
            raise NotImplementedError('Activation function {} not implemented for inversion'.format(name))


def inverse_softplus(y, threshold=20):
    # For large y, softplus(x) is approximately x, so we can return y directly.
    # This avoids numerical overflow for the exponential operation.
    is_large = y > threshold
    # For values of y that are not too large, compute the inverse softplus normally.
    x_small = jnp.log(jnp.exp(y) - 1)
    # Use the approximation for large values of y.
    x_large = y
    # Select the appropriate values based on the condition.
    x = jnp.where(is_large, x_large, x_small)
    return x


def vector_to_lower_triangular(vector, L):
    # Initialize an LxL matrix of zeros
    matrix = jnp.zeros((L, L))

    # Get the indices for the lower triangular part
    tril_indices = jnp.tril_indices(L, -1)

    # Fill the lower triangular part of the matrix with the vector elements
    matrix = matrix.at[tril_indices].set(vector.ravel())

    return matrix