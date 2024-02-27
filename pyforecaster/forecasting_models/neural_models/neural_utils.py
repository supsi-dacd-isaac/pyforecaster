from functools import partial

from jax import jit
from jax import numpy as jnp
from jax import random


def jitting_wrapper(fun, model, **kwargs):
    return jit(partial(fun, model=model, **kwargs))


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

def reproject_weights(params, rec_stable=False, monotone=False):
    # Loop through each layer and reproject the input-convex weights
    for layer_name in params['params']:
        if 'PICNNLayer' in layer_name:
            if 'wz' in params['params'][layer_name].keys():
                _reproject(params['params'], layer_name, rec_stable=rec_stable, monotone=monotone or ('monotone' in layer_name))
            else:
                for name in params['params'][layer_name].keys():
                    _reproject(params['params'][layer_name], name, rec_stable=rec_stable, monotone=monotone or ('monotone' in layer_name))

    return params

def _reproject(params, layer_name, rec_stable=False, monotone=False):
    if ('monotone' in layer_name) or monotone:
        for name in {'wz', 'wy'} & set(params[layer_name].keys()):
            params[layer_name][name]['kernel'] = jnp.maximum(0, params[layer_name][name]['kernel'])
    else:
        params[layer_name]['wz']['kernel'] = jnp.maximum(0, params[layer_name]['wz']['kernel'])
        if rec_stable:
            params[layer_name]['wy']['kernel'] = jnp.maximum(0, params[layer_name]['wy']['kernel'])
