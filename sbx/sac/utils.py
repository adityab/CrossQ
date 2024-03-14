import os

import jax
import jax.numpy as jnp
import flax.linen as nn

def is_slurm_job():
    """Checks whether the script is run within slurm"""
    return bool(len({k: v for k, v in os.environ.items() if 'SLURM' in k}))


class ReLU(nn.Module):
    def __call__(self, x):
        return nn.relu(x)

class ReLU6(nn.Module):
    def __call__(self, x):
        return nn.relu6(x)

class Tanh(nn.Module):
    def __call__(self, x):
        return nn.tanh(x)

class Sin(nn.Module):
    def __call__(self, x):
        return jnp.sin(x)

class Elu(nn.Module):
    def __call__(self, x):
        return nn.elu(x)

class GLU(nn.Module):
    def __call__(self, x):
        return nn.glu(x)
    
class LayerNormedReLU(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.LayerNorm()(nn.relu(x))

class ReLUOverMax(nn.Module):
    def __call__(self, x):
        act = nn.relu(x)
        return act / (jnp.max(act) + 1e-6)

activation_fn = {
    # unbounded
    "relu": ReLU,
    "elu": Elu,
    "glu": GLU,
    # bounded
    "tanh": Tanh,
    "sin": Sin,
    "relu6": ReLU6,
    # unbounded with normalizer
    "layernormed_relu": LayerNormedReLU, # with normalizer
    "relu_over_max": ReLUOverMax, # with normalizer
}