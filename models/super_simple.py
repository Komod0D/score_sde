import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import functools

from . import utils, layers, layerspp, normalization
default_init = layers.default_init
get_act = layers.get_act


@utils.register_model(name='ssim')

class SSimple(nn.Module):
    """The most basic architecture imaginable: takes 2D input and continuous time"""
    
    config: ml_collections.ConfigDict
    
    @nn.compact
    def __call__(self, x, t, train=True):
        config = self.config
        nf = config.model.nf
        act = get_act(config)
        out_shape = x.shape[-1]
        used_sigmas = t
        
        x_reshape = jnp.squeeze(x).reshape(-1, out_shape)

        temb = [t - 0.5, jnp.cos(2 * jnp.pi * t), jnp.sin(2 * jnp.pi * t), -jnp.cos(4 * jnp.pi * t)]
        temb = jnp.stack(temb, axis=1)
        
        xemb = jnp.concatenate([x_reshape, temb], axis=1)
        h1 = act(nn.Dense(nf)(xemb))
        h2 = act(nn.Dense(nf)(h1))
        h3 = act(nn.Dense(nf)(h2))
        out = nn.Dense(out_shape)(h3)[:, None, None, :]

        out = out / used_sigmas.reshape(-1, 1, 1, 1)
        return out