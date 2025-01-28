import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import functools

from . import utils, layers, layerspp, normalization
default_init = layers.default_init
get_act = layers.get_act


@utils.register_model(name='tsim')

class SSimpleDeep(nn.Module):
    """The most basic architecture imaginable: takes 2D input and continuous time"""
    
    config: ml_collections.ConfigDict
    
    @nn.compact
    def __call__(self, x, t, train=True):
        config = self.config
        nf = config.model.nf
        act = get_act(config)
        x = jnp.squeeze(x)
        out_shape = x.shape[-1]

        assert config.training.continuous, "Fourier features are only used for continuous training."
        used_sigmas = t
        temb = layerspp.GaussianFourierProjection(
                embedding_size=nf,
                scale=config.model.fourier_scale)(jnp.log(used_sigmas))
        
        temb = act(nn.Dense(2 * nf, kernel_init=default_init())(temb))

        xemb = act(nn.Dense(2 * nf, kernel_init=default_init())(x))
        xemb = nn.Dense(2 * nf, kernel_init=default_init())(xemb)
        mid = temb + xemb
        
        for _ in range(self.config.model.num_res_blocks):
            mid = act(nn.Dense(2 * nf, kernel_init=default_init())(mid)) + temb
        
        mid = act(nn.Dense(2 * nf, kernel_init=default_init())(mid))
        mid = act(nn.Dense(2 * nf, kernel_init=default_init())(mid))

        out = nn.Dense(out_shape, kernel_init=default_init())(mid)[:, None, None, :]

        out = out / used_sigmas.reshape(-1, 1, 1, 1)
        return out