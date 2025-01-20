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
        out_shape = 2

        assert config.training.continuous, "Fourier features are only used for continuous training."
        used_sigmas = t
        temb = layerspp.GaussianFourierProjection(
                embedding_size=nf,
                scale=config.model.fourier_scale)(jnp.log(used_sigmas))
        
        temb = act(nn.Dense(2 * nf, kernel_init=default_init())(temb))[:, None, None, :]

        xemb = act(nn.Dense(2 * nf, kernel_init=default_init())(x))
        xemb = nn.Dense(2 * nf, kernel_init=default_init())(xemb)
        h = temb + xemb

        
        mid = act(nn.Dense(2 * nf, kernel_init=default_init())(h)) + temb
        mid2 = act(nn.Dense(2 * nf, kernel_init=default_init())(mid)) + temb

        out = nn.Dense(out_shape, kernel_init=default_init())(mid2)


        out = out / used_sigmas.reshape(-1, 1, 1, 1)
        return out