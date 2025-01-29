import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import functools

from . import utils, layers, layerspp, normalization
default_init = layers.default_init
get_act = layers.get_act


@utils.register_model(name='ssim_conv')

class SSimpleConv(nn.Module):
    """Simple Convolutional Network"""
    
    config: ml_collections.ConfigDict
    
    @nn.compact
    def __call__(self, x, t, train=True):
        config = self.config
        nf = config.model.nf
        strides = config.model.strides
        kernel = config.model.kernel
        act = get_act(config)
        out_shape = x.shape[-1]

        assert config.training.continuous, "Fourier features are only used for continuous training."
        used_sigmas = t
        
        temb = layerspp.GaussianFourierProjection(
                embedding_size=nf,
                scale=config.model.fourier_scale)(jnp.log(used_sigmas))
        
        temb = act(nn.Dense(2 * nf, kernel_init=default_init())(temb))[:, None, None, :]

        conv1 = act(nn.Conv(nf, kernel_size=kernel, strides=strides, padding='SAME',
                            kernel_init=default_init()))(x)
        conv2 = act(nn.Conv(2 * nf, kernel_size=kernel, strides=strides, padding='SAME',
                            kernel_init=default_init()))(conv1)

        h = temb + conv2
        mid = act(nn.Dense(2 * nf, kernel_init=default_init())(h))
        
        up2 = act(nn.ConvTranspose(2 * nf, kernel_size=kernel, kernel_init=default_init(),
                                   strides=strides)(mid))
        up1 = act(nn.ConvTranspose(nf, kernel_size=kernel, kernel_init=default_init(), strides=strides)(up2))

        
        final = act(nn.Dense(2 * nf, kernel_init=default_init())(up1))
        out = nn.Dense(out_shape, kernel_init=default_init())(final)

        out = out / used_sigmas.reshape(-1, 1, 1, 1)
        return out