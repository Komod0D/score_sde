from flax import linen as nn
from jax import numpy as jnp
import ml_collections
from models import utils

from . import layers

get_act = layers.get_act


from flax import linen as nn
from jax import numpy as jnp
import ml_collections
from models import utils

from . import layers

get_act = layers.get_act

@utils.register_model(name='cls_test')
class TestClassifier(nn.Module):

    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, t, train=False):
        config = self.config
        nf = config.model.nf
        act = get_act(config)
        
        x_sq = jnp.squeeze(x)

        h1 = act(nn.Dense(nf)(x_sq))
        h2 = act(nn.Dense(nf)(h1))
        h3 = act(nn.Dense(nf)(h2))
        h4 = act(nn.Dense(nf)(h3))
        h5 = act(nn.Dense(nf)(h4))
        h6 = act(nn.Dense(nf)(h5))


        out = nn.Dense(1)(h3)

        return out


@utils.register_model(name='ssim_cls')
class TestClassifier(nn.Module):

    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, t, train=False):
        config = self.config
        nf = config.model.nf
        act = get_act(config)
        
        x_sq = jnp.squeeze(x).reshape(-1, x.shape[-1])

        temb = [t - 0.5, jnp.cos(2 * jnp.pi * t), jnp.sin(2 * jnp.pi * t), -jnp.cos(4 * jnp.pi * t)]
        temb = jnp.stack(temb, axis=1)
        xemb = jnp.concatenate([x_sq, temb], axis=1)

        h1 = act(nn.Dense(nf)(xemb))
        h2 = act(nn.Dense(nf)(h1))
        h3 = act(nn.Dense(nf)(h2))
        h4 = act(nn.Dense(nf)(h3))
        h5 = act(nn.Dense(nf)(h4))
        h6 = act(nn.Dense(nf)(h5))


        out = nn.Dense(1)(h6)

        return out