from collections import OrderedDict, namedtuple
import jax
import jax.numpy as jnp
from dataclasses import dataclass
r"""A personal rewrite of the pytorch.nn model for learning purposes."""


@dataclass()
class Linear:
    r"""Linear nn layer with learnable weights w and bias b. 
        The return on input data x is y = x @ w.T + b.

    Paramaters:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    """
    in_features: int
    out_features: int
    bias: bool = True
    device: str = None
    nonlinearity: str = 'relu'
    dtype: any = jnp.float32

    def init(self, key):

        if self.nonlinearity == 'tanh':
            a = 5/3.
        else:
            a = 2.0**0.5

        params = {}
        key, w_key, b_key = jax.random.split(key, num=3)
        w = jax.random.normal(w_key, (self.in_features, self.out_features),
                              dtype=self.dtype)*a/(self.in_features)**0.5
        params['weights'] = w

        if self.bias is True:
            b = jax.random.normal(b_key, (self.out_features,),
                                  dtype=self.dtype)
            params['bias'] = b

        self.num_params = sum(params[p].size for p in params)

        return key, params

    def __call__(self, params, x):
        r"""The 'forward pass' returning y = x @ w.T + b."""
        self.out = x @ params['weights']
        if params['bias'] is not None:
            self.out += params['bias']
        return self.out

    def __repr__(self):
        mystr = f"Linear(in_features={self.in_features}, "
        mystr += f"out_features={self.out_features}, bias={self.bias}), "
        mystr += f"Total parameters: {self.num_params}"
        return mystr


@dataclass
class Embedding:
    r"""An embedding layer with learnable weights.
    The return on input data x is y = w[x]

    Paramaters:
        num_embeddings: Number of features to embed.
        embedding_dim: The dimension size to embed those features.
    """
    num_embeddings: int
    embedding_dim: int
    device: str = None
    nonlinearity: str = 'relu'
    dtype: any = jnp.float32

    def init(self, key):

        params = {}
        key, e_key = jax.random.split(key)
        w = jax.random.normal(e_key, 
            (self.num_embeddings, self.embedding_dim), dtype=self.dtype)
        params['weights'] = w

        self.num_params = sum(params[p].size for p in params)

        return key, params

    def __call__(self, params, x):
        out = params['weights'][x]
        return out

    def __repr__(self):
        mystr = f"Embedding(num_embeddings={self.num_embeddings}, "
        mystr += f"embedding_dim={self.embedding_dim}), "
        mystr += f"Total parameters: {self.num_params}"
        return mystr
