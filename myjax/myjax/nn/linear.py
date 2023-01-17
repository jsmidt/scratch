import jax
import jax.numpy as jnp

class Linear:
    r"""Linear nn layer with learnable weights w and bias b. 
        The return on input data x is y = x @ w.T + b.

    Paramaters:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    """

    def __init__(self, in_features, out_features, bias=True, nonlinearity=None, device=None, dtype=jnp.float32) -> None:
        # super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.dtype = dtype
        self.nonlinearity = nonlinearity

    def init(self, key):
        '''Initialize with a key'''

        # Scale weights for nonlinearity
        if self.nonlinearity == 'tanh':
            a = 5/3.
        elif self.nonlinearity is None:
            a = 1.0
        else:
            a = 2.0**0.5

        # Set weights and bias as parameters
        params = {}
        key, w_key, b_key = jax.random.split(key, num=3)
        params['weights'] = jax.random.normal(key=w_key,
                                              shape=(self.in_features,
                                                     self.out_features)
                                              )*a/(self.in_features)**0.5
        # Only use bias if desired
        if self.bias == True:
            params['bias'] = jax.random.normal(key=b_key,
                                               shape=(self.out_features,)) * 0.1

        # Record number of parameters
        self.num_params = sum(params[p].size for p in params)

        # Return parameters and new key
        return key, params

    def __call__(self, params: dict, x: jnp.array) -> jnp.array:
        '''The return on input data x is y = x @ w.T + b.'''
        out = x @ params['weights']
        if self.bias == True:
            out += params['bias']
        return out

    def __repr__(self) -> str:
        mystr = f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias}), Total parameters: {self.num_params}"
        return mystr

class Relu:
    r"""Applies the rectified linear unit function element-wise:
    """

    def __call__(self, params, x):
        return jax.nn.relu(x)

    def init(self, key):
        params = jnp.array([])
        self.num_params = sum(params[p].size for p in params)
        return key, params

    def __repr__(self):
        mystr = f"Relu(), Total parameters: {self.num_params}"
        return mystr

