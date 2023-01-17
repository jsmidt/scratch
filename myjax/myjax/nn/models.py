import jax
import jax.numpy as jnp

class Model:
    '''Base class for model types'''

    def __init__(self) -> None:
        pass

    # Every model needs an init
    def init(self, key) -> None:
        NotImplementedError

    # Models use the "forward" method for call
    def __call__(self, params, x) -> jnp.array:
        return self.forward(params, x)

    # Every model needs a forward routine. See call.
    def forward(self, params, x) -> jnp.array:
        NotImplementedError


class Sequential(Model):
    '''Sequential model'''

    def __init__(self, layers) -> None:
        self.layers = layers

    # Init model with a key
    def init(self, key):
        '''Initialize model with a key'''

        # Parameters for each later
        params = {}
        self.num_params = 0
        for i, layer in enumerate(self.layers):
            # Initialize each layer with a key
            key, l_key = jax.random.split(key)
            key, paramsl = layer.init(l_key)
            params[i] = paramsl
            #if paramsl is not None:
            #    self.num_params += layer.num_params

        # Record total number of parameters
        self.num_params = sum([l.num_params for l in self.layers])

        return key, params

    # Forward routine when called
    def forward(self, params, x) -> jnp.array:
        '''Evaluate model for batch x'''
        for i, layer in enumerate(self.layers):
            x = layer(params[i], x)
        return x

    # Print model info
    def __repr__(self):
        '''Model info as a string'''

        # Name of model
        mystr = f"{self.__class__.__name__}("

        # List each layer
        for i, layer in enumerate(self.layers):
            mystr += f"\n  ({i}): {layer}, "
        mystr += f"\n)"

        # Print total parameters of model
        mystr += f"\nTotal parameters: {self.num_params}"
        return mystr

