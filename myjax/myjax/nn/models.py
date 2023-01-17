import jax
import jax.numpy as jnp


class Model:
    '''Base class for model types'''
    # Every model needs an init

    def __init__(self) -> None:
        self.model = None

    # Every model needs an initialization
    def init(self, key):
        key, m_key = jax.random.split(key)
        key, params = self.model.init(key)
        return key, params

    # Every model needs an training step
    def training_step(self):
        NotImplementedError

    # Some models needs an validation step
    def validation_step(self):
        NotImplementedError

    # Some models needs an predict step
    def predict_step(self):
        NotImplementedError

    # Every model need to configure optimzers
    def configure_optimizers(self):
        NotImplementedError

    def __repr__(self) -> str:
        return self.model.__repr__()
