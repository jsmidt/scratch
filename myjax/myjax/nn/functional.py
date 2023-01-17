import jax
import jax.numpy

def mse_loss(input, target):
    return jnp.mean((input-target)**2)
