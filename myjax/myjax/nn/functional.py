import jax
import jax.numpy as jnp


# Take functions preferably from jax.nn package
# This file if for those not in there, like MSE

def mse_loss(input, target):
    return jnp.mean((input-target)**2)
