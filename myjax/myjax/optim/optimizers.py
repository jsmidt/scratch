import jax
import jax.numpy as np


class Optimzer:
    def __init__(self, learning_rate=0.001, momentum=0, weight_decay=0) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

    def step(self):
        NotImplementedError


class SGD(Optimzer):
    def __init__(self, learning_rate=0.001, momentum=0.0, weight_decay=0.0) -> None:
        super().__init__(learning_rate, momentum, weight_decay)

    def step(self, params, grads):

        # SGD step with weight decay
        params = jax.tree_map(
            lambda p, g: p - self.learning_rate * (g + self.weight_decay*p), params, grads)

        return params
