import torch

class Optimizer:
    def __init__(self, params, lr=0.01):
        """Minibatch stochastic gradient descent."""
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param.grad = None


class SGD(Optimizer):

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad
