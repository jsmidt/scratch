from collections import OrderedDict, namedtuple
import torch
import random
r"""A personal rewrite of the pytorch.nn model for learning purposes."""


class Module:
    r"""The base Module class for modules in nn."""

    def __init__(self) -> None:
        super().__setattr__('training', True)
        super().__setattr__('_parameters', OrderedDict())

    def forward(self):
        self()


class Linear(Module):
    r"""Linear nn layer with learnable weights w and bias b. 
        The return on input data x is y = x @ w.T + b.

    Paramaters:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    """

    def __init__(self, in_features, out_features, bias=True, device=None,
                 dtype=torch.float32, nonlinearity='relu') -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bbias = bias

        if nonlinearity == 'tanh':
            a = 5/3.
        else:
            a = 2.0**0.5

        self.weight = torch.randn(
            (in_features, out_features))*a/(in_features)**0.5
        self.bias = torch.randn(out_features) * 0.1 if bias else None

    def __call__(self, x):
        r"""The 'forward pass' returning y = x @ w.T + b."""
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        r"""The parameters to train."""
        params = [self.weight] + ([] if self.bias is None else [self.bias])
        return params

    def __repr__(self):
        params = self.parameters()
        num_params = sum(p.nelement() for p in params)
        mystr = f"Linear(in_features={self.in_features}, "
        mystr += f"out_features={self.out_features}, bias={self.bbias}), "
        mystr += f"Total parameters: {num_params}"
        return mystr


class Embedding(Module):
    r"""An embedding layer with learnable weights.
    The return on input data x is y = w[x]

    Paramaters:
        num_embeddings: Number of features to embed.
        embedding_dim: The dimension size to embed those features.
    """

    def __init__(self, num_embeddings, embedding_dim) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        params = [self.weight]
        return params

    def __repr__(self):
        params = self.parameters()
        num_params = sum(p.nelement() for p in params)
        mystr = f"Embedding(num_embeddings={self.num_embeddings}, "
        mystr += f"embedding_dim={self.embedding_dim}), "
        mystr += f"Total parameters: {num_params}"
        return mystr


class Tanh(Module):
    r"""Applies the Hyperbolic Tangent (Tanh) function element-wise.
    """

    def __call__(self, x):
        return torch.tanh(x)

    def parameters(self):
        params = []
        return params

    def __repr__(self):
        params = self.parameters()
        num_params = sum(p.nelement() for p in params)
        mystr = f"Tanh(), "
        mystr += f"Total parameters: {num_params}"
        return mystr


class Relu(Module):
    r"""Applies the rectified linear unit function element-wise:
    """

    def __call__(self, x):
        return x * (x > 0).float()

    def parameters(self):
        params = []
        return params

    def __repr__(self):
        params = self.parameters()
        num_params = sum(p.nelement() for p in params)
        mystr = f"Relu(), "
        mystr += f"Total parameters: {num_params}"
        return mystr


class Flatten:
    '''Flattens input by reshaping it into a one-dimensional tensor. '''

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, x):
        # self.out = x.view(x.shape[0], -1)
        # return x.view(-1, x.numel())
        return x.flatten(self.start_dim, self.end_dim)

    def parameters(self):
        params = []
        return params

    def __repr__(self):
        params = self.parameters()
        num_params = sum(p.nelement() for p in params)
        mystr = f"Flatten(), "
        mystr += f"Total parameters: {num_params}"
        return mystr


class BatchNorm1d:
    '''Applies Batch Normalization over 1D input'''

    def __init__(self, num_features, eps=1e-05, momentum=0.1):

        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.num_features = num_features
        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True, unbiased=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean)/torch.sqrt(xvar + self.eps)
        self.out = self.gamma*xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * \
                    self.running_mean + self.momentum*xmean
                self.running_var = (1 - self.momentum) * \
                    self.running_var + self.momentum*xvar
        return self.out

    def parameters(self):
        params = [self.gamma, self.beta]
        return params

    def __repr__(self):
        params = self.parameters()
        num_params = sum(p.nelement() for p in params)
        mystr = f"Batchnorm1D(num_features={self.num_features}), "
        mystr += f"Total parameters: {num_params}"
        return mystr


class Sequential(Module):
    '''An nn model from seuqntial layers. '''

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        params = [p for layer in self.layers for p in layer.parameters()]
        return params

    def eval(self):
        for layer in self.layers:
            layer.training = False
    
    def train(self):
        for layer in self.layers:
            layer.training = True

    def __repr__(self):
        params = self.parameters()
        num_params = sum(p.nelement() for p in params)

        # Name of model
        mystr = f"{self.__class__.__name__}("
        # List each layer
        for i, layer in enumerate(self.layers):
            mystr += f"\n  ({i}): {layer}, "
        mystr += f"\n)"
        # Print total parameters of model
        mystr += f"\nTotal parameters: {num_params}"
        return mystr



