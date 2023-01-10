from collections import OrderedDict, namedtuple
import torch
r"""A personal rewrite of the pytorch.nn model for learning purposes."""


class Module:
    r"""The base Module class for modules in nn."""

    def __init__(self) -> None:
        super().__setattr__('training', True)
        super().__setattr__('_parameters', OrderedDict())


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
        self.bias = torch.randn(out_features) * 0.1

    def __call__(self, x):
        r"""The 'forward pass' returning y = x @ w.T + b."""
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        r"""The parameters to train."""
        params =  [self.weight] + ([] if self.bias is None else [self.bias])
        num_params = sum(p.nelement() for p in params)
        return params, num_params

    def __repr__(self):
        _, num_params = self.parameters()
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
        num_params = sum(p.nelement() for p in params)
        return params, num_params

    def __repr__(self):
        _, num_params = self.parameters()
        mystr = f"Embedding(num_embeddings={self.num_embeddings}, "
        mystr += f"embedding_dim={self.embedding_dim}), "
        mystr += f"Total parameters: {num_params}"
        return mystr
