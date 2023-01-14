import torch
import torch.nn.functional as F

def softmax(logits, dim=1):
    # Allowed to use softmax because it's in Jax
    return F.softmax(logits, dim)

def cross_entropy(logits, targets,dim=1):
    # Allowed to use softmax because it's in Jax, gives log(Probabilities)
    lprobs = F.log_softmax(logits, dim)

    # Cross entropy picks out the - log probability of the target and averages
    loss = -lprobs[torch.arange(logits.shape[0]), targets].mean()

    return loss
