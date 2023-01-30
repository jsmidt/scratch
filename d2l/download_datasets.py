import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST, CIFAR10



#################################
#
#  MNIST
#
#################################

dataset = MNIST(root="data", download=True, transform=ToTensor())


dataset = CIFAR10(root="data", download=True, transform=ToTensor())


#################################
#
#  Fashion MNIST
#
#################################

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
