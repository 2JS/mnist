import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import pytorch_lightning as pl

from model import AutoEncoder

train_dataset = MNIST(root='data/', train=True, download=True, transform=ToTensor()
)
test_dataset = MNIST(root='data/', train=False, download=True, transform=ToTensor()
)

train_dataloader = DataLoader(train_dataset)

autoencoder = AutoEncoder()

trainer = pl.Trainer()
trainer.fit(autoencoder, train_dataloader)