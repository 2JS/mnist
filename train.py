import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model import AutoEncoder

train_dataset = MNIST(root='data/', train=True, download=True, transform=ToTensor()
)
test_dataset = MNIST(root='data/', train=False, download=True, transform=ToTensor()
)

train_dataloader = DataLoader(train_dataset)
test_dataloader = DataLoader(test_dataset)

autoencoder = AutoEncoder()

logger = WandbLogger(project='mnist', log_model='all')

trainer = pl.Trainer(
    max_epochs=10,
    gpus=-1,
    logger=logger,
)
logger.watch(autoencoder)

trainer.fit(autoencoder, train_dataloader, test_dataloader)
