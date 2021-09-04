import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self,
        data_dir='data/',
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = torch.jit.script(nn.Sequential(
            # T.ColorJitter(brightness=0.4, contrast=0.6, saturation=0.6, hue=0.05),
            # T.RandomGrayscale(p=0.1),
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomVerticalFlip(p=0.5),
            # T.RandomResizedCrop(size=(32,32), scale=(0.7,1.0)),
            # T.RandomAffine(degrees=(-45, 45), translate=(0.1, 0.1), scale=(0.9, 1.2)),
            T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        ))

    def setup(self, stage = None):
        self.train_dataset = CIFAR10(root=self.data_dir, train=True, download=True, transform=T.ToTensor()
        )
        self.valid_dataset = CIFAR10(root=self.data_dir, train=False, download=True, transform=T.ToTensor()
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=2**7, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=2**7)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if not self.trainer.training:
            return batch

        x, y = batch
        x = (x * 255).to(torch.uint8)
        x = self.train_transform(x)
        x = x.to(torch.float32) / 255
        return x, y
    