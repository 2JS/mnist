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

    def prepare_data(self):
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage = None):
        self.train_dataset = CIFAR10(root='data/', train=True, download=True, transform=T.ToTensor()
        )
        self.valid_dataset = CIFAR10(root='data/', train=False, download=True, transform=T.ToTensor()
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=2**4, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=2**7)
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        if not self.trainer.training:
            return batch
        
        x = batch[0]
        x = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)(x)
        batch[0] = x
        return batch
    