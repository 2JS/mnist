import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as metrics
import pytorch_lightning as pl


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 10), nn.Softmax()
        )
        self.train_acc = metrics.Accuracy(compute_on_step=False)
        self.valid_acc = metrics.Accuracy(compute_on_step=False)

    def forward(self, x):
        logits = self.layers(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train loss', loss)
        self.train_acc(logits, y)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train accuracy', self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('valid loss', loss)
        self.valid_acc(logits, y)

    def validation_epoch_end(self, outputs):
        self.log('valid accuracy', self.valid_acc.compute())
        self.valid_acc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-4)


class AutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
