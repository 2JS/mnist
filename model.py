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

class ConvClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.Dropout(p=0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )
        self.train_acc = metrics.Accuracy()
        self.valid_acc = metrics.Accuracy(compute_on_step=False)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train loss', loss)
        self.log('train accuracy', self.train_acc(logits, y))
        self.train_acc.reset()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('valid loss', loss)
        self.valid_acc(logits, y)
        return loss

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


class ConvAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 32, 14, 14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 64, 7, 7
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, padding=1),
            # 128, 4, 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128*4*4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            # 128, 4, 4
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 64, 7, 7
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 32, 14, 14
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 16, 28, 28
            nn.Conv2d(16, 1, kernel_size=3, padding=1, padding_mode='replicate'),
            # 1, 28, 28
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_hat = F.leaky_relu_(1-F.leaky_relu_(1-self.decoder(z), 0.1), 0.1)
        loss = F.mse_loss(x_hat, x)
        self.log('train loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_hat = F.leaky_relu_(1-F.leaky_relu_(1-self.decoder(z), 0.1), 0.1)
        loss = F.mse_loss(x_hat, x)
        self.log('valid loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
