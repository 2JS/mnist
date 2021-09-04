import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as metrics
import pytorch_lightning as pl

from layers import *


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 10),
        )
        self.train_acc = metrics.Accuracy(compute_on_step=False)
        self.valid_acc = metrics.Accuracy(compute_on_step=False)

    def forward(self, x):
        logits = self.layers(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
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
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(8192, 512),
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

class MLPMixerClassifier(pl.LightningModule):
    def __init__(
        self,
        seq_len=100,
        in_channels=5,
        num_features=32,
        expansion_factor=2,
        num_layers=3,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, num_features, kernel_size=7, padding=3)
        self.mixers = nn.Sequential(
            *[
                MixerLayer(
                    seq_len=seq_len,
                    num_features=num_features,
                    d_c=expansion_factor*seq_len,
                    d_s=expansion_factor*num_features,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(seq_len*num_features, num_classes)

        self.train_acc = metrics.Accuracy()
        self.valid_acc = metrics.Accuracy(compute_on_step=False)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2, 3)
        x = x.permute(0, 2, 1)
        x = self.mixers(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

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


class ResNet(pl.LightningModule):
    def __init__(self, block, num_block, num_classes=10, init_weights=True):
        super().__init__()

        self.in_channels=64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weights inittialization
        if init_weights:
            self._initialize_weights()

        self.train_acc = metrics.Accuracy()
        self.valid_acc = metrics.Accuracy(compute_on_step=False)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer


    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(BottleNeck, [3,4,6,3])

def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])