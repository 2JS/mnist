import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model import *
from datamodule import CIFAR10DataModule

dm = CIFAR10DataModule()

model = MLPMixerClassifier(seq_len=32*32, in_channels=3, num_features=128, num_layers=5, num_classes=10)

logger = WandbLogger(project='cifar10', log_model='all')

trainer = pl.Trainer(
    max_epochs=10,
    # gpus=-1,
    logger=logger,
)
logger.watch(model)

trainer.fit(model, dm)
