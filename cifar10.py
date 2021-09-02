import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from model import *
from datamodule import CIFAR10DataModule

if __name__=='__main__':
    dm = CIFAR10DataModule()

    model = resnet34()

    logger = WandbLogger(project='cifar10', log_model=True)

    trainer = pl.Trainer(
        max_epochs=100,
        gpus=-1,
        logger=logger,
    )
    logger.watch(model)

    trainer.fit(model, dm)
