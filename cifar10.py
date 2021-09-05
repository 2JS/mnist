import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pathlib import Path

from model import *
from datamodule import CIFAR10DataModule

if __name__=='__main__':
    pl.seed_everything(0)

    logger = WandbLogger(project='cifar10', log_model=True, id='25lxoitk')

    trainer = pl.Trainer(
        max_epochs=200,
        gpus=-1,
        logger=logger,
    )

    checkpoint_reference = f'teamvkik/cifar10/model-25lxoitk:latest'
    artifact = trainer.logger.experiment.use_artifact(checkpoint_reference, type='model')
    artifact_dir = artifact.download()

    dm = CIFAR10DataModule()

    model = ConvClassifier.load_from_checkpoint(Path(artifact_dir)/'model.ckpt')

    logger.watch(model)

    trainer.fit(model, dm)
