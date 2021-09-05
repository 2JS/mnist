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

    logger = WandbLogger(project='cifar10', log_model=True)

    trainer = pl.Trainer(
        max_epochs=200,
        gpus=-1,
        logger=logger,
    )

    checkpoint_reference = 'teamvkik/cifar10/model-1pgkfk53:v0'
    run = trainer.logger.experiment
    artifact = run.use_artifact(checkpoint_reference, type='model')
    artifact_dir = artifact.download()

    dm = CIFAR10DataModule()

    model = ConvClassifier.load_from_checkpoint(Path(artifact_dir)/'model.ckpt')

    logger.watch(model)

    trainer.fit(model, dm)
