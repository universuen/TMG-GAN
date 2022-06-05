import pytorch_lightning as pl
import torch

from src import config, models
from src.data_modules import GANDataModule


class GAN:
    def __init__(self, target_class: int):
        self.model = models.GANModule(20)
        self.trainer = pl.Trainer(
            max_epochs=config.gan_config.epochs,
            accelerator=config.device,
            devices=1,
        )
        self.target_class = target_class

    def fit(self):
        self.trainer.fit(
            model=self.model,
            datamodule=GANDataModule(self.target_class),
        )

    def test(self):
        self.trainer.test(
            model=self.model,
            datamodule=GANDataModule(self.target_class),
        )
