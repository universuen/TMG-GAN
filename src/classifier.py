import pytorch_lightning as pl

from src import config, models
from src.data_modules import ClassifierDataModule


class Classifier:
    def __init__(self, features_num: int, labels_num: int):
        self.model = models.ClassifierModel(features_num, labels_num)
        self.trainer = pl.Trainer(
            max_epochs=config.classifier_config.epochs,
            accelerator=config.device,
            devices=1,
        )

    def fit(self):
        self.trainer.fit(
            model=self.model,
            datamodule=ClassifierDataModule(),
        )

    def test(self):
        self.trainer.test(
            model=self.model,
            datamodule=ClassifierDataModule(),
        )
