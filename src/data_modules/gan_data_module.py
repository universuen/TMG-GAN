from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from src import config
from ._dataset import Dataset


class GANDataModule(LightningDataModule):

    def __init__(self, target_class: int):
        super().__init__()
        self.data: Dataset = None
        self.target_class = target_class

    def setup(self, stage: Optional[str] = None) -> None:
        data = Dataset(
            samples_path=config.path_config.datasets / 'x_train.csv',
            labels_path=config.path_config.datasets / 'y_train.csv',
        )
        target_samples = []
        target_labels = []
        for sample, label in data:
            if label == self.target_class:
                target_samples.append(sample)
                target_labels.append(label)
        data.samples = torch.stack(target_samples)
        data.labels = torch.stack(target_labels)

        self.data = data

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data,
            batch_size=config.gan_config.batch_size,
            num_workers=4
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data,
            batch_size=len(self.data),
            num_workers=4
        )

