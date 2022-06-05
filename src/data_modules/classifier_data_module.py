from typing import Optional

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from src import config
from ._dataset import Dataset


class ClassifierDataModule(LightningDataModule):

    def __init__(self):
        super().__init__()
        self.data: Dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.data = Dataset(
                samples_path=config.path_config.datasets / 'x_train.csv',
                labels_path=config.path_config.datasets / 'y_train.csv',
            )
        elif stage == 'test':
            self.data = Dataset(
                samples_path=config.path_config.datasets / 'x_test.csv',
                labels_path=config.path_config.datasets / 'y_test.csv',
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data,
            batch_size=config.classifier_config.batch_size,
            num_workers=4,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data,
            batch_size=len(self.data),
            num_workers=4,
        )
