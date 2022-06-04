import torch
import pytorch_lightning as pl
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm

from src import config, datasets, logger, models


class Classifier:
    def __init__(self):
        self.model = models.ClassifierModel(datasets.feature_num, datasets.label_num)
        self.trainer = pl.Trainer(
            max_epochs=config.classifier_config.epochs,
            accelerator=config.device,
            devices=1,
        )
        self.logger = logger.Logger(self.__class__.__name__)
        self.metrics = {
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'Accuracy': 0.0,
        }

    def fit(self, dataset: datasets.TrDataset):
        self.trainer.fit(
            model=self.model,
            train_dataloaders=DataLoader(
                dataset=dataset,
                batch_size=config.classifier_config.batch_size,
                num_workers=4,
            ),
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.model(x), dim=1)

    def test(self, dataset: datasets.TeDataset):
        if self.model is None:
            self.model = models.ClassifierModel(datasets.feature_num, datasets.label_num)
        predicted_labels = self.predict(dataset.features).cpu()
        real_labels = dataset.labels.cpu()
        self.metrics['Precision'] = metrics.precision_score(
            y_true=real_labels,
            y_pred=predicted_labels,
            average='macro',
            zero_division=0,
        )
        self.metrics['Recall'] = metrics.recall_score(
            y_true=real_labels,
            y_pred=predicted_labels,
            average='macro',
            zero_division=0,
        )
        self.metrics['F1'] = metrics.f1_score(
            y_true=real_labels,
            y_pred=predicted_labels,
            average='macro',
            zero_division=0,
        )
        self.metrics['Accuracy'] = metrics.accuracy_score(
            y_true=real_labels,
            y_pred=predicted_labels,
        )
