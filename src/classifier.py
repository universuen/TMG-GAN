import torch
import numpy as np
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from sklearn import metrics
from torch.utils.data import DataLoader

from src import config, datasets, logger, models


class Classifier:
    def __init__(self, name: str):
        self.name = f'{name}_classifier'
        self.model = models.CDModel(datasets.feature_num, datasets.label_num).to(config.device)
        self.logger = logger.Logger(name)
        self.confusion_matrix: np.ndarray = None
        self.metrics = {
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'Accuracy': 0.0,
        }

    def fit(self, dataset: datasets.TrDataset):
        self.model.train()
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')
        optimizer = Adam(
            params=self.model.parameters(),
            lr=config.classifier_config.lr,
        )
        dl = DataLoader(dataset, config.classifier_config.batch_size, shuffle=True)
        for e in range(config.classifier_config.epochs):
            for idx, (samples, labels) in enumerate(dl):
                print(f'\repoch {e + 1} / {config.classifier_config.epochs}: {(idx + 1) / len(dl): .2%}', end='')
                self.model.zero_grad()
                prediction = self.model(samples)[1]
                loss = cross_entropy(
                    input=prediction,
                    target=labels,
                )
                loss.backward()
                optimizer.step()
        print('')
        self.model.eval()
        self.logger.info('Finished training')

    def predict(self, x: torch.Tensor, use_prob: bool = False) -> torch.Tensor:
        with torch.no_grad():
            prob = self.model(x)[1]
        if use_prob:
            return prob.squeeze(dim=1).detach()
        else:
            return torch.argmax(prob, dim=1)

    def test(self, dataset: datasets.TeDataset):
        self.model = self.model.cpu()
        predicted_labels = self.predict(dataset.samples.cpu())
        real_labels = dataset.labels.cpu()
        self.confusion_matrix = metrics.multilabel_confusion_matrix(
            y_true=real_labels,
            y_pred=predicted_labels,
            labels=[i for i in range(datasets.label_num)]
        )
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
        self.model = self.model.to(config.device)
