import torch
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from sklearn import metrics
from torch.utils.data import DataLoader

from src import config, datasets, logger, models


class Classifier:
    def __init__(self, name: str):
        self.name = f'{name}_classifier'
        self.model = models.ClassifierModel(datasets.feature_num, datasets.label_num).to(config.device)
        self.logger = logger.Logger(name)
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
        dl = DataLoader(dataset, config.classifier_config.batch_size)
        for e in range(config.classifier_config.epochs):
            for idx, (samples, labels) in enumerate(dl):
                print(f'\repoch {e + 1} / {config.classifier_config.epochs}: {idx / len(dl): .2%}', end='')
                self.model.zero_grad()
                prediction = self.model(samples)
                loss = cross_entropy(
                    input=prediction,
                    target=labels,
                )
                loss.backward()
                optimizer.step()

        self.model.eval()
        self.logger.info('Finished training')

    def predict(self, x: torch.Tensor, use_prob: bool = False) -> torch.Tensor:
        x = x.to(config.device)
        with torch.no_grad():
            prob = self.model(x)
        if use_prob:
            return prob.squeeze(dim=1).detach()
        else:
            return torch.argmax(prob, dim=1)

    def test(self, dataset: datasets.TeDataset):
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
