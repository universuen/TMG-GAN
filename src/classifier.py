import torch
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from sklearn.metrics import accuracy_score

from src import config, datasets, logger, models


class Classifier:
    def __init__(self, name: str):
        self.name = f'{name}_classifier'
        self.model = models.ClassifierModel(122, 5).to(config.device)
        self.logger = logger.Logger(name)
        self.metrics = {
            'Acc': 0.0,
        }

    def fit(self, dataset: datasets.TrDataset):
        self.model.train()
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')
        optimizer = Adam(
                params=self.model.parameters(),
                lr=config.classifier_config.lr,
                betas=(0.5, 0.9),
            )
        x, labels = dataset.features, dataset.labels
        for _ in range(config.classifier_config.epochs):
            self.model.zero_grad()
            prediction = self.model(x)
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
        self.metrics['Acc'] = accuracy_score(
            dataset.labels.cpu().numpy(),
            self.predict(dataset.features).cpu().numpy()
        )


