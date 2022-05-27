import torch

from src import datasets, rbm, Logger


class PartialDBN:
    def __init__(
            self,
            in_features: int,
            out_features: int,
    ):
        self.layers = [
            rbm.RBM(in_features, 128),
            rbm.RBM(128, 64),
            rbm.RBM(64, out_features),
        ]
        self.logger = Logger(self.__class__.__name__)

    def fit(self, dataset: datasets.TrDataset):
        self.logger.info('Started training.')
        x = dataset.features
        for i in self.layers:
            i.fit(x)
            x = i.sample_hidden(x)
        self.logger.info('Finished training.')

    def extract(self, x: torch.Tensor):
        for i in self.layers:
            x = i.sample_hidden(x)
        return x

