from torch import nn

from src.models._model import Model


class GeneratorModel(Model):
    def __init__(self, z_size: int, feature_num: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_size, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, feature_num),
        )
