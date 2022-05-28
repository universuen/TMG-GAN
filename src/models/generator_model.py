import torch
from torch import nn

from src.models._model import Model
from src import config


class GeneratorModel(Model):
    def __init__(self, z_size: int, feature_num: int):
        super().__init__()
        self.z_size = z_size
        self.model = nn.Sequential(
            nn.Linear(z_size, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, feature_num),
        )

    def make_samples(self, num: int) -> torch.Tensor:
        z = torch.randn(num, self.z_size, device=config.device)
        return self.forward(z)
