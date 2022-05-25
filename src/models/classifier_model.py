import torch
from torch import nn

from src.models._model import Model


class ClassifierModel(Model):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.main_model = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.2),
        )
        self.last_layer = nn.Sequential(
            nn.Linear(32, out_features),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.apply(self._init_weights)
            self.initialized = True
        x = self.main_model(x)
        x = self.last_layer(x)
        return x
