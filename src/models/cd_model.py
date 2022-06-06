import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from src.models._model import Model


class CDModel(Model):
    def __init__(self, in_features: int, label_num: int):
        super().__init__()
        self.main_model = nn.Sequential(
            spectral_norm(nn.Linear(in_features, 512)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(512, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 32)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(32, 16)),
            nn.LeakyReLU(0.2),
        )
        self.hidden_status: torch.Tensor = None
        self.c_last_layer = nn.Sequential(
            nn.Linear(16, label_num),
            nn.Softmax(dim=1),
        )
        self.d_last_layer = nn.Sequential(
            spectral_norm(nn.Linear(16, 1)),
        )

    def c_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.apply(self._init_weights)
            self.initialized = True
        x = self.main_model(x)
        self.hidden_status = x
        x = self.c_last_layer(x)
        return x

    def d_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.apply(self._init_weights)
            self.initialized = True
        x = self.main_model(x)
        self.hidden_status = x
        x = self.d_last_layer(x)
        return x
