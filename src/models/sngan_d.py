import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class SNGANDModel(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Linear(in_features, 1024)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(512, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 32)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(32, 16)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(16, 1)),
        )
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    @staticmethod
    def _init_weights(layer: nn.Module):
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, 0.0, 0.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif type(layer) == nn.BatchNorm1d:
            nn.init.normal_(layer.weight, 1.0, 0.02)
            nn.init.constant_(layer.bias, 0)