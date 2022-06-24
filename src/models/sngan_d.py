import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from src.utils import init_weights


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
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)
