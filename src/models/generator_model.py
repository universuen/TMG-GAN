import torch
from torch import nn

from src import config
from src.utils import init_weights


class GeneratorModel(nn.Module):
    def __init__(self, z_size: int, feature_num: int):
        super().__init__()
        self.z_size = z_size
        self.main_model = nn.Sequential(
            nn.Linear(z_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.hidden_status: torch.Tensor = None
        self.last_layer = nn.Sequential(
            nn.Linear(1024, feature_num),
            nn.Sigmoid(),
        )
        self.apply(init_weights)

    def generate_samples(self, num: int) -> torch.Tensor:
        z = torch.randn(num, self.z_size, device=config.device)
        return self.forward(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main_model(x)
        self.hidden_status = x
        return torch.reshape(self.last_layer(x), [-1, 28, 28])

