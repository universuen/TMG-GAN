import torch
from torch import nn

from src import config
from src.utils import init_weights


class GeneratorModel(nn.Module):
    def __init__(self, z_size: int, feature_num: int):
        super().__init__()
        self.z_size = z_size
        self.main_model = nn.Sequential(
            # z_size * 1 * 1
            nn.ConvTranspose2d(z_size, 64, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # 64 * 4 * 4
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # 32 * 8 * 8
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16 * 16 * 16
        )
        self.hidden_status: torch.Tensor = None
        self.last_layer = nn.Sequential(
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2),
            # 3 * 32 * 32
            nn.Tanh()
        )
        self.apply(init_weights)

    def generate_samples(self, num: int) -> torch.Tensor:
        z = torch.randn(num, self.z_size, device=config.device)
        return self.forward(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.z_size, 1, 1)
        x = self.main_model(x)
        self.hidden_status = x
        return torch.reshape(self.last_layer(x), [-1, 3, 32, 32])

