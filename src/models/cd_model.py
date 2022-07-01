import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from src.utils import init_weights


class CDModel(nn.Module):
    def __init__(self, in_features: int, label_num: int):
        super().__init__()
        self.resize = nn.Linear(in_features, 78)
        self.main_model = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 32, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.2),
            spectral_norm(nn.Linear(320, 256)),
            nn.ReLU(),
        )
        self.hidden_status: torch.Tensor = None
        self.c_last_layer = nn.Sequential(
            nn.Linear(256, label_num),
            nn.Softmax(dim=1),
        )
        self.d_last_layer = nn.Sequential(
            spectral_norm(nn.Linear(256, 1)),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.resize(x)
        x = torch.reshape(x, [-1, 1, 13, 6])
        x = self.main_model(x)
        self.hidden_status = x
        return self.d_last_layer(x), self.c_last_layer(x)
