import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model: nn.Module = None
        self.initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.apply(self._init_weights)
            self.initialized = True
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
