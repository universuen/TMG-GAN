import torch

from src.models.cd_model import CDModel


class ClassifierModel(CDModel):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_forward(x)
