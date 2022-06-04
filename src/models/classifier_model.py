import torch
from torch import nn
from torch.nn.functional import cross_entropy

from src.config import classifier_config
from ._model import Model


class ClassifierModel(Model):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.main_body = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.LeakyReLU(),
        )
        self.hidden_status: torch.Tensor = None
        self.out_layer = nn.Sequential(
            nn.Linear(16, out_features),
            nn.Softmax(1),
        )
        self.full_state_update = False

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        self.hidden_status = self.main_body(x)
        return self.out_layer(self.hidden_status)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=classifier_config.lr)

    def training_step(
            self,
            batch: torch.Tensor,
            batch_idx: int,
    ):
        x, labels = batch
        prediction = self(x)
        loss = cross_entropy(
            input=prediction,
            target=labels,
        )
        # self.log('training_loss', loss)
        return loss
