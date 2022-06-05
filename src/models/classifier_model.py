import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torchmetrics.functional import precision, recall, f1_score, accuracy
from pytorch_lightning import LightningModule

from src.config import classifier_config


class ClassifierModel(LightningModule):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.main_body = nn.Sequential(
            nn.Linear(num_features, 64),
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
            nn.Linear(16, num_classes),
            nn.Softmax(1),
        )

        self.num_classes = num_classes

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(layer: nn.Module):
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, 0.0, 0.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        self.hidden_status = self.main_body(x)
        return self.out_layer(self.hidden_status)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=classifier_config.lr)

    def training_step(
            self,
            batch: list[torch.Tensor],
            batch_idx: int,
    ):
        x, labels = batch
        prob = self.forward(x)
        loss = cross_entropy(
            input=prob,
            target=labels,
        )
        self.log('training_loss', loss, prog_bar=True)
        return loss

    def test_step(
            self,
            batch: list[torch.Tensor],
            batch_idx: int,
    ):
        x, labels = batch
        prediction = torch.argmax(self.forward(x), dim=1)
        self.log('Precision', precision(prediction, labels, average='macro', num_classes=self.num_classes))
        self.log('Recall', recall(prediction, labels, average='macro', num_classes=self.num_classes))
        self.log('F1', f1_score(prediction, labels, average='macro', num_classes=self.num_classes))
        self.log('Accuracy', accuracy(prediction, labels, average='macro', num_classes=self.num_classes))
