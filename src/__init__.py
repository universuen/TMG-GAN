from src import (
    config,
    models,
    utils,
    data_modules,
)

from src.logger import Logger
from src.classifier import Classifier
from src.gan import GAN

from torchmetrics import Metric

Metric.full_state_update = False
