import torch
from torch import Tensor

from src.datasets.tr_dataset import TrDataset
from src.datasets.te_dataset import TeDataset

tr_samples = torch.zeros([100, 10])
tr_labels = torch.zeros([100])

te_samples: Tensor = None
te_labels: Tensor = None

feature_num: int = None
label_num: int = None
