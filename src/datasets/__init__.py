from torch import Tensor

from src.datasets.tr_dataset import TrDataset
from src.datasets.te_dataset import TeDataset

tr_features: Tensor = None
tr_labels: Tensor = None

te_features: Tensor = None
te_labels: Tensor = None

feature_num: int = None
label_num: int = None
