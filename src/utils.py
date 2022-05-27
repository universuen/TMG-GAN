import random

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

import src


def set_random_state(seed: int = None) -> None:
    if seed is None:
        seed = src.config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_datasets():
    set_random_state()
    features, labels = make_classification(
        n_samples=100000,
        n_features=100,
        n_informative=90,
        n_classes=10,
    )
    src.datasets.feature_num = 100
    src.datasets.label_num = 10
    features = minmax_scale(features)
    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(labels).type(torch.LongTensor)
    temp = train_test_split(
        features,
        labels,
        test_size=0.1,
    )
    src.datasets.tr_features, src.datasets.te_features, src.datasets.tr_labels, src.datasets.te_labels = temp
