import random

import numpy as np
import torch
from sklearn.datasets import make_classification, make_blobs
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
    features, labels = make_blobs(1000, n_features=100, centers=5)
    src.datasets.feature_num = 100
    src.datasets.label_num = 5
    features = minmax_scale(features)
    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(labels).type(torch.LongTensor)
    temp = train_test_split(
        features,
        labels,
        test_size=0.1,
    )
    src.datasets.tr_features, src.datasets.te_features, src.datasets.tr_labels, src.datasets.te_labels = temp


def turn_on_test_mode():
    src.config.gan_config.epochs = 3
    src.config.gan_config.batch_size = 100
    src.config.classifier_config.epochs = 3
