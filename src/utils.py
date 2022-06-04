import random

import numpy as np
import torch
import pandas as pd
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
    tr_features = pd.read_csv(src.config.path_config.datasets / 'x_train.csv')
    tr_features = torch.tensor(tr_features.values, dtype=torch.float)
    src.datasets.tr_features = tr_features

    tr_labels = pd.read_csv(src.config.path_config.datasets / 'y_train.csv')
    tr_labels = torch.tensor(tr_labels.values, dtype=torch.float).squeeze().argmax(1).type(torch.LongTensor)
    src.datasets.tr_labels = tr_labels

    te_features = pd.read_csv(src.config.path_config.datasets / 'x_test.csv')
    te_features = torch.tensor(te_features.values, dtype=torch.float)
    src.datasets.te_features = te_features

    te_labels = pd.read_csv(src.config.path_config.datasets / 'y_test.csv')
    te_labels = torch.tensor(te_labels.values, dtype=torch.float).squeeze().argmax(1).type(torch.LongTensor)
    src.datasets.te_labels = te_labels

    src.datasets.feature_num = len(tr_features[0])
    src.datasets.label_num = max(tr_labels).item() + 1

    # src.datasets.feature_num = 100
    # src.datasets.label_num = 5
    # features, labels = make_blobs(1000, n_features=src.datasets.feature_num, centers=src.datasets.label_num)
    # features = minmax_scale(features)
    # features = torch.tensor(features, dtype=torch.float)
    # labels = torch.tensor(labels).type(torch.LongTensor)
    # temp = train_test_split(
    #     features,
    #     labels,
    #     test_size=0.1,
    # )
    # src.datasets.tr_features, src.datasets.te_features, src.datasets.tr_labels, src.datasets.te_labels = temp


def turn_on_test_mode():
    print('*************TEST MODE IS ON!*************')
    src.config.gan_config.epochs = 3
    src.config.gan_config.batch_size = 100
    src.config.classifier_config.epochs = 3
