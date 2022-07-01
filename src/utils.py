import random

import numpy as np
import torch
import pandas as pd
from torch import nn
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


def set_dataset_values():
    src.datasets.feature_num = len(src.datasets.tr_samples[0])
    src.datasets.label_num = int(max(src.datasets.tr_labels).item()) + 1


def prepare_datasets(name: str = None):
    if name is not None:
        tr_features = pd.read_csv(src.config.path_config.datasets / name / 'x_train.csv')
        tr_features = torch.tensor(tr_features.values, dtype=torch.float)
        src.datasets.tr_samples = tr_features

        tr_labels = pd.read_csv(src.config.path_config.datasets / name / 'y_train.csv')
        try:
            tr_labels = torch.tensor(tr_labels.values, dtype=torch.float).squeeze().argmax(1).type(torch.LongTensor)
        except IndexError:
            tr_labels = torch.tensor(tr_labels.values, dtype=torch.float).squeeze().type(torch.LongTensor)
        src.datasets.tr_labels = tr_labels

        te_features = pd.read_csv(src.config.path_config.datasets / name / 'x_test.csv')
        te_features = torch.tensor(te_features.values, dtype=torch.float)
        src.datasets.te_samples = te_features

        te_labels = pd.read_csv(src.config.path_config.datasets / name / 'y_test.csv')
        try:
            te_labels = torch.tensor(te_labels.values, dtype=torch.float).squeeze().argmax(1).type(torch.LongTensor)
        except IndexError:
            te_labels = torch.tensor(te_labels.values, dtype=torch.float).squeeze().type(torch.LongTensor)
        src.datasets.te_labels = te_labels
        set_dataset_values()
    else:
        src.datasets.feature_num = 30
        src.datasets.label_num = 5
        samples, labels = make_blobs(1000, n_features=src.datasets.feature_num, centers=src.datasets.label_num)
        # samples, labels = make_classification(
        #     n_samples=1000,
        #     n_features=src.datasets.feature_num,
        #     n_informative=src.datasets.feature_num - 2,
        #     n_redundant=0,
        #     n_classes=5,
        #     n_clusters_per_class=2,
        #     weights=[0.5, 0.3, 0.1, 0.05, 0.05],
        # )
        samples = minmax_scale(samples)
        samples = torch.tensor(samples, dtype=torch.float)
        labels = torch.tensor(labels).type(torch.LongTensor)
        temp = train_test_split(
            samples,
            labels,
            test_size=0.1,
        )
        src.datasets.tr_samples, src.datasets.te_samples, src.datasets.tr_labels, src.datasets.te_labels = temp


def transfer_to_binary():
    for idx, item in enumerate(src.datasets.tr_labels):
        if item > 0:
            src.datasets.tr_labels[idx] = 1
    for idx, item in enumerate(src.datasets.te_labels):
        if item > 0:
            src.datasets.te_labels[idx] = 1


def turn_on_test_mode():
    src.datasets.tr_samples = src.datasets.tr_samples[:1000]
    src.datasets.tr_labels = src.datasets.tr_labels[:1000]
    src.datasets.te_samples = src.datasets.te_samples[:1000]
    src.datasets.te_labels = src.datasets.te_labels[:1000]
    src.config.gan_config.epochs = 1
    src.config.classifier_config.epochs = 1


def init_weights(layer: nn.Module):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, 0.0, 0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif type(layer) == nn.BatchNorm1d:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0)
