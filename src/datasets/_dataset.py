import torch


from src import datasets, config


class Dataset:
    def __init__(self, training: bool = True):
        if training:
            self.features = datasets.tr_features
            self.labels = datasets.tr_labels
        else:
            self.features = datasets.te_features
            self.labels = datasets.te_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]
