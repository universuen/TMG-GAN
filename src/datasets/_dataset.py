import torch


from src import datasets, config


class Dataset:
    def __init__(self, training: bool = True):
        if training:
            self.samples = datasets.tr_features.to(config.device)
            self.labels = datasets.tr_labels.to(config.device)
        else:
            self.samples = datasets.te_features.to(config.device)
            self.labels = datasets.te_labels.to(config.device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.samples[idx], self.labels[idx]
