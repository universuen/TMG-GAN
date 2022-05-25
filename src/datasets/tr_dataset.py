import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import pandas as pd

from src.config import device, path_config


class TrDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.features = pd.read_csv(path_config.data / 'tr_feature.csv', header=None)
        self.features = torch.tensor(self.features.values, dtype=torch.float)
        self.labels = pd.read_csv(path_config.data / 'tr_label.csv', header=None)
        self.labels = torch.tensor(self.labels.values, dtype=torch.float).squeeze().type(torch.LongTensor)
        # self.labels = one_hot(self.labels).squeeze(dim=1)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

