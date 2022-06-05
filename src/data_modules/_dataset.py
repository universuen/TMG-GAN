from pathlib import Path

import torch
from torch.utils.data import Dataset as Dataset_
import pandas as pd


class Dataset(Dataset_):
    def __init__(
            self,
            samples_path: Path,
            labels_path: Path,
    ):
        samples = pd.read_csv(samples_path)
        self.samples = torch.tensor(
            data=samples.values,
            dtype=torch.float,
        )
        labels = pd.read_csv(labels_path)
        self.labels = torch.tensor(
            data=labels.values,
            dtype=torch.float
        ).squeeze().argmax(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.samples[idx], self.labels[idx]
