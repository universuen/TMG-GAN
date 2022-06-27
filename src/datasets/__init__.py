import torch
from torchvision import datasets, transforms

from src.config import path_config
from src.datasets.tr_dataset import TrDataset
from src.datasets.te_dataset import TeDataset

tf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1),
    ]
)

tr_ds = datasets.MNIST(
    root=path_config.data,
    train=True,
    transform=tf,
    download=True
)
tr_samples, tr_labels = [], []
for i, j in tr_ds:
    tr_samples.append(i)
    tr_labels.append(j)
tr_samples = torch.cat(tr_samples)
tr_labels = torch.tensor(tr_labels)

te_ds = datasets.MNIST(
    root=path_config.data,
    train=False,
    transform=tf,
    download=True
)
te_samples, te_labels = [], []
for i, j in te_ds:
    te_samples.append(i)
    te_labels.append(j)
te_samples = torch.cat(te_samples)
te_labels = torch.tensor(te_labels)

feature_num = len(tr_samples[0].flatten())
label_num = max(tr_labels).item() + 1

pass
