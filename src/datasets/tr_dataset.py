from src.datasets._dataset import Dataset


class TrDataset(Dataset):
    def __init__(self):
        super().__init__(training=True)
