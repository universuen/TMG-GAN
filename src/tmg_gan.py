import torch

import src


class TMGGAN:
    def __init__(self, g_num: int, reduced_dim_num: int = 30):
        self.logger = src.Logger(self.__class__.__name__)
        self.generators = [src.models.GeneratorModel(src.config.gan_config, reduced_dim_num) for _ in range(g_num)]
        self.cd = src.models.CDModel(reduced_dim_num, src.datasets.label_num)
        self.samples = dict()

    def fit(self, dataset: src.datasets.TrDataset):
        assert min(dataset.labels) == 0
        assert max(dataset.labels) == len(self.generators) - 1

    def _divide_samples(self):
        pass


