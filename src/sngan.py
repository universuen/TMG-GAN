import random

import torch
from matplotlib import pyplot as plt

from src import models, config, datasets


class SNGAN:

    def __init__(self):
        self.discriminators = [
            models.SNGANDModel(datasets.feature_num).to(config.device)
            for _ in range(datasets.label_num)
        ]
        self.generators = [
            models.GeneratorModel(config.gan_config.z_size, datasets.feature_num).to(config.device)
            for _ in range(datasets.label_num)
        ]
        self.samples = dict()

    def fit(self, dataset):

        for i in self.discriminators:
            i.train()
        for i in self.generators:
            i.train()

        self._divide_samples(dataset)
        d_optimizers = [
            torch.optim.Adam(
                params=self.discriminators[i].parameters(),
                lr=config.gan_config.cd_lr,
                betas=(0.5, 0.999),
            )
            for i in range(datasets.label_num)
        ]
        g_optimizers = [
            torch.optim.Adam(
                params=self.generators[i].parameters(),
                lr=config.gan_config.cd_lr,
                betas=(0.5, 0.999),
            )
            for i in range(datasets.label_num)
        ]

        for e in range(config.gan_config.epochs):
            print(f'\r{(e + 1) / config.gan_config.epochs: .2%}', end='')
            for target_label in range(datasets.label_num):
                # train D
                for _ in range(config.gan_config.cd_loop_num):
                    d_optimizers[target_label].zero_grad()
                    real_samples = self._get_target_samples(target_label, config.gan_config.batch_size)
                    score_real = self.discriminators[target_label](real_samples).mean()
                    loss_real = - score_real
                    generated_samples = self.generators[target_label].generate_samples(config.gan_config.batch_size)
                    score_generated = self.discriminators[target_label](generated_samples).mean()
                    loss_generated = score_generated
                    d_loss = (loss_real + loss_generated) / 2
                    d_loss.backward()
                    d_optimizers[target_label].step()
                # train G
                for _ in range(config.gan_config.g_loop_num):
                    g_optimizers[target_label].zero_grad()
                    generated_samples = self.generators[target_label].generate_samples(config.gan_config.batch_size)
                    score_generated = self.discriminators[target_label](generated_samples).mean()
                    g_loss = - score_generated
                    g_loss.backward()
                    g_optimizers[target_label].step()
                if e % 10 == 0:
                    with torch.no_grad():
                        for i in self.generators:
                            i.eval()
                        images = torch.cat([self.generate_samples(i, 10) for i in range(10)])
                        for i in self.generators:
                            i.train()
                    f, axs = plt.subplots(10, 10)
                    for i in range(10):
                        for j in range(10):
                            axs[i, j].imshow(images[i * 10 + j].permute(1, 2, 0))
                            axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                    plt.savefig(config.path_config.gan_outs / f'sngan_{e}.jpg')
                    plt.close(f)
        print()
        for i in self.discriminators:
            i.eval()
        for i in self.generators:
            i.eval()

    def _divide_samples(self, dataset: datasets.TrDataset) -> None:
        for sample, label in dataset:
            label = label.item()
            if label not in self.samples.keys():
                self.samples[label] = sample.unsqueeze(0)
            else:
                self.samples[label] = torch.cat([self.samples[label], sample.unsqueeze(0)])

    def _get_target_samples(self, label: int, num: int) -> torch.Tensor:
        return torch.stack(
            random.choices(
                self.samples[label],
                k=num,
            )
        )

    def generate_samples(self, target_label: int, num: int):
        return self.generators[target_label].generate_samples(num).cpu().detach()
