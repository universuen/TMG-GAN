import random

import torch
from torch.nn.functional import binary_cross_entropy

from src import models, config, datasets


class GAN:

    def __init__(self):
        self.name = 'GAN'
        self.discriminator = models.GANDModel(datasets.feature_num).to(config.device)

        self.generator = models.GeneratorModel(config.gan_config.z_size, datasets.feature_num).to(config.device)

        self.samples = dict()

    def fit(self, dataset):

        self.discriminator.train()
        self.generator.train()

        if len(self.samples) == 0:
            self._divide_samples(dataset)
        d_optimizer = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=config.gan_config.cd_lr,
            betas=(0.5, 0.999),
        )

        g_optimizer = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=config.gan_config.cd_lr,
            betas=(0.5, 0.999),
        )

        for e in range(config.gan_config.epochs):
            print(f'\r{(e + 1) / config.gan_config.epochs: .2%}', end='')
            for target_label in range(datasets.label_num):
                # train D
                for _ in range(config.gan_config.cd_loop_num):
                    d_optimizer.zero_grad()
                    real_samples = self._get_target_samples(target_label, config.gan_config.batch_size)
                    score_real = self.discriminator(real_samples)
                    loss_real = binary_cross_entropy(score_real, torch.ones_like(score_real))
                    generated_samples = self.generator.generate_samples(config.gan_config.batch_size)
                    score_generated = self.discriminator(generated_samples)
                    loss_generated = binary_cross_entropy(
                        score_generated,
                        torch.zeros_like(score_generated, device=config.device)
                    )
                    d_loss = (loss_real + loss_generated) / 2
                    d_loss.backward()
                    d_optimizer.step()
                # train G
                for _ in range(config.gan_config.g_loop_num):
                    g_optimizer.zero_grad()
                    generated_samples = self.generator.generate_samples(config.gan_config.batch_size)
                    score_generated = self.discriminator(generated_samples)
                    g_loss = binary_cross_entropy(
                        score_generated,
                        torch.ones_like(score_generated, device=config.device)
                    )
                    g_loss.backward()
                    g_optimizer.step()
        print()
        self.discriminator.eval()
        self.generator.eval()

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
        return self.generator.generate_samples(num).cpu().detach()
