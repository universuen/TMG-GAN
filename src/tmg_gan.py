import torch
from torch.utils.data import DataLoader

from src import models, config, datasets


class TMGGAN:
    def __init__(self):
        self.discriminator = models.CDModel(datasets.feature_num, datasets.label_num).to(config.device)
        self.generator = models.GeneratorModel(config.gan_config.z_size, datasets.feature_num).to(config.device)

    def fit(self, dataset):
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
        dl = DataLoader(dataset, batch_size=config.gan_config.batch_size)
        for e in range(config.gan_config.epochs):
            for idx, (real_samples, labels) in enumerate(dl):
                print(f'\repoch {e + 1} / {config.gan_config.epochs}: {idx / len(dl): .2%}', end='')
                # train D
                for _ in range(config.gan_config.cd_loop_num):
                    d_optimizer.zero_grad()
                    score_real = self.discriminator.d_forward(real_samples).mean()
                    generated_samples = self.generator.generate_samples(len(real_samples))
                    score_generated = self.discriminator.d_forward(generated_samples).mean()
                    d_loss = (score_generated - score_real) / 2
                    d_loss.backward()
                    d_optimizer.step()

                # train G
                for _ in range(config.gan_config.g_loop_num):
                    g_optimizer.zero_grad()
                    generated_samples = self.generator.generate_samples(len(real_samples))
                    score_generated = self.discriminator.d_forward(generated_samples).mean()
                    g_loss = -score_generated
                    g_loss.backward()
                    g_optimizer.step()

    def generate_samples(self, num: int):
        return self.generator.generate_samples(num).cpu().detach()
