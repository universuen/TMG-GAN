import random

import torch
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter

from src import models, config, datasets


class TMGGAN:

    def __init__(self):
        self.cd = models.CDModel(datasets.feature_num, datasets.label_num).to(config.device)
        self.generators = [
            models.GeneratorModel(config.gan_config.z_size, datasets.feature_num).to(config.device)
            for _ in range(datasets.label_num)
        ]
        self.samples = dict()
        self.writer = SummaryWriter()

    def fit(self, dataset):
        self._divide_samples(dataset)
        cd_optimizer = torch.optim.Adam(
            params=self.cd.parameters(),
            lr=config.gan_config.cd_lr,
            betas=(0.5, 0.999),
        )
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
            # prepare data
            sample_num_per_class = config.gan_config.batch_size // datasets.label_num
            real_labels = []
            real_samples = []
            for i in range(datasets.label_num):
                real_labels.extend([i for _ in range(sample_num_per_class)])
                real_samples.append(self._get_target_samples(i, sample_num_per_class))
            real_labels = torch.tensor(real_labels, device=config.device)
            real_samples = torch.cat(real_samples)
            c_loss: torch.Tensor = None
            d_loss: torch.Tensor = None
            g_loss: torch.Tensor = None

            # train C and D
            for _ in range(config.gan_config.cd_loop_num):
                cd_optimizer.zero_grad()
                score_real, predicted_labels = self.cd(real_samples)
                score_real = score_real.mean()
                generated_samples = [
                    self.generators[i].generate_samples(sample_num_per_class)
                    for i in range(datasets.label_num)
                ]
                generated_samples = torch.cat(generated_samples)
                score_generated = self.cd(generated_samples)[0].mean()
                d_loss = (score_generated - score_real) / 2
                c_loss = cross_entropy(
                    input=predicted_labels,
                    target=real_labels,
                )
                loss = d_loss + c_loss
                loss.backward()
                cd_optimizer.step()
            self.writer.add_scalars(
                'cd_loss',
                {
                    'c_loss': c_loss,
                    'd_loss': d_loss,
                },
                e,
            )

            # train G
            for _ in range(config.gan_config.g_loop_num):
                for i in range(datasets.label_num):
                    g_optimizers[i].zero_grad()
                generated_samples = [
                    self.generators[i].generate_samples(sample_num_per_class)
                    for i in range(datasets.label_num)
                ]
                generated_samples = torch.cat(generated_samples)
                score_generated, predicted_labels = self.cd(generated_samples)
                score_generated = score_generated.mean()
                loss_label = cross_entropy(
                    input=predicted_labels,
                    target=real_labels,
                )
                g_loss = -score_generated + loss_label
                g_loss.backward()
                for i in range(datasets.label_num):
                    g_optimizers[i].step()
            self.writer.add_scalar('g_loss', g_loss, e)
        self.writer.close()

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
