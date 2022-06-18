import random

import torch
from torch.nn.functional import cross_entropy, cosine_similarity

from src import models, config, datasets


class TMGGAN:

    def __init__(self):
        self.cd = models.CDModel(datasets.feature_num, datasets.label_num).to(config.device)
        self.generators = [
            models.GeneratorModel(config.gan_config.z_size, datasets.feature_num).to(config.device)
            for _ in range(datasets.label_num)
        ]
        self.samples = dict()

    def fit(self, dataset):

        self.cd.train()
        for i in self.generators:
            i.train()

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

            for target_label in range(datasets.label_num):
                # train C and D
                for _ in range(config.gan_config.cd_loop_num):
                    cd_optimizer.zero_grad()
                    real_samples = self._get_target_samples(target_label, config.gan_config.batch_size)
                    score_real, predicted_labels = self.cd(real_samples)
                    score_real = score_real.mean()
                    generated_samples = self.generators[target_label].generate_samples(config.gan_config.batch_size)
                    score_generated = self.cd(generated_samples)[0].mean()
                    d_loss = (score_generated - score_real) / 2
                    c_loss = cross_entropy(
                        input=predicted_labels,
                        target=torch.full([len(predicted_labels)], target_label, device=config.device),
                    )
                    loss = d_loss + c_loss
                    loss.backward()
                    cd_optimizer.step()

                # train G
                for _ in range(config.gan_config.g_loop_num):
                    g_optimizers[target_label].zero_grad()
                    generated_samples = self.generators[target_label].generate_samples(config.gan_config.batch_size)
                    real_samples = self._get_target_samples(target_label, config.gan_config.batch_size)
                    self.cd(real_samples)
                    hidden_real = self.cd.hidden_status
                    score_generated, predicted_labels = self.cd(generated_samples)
                    hidden_generated = self.cd.hidden_status
                    cd_hidden_loss = - cosine_similarity(hidden_real, hidden_generated).mean()
                    score_generated = score_generated.mean()
                    loss_label = cross_entropy(
                        input=predicted_labels,
                        target=torch.full([len(predicted_labels)], target_label, device=config.device),
                    )
                    if e < 1000:
                        cd_hidden_loss = 0
                    g_loss = -score_generated + loss_label + cd_hidden_loss
                    g_loss.backward()
                    g_optimizers[target_label].step()
            for i in g_optimizers:
                i.zero_grad()
            for i in self.generators:
                i.generate_samples(3)
            g_hidden_losses = []
            for i, _ in enumerate(self.generators):
                for j, _ in enumerate(self.generators):
                    if i == j:
                        continue
                    else:
                        g_hidden_losses.append(
                            torch.dot(
                                self.generators[i].hidden_status.flatten(),
                                self.generators[j].hidden_status.flatten(),
                            )
                        )
            g_hidden_loss = torch.mean(torch.stack(g_hidden_losses))
            g_hidden_loss.backward()
            for i in g_optimizers:
                i.step()
        print('')
        self.cd.eval()
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

    def generate_qualified_samples(self, target_label: int, num: int):
        result = []
        patience = 10
        while len(result) < num:
            sample = self.generators[target_label].generate_samples(1)
            label = torch.argmax(self.cd(sample)[1])
            if label == target_label or patience == 0:
                result.append(sample.cpu().detach())
                patience = 10
            else:
                patience -= 1
        return torch.cat(result)
