import random

import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm

import src


class TMGGAN:
    def __init__(self, g_num: int, reduced_dim_num: int = 30):
        self.logger = src.Logger(self.__class__.__name__)
        self.generators = [
            src.models.GeneratorModel(src.config.gan_config.z_size, reduced_dim_num).to(src.config.device)
            for _ in range(g_num)
        ]
        self.cd = src.models.CDModel(reduced_dim_num, src.datasets.label_num).to(src.config.device)
        self.dbn = src.PartialDBN(src.datasets.feature_num, reduced_dim_num)
        self.samples = dict()

    def fit(self, dataset: src.datasets.TrDataset):
        assert min(dataset.labels) == 0
        assert max(dataset.labels).item() == len(self.generators) - 1

        self.logger.info('Started training DBN')
        self.dbn.fit(dataset)
        self.logger.info('Finished training DBN')

        self.logger.info('Started preprocessing samples')
        dataset.features = self.dbn.extract(dataset.features)
        self._divide_samples(dataset)
        self.logger.info('Finished preprocessing samples')

        self.logger.info('Started training TMG-GAN')
        for label in self.samples.keys():
            self.generators[label].train()
        g_optimizers = [
            torch.optim.Adam(
                params=i.parameters(),
                lr=src.config.gan_config.g_lr,
                betas=(0.5, 0.999),
            )
            for i in self.generators
        ]
        cd_optimizer = torch.optim.Adam(
            params=self.cd.parameters(),
            lr=src.config.gan_config.cd_lr,
            betas=(0.5, 0.999),
        )
        for _ in tqdm(range(src.config.gan_config.epochs)):
            for __ in range(src.config.gan_config.cd_loop_num):
                real_x = []
                for label in self.samples.keys():
                    real_x.append(self._get_target_samples(label, src.config.gan_config.batch_size))
                real_x = torch.cat(real_x)

                fake_x = []
                for label in self.samples.keys():
                    fake_x.append(self.generators[label].make_samples(src.config.gan_config.batch_size).detach())
                fake_x = torch.cat(fake_x)

                real_labels = []
                for label in self.samples.keys():
                    real_labels.extend([label for _ in range(src.config.gan_config.batch_size)])
                real_labels = torch.tensor(real_labels, device=src.config.device)

                fake_labels = []
                for label in self.samples.keys():
                    fake_labels.extend([label for _ in range(src.config.gan_config.batch_size)])
                fake_labels = torch.tensor(fake_labels, device=src.config.device)

                self.cd.zero_grad()

                prediction_real = self.cd.d_forward(real_x)
                loss_real = - prediction_real.mean()

                prediction_fake = self.cd.d_forward(fake_x)
                loss_fake = prediction_fake.mean()

                predicted_labels = self.cd.c_forward(torch.cat([real_x, fake_x]))
                label_loss = cross_entropy(
                    input=predicted_labels,
                    target=torch.cat([real_labels, fake_labels])
                )

                loss = loss_real + loss_fake + label_loss
                loss.backward()
                cd_optimizer.step()

            for __ in range(src.config.gan_config.g_loop_num):
                loss: torch.Tensor = 0
                hidden_statuses = []

                for label in self.samples.keys():

                    self.generators[label].zero_grad()
                    fake_x = self.generators[label].make_samples(src.config.gan_config.batch_size)
                    prediction = self.cd.d_forward(fake_x)
                    loss += - prediction.mean()

                    predicted_labels = self.cd.c_forward(fake_x)
                    real_labels = torch.tensor(
                        data=[label for ___ in range(src.config.gan_config.batch_size)],
                        device=src.config.device
                    )
                    loss += cross_entropy(
                        input=predicted_labels,
                        target=real_labels,
                    )
                    hidden_statuses.append(self.cd.hidden_status.flatten())

                # for i in range(len(self.generators)):
                #     for j in range(len(self.generators)):
                #         if i == j:
                #             continue
                #         else:
                #             loss += torch.dot(hidden_statuses[i], hidden_statuses[j])
                loss.backward()
                for i in g_optimizers:
                    i.step()

        for label in self.samples.keys():
            self.generators[label].eval()
        self.logger.info('Finished training TMG-GAN')

    def _divide_samples(self, dataset: src.datasets.TrDataset) -> None:
        for feature, label in dataset:
            label = label.item()
            if label not in self.samples.keys():
                self.samples[label] = feature.unsqueeze(0)
            else:
                self.samples[label] = torch.cat([self.samples[label], feature.unsqueeze(0)])

    def _get_target_samples(self, label: int, num: int) -> torch.Tensor:
        return torch.stack(
            random.choices(
                self.samples[label],
                k=num,
            )
        )

    def make_samples(self, label: int, num: int):
        return self.generators[label].make_samples(num)
