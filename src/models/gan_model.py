from collections import OrderedDict

import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.nn import Module
from pytorch_lightning import LightningModule
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src import config
from src.config import gan_config


class _GeneratorModel(Module):
    def __init__(self, z_size: int, num_features: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(z_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, num_features),
        )

    def forward(self, real_samples: torch.Tensor) -> torch.Tensor:
        return self.body(real_samples)


class _DiscriminatorModule(Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.main_body = nn.Sequential(
            spectral_norm(nn.Linear(num_features, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 256)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(256, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 16)),
            nn.LeakyReLU(0.2),
        )
        self.hidden_status: torch.Tensor = None
        self.out_layer = spectral_norm(nn.Linear(16, 1))

    def forward(self, real_samples: torch.Tensor) -> torch.Tensor:
        self.hidden_status = self.main_body(real_samples)
        return self.out_layer(self.hidden_status)


class GANModule(LightningModule):
    def __init__(self, num_features: int):
        super().__init__()
        self.generator = _GeneratorModel(gan_config.z_size, num_features)
        self.discriminator = _DiscriminatorModule(num_features)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(layer: nn.Module):
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, 0.0, 0.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif type(layer) == nn.BatchNorm1d:
            nn.init.normal_(layer.weight, 1.0, 0.02)
            nn.init.constant_(layer.bias, 0)

    def forward(self, real_samples: torch.Tensor, generating: bool) -> torch.Tensor:
        if generating:
            return self.generator(real_samples)
        else:
            return self.discriminator(real_samples)

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=gan_config.g_lr,
            betas=(0.5, 0.999),
        )
        d_optimizer = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=gan_config.g_lr,
            betas=(0.5, 0.999),
        )
        return d_optimizer, g_optimizer

    def training_step(
            self,
            batch: list[torch.Tensor],
            batch_idx: int,
            optimizer_idx: int,
    ) -> OrderedDict:
        real_samples, _ = batch
        z = torch.randn(real_samples.shape[0], gan_config.z_size)
        z = z.type_as(real_samples)

        # train D
        if optimizer_idx == 0:
            score_real = self.forward(real_samples, generating=False).mean()
            generated_samples = self.forward(z, generating=True)
            score_generated = self.forward(generated_samples, generating=False).mean()
            d_loss = (score_generated - score_real) / 2
            tqdm_dict = {"d_loss": d_loss}
            return OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})

        # train G
        if optimizer_idx == 1:
            generated_samples = self.forward(z, generating=True)
            score_generated = self.forward(generated_samples, generating=False).mean()
            g_loss = -score_generated
            tqdm_dict = {"g_loss": g_loss}
            return OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})

    def test_step(
            self,
            batch: list[torch.Tensor],
            batch_idx: int
    ):
        real_samples, labels = batch
        z = torch.randn(len(real_samples), gan_config.z_size)
        z = z.type_as(real_samples)
        generated_samples = self.forward(z, generating=True).detach()
        real_samples = real_samples.cpu()
        generated_samples = generated_samples.cpu()
        labels = labels.cpu()
        real_samples = torch.cat([real_samples, generated_samples])
        labels = torch.cat([labels, torch.full([len(generated_samples)], -1)])

        embedded_x = TSNE(
            learning_rate='auto',
            init='random',
            random_state=config.seed,
        ).fit_transform(real_samples)
        sns.scatterplot(
            x=embedded_x[:, 0],
            y=embedded_x[:, 1],
            hue=labels,
            palette="deep"
        )
        plt.show()
