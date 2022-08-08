

import torch

import src

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_data(gan):
    x = src.datasets.TrDataset().samples.cpu()
    y = src.datasets.TrDataset().labels.cpu()
    y = torch.zeros_like(y)

    for i in range(src.datasets.label_num):
        x = torch.cat([x, gan.generate_samples(i, len(gan.samples[i]) // 3)])
        y = torch.cat([y, torch.full([len(gan.samples[i]) // 3], -1)])

    embedded_x = TSNE(
        learning_rate='auto',
        init='random',
        random_state=src.config.seed,
    ).fit_transform(x)
    return embedded_x, y


if __name__ == '__main__':
    src.utils.prepare_datasets()
    # src.utils.turn_on_test_mode()
    # src.utils.set_random_state()
    src.config.gan_config.epochs = 500
    gans = [src.GAN(), src.WGAN(), src.SNGAN(), src.TMGGAN()]
    _, axes = plt.subplots(4, 4, figsize=(20, 20))
    for i, gan in enumerate(gans):
        current_epoch = 0
        for j in range(4):
            gan.fit(src.datasets.TrDataset())
            current_epoch += src.config.gan_config.epochs
            x, y = get_data(gan)
            colors = []
            for k in y:
                colors.append('tab:orange' if k == -1 else 'tab:blue')
            axes[i][j].scatter(
                x[:, 0],
                x[:, 1],
                c=colors,
            )
            if j == 0:
                axes[i][j].set_ylabel(gan.name, fontsize=18)
            if i == 3:
                axes[i][j].set_xlabel(f'{current_epoch} epochs', fontsize=18)
            # axes[i][j].get_legend().
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
            plt.savefig('gan.jpg')
    # plt.show()
