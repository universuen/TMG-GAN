import context

import torch

import src

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if __name__ == '__main__':
    # src.utils.turn_on_test_mode()
    src.utils.prepare_datasets()
    src.utils.set_random_state()
    tmg_gan = src.TMGGAN()
    tmg_gan.fit(src.datasets.TrDataset())

    x = src.datasets.TrDataset().samples.cpu()
    y = src.datasets.TrDataset().labels.cpu()

    x = torch.cat([x, tmg_gan.generate_samples(100)])
    y = torch.cat([y, torch.full([100], -1)])
    embedded_x = TSNE(
        learning_rate='auto',
        init='random',
        random_state=src.config.seed,
    ).fit_transform(x)
    sns.scatterplot(
        x=embedded_x[:, 0],
        y=embedded_x[:, 1],
        hue=y,
        palette="deep"
    )
    plt.show()
