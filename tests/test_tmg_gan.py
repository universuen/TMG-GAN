import torch

import context

import src

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if __name__ == '__main__':
    # src.utils.turn_on_test_mode()
    src.utils.prepare_datasets()
    src.utils.set_random_state()
    tmg_gan = src.TMGGAN(src.datasets.label_num, 30)
    tmg_gan.fit(src.datasets.TrDataset())

    x = tmg_gan.dbn.extract(src.datasets.TrDataset().features).cpu()
    y = src.datasets.TrDataset().labels.cpu()

    x = torch.cat([x, tmg_gan.make_samples(0, 100).cpu().detach()])
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
