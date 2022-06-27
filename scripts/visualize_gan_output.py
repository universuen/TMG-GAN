import context

import torch
from matplotlib import pyplot as plt


import src
from src import utils

GAN = src.TMGGAN


if __name__ == '__main__':
    # utils.set_random_state()
    # utils.turn_on_test_mode()

    gan = GAN()
    gan.fit(src.datasets.TrDataset())
    images = torch.cat([gan.generate_samples(i, 10) for i in range(10)])
    _, axs = plt.subplots(10, 10)
    for i in range(10):
        for j in range(10):
            axs[i, j].imshow(images[i * 10 + j], cmap='gray')
            axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig('gan_output.jpg')
    plt.show()
