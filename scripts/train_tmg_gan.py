import context

import pickle

import torch

import src
from src import Classifier, datasets, utils

dataset = 'CICIDS2017'
# dataset = 'KDDCUP99'
# dataset = 'NSL-KDD'

if __name__ == '__main__':
    utils.set_random_state()
    utils.prepare_datasets(dataset)
    utils.turn_on_test_mode()

    src.utils.set_random_state()
    tmg_gan = src.TMGGAN('Linear')
    tmg_gan.fit(src.datasets.TrDataset())
    # generate
    cnt = [len(tmg_gan.samples[i]) for i in range(datasets.label_num)]
    print(cnt)
    delta = cnt[0] - sum(cnt[1:])
    if delta <= 0:
        print('no need to generate')
    else:
        num = delta // (datasets.label_num - 1)
        for i in range(1, datasets.label_num):
            generated_samples = tmg_gan.generate_qualified_samples(i, num)
            generated_labels = torch.full([num], i)
            datasets.tr_samples = torch.cat([datasets.tr_samples, generated_samples])
            datasets.tr_labels = torch.cat([datasets.tr_labels, generated_labels])

    with open('data.pkl', 'wb') as f:
        pickle.dump(
            (
                datasets.tr_samples.numpy(),
                datasets.tr_labels.numpy(),
                datasets.te_samples.numpy(),
                datasets.te_labels.numpy(),
            ),
            f,
        )
