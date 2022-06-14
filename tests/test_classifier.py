import context

import pickle

import torch

import src
from src import Classifier, datasets, utils


if __name__ == '__main__':
    # utils.turn_on_test_mode()

    utils.set_random_state()
    utils.prepare_datasets()

    utils.set_random_state()
    clf = Classifier('test_-1')
    clf.test(datasets.TeDataset())
    print(clf.confusion_matrix)
    print(clf.metrics)

    utils.set_random_state()
    clf = Classifier('test_0')
    clf.fit(datasets.TrDataset())
    clf.test(datasets.TeDataset())
    print(clf.confusion_matrix)
    print(clf.metrics)

    src.utils.set_random_state()
    tmg_gan = src.TMGGAN()
    tmg_gan.fit(src.datasets.TrDataset())

    # count the max number of samples
    max_cnt = max([len(tmg_gan.samples[i]) for i in range(datasets.label_num)])
    # generate samples
    for i in range(datasets.label_num):
        cnt_generated = max_cnt - len(tmg_gan.samples[i])
        if cnt_generated > 0:
            generated_samples = tmg_gan.generate_samples(i, cnt_generated)
            generated_labels = torch.full([cnt_generated], i)
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

    clf = Classifier('test_1')
    clf.fit(datasets.TrDataset())
    torch.cuda.empty_cache()
    clf.test(datasets.TeDataset())
    print(clf.confusion_matrix)
    print(clf.metrics)
