import context

import pickle

import torch

import src
from src import Classifier, datasets, utils

dataset = 'KDDCUP99'
# dataset = 'NSL-KDD'

if __name__ == '__main__':
    # utils.turn_on_test_mode()

    utils.set_random_state()
    utils.prepare_datasets(dataset)
    gan = src.SNGAN()
    gan.fit(src.datasets.TrDataset())
    # count the max number of samples
    max_cnt = max([len(gan.samples[i]) for i in range(datasets.label_num)])
    # generate samples
    for i in range(datasets.label_num):
        cnt_generated = max_cnt - len(gan.samples[i])
        if cnt_generated > 0:
            generated_samples = gan.generate_samples(i, cnt_generated)
            generated_labels = torch.full([cnt_generated], i)
            datasets.tr_samples = torch.cat([datasets.tr_samples, generated_samples])
            datasets.tr_labels = torch.cat([datasets.tr_labels, generated_labels])

    utils.set_random_state()
    clf = Classifier('SNGAN')
    clf.fit(datasets.TrDataset())
    torch.cuda.empty_cache()
    clf.test(datasets.TeDataset())
    print(clf.confusion_matrix)
    print(clf.metrics)
