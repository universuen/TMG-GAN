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
    utils.set_random_state()
    clf = Classifier('test_0')
    clf.fit(datasets.TrDataset())
    clf.test(datasets.TeDataset())
    print(clf.confusion_matrix)
    print(clf.metrics)

