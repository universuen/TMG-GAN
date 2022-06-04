import context

from src import Classifier, datasets, utils


if __name__ == '__main__':
    utils.prepare_datasets()
    utils.set_random_state()
    utils.turn_on_test_mode()
    clf = Classifier()
    clf.test(datasets.TeDataset())
    print(clf.metrics)
    clf.fit(datasets.TrDataset())
    clf.test(datasets.TeDataset())
    print(clf.metrics)
