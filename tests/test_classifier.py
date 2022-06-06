import context

from src import Classifier, datasets, utils


if __name__ == '__main__':
    # utils.turn_on_test_mode()
    utils.prepare_datasets()
    utils.set_random_state()
    clf = Classifier('test')
    clf.test(datasets.TeDataset())
    print(clf.metrics)
    clf.fit(datasets.TrDataset())
    clf.test(datasets.TeDataset())
    print(clf.metrics)
