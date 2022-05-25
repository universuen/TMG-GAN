import context

from src import Classifier, datasets


if __name__ == '__main__':
    clf = Classifier('test')
    clf.test(datasets.TeDataset())
    print(clf.metrics)
    clf.fit(datasets.TrDataset())
    clf.test(datasets.TeDataset())
    print(clf.metrics)
