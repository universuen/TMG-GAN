import context

from src import Classifier, utils

if __name__ == '__main__':
    utils.prepare_datasets()
    utils.set_random_state()
    utils.turn_on_test_mode()
    clf = Classifier(20, 5)

    clf.test()
    clf.fit()
    clf.test()
