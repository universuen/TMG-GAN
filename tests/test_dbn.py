from src import datasets, utils, PartialDBN, Classifier


if __name__ == '__main__':
    utils.prepare_datasets()

    utils.set_random_state()
    tr_dataset = datasets.TrDataset()
    te_dataset = datasets.TeDataset()
    c = Classifier()
    c.fit(tr_dataset)
    c.test(te_dataset)
    print(c.metrics)

    utils.set_random_state()
    dbn = PartialDBN(datasets.feature_num, 50)
    dbn.fit(tr_dataset)
    tr_dataset.features = dbn.extract(tr_dataset.features)
    te_dataset.features = dbn.extract(te_dataset.features)
    c = Classifier()
    c.fit(tr_dataset)
    c.test(te_dataset)
    print(c.metrics)


