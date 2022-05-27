from src import datasets, utils, PartialDBN


if __name__ == '__main__':
    utils.prepare_datasets()
    utils.set_random_state()
    dbn = PartialDBN(datasets.feature_num, 30)
    dbn.fit(datasets.TrDataset())
    x = dbn.extract(datasets.TeDataset().features)
    print(x.shape)

