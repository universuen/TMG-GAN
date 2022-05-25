import context

from src import datasets


if __name__ == '__main__':
    tr_dataset = datasets.TrDataset()
    print(len(tr_dataset), len(tr_dataset.features), len(tr_dataset.labels))
    print(tr_dataset.features[0])
    print(type(tr_dataset.features[0]))
    print(tr_dataset.labels[0])
    print(type(tr_dataset.labels[0]))

    te_dataset = datasets.TeDataset()
    print(len(te_dataset), len(te_dataset.features), len(te_dataset.labels))
    print(te_dataset.features[0])
    print(type(te_dataset.features[0]))
    print(te_dataset.labels[0])
    print(type(te_dataset.labels[0]))
