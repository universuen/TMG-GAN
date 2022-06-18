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

    # select features
    lens = (len(datasets.tr_samples), len(datasets.te_samples))
    samples = torch.cat(
        [
            datasets.tr_samples,
            datasets.te_samples,
        ]
    )
    labels = torch.cat(
        [
            datasets.tr_labels,
            datasets.te_labels,
        ]
    )
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import minmax_scale

    pca = PCA(n_components=25)
    samples = torch.from_numpy(
        minmax_scale(
            pca.fit_transform(samples, labels)
        )
    ).float()
    samples = (samples - samples.min())
    datasets.tr_samples, datasets.te_samples = torch.split(samples, lens)
    utils.set_dataset_values()
    print(datasets.feature_num)

    src.utils.set_random_state()
    tmg_gan = src.TMGGAN()
    tmg_gan.fit(src.datasets.TrDataset())
    # count the max number of samples
    max_cnt = max([len(tmg_gan.samples[i]) for i in range(datasets.label_num)])
    # generate samples
    for i in range(datasets.label_num):
        cnt_generated = max_cnt - len(tmg_gan.samples[i])
        if cnt_generated > 0:
            generated_samples = tmg_gan.generate_qualified_samples(i, cnt_generated)
            generated_labels = torch.full([cnt_generated], i)
            datasets.tr_samples = torch.cat([datasets.tr_samples, generated_samples])
            datasets.tr_labels = torch.cat([datasets.tr_labels, generated_labels])

    utils.set_random_state()
    clf = Classifier('TMG_GAN')
    clf.model = tmg_gan.cd
    clf.fit(datasets.TrDataset())
    torch.cuda.empty_cache()
    clf.test(datasets.TeDataset())
    print(clf.confusion_matrix)
    print(clf.metrics)
