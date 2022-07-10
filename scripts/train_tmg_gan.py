import context

import pickle

import torch
from sklearn.metrics import classification_report

import src
from src import datasets, utils

dataset = 'CICIDS2017'
# dataset = 'KDDCUP99'
# dataset = 'NSL-KDD'

ratio: float = 1

if __name__ == '__main__':
    utils.set_random_state()
    utils.prepare_datasets(dataset)

    num_4 = int(1476 * ratio)
    cnt_4 = 0
    new_tr_samples = []
    new_tr_labels = []
    for i, j in zip(datasets.tr_samples, datasets.tr_labels):
        if j == 0:
            new_tr_samples.append(i)
            new_tr_labels.append(j)
        elif j == 4:
            if cnt_4 <= num_4:
                new_tr_samples.append(i)
                new_tr_labels.append(torch.ones_like(j))
                cnt_4 += 1
    datasets.tr_samples = torch.stack(new_tr_samples)
    datasets.tr_labels = torch.stack(new_tr_labels)

    num_4 = int(706 * ratio)
    cnt_4 = 0
    new_te_samples = []
    new_te_labels = []
    for i, j in zip(datasets.te_samples, datasets.te_labels):
        if j == 0:
            new_te_samples.append(i)
            new_te_labels.append(j)
        elif j == 4:
            if cnt_4 <= num_4:
                new_te_samples.append(i)
                new_te_labels.append(torch.ones_like(j))
                cnt_4 += 1
    datasets.te_samples = torch.stack(new_te_samples)
    datasets.te_labels = torch.stack(new_te_labels)

    datasets.label_num = 2

    # utils.turn_on_test_mode()
    src.utils.set_random_state()
    tmg_gan = src.TMGGAN()
    tmg_gan.fit(src.datasets.TrDataset())
    # generate
    cnt = [len(tmg_gan.samples[i]) for i in range(datasets.label_num)]
    print(cnt)
    delta = cnt[0] - sum(cnt[1:])
    if delta <= 0:
        print('no need to generate')
    else:
        num = delta // (datasets.label_num - 1)
        for i in range(1, datasets.label_num):
            generated_samples = tmg_gan.generate_qualified_samples(i, num)
            generated_labels = torch.full([num], i)
            datasets.tr_samples = torch.cat([datasets.tr_samples, generated_samples])
            datasets.tr_labels = torch.cat([datasets.tr_labels, generated_labels])

    with open(f'data_{ratio}.pkl', 'wb') as f:
        pickle.dump(
            (
                datasets.tr_samples.numpy(),
                datasets.tr_labels.numpy(),
                datasets.te_samples.numpy(),
                datasets.te_labels.numpy(),
            ),
            f,
        )
