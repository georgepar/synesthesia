import numpy as np
import scipy.io
from tensorflow.python.framework import dtypes

import corrnet.config as corrnet_cfg
import corrnet.data.dataset


def parse_image_embeddings(dataset):
    ds = corrnet_cfg.zeroshot.ds[dataset]
    res101 = scipy.io.loadmat(ds.res101)
    res101_features = res101['features'].transpose()
    res101_labels = res101['labels'].flatten()
    return res101_features, res101_labels


def parse_class_embeddings(dataset):
    ds = corrnet_cfg.zeroshot.ds[dataset]
    with open(ds.class_embeddings) as f:
        lines = [l.strip().split(' ') for l in f.readlines()]
        class_embeddings = {
            l[0]: np.array([float(n) for n in l[1:]]) for l in lines}
    return class_embeddings


def embeddings_for_labels(dataset, class_embeddings, labels):
    ds = corrnet_cfg.zeroshot.ds[dataset]
    all_classes = scipy.io.loadmat(ds.att_splits)['allclasses_names']
    classes = {i+1: c[0] for i, c in enumerate(all_classes.flatten().tolist())}
    label_embeddings = np.array([class_embeddings[classes[l]] for l in labels])
    return label_embeddings


def parse_label_embeddings(dataset, labels):
    class_embeddings = parse_class_embeddings(dataset)
    label_embeddings = embeddings_for_labels(dataset,
                                             class_embeddings,
                                             labels)
    return label_embeddings


def get_split_indexes(dataset, sanitize_trainval=True):
    ds = corrnet_cfg.zeroshot.ds[dataset]
    attr_splits = scipy.io.loadmat(ds.att_splits)
    test_seen = attr_splits['test_seen_loc'].flatten() - 1
    test_unseen = attr_splits['test_unseen_loc'].flatten() - 1
    train = attr_splits['train_loc'].flatten() - 1
    val = attr_splits['val_loc'].flatten() - 1
    if sanitize_trainval:
        train = train[~np.isin(train, test_seen)]
        val = val[~np.isin(val, test_seen)]
    # if corrnet_cfg.DEBUG and sanitize_trainval:
    #     trainval = attr_splits['trainval_loc'].flatten() - 1
    #     assert np.all(
    #         np.sort(trainval) == np.sort(np.concatenate(train, val))), (
    #         'Trainval split does not conform with train and val splits')
    return train, val, test_seen, test_unseen


def create_split_dataset(embeddings, labels, split_idx):
    options = dict(dtype=dtypes.float32, seed=None)
    split_embeddings = embeddings[split_idx, :]
    split_labels = labels[split_idx]
    dataset = corrnet.data.dataset.EmbeddingsDataSet(
        split_embeddings,
        split_labels,
        **options)
    return dataset


def create_embeddings_dataset(dataset, embeddings, labels):
    train_idx, val_idx, test_seen_idx, test_unseen_idx = get_split_indexes(
        dataset, sanitize_trainval=True)

    train_dataset = create_split_dataset(embeddings, labels, train_idx)
    validation_dataset = create_split_dataset(embeddings, labels, val_idx)
    test_seen_dataset = create_split_dataset(embeddings, labels, test_seen_idx)
    test_unseen_dataset = create_split_dataset(embeddings,
                                               labels,
                                               test_unseen_idx)

    return corrnet.data.dataset.ZSLDatasets(
        train=train_dataset,
        validation=validation_dataset,
        test_seen=test_seen_dataset,
        test_unseen=test_unseen_dataset)
