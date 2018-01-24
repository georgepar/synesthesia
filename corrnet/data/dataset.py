from __future__ import absolute_import
from __future__ import division

import collections
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed


ZSLDatasets = collections.namedtuple('ZSLDatasets', ['train',
                                                     'validation',
                                                     'test_seen',
                                                     'test_unseen'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class Dataset(object):
    def __init__(self,
                 data,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed
        # is returned
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid dtype {}, expected uint8 or '
                            'float32'.format(dtype))
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert data.shape[0] == labels.shape[0], (
                'data.shape: {} labels.shape: {}'
                .format(data.shape, labels.shape))
            self._num_examples = data.shape[0]

        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_embedding = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return ([fake_embedding for _ in xrange(batch_size)],
                    [fake_label for _ in xrange(batch_size)])
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._data = self.data[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data = self.data[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            labels_new_part = self._labels[start:end]
            concat_imgs = np.concatenate(
                (data_rest_part, data_new_part), axis=0)
            concat_labels = np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
            return concat_imgs, concat_labels
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._labels[start:end]


class EmbeddingsDataSet(Dataset):
    def __init__(self,
                 embeddings,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 seed=None):
        """Construct an Embeddings DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        super(EmbeddingsDataSet, self).__init__(
            embeddings,
            labels,
            fake_data=fake_data,
            one_hot=one_hot,
            dtype=dtype,
            seed=seed)

    @property
    def embeddings(self):
        return self._data


class ImageDataSet(Dataset):
    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 seed=None):
        """Construct an Image DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        if not fake_data and dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)

        super(ImageDataSet, self).__init__(
            images,
            labels,
            fake_data=fake_data,
            one_hot=one_hot,
            dtype=dtype,
            seed=seed)

    @property
    def images(self):
        return self._data
