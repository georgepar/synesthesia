from functools import partial

import numpy as np
from tensorflow.examples.tutorials import mnist
from tensorflow.python.framework import dtypes

import corrnet.data.dataset

MNIST = mnist.input_data.read_data_sets('MNIST_data', one_hot=True)


def image_half(shape, which_half, img):
    if img.shape != shape:
        img = np.reshape(img, shape)
    half_columns = shape[1] // 2
    if which_half == 'left':
        half_img = img[:, :half_columns]
    elif which_half == 'right':
        half_img = img[:, half_columns:]
    else:
        return None
    return np.reshape(half_img, (shape[0] * half_columns))


mnist_img_left_half = partial(image_half, (28, 28), 'left')
mnist_img_right_half = partial(image_half, (28, 28), 'right')
map_mnist_take_left_half = partial(
    np.apply_along_axis, mnist_img_left_half, 1)
map_mnist_take_right_half = partial(
    np.apply_along_axis, mnist_img_right_half, 1)


def create_view_dataset(images, labels, take_half_func):
    halves = take_half_func(images)
    options = dict(dtype=dtypes.float32, seed=None)
    dataset = corrnet.data.dataset.ImageDataSet(halves, labels, **options)
    return dataset


def left_view_dataset():
    train_dataset = create_view_dataset(
        MNIST.train.images,
        MNIST.train.labels,
        map_mnist_take_left_half)
    validation_dataset = create_view_dataset(
        MNIST.validation.images,
        MNIST.validation.labels,
        map_mnist_take_left_half)
    test_dataset = create_view_dataset(
        MNIST.test.images,
        MNIST.test.labels,
        map_mnist_take_left_half)

    return corrnet.data.dataset.Datasets(
        train=train_dataset,
        validation=validation_dataset,
        test=test_dataset)


def right_view_dataset():
    train_dataset = create_view_dataset(
        MNIST.train.images,
        MNIST.train.labels,
        map_mnist_take_right_half)
    validation_dataset = create_view_dataset(
        MNIST.validation.images,
        MNIST.validation.labels,
        map_mnist_take_right_half)
    test_dataset = create_view_dataset(
        MNIST.test.images,
        MNIST.test.labels,
        map_mnist_take_right_half)

    return corrnet.data.dataset.Datasets(
        train=train_dataset,
        validation=validation_dataset,
        test=test_dataset)
