from __future__ import division
import tensorflow as tf
import toolz


def get_num_iterations(num_samples, batch_size):
    return int(num_samples / batch_size)


def square_sum(x):
    return tf.reduce_sum(tf.square(x))


def l2_error(x1, x2):
    return square_sum(tf.subtract(x1, x2))


def compose_layers(layers):
    # Compose apply functions for the list of layers
    # reversing the layer list to apply in the correct order
    if len(layers) == 0:
        return lambda x: x
    elif len(layers) == 1:
        return layers[0]
    else:
        return toolz.functoolz.compose(*reversed([l for l in layers]))
