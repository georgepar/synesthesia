import tensorflow as tf


def leaky_relu(x, alpha=0.05, max_value=None):
    """ReLU.

    alpha: slope of negative section.
    """
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=x.dtype.base_dtype),
                             tf.cast(max_value, dtype=x.dtype.base_dtype))
    x -= tf.constant(alpha, dtype=x.dtype.base_dtype) * negative_part
    return x
