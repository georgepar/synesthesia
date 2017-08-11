from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn

from tensorflow.python.layers import base

import corrnet.activations


class MultiInputDense(base.Layer):
    """Densely-connected layer class with multiple inputs.
    This layer implements the operation:
    `outputs = activation(inputs1.kernel1 + inputs2.kernel2 + bias)`
    From the definition of outputs it is evident that all inputs must have
    the same innermost dimension input.shape[-1]
    Where `activation` is the activation function passed as the `activation`
    argument (if not `None`), `kernel` is a weights matrix created by the
    layer, and `bias` is a bias vector created by the layer
    (only if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then it is
    flattened prior to the initial matrix multiply by `kernel`.
    Arguments:
        units: Integer or Long, dimensionality of the output space.
        activation: Activation function (callable). Set it to None to
            maintain a linear activation.
        use_bias: Boolean, whether the layer uses a bias.
        kernel_initializer: Initializer function for the weight matrix.
            If `None` (default), weights are initialized using the default
            initializer used by `tf.get_variable`.
        bias_initializer: Initializer function for the bias.
        kernel_regularizer: Regularizer function for the weight matrix.
        bias_regularizer: Regularizer function for the bias.
        activity_regularizer: Regularizer function for the output.
        trainable: Boolean, if `True` also add variables to the
            graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in
            such cases.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
    Properties:
        units: Python integer, dimensionality of the output space.
        activation: Activation function (callable).
        use_bias: Boolean, whether the layer uses a bias.
        kernel_initializer: Initializer instance (or name) for the weight
            matrix.
        bias_initializer: Initializer instance (or name) for the bias.
        kernel_regularizer: Regularizer instance for the weight matrix
            (callable)
        bias_regularizer: Regularizer instance for the bias (callable).
        activity_regularizer: Regularizer instance for the output (callable)
        kernel: Weight matrix (TensorFlow variable or tensor).
        bias: Bias vector, if applicable (TensorFlow variable or tensor).
    """

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(MultiInputDense, self).__init__(
            trainable=trainable, name=name, **kwargs)
        self.bias = None
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernels = []
        self.input_spec = [base.InputSpec(min_ndim=2),
                           base.InputSpec(min_ndim=2)]

    def build(self, input_shapes):
        input_shapes = [tensor_shape.TensorShape(input_shape)
                        for input_shape in input_shapes]
        for i, input_shape in enumerate(input_shapes):
            if input_shape[-1].value is None:
                raise ValueError('The last dimension of the inputs to '
                                 '`MultiInputDense` should be defined.'
                                 ' Found `None`.')
            self.input_spec[i] = base.InputSpec(
                min_ndim=2, axes={-1: input_shape[-1].value})
            self.kernels.append(self.add_variable(
                'kernel_input_{}'.format(i),
                shape=[input_shape[-1].value, self.units],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                dtype=self.dtype,
                trainable=True))
        if self.use_bias:
            self.bias = self.add_variable('bias',
                                          shape=[self.units, ],
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          dtype=self.dtype,
                                          trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs,  **kwargs):
        inputs = [ops.convert_to_tensor(input_tensor, dtype=self.dtype)
                  for input_tensor in inputs]

        outputs_wrt_input = [
            tf.matmul(input_tensor, kernel)
            for input_tensor, kernel
            in zip(inputs, self.kernels)]
        outputs = tf.add_n(outputs_wrt_input)
        if self.use_bias:
            # outputs_wrt_input = [
            #    nn.bias_add(out, self.bias)
            #    for out in outputs_wrt_input]
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            # outputs_wrt_input = [
            #    self.activation(out)
            #    for out in outputs_wrt_input]
            outputs = self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def _compute_output_shape(self, input_shapes):
        # All inputs must have the same size
        input_shape = tensor_shape.TensorShape(input_shapes[0])
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be'
                ' defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)


def multi_input_dense_layer(
        inputs, units,
        activation=corrnet.activations.leaky_relu,
        use_bias=True,
        kernel_initializer=initializers.xavier_initializer(),
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        trainable=True,
        name=None,
        reuse=None):
    """Functional interface for the multi input densely-connected layer.
    This layer implements the operation:
    `outputs = activation(inputs1.kernel1 + inputs2.kernel2 + bias)`
    Where `activation` is the activation function passed as the `activation`
    argument (if not `None`), `kernel` is a weights matrix created by the
    layer, and `bias` is a bias vector created by the layer
    (only if `use_bias` is `True`).
    Note: if the `inputs` tensor has a rank greater than 2, then it is
    flattened prior to the initial matrix multiply by `kernel`.
    Arguments:
      inputs: a list of Tensor inputs.
      units: Integer or Long, dimensionality of the output space.
      activation: Activation function (callable). Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: Initializer function for the weight matrix.
        If `None` (default), weights are initialized using the default
        initializer used by `tf.get_variable`.
      bias_initializer: Initializer function for the bias.
      kernel_regularizer: Regularizer function for the weight matrix.
      bias_regularizer: Regularizer function for the bias.
      activity_regularizer: Regularizer function for the output.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      name: String, the name of the layer.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      The created layer. Use .apply method to pass inputs
    """
    layer = MultiInputDense(
        units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name,
        dtype=inputs[0].dtype.base_dtype,
        _scope=name,
        _reuse=reuse)
    return layer  # .apply(inputs)


def dense_layer(
        inputs, units,
        activation=corrnet.activations.leaky_relu,
        use_bias=True,
        kernel_initializer=initializers.xavier_initializer(),
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        trainable=True,
        name=None,
        reuse=None):
    """Functional interface for the densely-connected layer.
    This layer implements the operation:
    `outputs = activation(inputs.kernel + bias)`
    Where `activation` is the activation function passed as the `activation`
    argument (if not `None`), `kernel` is a weights matrix created by the
    layer, and `bias` is a bias vector created by the layer
    (only if `use_bias` is `True`).
    Note: if the `inputs` tensor has a rank greater than 2, then it is
    flattened prior to the initial matrix multiply by `kernel`.
    Arguments:
      inputs: Tensor input.
      units: Integer or Long, dimensionality of the output space.
      activation: Activation function (callable). Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: Initializer function for the weight matrix.
        If `None` (default), weights are initialized using the default
        initializer used by `tf.get_variable`.
      bias_initializer: Initializer function for the bias.
      kernel_regularizer: Regularizer function for the weight matrix.
      bias_regularizer: Regularizer function for the bias.
      activity_regularizer: Regularizer function for the output.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      name: String, the name of the layer.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
        The created layer
    """
    layer = core_layers.Dense(
        units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name,
        dtype=inputs[0].dtype.base_dtype,
        _scope=name,
        _reuse=reuse)
    return layer  # .apply(inputs)


def dropout(keep_prob=0.5,
            noise_shape=None,
            name=None):
    """Returns a dropout op applied to the input.
    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the
    expected sum is unchanged.
    Args:
    inputs: The tensor to pass to the nn.dropout op.
    keep_prob: A scalar `Tensor` with the same type as x. The probability
      that each element is kept.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    is_training: A bool `Tensor` indicating whether or not the model
      is in training mode. If so, dropout is applied and values scaled.
      Otherwise, inputs is returned.
    outputs_collections: Collection to add the outputs.
    scope: Optional scope for name_scope.
    Returns:
    A tensor representing the output of the operation.
    """
    layer = core_layers.Dropout(rate=1 - keep_prob,
                                noise_shape=noise_shape,
                                name=name,
                                _scope=name)
    return layer
