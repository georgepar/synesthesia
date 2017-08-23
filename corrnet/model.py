import tensorflow as tf
import tensorflow.contrib.keras as keras

import corrnet.activations
import corrnet.layers

from corrnet.utils import compose_layers


def build_common_layer(inputs, hdim):
    with tf.name_scope('common_layer'):
        h1 = corrnet.layers.dense_layer(
                inputs[0],
                hdim,
                name='common_{}'.format('input1'),
                activation=corrnet.activations.leaky_relu)
        h2 = corrnet.layers.dense_layer(
                inputs[1],
                hdim,
                name='common_{}'.format('input2'),
                activation=corrnet.activations.leaky_relu)
        h = keras.layers.Add(name='common_add_layer')
        h_o = h([h1, h2])
        return h, h_o


def build_encoder_for_input(nn_input, input_name, encoder_hidden_units):
    with tf.name_scope('encoder_{}'.format(input_name)):
        encoder_layers = []
        for i, hidden_units in enumerate(encoder_hidden_units):
            layer = corrnet.layers.dense_layer(
                nn_input,
                hidden_units,
                name='h{}_{}'.format(i, input_name),
                activation=corrnet.activations.leaky_relu)
            nn_input = layer(nn_input)
            encoder_layers.append(layer)
        encoder = compose_layers(encoder_layers)
        return encoder


def build_encoder(inputs, hidden_units):
    encoder1 = build_encoder_for_input(inputs[0], 'input1', hidden_units)
    encoder2 = build_encoder_for_input(inputs[1], 'input2', hidden_units)
    return encoder1, encoder2


def build_decoder_for_input(hidden, input_name, decoder_hidden_units):
    with tf.name_scope('decoder_{}'.format(input_name)):
        decoder_layers = []
        for i, hidden_units in enumerate(decoder_hidden_units):
            layer = corrnet.layers.dense_layer(
                hidden,
                hidden_units,
                name='r{}_{}'.format(i, input_name),
                activation=corrnet.activations.leaky_relu)
            hidden = layer(hidden)
            decoder_layers.append(layer)
        decoder = compose_layers(decoder_layers)
        return decoder


def build_decoder(common_representation, hidden_units):
    decoder1 = build_decoder_for_input(
        common_representation, 'input1', hidden_units)
    decoder2 = build_decoder_for_input(
        common_representation, 'input2', hidden_units)
    return decoder1, decoder2
