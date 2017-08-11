import tensorflow as tf

from corrnet.utils import square_sum, l2_error, compose_layers


def correlation(h1, h2, lamda=0.02):
    """
    Calculates the correlation between the common hidden layer outputs wrt each
    input
    Args:
        h1: Common hidden layer output wrt input1
        h2: Common hidden layer output wrt input2
        lamda: parameter to scale correlation in the loss function
    Returns:
        The correlation between hidden_wrt_input_1, hidden_wrt_input2 scaled by
        lamda
        - lamda * corr(h(X1), h(X2))
    """
    h1_mean = tf.reduce_mean(h1)
    h2_mean = tf.reduce_mean(h2)
    h1_centered = tf.subtract(h1, h1_mean)
    h2_centered = tf.subtract(h2, h2_mean)
    corr_nr = tf.reduce_sum(tf.multiply(h1_centered, h2_centered))
    corr_dr1 = tf.sqrt(square_sum(h1_centered))
    corr_dr2 = tf.sqrt(square_sum(h2_centered))
    corr_dr = tf.add(tf.multiply(corr_dr1, corr_dr2), 1e-6)
    corr = tf.multiply(tf.div(corr_nr, corr_dr), -lamda)
    return corr


def calculate_loss(input_layers1,
                   input_layers2,
                   reconstruction_layers1,
                   reconstruction_layers2,
                   common_layer,
                   inputs,
                   loss_terms,
                   lamda=0.02):
    """
    Calculate the loss function given the network layers and the inputs.
    Currently supports 2 inputs

Graph 1:
input1 -> in1_layer1 -> ... -> in1_layerN     rec1_layer1 -> ... -> rec1_layerN
                                      \      /                          \
                                      common=h                        concat=g
                                      /      \                          /
input2 -> in2_layer1 -> ... -> in2_layerN     rec2_layer1 -> ... -> rec2_layerN

    Args:
        input_layers1: list of the input layers for input1,
            i.e. [in1_layer1, in1_layer2, ..., in1_layerN], see Graph1
        input_layers2: list of the input layers for input2,
            i.e. [in2_layer1, in2_layer2, ..., in2_layerN], see Graph1
        reconstruction_layers1: list of the reconstruction layers for input1,
            i.e. [rec1_layer1, rec1_layer2, ..., rec1_layerN], see Graph1
        reconstruction_layers2: list of the reconstruction layers for input2,
            i.e. [rec2_layer1, rec2_layer2, ..., rec2_layerN], see Graph1
        common_layer: the common layer, i.e common from Graph 1
        inputs: the list of inputs, i.e. [input1, input2] from Graph 1
        loss_terms: list of strings: The loss terms to be incorporated in the
            loss function. Can contain:
            l1 -> The total reconstruction loss given both inputs
            l2 -> The reconstruction loss wrt input1
            l3 -> The reconstruction loss wrt input2
            l4 -> the correlation btw the hidden layer outputs wrt input1 and
                input2 scaled by lamda
        lamda: Scaling coefficient for the correlation loss term
    Returns:
        loss = l1 + l2 + l3 - l4
        The terms are conditioned on the specified loss terms. E.g. if
        loss_terms = ['l1', 'l2', 'l3'] then
        loss = l1 + l2 + l3
    """
    if len(loss_terms) == 0:
        raise ValueError("Must pass at least one loss term of l1, l2, l3, l4")
    input1_subnet = compose_layers(input_layers1)
    input2_subnet = compose_layers(input_layers2)
    rec1_subnet = compose_layers(reconstruction_layers1)
    rec2_subnet = compose_layers(reconstruction_layers2)

    def h(in_list): return common_layer([input1_subnet(in_list[0]),
                                         input2_subnet(in_list[1])])

    def g(common_out): return tf.concat([rec1_subnet(common_out),
                                         rec2_subnet(common_out)], 1)

    concat_inputs = tf.concat(inputs, 1)
    losses = []

    h1_corr = None
    h2_corr = None

    if 'l1' in loss_terms:
        losses.append(l2_error(concat_inputs, g(h(inputs))))
    if 'l2' in loss_terms:
        x1_zero_input = [inputs[0], tf.zeros_like(inputs[1])]
        h1 = h(x1_zero_input)
        # Cache to reuse in correlation
        if 'l4' in loss_terms:
            h1_corr = h1
        losses.append(l2_error(concat_inputs, g(h1)))
    if 'l3' in loss_terms:
        zero_x2_input = [tf.zeros_like(inputs[0]), inputs[1]]
        h2 = h(zero_x2_input)
        # Cache to reuse in correlation
        if 'l4' in loss_terms:
            h2_corr = h2
        losses.append(l2_error(concat_inputs, g(h2)))

    if 'l4' in loss_terms:
        if h1_corr is None:
            x1_zero_input = [inputs[0], tf.zeros_like(inputs[1])]
            h1_corr = h(x1_zero_input)
        if h2_corr is None:
            zero_x2_input = [tf.zeros_like(inputs[0]), inputs[1]]
            h2_corr = h(zero_x2_input)
        losses.append(correlation(h1_corr, h2_corr, lamda=lamda))

    loss = tf.add_n(losses)
    return loss
