import os


class paths(object):
    base = '/home/geopar/PycharmProjects/corrnets/'
    mnist = os.path.join(base, 'MNIST_data')
    checkpoint = os.path.join(base, 'my_model_final.ckpt')


class net(object):
    n_inputs = 28 * 14

    class encoder_layers(object):
        # first layer has 500 hidden units second 300
        hidden_units = [500, 300]

    class common_layer(object):
        hidden_units = 50

    class decoder_layers(object):
        hidden_units = [300, 500]


class train(object):
    learning_rate = 0.01
    n_epochs = 40
    batch_size = 50
