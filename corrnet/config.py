import os


DEBUG = True


class paths(object):
    base = '/home/geopar/projects/corrnet/'
    mnist = os.path.join(base, 'MNIST_data')
    checkpoint = os.path.join(base, 'my_model_final.ckpt')
    zeroshot = os.path.join(base, 'xlsa17/data')


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


class _zsl_dataset(object):
    def __init__(self, path):
        self.path = path
        self.classes = os.path.join(path, 'allclasses.txt')
        self.att_splits = os.path.join(path, 'att_splits.mat')
        self.binary_att_splits = os.path.join(path, 'binaryAtt_splits.mat')
        self.res101 = os.path.join(path, 'res101.mat')
        self.testclasses = os.path.join(path, 'testclasses.txt')
        self.trainclasses = [
            os.path.join(path, 'trainclasses1.txt'),
            os.path.join(path, 'trainclasses2.txt'),
            os.path.join(path, 'trainclasses3.txt')
        ]
        self.valclasses = [
            os.path.join(path, 'valclasses1.txt'),
            os.path.join(path, 'valclasses2.txt'),
            os.path.join(path, 'valclasses3.txt')
        ]
        self.trainvalclasses = os.path.join(path, 'trainvalclasses.txt')
        self.class_embeddings = os.path.join(path, 'classes.en.vec')


class _awa1(_zsl_dataset):
    def __init__(self):
        self.path = os.path.join(paths.zeroshot, 'AWA1')
        super(_awa1, self).__init__(self.path)


class _awa2(_zsl_dataset):
    def __init__(self):
        self.path = os.path.join(paths.zeroshot, 'AWA2')
        super(_awa2, self).__init__(self.path)


class _apy(_zsl_dataset):
    def __init__(self):
        self.path = os.path.join(paths.zeroshot, 'APY')
        super(_apy, self).__init__(self.path)


class _cub(_zsl_dataset):
    def __init__(self):
        self.path = os.path.join(paths.zeroshot, 'CUB')
        super(_cub, self).__init__(self.path)


class _sun(_zsl_dataset):
    def __init__(self):
        self.path = os.path.join(paths.zeroshot, 'SUN')
        super(_sun, self).__init__(self.path)


class zeroshot(object):
    ds = {
        'apy': _apy(),
        'awa1': _awa1(),
        'awa2': _awa2(),
        'cub': _cub(),
        'sun': _sun()
    }