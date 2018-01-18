import chainer
import math


class GeneratorResBlock(chainer.Chain):
    def __init__(self):
        super().__init__(
            c1=chainer.links.Convolution2D(64, 64, ksize=3, stride=1, pad=1, initialW=chainer.initializers.Normal((0.02 * math.sqrt(64 * 3 * 3)) / math.sqrt(64))),
            bn1=chainer.links.BatchNormalization(64),
            c2=chainer.links.Convolution2D(64, 64, ksize=3, stride=1, pad=1, initialW=chainer.initializers.Normal((0.02 * math.sqrt(64 * 3 * 3)) / math.sqrt(64))),
            bn2=chainer.links.BatchNormalization(64),
        )

    def __call__(self, x: chainer.Variable):
        with chainer.using_config("test", False):
            h = chainer.functions.relu(self.bn1(self.c1(x)))
        h = self.bn2(self.c2(h))
        return h + x  # residual


class Generator(chainer.Chain):
    def __init__(self):
        super().__init__(
            first=chainer.links.Convolution2D(3, 64, ksize=3, stride=1, pad=1, initialW=chainer.initializers.Normal((0.02 * math.sqrt(3 * 3 * 3)) / math.sqrt(3))),
            res1=GeneratorResBlock(),
            res2=GeneratorResBlock(),
            res3=GeneratorResBlock(),
            res4=GeneratorResBlock(),
            res5=GeneratorResBlock(),
            conv_mid=chainer.links.Convolution2D(64, 64, ksize=3, stride=1, pad=1, initialW=chainer.initializers.Normal((0.02 * math.sqrt(64 * 3 * 3)) / math.sqrt(64))),
            bn_mid=chainer.links.BatchNormalization(64),
            conv_output=chainer.links.Convolution2D(64, 3, ksize=3, stride=1, pad=1, initialW=chainer.initializers.Normal((0.02 * math.sqrt(64 * 3 * 3)) / math.sqrt(64)))
        )

    def __call__(self, x: chainer.Variable):
        h = self.first(x)
        h = first = chainer.functions.relu(h)

        with chainer.using_config("test", False):
            h = self.res1(h)
            h = self.res2(h)
            h = self.res3(h)
            h = self.res4(h)
            h = self.res5(h)
            mid = self.bn_mid(self.conv_mid(h))

        h = first + mid

        h = self.conv_output(h)
        return h


class Discriminator(chainer.Chain):
    def __init__(self):
        super().__init__(
            conv_input=chainer.links.Convolution2D(3, 64, ksize=3, stride=1, pad=0, initialW=chainer.initializers.Normal((0.02 * math.sqrt(3 * 3 * 3)) / math.sqrt(3))),
            c1=chainer.links.Convolution2D(64, 64, ksize=3, stride=2, pad=0, initialW=chainer.initializers.Normal((0.02 * math.sqrt(64 * 3 * 3)) / math.sqrt(64))),
            bn1=chainer.links.BatchNormalization(64),
            c2=chainer.links.Convolution2D(64, 128, ksize=3, stride=1, pad=0, initialW=chainer.initializers.Normal((0.02 * math.sqrt(128 * 3 * 3)) / math.sqrt(64))),
            bn2=chainer.links.BatchNormalization(128),
            c3=chainer.links.Convolution2D(128, 128, ksize=3, stride=2, pad=0, initialW=chainer.initializers.Normal((0.02 * math.sqrt(128 * 3 * 3)) / math.sqrt(128))),
            bn3=chainer.links.BatchNormalization(128),
            c4=chainer.links.Convolution2D(128, 256, ksize=3, stride=1, pad=0, initialW=chainer.initializers.Normal((0.02 * math.sqrt(128 * 3 * 3)) / math.sqrt(128))),
            bn4=chainer.links.BatchNormalization(256),
            c5=chainer.links.Convolution2D(256, 256, ksize=3, stride=2, pad=0, initialW=chainer.initializers.Normal((0.02 * math.sqrt(256 * 3 * 3)) / math.sqrt(256))),
            bn5=chainer.links.BatchNormalization(256),
            c6=chainer.links.Convolution2D(256, 512, ksize=3, stride=1, pad=0, initialW=chainer.initializers.Normal((0.02 * math.sqrt(256 * 3 * 3)) / math.sqrt(256))),
            bn6=chainer.links.BatchNormalization(512),
            c7=chainer.links.Convolution2D(512, 512, ksize=3, stride=2, pad=0, initialW=chainer.initializers.Normal((0.02 * math.sqrt(512 * 3 * 3)) / math.sqrt(512))),
            bn7=chainer.links.BatchNormalization(512),
            linear1=chainer.links.Linear(in_size=4608, out_size=1024),
            linear2=chainer.links.Linear(in_size=None, out_size=2),
        )

    def __call__(self, x):
        h = self.conv_input(x)
        
        with chainer.using_config("test", False):
            h = self.bn1(chainer.functions.elu(self.c1(h)))
            h = self.bn2(chainer.functions.elu(self.c2(h)))
            h = self.bn3(chainer.functions.elu(self.c3(h)))
            h = self.bn4(chainer.functions.elu(self.c4(h)))
            h = self.bn5(chainer.functions.elu(self.c5(h)))
            h = self.bn6(chainer.functions.elu(self.c6(h)))
            h = self.bn7(chainer.functions.elu(self.c7(h)))
        h = chainer.functions.elu(self.linear1(h))
        h = chainer.functions.sigmoid(self.linear2(h))
        return h

