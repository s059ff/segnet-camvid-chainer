import chainer
import chainer.functions as F
import chainer.links as L


class SegNet(chainer.Chain):

    def __init__(self):
        super(SegNet, self).__init__()

        def same(ksize):
            return ksize // 2

        def valid():
            return 0

        with self.init_scope():

            initialW = chainer.initializers.HeNormal()

            # Encoding layers.
            self.conv1 = L.Convolution2D(None, 64, ksize=(3, 3), nobias=True, initialW=initialW, pad=(same(3), same(3)))
            self.bn1 = L.BatchNormalization(64, initial_beta=0.001)

            self.conv2 = L.Convolution2D(None, 64, ksize=(3, 3), nobias=True, initialW=initialW, pad=(same(3), same(3)))
            self.bn2 = L.BatchNormalization(64, initial_beta=0.001)

            self.conv3 = L.Convolution2D(None, 64, ksize=(3, 3), nobias=True, initialW=initialW, pad=(same(3), same(3)))
            self.bn3 = L.BatchNormalization(64, initial_beta=0.001)

            self.conv4 = L.Convolution2D(None, 64, ksize=(3, 3), nobias=True, initialW=initialW, pad=(same(3), same(3)))
            self.bn4 = L.BatchNormalization(64, initial_beta=0.001)

            # Decode layers.
            self.conv_decode4 = L.Convolution2D(None, 64, ksize=(3, 3), nobias=True, initialW=initialW, pad=(same(3), same(3)))
            self.bn_decode4 = L.BatchNormalization(64, initial_beta=0.001)

            self.conv_decode3 = L.Convolution2D(None, 64, ksize=(3, 3), nobias=True, initialW=initialW, pad=(same(3), same(3)))
            self.bn_decode3 = L.BatchNormalization(64, initial_beta=0.001)

            self.conv_decode2 = L.Convolution2D(None, 64, ksize=(3, 3), nobias=True, initialW=initialW, pad=(same(3), same(3)))
            self.bn_decode2 = L.BatchNormalization(64, initial_beta=0.001)

            self.conv_decode1 = L.Convolution2D(None, 64, ksize=(3, 3), nobias=True, initialW=initialW, pad=(same(3), same(3)))
            self.bn_decode1 = L.BatchNormalization(64, initial_beta=0.001)

            self.conv_classifier = L.Convolution2D(None, 1, ksize=(1, 1), pad=(valid(), valid()))

    def __call__(self, h):

        # Encode.
        h = self.conv1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h, indices1 = F.max_pooling_2d(h, (2, 2), cover_all=False, return_indices=True)

        h = self.conv2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h, indices2 = F.max_pooling_2d(h, (2, 2), cover_all=False, return_indices=True)

        h = self.conv3(h)
        h = self.bn3(h)
        h = F.relu(h)
        h, indices3 = F.max_pooling_2d(h, (2, 2), cover_all=False, return_indices=True)

        h = self.conv4(h)
        h = self.bn4(h)
        h = F.relu(h)
        # h, indices4 = F.max_pooling_2d(h, (2, 2), cover_all=False, return_indices=True)

        # Decode.
        # h = F.upsampling_2d(h, indices4, (2, 2))
        h = self.conv_decode4(h)
        h = self.bn_decode4(h)
        h = F.relu(h)

        h = F.upsampling_2d(h, indices3, (2, 2), cover_all=False)
        h = self.conv_decode3(h)
        h = self.bn_decode3(h)
        h = F.relu(h)

        h = F.upsampling_2d(h, indices2, (2, 2), cover_all=False)
        h = self.conv_decode2(h)
        h = self.bn_decode2(h)
        h = F.relu(h)

        h = F.upsampling_2d(h, indices1, (2, 2), cover_all=False)
        h = self.conv_decode1(h)
        h = self.bn_decode1(h)
        h = F.relu(h)

        h = self.conv_classifier(h)

        return h
