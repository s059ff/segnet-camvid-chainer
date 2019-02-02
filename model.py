import chainer
import chainer.functions as F
import chainer.links as L


class SegNet(chainer.Chain):

    def __init__(self):
        super(SegNet, self).__init__()

        def same(ksize):
            return tuple(map(lambda x: x // 2, ksize))

        def valid():
            return (0, 0)

        with self.init_scope():

            initialW = chainer.initializers.HeNormal()

            # Encoding layers.
            self.conv1 = L.Convolution2D(None, 64, ksize=(7, 7), nobias=True, initialW=initialW, pad=same((7, 7)))
            self.bn1 = L.BatchNormalization(64, initial_beta=0.001)

            self.conv2 = L.Convolution2D(None, 64, ksize=(7, 7), nobias=True, initialW=initialW, pad=same((7, 7)))
            self.bn2 = L.BatchNormalization(64, initial_beta=0.001)

            self.conv3 = L.Convolution2D(None, 64, ksize=(7, 7), nobias=True, initialW=initialW, pad=same((7, 7)))
            self.bn3 = L.BatchNormalization(64, initial_beta=0.001)

            self.conv4 = L.Convolution2D(None, 64, ksize=(7, 7), nobias=True, initialW=initialW, pad=same((7, 7)))
            self.bn4 = L.BatchNormalization(64, initial_beta=0.001)

            # Decode layers.
            self.conv5 = L.Convolution2D(None, 64, ksize=(7, 7), nobias=True, initialW=initialW, pad=same((7, 7)))
            self.bn5 = L.BatchNormalization(64, initial_beta=0.001)

            self.conv6 = L.Convolution2D(None, 64, ksize=(7, 7), nobias=True, initialW=initialW, pad=same((7, 7)))
            self.bn6 = L.BatchNormalization(64, initial_beta=0.001)

            self.conv7 = L.Convolution2D(None, 64, ksize=(7, 7), nobias=True, initialW=initialW, pad=same((7, 7)))
            self.bn7 = L.BatchNormalization(64, initial_beta=0.001)

            self.conv8 = L.Convolution2D(None, 64, ksize=(7, 7), nobias=True, initialW=initialW, pad=same((7, 7)))
            self.bn8 = L.BatchNormalization(64, initial_beta=0.001)

            self.conv9 = L.Convolution2D(None, 1, ksize=(1, 1), pad=valid())

    def __call__(self, h):

        # Encode.
        h = self.conv1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h, indices1 = F.max_pooling_2d(h, ksize=(2, 2), cover_all=False, return_indices=True)

        h = self.conv2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h, indices2 = F.max_pooling_2d(h, ksize=(2, 2), cover_all=False, return_indices=True)

        h = self.conv3(h)
        h = self.bn3(h)
        h = F.relu(h)
        h, indices3 = F.max_pooling_2d(h, ksize=(2, 2), cover_all=False, return_indices=True)

        h = self.conv4(h)
        h = self.bn4(h)
        h = F.relu(h)
        # h, indices4 = F.max_pooling_2d(h, ksize=(2, 2), cover_all=False, return_indices=True)

        # Decode.
        # h = F.upsampling_2d(h, indices4, ksize=(2, 2))
        h = self.conv5(h)
        h = self.bn5(h)

        h = F.upsampling_2d(h, indices3, ksize=(2, 2), cover_all=False)
        h = self.conv6(h)
        h = self.bn6(h)

        h = F.upsampling_2d(h, indices2, ksize=(2, 2), cover_all=False)
        h = self.conv7(h)
        h = self.bn7(h)

        h = F.upsampling_2d(h, indices1, ksize=(2, 2), cover_all=False)
        h = self.conv8(h)
        h = self.bn8(h)

        h = self.conv9(h)

        return h
