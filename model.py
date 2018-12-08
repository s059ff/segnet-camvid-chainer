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

            # Encoding layers.
            self.conv1 = L.Convolution2D(
                None, 64, (3, 3), pad=(same(3), same(3)))
            self.bn1 = L.BatchNormalization(64)

            self.conv2 = L.Convolution2D(
                None, 128, (3, 3), pad=(same(3), same(3)))
            self.bn2 = L.BatchNormalization(128)

            self.conv3 = L.Convolution2D(
                None, 256, (3, 3), pad=(same(3), same(3)))
            self.bn3 = L.BatchNormalization(256)

            self.conv4 = L.Convolution2D(
                None, 512, (3, 3), pad=(same(3), same(3)))
            self.bn4 = L.BatchNormalization(512)

            # Decode layers.
            self.conv5 = L.Convolution2D(
                None, 512, (3, 3), pad=(same(3), same(3)))
            self.bn5 = L.BatchNormalization(512)

            self.conv6 = L.Convolution2D(
                None, 256, (3, 3), pad=(same(3), same(3)))
            self.bn6 = L.BatchNormalization(256)

            self.conv7 = L.Convolution2D(
                None, 128, (3, 3), pad=(same(3), same(3)))
            self.bn7 = L.BatchNormalization(128)

            self.conv8 = L.Convolution2D(
                None, 64, (3, 3), pad=(same(3), same(3)))
            self.bn8 = L.BatchNormalization(64)

            self.conv9 = L.Convolution2D(
                None, 1, (1, 1), pad=(valid(), valid()))

    def __call__(self, h):

        # Encode.
        h = self.conv1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, (2, 2))

        h = self.conv2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, (2, 2))

        h = self.conv3(h)
        h = self.bn3(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, (2, 2))

        h = self.conv4(h)
        h = self.bn4(h)
        h = F.relu(h)

        # Decode.
        h = self.conv5(h)
        h = self.bn5(h)
        h = F.relu(h)

        h = F.unpooling_2d(h, (2, 2), cover_all=False)
        h = self.conv6(h)
        h = self.bn6(h)
        h = F.relu(h)

        h = F.unpooling_2d(h, (2, 2), cover_all=False)
        h = self.conv7(h)
        h = self.bn7(h)
        h = F.relu(h)

        h = F.unpooling_2d(h, (2, 2), cover_all=False)
        h = self.conv8(h)
        h = self.bn8(h)
        h = F.relu(h)

        h = self.conv9(h)
        h = F.sigmoid(h)

        return h
