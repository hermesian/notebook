import chainer
import chainer.functions as F
import chainer.links as L


class MnistCNN(chainer.Chain):

    """An example of convolutional neural network for MNIST dataset.
       refered page:
       http://ttlg.hateblo.jp/entry/2016/02/11/181322

    """

    def __init__(self, channel=1, c1=16, c2=32, c3=64, f1=256, \
                 f2=512, filter_size1=3, filter_size2=3, filter_size3=3):
        super(MnistCNN, self).__init__(
            conv1=L.Convolution2D(channel, c1, filter_size1),
            conv2=L.Convolution2D(c1, c2, filter_size2),
            conv3=L.Convolution2D(c2, c3, filter_size3),
            l1=L.Linear(f1, f2),
            l2=L.Linear(f2, 10)
        )

    def __call__(self, x):
        # param x --- chainer.Variable of array
        x.data = x.data.reshape((len(x.data), 1, 28, 28))
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(F.relu(self.l1(h)))
        y = self.l2(h)
        return y

