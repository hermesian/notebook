import chainer
import chainer.functions as F
import chainer.links as L


class CNN(chainer.Chain):

    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 32, 5, pad=2),
            conv2=L.Convolution2D(32, 64, 5, pad=2),
            fl3=L.Linear(7*7*64, 1024),
            fl4=L.Linear(1024, 10)
        )
        self.train = True

    def __call__(self, x):
        h_conv1 = F.relu(self.conv1(x))
        h_pool1 = F.max_pooling_2d(h_conv1, 2)

        h_conv2 = F.relu(self.conv2(h_pool1))
        h_pool2 = F.max_pooling_2d(h_conv2, 2)

        h_fc1 = F.relu(self.fl3(h_pool2))
        return F.dropout(h_fc1, ratio=0.5, train=self.train)
